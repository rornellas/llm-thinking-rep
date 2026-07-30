#!/usr/bin/env python3
"""Run one preregistered seed of routing-set distillation v3."""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pre_qwen_certification.alignment_tolerant import SharedLowRankResidualMoE
from pre_qwen_certification.controlled_transplant import _tiny_config
from pre_qwen_certification.data import find_duplicates, jaccard, text_sha256, word_shingles
from pre_qwen_certification.modal import ConventionalSwiGLUMoE
from pre_qwen_certification.routing_set_data import (
    generate_routing_set_hypothesis_documents,
    generate_routing_set_ood_documents,
)
from pre_qwen_certification.routing_set_distillation import (
    JointRoutingSetWeights,
    RoutingSetWeights,
    distill_routing_set_student,
    evaluate_routing_set_fidelity,
    joint_fine_tune_routing_set,
)
from pre_qwen_certification.teacher_width_data import generate_width_documents
from pre_qwen_certification.tiny_lm import (
    CharacterCorpus,
    TinyMoELanguageModel,
    capture_training_layer,
    distill_layer_student,
    evaluate_closed_loop,
    evaluate_local_student,
    install_student,
    make_narrow_student,
)


def stable_seed(*parts: object) -> int:
    payload = "::".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**31 - 1)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_head() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout.strip()


def require_clean() -> None:
    pathspec = [
        ".",
        ":(exclude)runs/**",
        ":(exclude)results/**",
        ":(exclude)artifacts/**",
    ]
    for command in (
        ["git", "diff", "--quiet", "--", *pathspec],
        ["git", "diff", "--cached", "--quiet", "--", *pathspec],
    ):
        if subprocess.run(command, cwd=ROOT, check=False).returncode != 0:
            raise RuntimeError("tracked source/config/test files must be committed and clean")


def resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def source_seed_paths(source_dir: Path, seed: int) -> tuple[Path, Path]:
    nested = source_dir / f"seed-{seed}"
    nested_checkpoint = nested / f"frozen-candidates-seed-{seed}.pt"
    nested_payload = nested / f"seed-{seed}.json"
    if nested_checkpoint.exists() and nested_payload.exists():
        return nested_checkpoint, nested_payload
    return (
        source_dir / f"frozen-candidates-seed-{seed}.pt",
        source_dir / f"seed-{seed}.json",
    )


def cross_split_audit(
    train_docs: list, hypothesis_docs: list, ood_docs: list, threshold: float
) -> dict[str, Any]:
    groups = {"train": train_docs, "hypothesis": hypothesis_docs, "ood": ood_docs}
    exact: list[dict[str, str]] = []
    near: list[dict[str, Any]] = []
    maximum = 0.0
    labels = list(groups)
    for left_index, left_label in enumerate(labels):
        for right_label in labels[left_index + 1 :]:
            for left in groups[left_label]:
                left_hash = text_sha256(left.text)
                left_shingles = word_shingles(left.text)
                for right in groups[right_label]:
                    if left_hash == text_sha256(right.text):
                        exact.append({"left": left.document_id, "right": right.document_id})
                    score = jaccard(left_shingles, word_shingles(right.text))
                    maximum = max(maximum, score)
                    if score >= threshold:
                        near.append(
                            {
                                "left": left.document_id,
                                "right": right.document_id,
                                "left_split": left_label,
                                "right_split": right_label,
                                "jaccard": float(score),
                            }
                        )
    return {
        "exact_cross_split_duplicates": exact,
        "near_cross_split_pairs": near,
        "maximum_cross_split_jaccard": float(maximum),
        "threshold": float(threshold),
    }


@torch.no_grad()
def routing_window_metrics(
    teacher: TinyMoELanguageModel,
    module: torch.nn.Module,
    corpus: CharacterCorpus,
    *,
    split: str,
    windows_per_document: int,
    seed: int,
    diagnostic_weights: RoutingSetWeights,
) -> tuple[dict[str, float], dict[tuple[str, int], dict[str, float]]]:
    windows = corpus.fixed_windows(split, windows_per_document=windows_per_document, seed=seed)
    teacher_moe = teacher.blocks[0].moe
    if not isinstance(teacher_moe, ConventionalSwiGLUMoE):
        raise TypeError("teacher layer is not conventional")
    rows: dict[tuple[str, int], dict[str, float]] = {}
    values: list[dict[str, float]] = []
    teacher.eval(); module.eval()
    for window in windows:
        _, _, capture = teacher(window.inputs[None, :], collect_layer=0)
        if capture is None:
            raise AssertionError("missing teacher capture")
        current = evaluate_routing_set_fidelity(
            teacher_moe,
            module,
            capture.moe_input,
            capture.routing.top_ids,
            capture.routing.weights,
            diagnostic_weights,
        )
        row = {
            "expert_nrmse": max(current["expert"], 0.0) ** 0.5,
            "counterfactual_nrmse": max(current["counterfactual"], 0.0) ** 0.5,
            "geometry_mse": current["geometry"],
            "expert_cosine_penalty": current["cosine"],
            "routing_self_error": current["self_error"],
            "routing_cross_error": current["cross_error"],
            "routing_aggregate_error": current["aggregate_error"],
        }
        rows[(window.document_id, window.start)] = row
        values.append(row)
    summary = {
        key: float(sum(row[key] for row in values) / len(values)) for key in values[0]
    }
    return summary, rows


def evaluate_module(
    teacher: TinyMoELanguageModel,
    module: torch.nn.Module,
    corpus: CharacterCorpus,
    *,
    seed: int,
    candidate: str,
    phase: str,
    windows_per_document: int,
    evaluation_seed: int,
    diagnostic_weights: RoutingSetWeights,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    installed = install_student(teacher, module, layer_id=0)
    summary: dict[str, Any] = {}
    records: list[dict[str, Any]] = []
    for split in ("hypothesis", "ood"):
        local, local_rows = evaluate_local_student(
            teacher,
            module,
            corpus,
            split=split,
            layer_id=0,
            windows_per_document=windows_per_document,
            seed=evaluation_seed,
        )
        routing_summary, routing_rows = routing_window_metrics(
            teacher,
            module,
            corpus,
            split=split,
            windows_per_document=windows_per_document,
            seed=evaluation_seed,
            diagnostic_weights=diagnostic_weights,
        )
        closed, closed_rows = evaluate_closed_loop(
            teacher,
            installed,
            corpus,
            split=split,
            windows_per_document=windows_per_document,
            seed=evaluation_seed,
        )
        summary[split] = {
            "local": local,
            "routing_set": routing_summary,
            "closed_loop": closed,
        }
        local_index = {
            (str(row["document_id"]), int(row["start"])): float(row["nrmse"])
            for row in local_rows
        }
        for row in closed_rows:
            key = (str(row["document_id"]), int(row["start"]))
            records.append(
                {
                    **row,
                    **routing_rows[key],
                    "seed": int(seed),
                    "candidate": candidate,
                    "phase": phase,
                    "evaluation_split": split,
                    "local_nrmse": local_index[key],
                }
            )
    return summary, records


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/pre_qwen_alignment_routing_set_v3.yaml",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "results/pre-qwen-alignment-routing-set/v3",
    )
    args = parser.parse_args()

    config: dict[str, Any] = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if args.seed not in [int(value) for value in config["seeds"]]:
        parser.error("seed is not preregistered")
    require_clean()
    source_commit = git_head()
    torch.set_num_threads(int(config["threads"]))
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass

    tiny = _tiny_config(config["model"])
    source_cfg = config["source"]
    source_dir = resolve(str(source_cfg["teacher_width_results"]))
    source_config_path = resolve(str(source_cfg["teacher_width_config"]))
    source_verdict = source_dir / "VERDICT.md"
    required = str(source_cfg["required_source_verdict"])
    if required not in source_verdict.read_text(encoding="utf-8"):
        raise RuntimeError(f"required source verdict not found: {required}")

    checkpoint_path, source_payload_path = source_seed_paths(source_dir, args.seed)
    source_payload = json.loads(source_payload_path.read_text(encoding="utf-8"))
    if sha256_file(checkpoint_path) != str(source_payload["metadata"]["checkpoint_sha256"]):
        raise RuntimeError("source checkpoint hash mismatch")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if str(checkpoint["configuration_sha256"]) != sha256_file(source_config_path):
        raise RuntimeError("source configuration hash mismatch")

    data_cfg = config["data"]
    train_docs = generate_width_documents(
        split=str(data_cfg["train_split"]),
        documents=int(data_cfg["train_documents"]),
        seed=int(data_cfg["train_seed"]),
    )
    vocabulary = list(checkpoint["vocabulary"])
    train_corpus = CharacterCorpus({"train": train_docs}, tiny.seq_len, vocabulary=vocabulary)
    teacher = TinyMoELanguageModel(len(vocabulary), tiny)
    teacher.load_state_dict(checkpoint["teacher_state"])
    teacher.eval()
    teacher_moe = teacher.blocks[0].moe
    if not isinstance(teacher_moe, ConventionalSwiGLUMoE):
        raise TypeError("source teacher layer is not conventional")

    captured = capture_training_layer(
        teacher,
        train_corpus,
        split="train",
        layer_id=0,
        batches=int(data_cfg["capture_batches"]),
        batch_size=int(data_cfg["capture_batch_size"]),
        seed=stable_seed(args.seed, config["protocol_version"], "capture"),
    )

    trainable_rows = [
        row for row in config["candidates"] if row["architecture"] == "shared_full_rank_base_plus_expert_bilateral_low_rank_residual"
    ]
    initial_by_rank: dict[int, SharedLowRankResidualMoE] = {}
    candidates: dict[str, torch.nn.Module] = {}
    candidate_objectives: dict[str, RoutingSetWeights] = {}
    for row in trainable_rows:
        rank = int(row["rank"])
        if rank not in initial_by_rank:
            initial_by_rank[rank] = SharedLowRankResidualMoE.from_teacher(teacher_moe, rank=rank)
        name = str(row["name"])
        candidates[name] = copy.deepcopy(initial_by_rank[rank])
        candidate_objectives[name] = RoutingSetWeights.from_mapping(
            config["objectives"][str(row["objective"])]
        )

    baseline = make_narrow_student(teacher_moe, d_ff=26)
    baseline.load_state_dict(checkpoint["candidate_states"][str(source_cfg["baseline_state"])])
    baseline.router.weight.requires_grad_(False)
    candidates["narrow65-frozen-baseline"] = baseline

    full_control = copy.deepcopy(teacher_moe)
    full_control.load_state_dict(checkpoint["candidate_states"][str(source_cfg["full_control_state"])])
    full_control.router.weight.requires_grad_(False)
    candidates["full-continuation-control"] = full_control
    initial = {name: copy.deepcopy(module) for name, module in candidates.items()}

    training = config["training"]
    joint_weights = JointRoutingSetWeights.from_mapping(training["joint_weights"])
    histories: dict[str, Any] = {}
    for name in [str(row["name"]) for row in trainable_rows]:
        # All candidates of the same rank begin with the identical aggregate-only
        # warm-up and identical sampled minibatches. Objective-specific behavior
        # begins only in the local stage.
        histories[name] = {
            "stage1": distill_layer_student(
                candidates[name],
                captured,
                steps=int(training["stage1_steps"]),
                batch_size=int(training["local_batch_size"]),
                learning_rate=float(training["stage1_learning_rate"]),
                seed=stable_seed(args.seed, config["protocol_version"], "stage1", int(getattr(candidates[name], "rank", 0))),
            )
        }
        histories[name]["local"] = distill_routing_set_student(
            candidates[name],
            teacher_moe,
            captured,
            steps=int(training["local_steps"]),
            batch_size=int(training["local_batch_size"]),
            learning_rate=float(training["local_learning_rate"]),
            seed=stable_seed(args.seed, config["protocol_version"], "local"),
            objective_weights=candidate_objectives[name],
        )
        candidates[name], histories[name]["joint"] = joint_fine_tune_routing_set(
            teacher,
            candidates[name],
            train_corpus,
            layer_id=0,
            steps=int(training["joint_steps"]),
            batch_size=int(tiny.batch_size),
            learning_rate=float(training["joint_learning_rate"]),
            seed=stable_seed(args.seed, config["protocol_version"], "joint"),
            local_weights=candidate_objectives[name],
            joint_weights=joint_weights,
        )
        print(f"trained seed={args.seed} candidate={name}", flush=True)

    histories["narrow65-frozen-baseline"] = {"source": "frozen teacher-width state"}
    histories["full-continuation-control"] = {"source": "frozen teacher-width state"}

    args.output_dir.mkdir(parents=True, exist_ok=True)
    frozen_path = args.output_dir / f"frozen-candidates-seed-{args.seed}.pt"
    accounting: dict[str, dict[str, float | int]] = {}
    for name, module in candidates.items():
        if isinstance(module, SharedLowRankResidualMoE):
            accounting[name] = module.accounting().as_dict()
        elif isinstance(module, ConventionalSwiGLUMoE):
            ratio = module.geometry.d_ff / teacher_moe.geometry.d_ff
            accounting[name] = {"parameter_ratio": ratio, "compute_ratio": ratio}
        else:
            raise TypeError(type(module).__name__)
    torch.save(
        {
            "teacher_state": teacher.state_dict(),
            "candidate_states": {name: module.state_dict() for name, module in candidates.items()},
            "initial_states": {name: module.state_dict() for name, module in initial.items()},
            "candidate_types": {name: type(module).__name__ for name, module in candidates.items()},
            "candidate_accounting": accounting,
            "vocabulary": vocabulary,
            "seed": args.seed,
            "source_commit": source_commit,
            "source_teacher_checkpoint": str(checkpoint_path.relative_to(ROOT)),
            "source_teacher_checkpoint_sha256": sha256_file(checkpoint_path),
            "configuration_sha256": sha256_file(args.config),
            "protocol_version": config["protocol_version"],
        },
        frozen_path,
    )

    # Held-out data is materialized only after every candidate state is frozen.
    hypothesis_docs = generate_routing_set_hypothesis_documents(
        split=str(data_cfg["hypothesis_split"]),
        documents=int(data_cfg["hypothesis_documents"]),
        seed=int(data_cfg["hypothesis_seed"]),
    )
    ood_docs = generate_routing_set_ood_documents(split=str(data_cfg["ood_split"]))
    eval_corpus = CharacterCorpus(
        {"hypothesis": hypothesis_docs, "ood": ood_docs}, tiny.seq_len, vocabulary=vocabulary
    )
    data_audit = cross_split_audit(
        train_docs, hypothesis_docs, ood_docs, float(data_cfg["near_duplicate_threshold"])
    )
    data_audit["within_all_findings"] = [
        finding.__dict__
        for finding in find_duplicates(
            train_docs + hypothesis_docs + ood_docs,
            near_duplicate_threshold=float(data_cfg["near_duplicate_threshold"]),
        )
    ]

    diagnostic_weights = RoutingSetWeights.from_mapping(config["objectives"]["routing_set_v3"])
    candidate_order = [str(row["name"]) for row in config["candidates"]]
    evaluations: dict[str, Any] = {}
    records: list[dict[str, Any]] = []
    for name in candidate_order:
        evaluations[name] = {}
        for phase, module in (("initial", initial[name]), ("final", candidates[name])):
            summary, current = evaluate_module(
                teacher,
                module,
                eval_corpus,
                seed=args.seed,
                candidate=name,
                phase=phase,
                windows_per_document=int(data_cfg["windows_per_document"]),
                evaluation_seed=int(data_cfg["evaluation_window_seed"]),
                diagnostic_weights=diagnostic_weights,
            )
            evaluations[name][phase] = summary
            records.extend(current)
            print(
                f"evaluated seed={args.seed} candidate={name} phase={phase} "
                f"hyp={summary['hypothesis']['closed_loop']['loss_delta']:+.6f}",
                flush=True,
            )

    payload = {
        "metadata": {
            "protocol_version": config["protocol_version"],
            "seed": args.seed,
            "source_commit": source_commit,
            "configuration_sha256": sha256_file(args.config),
            "checkpoint_sha256": sha256_file(frozen_path),
            "source_teacher_checkpoint": str(checkpoint_path.relative_to(ROOT)),
            "source_teacher_checkpoint_sha256": sha256_file(checkpoint_path),
            "source_teacher_payload_sha256": sha256_file(source_payload_path),
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "source_teacher_tail_change": float(source_payload["metadata"]["teacher_tail_change"]),
            "captured_tokens": int(len(captured.inputs)),
            "parameter_ratios": {name: float(accounting[name]["parameter_ratio"]) for name in candidate_order},
            "compute_ratios": {name: float(accounting[name]["compute_ratio"]) for name in candidate_order},
            "accounting": accounting,
            "data_audit": data_audit,
        },
        "training": histories,
        "evaluations": evaluations,
        "records": records,
    }
    path = args.output_dir / f"seed-{args.seed}.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
