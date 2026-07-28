#!/usr/bin/env python3
"""Run one preregistered seed of the fresh teacher-informed width replication."""
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

from pre_qwen_certification.controlled_transplant import _tiny_config
from pre_qwen_certification.data import find_duplicates, jaccard, text_sha256, word_shingles
from pre_qwen_certification.modal import ConventionalSwiGLUMoE
from pre_qwen_certification.teacher_width_data import (
    generate_width_documents,
    generate_width_ood_documents,
)
from pre_qwen_certification.tiny_lm import (
    CharacterCorpus,
    TinyMoELanguageModel,
    capture_training_layer,
    distill_layer_student,
    evaluate_closed_loop,
    evaluate_local_student,
    install_student,
    joint_fine_tune_transplant,
    make_narrow_student,
    train_teacher,
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
    # Scientific source must be immutable. Generated evidence under runs/,
    # results/, and artifacts/ is allowed to change during a workflow run.
    pathspec = [
        ".",
        ":(exclude)runs/**",
        ":(exclude)results/**",
        ":(exclude)artifacts/**",
    ]
    commands = (
        ["git", "diff", "--quiet", "--", *pathspec],
        ["git", "diff", "--cached", "--quiet", "--", *pathspec],
    )
    for command in commands:
        if subprocess.run(command, cwd=ROOT, check=False).returncode != 0:
            raise RuntimeError(
                "tracked source/config/test files must be committed and clean before a seed run"
            )


def cross_split_audit(train_docs: list, hypothesis_docs: list, ood_docs: list, threshold: float) -> dict[str, Any]:
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
                        near.append({
                            "left": left.document_id,
                            "right": right.document_id,
                            "left_split": left_label,
                            "right_split": right_label,
                            "jaccard": float(score),
                        })
    return {
        "exact_cross_split_duplicates": exact,
        "near_cross_split_pairs": near,
        "maximum_cross_split_jaccard": float(maximum),
        "threshold": float(threshold),
    }


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
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    installed = install_student(teacher, module, layer_id=0)
    summary: dict[str, Any] = {}
    records: list[dict[str, Any]] = []
    for split in ("hypothesis", "ood"):
        local, local_rows = evaluate_local_student(
            teacher, module, corpus, split=split, layer_id=0,
            windows_per_document=windows_per_document, seed=evaluation_seed,
        )
        closed, closed_rows = evaluate_closed_loop(
            teacher, installed, corpus, split=split,
            windows_per_document=windows_per_document, seed=evaluation_seed,
        )
        summary[split] = {"local": local, "closed_loop": closed}
        local_index = {(str(row["document_id"]), int(row["start"])): float(row["nrmse"]) for row in local_rows}
        for row in closed_rows:
            key = (str(row["document_id"]), int(row["start"]))
            records.append({
                **row,
                "seed": int(seed),
                "candidate": candidate,
                "phase": phase,
                "evaluation_split": split,
                "local_nrmse": local_index[key],
            })
    return summary, records


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/pre_qwen_teacher_width_fresh_v1.yaml")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/pre-qwen-teacher-width-fresh/v1")
    args = parser.parse_args()

    config: dict[str, Any] = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if args.seed not in [int(v) for v in config["seeds"]]:
        parser.error("seed is not preregistered")
    require_clean()
    source_commit = git_head()
    torch.set_num_threads(int(config["threads"]))
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass

    tiny = _tiny_config(config["model"])
    data_cfg = config["data"]
    train_docs = generate_width_documents(
        split=str(data_cfg["train_split"]),
        documents=int(data_cfg["train_documents"]),
        seed=int(data_cfg["train_seed"]),
    )
    train_corpus = CharacterCorpus({"train": train_docs}, tiny.seq_len)
    teacher, teacher_history = train_teacher(train_corpus, tiny, seed=args.seed)
    teacher.eval()
    teacher_moe = teacher.blocks[0].moe
    if not isinstance(teacher_moe, ConventionalSwiGLUMoE):
        raise TypeError("teacher layer is not conventional")

    captured = capture_training_layer(
        teacher, train_corpus, split="train", layer_id=0,
        batches=int(data_cfg["capture_batches"]),
        batch_size=int(data_cfg["capture_batch_size"]),
        seed=stable_seed(args.seed, config["protocol_version"], "capture"),
    )

    widths = {str(k): int(v) for k, v in config["widths"].items()}
    candidates: dict[str, torch.nn.Module] = {
        "magnitude-init-35": make_narrow_student(teacher_moe, d_ff=widths["anchor"]),
        "magnitude-init-50": make_narrow_student(teacher_moe, d_ff=widths["comparator"]),
        "magnitude-init-65": make_narrow_student(teacher_moe, d_ff=widths["primary"]),
        "magnitude-init-75": make_narrow_student(teacher_moe, d_ff=widths["capacity"]),
        "full-continuation-control": copy.deepcopy(teacher_moe),
    }
    candidates["full-continuation-control"].router.weight.requires_grad_(False)
    initial = {name: copy.deepcopy(module) for name, module in candidates.items()}

    training = config["training"]
    stage_seed = stable_seed(args.seed, config["protocol_version"], "stage1")
    local_seed = stable_seed(args.seed, config["protocol_version"], "local")
    joint_seed = stable_seed(args.seed, config["protocol_version"], "joint")
    histories: dict[str, Any] = {}
    for name in ("magnitude-init-35", "magnitude-init-50", "magnitude-init-65", "magnitude-init-75"):
        histories[name] = {
            "stage1": distill_layer_student(
                candidates[name], captured,
                steps=int(training["stage1_steps"]),
                batch_size=int(training["local_batch_size"]),
                learning_rate=float(tiny.student_learning_rate),
                seed=stage_seed,
            )
        }
        histories[name]["local"] = distill_layer_student(
            candidates[name], captured,
            steps=int(training["local_steps"]),
            batch_size=int(training["local_batch_size"]),
            learning_rate=float(tiny.student_learning_rate),
            seed=local_seed,
        )
    histories["full-continuation-control"] = {"stage1": [], "local": []}

    for name in candidates:
        candidates[name], histories[name]["joint"] = joint_fine_tune_transplant(
            teacher, candidates[name], train_corpus, layer_id=0,
            steps=int(training["joint_steps"]),
            batch_size=int(tiny.batch_size),
            learning_rate=float(tiny.student_learning_rate),
            seed=joint_seed,
            local_weight=float(training["joint_local_weight"]),
            kl_weight=float(training["joint_kl_weight"]),
            ce_weight=float(training["joint_ce_weight"]),
        )
        print(f"trained seed={args.seed} candidate={name}", flush=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.output_dir / f"frozen-candidates-seed-{args.seed}.pt"
    torch.save({
        "teacher_state": teacher.state_dict(),
        "candidate_states": {name: module.state_dict() for name, module in candidates.items()},
        "initial_states": {name: module.state_dict() for name, module in initial.items()},
        "vocabulary": list(train_corpus.itos),
        "seed": args.seed,
        "source_commit": source_commit,
        "configuration_sha256": sha256_file(args.config),
        "protocol_version": config["protocol_version"],
    }, checkpoint_path)

    # Held-out families are materialized only after every candidate is frozen.
    hypothesis_docs = generate_width_documents(
        split=str(data_cfg["hypothesis_split"]),
        documents=int(data_cfg["hypothesis_documents"]),
        seed=int(data_cfg["hypothesis_seed"]),
    )
    ood_docs = generate_width_ood_documents(split=str(data_cfg["ood_split"]))
    eval_corpus = CharacterCorpus(
        {"hypothesis": hypothesis_docs, "ood": ood_docs},
        tiny.seq_len,
        vocabulary=train_corpus.itos,
    )
    data_audit = cross_split_audit(
        train_docs, hypothesis_docs, ood_docs,
        float(data_cfg["near_duplicate_threshold"]),
    )
    data_audit["within_all_findings"] = [
        finding.__dict__ for finding in find_duplicates(
            train_docs + hypothesis_docs + ood_docs,
            near_duplicate_threshold=float(data_cfg["near_duplicate_threshold"]),
        )
    ]

    evaluations: dict[str, Any] = {}
    records: list[dict[str, Any]] = []
    for name in candidates:
        evaluations[name] = {}
        for phase, module in (("initial", initial[name]), ("final", candidates[name])):
            summary, current = evaluate_module(
                teacher, module, eval_corpus, seed=args.seed, candidate=name, phase=phase,
                windows_per_document=int(data_cfg["windows_per_document"]),
                evaluation_seed=int(data_cfg["evaluation_window_seed"]),
            )
            evaluations[name][phase] = summary
            records.extend(current)
            print(
                f"evaluated seed={args.seed} candidate={name} phase={phase} "
                f"hyp={summary['hypothesis']['closed_loop']['loss_delta']:+.6f}", flush=True,
            )

    ratios = {
        name: (module.geometry.d_ff / teacher_moe.geometry.d_ff if isinstance(module, ConventionalSwiGLUMoE) else 1.0)
        for name, module in candidates.items()
    }
    payload = {
        "metadata": {
            "protocol_version": config["protocol_version"],
            "seed": args.seed,
            "source_commit": source_commit,
            "configuration_sha256": sha256_file(args.config),
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "teacher_history": teacher_history,
            "teacher_tail_change": float(teacher_history[-1]["language_loss"] - teacher_history[-2]["language_loss"]),
            "captured_tokens": int(len(captured.inputs)),
            "parameter_ratios": ratios,
            "compute_ratios": ratios,
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
