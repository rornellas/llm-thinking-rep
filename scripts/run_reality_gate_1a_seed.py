#!/usr/bin/env python3
"""Run one scale/seed cell of Reality Gate 1A."""
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

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pre_qwen_certification.controlled_transplant import _tiny_config
from pre_qwen_certification.heterogeneous_rank import (
    HeterogeneousSharedLowRankResidualMoE,
    allocate_static_ranks,
    effective_residual_rank,
    residual_mode_utilities,
    route_frequencies,
    uniform_rank_vector,
)
from pre_qwen_certification.modal import ConventionalSwiGLUMoE
from pre_qwen_certification.reality_gate import (
    PlateauRule,
    restore_teacher,
    routing_distribution,
    train_teacher_to_plateau,
)
from pre_qwen_certification.reality_gate_data import (
    ArrayTokenCorpus,
    documents_from_prepared_split,
    load_prepared_arrays,
    sha256_file,
)
from pre_qwen_certification.tiny_lm import (
    capture_training_layer,
    distill_layer_student,
    evaluate_closed_loop,
    evaluate_local_student,
    install_student,
    joint_fine_tune_transplant,
    make_narrow_student,
)


def stable_seed(*parts: object) -> int:
    payload = "::".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**31 - 1)


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
        ":(exclude)data/reality-gate-1a/**",
    ]
    for command in (
        ["git", "diff", "--quiet", "--", *pathspec],
        ["git", "diff", "--cached", "--quiet", "--", *pathspec],
    ):
        if subprocess.run(command, cwd=ROOT, check=False).returncode != 0:
            raise RuntimeError("scientific source/config/test files must be committed and clean")


def _make_corpora(arrays: dict[str, np.ndarray], manifest: dict[str, Any], cfg: dict[str, Any], scale: dict[str, Any]):
    seq_len = int(scale["model"]["seq_len"])
    vocab_size = int(manifest["vocab_size"])
    data = cfg["data"]
    minimum_tokens = seq_len + 2
    train_docs = documents_from_prepared_split(
        arrays["train"],
        prefix="wikitext-train",
        domain="wikitext-train",
        maximum_document_tokens=int(data["train_document_tokens"]),
        minimum_document_tokens=minimum_tokens,
        maximum_total_tokens=int(data["maximum_train_tokens"]),
    )
    calibration_docs = documents_from_prepared_split(
        arrays["validation"],
        prefix="wikitext-calibration",
        domain="wikitext-validation",
        maximum_document_tokens=int(data["calibration_document_tokens"]),
        minimum_document_tokens=minimum_tokens,
        maximum_total_tokens=int(data["maximum_calibration_tokens"]),
    )
    train_corpus = ArrayTokenCorpus(
        {"train": train_docs, "calibration": calibration_docs},
        seq_len=seq_len,
        vocab_size=vocab_size,
    )
    return train_corpus, train_docs, calibration_docs


def _make_eval_corpus(arrays: dict[str, np.ndarray], manifest: dict[str, Any], cfg: dict[str, Any], scale: dict[str, Any]):
    seq_len = int(scale["model"]["seq_len"])
    vocab_size = int(manifest["vocab_size"])
    data = cfg["data"]
    minimum_tokens = seq_len + 2
    hypothesis_docs = documents_from_prepared_split(
        arrays["test"],
        prefix="wikitext-hypothesis",
        domain="wikitext-test",
        maximum_document_tokens=int(data["hypothesis_document_tokens"]),
        minimum_document_tokens=minimum_tokens,
        maximum_total_tokens=int(data["maximum_hypothesis_tokens"]),
    )
    ood_docs = documents_from_prepared_split(
        arrays["ood"],
        prefix="structured-ood",
        domain="structured-ood",
        maximum_document_tokens=int(data["ood_document_tokens"]),
        minimum_document_tokens=minimum_tokens,
        maximum_total_tokens=int(data["maximum_ood_tokens"]),
    )
    corpus = ArrayTokenCorpus(
        {"hypothesis": hypothesis_docs, "ood": ood_docs},
        seq_len=seq_len,
        vocab_size=vocab_size,
    )
    return corpus, hypothesis_docs, ood_docs


def _document_hashes(documents) -> dict[str, str]:
    grouped: dict[str, list] = {}
    for document in documents:
        grouped.setdefault(str(document.document_id), []).append(document)
    result: dict[str, str] = {}
    for document_id, chunks in grouped.items():
        digest = hashlib.sha256()
        for chunk in sorted(chunks, key=lambda value: int(value.source_offset)):
            digest.update(chunk.tokens.numpy().tobytes())
        result[document_id] = digest.hexdigest()
    return result


def _routing_only_utilities(n_experts: int, max_rank: int) -> torch.Tensor:
    # Diminishing utility makes routing frequency the only expert-specific signal.
    values = 1.0 / torch.sqrt(torch.arange(1, max_rank + 1, dtype=torch.float64))
    return values[None, :].repeat(n_experts, 1)


def _allocation(
    teacher_moe: ConventionalSwiGLUMoE,
    frequencies: torch.Tensor,
    *,
    uniform_rank: int,
    max_rank: int,
    kind: str,
) -> tuple[int, ...]:
    if kind == "spectral":
        utilities = residual_mode_utilities(teacher_moe, max_rank=max_rank)
    elif kind == "routing":
        utilities = _routing_only_utilities(teacher_moe.geometry.n_experts, max_rank)
    else:
        raise ValueError(kind)
    return allocate_static_ranks(
        utilities,
        frequencies,
        total_rank_budget=teacher_moe.geometry.n_experts * uniform_rank,
        expected_active_rank_budget=teacher_moe.geometry.top_k * uniform_rank,
        min_rank=int(1),
        max_rank=max_rank,
    )


def _evaluate_module(
    teacher,
    module,
    corpus,
    *,
    scale_name: str,
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
            teacher,
            module,
            corpus,
            split=split,
            layer_id=0,
            windows_per_document=windows_per_document,
            seed=evaluation_seed,
        )
        closed, closed_rows = evaluate_closed_loop(
            teacher,
            installed,
            corpus,
            split=split,
            windows_per_document=windows_per_document,
            seed=evaluation_seed,
        )
        summary[split] = {"local": local, "closed_loop": closed}
        local_index = {
            (str(row["document_id"]), int(row["start"])): float(row["nrmse"])
            for row in local_rows
        }
        for row in closed_rows:
            key = (str(row["document_id"]), int(row["start"]))
            records.append(
                {
                    **row,
                    "scale": scale_name,
                    "seed": int(seed),
                    "candidate": candidate,
                    "phase": phase,
                    "evaluation_split": split,
                    "local_nrmse": local_index[key],
                }
            )
    return summary, records


def _trajectory_cell(
    teacher,
    train_corpus,
    *,
    scale_name: str,
    seed: int,
    label: str,
    uniform_rank: int,
    max_rank: int,
    cfg: dict[str, Any],
) -> dict[str, Any]:
    teacher_moe = teacher.blocks[0].moe
    if not isinstance(teacher_moe, ConventionalSwiGLUMoE):
        raise TypeError("teacher MoE is not conventional")
    data_cfg = cfg["data"]
    training = cfg["training"]
    captured = capture_training_layer(
        teacher,
        train_corpus,
        split="train",
        layer_id=0,
        batches=int(data_cfg["trajectory_capture_batches"]),
        batch_size=int(data_cfg["capture_batch_size"]),
        seed=stable_seed(seed, cfg["protocol_version"], scale_name, label, "trajectory-capture"),
    )
    frequencies = route_frequencies(captured.top_ids, teacher_moe.geometry.n_experts)
    uniform_ranks = uniform_rank_vector(teacher_moe.geometry.n_experts, uniform_rank)
    spectral_ranks = _allocation(
        teacher_moe, frequencies, uniform_rank=uniform_rank, max_rank=max_rank, kind="spectral"
    )
    students = {
        "uniform": HeterogeneousSharedLowRankResidualMoE.from_teacher(
            teacher_moe, ranks=uniform_ranks
        ),
        "heterogeneous-spectral": HeterogeneousSharedLowRankResidualMoE.from_teacher(
            teacher_moe, ranks=spectral_ranks
        ),
    }
    result: dict[str, Any] = {
        "label": label,
        "route_frequencies": [float(value) for value in frequencies],
        "effective_residual_rank_95": effective_residual_rank(teacher_moe),
        "ranks": {
            "uniform": list(uniform_ranks),
            "heterogeneous-spectral": list(spectral_ranks),
        },
        "candidates": {},
    }
    stage_seed = stable_seed(seed, cfg["protocol_version"], scale_name, label, "trajectory-distill")
    for name, student in students.items():
        history = distill_layer_student(
            student,
            captured,
            steps=int(training["trajectory_local_steps"]),
            batch_size=int(training["local_batch_size"]),
            learning_rate=float(teacher.config.student_learning_rate),
            seed=stage_seed,
        )
        local, _ = evaluate_local_student(
            teacher,
            student,
            train_corpus,
            split="calibration",
            layer_id=0,
            windows_per_document=int(data_cfg["trajectory_windows_per_document"]),
            seed=int(data_cfg["calibration_window_seed"]),
        )
        installed = install_student(teacher, student, layer_id=0)
        closed, _ = evaluate_closed_loop(
            teacher,
            installed,
            train_corpus,
            split="calibration",
            windows_per_document=int(data_cfg["trajectory_windows_per_document"]),
            seed=int(data_cfg["calibration_window_seed"]),
        )
        result["candidates"][name] = {
            "history": history,
            "local": local,
            "closed_loop": closed,
            "accounting": student.accounting(
                frequencies, uniform_reference_rank=uniform_rank
            ).as_dict(),
        }
    return result


def _run_pilot(
    teacher,
    train_corpus,
    *,
    scale_name: str,
    seed: int,
    uniform_rank: int,
    max_rank: int,
    plateau: dict[str, Any],
    cfg: dict[str, Any],
    output_dir: Path,
    source_commit: str,
    config_path: Path,
    data_root: Path,
) -> int:
    """Engineering pilot: no test/OOD arrays are opened."""
    teacher_moe = teacher.blocks[0].moe
    if not isinstance(teacher_moe, ConventionalSwiGLUMoE):
        raise TypeError("pilot teacher layer is not conventional")
    data_cfg = cfg["data"]
    captured = capture_training_layer(
        teacher,
        train_corpus,
        split="train",
        layer_id=0,
        batches=int(data_cfg["trajectory_capture_batches"]),
        batch_size=int(data_cfg["capture_batch_size"]),
        seed=stable_seed(seed, cfg["protocol_version"], scale_name, "pilot-capture"),
    )
    frequencies = route_frequencies(captured.top_ids, teacher_moe.geometry.n_experts)
    uniform_ranks = uniform_rank_vector(teacher_moe.geometry.n_experts, uniform_rank)
    spectral_ranks = _allocation(
        teacher_moe, frequencies, uniform_rank=uniform_rank, max_rank=max_rank, kind="spectral"
    )
    students = {
        "uniform-rank": HeterogeneousSharedLowRankResidualMoE.from_teacher(
            teacher_moe, ranks=uniform_ranks
        ),
        "heterogeneous-spectral": HeterogeneousSharedLowRankResidualMoE.from_teacher(
            teacher_moe, ranks=spectral_ranks
        ),
    }
    pilot_result: dict[str, Any] = {
        "protocol_version": cfg["protocol_version"],
        "source_commit": source_commit,
        "configuration_sha256": sha256_file(config_path),
        "data_manifest_sha256": sha256_file(data_root / "manifest.json"),
        "scale": scale_name,
        "seed": seed,
        "plateau": plateau,
        "rank_vectors": {
            "uniform-rank": list(uniform_ranks),
            "heterogeneous-spectral": list(spectral_ranks),
        },
        "candidates": {},
        "heldout_loaded": False,
    }
    pilot_steps = int(cfg["pilot"]["student_steps"])
    pilot_seed = stable_seed(seed, cfg["protocol_version"], scale_name, "pilot-distill")
    for name, student in students.items():
        history = distill_layer_student(
            student,
            captured,
            steps=pilot_steps,
            batch_size=int(cfg["training"]["local_batch_size"]),
            learning_rate=float(teacher.config.student_learning_rate),
            seed=pilot_seed,
        )
        local, _ = evaluate_local_student(
            teacher,
            student,
            train_corpus,
            split="calibration",
            layer_id=0,
            windows_per_document=int(data_cfg["trajectory_windows_per_document"]),
            seed=int(data_cfg["calibration_window_seed"]),
        )
        installed = install_student(teacher, student, layer_id=0)
        closed, _ = evaluate_closed_loop(
            teacher,
            installed,
            train_corpus,
            split="calibration",
            windows_per_document=int(data_cfg["trajectory_windows_per_document"]),
            seed=int(data_cfg["calibration_window_seed"]),
        )
        pilot_result["candidates"][name] = {
            "history": history,
            "local": local,
            "closed_loop": closed,
            "accounting": student.accounting(
                frequencies, uniform_reference_rank=uniform_rank
            ).as_dict(),
        }
    pilot_dir = output_dir / "pilot"
    pilot_dir.mkdir(parents=True, exist_ok=True)
    path = pilot_dir / f"{scale_name}-seed-{seed}.json"
    path.write_text(json.dumps(pilot_result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(path)
    return 0 if bool(plateau["plateau_reached"]) else 3


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scale", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument(
        "--config", type=Path, default=ROOT / "configs/reality_gate_1a.yaml"
    )
    parser.add_argument(
        "--data-root", type=Path, default=ROOT / "data/reality-gate-1a"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "results/reality-gate-1a"
    )
    parser.add_argument("--pilot", action="store_true")
    args = parser.parse_args()

    cfg: dict[str, Any] = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if args.scale not in cfg["scales"]:
        parser.error("scale is not preregistered")
    if args.seed not in [int(value) for value in cfg["seeds"]] and not args.pilot:
        parser.error("seed is not preregistered")
    require_clean()
    source_commit = git_head()
    torch.set_num_threads(int(cfg["threads"]))
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass

    scale = cfg["scales"][args.scale]
    model_cfg = dict(scale["model"])
    plateau_values = dict(scale["plateau"])
    if args.pilot:
        plateau_values["minimum_steps"] = int(cfg["pilot"]["minimum_steps"])
        plateau_values["maximum_steps"] = int(cfg["pilot"]["maximum_steps"])
        plateau_values["evaluation_interval"] = int(cfg["pilot"]["evaluation_interval"])
        plateau_values["patience"] = int(cfg["pilot"]["patience"])
        model_cfg["teacher_steps"] = plateau_values["maximum_steps"]
    tiny = _tiny_config(model_cfg)
    rule = PlateauRule.from_mapping(plateau_values)
    arrays, data_manifest = load_prepared_arrays(
        args.data_root, splits=("train", "validation")
    )
    train_corpus, train_docs, calibration_docs = _make_corpora(
        arrays, data_manifest, cfg, scale
    )

    teacher, teacher_history, snapshots, plateau = train_teacher_to_plateau(
        train_corpus,
        tiny,
        seed=args.seed,
        rule=rule,
        trajectory_fractions=[float(value) for value in scale["trajectory_fractions"]],
        layer_id=0,
    )
    if args.pilot:
        payload = {
            "protocol_version": cfg["protocol_version"],
            "pilot": True,
            "scale": args.scale,
            "seed": args.seed,
            "source_commit": source_commit,
            "plateau": plateau,
            "teacher_history": teacher_history,
            "data_revision": data_manifest["revision"],
        }
        args.output_dir.mkdir(parents=True, exist_ok=True)
        path = args.output_dir / "pilot.json"
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(path)
        return 0

    uniform_rank = int(scale["uniform_rank"])
    max_rank = int(scale["max_rank"])
    if args.pilot:
        return _run_pilot(
            teacher,
            train_corpus,
            scale_name=args.scale,
            seed=args.seed,
            uniform_rank=uniform_rank,
            max_rank=max_rank,
            plateau=plateau,
            cfg=cfg,
            output_dir=args.output_dir,
            source_commit=source_commit,
            config_path=args.config,
            data_root=args.data_root,
        )

    trajectory: list[dict[str, Any]] = []
    for label, state in snapshots.items():
        snapshot_teacher = restore_teacher(
            state, vocab_size=train_corpus.vocab_size, config=tiny
        )
        trajectory.append(
            _trajectory_cell(
                snapshot_teacher,
                train_corpus,
                scale_name=args.scale,
                seed=args.seed,
                label=label,
                uniform_rank=uniform_rank,
                max_rank=max_rank,
                cfg=cfg,
            )
        )

    teacher_moe = teacher.blocks[0].moe
    if not isinstance(teacher_moe, ConventionalSwiGLUMoE):
        raise TypeError("final teacher layer is not conventional")
    data_cfg = cfg["data"]
    captured = capture_training_layer(
        teacher,
        train_corpus,
        split="train",
        layer_id=0,
        batches=int(data_cfg["final_capture_batches"]),
        batch_size=int(data_cfg["capture_batch_size"]),
        seed=stable_seed(args.seed, cfg["protocol_version"], args.scale, "final-capture"),
    )
    frequencies = route_frequencies(captured.top_ids, teacher_moe.geometry.n_experts)
    uniform_ranks = uniform_rank_vector(teacher_moe.geometry.n_experts, uniform_rank)
    spectral_ranks = _allocation(
        teacher_moe, frequencies, uniform_rank=uniform_rank, max_rank=max_rank, kind="spectral"
    )
    routing_ranks = _allocation(
        teacher_moe, frequencies, uniform_rank=uniform_rank, max_rank=max_rank, kind="routing"
    )
    narrow_width = max(1, int(round(teacher_moe.geometry.d_ff * float(scale["narrow_fraction"]))))
    candidates: dict[str, torch.nn.Module] = {
        "uniform-rank": HeterogeneousSharedLowRankResidualMoE.from_teacher(
            teacher_moe, ranks=uniform_ranks
        ),
        "heterogeneous-spectral": HeterogeneousSharedLowRankResidualMoE.from_teacher(
            teacher_moe, ranks=spectral_ranks
        ),
        "heterogeneous-routing": HeterogeneousSharedLowRankResidualMoE.from_teacher(
            teacher_moe, ranks=routing_ranks
        ),
        "narrow65": make_narrow_student(teacher_moe, d_ff=narrow_width),
        "full-identity-control": copy.deepcopy(teacher_moe),
    }
    candidates["full-identity-control"].router.weight.requires_grad_(False)
    initial = {name: copy.deepcopy(module) for name, module in candidates.items()}

    training = cfg["training"]
    local_seed = stable_seed(args.seed, cfg["protocol_version"], args.scale, "final-local")
    joint_seed = stable_seed(args.seed, cfg["protocol_version"], args.scale, "final-joint")
    histories: dict[str, Any] = {}
    for name in ("uniform-rank", "heterogeneous-spectral", "heterogeneous-routing", "narrow65"):
        histories[name] = {
            "local": distill_layer_student(
                candidates[name],
                captured,
                steps=int(training["final_local_steps"]),
                batch_size=int(training["local_batch_size"]),
                learning_rate=float(tiny.student_learning_rate),
                seed=local_seed,
            )
        }
        candidates[name], histories[name]["joint"] = joint_fine_tune_transplant(
            teacher,
            candidates[name],
            train_corpus,
            layer_id=0,
            steps=int(training["joint_steps"]),
            batch_size=int(tiny.batch_size),
            learning_rate=float(tiny.student_learning_rate),
            seed=joint_seed,
            local_weight=float(training["joint_weights"]["local"]),
            kl_weight=float(training["joint_weights"]["kl"]),
            ce_weight=float(training["joint_weights"]["ce"]),
        )
        print(f"trained scale={args.scale} seed={args.seed} candidate={name}", flush=True)
    histories["full-identity-control"] = {"local": [], "joint": []}

    cell_dir = args.output_dir / args.scale
    cell_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = cell_dir / f"frozen-candidates-seed-{args.seed}.pt"
    torch.save(
        {
            "teacher_state": teacher.state_dict(),
            "candidate_states": {name: module.state_dict() for name, module in candidates.items()},
            "initial_states": {name: module.state_dict() for name, module in initial.items()},
            "candidate_types": {name: type(module).__name__ for name, module in candidates.items()},
            "rank_vectors": {
                "uniform-rank": list(uniform_ranks),
                "heterogeneous-spectral": list(spectral_ranks),
                "heterogeneous-routing": list(routing_ranks),
            },
            "seed": args.seed,
            "scale": args.scale,
            "source_commit": source_commit,
            "configuration_sha256": sha256_file(args.config),
            "data_manifest_sha256": sha256_file(args.data_root / "manifest.json"),
            "protocol_version": cfg["protocol_version"],
        },
        checkpoint_path,
    )

    # The test and OOD arrays are loaded into an evaluation corpus only after
    # candidates and checkpoint hashes are frozen.
    heldout_arrays, heldout_manifest = load_prepared_arrays(
        args.data_root, splits=("test", "ood")
    )
    if heldout_manifest != data_manifest:
        raise RuntimeError("data manifest changed after candidate freeze")
    eval_corpus, hypothesis_docs, ood_docs = _make_eval_corpus(
        heldout_arrays, data_manifest, cfg, scale
    )
    train_hashes = set(_document_hashes(train_docs).values())
    calibration_hashes = set(_document_hashes(calibration_docs).values())
    hypothesis_hashes = set(_document_hashes(hypothesis_docs).values())
    ood_hashes = set(_document_hashes(ood_docs).values())
    overlap = {
        "train_calibration": sorted(train_hashes & calibration_hashes),
        "train_hypothesis": sorted(train_hashes & hypothesis_hashes),
        "train_ood": sorted(train_hashes & ood_hashes),
        "calibration_hypothesis": sorted(calibration_hashes & hypothesis_hashes),
        "calibration_ood": sorted(calibration_hashes & ood_hashes),
        "hypothesis_ood": sorted(hypothesis_hashes & ood_hashes),
    }

    evaluations: dict[str, Any] = {}
    records: list[dict[str, Any]] = []
    for name, module in candidates.items():
        evaluations[name] = {}
        for phase, current_module in (("initial", initial[name]), ("final", module)):
            summary, rows = _evaluate_module(
                teacher,
                current_module,
                eval_corpus,
                scale_name=args.scale,
                seed=args.seed,
                candidate=name,
                phase=phase,
                windows_per_document=int(data_cfg["evaluation_windows_per_document"]),
                evaluation_seed=int(data_cfg["evaluation_window_seed"]),
            )
            evaluations[name][phase] = summary
            records.extend(rows)

    train_accounting: dict[str, Any] = {}
    heldout_accounting: dict[str, Any] = {}
    for name, module in candidates.items():
        if isinstance(module, HeterogeneousSharedLowRankResidualMoE):
            train_accounting[name] = module.accounting(
                frequencies, uniform_reference_rank=uniform_rank
            ).as_dict()
            heldout_accounting[name] = {}
            for split in ("hypothesis", "ood"):
                heldout_freq = routing_distribution(
                    teacher,
                    eval_corpus,
                    split=split,
                    layer_id=0,
                    windows_per_document=int(data_cfg["evaluation_windows_per_document"]),
                    seed=int(data_cfg["evaluation_window_seed"]),
                )
                heldout_accounting[name][split] = module.accounting(
                    heldout_freq, uniform_reference_rank=uniform_rank
                ).as_dict()
        elif isinstance(module, ConventionalSwiGLUMoE):
            ratio = module.geometry.d_ff / teacher_moe.geometry.d_ff
            train_accounting[name] = {
                "parameter_ratio": float(ratio),
                "compute_ratio": float(ratio),
                "expected_active_rank": None,
            }
            heldout_accounting[name] = {
                split: dict(train_accounting[name]) for split in ("hypothesis", "ood")
            }

    payload = {
        "metadata": {
            "protocol_version": cfg["protocol_version"],
            "scale": args.scale,
            "seed": args.seed,
            "source_commit": source_commit,
            "configuration_sha256": sha256_file(args.config),
            "data_manifest_sha256": sha256_file(args.data_root / "manifest.json"),
            "data_revision": data_manifest["revision"],
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "plateau": plateau,
            "teacher_history": teacher_history,
            "rank_vectors": {
                "uniform-rank": list(uniform_ranks),
                "heterogeneous-spectral": list(spectral_ranks),
                "heterogeneous-routing": list(routing_ranks),
            },
            "route_frequencies": [float(value) for value in frequencies],
            "train_accounting": train_accounting,
            "heldout_accounting": heldout_accounting,
            "document_hash_overlap": overlap,
            "document_counts": {
                "train_chunks": len(train_docs),
                "calibration_chunks": len(calibration_docs),
                "hypothesis_chunks": len(hypothesis_docs),
                "ood_chunks": len(ood_docs),
                "train_articles": len({doc.document_id for doc in train_docs}),
                "calibration_articles": len({doc.document_id for doc in calibration_docs}),
                "hypothesis_articles": len({doc.document_id for doc in hypothesis_docs}),
                "ood_articles": len({doc.document_id for doc in ood_docs}),
            },
        },
        "trajectory": trajectory,
        "training": histories,
        "evaluations": evaluations,
        "records": records,
    }
    output_path = cell_dir / f"seed-{args.seed}.json"
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
