#!/usr/bin/env python3
"""Run one scale/seed cell of Native Compact Gate 2A."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Iterable

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pre_qwen_certification.native_compact import (
    CANDIDATES,
    NativeArchitectureSpec,
    build_paired_candidate_models,
    candidate_accounting,
    evaluate_native_model,
    restore_phase_models,
    route_health,
    train_matched_candidates,
)
from pre_qwen_certification.reality_gate import routing_distribution
from pre_qwen_certification.reality_gate_data import (
    ArrayTokenCorpus,
    TokenDocument,
    documents_from_prepared_split,
    load_prepared_arrays,
    sha256_file,
)
from pre_qwen_certification.tiny_lm import TinyLMConfig


def git_head() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


def require_clean() -> None:
    pathspec = [
        ".",
        ":(exclude)data/native-compact-gate-2a/**",
        ":(exclude)results/native-compact-gate-2a/**",
        ":(exclude)runs/native-compact-gate-2a/**",
    ]
    for command in (
        ["git", "diff", "--quiet", "--", *pathspec],
        ["git", "diff", "--cached", "--quiet", "--", *pathspec],
    ):
        if subprocess.run(command, cwd=ROOT, check=False).returncode != 0:
            raise RuntimeError("scientific source/config/test files must be committed and clean")


def _tiny_config(values: dict[str, Any], *, steps: int | None = None) -> TinyLMConfig:
    return TinyLMConfig(
        seq_len=int(values["seq_len"]),
        batch_size=int(values["batch_size"]),
        d_model=int(values["d_model"]),
        n_heads=int(values["n_heads"]),
        n_layers=int(values["n_layers"]),
        d_ff=int(values["d_ff"]),
        n_experts=int(values["n_experts"]),
        top_k=int(values["top_k"]),
        teacher_steps=int(values["training_steps"] if steps is None else steps),
        student_steps=0,
        learning_rate=float(values["learning_rate"]),
        student_learning_rate=float(values["learning_rate"]),
        weight_decay=float(values["weight_decay"]),
        aux_weight=float(values["aux_weight"]),
        grad_clip=float(values["grad_clip"]),
    )


def _training_corpus(arrays, manifest, cfg, scale_cfg):
    data = cfg["data"]
    seq_len = int(scale_cfg["model"]["seq_len"])
    minimum = seq_len + 2
    train_docs = documents_from_prepared_split(
        arrays["train"],
        prefix="native-train",
        domain="wikitext103-train",
        maximum_document_tokens=int(data["maximum_document_tokens"]),
        minimum_document_tokens=minimum,
        maximum_total_tokens=int(data["maximum_train_tokens"]),
    )
    calibration_docs = documents_from_prepared_split(
        arrays["validation"],
        prefix="native-calibration",
        domain="wikitext103-validation",
        maximum_document_tokens=int(data["maximum_document_tokens"]),
        minimum_document_tokens=minimum,
        maximum_total_tokens=int(data["maximum_calibration_tokens"]),
    )
    corpus = ArrayTokenCorpus(
        {"train": train_docs, "calibration": calibration_docs},
        seq_len=seq_len,
        vocab_size=int(manifest["vocab_size"]),
    )
    return corpus, train_docs, calibration_docs


def _evaluation_corpus(arrays, manifest, cfg, scale_cfg):
    data = cfg["data"]
    seq_len = int(scale_cfg["model"]["seq_len"])
    minimum = seq_len + 2
    hypothesis_docs = documents_from_prepared_split(
        arrays["test"],
        prefix="native-hypothesis",
        domain="wikitext103-test",
        maximum_document_tokens=int(data["maximum_document_tokens"]),
        minimum_document_tokens=minimum,
        maximum_total_tokens=int(data["maximum_hypothesis_tokens"]),
    )
    ood_docs = documents_from_prepared_split(
        arrays["ood"],
        prefix="native-ood",
        domain="structured-ood-v2",
        maximum_document_tokens=int(data["maximum_document_tokens"]),
        minimum_document_tokens=minimum,
        maximum_total_tokens=int(data["maximum_ood_tokens"]),
    )
    corpus = ArrayTokenCorpus(
        {"hypothesis": hypothesis_docs, "ood": ood_docs},
        seq_len=seq_len,
        vocab_size=int(manifest["vocab_size"]),
    )
    return corpus, hypothesis_docs, ood_docs


def _hash_documents(documents: Iterable[TokenDocument]) -> set[str]:
    return {
        hashlib.sha256(document.tokens.numpy().tobytes()).hexdigest()
        for document in documents
    }


def _overlaps(groups: dict[str, list[TokenDocument]]) -> dict[str, bool]:
    hashes = {name: _hash_documents(documents) for name, documents in groups.items()}
    names = list(hashes)
    result: dict[str, bool] = {}
    for left_index, left in enumerate(names):
        for right in names[left_index + 1 :]:
            result[f"{left}__{right}"] = bool(hashes[left] & hashes[right])
    return result


def _heldout_route_health(model, corpus, split: str, cfg: dict[str, Any]) -> dict[str, Any]:
    layers: dict[str, Any] = {}
    for layer_id in range(int(model.config.n_layers)):
        distribution = routing_distribution(
            model,
            corpus,
            split=split,
            layer_id=layer_id,
            windows_per_document=int(cfg["evaluation"]["windows_per_document"]),
            seed=int(cfg["evaluation"]["window_seed"]),
        )
        layers[str(layer_id)] = {
            "distribution": [float(value) for value in distribution],
            "health": route_health(distribution, model.config.top_k),
        }
    return {
        "layers": layers,
        "health": {
            "dead_experts": max(
                int(value["health"]["dead_experts"]) for value in layers.values()
            ),
            "minimum_normalized_entropy": min(
                float(value["health"]["normalized_entropy"]) for value in layers.values()
            ),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scale", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/native_compact_gate_2a.yaml",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=ROOT / "data/native-compact-gate-2a",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "results/native-compact-gate-2a",
    )
    parser.add_argument("--pilot", action="store_true")
    args = parser.parse_args()

    require_clean()
    cfg: dict[str, Any] = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if args.scale not in cfg["scales"]:
        raise ValueError(f"unknown scale: {args.scale}")
    if not args.pilot and int(args.seed) not in [int(value) for value in cfg["seeds"]]:
        raise ValueError("scientific seed is not preregistered")
    scale_cfg = cfg["scales"][args.scale]
    steps = int(cfg["pilot"]["steps"] if args.pilot else scale_cfg["model"]["training_steps"])
    model_cfg = _tiny_config(scale_cfg["model"], steps=steps)
    spec = NativeArchitectureSpec(
        native_rank=int(scale_cfg["native_rank"]),
        narrow_d_ff=int(scale_cfg["narrow_d_ff"]),
    )
    train_arrays, manifest = load_prepared_arrays(
        args.data_root,
        splits=("train", "validation"),
    )
    training_corpus, train_docs, calibration_docs = _training_corpus(
        train_arrays, manifest, cfg, scale_cfg
    )
    models = build_paired_candidate_models(
        training_corpus.vocab_size,
        model_cfg,
        spec,
        seed=int(args.seed),
    )
    accounting = {
        name: value.as_dict()
        for name, value in candidate_accounting(models, model_cfg).items()
    }
    training = train_matched_candidates(
        models,
        training_corpus,
        model_cfg,
        steps=steps,
        evaluation_interval=int(
            cfg["pilot"]["evaluation_interval"]
            if args.pilot
            else scale_cfg["evaluation_interval"]
        ),
        calibration_windows_per_document=int(
            cfg["evaluation"]["calibration_windows_per_document"]
        ),
        calibration_seed=int(cfg["evaluation"]["calibration_seed"]),
        seed=int(args.seed),
    )
    source_commit = git_head()
    common_metadata = {
        "protocol_version": str(cfg["protocol_version"]),
        "source_commit": source_commit,
        "configuration_sha256": sha256_file(args.config),
        "data_manifest_sha256": sha256_file(args.data_root / "manifest.json"),
        "scale": str(args.scale),
        "seed": int(args.seed),
        "model": dict(scale_cfg["model"]),
        "native_rank": int(spec.native_rank),
        "narrow_d_ff": int(spec.narrow_d_ff),
        "accounting": accounting,
        "training": {
            "steps": int(training["steps"]),
            "batch_size": int(training["batch_size"]),
            "training_tokens_per_candidate": int(training["training_tokens_per_candidate"]),
            "best_step": training["best_step"],
            "best_calibration_loss": training["best_calibration_loss"],
            "histories": training["histories"],
        },
    }

    if args.pilot:
        pilot_dir = args.output_dir / "pilot"
        pilot_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "metadata": {
                **common_metadata,
                "pilot": True,
                "heldout_loaded": False,
            }
        }
        path = pilot_dir / f"{args.scale}-seed-{args.seed}.json"
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(path)
        return 0

    frozen = {
        "protocol_version": str(cfg["protocol_version"]),
        "source_commit": source_commit,
        "scale": str(args.scale),
        "seed": int(args.seed),
        "final_states": training["final_states"],
        "best_states": training["best_states"],
        "best_step": training["best_step"],
    }
    scale_dir = args.output_dir / args.scale
    scale_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = scale_dir / f"frozen-candidates-seed-{args.seed}.pt"
    torch.save(frozen, checkpoint_path)
    checkpoint_hash = sha256_file(checkpoint_path)

    heldout_arrays, heldout_manifest = load_prepared_arrays(
        args.data_root,
        splits=("test", "ood"),
    )
    if heldout_manifest != manifest:
        raise RuntimeError("data manifest changed between training and heldout opening")
    evaluation_corpus, hypothesis_docs, ood_docs = _evaluation_corpus(
        heldout_arrays, manifest, cfg, scale_cfg
    )
    phases = {
        "final": restore_phase_models(models, training["final_states"]),
        "best-calibration": restore_phase_models(models, training["best_states"]),
    }
    summaries: dict[str, Any] = {}
    records: list[dict[str, Any]] = []
    routing: dict[str, Any] = {}
    for phase, phase_models in phases.items():
        summaries[phase] = {}
        routing[phase] = {}
        for candidate in CANDIDATES:
            summaries[phase][candidate] = {}
            routing[phase][candidate] = {}
            for split in ("hypothesis", "ood"):
                summary, rows = evaluate_native_model(
                    phase_models[candidate],
                    evaluation_corpus,
                    split=split,
                    windows_per_document=int(cfg["evaluation"]["windows_per_document"]),
                    evaluation_seed=int(cfg["evaluation"]["window_seed"]),
                )
                summaries[phase][candidate][split] = summary
                routing[phase][candidate][split] = _heldout_route_health(
                    phase_models[candidate], evaluation_corpus, split, cfg
                )
                for row in rows:
                    records.append(
                        {
                            **row,
                            "scale": str(args.scale),
                            "seed": int(args.seed),
                            "candidate": candidate,
                            "phase": phase,
                            "evaluation_split": split,
                        }
                    )

    payload = {
        "metadata": {
            **common_metadata,
            "pilot": False,
            "heldout_loaded_after_candidate_freeze": True,
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_sha256": checkpoint_hash,
            "document_hash_overlap": _overlaps(
                {
                    "train": train_docs,
                    "calibration": calibration_docs,
                    "hypothesis": hypothesis_docs,
                    "ood": ood_docs,
                }
            ),
            "routing": routing,
        },
        "summaries": summaries,
        "records": records,
    }
    result_path = scale_dir / f"seed-{args.seed}.json"
    result_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(result_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
