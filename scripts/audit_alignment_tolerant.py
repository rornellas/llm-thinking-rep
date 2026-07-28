#!/usr/bin/env python3
"""Independent recalculation for the alignment-tolerant shared-factor screen."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def seed_payload_path(output_dir: Path, seed: int) -> Path:
    nested = output_dir / f"seed-{seed}" / f"seed-{seed}.json"
    if nested.exists():
        return nested
    return output_dir / f"seed-{seed}.json"


def crossed_matrix(
    records: Sequence[Mapping[str, object]], value_key: str
) -> tuple[np.ndarray, list[object], list[object]]:
    seeds = sorted({row["seed"] for row in records}, key=str)
    documents = sorted({row["document_id"] for row in records}, key=str)
    seed_index = {value: index for index, value in enumerate(seeds)}
    document_index = {value: index for index, value in enumerate(documents)}
    buckets: list[list[list[float]]] = [
        [[] for _ in documents] for _ in seeds
    ]
    for row in records:
        value = float(row[value_key])
        if not np.isfinite(value):
            raise ValueError(f"non-finite record: {row}")
        buckets[seed_index[row["seed"]]][document_index[row["document_id"]]].append(value)
    matrix = np.full((len(seeds), len(documents)), np.nan, dtype=np.float64)
    for seed_id, seed_buckets in enumerate(buckets):
        for document_id, values in enumerate(seed_buckets):
            if values:
                matrix[seed_id, document_id] = float(np.mean(values))
    return matrix, seeds, documents


def independent_bootstrap(
    records: Sequence[Mapping[str, object]],
    *,
    value_key: str,
    samples: int,
    random_seed: int,
    confidence: float,
) -> dict[str, Any]:
    matrix, seeds, documents = crossed_matrix(records, value_key)
    observed = float(np.nanmean(matrix))
    rng = np.random.default_rng(random_seed)
    draws = np.empty(samples, dtype=np.float64)
    for index in range(samples):
        seed_ids = rng.integers(0, matrix.shape[0], size=matrix.shape[0])
        document_ids = rng.integers(0, matrix.shape[1], size=matrix.shape[1])
        draws[index] = float(np.nanmean(matrix[np.ix_(seed_ids, document_ids)]))
    alpha = 1.0 - confidence
    return {
        "mean": observed,
        "lcb": float(np.quantile(draws, alpha / 2.0)),
        "ucb": float(np.quantile(draws, 1.0 - alpha / 2.0)),
        "effective_cells": int(np.isfinite(matrix).sum()),
        "training_seeds": [str(value) for value in seeds],
        "documents": [str(value) for value in documents],
    }


def selected(
    payloads: list[dict[str, Any]], candidate: str, split: str
) -> list[dict[str, Any]]:
    return [
        row
        for payload in payloads
        for row in payload["records"]
        if row["candidate"] == candidate
        and row["phase"] == "final"
        and row["evaluation_split"] == split
    ]


def paired(
    payloads: list[dict[str, Any]], left: str, right: str
) -> list[dict[str, Any]]:
    rows = [
        row
        for payload in payloads
        for row in payload["records"]
        if row["candidate"] in {left, right}
        and row["phase"] == "final"
        and row["evaluation_split"] == "hypothesis"
    ]
    index = {
        (
            int(row["seed"]),
            str(row["document_id"]),
            int(row["start"]),
            str(row["candidate"]),
        ): float(row["loss_delta"])
        for row in rows
    }
    keys = sorted(
        (seed, document, start)
        for seed, document, start, candidate in index
        if candidate == left
    )
    return [
        {
            "seed": seed,
            "document_id": document,
            "start": start,
            "difference": index[(seed, document, start, left)]
            - index[(seed, document, start, right)],
        }
        for seed, document, start in keys
    ]


def close(left: float, right: float, tolerance: float = 1e-10) -> bool:
    return abs(float(left) - float(right)) <= tolerance


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/pre_qwen_alignment_tolerant_v1.yaml",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "results/pre-qwen-alignment-tolerant/v1",
    )
    parser.add_argument("--allow-provisional-decision", action="store_true")
    args = parser.parse_args()

    config: dict[str, Any] = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    metrics = json.loads((args.output_dir / "metrics.json").read_text(encoding="utf-8"))
    seeds = [int(value) for value in config["seeds"]]
    payloads = [
        json.loads(seed_payload_path(args.output_dir, seed).read_text(encoding="utf-8"))
        for seed in seeds
    ]
    names = [str(row["name"]) for row in config["candidates"]]
    stats = config["statistics"]
    mismatches: list[str] = []
    checks: dict[str, bool] = {}

    actual_config_hash = sha256_file(args.config)
    checks["config_hash_consistent"] = all(
        payload["metadata"]["configuration_sha256"] == actual_config_hash
        for payload in payloads
    ) and metrics["metadata"]["configuration_sha256"] == actual_config_hash
    if not checks["config_hash_consistent"]:
        mismatches.append("configuration hash mismatch")

    checks["candidate_checkpoint_hashes"] = True
    checks["source_checkpoint_hashes"] = True
    checks["clean_data_audit"] = True
    for seed, payload in zip(seeds, payloads, strict=True):
        seed_dir = seed_payload_path(args.output_dir, seed).parent
        candidate_checkpoint = seed_dir / f"frozen-candidates-seed-{seed}.pt"
        if not candidate_checkpoint.exists() or sha256_file(candidate_checkpoint) != payload["metadata"]["checkpoint_sha256"]:
            checks["candidate_checkpoint_hashes"] = False
        source_checkpoint = ROOT / payload["metadata"]["source_teacher_checkpoint"]
        if not source_checkpoint.exists() or sha256_file(source_checkpoint) != payload["metadata"]["source_teacher_checkpoint_sha256"]:
            checks["source_checkpoint_hashes"] = False
        audit = payload["metadata"]["data_audit"]
        if audit["exact_cross_split_duplicates"] or audit["near_cross_split_pairs"]:
            checks["clean_data_audit"] = False
    for name in (
        "candidate_checkpoint_hashes",
        "source_checkpoint_hashes",
        "clean_data_audit",
    ):
        if not checks[name]:
            mismatches.append(name)

    recomputed: dict[str, Any] = {}
    for index, name in enumerate(names):
        recomputed[name] = {}
        for split, offset in (("hypothesis", 100 + index), ("ood", 200 + index)):
            rows = selected(payloads, name, split)
            result = independent_bootstrap(
                rows,
                value_key="loss_delta",
                samples=int(stats["bootstrap_samples"]),
                random_seed=int(stats["bootstrap_seed"]) + offset,
                confidence=float(stats["confidence"]),
            )
            per_seed = {
                str(seed): float(
                    np.mean(
                        [
                            float(row["loss_delta"])
                            for row in rows
                            if int(row["seed"]) == seed
                        ]
                    )
                )
                for seed in seeds
            }
            result["per_seed"] = per_seed
            recomputed[name][split] = result
            reported = metrics["candidates"][name][split]
            for key in ("mean", "lcb", "ucb"):
                if not close(result[key], reported[key]):
                    mismatches.append(f"{name}/{split}/{key}")
            for seed in seeds:
                if not close(per_seed[str(seed)], reported["per_seed"][str(seed)]):
                    mismatches.append(f"{name}/{split}/per_seed/{seed}")

        parameter_values = [
            float(payload["metadata"]["parameter_ratios"][name]) for payload in payloads
        ]
        compute_values = [
            float(payload["metadata"]["compute_ratios"][name]) for payload in payloads
        ]
        if not close(float(np.mean(parameter_values)), metrics["candidates"][name]["parameter_ratio"]):
            mismatches.append(f"{name}/parameter_ratio")
        if not close(float(np.mean(compute_values)), metrics["candidates"][name]["compute_ratio"]):
            mismatches.append(f"{name}/compute_ratio")

    comparisons = {
        "primary_minus_narrow65": (
            "shared-lora-r5",
            "narrow65-frozen-baseline",
            500,
        ),
        "capacity_minus_narrow65": (
            "shared-lora-r6",
            "narrow65-frozen-baseline",
            501,
        ),
    }
    recomputed_comparisons: dict[str, Any] = {}
    for label, (left, right, offset) in comparisons.items():
        result = independent_bootstrap(
            paired(payloads, left, right),
            value_key="difference",
            samples=int(stats["bootstrap_samples"]),
            random_seed=int(stats["bootstrap_seed"]) + offset,
            confidence=float(stats["confidence"]),
        )
        recomputed_comparisons[label] = result
        reported = metrics["comparisons"][label]
        for key in ("mean", "lcb", "ucb"):
            if not close(result[key], reported[key]):
                mismatches.append(f"{label}/{key}")

    expected_rows = {str(row["name"]): row for row in config["candidates"]}
    arithmetic_checks: dict[str, bool] = {}
    arithmetic_tolerance = float(config["gates"]["arithmetic_tolerance"])
    for name in names:
        parameter_ratio = float(metrics["candidates"][name]["parameter_ratio"])
        compute_ratio = float(metrics["candidates"][name]["compute_ratio"])
        arithmetic_checks[name] = (
            abs(parameter_ratio - float(expected_rows[name]["expected_parameter_ratio"]))
            <= arithmetic_tolerance
            and abs(compute_ratio - float(expected_rows[name]["expected_compute_ratio"]))
            <= arithmetic_tolerance
        )
        if arithmetic_checks[name] != bool(metrics["arithmetic_checks"][name]):
            mismatches.append(f"{name}/arithmetic_check")

    if not args.allow_provisional_decision:
        gates_cfg = config["gates"]
        primary = metrics["candidates"]["shared-lora-r5"]
        capacity = metrics["candidates"]["shared-lora-r6"]
        full = metrics["candidates"]["full-continuation-control"]
        expected_gates = {
            "primary_hypothesis_pass": primary["hypothesis"]["ucb"] <= float(gates_cfg["primary_hypothesis_ucb_max"]),
            "primary_ood_pass": primary["ood"]["ucb"] <= float(gates_cfg["primary_ood_ucb_max"]),
            "every_seed_primary_pass": max(primary["hypothesis"]["per_seed"].values()) <= float(gates_cfg["every_seed_primary_hypothesis_delta_max"]),
            "primary_vs_narrow65_pass": metrics["comparisons"]["primary_minus_narrow65"]["ucb"] <= float(gates_cfg["primary_minus_narrow65_ucb_max"]),
            "primary_parameter_budget_pass": primary["parameter_ratio"] < float(gates_cfg["primary_parameter_ratio_strict_max"]),
            "primary_compute_budget_pass": primary["compute_ratio"] < float(gates_cfg["primary_compute_ratio_strict_max"]),
            "capacity_hypothesis_pass": capacity["hypothesis"]["ucb"] <= float(gates_cfg["capacity_hypothesis_ucb_max"]),
            "capacity_vs_narrow65_pass": metrics["comparisons"]["capacity_minus_narrow65"]["ucb"] <= float(gates_cfg["capacity_minus_narrow65_ucb_max"]),
            "full_control_hypothesis_pass": full["hypothesis"]["ucb"] <= float(gates_cfg["full_control_hypothesis_ucb_max"]),
            "full_control_ood_pass": full["ood"]["ucb"] <= float(gates_cfg["full_control_ood_ucb_max"]),
            "all_arithmetic_pass": all(arithmetic_checks.values()),
            "clean_data_audit_pass": checks["clean_data_audit"],
            "independent_audit_pass": True,
        }
        for name, value in expected_gates.items():
            if bool(metrics["decision"]["gates"][name]) != value:
                mismatches.append(f"decision gate {name}")
        primary_keys = (
            "primary_hypothesis_pass",
            "primary_ood_pass",
            "every_seed_primary_pass",
            "primary_vs_narrow65_pass",
            "primary_parameter_budget_pass",
            "primary_compute_budget_pass",
            "full_control_hypothesis_pass",
            "full_control_ood_pass",
            "all_arithmetic_pass",
            "clean_data_audit_pass",
            "independent_audit_pass",
        )
        promising_keys = (
            "capacity_hypothesis_pass",
            "capacity_vs_narrow65_pass",
            "full_control_hypothesis_pass",
            "full_control_ood_pass",
            "all_arithmetic_pass",
            "clean_data_audit_pass",
            "independent_audit_pass",
        )
        if all(expected_gates[key] for key in primary_keys):
            expected_verdict = "ALIGNMENT_TOLERANT_SHARED_LORA_PASS"
        elif all(expected_gates[key] for key in promising_keys):
            expected_verdict = "ALIGNMENT_TOLERANT_SHARED_LORA_PROMISING_ONLY"
        else:
            expected_verdict = "ALIGNMENT_TOLERANT_SHARED_LORA_FAIL"
        if metrics["decision"]["verdict"] != expected_verdict:
            mismatches.append("final verdict")

    checks["statistics_recomputed"] = not any(
        "/mean" in item
        or "/lcb" in item
        or "/ucb" in item
        or "/per_seed/" in item
        for item in mismatches
    )
    checks["ratios_recomputed"] = not any(
        item.endswith("parameter_ratio") or item.endswith("compute_ratio")
        for item in mismatches
    )
    audit_passed = not mismatches
    result = {
        "audit_passed": audit_passed,
        "mode": "provisional-statistics" if args.allow_provisional_decision else "final-decision",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "checks": checks,
        "mismatches": mismatches,
        "recomputed": {
            "candidates": recomputed,
            "comparisons": recomputed_comparisons,
            "arithmetic_checks": arithmetic_checks,
        },
    }
    audit_dir = args.output_dir / "adversarial-audit"
    audit_dir.mkdir(parents=True, exist_ok=True)
    (audit_dir / "audit.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    verdict = "PASS" if audit_passed else "FAIL"
    (audit_dir / "VERDICT.md").write_text(
        "\n".join(
            [
                "# Independent audit — alignment-tolerant v1",
                "",
                f"**Audit:** **{verdict}**",
                f"**Mode:** `{result['mode']}`",
                f"**Mismatches:** `{len(mismatches)}`",
                "",
                *[f"- `{item}`" for item in mismatches],
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print((audit_dir / "VERDICT.md").read_text(encoding="utf-8"))
    return 0 if audit_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
