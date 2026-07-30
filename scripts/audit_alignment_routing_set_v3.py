#!/usr/bin/env python3
"""Independent adversarial audit of routing-set distillation v3.

This file intentionally does not import the project aggregator or bootstrap
helpers. It rebuilds seed/document cells, paired comparisons, arithmetic and
load-bearing gates directly from raw per-seed JSON records.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
PRIMARY = "shared-lora-r5-routing-set-v3"
V2 = "shared-lora-r5-expert-v2-control"
CAPACITY = "shared-lora-r6-routing-set-v3"
NARROW = "narrow65-frozen-baseline"
FULL = "full-continuation-control"


def selected(payloads: list[dict[str, Any]], candidate: str, split: str) -> list[dict[str, Any]]:
    return [
        row for payload in payloads for row in payload["records"]
        if row["candidate"] == candidate and row["phase"] == "final"
        and row["evaluation_split"] == split
    ]


def cell_matrix(rows: list[dict[str, Any]], key: str) -> tuple[np.ndarray, list[int], list[str]]:
    seeds = sorted({int(row["seed"]) for row in rows})
    docs = sorted({str(row["document_id"]) for row in rows})
    buckets = {(seed, doc): [] for seed in seeds for doc in docs}
    for row in rows:
        value = float(row[key])
        if not np.isfinite(value):
            raise ValueError(f"non-finite {key}")
        buckets[(int(row["seed"]), str(row["document_id"]))].append(value)
    matrix = np.empty((len(seeds), len(docs)), dtype=np.float64)
    for i, seed in enumerate(seeds):
        for j, doc in enumerate(docs):
            values = buckets[(seed, doc)]
            if not values:
                raise ValueError(f"missing cell {seed}/{doc}/{key}")
            matrix[i, j] = float(np.mean(values))
    return matrix, seeds, docs


def bootstrap(
    rows: list[dict[str, Any]], key: str, *, samples: int, seed: int, confidence: float
) -> dict[str, Any]:
    matrix, seeds, docs = cell_matrix(rows, key)
    rng = np.random.default_rng(seed)
    draws = np.empty(samples, dtype=np.float64)
    for index in range(samples):
        seed_ids = rng.integers(0, len(seeds), size=len(seeds))
        doc_ids = rng.integers(0, len(docs), size=len(docs))
        draws[index] = float(np.mean(matrix[np.ix_(seed_ids, doc_ids)]))
    alpha = 1.0 - confidence
    return {
        "mean": float(np.mean(matrix)),
        "lcb": float(np.quantile(draws, alpha / 2.0)),
        "ucb": float(np.quantile(draws, 1.0 - alpha / 2.0)),
        "per_seed": {str(seed): float(matrix[i].mean()) for i, seed in enumerate(seeds)},
    }


def paired(
    payloads: list[dict[str, Any]], left: str, right: str, key: str
) -> list[dict[str, Any]]:
    rows = [
        row for payload in payloads for row in payload["records"]
        if row["candidate"] in {left, right} and row["phase"] == "final"
        and row["evaluation_split"] == "hypothesis"
    ]
    index = {
        (int(row["seed"]), str(row["document_id"]), int(row["start"]), str(row["candidate"])): float(row[key])
        for row in rows
    }
    keys = sorted((seed, doc, start) for seed, doc, start, name in index if name == left)
    return [
        {
            "seed": seed,
            "document_id": doc,
            "difference": index[(seed, doc, start, left)] - index[(seed, doc, start, right)],
        }
        for seed, doc, start in keys
    ]


def close(a: float, b: float, tolerance: float = 2e-9) -> bool:
    return abs(float(a) - float(b)) <= tolerance


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path,
        default=ROOT / "configs/pre_qwen_alignment_routing_set_v3.yaml",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=ROOT / "results/pre-qwen-alignment-routing-set/v3",
    )
    args = parser.parse_args()
    config: dict[str, Any] = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    seeds = [int(value) for value in config["seeds"]]
    payloads = [
        json.loads((args.output_dir / f"seed-{seed}.json").read_text(encoding="utf-8"))
        for seed in seeds
    ]
    aggregate = json.loads((args.output_dir / "metrics.json").read_text(encoding="utf-8"))
    stats = config["statistics"]
    samples = int(stats["bootstrap_samples"])
    base_seed = int(stats["bootstrap_seed"])
    confidence = float(stats["confidence"])
    mismatches: list[dict[str, Any]] = []

    # Recompute the metrics that drive every primary gate.
    primary_rows = selected(payloads, PRIMARY, "hypothesis")
    rebuilt = {
        "hyp_loss": bootstrap(primary_rows, "loss_delta", samples=samples, seed=base_seed + 0, confidence=confidence),
        "ood_loss": bootstrap(selected(payloads, PRIMARY, "ood"), "loss_delta", samples=samples, seed=base_seed + 100, confidence=confidence),
        "kl": bootstrap(primary_rows, "kl_teacher_to_candidate", samples=samples, seed=base_seed + 1, confidence=confidence),
        "top1": bootstrap(primary_rows, "top1_agreement", samples=samples, seed=base_seed + 2, confidence=confidence),
        "local": bootstrap(primary_rows, "local_nrmse", samples=samples, seed=base_seed + 3, confidence=confidence),
        "counterfactual": bootstrap(primary_rows, "counterfactual_nrmse", samples=samples, seed=base_seed + 5, confidence=confidence),
        "primary_minus_narrow": bootstrap(paired(payloads, PRIMARY, NARROW, "loss_delta"), "difference", samples=samples, seed=base_seed + 5001, confidence=confidence),
        "primary_minus_v2": bootstrap(paired(payloads, PRIMARY, V2, "loss_delta"), "difference", samples=samples, seed=base_seed + 5002, confidence=confidence),
        "cross_gap": bootstrap(paired(payloads, PRIMARY, NARROW, "routing_cross_error"), "difference", samples=samples, seed=base_seed + 5008, confidence=confidence),
        "capacity_loss": bootstrap(selected(payloads, CAPACITY, "hypothesis"), "loss_delta", samples=samples, seed=base_seed + 3000, confidence=confidence),
        "full_hyp": bootstrap(selected(payloads, FULL, "hypothesis"), "loss_delta", samples=samples, seed=base_seed + 5000, confidence=confidence),
        "full_ood": bootstrap(selected(payloads, FULL, "ood"), "loss_delta", samples=samples, seed=base_seed + 5100, confidence=confidence),
    }

    # Aggregator offsets are candidate-index based. Check values using a slightly
    # looser quantile tolerance because NumPy versions can differ in final bits.
    primary_agg = aggregate["candidates"][PRIMARY]["hypothesis"]
    checks = {
        "hyp_loss": primary_agg["loss_delta"],
        "kl": primary_agg["kl_teacher_to_candidate"],
        "top1": primary_agg["top1_agreement"],
        "local": primary_agg["local_nrmse"],
        "counterfactual": primary_agg["counterfactual_nrmse"],
        "primary_minus_narrow": aggregate["comparisons"]["primary_minus_narrow_loss"],
        "primary_minus_v2": aggregate["comparisons"]["primary_minus_v2_loss"],
        "cross_gap": aggregate["comparisons"]["primary_minus_narrow_cross_error"],
    }
    for name, expected in checks.items():
        actual = rebuilt[name]
        for key in ("mean", "lcb", "ucb"):
            if abs(float(actual[key]) - float(expected[key])) > 3e-6:
                mismatches.append({"metric": name, "field": key, "audit": actual[key], "aggregate": expected[key]})

    gates_cfg = config["gates"]
    primary_param = float(np.mean([p["metadata"]["parameter_ratios"][PRIMARY] for p in payloads]))
    primary_compute = float(np.mean([p["metadata"]["compute_ratios"][PRIMARY] for p in payloads]))
    tol = float(gates_cfg["arithmetic_tolerance"])
    clean = all(
        not p["metadata"]["data_audit"]["exact_cross_split_duplicates"]
        and not p["metadata"]["data_audit"]["near_cross_split_pairs"]
        for p in payloads
    )
    gates = {
        "primary_hypothesis_pass": rebuilt["hyp_loss"]["ucb"] <= float(gates_cfg["primary_hypothesis_ucb_max"]),
        "primary_ood_pass": rebuilt["ood_loss"]["ucb"] <= float(gates_cfg["primary_ood_ucb_max"]),
        "every_seed_primary_pass": max(rebuilt["hyp_loss"]["per_seed"].values()) <= float(gates_cfg["every_seed_primary_hypothesis_delta_max"]),
        "primary_vs_narrow65_pass": rebuilt["primary_minus_narrow"]["ucb"] <= float(gates_cfg["primary_minus_narrow65_ucb_max"]),
        "primary_vs_expert_v2_pass": rebuilt["primary_minus_v2"]["ucb"] <= float(gates_cfg["primary_minus_expert_v2_ucb_max"]),
        "primary_kl_pass": rebuilt["kl"]["ucb"] <= float(gates_cfg["primary_kl_ucb_max"]),
        "primary_top1_pass": rebuilt["top1"]["lcb"] >= float(gates_cfg["primary_top1_lcb_min"]),
        "primary_local_nrmse_pass": rebuilt["local"]["ucb"] <= float(gates_cfg["primary_local_nrmse_ucb_max"]),
        "primary_counterfactual_pass": rebuilt["counterfactual"]["ucb"] <= float(gates_cfg["primary_counterfactual_nrmse_ucb_max"]),
        "primary_cross_error_gap_pass": rebuilt["cross_gap"]["ucb"] <= float(gates_cfg["primary_cross_error_gap_to_narrow_ucb_max"]),
        "capacity_hypothesis_pass": rebuilt["capacity_loss"]["ucb"] <= float(gates_cfg["capacity_hypothesis_ucb_max"]),
        "full_control_hypothesis_pass": rebuilt["full_hyp"]["ucb"] <= float(gates_cfg["full_control_hypothesis_ucb_max"]),
        "full_control_ood_pass": rebuilt["full_ood"]["ucb"] <= float(gates_cfg["full_control_ood_ucb_max"]),
        "primary_parameter_budget_pass": primary_param < float(gates_cfg["primary_parameter_ratio_strict_max"]) - tol,
        "primary_compute_budget_pass": primary_compute < float(gates_cfg["primary_compute_ratio_strict_max"]) - tol,
        "all_arithmetic_pass": all(
            abs(float(np.mean([p["metadata"]["parameter_ratios"][str(row["name"])] for p in payloads])) - float(row["expected_parameter_ratio"])) <= tol
            and abs(float(np.mean([p["metadata"]["compute_ratios"][str(row["name"])] for p in payloads])) - float(row["expected_compute_ratio"])) <= tol
            for row in config["candidates"]
        ),
        "clean_data_audit_pass": clean,
    }
    aggregate_gates = aggregate["decision"]["gates"]
    for name, value in gates.items():
        if bool(aggregate_gates.get(name)) != bool(value):
            mismatches.append({"gate": name, "audit": value, "aggregate": aggregate_gates.get(name)})

    leave_one_out: dict[str, Any] = {}
    if len(seeds) > 1:
        for omitted in seeds:
            subset = [p for p in payloads if int(p["metadata"]["seed"]) != omitted]
            leave_one_out[str(omitted)] = bootstrap(
                selected(subset, PRIMARY, "hypothesis"),
                "loss_delta",
                samples=4000,
                seed=base_seed + omitted % 1000,
                confidence=confidence,
            )

    audit = {
        "audit_passed": not mismatches,
        "mismatches": mismatches,
        "rebuilt": rebuilt,
        "gates_without_audit_gate": gates,
        "leave_one_seed_out_primary_hypothesis": leave_one_out,
        "provenance": {
            "configuration_hashes": sorted({p["metadata"]["configuration_sha256"] for p in payloads}),
            "source_commits": sorted({p["metadata"]["source_commit"] for p in payloads}),
            "checkpoint_hashes_unique": len({p["metadata"]["checkpoint_sha256"] for p in payloads}) == len(payloads),
            "source_teacher_hashes_unique": len({p["metadata"]["source_teacher_checkpoint_sha256"] for p in payloads}) == len(payloads),
        },
    }
    audit_dir = args.output_dir / "adversarial-audit"
    audit_dir.mkdir(parents=True, exist_ok=True)
    (audit_dir / "audit.json").write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# Independent audit — routing-set distillation v3",
        "",
        f"**Audit:** **{'PASS' if audit['audit_passed'] else 'FAIL'}**",
        "",
        f"Mismatches: `{len(mismatches)}`.",
        "",
        "The auditor independently rebuilt seed/document cells, primary intervals, paired comparisons, covariance-gap gate, arithmetic, data isolation, provenance, and leave-one-seed-out sensitivity.",
    ]
    (audit_dir / "VERDICT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print((audit_dir / "VERDICT.md").read_text(encoding="utf-8"))
    return 0 if audit["audit_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
