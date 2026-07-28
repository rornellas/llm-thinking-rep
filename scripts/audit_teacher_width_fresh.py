#!/usr/bin/env python3
"""Independent adversarial reconstruction of the fresh width replication.

This implementation intentionally does not import the project aggregator or its
bootstrap helpers.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]


def matrix(records: list[dict[str, Any]], key: str) -> tuple[np.ndarray, list[int], list[str]]:
    seeds = sorted({int(row["seed"]) for row in records})
    documents = sorted({str(row["document_id"]) for row in records})
    result = np.full((len(seeds), len(documents)), np.nan, dtype=np.float64)
    for i, seed in enumerate(seeds):
        for j, document in enumerate(documents):
            values = [float(row[key]) for row in records if int(row["seed"]) == seed and str(row["document_id"]) == document]
            if values:
                result[i, j] = float(np.mean(values))
    return result, seeds, documents


def bootstrap(records: list[dict[str, Any]], key: str, *, samples: int, seed: int, confidence: float) -> dict[str, Any]:
    values, seeds, documents = matrix(records, key)
    generator = np.random.default_rng(seed)
    draws = np.empty(samples, dtype=np.float64)
    for index in range(samples):
        rows = generator.integers(0, values.shape[0], size=values.shape[0])
        columns = generator.integers(0, values.shape[1], size=values.shape[1])
        draws[index] = float(np.nanmean(values[np.ix_(rows, columns)]))
    alpha = 1.0 - confidence
    return {
        "mean": float(np.nanmean(values)),
        "lcb": float(np.quantile(draws, alpha / 2.0)),
        "ucb": float(np.quantile(draws, 1.0 - alpha / 2.0)),
        "per_seed": {
            str(seed_value): float(np.nanmean(values[row_index]))
            for row_index, seed_value in enumerate(seeds)
        },
        "documents": documents,
    }


def selected(payloads, candidate: str, split: str) -> list[dict[str, Any]]:
    return [row for payload in payloads for row in payload["records"] if row["candidate"] == candidate and row["phase"] == "final" and row["evaluation_split"] == split]


def paired(payloads, left: str, right: str) -> list[dict[str, Any]]:
    rows = [row for payload in payloads for row in payload["records"] if row["candidate"] in {left, right} and row["phase"] == "final" and row["evaluation_split"] == "hypothesis"]
    index = {(int(row["seed"]), str(row["document_id"]), int(row["start"]), str(row["candidate"])): float(row["loss_delta"]) for row in rows}
    keys = sorted((seed, document, start) for seed, document, start, candidate in index if candidate == left)
    return [{"seed": seed, "document_id": document, "start": start, "difference": index[(seed, document, start, left)] - index[(seed, document, start, right)]} for seed, document, start in keys]


def close(a: float, b: float, tolerance: float = 1e-12) -> bool:
    return abs(float(a) - float(b)) <= tolerance


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs/pre_qwen_teacher_width_fresh_v1.yaml")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/pre-qwen-teacher-width-fresh/v1")
    args = parser.parse_args()
    config: dict[str, Any] = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    aggregate = json.loads((args.output_dir / "metrics.json").read_text(encoding="utf-8"))
    seeds = [int(v) for v in config["seeds"]]
    payloads = [json.loads((args.output_dir / f"seed-{seed}.json").read_text(encoding="utf-8")) for seed in seeds]
    stats = config["statistics"]
    samples = int(stats["bootstrap_samples"])
    base_seed = int(stats["bootstrap_seed"])
    confidence = float(stats["confidence"])
    names = [str(row["name"]) for row in config["candidates"]]
    mismatches: list[dict[str, Any]] = []
    rebuilt: dict[str, Any] = {}
    for index, name in enumerate(names):
        rebuilt[name] = {}
        for split, offset in (("hypothesis", 100), ("ood", 200)):
            value = bootstrap(selected(payloads, name, split), "loss_delta", samples=samples, seed=base_seed + offset + index, confidence=confidence)
            rebuilt[name][split] = value
            expected = aggregate["candidates"][name][split]
            for key in ("mean", "lcb", "ucb"):
                if not close(value[key], expected[key]):
                    mismatches.append({"candidate": name, "split": split, "key": key, "audit": value[key], "aggregate": expected[key]})
        expected_ratio = float(np.mean([payload["metadata"]["parameter_ratios"][name] for payload in payloads]))
        if not close(expected_ratio, aggregate["candidates"][name]["parameter_ratio"]):
            mismatches.append({"candidate": name, "issue": "parameter ratio"})
    comparison = bootstrap(paired(payloads, "magnitude-init-65", "magnitude-init-50"), "difference", samples=samples, seed=base_seed + 500, confidence=confidence)
    for key in ("mean", "lcb", "ucb"):
        if not close(comparison[key], aggregate["comparisons"]["primary_minus_50"][key]):
            mismatches.append({"comparison": "primary_minus_50", "key": key})

    gates_cfg = config["gates"]
    primary = rebuilt["magnitude-init-65"]
    capacity = rebuilt["magnitude-init-75"]
    anchor = rebuilt["magnitude-init-35"]
    full = rebuilt["full-continuation-control"]
    tol = float(gates_cfg["arithmetic_tolerance"])
    primary_param = float(np.mean([payload["metadata"]["parameter_ratios"]["magnitude-init-65"] for payload in payloads]))
    primary_compute = float(np.mean([payload["metadata"]["compute_ratios"]["magnitude-init-65"] for payload in payloads]))
    clean = all(
        not payload["metadata"]["data_audit"]["exact_cross_split_duplicates"]
        and not payload["metadata"]["data_audit"]["near_cross_split_pairs"]
        for payload in payloads
    )
    gates = {
        "primary_hypothesis_pass": primary["hypothesis"]["ucb"] <= float(gates_cfg["primary_hypothesis_ucb_max"]),
        "primary_ood_pass": primary["ood"]["ucb"] <= float(gates_cfg["primary_ood_ucb_max"]),
        "every_seed_primary_pass": max(primary["hypothesis"]["per_seed"].values()) <= float(gates_cfg["every_seed_primary_hypothesis_delta_max"]),
        "primary_vs_comparator_pass": comparison["ucb"] <= float(gates_cfg["primary_minus_comparator_ucb_max"]),
        "capacity_pass": capacity["hypothesis"]["ucb"] <= float(gates_cfg["capacity_hypothesis_ucb_max"]),
        "anchor_failure_reproduced": anchor["hypothesis"]["lcb"] >= float(gates_cfg["anchor_hypothesis_lcb_min"]),
        "full_control_hypothesis_pass": full["hypothesis"]["ucb"] <= float(gates_cfg["full_control_hypothesis_ucb_max"]),
        "full_control_ood_pass": full["ood"]["ucb"] <= float(gates_cfg["full_control_ood_ucb_max"]),
        "exact_primary_parameter_ratio_pass": abs(primary_param - float(gates_cfg["exact_primary_parameter_ratio"])) <= tol,
        "exact_primary_compute_ratio_pass": abs(primary_compute - float(gates_cfg["exact_primary_compute_ratio"])) <= tol,
        "clean_data_audit_pass": clean,
    }
    aggregate_gates = aggregate["decision"]["gates"]
    for name, value in gates.items():
        if bool(aggregate_gates.get(name)) != bool(value):
            mismatches.append({"gate": name, "audit": value, "aggregate": aggregate_gates.get(name)})
    leave_one_out: dict[str, Any] = {}
    for omitted in seeds:
        subset = [payload for payload in payloads if int(payload["metadata"]["seed"]) != omitted]
        rows = selected(subset, "magnitude-init-65", "hypothesis")
        leave_one_out[str(omitted)] = bootstrap(rows, "loss_delta", samples=4000, seed=base_seed + omitted % 1000, confidence=confidence)
    audit = {
        "audit_passed": not mismatches,
        "mismatches": mismatches,
        "rebuilt": rebuilt,
        "comparison": comparison,
        "gates_without_audit_gate": gates,
        "leave_one_seed_out_primary_hypothesis": leave_one_out,
        "provenance": {
            "configuration_hashes": sorted({payload["metadata"]["configuration_sha256"] for payload in payloads}),
            "source_commits": sorted({payload["metadata"]["source_commit"] for payload in payloads}),
            "checkpoint_hashes_unique": len({payload["metadata"]["checkpoint_sha256"] for payload in payloads}) == len(payloads),
        },
    }
    audit_dir = args.output_dir / "adversarial-audit"
    audit_dir.mkdir(parents=True, exist_ok=True)
    (audit_dir / "audit.json").write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# Independent audit — teacher-informed width fresh replication",
        "",
        f"**Audit:** **{'PASS' if audit['audit_passed'] else 'FAIL'}**",
        "",
        f"Mismatches: `{len(mismatches)}`.",
        "",
        "The auditor independently rebuilt seed/document cells, crossed intervals, the paired 65%-minus-50% comparison, arithmetic ratios, data-overlap gates, and leave-one-seed-out sensitivity.",
    ]
    (audit_dir / "VERDICT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print((audit_dir / "VERDICT.md").read_text(encoding="utf-8"))
    return 0 if audit["audit_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
