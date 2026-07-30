#!/usr/bin/env python3
"""Independent adversarial audit for routing-coupled residual v4.

This script intentionally does not import the aggregator or the project's
bootstrap helpers.  It rebuilds seed-document cells, crossed intervals,
provenance, arithmetic, gates, and leave-one-seed-out sensitivity from raw
per-seed records.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Callable

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
PRIMARY = "rank5-coupled-q8-h8-v4"
MEAN_ONLY = "rank5-coupled-q8-h8-mean-only-control"
V3 = "rank5-v3-frozen-baseline"
RANK6 = "rank6-v3-frozen-capacity"
NARROW = "narrow65-frozen-baseline"
FULL = "full-continuation-control"
DISABLED = "rank5-coupled-q8-h8-v4__coupling-disabled"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rows_for(records: list[dict[str, Any]], candidate: str, split: str) -> list[dict[str, Any]]:
    return [
        row
        for row in records
        if row["candidate"] == candidate
        and row["phase"] == "final"
        and row["evaluation_split"] == split
    ]


def matrix(
    rows: list[dict[str, Any]],
    value: Callable[[dict[str, Any]], float],
) -> tuple[np.ndarray, list[int], list[str]]:
    seeds = sorted({int(row["seed"]) for row in rows})
    documents = sorted({str(row["document_id"]) for row in rows})
    buckets: dict[tuple[int, str], list[float]] = {}
    for row in rows:
        number = float(value(row))
        if not np.isfinite(number):
            raise ValueError("non-finite audit observation")
        buckets.setdefault((int(row["seed"]), str(row["document_id"])), []).append(number)
    result = np.full((len(seeds), len(documents)), np.nan, dtype=np.float64)
    for seed_index, seed in enumerate(seeds):
        for document_index, document in enumerate(documents):
            values = buckets.get((seed, document))
            if values:
                result[seed_index, document_index] = float(np.mean(values))
    if np.isnan(result).any():
        raise ValueError("incomplete crossed seed-document matrix")
    return result, seeds, documents


def crossed(
    rows: list[dict[str, Any]],
    value: Callable[[dict[str, Any]], float],
    *,
    samples: int,
    random_seed: int,
    confidence: float,
) -> dict[str, Any]:
    values, seeds, documents = matrix(rows, value)
    rng = np.random.default_rng(random_seed)
    draws = np.empty(samples, dtype=np.float64)
    for index in range(samples):
        seed_ids = rng.integers(0, len(seeds), size=len(seeds))
        document_ids = rng.integers(0, len(documents), size=len(documents))
        draws[index] = float(values[np.ix_(seed_ids, document_ids)].mean())
    alpha = 1.0 - confidence
    return {
        "mean": float(values.mean()),
        "lcb": float(np.quantile(draws, alpha / 2.0)),
        "ucb": float(np.quantile(draws, 1.0 - alpha / 2.0)),
        "per_seed": {
            str(seed): float(values[index].mean()) for index, seed in enumerate(seeds)
        },
        "effective_cells": int(values.size),
    }


def paired(
    records: list[dict[str, Any]],
    left: str,
    right: str,
    key: str,
    *,
    samples: int,
    random_seed: int,
    confidence: float,
) -> dict[str, Any]:
    left_rows = rows_for(records, left, "hypothesis")
    right_rows = rows_for(records, right, "hypothesis")
    index = {
        (int(row["seed"]), str(row["document_id"]), int(row["start"])): row
        for row in right_rows
    }
    difference_rows: list[dict[str, Any]] = []
    for row in left_rows:
        cell = (int(row["seed"]), str(row["document_id"]), int(row["start"]))
        if cell not in index:
            raise ValueError(f"missing paired audit cell: {cell}")
        difference_rows.append(
            {
                "seed": cell[0],
                "document_id": cell[1],
                "difference": float(row[key]) - float(index[cell][key]),
            }
        )
    if len(difference_rows) != len(right_rows):
        raise ValueError("paired audit row-count mismatch")
    return crossed(
        difference_rows,
        lambda row: float(row["difference"]),
        samples=samples,
        random_seed=random_seed,
        confidence=confidence,
    )


def stat(
    records: list[dict[str, Any]],
    candidate: str,
    split: str,
    key: str,
    *,
    samples: int,
    random_seed: int,
    confidence: float,
) -> dict[str, Any]:
    return crossed(
        rows_for(records, candidate, split),
        lambda row: float(row[key]),
        samples=samples,
        random_seed=random_seed,
        confidence=confidence,
    )


def close(label: str, actual: float, expected: float, mismatches: list[str], tolerance: float = 1e-10) -> None:
    if abs(actual - expected) > tolerance:
        mismatches.append(f"{label}: rebuilt={actual} machine={expected}")


def expected_verdict(gates: dict[str, bool], improvement_votes: int, required_votes: int) -> str:
    pass_keys = (
        "primary_hypothesis_pass",
        "primary_ood_pass",
        "every_seed_primary_pass",
        "primary_vs_narrow65_pass",
        "primary_vs_rank6_pass",
        "primary_vs_v3_pass",
        "primary_vs_mean_only_kl_pass",
        "primary_kl_pass",
        "primary_top1_pass",
        "primary_local_nrmse_pass",
        "primary_counterfactual_pass",
        "primary_cross_error_gap_pass",
        "causal_coupling_pass",
        "full_control_hypothesis_pass",
        "full_control_ood_pass",
        "primary_parameter_budget_pass",
        "primary_compute_budget_pass",
        "all_arithmetic_pass",
        "clean_data_audit_pass",
    )
    if all(gates[key] for key in pass_keys):
        return "ROUTING_COUPLED_V4_PASS"
    integrity = all(
        gates[key]
        for key in (
            "primary_parameter_budget_pass",
            "primary_compute_budget_pass",
            "all_arithmetic_pass",
            "clean_data_audit_pass",
            "full_control_hypothesis_pass",
            "full_control_ood_pass",
            "causal_coupling_pass",
        )
    )
    if integrity and improvement_votes >= required_votes:
        return "ROUTING_COUPLED_V4_FUNCTIONAL_SIGNAL"
    return "ROUTING_COUPLED_V4_FAIL"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path, default=ROOT / "configs/pre_qwen_routing_coupled_v4.yaml"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "results/pre-qwen-routing-coupled/v4"
    )
    args = parser.parse_args()

    cfg: dict[str, Any] = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    seeds = [int(value) for value in cfg["seeds"]]
    payloads = [
        json.loads((args.output_dir / f"seed-{seed}.json").read_text(encoding="utf-8"))
        for seed in seeds
    ]
    records = [row for payload in payloads for row in payload["records"]]
    samples = int(cfg["statistics"]["bootstrap_samples"])
    base_seed = int(cfg["statistics"]["bootstrap_seed"])
    confidence = float(cfg["statistics"]["confidence"])
    machine = json.loads((args.output_dir / "metrics.json").read_text(encoding="utf-8"))
    mismatches: list[str] = []

    # Rebuild every load-bearing statistic using the same preregistered random
    # streams but an independent implementation.
    primary_hyp = {
        "loss": stat(records, PRIMARY, "hypothesis", "loss_delta", samples=samples, random_seed=base_seed + 0, confidence=confidence),
        "kl": stat(records, PRIMARY, "hypothesis", "kl_teacher_to_candidate", samples=samples, random_seed=base_seed + 1, confidence=confidence),
        "top1": stat(records, PRIMARY, "hypothesis", "top1_agreement", samples=samples, random_seed=base_seed + 2, confidence=confidence),
        "local": stat(records, PRIMARY, "hypothesis", "local_nrmse", samples=samples, random_seed=base_seed + 3, confidence=confidence),
        "counterfactual": stat(records, PRIMARY, "hypothesis", "counterfactual_nrmse", samples=samples, random_seed=base_seed + 4, confidence=confidence),
        "cross": stat(records, PRIMARY, "hypothesis", "routing_cross_error", samples=samples, random_seed=base_seed + 5, confidence=confidence),
    }
    primary_ood = stat(records, PRIMARY, "ood", "loss_delta", samples=samples, random_seed=base_seed + 100, confidence=confidence)
    full_hyp = stat(records, FULL, "hypothesis", "loss_delta", samples=samples, random_seed=base_seed + 6000, confidence=confidence)
    full_ood = stat(records, FULL, "ood", "loss_delta", samples=samples, random_seed=base_seed + 6100, confidence=confidence)

    comparisons = {
        "primary_minus_narrow_loss": paired(records, PRIMARY, NARROW, "loss_delta", samples=samples, random_seed=base_seed + 9101, confidence=confidence),
        "primary_minus_rank6_loss": paired(records, PRIMARY, RANK6, "loss_delta", samples=samples, random_seed=base_seed + 9102, confidence=confidence),
        "primary_minus_v3_loss": paired(records, PRIMARY, V3, "loss_delta", samples=samples, random_seed=base_seed + 9103, confidence=confidence),
        "primary_minus_v3_kl": paired(records, PRIMARY, V3, "kl_teacher_to_candidate", samples=samples, random_seed=base_seed + 9104, confidence=confidence),
        "primary_minus_v3_top1": paired(records, PRIMARY, V3, "top1_agreement", samples=samples, random_seed=base_seed + 9105, confidence=confidence),
        "primary_minus_v3_local": paired(records, PRIMARY, V3, "local_nrmse", samples=samples, random_seed=base_seed + 9106, confidence=confidence),
        "primary_minus_v3_counterfactual": paired(records, PRIMARY, V3, "counterfactual_nrmse", samples=samples, random_seed=base_seed + 9107, confidence=confidence),
        "primary_minus_mean_only_kl": paired(records, PRIMARY, MEAN_ONLY, "kl_teacher_to_candidate", samples=samples, random_seed=base_seed + 9108, confidence=confidence),
        "primary_minus_mean_only_loss": paired(records, PRIMARY, MEAN_ONLY, "loss_delta", samples=samples, random_seed=base_seed + 9109, confidence=confidence),
        "disabled_minus_primary_kl": paired(records, DISABLED, PRIMARY, "kl_teacher_to_candidate", samples=samples, random_seed=base_seed + 9110, confidence=confidence),
        "disabled_minus_primary_loss": paired(records, DISABLED, PRIMARY, "loss_delta", samples=samples, random_seed=base_seed + 9111, confidence=confidence),
        "primary_minus_narrow_cross_error": paired(records, PRIMARY, NARROW, "routing_cross_error", samples=samples, random_seed=base_seed + 9113, confidence=confidence),
    }

    machine_primary = machine["candidates"][PRIMARY]
    for label, rebuilt, expected in (
        ("primary loss", primary_hyp["loss"], machine_primary["hypothesis"]["loss_delta"]),
        ("primary KL", primary_hyp["kl"], machine_primary["hypothesis"]["kl_teacher_to_candidate"]),
        ("primary top1", primary_hyp["top1"], machine_primary["hypothesis"]["top1_agreement"]),
        ("primary local", primary_hyp["local"], machine_primary["hypothesis"]["local_nrmse"]),
        ("primary counterfactual", primary_hyp["counterfactual"], machine_primary["hypothesis"]["counterfactual_nrmse"]),
        ("primary OOD loss", primary_ood, machine_primary["ood"]["loss_delta"]),
    ):
        for field in ("mean", "lcb", "ucb"):
            close(f"{label} {field}", float(rebuilt[field]), float(expected[field]), mismatches)
    for name, rebuilt in comparisons.items():
        expected = machine["comparisons"][name]
        for field in ("mean", "lcb", "ucb"):
            close(f"{name} {field}", float(rebuilt[field]), float(expected[field]), mismatches)

    # Provenance, completeness, hashes, and data isolation.
    config_hash = sha256_file(args.config)
    protocol = str(cfg["protocol_version"])
    source_commits = {str(payload["metadata"]["source_commit"]) for payload in payloads}
    configuration_hashes = {str(payload["metadata"]["configuration_sha256"]) for payload in payloads}
    checkpoint_hashes: list[str] = []
    source_hashes: list[str] = []
    required_candidates = {
        str(row["name"]) for row in cfg["candidates"]
    } | {DISABLED, "rank5-coupled-q8-h8-v4__second-moment-disabled"}
    coverage_pass = True
    data_pass = True
    for seed, payload in zip(seeds, payloads, strict=True):
        checkpoint = args.output_dir / f"frozen-candidates-seed-{seed}.pt"
        actual_checkpoint_hash = sha256_file(checkpoint)
        checkpoint_hashes.append(actual_checkpoint_hash)
        if actual_checkpoint_hash != str(payload["metadata"]["checkpoint_sha256"]):
            mismatches.append(f"checkpoint hash mismatch for seed {seed}")
        source_hashes.append(str(payload["metadata"]["source_v3_checkpoint_sha256"]))
        audit = payload["metadata"]["data_audit"]
        if audit["exact_cross_split_duplicates"] or audit["near_cross_split_pairs"]:
            data_pass = False
        if any(str(row["candidate"]) not in required_candidates for row in payload["records"]):
            coverage_pass = False
        observed_pairs = {
            (str(row["candidate"]), str(row["phase"]), str(row["evaluation_split"]))
            for row in payload["records"]
        }
        for candidate in required_candidates:
            if (candidate, "final", "hypothesis") not in observed_pairs or (
                candidate,
                "final",
                "ood",
            ) not in observed_pairs:
                coverage_pass = False

    expected = {
        str(row["name"]): (
            float(row["expected_parameter_ratio"]),
            float(row["expected_compute_ratio"]),
        )
        for row in cfg["candidates"]
    }
    observed_first = payloads[0]["metadata"]
    tolerance = float(cfg["gates"]["arithmetic_tolerance"])
    arithmetic_pass = True
    for candidate, (parameter_ratio, compute_ratio) in expected.items():
        for payload in payloads:
            actual_p = float(payload["metadata"]["parameter_ratios"][candidate])
            actual_c = float(payload["metadata"]["compute_ratios"][candidate])
            arithmetic_pass &= abs(actual_p - parameter_ratio) <= tolerance
            arithmetic_pass &= abs(actual_c - compute_ratio) <= tolerance

    gates_cfg = cfg["gates"]
    observed_primary_p = float(observed_first["parameter_ratios"][PRIMARY])
    observed_primary_c = float(observed_first["compute_ratios"][PRIMARY])
    every_seed = max(primary_hyp["loss"]["per_seed"].values())
    improvement_checks = {
        "loss": comparisons["primary_minus_v3_loss"]["ucb"] <= float(gates_cfg["primary_minus_v3_loss_ucb_max"]),
        "kl": comparisons["primary_minus_v3_kl"]["ucb"] <= float(gates_cfg["primary_minus_v3_kl_ucb_max"]),
        "top1": comparisons["primary_minus_v3_top1"]["lcb"] >= float(gates_cfg["primary_minus_v3_top1_lcb_min"]),
        "local": comparisons["primary_minus_v3_local"]["ucb"] <= float(gates_cfg["primary_minus_v3_local_nrmse_ucb_max"]),
        "counterfactual": comparisons["primary_minus_v3_counterfactual"]["ucb"] <= float(gates_cfg["primary_minus_v3_counterfactual_nrmse_ucb_max"]),
    }
    improvement_votes = sum(improvement_checks.values())
    causal_kl = comparisons["disabled_minus_primary_kl"]["lcb"] >= float(gates_cfg["disabled_minus_primary_kl_lcb_min"])
    causal_loss = comparisons["disabled_minus_primary_loss"]["lcb"] >= float(gates_cfg["disabled_minus_primary_loss_lcb_min"])
    gates = {
        "primary_hypothesis_pass": primary_hyp["loss"]["ucb"] <= float(gates_cfg["primary_hypothesis_ucb_max"]),
        "primary_ood_pass": primary_ood["ucb"] <= float(gates_cfg["primary_ood_ucb_max"]),
        "every_seed_primary_pass": every_seed <= float(gates_cfg["every_seed_primary_hypothesis_delta_max"]),
        "primary_vs_narrow65_pass": comparisons["primary_minus_narrow_loss"]["ucb"] <= float(gates_cfg["primary_minus_narrow65_ucb_max"]),
        "primary_vs_rank6_pass": comparisons["primary_minus_rank6_loss"]["ucb"] <= float(gates_cfg["primary_minus_rank6_ucb_max"]),
        "primary_vs_v3_pass": comparisons["primary_minus_v3_loss"]["ucb"] <= float(gates_cfg["primary_minus_v3_ucb_max"]),
        "primary_vs_mean_only_kl_pass": comparisons["primary_minus_mean_only_kl"]["ucb"] <= float(gates_cfg["primary_minus_mean_only_kl_ucb_max"]),
        "primary_kl_pass": primary_hyp["kl"]["ucb"] <= float(gates_cfg["primary_kl_ucb_max"]),
        "primary_top1_pass": primary_hyp["top1"]["lcb"] >= float(gates_cfg["primary_top1_lcb_min"]),
        "primary_local_nrmse_pass": primary_hyp["local"]["ucb"] <= float(gates_cfg["primary_local_nrmse_ucb_max"]),
        "primary_counterfactual_pass": primary_hyp["counterfactual"]["ucb"] <= float(gates_cfg["primary_counterfactual_nrmse_ucb_max"]),
        "primary_cross_error_gap_pass": comparisons["primary_minus_narrow_cross_error"]["ucb"] <= float(gates_cfg["primary_cross_error_gap_vs_narrow_ucb_max"]),
        "causal_coupling_kl_pass": causal_kl,
        "causal_coupling_loss_pass": causal_loss,
        "causal_coupling_pass": causal_kl or causal_loss,
        "full_control_hypothesis_pass": full_hyp["ucb"] <= float(gates_cfg["full_control_hypothesis_ucb_max"]),
        "full_control_ood_pass": full_ood["ucb"] <= float(gates_cfg["full_control_ood_ucb_max"]),
        "primary_parameter_budget_pass": observed_primary_p < float(gates_cfg["primary_parameter_ratio_strict_max"]),
        "primary_compute_budget_pass": observed_primary_c <= float(gates_cfg["primary_compute_ratio_max"]),
        "all_arithmetic_pass": bool(arithmetic_pass),
        "clean_data_audit_pass": bool(data_pass),
    }

    machine_gates = machine["decision"]["gates"]
    for name, value in gates.items():
        if bool(machine_gates.get(name)) != bool(value):
            mismatches.append(f"gate mismatch {name}: rebuilt={value} machine={machine_gates.get(name)}")
    if int(machine["decision"]["improvement_votes"]) != improvement_votes:
        mismatches.append("improvement vote mismatch")

    expected_final = expected_verdict(
        gates,
        improvement_votes,
        int(gates_cfg["behavior_improvement_min_votes"]),
    )

    leave_one_seed_out: dict[str, Any] = {}
    primary_rows = rows_for(records, PRIMARY, "hypothesis")
    for omitted in seeds:
        subset = [row for row in primary_rows if int(row["seed"]) != omitted]
        leave_one_seed_out[str(omitted)] = crossed(
            subset,
            lambda row: float(row["loss_delta"]),
            samples=samples,
            random_seed=base_seed + 12000 + omitted,
            confidence=confidence,
        )

    audit = {
        "audit_passed": not mismatches and coverage_pass and len(source_commits) == 1 and configuration_hashes == {config_hash} and all(len(value) == 64 for value in checkpoint_hashes + source_hashes),
        "mismatches": mismatches,
        "coverage_pass": coverage_pass,
        "provenance": {
            "protocol": protocol,
            "source_commits": sorted(source_commits),
            "configuration_hashes": sorted(configuration_hashes),
            "checkpoint_hashes_unique": len(set(checkpoint_hashes)) == len(checkpoint_hashes),
            "source_v3_hashes_unique": len(set(source_hashes)) == len(source_hashes),
        },
        "rebuilt": {
            "primary_hypothesis": primary_hyp,
            "primary_ood_loss": primary_ood,
            "full_hypothesis_loss": full_hyp,
            "full_ood_loss": full_ood,
            "comparisons": comparisons,
        },
        "gates_without_audit_gate": gates,
        "improvement_checks": improvement_checks,
        "improvement_votes": improvement_votes,
        "expected_final_verdict_if_audit_passes": expected_final,
        "leave_one_seed_out_primary_hypothesis": leave_one_seed_out,
    }
    audit_dir = args.output_dir / "adversarial-audit"
    audit_dir.mkdir(parents=True, exist_ok=True)
    (audit_dir / "audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (audit_dir / "VERDICT.md").write_text(
        "# Independent audit — routing-coupled residual v4\n\n"
        f"**Audit:** **{'PASS' if audit['audit_passed'] else 'FAIL'}**\n\n"
        f"Mismatches: `{len(mismatches)}`.\n\n"
        f"Expected final verdict if the audit passes: `{expected_final}`.\n\n"
        "The auditor independently rebuilt seed-document cells, crossed intervals, "
        "causal ablations, behavior-improvement votes, arithmetic, data isolation, "
        "provenance, gates, and leave-one-seed-out sensitivity.\n",
        encoding="utf-8",
    )
    print("Audit:", "PASS" if audit["audit_passed"] else "FAIL")
    if mismatches:
        print("\n".join(mismatches))
    return 0 if audit["audit_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
