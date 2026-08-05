#!/usr/bin/env python3
"""Independent factual/adversarial audit for Reality Gate 1A."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Callable

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
PRIMARY = "heterogeneous-spectral"
UNIFORM = "uniform-rank"
ROUTING = "heterogeneous-routing"
NARROW = "narrow65"
FULL = "full-identity-control"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rows_for(records: list[dict[str, Any]], scale: str, candidate: str, split: str) -> list[dict[str, Any]]:
    return [
        row
        for row in records
        if row["scale"] == scale
        and row["candidate"] == candidate
        and row["phase"] == "final"
        and row["evaluation_split"] == split
    ]


def matrix(rows: list[dict[str, Any]], value: Callable[[dict[str, Any]], float]) -> tuple[np.ndarray, list[int], list[str]]:
    seeds = sorted({int(row["seed"]) for row in rows})
    documents = sorted({str(row["document_id"]) for row in rows})
    buckets: dict[tuple[int, str], list[float]] = {}
    for row in rows:
        number = float(value(row))
        if not np.isfinite(number):
            raise ValueError("non-finite audit observation")
        buckets.setdefault((int(row["seed"]), str(row["document_id"])), []).append(number)
    values = np.full((len(seeds), len(documents)), np.nan, dtype=np.float64)
    for i, seed in enumerate(seeds):
        for j, document in enumerate(documents):
            current = buckets.get((seed, document))
            if current:
                values[i, j] = float(np.mean(current))
    if np.isnan(values).any():
        raise ValueError("incomplete seed-document matrix")
    return values, seeds, documents


def crossed(rows: list[dict[str, Any]], value: Callable[[dict[str, Any]], float], *, samples: int, random_seed: int, confidence: float) -> dict[str, Any]:
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
        "per_seed": {str(seed): float(values[i].mean()) for i, seed in enumerate(seeds)},
        "effective_cells": int(values.size),
    }


def stat(
    records: list[dict[str, Any]],
    scale: str,
    candidate: str,
    key: str,
    *,
    random_seed: int,
    cfg: dict[str, Any],
    split: str = "hypothesis",
) -> dict[str, Any]:
    return crossed(
        rows_for(records, scale, candidate, split),
        lambda row: float(row[key]),
        samples=int(cfg["statistics"]["bootstrap_samples"]),
        random_seed=random_seed,
        confidence=float(cfg["statistics"]["confidence"]),
    )


def paired(records: list[dict[str, Any]], scale: str, left: str, right: str, key: str, *, random_seed: int, cfg: dict[str, Any]) -> dict[str, Any]:
    left_rows = rows_for(records, scale, left, "hypothesis")
    right_rows = rows_for(records, scale, right, "hypothesis")
    index = {
        (int(row["seed"]), str(row["document_id"]), int(row["start"])): row
        for row in right_rows
    }
    differences: list[dict[str, Any]] = []
    for row in left_rows:
        cell = (int(row["seed"]), str(row["document_id"]), int(row["start"]))
        if cell not in index:
            raise ValueError(f"unpaired cell {cell}")
        differences.append(
            {
                "seed": cell[0],
                "document_id": cell[1],
                "difference": float(row[key]) - float(index[cell][key]),
            }
        )
    if len(differences) != len(right_rows):
        raise ValueError("paired row count mismatch")
    return crossed(
        differences,
        lambda row: float(row["difference"]),
        samples=int(cfg["statistics"]["bootstrap_samples"]),
        random_seed=random_seed,
        confidence=float(cfg["statistics"]["confidence"]),
    )


def close(label: str, rebuilt: dict[str, Any], machine: dict[str, Any], mismatches: list[str]) -> None:
    for key in ("mean", "lcb", "ucb"):
        if abs(float(rebuilt[key]) - float(machine[key])) > 1e-10:
            mismatches.append(f"{label}.{key}: rebuilt={rebuilt[key]} machine={machine[key]}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs/reality_gate_1a.yaml")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/reality-gate-1a")
    parser.add_argument("--data-root", type=Path, default=ROOT / "data/reality-gate-1a")
    args = parser.parse_args()

    cfg: dict[str, Any] = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    scales = list(cfg["scales"])
    seeds = [int(value) for value in cfg["seeds"]]
    payloads = [
        json.loads((args.output_dir / scale / f"seed-{seed}.json").read_text(encoding="utf-8"))
        for scale in scales
        for seed in seeds
    ]
    records = [row for payload in payloads for row in payload["records"]]
    machine = json.loads((args.output_dir / "metrics.json").read_text(encoding="utf-8"))
    base_seed = int(cfg["statistics"]["bootstrap_seed"])
    mismatches: list[str] = []
    rebuilt_scales: dict[str, Any] = {}

    candidate_order = (PRIMARY, UNIFORM, ROUTING, NARROW, FULL)
    comparison_specs = (
        ("primary_minus_uniform_loss", PRIMARY, UNIFORM, "loss_delta"),
        ("primary_minus_uniform_kl", PRIMARY, UNIFORM, "kl_teacher_to_candidate"),
        ("primary_minus_uniform_top1", PRIMARY, UNIFORM, "top1_agreement"),
        ("primary_minus_uniform_local", PRIMARY, UNIFORM, "local_nrmse"),
        ("primary_minus_routing_loss", PRIMARY, ROUTING, "loss_delta"),
        ("primary_minus_narrow_loss", PRIMARY, NARROW, "loss_delta"),
        ("primary_minus_narrow_kl", PRIMARY, NARROW, "kl_teacher_to_candidate"),
        ("primary_minus_narrow_top1", PRIMARY, NARROW, "top1_agreement"),
        ("primary_minus_narrow_local", PRIMARY, NARROW, "local_nrmse"),
    )
    metric_keys = ("loss_delta", "kl_teacher_to_candidate", "top1_agreement", "local_nrmse")
    for scale_index, scale in enumerate(scales):
        rebuilt_candidates: dict[str, Any] = {}
        for candidate_index, candidate in enumerate(candidate_order):
            rebuilt_candidates[candidate] = {}
            for metric_index, key in enumerate(metric_keys):
                rebuilt = stat(
                    records,
                    scale,
                    candidate,
                    key,
                    random_seed=base_seed + scale_index * 10000 + candidate_index * 1000 + metric_index,
                    cfg=cfg,
                )
                rebuilt_candidates[candidate][key] = rebuilt
                close(
                    f"{scale}.{candidate}.{key}",
                    rebuilt,
                    machine["scales"][scale]["candidates"][candidate]["hypothesis"][key],
                    mismatches,
                )
        rebuilt_candidates[PRIMARY]["ood_loss_delta"] = stat(
            records, scale, PRIMARY, "loss_delta",
            random_seed=base_seed + scale_index * 10000 + 100, cfg=cfg, split="ood"
        )
        rebuilt_candidates[PRIMARY]["ood_kl_teacher_to_candidate"] = stat(
            records, scale, PRIMARY, "kl_teacher_to_candidate",
            random_seed=base_seed + scale_index * 10000 + 101, cfg=cfg, split="ood"
        )
        rebuilt_candidates[FULL]["ood_loss_delta"] = stat(
            records, scale, FULL, "loss_delta",
            random_seed=base_seed + scale_index * 10000 + 4100, cfg=cfg, split="ood"
        )
        rebuilt_candidates[FULL]["ood_kl_teacher_to_candidate"] = stat(
            records, scale, FULL, "kl_teacher_to_candidate",
            random_seed=base_seed + scale_index * 10000 + 4101, cfg=cfg, split="ood"
        )
        for label, key, candidate in (
            ("primary ood loss", "ood_loss_delta", PRIMARY),
            ("primary ood kl", "ood_kl_teacher_to_candidate", PRIMARY),
            ("full ood loss", "ood_loss_delta", FULL),
            ("full ood kl", "ood_kl_teacher_to_candidate", FULL),
        ):
            machine_key = "loss_delta" if "loss" in label else "kl_teacher_to_candidate"
            close(
                f"{scale}.{label}",
                rebuilt_candidates[candidate][key],
                machine["scales"][scale]["candidates"][candidate]["ood"][machine_key],
                mismatches,
            )

        rebuilt_comparisons: dict[str, Any] = {}
        for index, (name, left, right, key) in enumerate(comparison_specs):
            rebuilt = paired(
                records,
                scale,
                left,
                right,
                key,
                random_seed=base_seed + scale_index * 10000 + 7000 + index,
                cfg=cfg,
            )
            rebuilt_comparisons[name] = rebuilt
            close(name, rebuilt, machine["scales"][scale]["comparisons"][name], mismatches)
        rebuilt_scales[scale] = {
            "candidates": rebuilt_candidates,
            "comparisons": rebuilt_comparisons,
        }

    config_hash = sha256_file(args.config)
    data_manifest_path = args.data_root / "manifest.json"
    data_hash = sha256_file(data_manifest_path)
    data_manifest = json.loads(data_manifest_path.read_text(encoding="utf-8"))
    data_files_pass = all(
        (args.data_root / name).exists()
        and sha256_file(args.data_root / name) == str(expected)
        for name, expected in data_manifest["sha256"].items()
    )
    provenance_pass = (
        {str(payload["metadata"]["protocol_version"]) for payload in payloads}
        == {str(cfg["protocol_version"])}
        and {str(payload["metadata"]["configuration_sha256"]) for payload in payloads}
        == {config_hash}
        and {str(payload["metadata"]["data_manifest_sha256"]) for payload in payloads}
        == {data_hash}
        and len({str(payload["metadata"]["source_commit"]) for payload in payloads}) == 1
    )
    checkpoint_pass = True
    checkpoint_hashes: list[str] = []
    for payload in payloads:
        scale = str(payload["metadata"]["scale"])
        seed = int(payload["metadata"]["seed"])
        path = args.output_dir / scale / f"frozen-candidates-seed-{seed}.pt"
        actual = sha256_file(path)
        checkpoint_hashes.append(actual)
        if actual != str(payload["metadata"]["checkpoint_sha256"]):
            checkpoint_pass = False
            mismatches.append(f"checkpoint hash mismatch {scale}/{seed}")
    checkpoint_pass &= len(set(checkpoint_hashes)) == len(checkpoint_hashes)
    data_pass = all(
        not any(payload["metadata"]["document_hash_overlap"].values()) for payload in payloads
    )
    plateau_pass = all(bool(payload["metadata"]["plateau"]["plateau_reached"]) for payload in payloads)

    # Recompute the final gate map independently from raw metadata and rebuilt statistics.
    gates_cfg = cfg["gates"]
    gates: dict[str, bool] = {}
    signal_scales = 0
    for scale in scales:
        current = [payload for payload in payloads if payload["metadata"]["scale"] == scale]
        machine_scale = machine["scales"][scale]
        comparisons = rebuilt_scales[scale]["comparisons"]
        primary = rebuilt_scales[scale]["candidates"][PRIMARY]
        accounting = machine_scale["candidates"][PRIMARY]["accounting"]
        uniform_accounting = machine_scale["candidates"][UNIFORM]["accounting"]
        behavior_votes = sum(
            (
                comparisons["primary_minus_uniform_kl"]["ucb"] <= float(gates_cfg["uniform_kl_difference_ucb_max"]),
                comparisons["primary_minus_uniform_top1"]["lcb"] >= float(gates_cfg["uniform_top1_difference_lcb_min"]),
                comparisons["primary_minus_uniform_local"]["ucb"] <= float(gates_cfg["uniform_local_difference_ucb_max"]),
            )
        )
        primary_worst_seed = max(primary["loss_delta"]["per_seed"].values())
        scale_gates = {
            "all_teachers_plateaued": all(bool(payload["metadata"]["plateau"]["plateau_reached"]) for payload in current),
            "clean_data": all(not any(payload["metadata"]["document_hash_overlap"].values()) for payload in current),
            "primary_parameter_budget": float(accounting["parameter_ratio_max"]) <= float(uniform_accounting["parameter_ratio_max"]) + float(gates_cfg["budget_tolerance"]),
            "primary_train_compute_budget": float(accounting["train_compute_ratio_max"]) <= float(uniform_accounting["train_compute_ratio_max"]) + float(gates_cfg["budget_tolerance"]),
            "primary_hypothesis_compute_budget": float(accounting["hypothesis_compute_ratio_max"]) <= float(uniform_accounting["hypothesis_compute_ratio_max"]) + float(gates_cfg["heldout_compute_tolerance"]),
            "primary_absolute_loss": primary["loss_delta"]["ucb"] <= float(gates_cfg["absolute_loss_ucb_max"]),
            "primary_absolute_kl": primary["kl_teacher_to_candidate"]["ucb"] <= float(gates_cfg["absolute_kl_ucb_max"]),
            "primary_absolute_top1": primary["top1_agreement"]["lcb"] >= float(gates_cfg["absolute_top1_lcb_min"]),
            "primary_absolute_local": primary["local_nrmse"]["ucb"] <= float(gates_cfg["absolute_local_nrmse_ucb_max"]),
            "primary_ood_loss": rebuilt_scales[scale]["candidates"][PRIMARY]["ood_loss_delta"]["ucb"] <= float(gates_cfg["ood_loss_ucb_max"]),
            "primary_ood_kl": rebuilt_scales[scale]["candidates"][PRIMARY]["ood_kl_teacher_to_candidate"]["ucb"] <= float(gates_cfg["ood_kl_ucb_max"]),
            "primary_every_seed_loss": primary_worst_seed <= float(gates_cfg["every_seed_loss_delta_max"]),
            "primary_vs_uniform_loss": comparisons["primary_minus_uniform_loss"]["ucb"] <= float(gates_cfg["uniform_loss_difference_ucb_max"]),
            "primary_vs_uniform_behavior": behavior_votes >= int(gates_cfg["uniform_behavior_min_votes"]),
            "primary_vs_routing_loss": comparisons["primary_minus_routing_loss"]["ucb"] <= float(gates_cfg["routing_loss_difference_ucb_max"]),
            "primary_vs_narrow_loss": comparisons["primary_minus_narrow_loss"]["ucb"] <= float(gates_cfg["narrow_loss_difference_ucb_max"]),
            "primary_vs_narrow_kl": comparisons["primary_minus_narrow_kl"]["ucb"] <= float(gates_cfg["narrow_kl_difference_ucb_max"]),
            "primary_vs_narrow_top1": comparisons["primary_minus_narrow_top1"]["lcb"] >= float(gates_cfg["narrow_top1_difference_lcb_min"]),
            "primary_vs_narrow_local": comparisons["primary_minus_narrow_local"]["ucb"] <= float(gates_cfg["narrow_local_difference_ucb_max"]),
            "full_identity": max(
                abs(float(rebuilt_scales[scale]["candidates"][FULL]["loss_delta"]["mean"])),
                abs(float(rebuilt_scales[scale]["candidates"][FULL]["ood_loss_delta"]["mean"])),
                float(rebuilt_scales[scale]["candidates"][FULL]["kl_teacher_to_candidate"]["ucb"]),
                float(rebuilt_scales[scale]["candidates"][FULL]["ood_kl_teacher_to_candidate"]["ucb"]),
            ) <= float(gates_cfg["identity_absolute_tolerance"]),
        }
        for name, value in scale_gates.items():
            gates[f"{scale}__{name}"] = bool(value)
            if bool(machine["decision"]["gates"].get(f"{scale}__{name}")) != bool(value):
                mismatches.append(f"gate mismatch {scale}__{name}")
        if (
            scale_gates["all_teachers_plateaued"]
            and scale_gates["clean_data"]
            and scale_gates["primary_parameter_budget"]
            and scale_gates["primary_train_compute_budget"]
            and comparisons["primary_minus_uniform_loss"]["ucb"] <= float(gates_cfg["signal_uniform_loss_ucb_max"])
            and behavior_votes >= int(gates_cfg["signal_behavior_min_votes"])
        ):
            signal_scales += 1

    small, medium = scales[0], scales[-1]
    scale_trend = (
        rebuilt_scales[medium]["comparisons"]["primary_minus_uniform_loss"]["mean"]
        <= rebuilt_scales[small]["comparisons"]["primary_minus_uniform_loss"]["mean"]
        + float(gates_cfg["maximum_scale_trend_regression"])
    )
    gates["scale_trend"] = bool(scale_trend)
    # Match the preregistration exactly: the routing-only comparator is a
    # load-bearing mechanism control, not an advisory metric.
    pass_required = list(gates)
    if all(gates[name] for name in pass_required):
        expected_verdict = "REALITY_GATE_1A_PASS"
    elif signal_scales >= int(gates_cfg["signal_min_scales"]):
        expected_verdict = "REALITY_GATE_1A_HETEROGENEOUS_RANK_SIGNAL"
    else:
        expected_verdict = "REALITY_GATE_1A_FAIL"

    leave_one_seed_out: dict[str, Any] = {}
    for scale_index, scale in enumerate(scales):
        primary_rows = rows_for(records, scale, PRIMARY, "hypothesis")
        uniform_rows = rows_for(records, scale, UNIFORM, "hypothesis")
        for omitted in seeds:
            primary_subset = [row for row in primary_rows if int(row["seed"]) != omitted]
            uniform_subset = [row for row in uniform_rows if int(row["seed"]) != omitted]
            paired_subset = []
            uniform_index = {
                (int(row["seed"]), str(row["document_id"]), int(row["start"])): row
                for row in uniform_subset
            }
            for row in primary_subset:
                cell = (int(row["seed"]), str(row["document_id"]), int(row["start"]))
                if cell not in uniform_index:
                    raise ValueError(f"missing leave-one-seed-out pair {cell}")
                paired_subset.append({
                    "seed": cell[0],
                    "document_id": cell[1],
                    "difference": float(row["loss_delta"]) - float(uniform_index[cell]["loss_delta"]),
                })
            leave_one_seed_out[f"{scale}__omit_{omitted}"] = crossed(
                paired_subset,
                lambda row: float(row["difference"]),
                samples=int(cfg["statistics"]["bootstrap_samples"]),
                random_seed=base_seed + 50000 + scale_index * 10000 + omitted,
                confidence=float(cfg["statistics"]["confidence"]),
            )

    audit_passed = (
        not mismatches
        and provenance_pass
        and checkpoint_pass
        and data_pass
        and data_files_pass
        and signal_scales == int(machine["decision"]["signal_scales"])
    )
    audit = {
        "audit_passed": bool(audit_passed),
        "mismatches": mismatches,
        "provenance_pass": provenance_pass,
        "checkpoint_pass": checkpoint_pass,
        "data_pass": data_pass,
        "data_files_pass": data_files_pass,
        "plateau_pass": plateau_pass,
        "rebuilt": rebuilt_scales,
        "gates_without_audit_gate": gates,
        "signal_scales": signal_scales,
        "expected_final_verdict_if_audit_passes": expected_verdict,
        "leave_one_seed_out_primary_minus_uniform_loss": leave_one_seed_out,
    }
    audit_dir = args.output_dir / "adversarial-audit"
    audit_dir.mkdir(parents=True, exist_ok=True)
    (audit_dir / "audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (audit_dir / "VERDICT.md").write_text(
        "# Independent audit — Reality Gate 1A\n\n"
        f"**Audit:** **{'PASS' if audit_passed else 'FAIL'}**\n\n"
        f"Mismatches: `{len(mismatches)}`.\n\n"
        f"Expected final verdict if audit passes: `{expected_verdict}`.\n\n"
        "The auditor independently rebuilt scale-specific seed-document statistics, paired comparisons, budgets, plateau status, data isolation, checkpoint hashes, leave-one-seed-out sensitivity, gate semantics, and the final decision.\n",
        encoding="utf-8",
    )
    print("Audit:", "PASS" if audit_passed else "FAIL")
    if mismatches:
        print("\n".join(mismatches))
    return 0 if audit_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
