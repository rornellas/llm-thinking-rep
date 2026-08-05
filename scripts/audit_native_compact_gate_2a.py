#!/usr/bin/env python3
"""Independent reconstruction and audit of Native Compact Gate 2A."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import yaml

FULL = "conventional-full"
NARROW = "conventional-narrow65"
PRIMARY = "native-shared-rank"
CANDIDATES = (FULL, NARROW, PRIMARY)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_payloads(output_dir: Path, scales: Sequence[str], seeds: Sequence[int]) -> list[dict[str, Any]]:
    payloads: list[dict[str, Any]] = []
    for scale in scales:
        for seed in seeds:
            path = output_dir / scale / f"seed-{seed}.json"
            payload = json.loads(path.read_text(encoding="utf-8"))
            metadata = payload["metadata"]
            if str(metadata["scale"]) != str(scale) or int(metadata["seed"]) != int(seed):
                raise ValueError(f"scale/seed mismatch: {path}")
            payloads.append(payload)
    return payloads


def rows_for(
    records: Iterable[Mapping[str, Any]],
    *,
    scale: str,
    candidate: str,
    phase: str,
    split: str,
) -> list[dict[str, Any]]:
    return [
        dict(row)
        for row in records
        if str(row["scale"]) == scale
        and str(row["candidate"]) == candidate
        and str(row["phase"]) == phase
        and str(row["evaluation_split"]) == split
    ]


def crossed_matrix(
    records: Sequence[Mapping[str, Any]],
    value_key: str,
) -> tuple[np.ndarray, list[int], list[str]]:
    seeds = sorted({int(row["seed"]) for row in records})
    documents = sorted({str(row["document_id"]) for row in records})
    seed_index = {value: index for index, value in enumerate(seeds)}
    document_index = {value: index for index, value in enumerate(documents)}
    buckets: list[list[list[float]]] = [[[] for _ in documents] for _ in seeds]
    for row in records:
        value = float(row[value_key])
        if not np.isfinite(value):
            raise ValueError("non-finite audit value")
        buckets[seed_index[int(row["seed"])]][document_index[str(row["document_id"])]].append(value)
    matrix = np.full((len(seeds), len(documents)), np.nan, dtype=np.float64)
    for seed_id, row in enumerate(buckets):
        for document_id, values in enumerate(row):
            if values:
                matrix[seed_id, document_id] = float(np.mean(values))
    return matrix, seeds, documents


def crossed_bootstrap(
    records: Sequence[Mapping[str, Any]],
    *,
    value_key: str,
    samples: int,
    random_seed: int,
    confidence: float,
) -> dict[str, Any]:
    matrix, seeds, documents = crossed_matrix(records, value_key)
    if matrix.size == 0 or np.all(np.isnan(matrix)):
        raise ValueError("no observations for audit bootstrap")
    rng = np.random.default_rng(random_seed)
    draws = np.empty(samples, dtype=np.float64)
    for index in range(samples):
        sampled_seeds = rng.integers(0, matrix.shape[0], size=matrix.shape[0])
        sampled_documents = rng.integers(0, matrix.shape[1], size=matrix.shape[1])
        draws[index] = float(np.nanmean(matrix[np.ix_(sampled_seeds, sampled_documents)]))
    alpha = 1.0 - confidence
    return {
        "mean": float(np.nanmean(matrix)),
        "lcb": float(np.quantile(draws, alpha / 2.0)),
        "ucb": float(np.quantile(draws, 1.0 - alpha / 2.0)),
        "confidence": confidence,
        "bootstrap_samples": samples,
        "training_seeds": [str(value) for value in seeds],
        "documents": documents,
        "effective_cells": int(np.isfinite(matrix).sum()),
        "per_seed": {
            str(seed): float(
                np.mean([float(row[value_key]) for row in records if int(row["seed"]) == seed])
            )
            for seed in seeds
        },
    }


def paired_rows(left: Sequence[Mapping[str, Any]], right: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    def cell(row: Mapping[str, Any]) -> tuple[int, str, int]:
        return int(row["seed"]), str(row["document_id"]), int(row["start"])

    right_index = {cell(row): row for row in right}
    if len(right_index) != len(right):
        raise ValueError("duplicate right-side paired cells")
    result: list[dict[str, Any]] = []
    for row in left:
        key = cell(row)
        if key not in right_index:
            raise ValueError(f"missing paired audit cell: {key}")
        result.append(
            {
                "seed": key[0],
                "document_id": key[1],
                "start": key[2],
                "difference": float(row["loss"]) - float(right_index[key]["loss"]),
            }
        )
    if len(result) != len(right):
        raise ValueError("paired audit row count mismatch")
    return result


def paired(
    records: Sequence[Mapping[str, Any]],
    cfg: Mapping[str, Any],
    *,
    scale: str,
    left: str,
    right: str,
    phase: str,
    split: str,
    salt: int,
) -> dict[str, Any]:
    rows = paired_rows(
        rows_for(records, scale=scale, candidate=left, phase=phase, split=split),
        rows_for(records, scale=scale, candidate=right, phase=phase, split=split),
    )
    return crossed_bootstrap(
        rows,
        value_key="difference",
        samples=int(cfg["statistics"]["bootstrap_samples"]),
        random_seed=int(cfg["statistics"]["bootstrap_seed"]) + int(salt),
        confidence=float(cfg["statistics"]["confidence"]),
    )


def approximately_equal(left: Any, right: Any, tolerance: float = 1e-12) -> bool:
    if isinstance(left, dict) and isinstance(right, dict):
        if set(left) != set(right):
            return False
        return all(approximately_equal(left[key], right[key], tolerance) for key in left)
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(
            approximately_equal(a, b, tolerance) for a, b in zip(left, right, strict=True)
        )
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        if np.isnan(float(left)) and np.isnan(float(right)):
            return True
        return abs(float(left) - float(right)) <= tolerance
    return left == right


def add_mismatch(mismatches: list[dict[str, Any]], name: str, expected: Any, observed: Any) -> None:
    if not approximately_equal(expected, observed):
        mismatches.append({"name": name, "expected": expected, "observed": observed})


def verify_data(root: Path, manifest: Mapping[str, Any], mismatches: list[dict[str, Any]]) -> bool:
    passed = True
    for filename, expected in manifest["sha256"].items():
        path = root / filename
        observed = sha256_file(path) if path.exists() else None
        if observed != str(expected):
            passed = False
            mismatches.append(
                {"name": f"data_sha256::{filename}", "expected": str(expected), "observed": observed}
            )
    return passed


def write_verdict(path: Path, audit: Mapping[str, Any]) -> None:
    lines = [
        "# Independent audit — Native Compact Gate 2A",
        "",
        f"**Audit:** **{'PASS' if audit['audit_passed'] else 'FAIL'}**",
        "",
        f"Mismatches: `{len(audit['mismatches'])}`.",
        "",
        f"Expected final verdict if audit passes: `{audit['expected_final_verdict_if_audit_passes']}`.",
        "",
        "The auditor independently reconstructed paired seed-document statistics, budgets, routing health, provenance, checkpoint hashes, data isolation, scale trend, and gate semantics.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    args = parser.parse_args()

    cfg: dict[str, Any] = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    metrics = json.loads((args.output_dir / "metrics.json").read_text(encoding="utf-8"))
    scales = list(cfg["scales"])
    seeds = [int(value) for value in cfg["seeds"]]
    payloads = load_payloads(args.output_dir, scales, seeds)
    records = [row for payload in payloads for row in payload["records"]]
    mismatches: list[dict[str, Any]] = []

    protocol = str(cfg["protocol_version"])
    config_hash = sha256_file(args.config)
    add_mismatch(mismatches, "protocol", protocol, metrics["protocol_version"])
    add_mismatch(mismatches, "configuration_sha256", config_hash, metrics["configuration_sha256"])
    payload_protocols = {str(payload["metadata"]["protocol_version"]) for payload in payloads}
    payload_configs = {str(payload["metadata"]["configuration_sha256"]) for payload in payloads}
    payload_manifests = {str(payload["metadata"]["data_manifest_sha256"]) for payload in payloads}
    payload_commits = {str(payload["metadata"]["source_commit"]) for payload in payloads}
    add_mismatch(mismatches, "payload_protocols", [protocol], sorted(payload_protocols))
    add_mismatch(mismatches, "payload_config_hashes", [config_hash], sorted(payload_configs))
    add_mismatch(mismatches, "payload_manifest_count", 1, len(payload_manifests))
    add_mismatch(mismatches, "payload_source_commit_count", 1, len(payload_commits))
    if payload_manifests:
        add_mismatch(
            mismatches,
            "metrics_data_manifest_sha256",
            next(iter(payload_manifests)),
            metrics["data_manifest_sha256"],
        )
    if payload_commits:
        add_mismatch(mismatches, "metrics_source_commit", next(iter(payload_commits)), metrics["source_commit"])

    data_manifest_path = args.data_root / "manifest.json"
    data_manifest = json.loads(data_manifest_path.read_text(encoding="utf-8"))
    data_manifest_hash = sha256_file(data_manifest_path)
    add_mismatch(mismatches, "data_manifest_file_hash", metrics["data_manifest_sha256"], data_manifest_hash)
    add_mismatch(mismatches, "data_schema_version", "native-compact-wikitext103-article-v1", data_manifest.get("schema_version"))
    add_mismatch(mismatches, "data_source", "Salesforce/wikitext", data_manifest.get("source"))
    add_mismatch(mismatches, "data_subset", "wikitext-103-raw-v1", data_manifest.get("subset"))
    add_mismatch(mismatches, "tokenizer_training_documents", 2048, data_manifest.get("tokenizer_training_documents"))
    data_pass = verify_data(args.data_root, data_manifest, mismatches)

    checkpoint_pass = True
    provenance_pass = True
    data_isolation_pass = True
    for payload in payloads:
        metadata = payload["metadata"]
        checkpoint_path = Path(str(metadata["checkpoint_path"]))
        if not checkpoint_path.exists():
            checkpoint_pass = False
            mismatches.append({"name": f"missing_checkpoint::{metadata['scale']}::{metadata['seed']}"})
        else:
            observed = sha256_file(checkpoint_path)
            if observed != str(metadata["checkpoint_sha256"]):
                checkpoint_pass = False
                mismatches.append(
                    {
                        "name": f"checkpoint_sha256::{metadata['scale']}::{metadata['seed']}",
                        "expected": str(metadata["checkpoint_sha256"]),
                        "observed": observed,
                    }
                )
        if not bool(metadata.get("heldout_loaded_after_candidate_freeze")):
            provenance_pass = False
            mismatches.append({"name": f"heldout_freeze::{metadata['scale']}::{metadata['seed']}"})
        if any(bool(value) for value in metadata["document_hash_overlap"].values()):
            data_isolation_pass = False
            mismatches.append({"name": f"document_overlap::{metadata['scale']}::{metadata['seed']}"})

    rebuilt: dict[str, Any] = {}
    gates_cfg = cfg["gates"]
    rebuilt_gates: dict[str, bool] = {}
    signal_scales = 0
    for scale_index, scale in enumerate(scales):
        current_payloads = [payload for payload in payloads if payload["metadata"]["scale"] == scale]
        candidate_loss_stats: dict[str, Any] = {}
        for candidate_index, candidate in enumerate(CANDIDATES):
            candidate_loss_stats[candidate] = {}
            for phase_index, phase in enumerate(("final", "best-calibration")):
                candidate_loss_stats[candidate][phase] = {}
                for split_index, split in enumerate(("hypothesis", "ood")):
                    rows = rows_for(
                        records, scale=scale, candidate=candidate, phase=phase, split=split
                    )
                    value = crossed_bootstrap(
                        rows,
                        value_key="loss",
                        samples=int(cfg["statistics"]["bootstrap_samples"]),
                        random_seed=(
                            int(cfg["statistics"]["bootstrap_seed"])
                            + scale_index * 100000
                            + candidate_index * 10000
                            + phase_index * 1000
                            + split_index * 100
                        ),
                        confidence=float(cfg["statistics"]["confidence"]),
                    )
                    candidate_loss_stats[candidate][phase][split] = value
                    add_mismatch(
                        mismatches,
                        f"candidate_loss::{scale}::{candidate}::{phase}::{split}",
                        value,
                        metrics["scales"][scale]["candidates"][candidate][phase][split]["loss"],
                    )

        comparisons = {
            "primary_minus_narrow_hypothesis": paired(
                records, cfg, scale=scale, left=PRIMARY, right=NARROW,
                phase="final", split="hypothesis", salt=scale_index * 100000 + 80001,
            ),
            "primary_minus_narrow_ood": paired(
                records, cfg, scale=scale, left=PRIMARY, right=NARROW,
                phase="final", split="ood", salt=scale_index * 100000 + 80002,
            ),
            "primary_minus_full_hypothesis": paired(
                records, cfg, scale=scale, left=PRIMARY, right=FULL,
                phase="final", split="hypothesis", salt=scale_index * 100000 + 80003,
            ),
            "best_primary_minus_narrow_hypothesis": paired(
                records, cfg, scale=scale, left=PRIMARY, right=NARROW,
                phase="best-calibration", split="hypothesis", salt=scale_index * 100000 + 80004,
            ),
        }
        for name, value in comparisons.items():
            add_mismatch(
                mismatches,
                f"comparison::{scale}::{name}",
                value,
                metrics["scales"][scale]["comparisons"][name],
            )
        accounting = current_payloads[0]["metadata"]["accounting"]
        for payload in current_payloads[1:]:
            add_mismatch(
                mismatches,
                f"accounting_seed_stability::{scale}::{payload['metadata']['seed']}",
                accounting,
                payload["metadata"]["accounting"],
            )
        add_mismatch(mismatches, f"accounting::{scale}", accounting, metrics["scales"][scale]["accounting"])
        clean_data = all(
            not any(payload["metadata"]["document_hash_overlap"].values())
            for payload in current_payloads
        )
        dead_experts = max(
            int(payload["metadata"]["routing"]["final"][PRIMARY]["hypothesis"]["health"]["dead_experts"])
            for payload in current_payloads
        )
        per_seed = comparisons["primary_minus_narrow_hypothesis"]["per_seed"]
        scale_gates = {
            "clean_data": clean_data,
            "primary_parameter_advantage": (
                float(accounting[NARROW]["expert_parameter_ratio"])
                - float(accounting[PRIMARY]["expert_parameter_ratio"])
                >= float(gates_cfg["minimum_expert_parameter_advantage"])
            ),
            "primary_compute_budget": float(accounting[PRIMARY]["expert_compute_ratio"])
            <= float(accounting[NARROW]["expert_compute_ratio"])
            + float(gates_cfg["budget_tolerance"]),
            "primary_hypothesis_noninferior": comparisons["primary_minus_narrow_hypothesis"]["ucb"]
            <= float(gates_cfg["primary_vs_narrow_hypothesis_loss_ucb_max"]),
            "primary_ood_noninferior": comparisons["primary_minus_narrow_ood"]["ucb"]
            <= float(gates_cfg["primary_vs_narrow_ood_loss_ucb_max"]),
            "primary_full_upper_bound": comparisons["primary_minus_full_hypothesis"]["ucb"]
            <= float(gates_cfg["primary_vs_full_hypothesis_loss_ucb_max"]),
            "every_seed_noninferior": max(float(value) for value in per_seed.values())
            <= float(gates_cfg["every_seed_primary_minus_narrow_loss_max"]),
            "routing_health": dead_experts <= int(gates_cfg["maximum_dead_experts"]),
        }
        for name, value in scale_gates.items():
            rebuilt_gates[f"{scale}__{name}"] = bool(value)
        if (
            scale_gates["clean_data"]
            and scale_gates["primary_parameter_advantage"]
            and scale_gates["primary_compute_budget"]
            and scale_gates["routing_health"]
            and comparisons["primary_minus_narrow_hypothesis"]["ucb"]
            <= float(gates_cfg["signal_primary_vs_narrow_hypothesis_loss_ucb_max"])
        ):
            signal_scales += 1
        rebuilt[scale] = {"candidate_loss": candidate_loss_stats, "comparisons": comparisons, "accounting": accounting, "gates": scale_gates}

    first_diff = float(rebuilt[scales[0]]["comparisons"]["primary_minus_narrow_hypothesis"]["mean"])
    last_diff = float(rebuilt[scales[-1]]["comparisons"]["primary_minus_narrow_hypothesis"]["mean"])
    scale_trend = last_diff <= first_diff + float(gates_cfg["maximum_scale_regression"])
    rebuilt_gates["scale_trend"] = bool(scale_trend)

    load_bearing = (
        "clean_data",
        "primary_parameter_advantage",
        "primary_compute_budget",
        "primary_hypothesis_noninferior",
        "primary_ood_noninferior",
        "every_seed_noninferior",
        "routing_health",
    )
    core_pass = all(
        rebuilt_gates[f"{scale}__{suffix}"]
        for scale in scales
        for suffix in load_bearing
    ) and scale_trend
    if core_pass:
        expected_verdict = "NATIVE_COMPACT_GATE_2A_PASS"
    elif signal_scales >= int(gates_cfg["signal_min_scales"]):
        expected_verdict = "NATIVE_COMPACT_GATE_2A_MECHANISM_SIGNAL"
    else:
        expected_verdict = "NATIVE_COMPACT_GATE_2A_FAIL"

    add_mismatch(
        mismatches,
        "expected_verdict",
        expected_verdict,
        metrics["decision"]["expected_if_audit_passes"],
    )
    for name, value in rebuilt_gates.items():
        if name in metrics["decision"]["gates"]:
            add_mismatch(mismatches, f"gate::{name}", bool(value), bool(metrics["decision"]["gates"][name]))
        else:
            mismatches.append({"name": f"missing_metric_gate::{name}"})

    audit = {
        "audit_passed": len(mismatches) == 0,
        "mismatches": mismatches,
        "expected_final_verdict_if_audit_passes": expected_verdict,
        "protocol_version": protocol,
        "source_commit": next(iter(payload_commits)) if len(payload_commits) == 1 else None,
        "configuration_sha256": config_hash,
        "data_manifest_sha256": data_manifest_hash,
        "data_pass": data_pass,
        "checkpoint_pass": checkpoint_pass,
        "provenance_pass": provenance_pass,
        "data_isolation_pass": data_isolation_pass,
        "signal_scales": signal_scales,
        "rebuilt_gates_without_audit": rebuilt_gates,
        "rebuilt": rebuilt,
    }
    audit_dir = args.output_dir / "adversarial-audit"
    audit_dir.mkdir(parents=True, exist_ok=True)
    (audit_dir / "audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    write_verdict(audit_dir / "VERDICT.md", audit)
    print(audit_dir / "audit.json")
    return 0 if audit["audit_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
