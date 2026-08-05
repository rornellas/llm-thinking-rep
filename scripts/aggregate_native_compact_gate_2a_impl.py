#!/usr/bin/env python3
"""Aggregate Native Compact Gate 2A across scales, seeds, and documents."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Iterable

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pre_qwen_certification.metrics import crossed_hierarchical_bootstrap
from pre_qwen_certification.native_compact import CANDIDATES, FULL, NARROW, PRIMARY


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_payloads(output_dir: Path, scales: list[str], seeds: list[int]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for scale in scales:
        for seed in seeds:
            path = output_dir / scale / f"seed-{seed}.json"
            if not path.exists():
                raise FileNotFoundError(path)
            payload = json.loads(path.read_text(encoding="utf-8"))
            metadata = payload["metadata"]
            if str(metadata["scale"]) != scale or int(metadata["seed"]) != seed:
                raise ValueError(f"scale/seed mismatch in {path}")
            result.append(payload)
    return result


def rows_for(
    records: Iterable[dict[str, Any]],
    *,
    scale: str,
    candidate: str,
    phase: str,
    split: str,
) -> list[dict[str, Any]]:
    return [
        row
        for row in records
        if str(row["scale"]) == scale
        and str(row["candidate"]) == candidate
        and str(row["phase"]) == phase
        and str(row["evaluation_split"]) == split
    ]


def bootstrap(rows: list[dict[str, Any]], key: str, cfg: dict[str, Any], salt: int) -> dict[str, Any]:
    result = crossed_hierarchical_bootstrap(
        rows,
        value_key=key,
        samples=int(cfg["statistics"]["bootstrap_samples"]),
        random_seed=int(cfg["statistics"]["bootstrap_seed"]) + int(salt),
        confidence=float(cfg["statistics"]["confidence"]),
    )
    result["per_seed"] = {
        str(seed): float(np.mean([float(row[key]) for row in rows if int(row["seed"]) == seed]))
        for seed in sorted({int(row["seed"]) for row in rows})
    }
    return result


def paired_rows(left: list[dict[str, Any]], right: list[dict[str, Any]], key: str = "loss") -> list[dict[str, Any]]:
    def cell(row: dict[str, Any]) -> tuple[int, str, int]:
        return int(row["seed"]), str(row["document_id"]), int(row["start"])

    index = {cell(row): row for row in right}
    if len(index) != len(right):
        raise ValueError("duplicate paired cells")
    result: list[dict[str, Any]] = []
    for row in left:
        current = cell(row)
        if current not in index:
            raise ValueError(f"missing paired cell {current}")
        result.append(
            {
                "seed": current[0],
                "document_id": current[1],
                "start": current[2],
                "difference": float(row[key]) - float(index[current][key]),
            }
        )
    if len(result) != len(right):
        raise ValueError("paired row count mismatch")
    return result


def paired(
    records: list[dict[str, Any]],
    cfg: dict[str, Any],
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
    return bootstrap(rows, "difference", cfg, salt)


def maturity(history: list[dict[str, Any]]) -> dict[str, float]:
    if len(history) < 3:
        return {"slope_per_step": float("nan"), "range": float("nan")}
    recent = history[-min(4, len(history)) :]
    steps = np.asarray([float(row["step"]) for row in recent])
    losses = np.asarray([float(row["validation_loss"]) for row in recent])
    slope = float(np.polyfit(steps - steps.mean(), losses, 1)[0])
    return {
        "slope_per_step": slope,
        "range": float(losses.max() - losses.min()),
        "final_validation_loss": float(losses[-1]),
    }


def write_verdict(path: Path, metrics: dict[str, Any]) -> None:
    lines = [
        "# Native Compact Gate 2A — native shared-rank training",
        "",
        f"**Decision:** **{metrics['decision']['verdict']}**",
        "",
    ]
    for scale, current in metrics["scales"].items():
        lines.extend(
            [
                f"## Scale `{scale}`",
                "",
                "| Candidate | Hyp loss | UCB95 | OOD loss | Expert params | Total params | Expert compute |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for candidate in CANDIDATES:
            stats = current["candidates"][candidate]
            accounting = current["accounting"][candidate]
            lines.append(
                "| {name} | {loss:.5f} | {ucb:.5f} | {ood:.5f} | {ep:.2%} | {tp:.2%} | {compute:.2%} |".format(
                    name=candidate,
                    loss=float(stats["final"]["hypothesis"]["loss"]["mean"]),
                    ucb=float(stats["final"]["hypothesis"]["loss"]["ucb"]),
                    ood=float(stats["final"]["ood"]["loss"]["mean"]),
                    ep=float(accounting["expert_parameter_ratio"]),
                    tp=float(accounting["total_parameter_ratio"]),
                    compute=float(accounting["expert_compute_ratio"]),
                )
            )
        lines.extend(["", "### Paired differences", ""])
        for name, value in current["comparisons"].items():
            lines.append(
                f"- `{name}`: `{value['mean']:+.6f}` 95% `[{value['lcb']:+.6f}, {value['ucb']:+.6f}]`."
            )
        lines.extend(["", "### Optimization maturity", ""])
        for candidate, by_seed in current["maturity"].items():
            slopes = [float(value["slope_per_step"]) for value in by_seed.values()]
            lines.append(f"- `{candidate}` mean terminal calibration slope: `{np.nanmean(slopes):+.3e}` per step.")
        lines.append("")
    lines.extend(["## Gates", ""])
    for name, value in metrics["decision"]["gates"].items():
        lines.append(f"- `{name}`: `{value}`.")
    lines.extend(
        [
            "",
            "The primary endpoint is final fixed-budget hypothesis loss; best-calibration results are secondary.",
            "Compute is an analytical expert-matrix proxy. No runtime speedup is claimed.",
            "`NO_GO_FOR_OLMOE_OR_QWEN` remains unchanged.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/native_compact_gate_2a.yaml",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "results/native-compact-gate-2a",
    )
    parser.add_argument("--audit-path", type=Path, default=None)
    args = parser.parse_args()

    cfg: dict[str, Any] = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    scales = list(cfg["scales"])
    seeds = [int(value) for value in cfg["seeds"]]
    payloads = load_payloads(args.output_dir, scales, seeds)
    protocol = str(cfg["protocol_version"])
    config_hash = sha256_file(args.config)
    if {str(payload["metadata"]["protocol_version"]) for payload in payloads} != {protocol}:
        raise ValueError("protocol mismatch")
    if {str(payload["metadata"]["configuration_sha256"]) for payload in payloads} != {config_hash}:
        raise ValueError("configuration hash mismatch")
    if len({str(payload["metadata"]["data_manifest_sha256"]) for payload in payloads}) != 1:
        raise ValueError("data manifest mismatch")
    if len({str(payload["metadata"]["source_commit"]) for payload in payloads}) != 1:
        raise ValueError("source commit mismatch")
    records = [row for payload in payloads for row in payload["records"]]

    gates_cfg = cfg["gates"]
    output_scales: dict[str, Any] = {}
    global_gates: dict[str, bool] = {}
    signal_scales = 0
    for scale_index, scale in enumerate(scales):
        current_payloads = [payload for payload in payloads if payload["metadata"]["scale"] == scale]
        candidate_stats: dict[str, Any] = {}
        for candidate_index, candidate in enumerate(CANDIDATES):
            candidate_stats[candidate] = {}
            for phase_index, phase in enumerate(("final", "best-calibration")):
                candidate_stats[candidate][phase] = {}
                for split_index, split in enumerate(("hypothesis", "ood")):
                    rows = rows_for(
                        records,
                        scale=scale,
                        candidate=candidate,
                        phase=phase,
                        split=split,
                    )
                    candidate_stats[candidate][phase][split] = {
                        key: bootstrap(
                            rows,
                            key,
                            cfg,
                            scale_index * 100000
                            + candidate_index * 10000
                            + phase_index * 1000
                            + split_index * 100
                            + key_index,
                        )
                        for key_index, key in enumerate(("loss", "entropy", "confidence"))
                    }
        comparisons = {
            "primary_minus_narrow_hypothesis": paired(
                records,
                cfg,
                scale=scale,
                left=PRIMARY,
                right=NARROW,
                phase="final",
                split="hypothesis",
                salt=scale_index * 100000 + 80001,
            ),
            "primary_minus_narrow_ood": paired(
                records,
                cfg,
                scale=scale,
                left=PRIMARY,
                right=NARROW,
                phase="final",
                split="ood",
                salt=scale_index * 100000 + 80002,
            ),
            "primary_minus_full_hypothesis": paired(
                records,
                cfg,
                scale=scale,
                left=PRIMARY,
                right=FULL,
                phase="final",
                split="hypothesis",
                salt=scale_index * 100000 + 80003,
            ),
            "best_primary_minus_narrow_hypothesis": paired(
                records,
                cfg,
                scale=scale,
                left=PRIMARY,
                right=NARROW,
                phase="best-calibration",
                split="hypothesis",
                salt=scale_index * 100000 + 80004,
            ),
        }
        accounting = {
            candidate: current_payloads[0]["metadata"]["accounting"][candidate]
            for candidate in CANDIDATES
        }
        for payload in current_payloads[1:]:
            if payload["metadata"]["accounting"] != current_payloads[0]["metadata"]["accounting"]:
                raise ValueError(f"accounting changed across seeds for {scale}")
        clean_data = all(
            not any(payload["metadata"]["document_hash_overlap"].values())
            for payload in current_payloads
        )
        primary_dead = max(
            int(
                payload["metadata"]["routing"]["final"][PRIMARY]["hypothesis"]["health"]["dead_experts"]
            )
            for payload in current_payloads
        )
        maturity_by_candidate = {
            candidate: {
                str(payload["metadata"]["seed"]): maturity(
                    payload["metadata"]["training"]["histories"][candidate]
                )
                for payload in current_payloads
            }
            for candidate in CANDIDATES
        }
        per_seed_difference = comparisons["primary_minus_narrow_hypothesis"]["per_seed"]
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
            "every_seed_noninferior": max(float(value) for value in per_seed_difference.values())
            <= float(gates_cfg["every_seed_primary_minus_narrow_loss_max"]),
            "routing_health": primary_dead <= int(gates_cfg["maximum_dead_experts"]),
        }
        for name, value in scale_gates.items():
            global_gates[f"{scale}__{name}"] = bool(value)
        if (
            scale_gates["clean_data"]
            and scale_gates["primary_parameter_advantage"]
            and scale_gates["primary_compute_budget"]
            and scale_gates["routing_health"]
            and comparisons["primary_minus_narrow_hypothesis"]["ucb"]
            <= float(gates_cfg["signal_primary_vs_narrow_hypothesis_loss_ucb_max"])
        ):
            signal_scales += 1
        output_scales[scale] = {
            "candidates": candidate_stats,
            "comparisons": comparisons,
            "accounting": accounting,
            "maturity": maturity_by_candidate,
            "gates": scale_gates,
        }

    small_diff = float(output_scales[scales[0]]["comparisons"]["primary_minus_narrow_hypothesis"]["mean"])
    medium_diff = float(output_scales[scales[-1]]["comparisons"]["primary_minus_narrow_hypothesis"]["mean"])
    scale_trend = medium_diff <= small_diff + float(gates_cfg["maximum_scale_regression"])
    global_gates["scale_trend"] = bool(scale_trend)

    audit_passed = False
    audit_summary: dict[str, Any] | None = None
    if args.audit_path is not None:
        audit_summary = json.loads(args.audit_path.read_text(encoding="utf-8"))
        audit_passed = bool(audit_summary.get("audit_passed"))
    global_gates["independent_audit"] = audit_passed

    load_bearing_suffixes = (
        "clean_data",
        "primary_parameter_advantage",
        "primary_compute_budget",
        "primary_hypothesis_noninferior",
        "primary_ood_noninferior",
        "every_seed_noninferior",
        "routing_health",
    )
    core_pass = all(
        global_gates[f"{scale}__{suffix}"]
        for scale in scales
        for suffix in load_bearing_suffixes
    ) and scale_trend
    if core_pass:
        expected = "NATIVE_COMPACT_GATE_2A_PASS"
    elif signal_scales >= int(gates_cfg["signal_min_scales"]):
        expected = "NATIVE_COMPACT_GATE_2A_MECHANISM_SIGNAL"
    else:
        expected = "NATIVE_COMPACT_GATE_2A_FAIL"
    if args.audit_path is None:
        verdict = "NATIVE_COMPACT_GATE_2A_PENDING_INDEPENDENT_AUDIT"
    elif audit_passed:
        verdict = expected
    else:
        verdict = "NATIVE_COMPACT_GATE_2A_FAIL_INTEGRITY"

    metrics = {
        "protocol_version": protocol,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_commit": payloads[0]["metadata"]["source_commit"],
        "configuration_sha256": config_hash,
        "data_manifest_sha256": payloads[0]["metadata"]["data_manifest_sha256"],
        "scales": output_scales,
        "decision": {
            "verdict": verdict,
            "expected_if_audit_passes": expected,
            "signal_scales": int(signal_scales),
            "gates": global_gates,
            "audit": audit_summary,
            "no_go_for_real_checkpoint": True,
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = args.output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_verdict(args.output_dir / "VERDICT.md", metrics)
    report = ROOT / "docs/results/2026-08-05-native-compact-gate-2a-analysis.md"
    report.parent.mkdir(parents=True, exist_ok=True)
    write_verdict(report, metrics)
    print(metrics_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
