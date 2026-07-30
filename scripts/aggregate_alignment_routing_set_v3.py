#!/usr/bin/env python3
"""Aggregate routing-set distillation v3 across seeds and documents."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pre_qwen_certification.metrics import crossed_hierarchical_bootstrap

PRIMARY = "shared-lora-r5-routing-set-v3"
V2 = "shared-lora-r5-expert-v2-control"
V1 = "shared-lora-r5-aggregate-v1-control"
CAPACITY = "shared-lora-r6-routing-set-v3"
NARROW = "narrow65-frozen-baseline"
FULL = "full-continuation-control"

METRICS = (
    "loss_delta",
    "kl_teacher_to_candidate",
    "top1_agreement",
    "local_nrmse",
    "expert_nrmse",
    "counterfactual_nrmse",
    "geometry_mse",
    "routing_self_error",
    "routing_cross_error",
    "routing_aggregate_error",
)


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


def summarize_metric(
    rows: list[dict[str, Any]],
    key: str,
    config: dict[str, Any],
    offset: int,
) -> dict[str, Any]:
    stats = config["statistics"]
    result = crossed_hierarchical_bootstrap(
        rows,
        value_key=key,
        seed_key="seed",
        document_key="document_id",
        samples=int(stats["bootstrap_samples"]),
        random_seed=int(stats["bootstrap_seed"]) + offset,
        confidence=float(stats["confidence"]),
    )
    result["per_seed"] = {
        str(seed): float(
            np.mean([float(row[key]) for row in rows if int(row["seed"]) == seed])
        )
        for seed in sorted({int(row["seed"]) for row in rows})
    }
    return result


def paired_rows(
    payloads: list[dict[str, Any]],
    left: str,
    right: str,
    key: str,
    *,
    split: str = "hypothesis",
) -> list[dict[str, Any]]:
    source = [
        row
        for payload in payloads
        for row in payload["records"]
        if row["candidate"] in {left, right}
        and row["phase"] == "final"
        and row["evaluation_split"] == split
    ]
    index = {
        (int(row["seed"]), str(row["document_id"]), int(row["start"]), str(row["candidate"])): float(row[key])
        for row in source
    }
    keys = sorted(
        (seed, document, start)
        for seed, document, start, candidate in index
        if candidate == left
    )
    missing = [
        (seed, document, start)
        for seed, document, start in keys
        if (seed, document, start, right) not in index
    ]
    if missing:
        raise ValueError(f"missing paired rows for {right}: {missing[:3]}")
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


def bootstrap_difference(
    payloads: list[dict[str, Any]],
    left: str,
    right: str,
    key: str,
    config: dict[str, Any],
    offset: int,
) -> dict[str, Any]:
    stats = config["statistics"]
    return crossed_hierarchical_bootstrap(
        paired_rows(payloads, left, right, key),
        value_key="difference",
        seed_key="seed",
        document_key="document_id",
        samples=int(stats["bootstrap_samples"]),
        random_seed=int(stats["bootstrap_seed"]) + offset,
        confidence=float(stats["confidence"]),
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/pre_qwen_alignment_routing_set_v3.yaml",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "results/pre-qwen-alignment-routing-set/v3",
    )
    parser.add_argument("--audit-path", type=Path, default=None)
    args = parser.parse_args()

    config: dict[str, Any] = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    seeds = [int(value) for value in config["seeds"]]
    payloads = [
        json.loads((args.output_dir / f"seed-{seed}.json").read_text(encoding="utf-8"))
        for seed in seeds
    ]
    names = [str(row["name"]) for row in config["candidates"]]

    candidates: dict[str, Any] = {}
    for candidate_index, name in enumerate(names):
        row: dict[str, Any] = {
            "parameter_ratio": float(
                np.mean([payload["metadata"]["parameter_ratios"][name] for payload in payloads])
            ),
            "compute_ratio": float(
                np.mean([payload["metadata"]["compute_ratios"][name] for payload in payloads])
            ),
        }
        for split_index, split in enumerate(("hypothesis", "ood")):
            rows = selected(payloads, name, split)
            row[split] = {
                key: summarize_metric(
                    rows,
                    key,
                    config,
                    1000 * candidate_index + 100 * split_index + metric_index,
                )
                for metric_index, key in enumerate(METRICS)
            }
        candidates[name] = row

    comparisons = {
        "primary_minus_narrow_loss": bootstrap_difference(
            payloads, PRIMARY, NARROW, "loss_delta", config, 5001
        ),
        "primary_minus_v2_loss": bootstrap_difference(
            payloads, PRIMARY, V2, "loss_delta", config, 5002
        ),
        "primary_minus_v1_loss": bootstrap_difference(
            payloads, PRIMARY, V1, "loss_delta", config, 5003
        ),
        "primary_minus_v2_kl": bootstrap_difference(
            payloads, PRIMARY, V2, "kl_teacher_to_candidate", config, 5004
        ),
        "primary_minus_v2_top1": bootstrap_difference(
            payloads, PRIMARY, V2, "top1_agreement", config, 5005
        ),
        "primary_minus_v2_counterfactual": bootstrap_difference(
            payloads, PRIMARY, V2, "counterfactual_nrmse", config, 5006
        ),
        "primary_minus_v2_cross_error": bootstrap_difference(
            payloads, PRIMARY, V2, "routing_cross_error", config, 5007
        ),
        "primary_minus_narrow_cross_error": bootstrap_difference(
            payloads, PRIMARY, NARROW, "routing_cross_error", config, 5008
        ),
        "v2_minus_narrow_cross_error": bootstrap_difference(
            payloads, V2, NARROW, "routing_cross_error", config, 5009
        ),
    }

    gates_cfg = config["gates"]
    primary = candidates[PRIMARY]
    capacity = candidates[CAPACITY]
    full = candidates[FULL]
    tol = float(gates_cfg["arithmetic_tolerance"])
    clean = all(
        not payload["metadata"]["data_audit"]["exact_cross_split_duplicates"]
        and not payload["metadata"]["data_audit"]["near_cross_split_pairs"]
        for payload in payloads
    )
    audit_pass = False
    if args.audit_path and args.audit_path.exists():
        audit_pass = bool(json.loads(args.audit_path.read_text(encoding="utf-8"))["audit_passed"])

    ph = primary["hypothesis"]
    gates = {
        "primary_hypothesis_pass": ph["loss_delta"]["ucb"]
        <= float(gates_cfg["primary_hypothesis_ucb_max"]),
        "primary_ood_pass": primary["ood"]["loss_delta"]["ucb"]
        <= float(gates_cfg["primary_ood_ucb_max"]),
        "every_seed_primary_pass": max(ph["loss_delta"]["per_seed"].values())
        <= float(gates_cfg["every_seed_primary_hypothesis_delta_max"]),
        "primary_vs_narrow65_pass": comparisons["primary_minus_narrow_loss"]["ucb"]
        <= float(gates_cfg["primary_minus_narrow65_ucb_max"]),
        "primary_vs_expert_v2_pass": comparisons["primary_minus_v2_loss"]["ucb"]
        <= float(gates_cfg["primary_minus_expert_v2_ucb_max"]),
        "primary_kl_pass": ph["kl_teacher_to_candidate"]["ucb"]
        <= float(gates_cfg["primary_kl_ucb_max"]),
        "primary_top1_pass": ph["top1_agreement"]["lcb"]
        >= float(gates_cfg["primary_top1_lcb_min"]),
        "primary_local_nrmse_pass": ph["local_nrmse"]["ucb"]
        <= float(gates_cfg["primary_local_nrmse_ucb_max"]),
        "primary_counterfactual_pass": ph["counterfactual_nrmse"]["ucb"]
        <= float(gates_cfg["primary_counterfactual_nrmse_ucb_max"]),
        "primary_cross_error_gap_pass": comparisons["primary_minus_narrow_cross_error"]["ucb"]
        <= float(gates_cfg["primary_cross_error_gap_to_narrow_ucb_max"]),
        "capacity_hypothesis_pass": capacity["hypothesis"]["loss_delta"]["ucb"]
        <= float(gates_cfg["capacity_hypothesis_ucb_max"]),
        "full_control_hypothesis_pass": full["hypothesis"]["loss_delta"]["ucb"]
        <= float(gates_cfg["full_control_hypothesis_ucb_max"]),
        "full_control_ood_pass": full["ood"]["loss_delta"]["ucb"]
        <= float(gates_cfg["full_control_ood_ucb_max"]),
        "primary_parameter_budget_pass": primary["parameter_ratio"]
        < float(gates_cfg["primary_parameter_ratio_strict_max"]) - tol,
        "primary_compute_budget_pass": primary["compute_ratio"]
        < float(gates_cfg["primary_compute_ratio_strict_max"]) - tol,
        "all_arithmetic_pass": all(
            abs(candidates[str(row["name"])]["parameter_ratio"] - float(row["expected_parameter_ratio"])) <= tol
            and abs(candidates[str(row["name"])]["compute_ratio"] - float(row["expected_compute_ratio"])) <= tol
            for row in config["candidates"]
        ),
        "clean_data_audit_pass": clean,
        "independent_audit_pass": audit_pass,
    }

    load_bearing = (
        "primary_hypothesis_pass",
        "primary_ood_pass",
        "every_seed_primary_pass",
        "primary_vs_narrow65_pass",
        "primary_vs_expert_v2_pass",
        "primary_kl_pass",
        "primary_top1_pass",
        "primary_local_nrmse_pass",
        "primary_counterfactual_pass",
        "primary_cross_error_gap_pass",
        "primary_parameter_budget_pass",
        "primary_compute_budget_pass",
        "all_arithmetic_pass",
        "clean_data_audit_pass",
        "independent_audit_pass",
        "full_control_hypothesis_pass",
        "full_control_ood_pass",
    )
    if all(gates[name] for name in load_bearing):
        verdict = "ALIGNMENT_TOLERANT_ROUTING_SET_V3_PASS"
    else:
        mechanism_votes = (
            comparisons["primary_minus_v2_kl"]["ucb"] <= 0.0,
            comparisons["primary_minus_v2_top1"]["lcb"] >= 0.0,
            comparisons["primary_minus_v2_counterfactual"]["ucb"] <= 0.0,
            comparisons["primary_minus_v2_cross_error"]["ucb"] <= 0.0,
        )
        absolute_core = all(
            gates[name]
            for name in (
                "primary_hypothesis_pass",
                "primary_ood_pass",
                "every_seed_primary_pass",
                "primary_parameter_budget_pass",
                "primary_compute_budget_pass",
                "all_arithmetic_pass",
                "clean_data_audit_pass",
                "independent_audit_pass",
            )
        )
        if absolute_core and sum(bool(value) for value in mechanism_votes) >= 2:
            verdict = "ALIGNMENT_TOLERANT_ROUTING_SET_V3_MECHANISM_SIGNAL"
        else:
            verdict = "ALIGNMENT_TOLERANT_ROUTING_SET_V3_FAIL"

    payload = {
        "metadata": {
            "protocol_version": config["protocol_version"],
            "configuration_sha256": payloads[0]["metadata"]["configuration_sha256"],
            "source_commits": sorted({payload["metadata"]["source_commit"] for payload in payloads}),
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "seeds": seeds,
            "teacher_tail_changes": {
                str(payload["metadata"]["seed"]): payload["metadata"]["source_teacher_tail_change"]
                for payload in payloads
            },
        },
        "candidates": candidates,
        "comparisons": comparisons,
        "decision": {
            "verdict": verdict,
            "gates": gates,
            "frozen_no_go_changed": False,
            "interpretation": config["decision_policy"][
                "pass" if verdict.endswith("_PASS") else "mechanism_signal" if verdict.endswith("MECHANISM_SIGNAL") else "fail"
            ],
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "metrics.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    lines = [
        "# Alignment-tolerant routing-set distillation v3",
        "",
        f"**Decision:** **{verdict}**",
        "",
        "| Candidate | Params | Compute | Hyp delta | UCB95 | KL | Top-1 | Local NRMSE | CF NRMSE | Cross error |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name in names:
        row = candidates[name]
        hyp = row["hypothesis"]
        lines.append(
            f"| {name} | {row['parameter_ratio']:.2%} | {row['compute_ratio']:.2%} | "
            f"{hyp['loss_delta']['mean']:+.5f} | {hyp['loss_delta']['ucb']:+.5f} | "
            f"{hyp['kl_teacher_to_candidate']['mean']:.5f} | {hyp['top1_agreement']['mean']:.2%} | "
            f"{hyp['local_nrmse']['mean']:.5f} | {hyp['counterfactual_nrmse']['mean']:.5f} | "
            f"{hyp['routing_cross_error']['mean']:+.5f} |"
        )
    lines += [
        "",
        f"- v3 minus narrow65 loss: `{comparisons['primary_minus_narrow_loss']['mean']:+.5f}`, 95% `[{comparisons['primary_minus_narrow_loss']['lcb']:+.5f}, {comparisons['primary_minus_narrow_loss']['ucb']:+.5f}]`.",
        f"- v3 minus expert-v2 loss: `{comparisons['primary_minus_v2_loss']['mean']:+.5f}`, 95% `[{comparisons['primary_minus_v2_loss']['lcb']:+.5f}, {comparisons['primary_minus_v2_loss']['ucb']:+.5f}]`.",
        f"- v3 minus narrow65 cross-error: `{comparisons['primary_minus_narrow_cross_error']['mean']:+.5f}`, 95% `[{comparisons['primary_minus_narrow_cross_error']['lcb']:+.5f}, {comparisons['primary_minus_narrow_cross_error']['ucb']:+.5f}]`.",
        "",
        "## Gates",
        "",
    ]
    lines.extend(f"- `{name}`: `{value}`." for name, value in gates.items())
    lines += [
        "",
        "The teachers are inherited fixed checkpoints; no plateau claim is made.",
        "No runtime claim is made. Ratios are exact expert-only analytical proxies.",
        "The frozen NO_GO_FOR_OLMOE_OR_QWEN is unchanged.",
    ]
    (args.output_dir / "VERDICT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print((args.output_dir / "VERDICT.md").read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
