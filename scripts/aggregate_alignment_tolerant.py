#!/usr/bin/env python3
"""Aggregate the preregistered alignment-tolerant shared-factor screen."""
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


def seed_payload_path(output_dir: Path, seed: int) -> Path:
    nested = output_dir / f"seed-{seed}" / f"seed-{seed}.json"
    if nested.exists():
        return nested
    return output_dir / f"seed-{seed}.json"


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


def summarize(
    rows: list[dict[str, Any]], config: dict[str, Any], offset: int
) -> dict[str, Any]:
    stats = config["statistics"]
    result = crossed_hierarchical_bootstrap(
        rows,
        value_key="loss_delta",
        seed_key="seed",
        document_key="document_id",
        samples=int(stats["bootstrap_samples"]),
        random_seed=int(stats["bootstrap_seed"]) + offset,
        confidence=float(stats["confidence"]),
    )
    result["per_seed"] = {
        str(seed): float(
            np.mean(
                [
                    float(row["loss_delta"])
                    for row in rows
                    if int(row["seed"]) == seed
                ]
            )
        )
        for seed in sorted({int(row["seed"]) for row in rows})
    }
    result["kl_mean"] = float(
        np.mean([float(row["kl_teacher_to_candidate"]) for row in rows])
    )
    result["top1_mean"] = float(
        np.mean([float(row["top1_agreement"]) for row in rows])
    )
    result["local_nrmse_mean"] = float(
        np.mean([float(row["local_nrmse"]) for row in rows])
    )
    return result


def paired(
    payloads: list[dict[str, Any]], left: str, right: str
) -> list[dict[str, Any]]:
    source = [
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
        for row in source
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
    parser.add_argument("--audit-path", type=Path, default=None)
    args = parser.parse_args()

    config: dict[str, Any] = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    seeds = [int(value) for value in config["seeds"]]
    payloads = [
        json.loads(seed_payload_path(args.output_dir, seed).read_text(encoding="utf-8"))
        for seed in seeds
    ]
    names = [str(row["name"]) for row in config["candidates"]]
    candidates: dict[str, Any] = {}
    for index, name in enumerate(names):
        candidates[name] = {
            "parameter_ratio": float(
                np.mean(
                    [payload["metadata"]["parameter_ratios"][name] for payload in payloads]
                )
            ),
            "compute_ratio": float(
                np.mean(
                    [payload["metadata"]["compute_ratios"][name] for payload in payloads]
                )
            ),
            "hypothesis": summarize(
                selected(payloads, name, "hypothesis"), config, 100 + index
            ),
            "ood": summarize(selected(payloads, name, "ood"), config, 200 + index),
        }

    stats = config["statistics"]
    primary_difference = crossed_hierarchical_bootstrap(
        paired(payloads, "shared-lora-r5", "narrow65-frozen-baseline"),
        value_key="difference",
        seed_key="seed",
        document_key="document_id",
        samples=int(stats["bootstrap_samples"]),
        random_seed=int(stats["bootstrap_seed"]) + 500,
        confidence=float(stats["confidence"]),
    )
    capacity_difference = crossed_hierarchical_bootstrap(
        paired(payloads, "shared-lora-r6", "narrow65-frozen-baseline"),
        value_key="difference",
        seed_key="seed",
        document_key="document_id",
        samples=int(stats["bootstrap_samples"]),
        random_seed=int(stats["bootstrap_seed"]) + 501,
        confidence=float(stats["confidence"]),
    )

    primary = candidates["shared-lora-r5"]
    capacity = candidates["shared-lora-r6"]
    full = candidates["full-continuation-control"]
    gates_cfg = config["gates"]
    tolerance = float(gates_cfg["arithmetic_tolerance"])
    expected = {str(row["name"]): row for row in config["candidates"]}
    arithmetic = {
        name: abs(
            candidates[name]["parameter_ratio"]
            - float(expected[name]["expected_parameter_ratio"])
        )
        <= tolerance
        and abs(
            candidates[name]["compute_ratio"]
            - float(expected[name]["expected_compute_ratio"])
        )
        <= tolerance
        for name in names
    }
    clean = all(
        not payload["metadata"]["data_audit"]["exact_cross_split_duplicates"]
        and not payload["metadata"]["data_audit"]["near_cross_split_pairs"]
        for payload in payloads
    )
    audit_pass = False
    if args.audit_path and args.audit_path.exists():
        audit_pass = bool(
            json.loads(args.audit_path.read_text(encoding="utf-8"))["audit_passed"]
        )

    gates = {
        "primary_hypothesis_pass": primary["hypothesis"]["ucb"]
        <= float(gates_cfg["primary_hypothesis_ucb_max"]),
        "primary_ood_pass": primary["ood"]["ucb"]
        <= float(gates_cfg["primary_ood_ucb_max"]),
        "every_seed_primary_pass": max(primary["hypothesis"]["per_seed"].values())
        <= float(gates_cfg["every_seed_primary_hypothesis_delta_max"]),
        "primary_vs_narrow65_pass": primary_difference["ucb"]
        <= float(gates_cfg["primary_minus_narrow65_ucb_max"]),
        "primary_parameter_budget_pass": primary["parameter_ratio"]
        < float(gates_cfg["primary_parameter_ratio_strict_max"]),
        "primary_compute_budget_pass": primary["compute_ratio"]
        < float(gates_cfg["primary_compute_ratio_strict_max"]),
        "capacity_hypothesis_pass": capacity["hypothesis"]["ucb"]
        <= float(gates_cfg["capacity_hypothesis_ucb_max"]),
        "capacity_vs_narrow65_pass": capacity_difference["ucb"]
        <= float(gates_cfg["capacity_minus_narrow65_ucb_max"]),
        "full_control_hypothesis_pass": full["hypothesis"]["ucb"]
        <= float(gates_cfg["full_control_hypothesis_ucb_max"]),
        "full_control_ood_pass": full["ood"]["ucb"]
        <= float(gates_cfg["full_control_ood_ucb_max"]),
        "all_arithmetic_pass": all(arithmetic.values()),
        "clean_data_audit_pass": clean,
        "independent_audit_pass": audit_pass,
    }

    primary_load_bearing = (
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
    promising_load_bearing = (
        "capacity_hypothesis_pass",
        "capacity_vs_narrow65_pass",
        "full_control_hypothesis_pass",
        "full_control_ood_pass",
        "all_arithmetic_pass",
        "clean_data_audit_pass",
        "independent_audit_pass",
    )
    if all(gates[name] for name in primary_load_bearing):
        verdict = "ALIGNMENT_TOLERANT_SHARED_LORA_PASS"
    elif all(gates[name] for name in promising_load_bearing):
        verdict = "ALIGNMENT_TOLERANT_SHARED_LORA_PROMISING_ONLY"
    else:
        verdict = "ALIGNMENT_TOLERANT_SHARED_LORA_FAIL"

    result = {
        "metadata": {
            "protocol_version": config["protocol_version"],
            "configuration_sha256": payloads[0]["metadata"]["configuration_sha256"],
            "source_commits": sorted(
                {payload["metadata"]["source_commit"] for payload in payloads}
            ),
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "seeds": seeds,
            "source_teacher_tail_changes": {
                str(payload["metadata"]["seed"]): payload["metadata"][
                    "source_teacher_tail_change"
                ]
                for payload in payloads
            },
        },
        "candidates": candidates,
        "comparisons": {
            "primary_minus_narrow65": primary_difference,
            "capacity_minus_narrow65": capacity_difference,
        },
        "arithmetic_checks": arithmetic,
        "decision": {
            "verdict": verdict,
            "gates": gates,
            "frozen_no_go_changed": False,
            "interpretation": config["decision_policy"][
                "pass"
                if verdict == "ALIGNMENT_TOLERANT_SHARED_LORA_PASS"
                else "promising"
                if verdict == "ALIGNMENT_TOLERANT_SHARED_LORA_PROMISING_ONLY"
                else "fail"
            ],
            "consequence": config["decision_policy"]["consequence"],
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "metrics.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    lines = [
        "# Alignment-tolerant shared low-rank residual screen v1",
        "",
        f"**Decision:** **{verdict}**",
        "",
        "| Candidate | Params | Compute | Hyp delta | UCB95 | OOD delta | UCB95 | Worst seed | KL hyp | Top-1 hyp |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name in names:
        row = candidates[name]
        lines.append(
            f"| {name} | {row['parameter_ratio']:.2%} | {row['compute_ratio']:.2%} | "
            f"{row['hypothesis']['mean']:+.5f} | {row['hypothesis']['ucb']:+.5f} | "
            f"{row['ood']['mean']:+.5f} | {row['ood']['ucb']:+.5f} | "
            f"{max(row['hypothesis']['per_seed'].values()):+.5f} | "
            f"{row['hypothesis']['kl_mean']:.5f} | {row['hypothesis']['top1_mean']:.3%} |"
        )
    lines += [
        "",
        f"- Rank-5 minus narrow65: mean `{primary_difference['mean']:+.5f}`, 95% `[{primary_difference['lcb']:+.5f}, {primary_difference['ucb']:+.5f}]`.",
        f"- Rank-6 minus narrow65: mean `{capacity_difference['mean']:+.5f}`, 95% `[{capacity_difference['lcb']:+.5f}, {capacity_difference['ucb']:+.5f}]`.",
        "",
        "## Gates",
        "",
    ]
    lines += [f"- `{name}`: `{value}`." for name, value in gates.items()]
    lines += [
        "",
        "## Scope",
        "",
        "The teachers and frozen baselines are inherited from the teacher-width replication; no plateau claim is made.",
        "The hypothesis and OOD documents are fresh and are materialized only after candidate freezing.",
        "Parameter and compute values are exact expert-only analytical proxies for this factorized execution; no runtime claim is made.",
        "The frozen NO_GO_FOR_OLMOE_OR_QWEN is unchanged.",
    ]
    (args.output_dir / "VERDICT.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print((args.output_dir / "VERDICT.md").read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
