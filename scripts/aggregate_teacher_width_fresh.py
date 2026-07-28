#!/usr/bin/env python3
"""Aggregate the fresh teacher-informed width replication."""
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


def selected(payloads, candidate: str, split: str) -> list[dict[str, Any]]:
    return [row for payload in payloads for row in payload["records"] if row["candidate"] == candidate and row["phase"] == "final" and row["evaluation_split"] == split]


def summarize(rows: list[dict[str, Any]], config: dict[str, Any], offset: int) -> dict[str, Any]:
    stats = config["statistics"]
    result = crossed_hierarchical_bootstrap(
        rows, value_key="loss_delta", seed_key="seed", document_key="document_id",
        samples=int(stats["bootstrap_samples"]),
        random_seed=int(stats["bootstrap_seed"]) + offset,
        confidence=float(stats["confidence"]),
    )
    result["per_seed"] = {
        str(seed): float(np.mean([float(row["loss_delta"]) for row in rows if int(row["seed"]) == seed]))
        for seed in sorted({int(row["seed"]) for row in rows})
    }
    result["kl_mean"] = float(np.mean([float(row["kl_teacher_to_candidate"]) for row in rows]))
    result["top1_mean"] = float(np.mean([float(row["top1_agreement"]) for row in rows]))
    result["local_nrmse_mean"] = float(np.mean([float(row["local_nrmse"]) for row in rows]))
    return result


def paired(payloads, left: str, right: str) -> list[dict[str, Any]]:
    source = [row for payload in payloads for row in payload["records"] if row["candidate"] in {left, right} and row["phase"] == "final" and row["evaluation_split"] == "hypothesis"]
    index = {(int(row["seed"]), str(row["document_id"]), int(row["start"]), str(row["candidate"])): float(row["loss_delta"]) for row in source}
    keys = sorted((seed, document, start) for seed, document, start, candidate in index if candidate == left)
    return [{"seed": seed, "document_id": document, "start": start, "difference": index[(seed, document, start, left)] - index[(seed, document, start, right)]} for seed, document, start in keys]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs/pre_qwen_teacher_width_fresh_v1.yaml")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/pre-qwen-teacher-width-fresh/v1")
    parser.add_argument("--audit-path", type=Path, default=None)
    args = parser.parse_args()
    config: dict[str, Any] = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    seeds = [int(v) for v in config["seeds"]]
    payloads = [json.loads((args.output_dir / f"seed-{seed}.json").read_text(encoding="utf-8")) for seed in seeds]
    names = [str(row["name"]) for row in config["candidates"]]
    candidates: dict[str, Any] = {}
    for index, name in enumerate(names):
        candidates[name] = {
            "parameter_ratio": float(np.mean([payload["metadata"]["parameter_ratios"][name] for payload in payloads])),
            "compute_ratio": float(np.mean([payload["metadata"]["compute_ratios"][name] for payload in payloads])),
            "hypothesis": summarize(selected(payloads, name, "hypothesis"), config, 100 + index),
            "ood": summarize(selected(payloads, name, "ood"), config, 200 + index),
        }
    pair_rows = paired(payloads, "magnitude-init-65", "magnitude-init-50")
    comparison = crossed_hierarchical_bootstrap(
        pair_rows, value_key="difference", seed_key="seed", document_key="document_id",
        samples=int(config["statistics"]["bootstrap_samples"]),
        random_seed=int(config["statistics"]["bootstrap_seed"]) + 500,
        confidence=float(config["statistics"]["confidence"]),
    )
    gates_cfg = config["gates"]
    primary = candidates["magnitude-init-65"]
    capacity = candidates["magnitude-init-75"]
    anchor = candidates["magnitude-init-35"]
    full = candidates["full-continuation-control"]
    tol = float(gates_cfg["arithmetic_tolerance"])
    clean = all(
        not payload["metadata"]["data_audit"]["exact_cross_split_duplicates"]
        and not payload["metadata"]["data_audit"]["near_cross_split_pairs"]
        for payload in payloads
    )
    audit_pass = False
    if args.audit_path and args.audit_path.exists():
        audit_pass = bool(json.loads(args.audit_path.read_text(encoding="utf-8"))["audit_passed"])
    gates = {
        "primary_hypothesis_pass": primary["hypothesis"]["ucb"] <= float(gates_cfg["primary_hypothesis_ucb_max"]),
        "primary_ood_pass": primary["ood"]["ucb"] <= float(gates_cfg["primary_ood_ucb_max"]),
        "every_seed_primary_pass": max(primary["hypothesis"]["per_seed"].values()) <= float(gates_cfg["every_seed_primary_hypothesis_delta_max"]),
        "primary_vs_comparator_pass": comparison["ucb"] <= float(gates_cfg["primary_minus_comparator_ucb_max"]),
        "capacity_pass": capacity["hypothesis"]["ucb"] <= float(gates_cfg["capacity_hypothesis_ucb_max"]),
        "anchor_failure_reproduced": anchor["hypothesis"]["lcb"] >= float(gates_cfg["anchor_hypothesis_lcb_min"]),
        "full_control_hypothesis_pass": full["hypothesis"]["ucb"] <= float(gates_cfg["full_control_hypothesis_ucb_max"]),
        "full_control_ood_pass": full["ood"]["ucb"] <= float(gates_cfg["full_control_ood_ucb_max"]),
        "exact_primary_parameter_ratio_pass": abs(primary["parameter_ratio"] - float(gates_cfg["exact_primary_parameter_ratio"])) <= tol,
        "exact_primary_compute_ratio_pass": abs(primary["compute_ratio"] - float(gates_cfg["exact_primary_compute_ratio"])) <= tol,
        "clean_data_audit_pass": clean,
        "independent_audit_pass": audit_pass,
    }
    load_bearing = (
        "primary_hypothesis_pass", "primary_ood_pass", "every_seed_primary_pass",
        "primary_vs_comparator_pass", "exact_primary_parameter_ratio_pass",
        "exact_primary_compute_ratio_pass", "clean_data_audit_pass", "independent_audit_pass",
    )
    if all(gates[name] for name in load_bearing) and gates["full_control_hypothesis_pass"] and gates["full_control_ood_pass"]:
        verdict = "TEACHER_WIDTH_65_REPLICATION_PASS"
    elif all(gates[name] for name in ("primary_hypothesis_pass", "primary_ood_pass", "every_seed_primary_pass", "exact_primary_parameter_ratio_pass", "exact_primary_compute_ratio_pass", "clean_data_audit_pass", "independent_audit_pass")):
        verdict = "TEACHER_WIDTH_65_REPLICATION_BORDERLINE"
    else:
        verdict = "TEACHER_WIDTH_65_REPLICATION_FAIL"
    payload = {
        "metadata": {
            "protocol_version": config["protocol_version"],
            "configuration_sha256": payloads[0]["metadata"]["configuration_sha256"],
            "source_commits": sorted({payload["metadata"]["source_commit"] for payload in payloads}),
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "seeds": seeds,
            "teacher_tail_changes": {str(payload["metadata"]["seed"]): payload["metadata"]["teacher_tail_change"] for payload in payloads},
        },
        "candidates": candidates,
        "comparisons": {"primary_minus_50": comparison},
        "decision": {
            "verdict": verdict,
            "gates": gates,
            "frozen_no_go_changed": False,
            "interpretation": config["decision_policy"]["consequence"],
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "metrics.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# Teacher-informed narrow width — fresh replication v1",
        "",
        f"**Decision:** **{verdict}**",
        "",
        "| Candidate | Params | Compute | Hyp Δ | UCB95 | OOD Δ | UCB95 | Worst seed | KL hyp | Top-1 hyp |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name in names:
        row = candidates[name]
        lines.append(
            f"| {name} | {row['parameter_ratio']:.1%} | {row['compute_ratio']:.1%} | "
            f"{row['hypothesis']['mean']:+.5f} | {row['hypothesis']['ucb']:+.5f} | "
            f"{row['ood']['mean']:+.5f} | {row['ood']['ucb']:+.5f} | "
            f"{max(row['hypothesis']['per_seed'].values()):+.5f} | "
            f"{row['hypothesis']['kl_mean']:.5f} | {row['hypothesis']['top1_mean']:.3%} |"
        )
    lines += [
        "",
        f"- Primary 65% minus 50%: mean `{comparison['mean']:+.5f}`, 95% `[{comparison['lcb']:+.5f}, {comparison['ucb']:+.5f}]`.",
        "",
        "## Gates",
        "",
    ]
    lines += [f"- `{name}`: `{value}`." for name, value in gates.items()]
    lines += [
        "",
        "No convergence claim is made for the teachers; tail changes are preserved in metrics.json.",
        "No runtime claim is made. Width ratios are exact routed matrix-operation and expert-parameter proxies.",
        "The frozen NO_GO_FOR_OLMOE_OR_QWEN is unchanged.",
    ]
    (args.output_dir / "VERDICT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print((args.output_dir / "VERDICT.md").read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
