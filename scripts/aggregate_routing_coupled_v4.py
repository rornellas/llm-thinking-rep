#!/usr/bin/env python3
"""Aggregate the preregistered routing-coupled residual v4 experiment."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Iterable

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pre_qwen_certification.metrics import crossed_hierarchical_bootstrap

PRIMARY = "rank5-coupled-q8-h8-v4"
MEAN_ONLY = "rank5-coupled-q8-h8-mean-only-control"
CAPACITY = "rank5-coupled-q12-h8-v4"
V3 = "rank5-v3-frozen-baseline"
RANK6 = "rank6-v3-frozen-capacity"
NARROW = "narrow65-frozen-baseline"
FULL = "full-continuation-control"
DISABLED = "rank5-coupled-q8-h8-v4__coupling-disabled"
SECOND_DISABLED = "rank5-coupled-q8-h8-v4__second-moment-disabled"

METRICS = (
    "loss_delta",
    "kl_teacher_to_candidate",
    "top1_agreement",
    "local_nrmse",
    "counterfactual_nrmse",
    "routing_cross_error",
    "correction_energy_ratio",
)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_payloads(output_dir: Path, seeds: list[int]) -> list[dict[str, Any]]:
    payloads: list[dict[str, Any]] = []
    for seed in seeds:
        path = output_dir / f"seed-{seed}.json"
        if not path.exists():
            raise FileNotFoundError(path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        if int(payload["metadata"]["seed"]) != seed:
            raise ValueError(f"seed mismatch in {path}")
        payloads.append(payload)
    return payloads


def final_rows(
    records: Iterable[dict[str, Any]], candidate: str, split: str
) -> list[dict[str, Any]]:
    return [
        row
        for row in records
        if str(row["candidate"]) == candidate
        and str(row["phase"]) == "final"
        and str(row["evaluation_split"]) == split
    ]


def bootstrap(
    rows: list[dict[str, Any]],
    value_key: str,
    cfg: dict[str, Any],
    salt: int,
) -> dict[str, Any]:
    if not rows:
        raise ValueError(f"no rows for {value_key}")
    result = crossed_hierarchical_bootstrap(
        rows,
        value_key=value_key,
        samples=int(cfg["statistics"]["bootstrap_samples"]),
        random_seed=int(cfg["statistics"]["bootstrap_seed"]) + salt,
        confidence=float(cfg["statistics"]["confidence"]),
    )
    per_seed: dict[str, float] = {}
    for seed in sorted({int(row["seed"]) for row in rows}):
        current = [float(row[value_key]) for row in rows if int(row["seed"]) == seed]
        per_seed[str(seed)] = sum(current) / len(current)
    result["per_seed"] = per_seed
    return result


def paired_rows(
    left: list[dict[str, Any]],
    right: list[dict[str, Any]],
    value_key: str,
    name: str,
) -> list[dict[str, Any]]:
    def key(row: dict[str, Any]) -> tuple[int, str, int]:
        return int(row["seed"]), str(row["document_id"]), int(row["start"])

    right_index = {key(row): row for row in right}
    if len(right_index) != len(right):
        raise ValueError(f"duplicate right cells for {name}")
    result: list[dict[str, Any]] = []
    for row in left:
        current_key = key(row)
        if current_key not in right_index:
            raise ValueError(f"unpaired cell {current_key} for {name}")
        result.append(
            {
                "seed": current_key[0],
                "document_id": current_key[1],
                "start": current_key[2],
                name: float(row[value_key]) - float(right_index[current_key][value_key]),
            }
        )
    if len(result) != len(right):
        raise ValueError(f"unequal paired row counts for {name}")
    return result


def paired_bootstrap(
    records: list[dict[str, Any]],
    left: str,
    right: str,
    split: str,
    value_key: str,
    cfg: dict[str, Any],
    salt: int,
) -> dict[str, Any]:
    name = "difference"
    rows = paired_rows(
        final_rows(records, left, split),
        final_rows(records, right, split),
        value_key,
        name,
    )
    return bootstrap(rows, name, cfg, salt)


def candidate_statistics(
    records: list[dict[str, Any]],
    candidate: str,
    cfg: dict[str, Any],
    salt: int,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for split_index, split in enumerate(("hypothesis", "ood")):
        rows = final_rows(records, candidate, split)
        result[split] = {
            metric: bootstrap(rows, metric, cfg, salt + split_index * 100 + index)
            for index, metric in enumerate(METRICS)
        }
    return result


def expected_ratios(cfg: dict[str, Any]) -> dict[str, dict[str, float]]:
    return {
        str(row["name"]): {
            "parameter_ratio": float(row["expected_parameter_ratio"]),
            "compute_ratio": float(row["expected_compute_ratio"]),
        }
        for row in cfg["candidates"]
    }


def observed_ratios(payloads: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    for payload in payloads:
        parameters = payload["metadata"]["parameter_ratios"]
        compute = payload["metadata"]["compute_ratios"]
        for name in parameters:
            current = {
                "parameter_ratio": float(parameters[name]),
                "compute_ratio": float(compute[name]),
            }
            if name in result and any(
                abs(result[name][key] - current[key]) > 1e-12 for key in current
            ):
                raise ValueError(f"ratio differs across seeds for {name}")
            result[name] = current
    return result


def clean_data(payloads: list[dict[str, Any]]) -> bool:
    for payload in payloads:
        audit = payload["metadata"]["data_audit"]
        if audit["exact_cross_split_duplicates"] or audit["near_cross_split_pairs"]:
            return False
        # Any document-level duplicate inside the full train/hypothesis/OOD set
        # reduces effective diversity and is disqualifying for this protocol.
        if audit.get("within_all_findings"):
            return False
    return True


def verdict_from_gates(gates: dict[str, bool | int], improvement_votes: int) -> str:
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
        "independent_audit_pass",
    )
    if all(bool(gates[key]) for key in pass_keys):
        return "ROUTING_COUPLED_V4_PASS"
    signal_integrity = (
        bool(gates["primary_parameter_budget_pass"])
        and bool(gates["primary_compute_budget_pass"])
        and bool(gates["all_arithmetic_pass"])
        and bool(gates["clean_data_audit_pass"])
        and bool(gates["full_control_hypothesis_pass"])
        and bool(gates["full_control_ood_pass"])
        and bool(gates["independent_audit_pass"])
        and bool(gates["causal_coupling_pass"])
    )
    if signal_integrity and improvement_votes >= int(
        gates.get("required_improvement_votes", 2)
    ):
        return "ROUTING_COUPLED_V4_FUNCTIONAL_SIGNAL"
    return "ROUTING_COUPLED_V4_FAIL"


def write_verdict(path: Path, metrics: dict[str, Any]) -> None:
    candidates = metrics["candidates"]
    order = [PRIMARY, MEAN_ONLY, CAPACITY, V3, RANK6, NARROW, FULL]
    lines = [
        "# Routing-coupled residual v4",
        "",
        f"**Decision:** **{metrics['decision']['verdict']}**",
        "",
        "| Candidate | Params | Compute | Hyp delta | UCB95 | KL | Top-1 | Local NRMSE | CF NRMSE | Cross error | Correction energy |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name in order:
        current = candidates[name]
        hyp = current["hypothesis"]
        lines.append(
            "| {name} | {p:.2%} | {c:.2%} | {loss:+.5f} | {loss_u:+.5f} | "
            "{kl:.5f} | {top:.2%} | {local:.5f} | {cf:.5f} | {cross:+.5f} | {energy:.5f} |".format(
                name=name,
                p=float(current["parameter_ratio"]),
                c=float(current["compute_ratio"]),
                loss=float(hyp["loss_delta"]["mean"]),
                loss_u=float(hyp["loss_delta"]["ucb"]),
                kl=float(hyp["kl_teacher_to_candidate"]["mean"]),
                top=float(hyp["top1_agreement"]["mean"]),
                local=float(hyp["local_nrmse"]["mean"]),
                cf=float(hyp["counterfactual_nrmse"]["mean"]),
                cross=float(hyp["routing_cross_error"]["mean"]),
                energy=float(hyp["correction_energy_ratio"]["mean"]),
            )
        )
    lines.extend(["", "## Load-bearing comparisons", ""])
    for name, current in metrics["comparisons"].items():
        lines.append(
            f"- `{name}`: mean `{current['mean']:+.6f}`, 95% "
            f"`[{current['lcb']:+.6f}, {current['ucb']:+.6f}]`."
        )
    lines.extend(["", "## Gates", ""])
    for name, value in metrics["decision"]["gates"].items():
        lines.append(f"- `{name}`: `{value}`.")
    lines.extend(
        [
            "",
            f"Behavior-improvement votes versus frozen v3: `{metrics['decision']['improvement_votes']}`.",
            "",
            "The teachers are inherited fixed checkpoints; no plateau claim is made.",
            "No runtime claim is made. Ratios are expert-only analytical proxies.",
            "The frozen `NO_GO_FOR_OLMOE_OR_QWEN` is unchanged.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/pre_qwen_routing_coupled_v4.yaml",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "results/pre-qwen-routing-coupled/v4",
    )
    parser.add_argument("--audit-path", type=Path, default=None)
    args = parser.parse_args()

    cfg: dict[str, Any] = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    seeds = [int(seed) for seed in cfg["seeds"]]
    payloads = load_payloads(args.output_dir, seeds)
    protocol = str(cfg["protocol_version"])
    config_hash = sha256_file(args.config)
    if {str(payload["metadata"]["protocol_version"]) for payload in payloads} != {protocol}:
        raise ValueError("protocol version mismatch")
    if {str(payload["metadata"]["configuration_sha256"]) for payload in payloads} != {config_hash}:
        raise ValueError("configuration hash mismatch")

    records = [row for payload in payloads for row in payload["records"]]
    configured_order = [str(row["name"]) for row in cfg["candidates"]]
    all_candidates = configured_order + [DISABLED, SECOND_DISABLED]
    statistics = {
        name: candidate_statistics(records, name, cfg, 1000 * index)
        for index, name in enumerate(all_candidates)
    }

    observed = observed_ratios(payloads)
    expected = expected_ratios(cfg)
    tolerance = float(cfg["gates"]["arithmetic_tolerance"])
    arithmetic = {
        name: (
            name in observed
            and abs(observed[name]["parameter_ratio"] - values["parameter_ratio"]) <= tolerance
            and abs(observed[name]["compute_ratio"] - values["compute_ratio"]) <= tolerance
        )
        for name, values in expected.items()
    }
    for name in configured_order:
        statistics[name]["parameter_ratio"] = observed[name]["parameter_ratio"]
        statistics[name]["compute_ratio"] = observed[name]["compute_ratio"]
    for name in (DISABLED, SECOND_DISABLED):
        statistics[name]["parameter_ratio"] = observed[PRIMARY]["parameter_ratio"]
        statistics[name]["compute_ratio"] = observed[PRIMARY]["compute_ratio"]

    comparisons = {
        "primary_minus_narrow_loss": paired_bootstrap(records, PRIMARY, NARROW, "hypothesis", "loss_delta", cfg, 9101),
        "primary_minus_rank6_loss": paired_bootstrap(records, PRIMARY, RANK6, "hypothesis", "loss_delta", cfg, 9102),
        "primary_minus_v3_loss": paired_bootstrap(records, PRIMARY, V3, "hypothesis", "loss_delta", cfg, 9103),
        "primary_minus_v3_kl": paired_bootstrap(records, PRIMARY, V3, "hypothesis", "kl_teacher_to_candidate", cfg, 9104),
        "primary_minus_v3_top1": paired_bootstrap(records, PRIMARY, V3, "hypothesis", "top1_agreement", cfg, 9105),
        "primary_minus_v3_local": paired_bootstrap(records, PRIMARY, V3, "hypothesis", "local_nrmse", cfg, 9106),
        "primary_minus_v3_counterfactual": paired_bootstrap(records, PRIMARY, V3, "hypothesis", "counterfactual_nrmse", cfg, 9107),
        "primary_minus_mean_only_kl": paired_bootstrap(records, PRIMARY, MEAN_ONLY, "hypothesis", "kl_teacher_to_candidate", cfg, 9108),
        "primary_minus_mean_only_loss": paired_bootstrap(records, PRIMARY, MEAN_ONLY, "hypothesis", "loss_delta", cfg, 9109),
        "disabled_minus_primary_kl": paired_bootstrap(records, DISABLED, PRIMARY, "hypothesis", "kl_teacher_to_candidate", cfg, 9110),
        "disabled_minus_primary_loss": paired_bootstrap(records, DISABLED, PRIMARY, "hypothesis", "loss_delta", cfg, 9111),
        "second_disabled_minus_primary_kl": paired_bootstrap(records, SECOND_DISABLED, PRIMARY, "hypothesis", "kl_teacher_to_candidate", cfg, 9112),
        "primary_minus_narrow_cross_error": paired_bootstrap(records, PRIMARY, NARROW, "hypothesis", "routing_cross_error", cfg, 9113),
    }

    audit_payload: dict[str, Any] | None = None
    audit_pass = False
    if args.audit_path is not None and args.audit_path.exists():
        audit_payload = json.loads(args.audit_path.read_text(encoding="utf-8"))
        audit_pass = bool(audit_payload.get("audit_passed")) and not audit_payload.get("mismatches")

    gates_cfg = cfg["gates"]
    primary_hyp = statistics[PRIMARY]["hypothesis"]
    primary_ood = statistics[PRIMARY]["ood"]
    full_hyp = statistics[FULL]["hypothesis"]
    full_ood = statistics[FULL]["ood"]
    every_seed = max(primary_hyp["loss_delta"]["per_seed"].values())
    cross_gap = comparisons["primary_minus_narrow_cross_error"]

    improvement_checks = {
        "loss": comparisons["primary_minus_v3_loss"]["ucb"] <= float(gates_cfg["primary_minus_v3_loss_ucb_max"]),
        "kl": comparisons["primary_minus_v3_kl"]["ucb"] <= float(gates_cfg["primary_minus_v3_kl_ucb_max"]),
        "top1": comparisons["primary_minus_v3_top1"]["lcb"] >= float(gates_cfg["primary_minus_v3_top1_lcb_min"]),
        "local": comparisons["primary_minus_v3_local"]["ucb"] <= float(gates_cfg["primary_minus_v3_local_nrmse_ucb_max"]),
        "counterfactual": comparisons["primary_minus_v3_counterfactual"]["ucb"] <= float(gates_cfg["primary_minus_v3_counterfactual_nrmse_ucb_max"]),
    }
    improvement_votes = sum(improvement_checks.values())
    causal_kl = comparisons["disabled_minus_primary_kl"]["lcb"] >= float(
        gates_cfg["disabled_minus_primary_kl_lcb_min"]
    )
    causal_loss = comparisons["disabled_minus_primary_loss"]["lcb"] >= float(
        gates_cfg["disabled_minus_primary_loss_lcb_min"]
    )

    gates: dict[str, bool | int] = {
        "primary_hypothesis_pass": primary_hyp["loss_delta"]["ucb"] <= float(gates_cfg["primary_hypothesis_ucb_max"]),
        "primary_ood_pass": primary_ood["loss_delta"]["ucb"] <= float(gates_cfg["primary_ood_ucb_max"]),
        "every_seed_primary_pass": every_seed <= float(gates_cfg["every_seed_primary_hypothesis_delta_max"]),
        "primary_vs_narrow65_pass": comparisons["primary_minus_narrow_loss"]["ucb"] <= float(gates_cfg["primary_minus_narrow65_ucb_max"]),
        "primary_vs_rank6_pass": comparisons["primary_minus_rank6_loss"]["ucb"] <= float(gates_cfg["primary_minus_rank6_ucb_max"]),
        "primary_vs_v3_pass": comparisons["primary_minus_v3_loss"]["ucb"] <= float(gates_cfg["primary_minus_v3_ucb_max"]),
        "primary_vs_mean_only_kl_pass": comparisons["primary_minus_mean_only_kl"]["ucb"] <= float(gates_cfg["primary_minus_mean_only_kl_ucb_max"]),
        "primary_kl_pass": primary_hyp["kl_teacher_to_candidate"]["ucb"] <= float(gates_cfg["primary_kl_ucb_max"]),
        "primary_top1_pass": primary_hyp["top1_agreement"]["lcb"] >= float(gates_cfg["primary_top1_lcb_min"]),
        "primary_local_nrmse_pass": primary_hyp["local_nrmse"]["ucb"] <= float(gates_cfg["primary_local_nrmse_ucb_max"]),
        "primary_counterfactual_pass": primary_hyp["counterfactual_nrmse"]["ucb"] <= float(gates_cfg["primary_counterfactual_nrmse_ucb_max"]),
        "primary_cross_error_gap_pass": cross_gap["ucb"] <= float(gates_cfg["primary_cross_error_gap_vs_narrow_ucb_max"]),
        "causal_coupling_kl_pass": causal_kl,
        "causal_coupling_loss_pass": causal_loss,
        "causal_coupling_pass": causal_kl or causal_loss,
        "full_control_hypothesis_pass": full_hyp["loss_delta"]["ucb"] <= float(gates_cfg["full_control_hypothesis_ucb_max"]),
        "full_control_ood_pass": full_ood["loss_delta"]["ucb"] <= float(gates_cfg["full_control_ood_ucb_max"]),
        "primary_parameter_budget_pass": observed[PRIMARY]["parameter_ratio"] < float(gates_cfg["primary_parameter_ratio_strict_max"]),
        "primary_compute_budget_pass": observed[PRIMARY]["compute_ratio"] <= float(gates_cfg["primary_compute_ratio_max"]),
        "all_arithmetic_pass": all(arithmetic.values()),
        "clean_data_audit_pass": clean_data(payloads),
        "independent_audit_pass": audit_pass,
        "required_improvement_votes": int(gates_cfg["behavior_improvement_min_votes"]),
    }
    verdict = verdict_from_gates(gates, improvement_votes)
    if audit_pass and audit_payload is not None:
        expected_verdict = audit_payload.get("expected_final_verdict_if_audit_passes")
        if expected_verdict is not None and str(expected_verdict) != verdict:
            raise RuntimeError(
                f"independent audit expected {expected_verdict}, aggregator produced {verdict}"
            )

    output: dict[str, Any] = {
        "metadata": {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "protocol_version": protocol,
            "configuration_sha256": config_hash,
            "seeds": seeds,
            "source_commits": sorted({str(payload["metadata"]["source_commit"]) for payload in payloads}),
        },
        "candidates": statistics,
        "comparisons": comparisons,
        "arithmetic": arithmetic,
        "improvement_checks": improvement_checks,
        "decision": {
            "verdict": verdict,
            "gates": gates,
            "improvement_votes": improvement_votes,
            "frozen_no_go_changed": False,
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "metrics.json").write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    write_verdict(args.output_dir / "VERDICT.md", output)
    print(verdict)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
