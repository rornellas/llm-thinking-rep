#!/usr/bin/env python3
"""Aggregate Reality Gate 1A across scales, seeds, and documents."""
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

PRIMARY = "heterogeneous-spectral"
UNIFORM = "uniform-rank"
ROUTING = "heterogeneous-routing"
NARROW = "narrow65"
FULL = "full-identity-control"
METRICS = ("loss_delta", "kl_teacher_to_candidate", "top1_agreement", "local_nrmse")


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
            if str(payload["metadata"]["scale"]) != scale or int(payload["metadata"]["seed"]) != seed:
                raise ValueError(f"scale/seed mismatch in {path}")
            result.append(payload)
    return result


def rows_for(records: Iterable[dict[str, Any]], scale: str, candidate: str, split: str) -> list[dict[str, Any]]:
    return [
        row
        for row in records
        if str(row["scale"]) == scale
        and str(row["candidate"]) == candidate
        and str(row["phase"]) == "final"
        and str(row["evaluation_split"]) == split
    ]


def bootstrap(rows: list[dict[str, Any]], key: str, cfg: dict[str, Any], salt: int) -> dict[str, Any]:
    if not rows:
        raise ValueError(f"no rows for {key}")
    result = crossed_hierarchical_bootstrap(
        rows,
        value_key=key,
        samples=int(cfg["statistics"]["bootstrap_samples"]),
        random_seed=int(cfg["statistics"]["bootstrap_seed"]) + salt,
        confidence=float(cfg["statistics"]["confidence"]),
    )
    per_seed: dict[str, float] = {}
    for seed in sorted({int(row["seed"]) for row in rows}):
        values = [float(row[key]) for row in rows if int(row["seed"]) == seed]
        per_seed[str(seed)] = sum(values) / len(values)
    result["per_seed"] = per_seed
    return result


def paired_rows(left: list[dict[str, Any]], right: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    def cell(row: dict[str, Any]) -> tuple[int, str, int]:
        return int(row["seed"]), str(row["document_id"]), int(row["start"])

    index = {cell(row): row for row in right}
    if len(index) != len(right):
        raise ValueError("duplicate paired right cells")
    result: list[dict[str, Any]] = []
    for row in left:
        current = cell(row)
        if current not in index:
            raise ValueError(f"missing paired cell: {current}")
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


def paired(records: list[dict[str, Any]], scale: str, left: str, right: str, key: str, cfg: dict[str, Any], salt: int) -> dict[str, Any]:
    return bootstrap(
        paired_rows(
            rows_for(records, scale, left, "hypothesis"),
            rows_for(records, scale, right, "hypothesis"),
            key,
        ),
        "difference",
        cfg,
        salt,
    )


def candidate_stats(records: list[dict[str, Any]], scale: str, candidate: str, cfg: dict[str, Any], salt: int) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for split_index, split in enumerate(("hypothesis", "ood")):
        rows = rows_for(records, scale, candidate, split)
        result[split] = {
            key: bootstrap(rows, key, cfg, salt + split_index * 100 + index)
            for index, key in enumerate(METRICS)
        }
    return result


def write_verdict(path: Path, metrics: dict[str, Any]) -> None:
    lines = [
        "# Reality Gate 1A — compressibility trajectory and heterogeneous static rank",
        "",
        f"**Decision:** **{metrics['decision']['verdict']}**",
        "",
    ]
    for scale, scale_data in metrics["scales"].items():
        lines.extend(
            [
                f"## Scale `{scale}`",
                "",
                "| Candidate | Hyp delta | UCB95 | KL | Top-1 | Local NRMSE | Params | Train compute | Hyp compute |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for candidate in (PRIMARY, UNIFORM, ROUTING, NARROW, FULL):
            current = scale_data["candidates"][candidate]
            hyp = current["hypothesis"]
            accounting = current["accounting"]
            lines.append(
                "| {candidate} | {loss:+.5f} | {ucb:+.5f} | {kl:.5f} | {top:.2%} | {local:.5f} | {params:.2%} | {train_compute:.2%} | {hyp_compute:.2%} |".format(
                    candidate=candidate,
                    loss=float(hyp["loss_delta"]["mean"]),
                    ucb=float(hyp["loss_delta"]["ucb"]),
                    kl=float(hyp["kl_teacher_to_candidate"]["mean"]),
                    top=float(hyp["top1_agreement"]["mean"]),
                    local=float(hyp["local_nrmse"]["mean"]),
                    params=float(accounting["parameter_ratio_max"]),
                    train_compute=float(accounting["train_compute_ratio_max"]),
                    hyp_compute=float(accounting["hypothesis_compute_ratio_max"]),
                )
            )
        lines.extend(["", "### Comparisons", ""])
        for name, value in scale_data["comparisons"].items():
            lines.append(
                f"- `{name}`: `{value['mean']:+.6f}` 95% `[{value['lcb']:+.6f}, {value['ucb']:+.6f}]`."
            )
        lines.extend(["", "### Plateau", ""])
        for seed, value in scale_data["plateau"].items():
            lines.append(f"- `{seed}`: reached=`{value['plateau_reached']}`, final_step=`{value['final_step']}`.")
        lines.append("")
    lines.extend(["## Gates", ""])
    for name, value in metrics["decision"]["gates"].items():
        lines.append(f"- `{name}`: `{value}`.")
    lines.extend(
        [
            "",
            "No runtime claim is made. Compute ratios are analytical expected routed-matrix proxies.",
            "The frozen `NO_GO_FOR_OLMOE_OR_QWEN` remains unchanged.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs/reality_gate_1a.yaml")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/reality-gate-1a")
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
        raise ValueError("data manifest differs across cells")
    records = [row for payload in payloads for row in payload["records"]]

    output_scales: dict[str, Any] = {}
    gates_cfg = cfg["gates"]
    global_gates: dict[str, bool] = {}
    signal_scales = 0
    for scale_index, scale in enumerate(scales):
        current_payloads = [payload for payload in payloads if payload["metadata"]["scale"] == scale]
        candidates = {
            candidate: candidate_stats(records, scale, candidate, cfg, scale_index * 10000 + index * 1000)
            for index, candidate in enumerate((PRIMARY, UNIFORM, ROUTING, NARROW, FULL))
        }
        comparisons: dict[str, Any] = {}
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
        for index, (name, left, right, key) in enumerate(comparison_specs):
            comparisons[name] = paired(
                records, scale, left, right, key, cfg, scale_index * 10000 + 7000 + index
            )

        accounting: dict[str, Any] = {}
        for candidate in (PRIMARY, UNIFORM, ROUTING, NARROW, FULL):
            rows = [payload["metadata"]["train_accounting"][candidate] for payload in current_payloads]
            hypothesis = [
                payload["metadata"]["heldout_accounting"][candidate]["hypothesis"]
                for payload in current_payloads
            ]
            ood = [
                payload["metadata"]["heldout_accounting"][candidate]["ood"]
                for payload in current_payloads
            ]
            accounting[candidate] = {
                "parameter_ratio_max": max(float(row["parameter_ratio"]) for row in rows),
                "train_compute_ratio_max": max(float(row["compute_ratio"]) for row in rows),
                "hypothesis_compute_ratio_max": max(float(row["compute_ratio"]) for row in hypothesis),
                "ood_compute_ratio_max": max(float(row["compute_ratio"]) for row in ood),
            }
            candidates[candidate]["accounting"] = accounting[candidate]

        plateau = {
            str(payload["metadata"]["seed"]): payload["metadata"]["plateau"]
            for payload in current_payloads
        }
        clean = all(
            not any(payload["metadata"]["document_hash_overlap"].values())
            for payload in current_payloads
        )
        behavior_votes = sum(
            (
                comparisons["primary_minus_uniform_kl"]["ucb"] <= float(gates_cfg["uniform_kl_difference_ucb_max"]),
                comparisons["primary_minus_uniform_top1"]["lcb"] >= float(gates_cfg["uniform_top1_difference_lcb_min"]),
                comparisons["primary_minus_uniform_local"]["ucb"] <= float(gates_cfg["uniform_local_difference_ucb_max"]),
            )
        )
        primary_worst_seed = max(
            candidates[PRIMARY]["hypothesis"]["loss_delta"]["per_seed"].values()
        )
        scale_gates = {
            "all_teachers_plateaued": all(bool(value["plateau_reached"]) for value in plateau.values()),
            "clean_data": clean,
            "primary_parameter_budget": accounting[PRIMARY]["parameter_ratio_max"] <= accounting[UNIFORM]["parameter_ratio_max"] + float(gates_cfg["budget_tolerance"]),
            "primary_train_compute_budget": accounting[PRIMARY]["train_compute_ratio_max"] <= accounting[UNIFORM]["train_compute_ratio_max"] + float(gates_cfg["budget_tolerance"]),
            "primary_hypothesis_compute_budget": accounting[PRIMARY]["hypothesis_compute_ratio_max"] <= accounting[UNIFORM]["hypothesis_compute_ratio_max"] + float(gates_cfg["heldout_compute_tolerance"]),
            "primary_absolute_loss": candidates[PRIMARY]["hypothesis"]["loss_delta"]["ucb"] <= float(gates_cfg["absolute_loss_ucb_max"]),
            "primary_absolute_kl": candidates[PRIMARY]["hypothesis"]["kl_teacher_to_candidate"]["ucb"] <= float(gates_cfg["absolute_kl_ucb_max"]),
            "primary_absolute_top1": candidates[PRIMARY]["hypothesis"]["top1_agreement"]["lcb"] >= float(gates_cfg["absolute_top1_lcb_min"]),
            "primary_absolute_local": candidates[PRIMARY]["hypothesis"]["local_nrmse"]["ucb"] <= float(gates_cfg["absolute_local_nrmse_ucb_max"]),
            "primary_ood_loss": candidates[PRIMARY]["ood"]["loss_delta"]["ucb"] <= float(gates_cfg["ood_loss_ucb_max"]),
            "primary_ood_kl": candidates[PRIMARY]["ood"]["kl_teacher_to_candidate"]["ucb"] <= float(gates_cfg["ood_kl_ucb_max"]),
            "primary_every_seed_loss": primary_worst_seed <= float(gates_cfg["every_seed_loss_delta_max"]),
            "primary_vs_uniform_loss": comparisons["primary_minus_uniform_loss"]["ucb"] <= float(gates_cfg["uniform_loss_difference_ucb_max"]),
            "primary_vs_uniform_behavior": behavior_votes >= int(gates_cfg["uniform_behavior_min_votes"]),
            "primary_vs_routing_loss": comparisons["primary_minus_routing_loss"]["ucb"] <= float(gates_cfg["routing_loss_difference_ucb_max"]),
            "primary_vs_narrow_loss": comparisons["primary_minus_narrow_loss"]["ucb"] <= float(gates_cfg["narrow_loss_difference_ucb_max"]),
            "primary_vs_narrow_kl": comparisons["primary_minus_narrow_kl"]["ucb"] <= float(gates_cfg["narrow_kl_difference_ucb_max"]),
            "primary_vs_narrow_top1": comparisons["primary_minus_narrow_top1"]["lcb"] >= float(gates_cfg["narrow_top1_difference_lcb_min"]),
            "primary_vs_narrow_local": comparisons["primary_minus_narrow_local"]["ucb"] <= float(gates_cfg["narrow_local_difference_ucb_max"]),
            "full_identity": max(
                abs(float(candidates[FULL]["hypothesis"]["loss_delta"]["mean"])),
                abs(float(candidates[FULL]["ood"]["loss_delta"]["mean"])),
                float(candidates[FULL]["hypothesis"]["kl_teacher_to_candidate"]["ucb"]),
                float(candidates[FULL]["ood"]["kl_teacher_to_candidate"]["ucb"]),
            ) <= float(gates_cfg["identity_absolute_tolerance"]),
        }
        if (
            scale_gates["all_teachers_plateaued"]
            and scale_gates["clean_data"]
            and scale_gates["primary_parameter_budget"]
            and scale_gates["primary_train_compute_budget"]
            and comparisons["primary_minus_uniform_loss"]["ucb"] <= float(gates_cfg["signal_uniform_loss_ucb_max"])
            and behavior_votes >= int(gates_cfg["signal_behavior_min_votes"])
        ):
            signal_scales += 1
        output_scales[scale] = {
            "candidates": candidates,
            "comparisons": comparisons,
            "accounting": accounting,
            "plateau": plateau,
            "behavior_votes_vs_uniform": behavior_votes,
            "gates": scale_gates,
            "trajectory": {
                str(payload["metadata"]["seed"]): payload["trajectory"]
                for payload in current_payloads
            },
        }
        for gate_name, value in scale_gates.items():
            global_gates[f"{scale}__{gate_name}"] = bool(value)

    small, medium = scales[0], scales[-1]
    scale_trend = (
        output_scales[medium]["comparisons"]["primary_minus_uniform_loss"]["mean"]
        <= output_scales[small]["comparisons"]["primary_minus_uniform_loss"]["mean"]
        + float(gates_cfg["maximum_scale_trend_regression"])
    )
    global_gates["scale_trend"] = bool(scale_trend)
    audit_pass = False
    audit_payload = None
    if args.audit_path is not None and args.audit_path.exists():
        audit_payload = json.loads(args.audit_path.read_text(encoding="utf-8"))
        audit_pass = bool(audit_payload.get("audit_passed")) and not audit_payload.get("mismatches")
    global_gates["independent_audit"] = audit_pass

    # Every preregistered gate, including the routing-only mechanism control,
    # is load-bearing for PASS. No comparison is silently advisory.
    pass_required = list(global_gates)
    if all(global_gates[name] for name in pass_required):
        verdict = "REALITY_GATE_1A_PASS"
    elif audit_pass and signal_scales >= int(gates_cfg["signal_min_scales"]):
        verdict = "REALITY_GATE_1A_HETEROGENEOUS_RANK_SIGNAL"
    else:
        verdict = "REALITY_GATE_1A_FAIL"

    output = {
        "metadata": {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "protocol_version": protocol,
            "configuration_sha256": config_hash,
            "data_manifest_sha256": sorted({str(payload["metadata"]["data_manifest_sha256"]) for payload in payloads})[0],
            "source_commits": sorted({str(payload["metadata"]["source_commit"]) for payload in payloads}),
            "scales": scales,
            "seeds": seeds,
        },
        "scales": output_scales,
        "decision": {
            "verdict": verdict,
            "gates": global_gates,
            "signal_scales": signal_scales,
            "frozen_no_go_changed": False,
        },
    }
    if audit_pass and audit_payload is not None:
        expected = audit_payload.get("expected_final_verdict_if_audit_passes")
        if expected is not None and str(expected) != verdict:
            raise RuntimeError(f"auditor expected {expected}, aggregator produced {verdict}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "metrics.json").write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    write_verdict(args.output_dir / "VERDICT.md", output)
    print(verdict)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
