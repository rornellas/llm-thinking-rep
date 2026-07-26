#!/usr/bin/env python3
"""Replicate modal-MoE trainability with seeds and capacity controls.

Controls:
- modal K=0: one shared SwiGLU expert; routing cannot create specialization;
- narrow conventional MoEs with d_ff 32/64/96, matching 25/50/75% of the
  full baseline's expert arithmetic, respectively;
- modal K=1/K=2 at 50/75% ideal expert arithmetic.

The comparison separates a genuine modal parameter-efficiency signal from a
task that simply does not need multiple experts.
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
from dataclasses import asdict
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Sequence


def load_source(path: Path):
    spec = importlib.util.spec_from_file_location("modal_trainability_source", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def one_run(source, dataset, seed: int, steps: int, variant: str, d_ff: int, rank: int | None):
    cfg = source.Config(
        d_ff=d_ff,
        steps=steps,
        eval_batches=10,
        eval_interval=100,
        batch_size=16,
        seq_len=64,
    )
    raw_variant = "modal" if rank is not None else "baseline"
    result, history = source.train_variant(raw_variant, rank, seed, dataset, cfg)
    if rank is None:
        label = f"baseline-dff{d_ff}"
        result.expert_parameter_ratio = d_ff / 128.0
        result.idealized_expert_compute_ratio = d_ff / 128.0
    else:
        label = f"modal-k{rank}"
    result.variant = label
    return result, history


def aggregate(results: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    seeds = sorted({int(row["seed"]) for row in results})
    by_seed = {(int(row["seed"]), row["variant"]): row for row in results}
    variants = sorted({row["variant"] for row in results})
    summary: list[dict[str, Any]] = []
    for variant in variants:
        rows = [by_seed[(seed, variant)] for seed in seeds]
        ratios = [row["final_validation_loss"] / by_seed[(seed, "baseline-dff128")]["final_validation_loss"] for seed, row in zip(seeds, rows, strict=True)]
        summary.append({
            "variant": variant,
            "runs": len(rows),
            "validation_loss_mean": mean(row["final_validation_loss"] for row in rows),
            "validation_loss_std": pstdev(row["final_validation_loss"] for row in rows),
            "loss_ratio_to_full_mean": mean(ratios),
            "loss_ratio_to_full_std": pstdev(ratios),
            "loss_ratio_to_full_max": max(ratios),
            "trainable_parameters_mean": mean(row["trainable_parameters"] for row in rows),
            "expert_parameter_ratio": rows[0]["expert_parameter_ratio"],
            "idealized_expert_compute_ratio": rows[0]["idealized_expert_compute_ratio"],
            "utilization_entropy_mean": mean(row["utilization_entropy"] for row in rows),
        })
    indexed = {row["variant"]: row for row in summary}
    k0 = indexed["modal-k0"]["loss_ratio_to_full_mean"]
    k1 = indexed["modal-k1"]["loss_ratio_to_full_mean"]
    k2 = indexed["modal-k2"]["loss_ratio_to_full_mean"]
    narrow32 = indexed["baseline-dff32"]["loss_ratio_to_full_mean"]
    narrow64 = indexed["baseline-dff64"]["loss_ratio_to_full_mean"]
    narrow96 = indexed["baseline-dff96"]["loss_ratio_to_full_mean"]
    k1_advantage = narrow64 - k1
    k2_advantage = narrow96 - k2
    specialization_identified = k0 > 1.03
    robust_modal = (
        k1 <= 1.04 and indexed["modal-k1"]["loss_ratio_to_full_max"] <= 1.06
        and k2 <= 1.03 and indexed["modal-k2"]["loss_ratio_to_full_max"] <= 1.05
        and k1_advantage >= 0.005 and k2_advantage >= 0.003
    )
    if robust_modal and specialization_identified:
        verdict = "ROBUST_SIGNAL"
    elif robust_modal:
        verdict = "MODAL_EFFICIENT_BUT_TASK_UNDERSPECIFIED"
    elif k1 <= narrow64 and k2 <= narrow96:
        verdict = "BORDERLINE"
    else:
        verdict = "FAIL"
    decision = {
        "verdict": verdict,
        "modal_k0_loss_ratio": k0,
        "modal_k1_loss_ratio": k1,
        "modal_k2_loss_ratio": k2,
        "narrow_dff32_loss_ratio": narrow32,
        "narrow_dff64_loss_ratio": narrow64,
        "narrow_dff96_loss_ratio": narrow96,
        "modal_k1_advantage_over_compute_matched_narrow": k1_advantage,
        "modal_k2_advantage_over_compute_matched_narrow": k2_advantage,
        "specialization_identified": specialization_identified,
        "rule": "A robust signal requires modal K1/K2 to generalize across all seeds and outperform conventional narrow MoEs at matched ideal expert compute. K0 >3% above baseline is required to show that the task actually benefits from expert-specific codes.",
    }
    return summary, decision


def write_outputs(output_dir: Path, payload: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "modal_robustness.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with (output_dir / "runs.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(payload["runs"][0].keys()))
        writer.writeheader(); writer.writerows(payload["runs"])
    with (output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(payload["summary"][0].keys()))
        writer.writeheader(); writer.writerows(payload["summary"])
    d = payload["decision"]
    lines = [
        "# Test 2.1 — multi-seed modal MoE robustness and capacity controls", "", f"**Decision:** **{d['verdict']}**", "",
        "| Variant | Runs | Expert params | Ideal expert compute | Validation loss | Loss/full | Worst seed | Router entropy |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["summary"]:
        lines.append(
            f"| {row['variant']} | {row['runs']} | {row['expert_parameter_ratio']:.2%} | "
            f"{row['idealized_expert_compute_ratio']:.2%} | {row['validation_loss_mean']:.4f} ± {row['validation_loss_std']:.4f} | "
            f"{row['loss_ratio_to_full_mean']:.3f}× | {row['loss_ratio_to_full_max']:.3f}× | {row['utilization_entropy_mean']:.3f} |"
        )
    lines += [
        "",
        f"- K1 advantage over compute-matched d_ff=64 baseline: `{d['modal_k1_advantage_over_compute_matched_narrow']:+.3%}` loss ratio.",
        f"- K2 advantage over compute-matched d_ff=96 baseline: `{d['modal_k2_advantage_over_compute_matched_narrow']:+.3%}` loss ratio.",
        f"- Specialization identified by K0 control: `{d['specialization_identified']}`.",
        "",
        "This remains a small character-language-model experiment. A positive result authorizes a larger token-level and specialization-forcing test; it does not establish billion-parameter scaling.",
    ]
    (output_dir / "VERDICT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--text", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("results/modal_robustness"))
    parser.add_argument("--seeds", default="1337,2027,31415")
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args(argv)
    source = load_source(args.source)
    source.torch.set_num_threads(args.threads)
    source.torch.set_num_interop_threads(1)
    dataset = source.CharDataset(args.text.read_text(encoding="utf-8"), 64)
    seeds = [int(value) for value in args.seeds.split(",") if value.strip()]
    variants = [
        ("baseline", 128, None),
        ("baseline", 32, None),
        ("baseline", 64, None),
        ("baseline", 96, None),
        ("modal", 128, 0),
        ("modal", 128, 1),
        ("modal", 128, 2),
    ]
    run_rows: list[dict[str, Any]] = []
    histories: dict[str, list[dict[str, float]]] = {}
    for seed in seeds:
        for _, d_ff, rank in variants:
            result, history = one_run(source, dataset, seed, args.steps, "modal" if rank is not None else "baseline", d_ff, rank)
            row = asdict(result)
            run_rows.append(row)
            histories[f"{result.variant}-seed{seed}"] = history
    summary, decision = aggregate(run_rows)
    payload = {
        "metadata": {"seeds": seeds, "steps": args.steps, "task": "Tiny Shakespeare character LM", "controls": ["modal K0", "compute-matched narrow conventional MoEs"]},
        "decision": decision,
        "summary": summary,
        "runs": run_rows,
        "history": histories,
    }
    write_outputs(args.output_dir, payload)
    print((args.output_dir / "VERDICT.md").read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
