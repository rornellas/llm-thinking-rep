#!/usr/bin/env python3
"""Replicate the extended marginal-utility controller across fresh seeds.

Each seed is executed in a fresh Python process so PyTorch thread pools and all
module-level state are independent. Lambda is selected separately for each seed
using train-split calibration only. The aggregate verdict requires held-out
quality and rank-compacted compute savings to survive every seed.
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Sequence


STATIC_K1_COMPUTE = 0.25048828125


def run_seed(args: argparse.Namespace, seed: int, output_dir: Path) -> dict[str, Any]:
    seed_dir = output_dir / f"seed-{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(args.runner_source),
        "--source",
        str(args.source),
        "--progressive-source",
        str(args.progressive_source),
        "--controller-source",
        str(args.controller_source),
        "--text",
        str(args.text),
        "--output-dir",
        str(seed_dir),
        "--steps",
        str(args.steps),
        "--seed",
        str(seed),
        "--threads",
        str(args.threads),
        "--train-batches",
        str(args.train_batches),
        "--tune-batches",
        str(args.tune_batches),
        "--test-batches",
        str(args.test_batches),
        "--controller-steps",
        str(args.controller_steps),
        "--controller-hidden",
        str(args.controller_hidden),
    ]
    log_path = seed_dir / "run.log"
    print("running:", " ".join(command), flush=True)
    with log_path.open("w", encoding="utf-8") as log:
        completed = subprocess.run(
            command,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    if completed.returncode != 0:
        tail = "\n".join(log_path.read_text(encoding="utf-8").splitlines()[-80:])
        raise RuntimeError(f"seed {seed} failed with code {completed.returncode}\n{tail}")
    payload = json.loads((seed_dir / "marginal_utility.json").read_text(encoding="utf-8"))
    result = payload["test_result"]
    return {
        "seed": seed,
        "lambda_nats": float(result["lambda_nats"]),
        "dynamic_loss": float(result["dynamic_loss"]),
        "static_k1_loss": float(result["static_k1_loss"]),
        "dynamic_delta_vs_k1": float(result["dynamic_delta_vs_k1"]),
        "dynamic_delta_ucb95": float(result["dynamic_delta_ucb95"]),
        "mean_rank": float(result["mean_rank"]),
        "p95_rank": float(result["p95_rank"]),
        "exact_compute_ratio": float(result["exact_compute_ratio"]),
        "bucket16_compute_ratio": float(result["bucket16_compute_ratio"]),
        "bucket32_compute_ratio": float(result["bucket32_compute_ratio"]),
        "compute_advantage_vs_k1": STATIC_K1_COMPUTE - float(result["bucket16_compute_ratio"]),
        "individual_verdict": payload["decision"]["verdict"],
    }


def make_decision(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    quality_all = all(
        row["dynamic_delta_vs_k1"] <= 0.010
        and row["dynamic_delta_ucb95"] <= 0.012
        for row in rows
    )
    strict_compute_all = all(row["bucket16_compute_ratio"] <= 0.2405 for row in rows)
    mean_compute = statistics.mean(row["bucket16_compute_ratio"] for row in rows)
    mean_advantage = STATIC_K1_COMPUTE - mean_compute
    if quality_all and strict_compute_all and mean_advantage >= 0.020:
        verdict = "ROBUST_MARGINAL_UTILITY_PASS"
    elif quality_all and mean_compute < STATIC_K1_COMPUTE:
        verdict = "ROBUST_MARGINAL_UTILITY_BORDERLINE"
    else:
        verdict = "ROBUST_MARGINAL_UTILITY_FAIL"
    return {
        "verdict": verdict,
        "quality_guard_passed_every_seed": quality_all,
        "strict_compute_guard_passed_every_seed": strict_compute_all,
        "mean_bucket16_compute_ratio": mean_compute,
        "mean_compute_advantage_vs_k1": mean_advantage,
        "worst_delta_vs_k1": max(row["dynamic_delta_vs_k1"] for row in rows),
        "worst_delta_ucb95": max(row["dynamic_delta_ucb95"] for row in rows),
        "worst_bucket16_compute_ratio": max(row["bucket16_compute_ratio"] for row in rows),
        "rule": (
            "PASS requires every seed within +0.010 nat of static K1, every "
            "paired-bootstrap UCB95 <= +0.012, every bucket16 compute <=24.05%, "
            "and mean compute advantage of at least two percentage points."
        ),
    }


def write_outputs(output_dir: Path, payload: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "marginal_utility_multiseed.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with (output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(payload["runs"][0].keys()))
        writer.writeheader()
        writer.writerows(payload["runs"])
    decision = payload["decision"]
    lines = [
        "# Test 2.11 — multi-seed marginal-utility controller",
        "",
        f"**Decision:** **{decision['verdict']}**",
        "",
        "| Seed | Lambda | Δ loss vs K1 | UCB95 | Mean K | p95 K | Exact compute | Bucket16 | Advantage vs K1 |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["runs"]:
        lines.append(
            f"| {row['seed']} | {row['lambda_nats']:.4f} | "
            f"{row['dynamic_delta_vs_k1']:+.4f} | {row['dynamic_delta_ucb95']:+.4f} | "
            f"{row['mean_rank']:.3f} | {row['p95_rank']:.0f} | "
            f"{row['exact_compute_ratio']:.3%} | {row['bucket16_compute_ratio']:.3%} | "
            f"{row['compute_advantage_vs_k1']:+.3%} |"
        )
    lines += [
        "",
        f"- Mean bucket16 compute: `{decision['mean_bucket16_compute_ratio']:.3%}`.",
        f"- Mean advantage over static K1: `{decision['mean_compute_advantage_vs_k1']:+.3%}`.",
        f"- Worst held-out Δ loss: `{decision['worst_delta_vs_k1']:+.4f}` nat.",
        f"- Worst paired-bootstrap UCB95: `{decision['worst_delta_ucb95']:+.4f}` nat.",
        f"- Worst bucket16 compute: `{decision['worst_bucket16_compute_ratio']:.3%}`.",
        "",
        "Every seed trains a new progressive Modal-MoE and new utility controllers. Utility price selection is confined to train-split calibration data.",
    ]
    (output_dir / "VERDICT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runner-source", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--progressive-source", type=Path, required=True)
    parser.add_argument("--controller-source", type=Path, required=True)
    parser.add_argument("--text", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seeds", default="61616,71717,81818")
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--train-batches", type=int, default=32)
    parser.add_argument("--tune-batches", type=int, default=24)
    parser.add_argument("--test-batches", type=int, default=24)
    parser.add_argument("--controller-steps", type=int, default=800)
    parser.add_argument("--controller-hidden", type=int, default=96)
    args = parser.parse_args(argv)

    started = time.perf_counter()
    seeds = [int(value) for value in args.seeds.split(",") if value.strip()]
    rows = [run_seed(args, seed, args.output_dir) for seed in seeds]
    payload = {
        "metadata": {
            "seeds": seeds,
            "steps": args.steps,
            "controller_steps": args.controller_steps,
            "experts": 64,
            "top_k": 8,
            "task": "Tiny Shakespeare character language modeling",
            "elapsed_seconds": time.perf_counter() - started,
            "lambda_grid": "extended through 0.060 nat per residual mode",
        },
        "runs": rows,
    }
    payload["decision"] = make_decision(rows)
    write_outputs(args.output_dir, payload)
    print((args.output_dir / "VERDICT.md").read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
