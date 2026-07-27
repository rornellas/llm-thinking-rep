#!/usr/bin/env python3
"""Test 2.8: statistically guarded, multi-seed rank-controller validation.

Test 2.7 produced a strong compute/quality curve but chose the most aggressive
threshold whose *point estimate* met a loose calibration limit.  The selected
point missed the pre-registered PASS boundary by 0.00105 nat on held-out data.

This experiment fixes the selection rule before observing new seeds:

* threshold is selected only on training-split calibration batches;
* selection uses a paired bootstrap 95% upper bound versus static K=1;
* the upper bound must be <= +0.003 nat;
* three fresh seeds are evaluated independently;
* hardware cost includes rank bucketing and tile padding after one upfront
  compaction, rather than the unrealistic maximum rank of contiguous tiles.

The dynamic forward is the exact mixed-rank graph from Test 2.7.  No expert
matrix is reconstructed.
"""
from __future__ import annotations

import argparse
import csv
import gc
import importlib.util
import json
import math
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn.functional as F


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@dataclass
class SeedResult:
    seed: int
    chosen_threshold: float
    tune_delta_mean: float
    tune_delta_ucb95: float
    baseline_loss: float
    static_k0_loss: float
    static_k1_loss: float
    static_k2_loss: float
    static_k3_loss: float
    dynamic_loss: float
    dynamic_delta_vs_k1: float
    dynamic_delta_ucb95: float
    mean_rank: float
    p50_rank: float
    p90_rank: float
    p95_rank: float
    exact_compute_ratio: float
    bucket16_compute_ratio: float
    bucket32_compute_ratio: float
    bucket64_compute_ratio: float
    compute_advantage_vs_k1: float
    pass_quality_compute: bool
    rank_counts: dict[str, int]


def paired_bootstrap_ucb(deltas: Sequence[float], seed: int, samples: int = 4000, quantile: float = 0.95) -> float:
    values = np.asarray(deltas, dtype=np.float64)
    if values.size == 0:
        raise ValueError("empty paired deltas")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, values.size, size=(samples, values.size))
    means = values[indices].mean(axis=1)
    return float(np.quantile(means, quantile))


def static_batch_losses(progressive: Any, model: Any, batches: Sequence[tuple[torch.Tensor, torch.Tensor]], rank: int) -> list[float]:
    progressive.set_active_rank(model, rank)
    model.eval()
    losses: list[float] = []
    with torch.no_grad():
        for tokens, targets in batches:
            logits, _, _ = model(tokens)
            losses.append(float(F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))))
    return losses


def bucketed_compute_ratio(
    rank_batches: Sequence[Sequence[np.ndarray]],
    *,
    tile_size: int,
    max_rank: int,
    controller_overhead: float,
    top_k: int = 8,
    d_model: int = 2048,
) -> float:
    """Cost after grouping tokens by required prefix and padding each mode bucket.

    Mode k executes for tokens with rank >= k.  Tokens are compacted once using
    their controller-predicted final rank.  This models grouped GEMMs without
    imposing the maximum rank on unrelated neighboring tokens.
    """
    padded_mode_tokens = 0
    original_tokens = 0
    residual_code_ops = 0
    for layers in rank_batches:
        for ranks in layers:
            flat = np.asarray(ranks, dtype=np.int64).reshape(-1)
            original_tokens += flat.size
            residual_code_ops += int(flat.sum())
            for mode in range(max_rank + 1):
                active = int(np.sum(flat >= mode))
                if active:
                    padded_mode_tokens += int(math.ceil(active / tile_size) * tile_size)
    matrix_ratio = padded_mode_tokens / (original_tokens * top_k)
    code_ratio = residual_code_ops / (original_tokens * d_model)
    return float(matrix_ratio + code_ratio + controller_overhead)


def exact_compute_ratio(rank_batches: Sequence[Sequence[np.ndarray]], controller_overhead: float) -> tuple[float, np.ndarray]:
    all_ranks = np.concatenate(
        [np.asarray(ranks, dtype=np.int64).reshape(-1) for layers in rank_batches for ranks in layers]
    )
    ratio = float(np.mean((all_ranks + 1.0) / 8.0 + all_ranks / 2048.0) + controller_overhead)
    return ratio, all_ranks


@torch.no_grad()
def evaluate_dynamic_batches(
    controller_module: Any,
    model: Any,
    controllers: Sequence[Any],
    batches: Sequence[tuple[torch.Tensor, torch.Tensor]],
    threshold: float,
    controller_overhead: float,
    max_rank: int,
) -> dict[str, Any]:
    model.eval()
    losses: list[float] = []
    rank_batches: list[list[np.ndarray]] = []
    for tokens, targets in batches:
        logits, ranks, _ = controller_module.forward_dynamic_model(
            model, tokens, controllers, threshold
        )
        losses.append(float(controller_module.token_losses(logits, targets).mean()))
        rank_batches.append([rank.detach().cpu().numpy() for rank in ranks])
    exact, all_ranks = exact_compute_ratio(rank_batches, controller_overhead)
    unique, counts = np.unique(all_ranks, return_counts=True)
    return {
        "batch_losses": losses,
        "validation_loss": float(np.mean(losses)),
        "mean_rank": float(np.mean(all_ranks)),
        "p50_rank": float(np.percentile(all_ranks, 50)),
        "p90_rank": float(np.percentile(all_ranks, 90)),
        "p95_rank": float(np.percentile(all_ranks, 95)),
        "rank_counts": {str(int(k)): int(v) for k, v in zip(unique, counts, strict=True)},
        "exact_compute_ratio": exact,
        "bucket16_compute_ratio": bucketed_compute_ratio(
            rank_batches, tile_size=16, max_rank=max_rank, controller_overhead=controller_overhead
        ),
        "bucket32_compute_ratio": bucketed_compute_ratio(
            rank_batches, tile_size=32, max_rank=max_rank, controller_overhead=controller_overhead
        ),
        "bucket64_compute_ratio": bucketed_compute_ratio(
            rank_batches, tile_size=64, max_rank=max_rank, controller_overhead=controller_overhead
        ),
    }


def select_threshold_guarded(
    controller_module: Any,
    model: Any,
    controllers: Sequence[Any],
    tune_batches: Sequence[tuple[torch.Tensor, torch.Tensor]],
    static_k1_batch_losses: Sequence[float],
    controller_overhead: float,
    max_rank: int,
    bootstrap_seed: int,
    margin_nats: float = 0.003,
) -> tuple[float, list[dict[str, Any]]]:
    candidates: list[dict[str, Any]] = []
    thresholds = np.arange(0.45, 0.7001, 0.025)
    for index, threshold in enumerate(thresholds):
        dynamic = evaluate_dynamic_batches(
            controller_module, model, controllers, tune_batches,
            float(threshold), controller_overhead, max_rank,
        )
        deltas = [
            dynamic_loss - static_loss
            for dynamic_loss, static_loss in zip(dynamic["batch_losses"], static_k1_batch_losses, strict=True)
        ]
        row = {
            **{key: value for key, value in dynamic.items() if key != "batch_losses"},
            "threshold": float(threshold),
            "delta_mean": float(np.mean(deltas)),
            "delta_ucb95": paired_bootstrap_ucb(deltas, bootstrap_seed + index),
            "guard_margin_nats": margin_nats,
        }
        row["guard_pass"] = row["delta_ucb95"] <= margin_nats
        candidates.append(row)
        print(
            f"threshold={threshold:.3f} tune-delta={row['delta_mean']:+.4f} "
            f"ucb95={row['delta_ucb95']:+.4f} exact={row['exact_compute_ratio']:.3%} "
            f"bucket16={row['bucket16_compute_ratio']:.3%}",
            flush=True,
        )
    feasible = [row for row in candidates if row["guard_pass"]]
    if feasible:
        chosen = min(feasible, key=lambda row: (row["bucket16_compute_ratio"], row["delta_ucb95"]))
    else:
        chosen = min(candidates, key=lambda row: (row["delta_ucb95"], row["bucket16_compute_ratio"]))
    return float(chosen["threshold"]), candidates


def make_batches(dataset: Any, split: str, count: int, batch_size: int, seed: int):
    generator = torch.Generator().manual_seed(seed)
    return [dataset.batch(split, batch_size, generator) for _ in range(count)]


def run_seed(
    source: Any,
    progressive: Any,
    controller_module: Any,
    dataset: Any,
    *,
    seed: int,
    steps: int,
    controller_steps: int,
    controller_hidden: int,
    train_batches_count: int,
    tune_batches_count: int,
    test_batches_count: int,
    tolerance: float,
) -> tuple[SeedResult, dict[str, Any]]:
    max_rank = 3
    cfg = source.Config(
        steps=steps,
        eval_interval=100,
        eval_batches=20,
        batch_size=16,
        seq_len=64,
    )
    print(f"\n===== seed {seed} =====", flush=True)
    baseline = progressive.train_baseline(source, dataset, cfg, seed)
    model, training_seconds = progressive.train_progressive(source, dataset, cfg, seed + 1, max_rank)

    train_features, train_labels, _ = controller_module.collect_controller_data(
        progressive, model, dataset, cfg,
        split="train", batches=train_batches_count, seed=seed + 100,
        max_rank=max_rank, tolerance=tolerance, keep_batches=False,
    )
    _, _, tune_batches = controller_module.collect_controller_data(
        progressive, model, dataset, cfg,
        split="train", batches=tune_batches_count, seed=seed + 200,
        max_rank=max_rank, tolerance=tolerance, keep_batches=True,
    )
    _, test_labels, test_batches = controller_module.collect_controller_data(
        progressive, model, dataset, cfg,
        split="validation", batches=test_batches_count, seed=seed + 300,
        max_rank=max_rank, tolerance=tolerance, keep_batches=True,
    )
    controllers, controller_training = controller_module.train_controllers(
        train_features, train_labels,
        max_rank=max_rank, hidden_dim=controller_hidden,
        steps=controller_steps, seed=seed + 400,
    )
    overhead = controller_module.olmoe_controller_overhead(controller_hidden, max_rank)

    static_tune = {
        rank: static_batch_losses(progressive, model, tune_batches, rank)
        for rank in range(max_rank + 1)
    }
    chosen_threshold, threshold_sweep = select_threshold_guarded(
        controller_module, model, controllers, tune_batches, static_tune[1],
        overhead, max_rank, seed + 500,
    )

    static_test_batches = {
        rank: static_batch_losses(progressive, model, test_batches, rank)
        for rank in range(max_rank + 1)
    }
    baseline_batches = static_batch_losses(progressive, baseline, test_batches, 0)
    dynamic = evaluate_dynamic_batches(
        controller_module, model, controllers, test_batches,
        chosen_threshold, overhead, max_rank,
    )
    test_deltas = [
        dynamic_loss - static_loss
        for dynamic_loss, static_loss in zip(dynamic["batch_losses"], static_test_batches[1], strict=True)
    ]
    delta_mean = float(np.mean(test_deltas))
    delta_ucb = paired_bootstrap_ucb(test_deltas, seed + 600)
    static_means = {rank: float(np.mean(values)) for rank, values in static_test_batches.items()}
    static_k1_compute = (1 + 1) / 8.0 + 1 / 2048.0
    compute_advantage = static_k1_compute - dynamic["bucket16_compute_ratio"]
    passed = bool(delta_mean <= 0.010 and compute_advantage >= 0.010)

    result = SeedResult(
        seed=seed,
        chosen_threshold=chosen_threshold,
        tune_delta_mean=next(row["delta_mean"] for row in threshold_sweep if abs(row["threshold"] - chosen_threshold) < 1e-9),
        tune_delta_ucb95=next(row["delta_ucb95"] for row in threshold_sweep if abs(row["threshold"] - chosen_threshold) < 1e-9),
        baseline_loss=float(np.mean(baseline_batches)),
        static_k0_loss=static_means[0],
        static_k1_loss=static_means[1],
        static_k2_loss=static_means[2],
        static_k3_loss=static_means[3],
        dynamic_loss=dynamic["validation_loss"],
        dynamic_delta_vs_k1=delta_mean,
        dynamic_delta_ucb95=delta_ucb,
        mean_rank=dynamic["mean_rank"],
        p50_rank=dynamic["p50_rank"],
        p90_rank=dynamic["p90_rank"],
        p95_rank=dynamic["p95_rank"],
        exact_compute_ratio=dynamic["exact_compute_ratio"],
        bucket16_compute_ratio=dynamic["bucket16_compute_ratio"],
        bucket32_compute_ratio=dynamic["bucket32_compute_ratio"],
        bucket64_compute_ratio=dynamic["bucket64_compute_ratio"],
        compute_advantage_vs_k1=compute_advantage,
        pass_quality_compute=passed,
        rank_counts=dynamic["rank_counts"],
    )
    diagnostics = {
        "seed": seed,
        "training_seconds": training_seconds,
        "controller_training": controller_training,
        "train_oracle_distribution": {
            str(rank): int(torch.sum(train_labels == rank)) for rank in range(max_rank + 1)
        },
        "test_oracle_distribution": {
            str(rank): int(torch.sum(test_labels == rank)) for rank in range(max_rank + 1)
        },
        "threshold_sweep": threshold_sweep,
        "test_batch_deltas": test_deltas,
    }
    del baseline, model, controllers, train_features, train_labels
    gc.collect()
    return result, diagnostics


def aggregate_decision(results: Sequence[SeedResult]) -> dict[str, Any]:
    deltas = np.asarray([row.dynamic_delta_vs_k1 for row in results])
    bucket16 = np.asarray([row.bucket16_compute_ratio for row in results])
    all_pass = all(row.pass_quality_compute for row in results)
    if all_pass and float(np.mean(deltas)) <= 0.008 and float(np.max(bucket16)) <= 0.30:
        verdict = "ROBUST_CONTROLLER_PASS"
    elif float(np.mean(deltas)) <= 0.015 and float(np.mean(bucket16)) <= 0.32:
        verdict = "ROBUST_CONTROLLER_BORDERLINE"
    else:
        verdict = "ROBUST_CONTROLLER_FAIL"
    return {
        "verdict": verdict,
        "all_seed_pass": all_pass,
        "mean_dynamic_delta_vs_k1": float(np.mean(deltas)),
        "max_dynamic_delta_vs_k1": float(np.max(deltas)),
        "mean_exact_compute_ratio": float(np.mean([row.exact_compute_ratio for row in results])),
        "mean_bucket16_compute_ratio": float(np.mean(bucket16)),
        "max_bucket16_compute_ratio": float(np.max(bucket16)),
        "rule": "PASS requires every fresh seed within +0.010 nat of static K1 with >=1 percentage point bucket16 compute advantage; aggregate mean delta <=0.008 and worst bucket16 compute <=30%.",
    }


def write_outputs(output_dir: Path, payload: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "controller_robustness.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    rows = payload["seed_results"]
    flat_rows = [{key: value for key, value in row.items() if key != "rank_counts"} for row in rows]
    with (output_dir / "seed_results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flat_rows[0].keys()))
        writer.writeheader(); writer.writerows(flat_rows)

    decision = payload["decision"]
    lines = [
        "# Test 2.8 — robust learned rank controller", "",
        f"**Decision:** **{decision['verdict']}**", "",
        "Thresholds were selected on train-split calibration data using a paired-bootstrap 95% upper bound of +0.003 nat versus static K=1.", "",
        "| Seed | Threshold | Dynamic loss | Δ vs K1 | Test UCB95 | Mean K | Exact compute | Bucket16 | Bucket32 | Bucket64 | Pass |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['seed']} | {row['chosen_threshold']:.3f} | {row['dynamic_loss']:.4f} | "
            f"{row['dynamic_delta_vs_k1']:+.4f} | {row['dynamic_delta_ucb95']:+.4f} | "
            f"{row['mean_rank']:.3f} | {row['exact_compute_ratio']:.3%} | "
            f"{row['bucket16_compute_ratio']:.3%} | {row['bucket32_compute_ratio']:.3%} | "
            f"{row['bucket64_compute_ratio']:.3%} | {'yes' if row['pass_quality_compute'] else 'no'} |"
        )
    lines += [
        "", "## Aggregate",
        f"- mean Δ loss vs static K1: `{decision['mean_dynamic_delta_vs_k1']:+.4f}` nat; worst `{decision['max_dynamic_delta_vs_k1']:+.4f}`.",
        f"- mean exact projected compute: `{decision['mean_exact_compute_ratio']:.3%}`.",
        f"- mean bucket16 projected compute: `{decision['mean_bucket16_compute_ratio']:.3%}`; worst `{decision['max_bucket16_compute_ratio']:.3%}`.",
        "", "Bucket cost assumes one upfront compaction by predicted final rank. Mode k is executed only for tokens requiring rank >= k, with each grouped GEMM padded to the indicated tile size. Compaction/scatter latency itself is not yet benchmarked.",
    ]
    (output_dir / "VERDICT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def self_test(controller_module: Any) -> None:
    ranks = [
        [np.asarray([0, 0, 1, 3, 2, 0, 1, 3, 0, 0], dtype=np.int64)],
        [np.asarray([0, 2, 2, 3, 0, 0, 1], dtype=np.int64)],
    ]
    overhead = 0.002
    exact, flat = exact_compute_ratio(ranks, overhead)
    bucket = bucketed_compute_ratio(ranks, tile_size=4, max_rank=3, controller_overhead=overhead)
    if exact <= 0 or bucket < exact or flat.size != 17:
        raise AssertionError((exact, bucket, flat.size))
    ucb = paired_bootstrap_ucb([0.0, 0.001, -0.001, 0.002], 17, samples=1000)
    if not math.isfinite(ucb):
        raise AssertionError(ucb)
    print(f"self-test passed: exact={exact:.4f} bucket4={bucket:.4f} ucb={ucb:.4f}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--progressive-source", type=Path, required=True)
    parser.add_argument("--controller-source", type=Path, required=True)
    parser.add_argument("--text", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("results/test-2-8/latest"))
    parser.add_argument("--seeds", default="10101,20202,30303")
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--controller-steps", type=int, default=600)
    parser.add_argument("--controller-hidden", type=int, default=64)
    parser.add_argument("--train-batches", type=int, default=32)
    parser.add_argument("--tune-batches", type=int, default=12)
    parser.add_argument("--test-batches", type=int, default=24)
    parser.add_argument("--oracle-tolerance", type=float, default=0.05)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args(argv)

    source = load_module("robust_base_source", args.source)
    progressive = load_module("robust_progressive_source", args.progressive_source)
    controller_module = load_module("robust_controller_source", args.controller_source)
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    if args.self_test:
        self_test(controller_module); return 0

    dataset = source.CharDataset(args.text.read_text(encoding="utf-8"), 64)
    seeds = [int(value) for value in args.seeds.split(",") if value.strip()]
    started = time.perf_counter()
    seed_results: list[SeedResult] = []
    diagnostics: list[dict[str, Any]] = []
    for seed in seeds:
        result, detail = run_seed(
            source, progressive, controller_module, dataset,
            seed=seed, steps=args.steps, controller_steps=args.controller_steps,
            controller_hidden=args.controller_hidden,
            train_batches_count=args.train_batches,
            tune_batches_count=args.tune_batches,
            test_batches_count=args.test_batches,
            tolerance=args.oracle_tolerance,
        )
        seed_results.append(result); diagnostics.append(detail)

    payload = {
        "metadata": {
            "seeds": seeds,
            "steps": args.steps,
            "controller_steps": args.controller_steps,
            "controller_hidden": args.controller_hidden,
            "train_batches": args.train_batches,
            "tune_batches": args.tune_batches,
            "test_batches": args.test_batches,
            "oracle_tolerance_nats": args.oracle_tolerance,
            "threshold_guard_ucb95_nats": 0.003,
            "elapsed_seconds": time.perf_counter() - started,
        },
        "decision": aggregate_decision(seed_results),
        "seed_results": [asdict(row) for row in seed_results],
        "diagnostics": diagnostics,
    }
    write_outputs(args.output_dir, payload)
    print((args.output_dir / "VERDICT.md").read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
