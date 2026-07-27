#!/usr/bin/env python3
"""Test 2.10: marginal-utility controller for nested Modal-MoE modes.

The ordinal controller predicts an oracle rank class.  That objective ignores
how expensive a mistake is: confusing K=0 with K=1 can be harmless for one
token and costly for another.  This experiment instead predicts the final
language-loss benefit of each nested prefix and chooses

    argmax_K  predicted_benefit(K) - lambda * K.

The multiplier ``lambda`` is selected only on train-split calibration batches
using a paired-bootstrap quality guard.  Evaluation uses a fresh validation
split and a genuinely mixed-rank forward pass.  The model geometry matches
OLMoE's 64 experts / top-8 routing, while width and corpus remain small.
"""
from __future__ import annotations

import argparse
import csv
import gc
import importlib.util
import json
import math
import random
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class MarginalUtilityController(nn.Module):
    """Predict cumulative final-loss benefit for K=1..Kmax over K=0."""

    def __init__(self, feature_dim: int, max_rank: int, hidden_dim: int, benefit_scale: float = 100.0) -> None:
        super().__init__()
        self.max_rank = max_rank
        self.benefit_scale = float(benefit_scale)
        self.net = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, max_rank),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features) / self.benefit_scale

    @torch.no_grad()
    def predict_rank(self, features: torch.Tensor, threshold: float) -> torch.Tensor:
        benefits = self(features)
        rank_costs = torch.arange(
            1, self.max_rank + 1, device=features.device, dtype=benefits.dtype
        ) * float(threshold)
        scores = benefits - rank_costs[None, :]
        scores = torch.cat(
            [torch.zeros(scores.shape[0], 1, device=scores.device, dtype=scores.dtype), scores],
            dim=-1,
        )
        return torch.argmax(scores, dim=-1)


def causal_mask(length: int, device: torch.device) -> torch.Tensor:
    mask = torch.full((length, length), float("-inf"), device=device)
    return torch.triu(mask, diagonal=1)


@torch.no_grad()
def forward_layer_ranks(
    controller_module: Any,
    model: nn.Module,
    tokens: torch.Tensor,
    layer_ranks: Sequence[int],
    *,
    collect_features: bool = False,
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    if len(layer_ranks) != len(model.blocks):
        raise ValueError((len(layer_ranks), len(model.blocks)))
    positions = torch.arange(tokens.shape[1], device=tokens.device)
    x = model.token_embedding(tokens) + model.position_embedding(positions)[None, :, :]
    mask = causal_mask(tokens.shape[1], tokens.device)
    features: list[torch.Tensor] = []
    for layer_index, block in enumerate(model.blocks):
        normalized = block.norm1(x)
        attention, _ = block.attn(
            normalized, normalized, normalized, attn_mask=mask, need_weights=False
        )
        x = x + attention
        moe_input = block.norm2(x)
        forced = torch.full(
            (tokens.numel(),),
            int(layer_ranks[layer_index]),
            device=tokens.device,
            dtype=torch.long,
        )
        moe_output, _, _, _, layer_features = controller_module.dynamic_moe_forward(
            block.moe, moe_input, forced_ranks=forced
        )
        x = x + moe_output
        if collect_features:
            features.append(layer_features.detach())
    return model.output(model.norm(x)), features


def token_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        targets.reshape(-1),
        reduction="none",
    )


@torch.no_grad()
def collect_utility_data(
    controller_module: Any,
    model: nn.Module,
    dataset: Any,
    cfg: Any,
    *,
    split: str,
    batches: int,
    seed: int,
    max_rank: int,
    keep_batches: bool,
) -> tuple[list[torch.Tensor], list[torch.Tensor], list[tuple[torch.Tensor, torch.Tensor]]]:
    generator = torch.Generator().manual_seed(seed)
    feature_chunks: list[list[torch.Tensor]] = [[] for _ in model.blocks]
    target_chunks: list[list[torch.Tensor]] = [[] for _ in model.blocks]
    stored: list[tuple[torch.Tensor, torch.Tensor]] = []
    model.eval()
    full_ranks = [max_rank] * len(model.blocks)

    for batch_index in range(batches):
        tokens, targets = dataset.batch(split, cfg.batch_size, generator)
        _, features = forward_layer_ranks(
            controller_module, model, tokens, full_ranks, collect_features=True
        )
        for layer_index in range(len(model.blocks)):
            base_config = full_ranks.copy()
            base_config[layer_index] = 0
            base_logits, _ = forward_layer_ranks(
                controller_module, model, tokens, base_config
            )
            base_loss = token_loss(base_logits, targets)
            benefits: list[torch.Tensor] = []
            for rank in range(1, max_rank + 1):
                config = full_ranks.copy()
                config[layer_index] = rank
                logits, _ = forward_layer_ranks(
                    controller_module, model, tokens, config
                )
                benefits.append(base_loss - token_loss(logits, targets))
            target = torch.stack(benefits, dim=-1)
            feature_chunks[layer_index].append(features[layer_index].cpu())
            target_chunks[layer_index].append(target.cpu())
        if keep_batches:
            stored.append((tokens.cpu(), targets.cpu()))
        if batch_index == 0 or (batch_index + 1) % 8 == 0:
            print(f"collected {split} utility batch {batch_index + 1}/{batches}", flush=True)

    return (
        [torch.cat(chunks, dim=0) for chunks in feature_chunks],
        [torch.cat(chunks, dim=0) for chunks in target_chunks],
        stored,
    )


def train_utility_controllers(
    feature_layers: Sequence[torch.Tensor],
    target_layers: Sequence[torch.Tensor],
    *,
    max_rank: int,
    hidden_dim: int,
    steps: int,
    seed: int,
) -> tuple[nn.ModuleList, list[dict[str, float]]]:
    set_seed(seed)
    controllers = nn.ModuleList()
    diagnostics: list[dict[str, float]] = []
    for layer_index, (features, targets) in enumerate(
        zip(feature_layers, target_layers, strict=True)
    ):
        controller = MarginalUtilityController(
            features.shape[-1], max_rank, hidden_dim
        )
        optimizer = torch.optim.AdamW(
            controller.parameters(), lr=2.0e-3, weight_decay=2.0e-3
        )
        generator = torch.Generator().manual_seed(seed + 1000 + layer_index)
        target_scaled = targets * controller.benefit_scale
        controller.train()
        final_loss = math.nan
        for step in range(1, steps + 1):
            indices = torch.randint(
                0,
                features.shape[0],
                (min(1536, features.shape[0]),),
                generator=generator,
            )
            x = features.index_select(0, indices)
            y = target_scaled.index_select(0, indices)
            prediction_scaled = controller.net(x)
            magnitude = y.abs().amax(dim=-1)
            weights = 1.0 + 2.0 * (magnitude / 2.0).clamp(0.0, 2.0)
            regression = F.smooth_l1_loss(
                prediction_scaled, y, reduction="none", beta=0.25
            ).mean(dim=-1)
            regression = torch.mean(regression * weights)

            lambdas = torch.rand(x.shape[0], generator=generator) * 2.0
            rank_costs = torch.arange(1, max_rank + 1, dtype=y.dtype)[None, :] * lambdas[:, None]
            true_scores = torch.cat(
                [torch.zeros(x.shape[0], 1), y - rank_costs], dim=-1
            )
            predicted_scores = torch.cat(
                [torch.zeros(x.shape[0], 1), prediction_scaled - rank_costs], dim=-1
            )
            best_rank = torch.argmax(true_scores, dim=-1)
            ranking = F.cross_entropy(predicted_scores, best_rank)
            loss = regression + 0.35 * ranking
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(controller.parameters(), 1.0)
            optimizer.step()
            final_loss = float(loss.detach())
            if step == 1 or step % 200 == 0 or step == steps:
                print(
                    f"utility-controller layer={layer_index} step={step}/{steps} "
                    f"loss={final_loss:.4f} regression={float(regression):.4f} "
                    f"ranking={float(ranking):.4f}",
                    flush=True,
                )
        controller.eval()
        with torch.no_grad():
            prediction = controller(features)
            mae = float(torch.mean(torch.abs(prediction - targets)))
            correlation_rows = []
            for rank in range(max_rank):
                a = prediction[:, rank].numpy()
                b = targets[:, rank].numpy()
                correlation_rows.append(
                    float(np.corrcoef(a, b)[0, 1])
                    if np.std(a) > 0 and np.std(b) > 0
                    else 0.0
                )
        controllers.append(controller)
        diagnostics.append(
            {
                "layer": float(layer_index),
                "final_training_loss": final_loss,
                "benefit_mae_nats": mae,
                "mean_rank_correlation": float(np.mean(correlation_rows)),
                "parameters": float(
                    sum(parameter.numel() for parameter in controller.parameters())
                ),
            }
        )
    return controllers, diagnostics


@torch.no_grad()
def static_batch_losses(
    progressive: Any,
    model: nn.Module,
    batches: Sequence[tuple[torch.Tensor, torch.Tensor]],
    rank: int,
) -> list[float]:
    progressive.set_active_rank(model, rank)
    model.eval()
    losses: list[float] = []
    for tokens, targets in batches:
        logits, _, _ = model(tokens)
        losses.append(float(token_loss(logits, targets).mean()))
    return losses


def paired_bootstrap_ucb(
    deltas: Sequence[float], seed: int, samples: int = 4000, quantile: float = 0.95
) -> float:
    values = np.asarray(deltas, dtype=np.float64)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, values.size, size=(samples, values.size))
    return float(np.quantile(values[indices].mean(axis=1), quantile))


def controller_overhead_ratio(
    hidden_dim: int,
    max_rank: int,
    *,
    d_model: int = 2048,
    summary_dim: int = 17,
) -> float:
    feature_dim = d_model + summary_dim
    controller_macs = (
        feature_dim * hidden_dim
        + hidden_dim * hidden_dim
        + hidden_dim * max_rank
    )
    original_expert_macs = 3 * 8 * 2048 * 1024
    return controller_macs / original_expert_macs


def exact_and_bucket_compute(
    rank_batches: Sequence[Sequence[np.ndarray]],
    controller_overhead: float,
    tile_size: int,
    max_rank: int,
) -> tuple[float, float, np.ndarray]:
    flat = np.concatenate(
        [np.asarray(ranks, dtype=np.int64).reshape(-1) for batch in rank_batches for ranks in batch]
    )
    exact = float(
        np.mean((flat + 1.0) / 8.0 + flat / 2048.0) + controller_overhead
    )
    padded_mode_tokens = 0
    total_tokens = 0
    residual_codes = 0
    for batch in rank_batches:
        for ranks in batch:
            values = np.asarray(ranks, dtype=np.int64).reshape(-1)
            total_tokens += values.size
            residual_codes += int(values.sum())
            for mode in range(max_rank + 1):
                active = int(np.sum(values >= mode))
                if active:
                    padded_mode_tokens += int(math.ceil(active / tile_size) * tile_size)
    bucket = (
        padded_mode_tokens / (total_tokens * 8.0)
        + residual_codes / (total_tokens * 2048.0)
        + controller_overhead
    )
    return exact, float(bucket), flat


@torch.no_grad()
def evaluate_dynamic(
    controller_module: Any,
    model: nn.Module,
    controllers: Sequence[MarginalUtilityController],
    batches: Sequence[tuple[torch.Tensor, torch.Tensor]],
    lambda_nats: float,
    overhead: float,
    max_rank: int,
) -> dict[str, Any]:
    model.eval()
    losses: list[float] = []
    rank_batches: list[list[np.ndarray]] = []
    for tokens, targets in batches:
        logits, ranks, _ = controller_module.forward_dynamic_model(
            model, tokens, controllers, lambda_nats
        )
        losses.append(float(token_loss(logits, targets).mean()))
        rank_batches.append([rank.cpu().numpy() for rank in ranks])
    exact, bucket16, flat = exact_and_bucket_compute(
        rank_batches, overhead, 16, max_rank
    )
    _, bucket32, _ = exact_and_bucket_compute(
        rank_batches, overhead, 32, max_rank
    )
    unique, counts = np.unique(flat, return_counts=True)
    return {
        "lambda_nats": float(lambda_nats),
        "batch_losses": losses,
        "validation_loss": float(np.mean(losses)),
        "mean_rank": float(np.mean(flat)),
        "p50_rank": float(np.percentile(flat, 50)),
        "p90_rank": float(np.percentile(flat, 90)),
        "p95_rank": float(np.percentile(flat, 95)),
        "exact_compute_ratio": exact,
        "bucket16_compute_ratio": bucket16,
        "bucket32_compute_ratio": bucket32,
        "rank_counts": {
            str(int(key)): int(value)
            for key, value in zip(unique, counts, strict=True)
        },
    }


def select_lambda(
    controller_module: Any,
    model: nn.Module,
    controllers: Sequence[MarginalUtilityController],
    tune_batches: Sequence[tuple[torch.Tensor, torch.Tensor]],
    static_k1_losses: Sequence[float],
    overhead: float,
    max_rank: int,
    seed: int,
) -> tuple[float, list[dict[str, Any]]]:
    candidates: list[dict[str, Any]] = []
    lambdas = np.linspace(0.0, 0.020, 21)
    for index, value in enumerate(lambdas):
        row = evaluate_dynamic(
            controller_module, model, controllers, tune_batches,
            float(value), overhead, max_rank,
        )
        deltas = [
            dynamic - static
            for dynamic, static in zip(
                row["batch_losses"], static_k1_losses, strict=True
            )
        ]
        row["delta_mean"] = float(np.mean(deltas))
        row["delta_ucb95"] = paired_bootstrap_ucb(
            deltas, seed + index
        )
        row["guard_pass"] = row["delta_ucb95"] <= 0.005
        candidates.append(row)
        print(
            f"lambda={value:.3f} delta={row['delta_mean']:+.4f} "
            f"ucb95={row['delta_ucb95']:+.4f} mean-K={row['mean_rank']:.3f} "
            f"bucket16={row['bucket16_compute_ratio']:.3%}",
            flush=True,
        )
    feasible = [row for row in candidates if row["guard_pass"]]
    if feasible:
        chosen = min(
            feasible,
            key=lambda row: (
                row["bucket16_compute_ratio"], row["delta_ucb95"]
            ),
        )
    else:
        chosen = min(
            candidates,
            key=lambda row: (
                row["delta_ucb95"], row["bucket16_compute_ratio"]
            ),
        )
    return float(chosen["lambda_nats"]), candidates


@dataclass
class TestResult:
    lambda_nats: float
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
    compute_advantage_vs_k1: float
    rank_counts: dict[str, int]


def make_decision(result: TestResult) -> dict[str, Any]:
    if (
        result.dynamic_delta_vs_k1 <= 0.010
        and result.dynamic_delta_ucb95 <= 0.012
        and result.bucket16_compute_ratio <= 0.2405
    ):
        verdict = "MARGINAL_UTILITY_PASS"
    elif (
        result.dynamic_delta_vs_k1 <= 0.015
        and result.bucket16_compute_ratio <= 0.2505
    ):
        verdict = "MARGINAL_UTILITY_BORDERLINE"
    else:
        verdict = "MARGINAL_UTILITY_FAIL"
    return {
        "verdict": verdict,
        "rule": "PASS requires held-out loss within +0.010 nat of static K1, paired-bootstrap UCB95 <= +0.012, and rank-compacted tile16 compute <=24.05% of original OLMoE expert projections.",
    }


def write_outputs(
    output_dir: Path,
    payload: dict[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "marginal_utility.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    sweep_rows = [
        {key: value for key, value in row.items() if key not in {"batch_losses", "rank_counts"}}
        for row in payload["lambda_sweep"]
    ]
    with (output_dir / "lambda_sweep.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(sweep_rows[0].keys())
        )
        writer.writeheader(); writer.writerows(sweep_rows)

    result = payload["test_result"]
    decision = payload["decision"]
    lines = [
        "# Test 2.10 — marginal-utility rank controller",
        "",
        f"**Decision:** **{decision['verdict']}**",
        "",
        f"Chosen utility price: `{result['lambda_nats']:.4f}` nat per residual mode (train-calibration only).",
        "",
        "| Policy | Validation loss | Δ vs K1 | Mean K | p95 K | Exact compute | Bucket16 | Bucket32 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
        f"| Static K=0 | {result['static_k0_loss']:.4f} | {result['static_k0_loss'] - result['static_k1_loss']:+.4f} | 0 | 0 | 12.500% | 12.500% | 12.500% |",
        f"| Static K=1 | {result['static_k1_loss']:.4f} | +0.0000 | 1 | 1 | 25.049% | 25.049% | 25.049% |",
        f"| Static K=2 | {result['static_k2_loss']:.4f} | {result['static_k2_loss'] - result['static_k1_loss']:+.4f} | 2 | 2 | 37.598% | 37.598% | 37.598% |",
        f"| Static K=3 | {result['static_k3_loss']:.4f} | {result['static_k3_loss'] - result['static_k1_loss']:+.4f} | 3 | 3 | 50.146% | 50.146% | 50.146% |",
        f"| **Marginal dynamic** | **{result['dynamic_loss']:.4f}** | **{result['dynamic_delta_vs_k1']:+.4f}** | **{result['mean_rank']:.3f}** | **{result['p95_rank']:.0f}** | **{result['exact_compute_ratio']:.3%}** | **{result['bucket16_compute_ratio']:.3%}** | **{result['bucket32_compute_ratio']:.3%}** |",
        "",
        f"- Test paired-bootstrap UCB95 vs static K1: `{result['dynamic_delta_ucb95']:+.4f}` nat.",
        f"- Bucket16 compute advantage vs K1: `{result['compute_advantage_vs_k1']:+.3%}`.",
        f"- Rank counts: `{result['rank_counts']}`.",
        "",
        "Each controller predicts the cumulative final-language-loss improvement of K=1,2,3 over K=0 for its layer. Inference chooses the prefix maximizing predicted benefit minus a calibrated price per residual mode. Reported bucket cost assumes one rank compaction before grouped GEMMs; scan/scatter latency remains for a kernel benchmark.",
    ]
    (output_dir / "VERDICT.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def self_test(controller_module: Any, progressive: Any) -> None:
    set_seed(17)
    moe = progressive.ProgressiveNeuronwiseMoE(
        12, 20, 7, 3, 3
    )
    x = torch.randn(2, 5, 12)
    features_dim = 12 + 17
    controller = MarginalUtilityController(
        features_dim, 3, 16
    )
    benefits = controller(torch.randn(10, features_dim))
    ranks = controller.predict_rank(
        torch.randn(10, features_dim), 0.01
    )
    if benefits.shape != (10, 3) or ranks.min() < 0 or ranks.max() > 3:
        raise AssertionError((benefits.shape, ranks))
    for rank in range(4):
        moe.active_rank = rank
        reference, _, _ = moe(x)
        forced = torch.full((10,), rank, dtype=torch.long)
        dynamic, _, _, _, _ = controller_module.dynamic_moe_forward(
            moe, x, forced_ranks=forced
        )
        error = float(torch.max(torch.abs(reference - dynamic)))
        if error > 2e-5:
            raise AssertionError((rank, error))
    print("self-test passed for utility decisions and exact mixed-rank algebra")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--progressive-source", type=Path, required=True)
    parser.add_argument("--controller-source", type=Path, required=True)
    parser.add_argument("--text", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("results/test-2-10/latest"))
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--seed", type=int, default=51515)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--train-batches", type=int, default=32)
    parser.add_argument("--tune-batches", type=int, default=24)
    parser.add_argument("--test-batches", type=int, default=24)
    parser.add_argument("--controller-steps", type=int, default=800)
    parser.add_argument("--controller-hidden", type=int, default=96)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args(argv)

    source = load_module("utility_base_source", args.source)
    progressive = load_module(
        "utility_progressive_source", args.progressive_source
    )
    controller_module = load_module(
        "utility_dynamic_source", args.controller_source
    )

    original_config = source.Config

    class Scale64Config(original_config):
        def __init__(self, *config_args, **config_kwargs):
            config_kwargs.update(
                {
                    "d_model": 96,
                    "n_heads": 4,
                    "n_layers": 2,
                    "d_ff": 128,
                    "n_experts": 64,
                    "top_k": 8,
                    "seq_len": 64,
                    "batch_size": 16,
                }
            )
            super().__init__(*config_args, **config_kwargs)

    source.Config = Scale64Config
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    if args.self_test:
        self_test(controller_module, progressive); return 0

    started = time.perf_counter()
    max_rank = 3
    cfg = source.Config(
        steps=args.steps,
        eval_interval=100,
        eval_batches=20,
    )
    dataset = source.CharDataset(
        args.text.read_text(encoding="utf-8"), cfg.seq_len
    )
    baseline = progressive.train_baseline(
        source, dataset, cfg, args.seed
    )
    model, progressive_seconds = progressive.train_progressive(
        source, dataset, cfg, args.seed + 1, max_rank
    )

    train_features, train_targets, _ = collect_utility_data(
        controller_module, model, dataset, cfg,
        split="train", batches=args.train_batches,
        seed=args.seed + 100, max_rank=max_rank,
        keep_batches=False,
    )
    _, _, tune_batches = collect_utility_data(
        controller_module, model, dataset, cfg,
        split="train", batches=args.tune_batches,
        seed=args.seed + 200, max_rank=max_rank,
        keep_batches=True,
    )
    _, _, test_batches = collect_utility_data(
        controller_module, model, dataset, cfg,
        split="validation", batches=args.test_batches,
        seed=args.seed + 300, max_rank=max_rank,
        keep_batches=True,
    )

    controllers, training_diagnostics = train_utility_controllers(
        train_features, train_targets,
        max_rank=max_rank,
        hidden_dim=args.controller_hidden,
        steps=args.controller_steps,
        seed=args.seed + 400,
    )
    overhead = controller_overhead_ratio(
        args.controller_hidden, max_rank
    )
    tune_k1 = static_batch_losses(
        progressive, model, tune_batches, 1
    )
    chosen_lambda, lambda_sweep = select_lambda(
        controller_module, model, controllers,
        tune_batches, tune_k1, overhead,
        max_rank, args.seed + 500,
    )

    static_test_batches = {
        rank: static_batch_losses(
            progressive, model, test_batches, rank
        )
        for rank in range(max_rank + 1)
    }
    baseline_batches = static_batch_losses(
        progressive, baseline, test_batches, 0
    )
    dynamic = evaluate_dynamic(
        controller_module, model, controllers,
        test_batches, chosen_lambda,
        overhead, max_rank,
    )
    deltas = [
        dynamic_loss - static_loss
        for dynamic_loss, static_loss in zip(
            dynamic["batch_losses"],
            static_test_batches[1],
            strict=True,
        )
    ]
    static_means = {
        rank: float(np.mean(values))
        for rank, values in static_test_batches.items()
    }
    result = TestResult(
        lambda_nats=chosen_lambda,
        baseline_loss=float(np.mean(baseline_batches)),
        static_k0_loss=static_means[0],
        static_k1_loss=static_means[1],
        static_k2_loss=static_means[2],
        static_k3_loss=static_means[3],
        dynamic_loss=dynamic["validation_loss"],
        dynamic_delta_vs_k1=float(np.mean(deltas)),
        dynamic_delta_ucb95=paired_bootstrap_ucb(
            deltas, args.seed + 600
        ),
        mean_rank=dynamic["mean_rank"],
        p50_rank=dynamic["p50_rank"],
        p90_rank=dynamic["p90_rank"],
        p95_rank=dynamic["p95_rank"],
        exact_compute_ratio=dynamic["exact_compute_ratio"],
        bucket16_compute_ratio=dynamic["bucket16_compute_ratio"],
        bucket32_compute_ratio=dynamic["bucket32_compute_ratio"],
        compute_advantage_vs_k1=0.25048828125 - dynamic["bucket16_compute_ratio"],
        rank_counts=dynamic["rank_counts"],
    )
    payload = {
        "metadata": {
            "seed": args.seed,
            "steps": args.steps,
            "controller_steps": args.controller_steps,
            "controller_hidden": args.controller_hidden,
            "train_batches": args.train_batches,
            "tune_batches": args.tune_batches,
            "test_batches": args.test_batches,
            "n_experts": 64,
            "top_k": 8,
            "progressive_training_seconds": progressive_seconds,
            "elapsed_seconds": time.perf_counter() - started,
            "calibration_guard_ucb95_nats": 0.005,
            "methodological_note": "Lambda selected only on train-split calibration batches; validation batches are disjoint. Utility targets use final token cross-entropy under layerwise prefix interventions.",
        },
        "decision": make_decision(result),
        "controller_training": training_diagnostics,
        "test_result": asdict(result),
        "lambda_sweep": lambda_sweep,
    }
    write_outputs(args.output_dir, payload)
    print((args.output_dir / "VERDICT.md").read_text(encoding="utf-8"))
    del baseline, model, controllers
    gc.collect()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
