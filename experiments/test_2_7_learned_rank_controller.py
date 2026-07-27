#!/usr/bin/env python3
"""Test 2.7: learn a causal controller for nested Modal-MoE prefixes.

The controller is evaluated on an actually dynamic execution graph. It always
executes mode 0, observes only information available before the additional
matrix modes (normalized token state, router summaries, and cheap statistics of
mode-0 gate/up/hidden projections), and predicts a final rank K in {0,1,2,3}.

This closes an important gap in Test 2.6: the token oracle there compared
uniform-rank model passes independently. Here, predicted ranks are applied per
token and per layer inside one causal forward pass, so their downstream effects
on later tokens and layers are included in the measured validation loss.
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import random
import sys
import time
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


class OrdinalRankController(nn.Module):
    """Predict P(K >= 1), P(K >= 2), ... from cheap pilot features."""

    def __init__(self, feature_dim: int, max_rank: int, hidden_dim: int) -> None:
        super().__init__()
        self.max_rank = max_rank
        self.net = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, max_rank),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)

    @torch.no_grad()
    def predict_rank(self, features: torch.Tensor, threshold: float) -> torch.Tensor:
        probabilities = torch.sigmoid(self(features))
        # An ordinal model must satisfy P(K>=1) >= P(K>=2) >= ... .  The
        # cumulative minimum makes inference monotone without leaking labels.
        probabilities = torch.cummin(probabilities, dim=-1).values
        return torch.sum(probabilities >= threshold, dim=-1).to(torch.long)


def vector_statistics(values: torch.Tensor) -> torch.Tensor:
    return torch.stack(
        [
            torch.sqrt(torch.mean(values.square(), dim=-1) + 1e-8),
            torch.mean(values.abs(), dim=-1),
            torch.amax(values.abs(), dim=-1),
            torch.mean((values > 0).to(values.dtype), dim=-1),
        ],
        dim=-1,
    )


def pilot_features(
    flat: torch.Tensor,
    logits: torch.Tensor,
    top_values: torch.Tensor,
    gate_mode0: torch.Tensor,
    up_mode0: torch.Tensor,
) -> torch.Tensor:
    probabilities = F.softmax(logits, dim=-1)
    route_weights = F.softmax(top_values, dim=-1)
    entropy = -torch.sum(probabilities * torch.log(probabilities.clamp_min(1e-9)), dim=-1)
    entropy = entropy / math.log(probabilities.shape[-1])
    route_entropy = -torch.sum(route_weights * torch.log(route_weights.clamp_min(1e-9)), dim=-1)
    route_entropy = route_entropy / math.log(max(2, route_weights.shape[-1]))
    margin = route_weights[:, 0] - route_weights[:, 1] if route_weights.shape[-1] > 1 else route_weights[:, 0]
    router_summary = torch.stack(
        [
            entropy,
            route_entropy,
            route_weights[:, 0],
            margin,
            torch.std(logits, dim=-1, unbiased=False),
        ],
        dim=-1,
    )
    hidden_mode0 = F.silu(gate_mode0) * up_mode0
    return torch.cat(
        [
            flat,
            router_summary,
            vector_statistics(gate_mode0),
            vector_statistics(up_mode0),
            vector_statistics(hidden_mode0),
        ],
        dim=-1,
    )


def router_auxiliary_loss(logits: torch.Tensor, top_ids: torch.Tensor, n_experts: int) -> torch.Tensor:
    probabilities = F.softmax(logits, dim=-1)
    importance = probabilities.mean(dim=0)
    assignments = F.one_hot(top_ids, n_experts).float().mean(dim=(0, 1))
    balance = n_experts * torch.sum(importance * assignments)
    z_loss = torch.mean(torch.logsumexp(logits, dim=-1).square())
    return balance + 0.1 * z_loss


def dynamic_moe_forward(
    moe: nn.Module,
    x: torch.Tensor,
    *,
    controller: OrdinalRankController | None = None,
    threshold: float = 0.5,
    forced_ranks: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Execute different nested prefix ranks for different tokens.

    For measurement we materialize all mode projections and mask unused modes.
    The reported deployment compute is based on the selected prefix. A future
    kernel will bucket tokens after the mode-0 pilot and skip unused matrices.
    """
    shape = x.shape
    flat = x.reshape(-1, moe.d_model)
    logits = moe.router(flat)
    top_values, top_ids = torch.topk(logits, moe.top_k, dim=-1)
    route_weights = F.softmax(top_values, dim=-1)

    all_modes = moe.modal_rank + 1
    gate_projected = torch.einsum("nd,kmd->nkm", flat, moe.gate_modes[:all_modes])
    up_projected = torch.einsum("nd,kmd->nkm", flat, moe.up_modes[:all_modes])
    features = pilot_features(flat, logits, top_values, gate_projected[:, 0], up_projected[:, 0])

    if forced_ranks is not None:
        ranks = forced_ranks.reshape(-1).to(device=x.device, dtype=torch.long)
    elif controller is not None:
        ranks = controller.predict_rank(features, threshold)
    else:
        raise ValueError("controller or forced_ranks is required")
    ranks = ranks.clamp_(0, moe.modal_rank)

    gate_codes = moe.selected_codes(moe.gate_codes, moe.modal_rank)[top_ids]
    up_codes = moe.selected_codes(moe.up_codes, moe.modal_rank)[top_ids]
    down_codes = moe.selected_codes(moe.down_codes, moe.modal_rank)[top_ids]

    gate_contributions = gate_projected[:, None, :, :] * gate_codes
    up_contributions = up_projected[:, None, :, :] * up_codes
    gate_prefixes = torch.cumsum(gate_contributions, dim=2)
    up_prefixes = torch.cumsum(up_contributions, dim=2)
    gather_index = ranks[:, None, None, None].expand(-1, moe.top_k, 1, moe.d_ff)
    gate = torch.gather(gate_prefixes, dim=2, index=gather_index).squeeze(2)
    up = torch.gather(up_prefixes, dim=2, index=gather_index).squeeze(2)
    hidden = F.silu(gate) * up

    mode_mask = (
        torch.arange(all_modes, device=x.device)[None, :] <= ranks[:, None]
    ).to(hidden.dtype)
    mode_inputs = torch.einsum(
        "nt,ntkm,ntm,nk->nkm",
        route_weights,
        down_codes,
        hidden,
        mode_mask,
    )
    output = torch.einsum("kdm,nkm->nd", moe.down_modes[:all_modes], mode_inputs)
    aux = router_auxiliary_loss(logits, top_ids, moe.n_experts)
    return output.reshape(shape), aux, top_ids, ranks, features


def causal_mask(length: int, device: torch.device) -> torch.Tensor:
    mask = torch.full((length, length), float("-inf"), device=device)
    return torch.triu(mask, diagonal=1)


def forward_dynamic_model(
    model: nn.Module,
    tokens: torch.Tensor,
    controllers: Sequence[OrdinalRankController] | None,
    threshold: float,
    *,
    forced_rank: int | None = None,
    collect_features: bool = False,
) -> tuple[torch.Tensor, list[torch.Tensor], list[torch.Tensor]]:
    positions = torch.arange(tokens.shape[1], device=tokens.device)
    x = model.token_embedding(tokens) + model.position_embedding(positions)[None, :, :]
    mask = causal_mask(tokens.shape[1], tokens.device)
    rank_rows: list[torch.Tensor] = []
    feature_rows: list[torch.Tensor] = []

    for layer_index, block in enumerate(model.blocks):
        normalized = block.norm1(x)
        attention, _ = block.attn(
            normalized,
            normalized,
            normalized,
            attn_mask=mask,
            need_weights=False,
        )
        x = x + attention
        moe_input = block.norm2(x)
        if forced_rank is None:
            if controllers is None:
                raise ValueError("controllers required for dynamic inference")
            forced = None
            controller = controllers[layer_index]
        else:
            forced = torch.full(
                (tokens.numel(),),
                forced_rank,
                device=tokens.device,
                dtype=torch.long,
            )
            controller = None
        moe_output, _, _, ranks, features = dynamic_moe_forward(
            block.moe,
            moe_input,
            controller=controller,
            threshold=threshold,
            forced_ranks=forced,
        )
        x = x + moe_output
        rank_rows.append(ranks.reshape(tokens.shape))
        if collect_features:
            feature_rows.append(features)

    logits = model.output(model.norm(x))
    return logits, rank_rows, feature_rows


def token_losses(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        targets.reshape(-1),
        reduction="none",
    )


@torch.no_grad()
def oracle_labels(
    progressive: Any,
    model: nn.Module,
    tokens: torch.Tensor,
    targets: torch.Tensor,
    max_rank: int,
    tolerance: float,
) -> torch.Tensor:
    losses: dict[int, torch.Tensor] = {}
    for rank in range(max_rank + 1):
        progressive.set_active_rank(model, rank)
        logits, _, _ = model(tokens)
        losses[rank] = token_losses(logits, targets)
    full = losses[max_rank]
    chosen = torch.full_like(full, max_rank, dtype=torch.long)
    for rank in range(max_rank):
        eligible = losses[rank] <= full + tolerance
        chosen = torch.where((chosen == max_rank) & eligible, torch.full_like(chosen, rank), chosen)
    progressive.set_active_rank(model, max_rank)
    return chosen


@torch.no_grad()
def collect_controller_data(
    progressive: Any,
    model: nn.Module,
    dataset: Any,
    cfg: Any,
    *,
    split: str,
    batches: int,
    seed: int,
    max_rank: int,
    tolerance: float,
    keep_batches: bool,
) -> tuple[list[torch.Tensor], torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]]]:
    generator = torch.Generator().manual_seed(seed)
    feature_chunks: list[list[torch.Tensor]] = [[] for _ in model.blocks]
    label_chunks: list[torch.Tensor] = []
    stored_batches: list[tuple[torch.Tensor, torch.Tensor]] = []
    model.eval()
    for batch_index in range(batches):
        tokens, targets = dataset.batch(split, cfg.batch_size, generator)
        labels = oracle_labels(progressive, model, tokens, targets, max_rank, tolerance)
        _, _, features = forward_dynamic_model(
            model,
            tokens,
            controllers=None,
            threshold=0.5,
            forced_rank=max_rank,
            collect_features=True,
        )
        for layer_index, layer_features in enumerate(features):
            feature_chunks[layer_index].append(layer_features.detach().cpu())
        label_chunks.append(labels.detach().cpu())
        if keep_batches:
            stored_batches.append((tokens.detach().cpu(), targets.detach().cpu()))
        if batch_index == 0 or (batch_index + 1) % 8 == 0:
            print(f"collected {split} controller batch {batch_index + 1}/{batches}", flush=True)
    return (
        [torch.cat(chunks, dim=0) for chunks in feature_chunks],
        torch.cat(label_chunks, dim=0),
        stored_batches,
    )


def ordinal_targets(labels: torch.Tensor, max_rank: int) -> torch.Tensor:
    thresholds = torch.arange(1, max_rank + 1, device=labels.device)
    return (labels[:, None] >= thresholds[None, :]).to(torch.float32)


def train_controllers(
    feature_layers: Sequence[torch.Tensor],
    labels: torch.Tensor,
    *,
    max_rank: int,
    hidden_dim: int,
    steps: int,
    seed: int,
) -> tuple[nn.ModuleList, list[dict[str, float]]]:
    set_seed(seed)
    controllers = nn.ModuleList()
    diagnostics: list[dict[str, float]] = []
    targets = ordinal_targets(labels, max_rank)
    positives = targets.sum(dim=0)
    negatives = targets.shape[0] - positives
    pos_weight = (negatives / positives.clamp_min(1.0)).clamp(0.5, 8.0) * 1.15

    for layer_index, features in enumerate(feature_layers):
        controller = OrdinalRankController(features.shape[1], max_rank, hidden_dim)
        optimizer = torch.optim.AdamW(controller.parameters(), lr=3e-3, weight_decay=1e-3)
        generator = torch.Generator().manual_seed(seed + 1000 + layer_index)
        controller.train()
        final_loss = math.nan
        for step in range(1, steps + 1):
            indices = torch.randint(0, features.shape[0], (min(1024, features.shape[0]),), generator=generator)
            x = features.index_select(0, indices)
            y = targets.index_select(0, indices)
            logits = controller(x)
            bce = F.binary_cross_entropy_with_logits(logits, y, pos_weight=pos_weight)
            monotonic = F.relu(logits[:, 1:] - logits[:, :-1]).mean() if max_rank > 1 else torch.zeros(())
            loss = bce + 0.05 * monotonic
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(controller.parameters(), 1.0)
            optimizer.step()
            final_loss = float(loss.detach())
            if step == 1 or step % 200 == 0 or step == steps:
                print(f"controller layer={layer_index} step={step}/{steps} loss={final_loss:.4f}", flush=True)
        controller.eval()
        controllers.append(controller)
        diagnostics.append(
            {
                "layer": float(layer_index),
                "final_training_loss": final_loss,
                "parameters": float(sum(parameter.numel() for parameter in controller.parameters())),
            }
        )
    return controllers, diagnostics


def olmoe_controller_overhead(hidden_dim: int, max_rank: int, summary_dim: int = 17) -> float:
    d_model, d_ff, top_k = 2048, 1024, 8
    controller_macs = (d_model + summary_dim) * hidden_dim + hidden_dim * max_rank
    original_expert_macs = 3 * top_k * d_model * d_ff
    return controller_macs / original_expert_macs


def olmoe_compute_from_ranks(ranks: np.ndarray, controller_overhead: float) -> float:
    ranks64 = ranks.astype(np.float64)
    return float(np.mean((ranks64 + 1.0) / 8.0 + ranks64 / 2048.0) + controller_overhead)


@torch.no_grad()
def evaluate_dynamic(
    model: nn.Module,
    controllers: Sequence[OrdinalRankController],
    batches: Sequence[tuple[torch.Tensor, torch.Tensor]],
    threshold: float,
    controller_overhead: float,
) -> dict[str, Any]:
    model.eval()
    losses: list[float] = []
    all_ranks: list[np.ndarray] = []
    for tokens, targets in batches:
        logits, ranks, _ = forward_dynamic_model(model, tokens, controllers, threshold)
        losses.append(float(token_losses(logits, targets).mean()))
        all_ranks.extend(rank.detach().cpu().numpy().reshape(-1) for rank in ranks)
    rank_array = np.concatenate(all_ranks)
    unique, counts = np.unique(rank_array, return_counts=True)
    return {
        "threshold": float(threshold),
        "validation_loss": float(np.mean(losses)),
        "validation_perplexity": float(math.exp(min(float(np.mean(losses)), 20.0))),
        "mean_rank": float(np.mean(rank_array)),
        "p50_rank": float(np.percentile(rank_array, 50)),
        "p90_rank": float(np.percentile(rank_array, 90)),
        "p95_rank": float(np.percentile(rank_array, 95)),
        "rank_counts": {str(int(key)): int(value) for key, value in zip(unique, counts, strict=True)},
        "projected_olmoe_compute_ratio": olmoe_compute_from_ranks(rank_array, controller_overhead),
        "controller_overhead_ratio": float(controller_overhead),
    }


@torch.no_grad()
def evaluate_static(
    progressive: Any,
    model: nn.Module,
    batches: Sequence[tuple[torch.Tensor, torch.Tensor]],
    max_rank: int,
) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    model.eval()
    for rank in range(max_rank + 1):
        losses: list[float] = []
        for tokens, targets in batches:
            progressive.set_active_rank(model, rank)
            logits, _, _ = model(tokens)
            losses.append(float(token_losses(logits, targets).mean()))
        params = (rank + 1) / 64.0 + rank / 2048.0
        compute = (rank + 1) / 8.0 + rank / 2048.0
        rows.append(
            {
                "rank": float(rank),
                "validation_loss": float(np.mean(losses)),
                "validation_perplexity": float(math.exp(min(float(np.mean(losses)), 20.0))),
                "projected_olmoe_parameter_ratio": float(params),
                "projected_olmoe_compute_ratio": float(compute),
            }
        )
    progressive.set_active_rank(model, max_rank)
    return rows


@torch.no_grad()
def classification_diagnostics(
    controllers: Sequence[OrdinalRankController],
    feature_layers: Sequence[torch.Tensor],
    labels: torch.Tensor,
    threshold: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    labels_np = labels.numpy()
    for layer_index, (controller, features) in enumerate(zip(controllers, feature_layers, strict=True)):
        predicted = controller.predict_rank(features, threshold).cpu().numpy()
        matrix = np.zeros((controller.max_rank + 1, controller.max_rank + 1), dtype=np.int64)
        for actual, guess in zip(labels_np, predicted, strict=True):
            matrix[int(actual), int(guess)] += 1
        rows.append(
            {
                "layer": layer_index,
                "accuracy": float(np.mean(predicted == labels_np)),
                "mean_absolute_error": float(np.mean(np.abs(predicted - labels_np))),
                "underprediction_rate": float(np.mean(predicted < labels_np)),
                "overprediction_rate": float(np.mean(predicted > labels_np)),
                "mean_predicted_rank": float(np.mean(predicted)),
                "mean_oracle_rank": float(np.mean(labels_np)),
                "confusion_matrix": matrix.tolist(),
            }
        )
    return rows


def choose_threshold(
    model: nn.Module,
    controllers: Sequence[OrdinalRankController],
    tune_batches: Sequence[tuple[torch.Tensor, torch.Tensor]],
    static_rows: Sequence[dict[str, float]],
    controller_overhead: float,
) -> tuple[float, list[dict[str, Any]], float]:
    static_k1_loss = next(row["validation_loss"] for row in static_rows if int(row["rank"]) == 1)
    quality_limit = static_k1_loss + 0.005
    candidates: list[dict[str, Any]] = []
    for threshold in np.linspace(0.10, 0.90, 17):
        row = evaluate_dynamic(model, controllers, tune_batches, float(threshold), controller_overhead)
        row["quality_limit"] = quality_limit
        row["within_quality_limit"] = row["validation_loss"] <= quality_limit
        candidates.append(row)
        print(
            f"threshold={threshold:.2f} tune-loss={row['validation_loss']:.4f} "
            f"mean-K={row['mean_rank']:.3f} compute={row['projected_olmoe_compute_ratio']:.3%}",
            flush=True,
        )
    feasible = [row for row in candidates if row["within_quality_limit"]]
    if feasible:
        chosen = min(feasible, key=lambda row: (row["projected_olmoe_compute_ratio"], row["validation_loss"]))
    else:
        chosen = min(candidates, key=lambda row: (row["validation_loss"], row["projected_olmoe_compute_ratio"]))
    return float(chosen["threshold"]), candidates, float(quality_limit)


def make_decision(
    static_test: Sequence[dict[str, float]],
    dynamic_test: dict[str, Any],
) -> dict[str, Any]:
    static_k1 = next(row for row in static_test if int(row["rank"]) == 1)
    loss_delta = dynamic_test["validation_loss"] - static_k1["validation_loss"]
    compute_advantage = static_k1["projected_olmoe_compute_ratio"] - dynamic_test["projected_olmoe_compute_ratio"]
    if loss_delta <= 0.010 and compute_advantage >= 0.010:
        verdict = "LEARNED_CONTROLLER_PASS"
    elif loss_delta <= 0.020 and compute_advantage > 0.0:
        verdict = "BORDERLINE"
    else:
        verdict = "FAIL"
    return {
        "verdict": verdict,
        "dynamic_minus_static_k1_loss_nats": float(loss_delta),
        "compute_advantage_over_static_k1": float(compute_advantage),
        "rule": "PASS requires dynamic loss within 0.01 nat of static K1 and at least one percentage point lower projected OLMoE expert compute, including controller MLP overhead.",
    }


def self_test(progressive: Any) -> None:
    set_seed(17)
    moe = progressive.ProgressiveNeuronwiseMoE(12, 20, 7, 3, 3)
    x = torch.randn(2, 5, 12)
    max_error = 0.0
    for rank in range(4):
        moe.active_rank = rank
        reference, _, _ = moe(x)
        forced = torch.full((x.shape[0] * x.shape[1],), rank, dtype=torch.long)
        dynamic, _, _, observed, features = dynamic_moe_forward(moe, x, forced_ranks=forced)
        error = float(torch.max(torch.abs(reference - dynamic)))
        max_error = max(max_error, error)
        if not torch.equal(observed, forced) or features.shape[0] != forced.shape[0]:
            raise AssertionError(rank)
    if max_error > 2e-5:
        raise AssertionError(f"dynamic prefix mismatch: {max_error}")
    controller = OrdinalRankController(features.shape[1], 3, 16)
    ranks = controller.predict_rank(features, 0.5)
    if ranks.min() < 0 or ranks.max() > 3:
        raise AssertionError(ranks)
    controller(features).sum().backward()
    if any(parameter.grad is None for parameter in controller.parameters()):
        raise AssertionError("missing controller gradients")
    print(f"self-test passed; maximum static/dynamic prefix error={max_error:.3e}")


def write_outputs(output_dir: Path, payload: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "learned_controller.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with (output_dir / "static_prefixes.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(payload["static_test"][0].keys()))
        writer.writeheader()
        writer.writerows(payload["static_test"])
    with (output_dir / "threshold_sweep.csv").open("w", newline="", encoding="utf-8") as handle:
        flat_rows = []
        for row in payload["threshold_sweep"]:
            flat_rows.append({key: value for key, value in row.items() if key != "rank_counts"})
        writer = csv.DictWriter(handle, fieldnames=list(flat_rows[0].keys()))
        writer.writeheader()
        writer.writerows(flat_rows)

    d = payload["decision"]
    dynamic = payload["dynamic_test"]
    lines = [
        "# Test 2.7 — learned causal rank controller",
        "",
        f"**Decision:** **{d['verdict']}**",
        "",
        f"Chosen threshold: `{payload['chosen_threshold']:.2f}` (selected only on training-calibration batches).",
        "",
        "| Policy | Validation loss | Mean K | p95 K | Projected OLMoE compute |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in payload["static_test"]:
        lines.append(
            f"| Static K={int(row['rank'])} | {row['validation_loss']:.4f} | {int(row['rank'])} | {int(row['rank'])} | {row['projected_olmoe_compute_ratio']:.3%} |"
        )
    lines.append(
        f"| **Learned dynamic** | **{dynamic['validation_loss']:.4f}** | **{dynamic['mean_rank']:.3f}** | **{dynamic['p95_rank']:.0f}** | **{dynamic['projected_olmoe_compute_ratio']:.3%}** |"
    )
    lines += [
        "",
        f"- Dynamic minus static-K1 loss: `{d['dynamic_minus_static_k1_loss_nats']:+.4f}` nat.",
        f"- Compute advantage over static K1: `{d['compute_advantage_over_static_k1']:+.3%}` of original OLMoE expert projections.",
        f"- Rank counts across all tokens and layers: `{dynamic['rank_counts']}`.",
        f"- Controller-only MLP overhead estimate: `{dynamic['controller_overhead_ratio']:.3%}` of original OLMoE expert projection MACs.",
        "",
        "The controller sees the local normalized token state, router summaries, and statistics of mode 0 only. Additional modes are selected before their projections. The validation pass actually applies different ranks per token and layer; it is not an oracle recombination of independent uniform-rank passes.",
    ]
    (output_dir / "VERDICT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--progressive-source", type=Path, required=True)
    parser.add_argument("--text", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("results/test-2-7/latest"))
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=9090)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--controller-train-batches", type=int, default=32)
    parser.add_argument("--controller-tune-batches", type=int, default=8)
    parser.add_argument("--test-batches", type=int, default=20)
    parser.add_argument("--controller-steps", type=int, default=600)
    parser.add_argument("--controller-hidden", type=int, default=64)
    parser.add_argument("--oracle-tolerance", type=float, default=0.05)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args(argv)

    source = load_module("controller_base_source", args.source)
    progressive = load_module("controller_progressive_source", args.progressive_source)
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    if args.self_test:
        self_test(progressive)
        return 0

    started = time.perf_counter()
    max_rank = 3
    cfg = source.Config(
        steps=args.steps,
        eval_interval=100,
        eval_batches=20,
        batch_size=16,
        seq_len=64,
    )
    dataset = source.CharDataset(args.text.read_text(encoding="utf-8"), cfg.seq_len)

    baseline = progressive.train_baseline(source, dataset, cfg, args.seed)
    progressive_model, progressive_seconds = progressive.train_progressive(
        source, dataset, cfg, args.seed + 1, max_rank
    )

    train_features, train_labels, _ = collect_controller_data(
        progressive,
        progressive_model,
        dataset,
        cfg,
        split="train",
        batches=args.controller_train_batches,
        seed=args.seed + 100,
        max_rank=max_rank,
        tolerance=args.oracle_tolerance,
        keep_batches=False,
    )
    tune_features, tune_labels, tune_batches = collect_controller_data(
        progressive,
        progressive_model,
        dataset,
        cfg,
        split="train",
        batches=args.controller_tune_batches,
        seed=args.seed + 200,
        max_rank=max_rank,
        tolerance=args.oracle_tolerance,
        keep_batches=True,
    )
    test_features, test_labels, test_batches = collect_controller_data(
        progressive,
        progressive_model,
        dataset,
        cfg,
        split="validation",
        batches=args.test_batches,
        seed=args.seed + 300,
        max_rank=max_rank,
        tolerance=args.oracle_tolerance,
        keep_batches=True,
    )

    controllers, controller_training = train_controllers(
        train_features,
        train_labels,
        max_rank=max_rank,
        hidden_dim=args.controller_hidden,
        steps=args.controller_steps,
        seed=args.seed + 400,
    )
    overhead = olmoe_controller_overhead(args.controller_hidden, max_rank)
    static_tune = evaluate_static(progressive, progressive_model, tune_batches, max_rank)
    chosen_threshold, threshold_sweep, tune_quality_limit = choose_threshold(
        progressive_model,
        controllers,
        tune_batches,
        static_tune,
        overhead,
    )
    static_test = evaluate_static(progressive, progressive_model, test_batches, max_rank)
    dynamic_test = evaluate_dynamic(
        progressive_model,
        controllers,
        test_batches,
        chosen_threshold,
        overhead,
    )
    classification = classification_diagnostics(
        controllers,
        test_features,
        test_labels,
        chosen_threshold,
    )
    baseline_rows = evaluate_static(progressive, baseline, test_batches, 0)
    baseline_loss = baseline_rows[0]["validation_loss"]
    decision = make_decision(static_test, dynamic_test)

    payload = {
        "metadata": {
            "seed": args.seed,
            "training_steps": args.steps,
            "progressive_training_seconds": progressive_seconds,
            "total_elapsed_seconds": time.perf_counter() - started,
            "controller_train_batches": args.controller_train_batches,
            "controller_tune_batches": args.controller_tune_batches,
            "test_batches": args.test_batches,
            "oracle_tolerance_nats": args.oracle_tolerance,
            "controller_hidden": args.controller_hidden,
            "controller_feature_dimension": int(train_features[0].shape[1]),
            "baseline_validation_loss": baseline_loss,
            "tune_quality_limit": tune_quality_limit,
            "methodological_note": "Threshold selection uses training-calibration data only; final metrics use the validation split. Dynamic ranks are executed inside one causal pass.",
        },
        "decision": decision,
        "chosen_threshold": chosen_threshold,
        "controller_training": controller_training,
        "controller_classification": classification,
        "train_oracle_distribution": {
            str(rank): int(torch.sum(train_labels == rank)) for rank in range(max_rank + 1)
        },
        "tune_oracle_distribution": {
            str(rank): int(torch.sum(tune_labels == rank)) for rank in range(max_rank + 1)
        },
        "test_oracle_distribution": {
            str(rank): int(torch.sum(test_labels == rank)) for rank in range(max_rank + 1)
        },
        "static_tune": static_tune,
        "threshold_sweep": threshold_sweep,
        "static_test": static_test,
        "dynamic_test": dynamic_test,
    }
    write_outputs(args.output_dir, payload)
    print((args.output_dir / "VERDICT.md").read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
