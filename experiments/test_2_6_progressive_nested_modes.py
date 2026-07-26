#!/usr/bin/env python3
"""Train nested neuron-wise modes and measure the token-level rank oracle."""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import random
import sys
import time
from dataclasses import asdict, dataclass
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


class ProgressiveNeuronwiseMoE(nn.Module):
    def __init__(self, d_model: int, d_ff: int, n_experts: int, top_k: int, modal_rank: int) -> None:
        super().__init__()
        if modal_rank < 1:
            raise ValueError("max modal rank must be positive")
        self.d_model = d_model
        self.d_ff = d_ff
        self.n_experts = n_experts
        self.top_k = top_k
        self.modal_rank = modal_rank
        self.n_modes = modal_rank + 1
        self.active_rank = modal_rank
        self.router = nn.Linear(d_model, n_experts, bias=False)
        self.gate_modes = nn.Parameter(torch.empty(self.n_modes, d_ff, d_model))
        self.up_modes = nn.Parameter(torch.empty(self.n_modes, d_ff, d_model))
        self.down_modes = nn.Parameter(torch.empty(self.n_modes, d_model, d_ff))
        self.gate_codes = nn.Parameter(torch.empty(n_experts, modal_rank, d_ff))
        self.up_codes = nn.Parameter(torch.empty(n_experts, modal_rank, d_ff))
        self.down_codes = nn.Parameter(torch.empty(n_experts, modal_rank, d_ff))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.router.weight, std=0.02)
        for bank in (self.gate_modes, self.up_modes, self.down_modes):
            for index, mode in enumerate(bank):
                nn.init.xavier_uniform_(mode)
                if index > 0:
                    mode.data.mul_(0.5 / math.sqrt(self.modal_rank))
        for codes in (self.gate_codes, self.up_codes, self.down_codes):
            nn.init.normal_(codes, std=0.7)

    def selected_codes(self, codes: torch.Tensor, rank: int) -> torch.Tensor:
        ones = torch.ones(codes.shape[0], 1, self.d_ff, dtype=codes.dtype, device=codes.device)
        if rank == 0:
            return ones
        return torch.cat([ones, codes[:, :rank]], dim=1)

    def forward(self, x: torch.Tensor):
        shape = x.shape
        flat = x.reshape(-1, self.d_model)
        logits = self.router(flat)
        top_values, top_ids = torch.topk(logits, self.top_k, dim=-1)
        route_weights = F.softmax(top_values, dim=-1)
        rank = int(max(0, min(self.active_rank, self.modal_rank)))
        modes = rank + 1
        gate_projected = torch.einsum("nd,kmd->nkm", flat, self.gate_modes[:modes])
        up_projected = torch.einsum("nd,kmd->nkm", flat, self.up_modes[:modes])
        selected_gate = self.selected_codes(self.gate_codes, rank)[top_ids]
        selected_up = self.selected_codes(self.up_codes, rank)[top_ids]
        selected_down = self.selected_codes(self.down_codes, rank)[top_ids]
        gate = torch.einsum("nkm,ntkm->ntm", gate_projected, selected_gate)
        up = torch.einsum("nkm,ntkm->ntm", up_projected, selected_up)
        hidden = F.silu(gate) * up
        mode_inputs = torch.einsum("nt,ntkm,ntm->nkm", route_weights, selected_down, hidden)
        output = torch.einsum("kdm,nkm->nd", self.down_modes[:modes], mode_inputs)
        probabilities = F.softmax(logits, dim=-1)
        importance = probabilities.mean(dim=0)
        assignments = F.one_hot(top_ids, self.n_experts).float().mean(dim=(0, 1))
        balance = self.n_experts * torch.sum(importance * assignments)
        z_loss = torch.mean(torch.logsumexp(logits, dim=-1) ** 2)
        return output.reshape(shape), balance + 0.1 * z_loss, top_ids


@dataclass
class PrefixResult:
    active_rank: int
    modes_executed: int
    validation_loss: float
    validation_perplexity: float
    loss_ratio_to_baseline: float
    loss_ratio_to_full_prefix: float
    projected_olmoe_parameter_ratio: float
    projected_olmoe_compute_ratio: float


def set_active_rank(model: nn.Module, rank: int) -> None:
    for block in model.blocks:
        if hasattr(block.moe, "active_rank"):
            block.moe.active_rank = rank


def olmoe_ratios(rank: int, d_model: int = 2048, d_ff: int = 1024, experts: int = 64, top_k: int = 8) -> tuple[float, float]:
    params = (rank + 1) / experts + rank / d_model
    compute = (rank + 1) / top_k + rank / d_model
    return params, compute


@torch.no_grad()
def evaluate_prefix(source, model, dataset, cfg, rank: int, seed: int, batches: int = 20) -> tuple[float, list[np.ndarray]]:
    set_active_rank(model, rank)
    model.eval()
    generator = torch.Generator().manual_seed(seed)
    losses = []
    token_losses: list[np.ndarray] = []
    for _ in range(batches):
        x, y = dataset.batch("validation", cfg.batch_size, generator)
        logits, _, _ = model(x)
        per_token = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1), reduction="none")
        losses.append(float(per_token.mean()))
        token_losses.append(per_token.cpu().numpy())
    model.train()
    return float(np.mean(losses)), token_losses


def train_baseline(source, dataset, cfg, seed: int):
    source.set_seed(seed)
    model = source.LanguageModel(len(dataset.vocab), cfg, "baseline", None)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    generator = torch.Generator().manual_seed(seed + 101)
    for step in range(1, cfg.steps + 1):
        x, y = dataset.batch("train", cfg.batch_size, generator)
        logits, aux, _ = model(x)
        ce = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))
        loss = ce + cfg.aux_weight * aux
        optimizer.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip); optimizer.step()
        if step == 1 or step % cfg.eval_interval == 0 or step == cfg.steps:
            print(f"baseline step={step}/{cfg.steps} ce={float(ce):.4f}", flush=True)
    return model


def train_progressive(source, dataset, cfg, seed: int, max_rank: int):
    source.set_seed(seed)
    source.ModalMoE = ProgressiveNeuronwiseMoE
    model = source.LanguageModel(len(dataset.vocab), cfg, "modal", max_rank)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    generator = torch.Generator().manual_seed(seed + 202)
    prefix_rng = random.Random(seed + 303)
    started = time.perf_counter()
    for step in range(1, cfg.steps + 1):
        x, y = dataset.batch("train", cfg.batch_size, generator)
        sampled_rank = prefix_rng.randrange(max_rank + 1)
        set_active_rank(model, max_rank)
        full_logits, full_aux, _ = model(x)
        full_ce = F.cross_entropy(full_logits.reshape(-1, full_logits.shape[-1]), y.reshape(-1))
        if sampled_rank == max_rank:
            sampled_ce, sampled_aux = full_ce, full_aux
        else:
            set_active_rank(model, sampled_rank)
            sampled_logits, sampled_aux, _ = model(x)
            sampled_ce = F.cross_entropy(sampled_logits.reshape(-1, sampled_logits.shape[-1]), y.reshape(-1))
        cost_weight = (sampled_rank + 1) / (max_rank + 1)
        loss = 0.55 * full_ce + 0.45 * sampled_ce + cfg.aux_weight * 0.5 * (full_aux + sampled_aux) + 0.002 * cost_weight
        optimizer.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip); optimizer.step()
        if step == 1 or step % cfg.eval_interval == 0 or step == cfg.steps:
            print(f"progressive step={step}/{cfg.steps} full={float(full_ce):.4f} sampled-rank={sampled_rank} sampled={float(sampled_ce):.4f}", flush=True)
    set_active_rank(model, max_rank)
    return model, time.perf_counter() - started


def oracle_distribution(losses_by_rank: dict[int, list[np.ndarray]], max_rank: int, tolerance: float) -> dict[str, Any]:
    flattened = {rank: np.concatenate(chunks) for rank, chunks in losses_by_rank.items()}
    full = flattened[max_rank]
    chosen = np.full(full.shape, max_rank, dtype=np.int32)
    for rank in range(max_rank):
        eligible = flattened[rank] <= full + tolerance
        chosen[(chosen == max_rank) & eligible] = rank
    counts = {str(rank): int(np.sum(chosen == rank)) for rank in range(max_rank + 1)}
    return {
        "tolerance_nats": tolerance,
        "tokens": int(chosen.size),
        "mean_rank": float(np.mean(chosen)),
        "mean_modes": float(np.mean(chosen + 1)),
        "p50_rank": float(np.percentile(chosen, 50)),
        "p90_rank": float(np.percentile(chosen, 90)),
        "p95_rank": float(np.percentile(chosen, 95)),
        "counts": counts,
        "projected_olmoe_mean_compute_ratio": float(np.mean((chosen + 1) / 8.0 + chosen / 2048.0)),
    }


def make_decision(prefixes: list[dict[str, Any]], oracles: list[dict[str, Any]]) -> dict[str, Any]:
    indexed = {int(row["active_rank"]): row for row in prefixes}
    oracle = next(item for item in oracles if abs(item["tolerance_nats"] - 0.05) < 1e-9)
    if indexed[1]["loss_ratio_to_baseline"] <= 1.05 and oracle["mean_rank"] <= 1.5 and oracle["p95_rank"] <= 3:
        verdict = "PROGRESSIVE_PASS"
    elif indexed[2]["loss_ratio_to_baseline"] <= 1.05 and oracle["mean_rank"] <= 2.25:
        verdict = "BORDERLINE"
    else:
        verdict = "FAIL"
    return {"verdict": verdict, "rule": "PASS requires K1 within 5% of baseline and token oracle mean rank <=1.5 at 0.05 nat tolerance."}


def self_test(source) -> None:
    source.set_seed(11)
    layer = ProgressiveNeuronwiseMoE(12, 20, 7, 3, 3)
    x = torch.randn(2, 4, 12)
    outputs = []
    for rank in range(4):
        layer.active_rank = rank
        output, _, _ = layer(x)
        if output.shape != x.shape or not torch.isfinite(output).all():
            raise AssertionError(rank)
        outputs.append(output)
    outputs[-1].sum().backward()
    if any(parameter.grad is None for parameter in layer.parameters()):
        raise AssertionError("missing gradients")
    print("self-test passed for ranks 0..3")


def write_outputs(output_dir: Path, payload: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "progressive_modes.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with (output_dir / "prefixes.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(payload["prefixes"][0].keys()))
        writer.writeheader(); writer.writerows(payload["prefixes"])
    lines = [
        "# Test 2.6 — nested progressive modes and token oracle", "", f"**Decision:** **{payload['decision']['verdict']}**", "",
        "| Active K | Modes | Validation loss | Loss/full baseline | Loss/full prefix | Projected OLMoE params | Projected OLMoE compute |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["prefixes"]:
        lines.append(f"| {row['active_rank']} | {row['modes_executed']} | {row['validation_loss']:.4f} | {row['loss_ratio_to_baseline']:.3f}× | {row['loss_ratio_to_full_prefix']:.3f}× | {row['projected_olmoe_parameter_ratio']:.3%} | {row['projected_olmoe_compute_ratio']:.3%} |")
    lines += ["", "## Token-level oracle"]
    for oracle in payload["oracles"]:
        lines.append(f"- tolerance `{oracle['tolerance_nats']:.2f}` nat: mean K `{oracle['mean_rank']:.3f}`, p95 `{oracle['p95_rank']:.0f}`, projected mean OLMoE compute `{oracle['projected_olmoe_mean_compute_ratio']:.3%}`, counts `{oracle['counts']}`.")
    lines += ["", "The oracle compares per-token cross-entropy of nested prefixes from the same checkpoint. It is an upper bound on a learned controller and does not include controller or scheduling overhead."]
    (output_dir / "VERDICT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--text", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("results/progressive_modes"))
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=8080)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args(argv)
    source = load_module("progressive_source", args.source)
    source.torch.set_num_threads(args.threads); source.torch.set_num_interop_threads(1)
    if args.self_test:
        self_test(source); return 0
    max_rank = 3
    cfg = source.Config(steps=args.steps, eval_interval=100, eval_batches=20, batch_size=16, seq_len=64)
    dataset = source.CharDataset(args.text.read_text(encoding="utf-8"), cfg.seq_len)
    baseline = train_baseline(source, dataset, cfg, args.seed)
    baseline_loss, _ = evaluate_prefix(source, baseline, dataset, cfg, 0, args.seed + 900, 20)
    progressive, elapsed = train_progressive(source, dataset, cfg, args.seed + 1, max_rank)
    losses_by_rank: dict[int, list[np.ndarray]] = {}
    prefix_rows: list[dict[str, Any]] = []
    full_loss = None
    raw_losses: dict[int, float] = {}
    for rank in range(max_rank + 1):
        loss, token_losses = evaluate_prefix(source, progressive, dataset, cfg, rank, args.seed + 901, 20)
        raw_losses[rank] = loss; losses_by_rank[rank] = token_losses
    full_loss = raw_losses[max_rank]
    for rank in range(max_rank + 1):
        params, compute = olmoe_ratios(rank)
        prefix_rows.append(asdict(PrefixResult(
            active_rank=rank, modes_executed=rank + 1, validation_loss=raw_losses[rank],
            validation_perplexity=float(math.exp(min(raw_losses[rank], 20.0))),
            loss_ratio_to_baseline=raw_losses[rank] / baseline_loss,
            loss_ratio_to_full_prefix=raw_losses[rank] / full_loss,
            projected_olmoe_parameter_ratio=params, projected_olmoe_compute_ratio=compute,
        )))
    oracles = [oracle_distribution(losses_by_rank, max_rank, tolerance) for tolerance in (0.02, 0.05, 0.10)]
    payload = {"metadata": {"max_rank": max_rank, "steps": args.steps, "seed": args.seed, "baseline_loss": baseline_loss, "progressive_training_seconds": elapsed}, "prefixes": prefix_rows, "oracles": oracles}
    payload["decision"] = make_decision(prefix_rows, oracles)
    write_outputs(args.output_dir, payload)
    print((args.output_dir / "VERDICT.md").read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
