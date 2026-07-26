#!/usr/bin/env python3
"""Train a more expressive neuron-wise modal MoE from initialization.

For each shared mode, every expert owns a vector of intermediate-neuron codes:

    G_e = sum_k Diag(a^G[e,k]) G_k
    U_e = sum_k Diag(a^U[e,k]) U_k
    D_e = sum_k D_k Diag(a^D[e,k])

The shared matrices are still multiplied only once per token. Expert-specific
codes are applied elementwise, and the down modes are evaluated after router-
weighted aggregation. For OLMoE dimensions, K=1 costs roughly 3.17% of expert
matrix parameters and 25.05% of top-8 projection arithmetic.
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


def load_source(path: Path):
    spec = importlib.util.spec_from_file_location("modal_source_neuronwise", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class NeuronwiseModalMoE(nn.Module):
    def __init__(self, d_model: int, d_ff: int, n_experts: int, top_k: int, modal_rank: int) -> None:
        super().__init__()
        if modal_rank < 0:
            raise ValueError("modal_rank must be non-negative")
        self.d_model = d_model
        self.d_ff = d_ff
        self.n_experts = n_experts
        self.top_k = top_k
        self.modal_rank = modal_rank
        self.n_modes = modal_rank + 1
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
                    mode.data.mul_(0.5 / math.sqrt(max(self.modal_rank, 1)))
        for codes in (self.gate_codes, self.up_codes, self.down_codes):
            if codes.numel():
                nn.init.normal_(codes, std=0.7)

    def expanded_codes(self, codes: torch.Tensor) -> torch.Tensor:
        ones = torch.ones(codes.shape[0], 1, self.d_ff, dtype=codes.dtype, device=codes.device)
        return torch.cat([ones, codes], dim=1)

    def forward(self, x: torch.Tensor):
        shape = x.shape
        flat = x.reshape(-1, self.d_model)
        logits = self.router(flat)
        top_values, top_ids = torch.topk(logits, self.top_k, dim=-1)
        route_weights = F.softmax(top_values, dim=-1)

        gate_projected = torch.einsum("nd,kmd->nkm", flat, self.gate_modes)
        up_projected = torch.einsum("nd,kmd->nkm", flat, self.up_modes)
        gate_code_bank = self.expanded_codes(self.gate_codes)
        up_code_bank = self.expanded_codes(self.up_codes)
        down_code_bank = self.expanded_codes(self.down_codes)
        selected_gate_codes = gate_code_bank[top_ids]
        selected_up_codes = up_code_bank[top_ids]
        selected_down_codes = down_code_bank[top_ids]

        gate = torch.einsum("nkm,ntkm->ntm", gate_projected, selected_gate_codes)
        up = torch.einsum("nkm,ntkm->ntm", up_projected, selected_up_codes)
        hidden = F.silu(gate) * up
        mode_inputs = torch.einsum("nt,ntkm,ntm->nkm", route_weights, selected_down_codes, hidden)
        output = torch.einsum("kdm,nkm->nd", self.down_modes, mode_inputs)
        aux = self.router_auxiliary_loss(logits, top_ids)
        return output.reshape(shape), aux, top_ids

    def router_auxiliary_loss(self, logits: torch.Tensor, top_ids: torch.Tensor) -> torch.Tensor:
        probabilities = F.softmax(logits, dim=-1)
        importance = probabilities.mean(dim=0)
        assignments = F.one_hot(top_ids, self.n_experts).float().mean(dim=(0, 1))
        balance = self.n_experts * torch.sum(importance * assignments)
        z_loss = torch.mean(torch.logsumexp(logits, dim=-1) ** 2)
        return balance + 0.1 * z_loss

    @torch.no_grad()
    def reference_reconstructed(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        flat = x.reshape(-1, self.d_model)
        logits = self.router(flat)
        top_values, top_ids = torch.topk(logits, self.top_k, dim=-1)
        route_weights = F.softmax(top_values, dim=-1)
        cg = self.expanded_codes(self.gate_codes)
        cu = self.expanded_codes(self.up_codes)
        cd = self.expanded_codes(self.down_codes)
        output = torch.zeros_like(flat)
        for expert_id in range(self.n_experts):
            where = torch.nonzero(top_ids == expert_id, as_tuple=False)
            if where.numel() == 0:
                continue
            token_ids, slots = where[:, 0], where[:, 1]
            gate_weight = torch.einsum("km,kmd->md", cg[expert_id], self.gate_modes)
            up_weight = torch.einsum("km,kmd->md", cu[expert_id], self.up_modes)
            down_weight = torch.einsum("km,kdm->dm", cd[expert_id], self.down_modes)
            selected = flat.index_select(0, token_ids)
            hidden = F.silu(F.linear(selected, gate_weight)) * F.linear(selected, up_weight)
            expert_out = F.linear(hidden, down_weight)
            expert_out = expert_out * route_weights[token_ids, slots].unsqueeze(-1)
            output.index_add_(0, token_ids, expert_out)
        return output.reshape(shape)


def exact_ratios(kind: str, rank: int, d_model: int, d_ff: int, n_experts: int, top_k: int) -> tuple[float, float]:
    original_parameters = 3 * n_experts * d_ff * d_model
    original_compute = 3 * top_k * d_ff * d_model
    shared = 3 * (rank + 1) * d_ff * d_model
    if kind == "neuronwise":
        code_parameters = 3 * n_experts * rank * d_ff
        code_compute = 3 * top_k * rank * d_ff
    else:
        code_parameters = 3 * n_experts * rank
        code_compute = 3 * top_k * rank
    return (shared + code_parameters) / original_parameters, (shared + code_compute) / original_compute


def run_variant(source, dataset, seed: int, steps: int, label: str, implementation, rank: int | None):
    source.ModalMoE = implementation
    cfg = source.Config(steps=steps, eval_batches=10, eval_interval=100, batch_size=16, seq_len=64)
    raw_variant = "baseline" if rank is None else "modal"
    result, history = source.train_variant(raw_variant, rank, seed, dataset, cfg)
    result.variant = label
    if rank is not None:
        kind = "neuronwise" if implementation is NeuronwiseModalMoE else "scalar"
        result.expert_parameter_ratio, result.idealized_expert_compute_ratio = exact_ratios(kind, rank, cfg.d_model, cfg.d_ff, cfg.n_experts, cfg.top_k)
    return result, history


def make_decision(results: list[dict[str, Any]]) -> dict[str, Any]:
    indexed = {row["variant"]: row for row in results}
    base = indexed["baseline"]["final_validation_loss"]
    ratios = {name: row["final_validation_loss"] / base for name, row in indexed.items()}
    improvement_k1 = ratios["scalar-k1"] - ratios["neuronwise-k1"]
    improvement_k2 = ratios["scalar-k2"] - ratios["neuronwise-k2"]
    if ratios["neuronwise-k1"] <= 1.02 and improvement_k1 >= 0.002:
        verdict = "PASS_K1"
    elif ratios["neuronwise-k2"] <= 1.015:
        verdict = "PASS_K2"
    elif ratios["neuronwise-k1"] <= ratios["scalar-k1"] and ratios["neuronwise-k2"] <= ratios["scalar-k2"]:
        verdict = "BORDERLINE"
    else:
        verdict = "NO_GAIN"
    olmoe = {
        "k1_parameter_ratio": exact_ratios("neuronwise", 1, 2048, 1024, 64, 8)[0],
        "k1_compute_ratio": exact_ratios("neuronwise", 1, 2048, 1024, 64, 8)[1],
        "k2_parameter_ratio": exact_ratios("neuronwise", 2, 2048, 1024, 64, 8)[0],
        "k2_compute_ratio": exact_ratios("neuronwise", 2, 2048, 1024, 64, 8)[1],
    }
    return {"verdict": verdict, "loss_ratios": ratios, "neuronwise_k1_advantage": improvement_k1, "neuronwise_k2_advantage": improvement_k2, "olmoe_budget_projection": olmoe}


def self_test() -> None:
    torch.manual_seed(13)
    layer = NeuronwiseModalMoE(12, 20, 7, 3, 2)
    layer.eval()
    x = torch.randn(3, 5, 12)
    fused, _, _ = layer(x)
    reference = layer.reference_reconstructed(x)
    error = float(torch.linalg.vector_norm(fused - reference) / torch.linalg.vector_norm(reference))
    if error > 3e-6:
        raise AssertionError(error)
    fused.sum().backward()
    if any(parameter.grad is None for parameter in layer.parameters()):
        raise AssertionError("missing gradient")
    print(f"self-test passed: fused/reconstructed error={error:.3e}")


def write_outputs(output_dir: Path, payload: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "neuronwise_modal_trainability.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with (output_dir / "results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(payload["results"][0].keys()))
        writer.writeheader(); writer.writerows(payload["results"])
    base = next(row for row in payload["results"] if row["variant"] == "baseline")["final_validation_loss"]
    lines = [
        "# Test 2.2 — neuron-wise modal codes trained from initialization", "", f"**Decision:** **{payload['decision']['verdict']}**", "",
        "| Variant | Expert params | Ideal expert compute | Validation loss | Loss/full | Parameters |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in payload["results"]:
        lines.append(f"| {row['variant']} | {row['expert_parameter_ratio']:.2%} | {row['idealized_expert_compute_ratio']:.2%} | {row['final_validation_loss']:.4f} | {row['final_validation_loss']/base:.3f}× | {row['trainable_parameters']:,} |")
    budget = payload["decision"]["olmoe_budget_projection"]
    lines += [
        "", f"Projected OLMoE K=1 budget: **{budget['k1_parameter_ratio']:.3%} parameters**, **{budget['k1_compute_ratio']:.3%} ideal expert compute**.",
        f"Projected OLMoE K=2 budget: **{budget['k2_parameter_ratio']:.3%} parameters**, **{budget['k2_compute_ratio']:.3%} ideal expert compute**.",
        "", "The extra expressivity comes from elementwise expert codes; the number of shared matrix multiplications is unchanged. The down branch remains aggregated before its shared matrices.",
    ]
    (output_dir / "VERDICT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--source", type=Path)
    parser.add_argument("--text", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("results/neuronwise_modal_trainability"))
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args(argv)
    if args.self_test:
        self_test(); return 0
    if args.source is None or args.text is None:
        parser.error("--source and --text are required")
    source = load_source(args.source)
    scalar_class = source.ModalMoE
    source.torch.set_num_threads(args.threads)
    source.torch.set_num_interop_threads(1)
    dataset = source.CharDataset(args.text.read_text(encoding="utf-8"), 64)
    variants = [
        ("baseline", scalar_class, None),
        ("scalar-k1", scalar_class, 1),
        ("scalar-k2", scalar_class, 2),
        ("neuronwise-k1", NeuronwiseModalMoE, 1),
        ("neuronwise-k2", NeuronwiseModalMoE, 2),
    ]
    results = []
    histories = {}
    for label, implementation, rank in variants:
        result, history = run_variant(source, dataset, args.seed, args.steps, label, implementation, rank)
        results.append(asdict(result)); histories[label] = history
    payload = {"metadata": {"task": "Tiny Shakespeare character LM", "steps": args.steps, "seed": args.seed}, "decision": make_decision(results), "results": results, "history": histories}
    write_outputs(args.output_dir, payload)
    print((args.output_dir / "VERDICT.md").read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
