#!/usr/bin/env python3
"""Can a modal MoE learn when the shared-mode constraint is imposed during training?

The post-training probes show that unconstrained OLMoE experts are not linearly
compressible in raw weight space. That does not imply the constrained
architecture is untrainable: optimization may choose a coordinated expert
basis if it is present from initialization.

This experiment compares a conventional sparse SwiGLU MoE against modal MoEs
trained from scratch on the same character-language-model task. The modal
layer executes the shared gate/up modes once per token and applies each shared
``down`` mode once after weighted aggregation; expert-specific state consists
only of scalar mode coefficients.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


@dataclass
class Config:
    seq_len: int = 64
    batch_size: int = 16
    d_model: int = 96
    n_heads: int = 4
    n_layers: int = 2
    d_ff: int = 128
    n_experts: int = 16
    top_k: int = 4
    steps: int = 400
    eval_interval: int = 100
    eval_batches: int = 20
    lr: float = 4e-4
    weight_decay: float = 0.05
    aux_weight: float = 0.01
    z_loss_weight: float = 1e-3
    grad_clip: float = 1.0


@dataclass
class Result:
    variant: str
    seed: int
    modal_rank: int | None
    trainable_parameters: int
    expert_parameter_ratio: float
    idealized_expert_compute_ratio: float
    final_train_loss: float
    final_validation_loss: float
    final_validation_perplexity: float
    best_validation_loss: float
    elapsed_seconds: float
    tokens_seen: int
    utilization_entropy: float
    min_expert_fraction: float
    max_expert_fraction: float


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def full_codes(codes: torch.Tensor) -> torch.Tensor:
    return torch.cat([torch.ones(codes.shape[0], 1, device=codes.device, dtype=codes.dtype), codes], dim=1)


class SparseBaselineMoE(nn.Module):
    def __init__(self, d_model: int, d_ff: int, n_experts: int, top_k: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.n_experts = n_experts
        self.top_k = top_k
        self.router = nn.Linear(d_model, n_experts, bias=False)
        self.gate = nn.Parameter(torch.empty(n_experts, d_ff, d_model))
        self.up = nn.Parameter(torch.empty(n_experts, d_ff, d_model))
        self.down = nn.Parameter(torch.empty(n_experts, d_model, d_ff))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.router.weight, std=0.02)
        for weights in (self.gate, self.up, self.down):
            for expert in weights:
                nn.init.xavier_uniform_(expert)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        shape = x.shape
        flat = x.reshape(-1, self.d_model)
        logits = self.router(flat)
        top_values, top_ids = torch.topk(logits, self.top_k, dim=-1)
        route_weights = F.softmax(top_values, dim=-1)
        output = torch.zeros_like(flat)
        for expert_id in range(self.n_experts):
            where = torch.nonzero(top_ids == expert_id, as_tuple=False)
            if where.numel() == 0:
                continue
            token_ids, slots = where[:, 0], where[:, 1]
            selected = flat.index_select(0, token_ids)
            gate = F.linear(selected, self.gate[expert_id])
            up = F.linear(selected, self.up[expert_id])
            hidden = F.silu(gate) * up
            expert_out = F.linear(hidden, self.down[expert_id])
            expert_out = expert_out * route_weights[token_ids, slots].unsqueeze(-1)
            output.index_add_(0, token_ids, expert_out)
        aux = router_auxiliary_loss(logits, top_ids, self.n_experts)
        return output.reshape(shape), aux, top_ids


class ModalMoE(nn.Module):
    def __init__(self, d_model: int, d_ff: int, n_experts: int, top_k: int, modal_rank: int) -> None:
        super().__init__()
        if modal_rank < 1:
            raise ValueError("modal_rank must be positive")
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
        self.gate_codes = nn.Parameter(torch.empty(n_experts, modal_rank))
        self.up_codes = nn.Parameter(torch.empty(n_experts, modal_rank))
        self.down_codes = nn.Parameter(torch.empty(n_experts, modal_rank))
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

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        shape = x.shape
        flat = x.reshape(-1, self.d_model)
        logits = self.router(flat)
        top_values, top_ids = torch.topk(logits, self.top_k, dim=-1)
        route_weights = F.softmax(top_values, dim=-1)
        gate_projected = torch.einsum("nd,kmd->nkm", flat, self.gate_modes)
        up_projected = torch.einsum("nd,kmd->nkm", flat, self.up_modes)
        gate_codes = full_codes(self.gate_codes).index_select(0, top_ids.reshape(-1)).reshape(flat.shape[0], self.top_k, self.n_modes)
        up_codes = full_codes(self.up_codes).index_select(0, top_ids.reshape(-1)).reshape(flat.shape[0], self.top_k, self.n_modes)
        gate = torch.einsum("nkm,ntk->ntm", gate_projected, gate_codes)
        up = torch.einsum("nkm,ntk->ntm", up_projected, up_codes)
        hidden = F.silu(gate) * up
        down_codes = full_codes(self.down_codes).index_select(0, top_ids.reshape(-1)).reshape(flat.shape[0], self.top_k, self.n_modes)
        mode_inputs = torch.einsum("nt,ntk,ntm->nkm", route_weights, down_codes, hidden)
        output = torch.einsum("kdm,nkm->nd", self.down_modes, mode_inputs)
        aux = router_auxiliary_loss(logits, top_ids, self.n_experts)
        return output.reshape(shape), aux, top_ids

    @torch.no_grad()
    def reference_reconstructed(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        flat = x.reshape(-1, self.d_model)
        logits = self.router(flat)
        top_values, top_ids = torch.topk(logits, self.top_k, dim=-1)
        route_weights = F.softmax(top_values, dim=-1)
        cg, cu, cd = full_codes(self.gate_codes), full_codes(self.up_codes), full_codes(self.down_codes)
        output = torch.zeros_like(flat)
        for expert_id in range(self.n_experts):
            where = torch.nonzero(top_ids == expert_id, as_tuple=False)
            if where.numel() == 0:
                continue
            token_ids, slots = where[:, 0], where[:, 1]
            gate_weight = torch.einsum("k,kmd->md", cg[expert_id], self.gate_modes)
            up_weight = torch.einsum("k,kmd->md", cu[expert_id], self.up_modes)
            down_weight = torch.einsum("k,kdm->dm", cd[expert_id], self.down_modes)
            selected = flat.index_select(0, token_ids)
            hidden = F.silu(F.linear(selected, gate_weight)) * F.linear(selected, up_weight)
            expert_out = F.linear(hidden, down_weight)
            expert_out = expert_out * route_weights[token_ids, slots].unsqueeze(-1)
            output.index_add_(0, token_ids, expert_out)
        return output.reshape(shape)


def router_auxiliary_loss(logits: torch.Tensor, top_ids: torch.Tensor, n_experts: int) -> torch.Tensor:
    probabilities = F.softmax(logits, dim=-1)
    importance = probabilities.mean(dim=0)
    assignments = F.one_hot(top_ids, n_experts).float().mean(dim=(0, 1))
    balance = n_experts * torch.sum(importance * assignments)
    z_loss = torch.mean(torch.logsumexp(logits, dim=-1) ** 2)
    return balance + 0.1 * z_loss


class Block(nn.Module):
    def __init__(self, cfg: Config, variant: str, modal_rank: int | None) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(cfg.d_model)
        self.attn = nn.MultiheadAttention(cfg.d_model, cfg.n_heads, batch_first=True, dropout=0.0)
        self.norm2 = nn.LayerNorm(cfg.d_model)
        if variant == "baseline":
            self.moe: nn.Module = SparseBaselineMoE(cfg.d_model, cfg.d_ff, cfg.n_experts, cfg.top_k)
        elif variant == "modal" and modal_rank is not None:
            self.moe = ModalMoE(cfg.d_model, cfg.d_ff, cfg.n_experts, cfg.top_k, modal_rank)
        else:
            raise ValueError((variant, modal_rank))

    def forward(self, x: torch.Tensor, causal_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]:
        normalized = self.norm1(x)
        attention, _ = self.attn(normalized, normalized, normalized, attn_mask=causal_mask, need_weights=False)
        x = x + attention
        moe_out, aux, ids = self.moe(self.norm2(x))
        return x + moe_out, aux, [ids]


class LanguageModel(nn.Module):
    def __init__(self, vocab_size: int, cfg: Config, variant: str, modal_rank: int | None) -> None:
        super().__init__()
        self.cfg = cfg
        self.token_embedding = nn.Embedding(vocab_size, cfg.d_model)
        self.position_embedding = nn.Embedding(cfg.seq_len, cfg.d_model)
        self.blocks = nn.ModuleList([Block(cfg, variant, modal_rank) for _ in range(cfg.n_layers)])
        self.norm = nn.LayerNorm(cfg.d_model)
        self.output = nn.Linear(cfg.d_model, vocab_size, bias=False)
        self.output.weight = self.token_embedding.weight
        nn.init.normal_(self.token_embedding.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)

    def forward(self, tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]:
        positions = torch.arange(tokens.shape[1], device=tokens.device)
        x = self.token_embedding(tokens) + self.position_embedding(positions)[None, :, :]
        mask = torch.full((tokens.shape[1], tokens.shape[1]), float("-inf"), device=tokens.device)
        mask = torch.triu(mask, diagonal=1)
        aux_total = torch.zeros((), device=tokens.device)
        routes: list[torch.Tensor] = []
        for block in self.blocks:
            x, aux, block_routes = block(x, mask)
            aux_total = aux_total + aux
            routes.extend(block_routes)
        return self.output(self.norm(x)), aux_total / len(self.blocks), routes


class CharDataset:
    def __init__(self, text: str, seq_len: int, split: float = 0.9) -> None:
        chars = sorted(set(text))
        self.vocab = chars
        self.stoi = {char: index for index, char in enumerate(chars)}
        encoded = torch.tensor([self.stoi[c] for c in text], dtype=torch.long)
        split_index = int(len(encoded) * split)
        self.train = encoded[:split_index]
        self.validation = encoded[split_index:]
        self.seq_len = seq_len

    def batch(self, split: str, batch_size: int, generator: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
        data = self.train if split == "train" else self.validation
        starts = torch.randint(0, len(data) - self.seq_len - 1, (batch_size,), generator=generator)
        x = torch.stack([data[i:i + self.seq_len] for i in starts])
        y = torch.stack([data[i + 1:i + self.seq_len + 1] for i in starts])
        return x, y


@torch.no_grad()
def evaluate(model: LanguageModel, dataset: CharDataset, cfg: Config, generator: torch.Generator) -> tuple[float, np.ndarray]:
    model.eval()
    losses: list[float] = []
    counts = np.zeros(cfg.n_experts, dtype=np.float64)
    for _ in range(cfg.eval_batches):
        x, y = dataset.batch("validation", cfg.batch_size, generator)
        logits, _, routes = model(x)
        losses.append(float(F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))))
        for ids in routes:
            counts += np.bincount(ids.cpu().numpy().reshape(-1), minlength=cfg.n_experts)
    model.train()
    return float(np.mean(losses)), counts / max(counts.sum(), 1.0)


def parameter_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_variant(variant: str, modal_rank: int | None, seed: int, dataset: CharDataset, cfg: Config) -> tuple[Result, list[dict[str, float]]]:
    set_seed(seed)
    model = LanguageModel(len(dataset.vocab), cfg, variant, modal_rank)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    train_gen = torch.Generator().manual_seed(seed + 101)
    eval_gen = torch.Generator().manual_seed(seed + 202)
    history: list[dict[str, float]] = []
    best_val = float("inf")
    started = time.perf_counter()
    final_train = float("nan")
    for step in range(1, cfg.steps + 1):
        x, y = dataset.batch("train", cfg.batch_size, train_gen)
        logits, aux, _ = model(x)
        language_loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))
        loss = language_loss + cfg.aux_weight * aux
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()
        final_train = float(language_loss.detach())
        if step == 1 or step % cfg.eval_interval == 0 or step == cfg.steps:
            val_loss, utilization = evaluate(model, dataset, cfg, eval_gen)
            best_val = min(best_val, val_loss)
            history.append({"step": float(step), "train_loss": final_train, "validation_loss": val_loss, "aux_loss": float(aux.detach())})
            print(f"variant={variant} rank={modal_rank} seed={seed} step={step}/{cfg.steps} train={final_train:.4f} val={val_loss:.4f} aux={float(aux.detach()):.4f}", flush=True)
    final_val, utilization = evaluate(model, dataset, cfg, eval_gen)
    nonzero = utilization[utilization > 0]
    entropy = float(-np.sum(nonzero * np.log(nonzero)) / math.log(cfg.n_experts)) if nonzero.size else 0.0
    expert_parameter_ratio = 1.0 if variant == "baseline" else (modal_rank + 1) / cfg.n_experts
    compute_ratio = 1.0 if variant == "baseline" else (modal_rank + 1) / cfg.top_k
    result = Result(
        variant=variant if modal_rank is None else f"modal-k{modal_rank}", seed=seed, modal_rank=modal_rank,
        trainable_parameters=parameter_count(model), expert_parameter_ratio=expert_parameter_ratio,
        idealized_expert_compute_ratio=compute_ratio, final_train_loss=final_train,
        final_validation_loss=final_val, final_validation_perplexity=float(math.exp(min(final_val, 20.0))),
        best_validation_loss=best_val, elapsed_seconds=time.perf_counter() - started,
        tokens_seen=cfg.steps * cfg.batch_size * cfg.seq_len, utilization_entropy=entropy,
        min_expert_fraction=float(np.min(utilization)), max_expert_fraction=float(np.max(utilization)),
    )
    return result, history


def decision(results: Sequence[Result]) -> dict:
    baseline = next(result for result in results if result.variant == "baseline")
    comparisons = []
    for result in results:
        if result.variant == "baseline":
            continue
        comparisons.append({
            "variant": result.variant,
            "relative_validation_loss": result.final_validation_loss / baseline.final_validation_loss,
            "absolute_validation_loss_delta": result.final_validation_loss - baseline.final_validation_loss,
            "expert_parameter_ratio": result.expert_parameter_ratio,
            "idealized_expert_compute_ratio": result.idealized_expert_compute_ratio,
        })
    k2 = next((x for x in comparisons if x["variant"] == "modal-k2"), None)
    k3 = next((x for x in comparisons if x["variant"] == "modal-k3"), None)
    verdict = "PASS" if k2 and k2["relative_validation_loss"] <= 1.10 else ("BORDERLINE" if k3 and k3["relative_validation_loss"] <= 1.10 else "FAIL")
    return {
        "verdict": verdict,
        "baseline_validation_loss": baseline.final_validation_loss,
        "comparisons": comparisons,
        "rule": "PASS: K=2 reaches <=10% relative loss penalty at 75% ideal expert compute and ~18.75% expert matrix parameters. BORDERLINE: K=3 reaches the same quality gate at compute parity.",
    }


def self_test() -> None:
    set_seed(5)
    module = ModalMoE(d_model=12, d_ff=20, n_experts=7, top_k=3, modal_rank=2)
    module.eval()
    x = torch.randn(2, 4, 12)
    direct, _, _ = module(x)
    reference = module.reference_reconstructed(x)
    error_value = float((torch.linalg.vector_norm(direct - reference) / torch.linalg.vector_norm(reference)).detach())
    if error_value > 2e-6:
        raise AssertionError(f"modal fused execution mismatch: {error_value}")
    direct.sum().backward()
    if any(parameter.grad is None for parameter in module.parameters()):
        raise AssertionError("missing gradient")
    print(f"self-test passed: fused/reconstructed relative error={error_value:.3e}")


def write_outputs(output_dir: Path, payload: dict) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "modal_moe_trainability.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    rows = payload["results"]
    with (output_dir / "results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader(); writer.writerows(rows)
    history_rows = [{"variant": variant, **item} for variant, history in payload["history"].items() for item in history]
    with (output_dir / "history.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history_rows[0].keys()))
        writer.writeheader(); writer.writerows(history_rows)
    lines = [
        "# Test 2.0 — modal MoE trainability under architectural constraint", "", f"**Decision:** **{payload['decision']['verdict']}**", "",
        "| Variant | Parameters | Expert parameter ratio | Ideal expert compute ratio | Validation loss | Loss vs baseline | Utilization entropy |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    baseline_loss = payload["decision"]["baseline_validation_loss"]
    for item in rows:
        lines.append(f"| {item['variant']} | {item['trainable_parameters']:,} | {item['expert_parameter_ratio']:.2%} | {item['idealized_expert_compute_ratio']:.2%} | {item['final_validation_loss']:.4f} | {item['final_validation_loss'] / baseline_loss:.3f}× | {item['utilization_entropy']:.3f} |")
    lines += [
        "",
        "The experiment asks whether optimization can coordinate experts around shared full-rank matrix modes when that structure is present from initialization. It does not claim transfer to billion-parameter training from a character-level task.",
        "The modal `down` projection is executed after aggregation, so the tested graph matches the intended future kernel algebra rather than reconstructing expert matrices.",
    ]
    (output_dir / "VERDICT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--text", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("results/modal_moe_trainability"))
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--modal-ranks", default="1,2,3")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args(argv)
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    if args.self_test:
        self_test(); return 0
    if args.text is None:
        parser.error("--text is required")
    text = args.text.read_text(encoding="utf-8")
    cfg = Config(steps=args.steps, batch_size=args.batch_size, seq_len=args.seq_len)
    dataset = CharDataset(text, cfg.seq_len)
    variants: list[tuple[str, int | None]] = [("baseline", None)] + [("modal", int(rank)) for rank in args.modal_ranks.split(",") if rank.strip()]
    results: list[Result] = []
    histories: dict[str, list[dict[str, float]]] = {}
    for variant, rank in variants:
        result, history = train_variant(variant, rank, args.seed, dataset, cfg)
        results.append(result); histories[result.variant] = history
    payload = {
        "metadata": {
            "task": "character-level next-token prediction", "dataset_chars": len(text), "vocab_size": len(dataset.vocab),
            "config": asdict(cfg), "modal_execution": "shared gate/up projections plus pre-down modal aggregation; no weight reconstruction",
        },
        "decision": decision(results), "results": [asdict(x) for x in results], "history": histories,
    }
    write_outputs(args.output_dir, payload)
    print((args.output_dir / "VERDICT.md").read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
