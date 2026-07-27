#!/usr/bin/env python3
"""Test 6.2: train progressive refinements through a hard event bottleneck.

Test 6.1's soft Hoyer regularizer did not make 90%-sparse post-hoc deltas
accurate enough.  This experiment changes the causal graph: during fine-tuning,
each refinement delta is passed through an exact per-token top-k event mask and
optional straight-through 4-bit quantizer *before* it reaches the residual
stream.

Variants:

* full-control: identical extra fine-tuning without an event bottleneck;
* topk90-fp: retain the largest 10% of delta coordinates;
* topk90-q4: same mask with fake 4-bit symmetric quantization;
* topk95-q4: retain the largest 5% with fake 4-bit quantization.

The event path receives most of the language-model objective; a smaller dense
K3 auxiliary objective preserves the underlying prefix.  Evaluation reports the
actual event-forward loss, dense K3 quality, bitmap+value traffic, and a random
mask control at equal density.  Top-k selection cost is not included in the
traffic ratio and must be measured in a future kernel.
"""
from __future__ import annotations

import argparse
import copy
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


EPS = 1e-12


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


@dataclass
class PolicyResult:
    training_variant: str
    evaluation_policy: str
    event_density: float
    quantization_bits: int | None
    validation_loss: float
    validation_perplexity: float
    delta_vs_own_dense_k3: float
    delta_vs_control_dense_k3: float
    random_mask_loss: float | None
    magnitude_advantage_over_random: float | None
    residual_bit_ratio_vs_dense_bf16: float
    p95_token_event_density: float
    dense_k3_loss: float
    dense_k1_loss: float
    elapsed_seconds: float


def causal_mask(length: int, device: torch.device) -> torch.Tensor:
    mask = torch.full((length, length), float("-inf"), device=device)
    return torch.triu(mask, diagonal=1)


def topk_mask(delta: torch.Tensor, density: float) -> torch.Tensor:
    width = delta.shape[-1]
    count = max(1, min(width, int(math.ceil(width * density))))
    indices = torch.topk(delta.abs(), k=count, dim=-1, sorted=False).indices
    mask = torch.zeros_like(delta, dtype=torch.bool)
    mask.scatter_(-1, indices, True)
    return mask


def random_mask(
    delta: torch.Tensor,
    density: float,
    generator: torch.Generator,
) -> torch.Tensor:
    width = delta.shape[-1]
    count = max(1, min(width, int(math.ceil(width * density))))
    scores = torch.rand(
        delta.shape,
        dtype=delta.dtype,
        device=delta.device,
        generator=generator,
    )
    indices = torch.topk(scores, k=count, dim=-1, sorted=False).indices
    mask = torch.zeros_like(delta, dtype=torch.bool)
    mask.scatter_(-1, indices, True)
    return mask


def fake_quantize_ste(
    values: torch.Tensor,
    bits: int | None,
) -> torch.Tensor:
    if bits is None:
        return values
    levels = (1 << (bits - 1)) - 1
    # Per-token scale avoids calibration leakage and follows local event range.
    scale = torch.amax(values.abs(), dim=-1, keepdim=True).clamp_min(1e-6)
    quantized = torch.round(values / scale * levels).clamp(-levels, levels)
    dequantized = quantized / levels * scale
    return values + (dequantized - values).detach()


def encode_delta(
    delta: torch.Tensor,
    *,
    density: float,
    bits: int | None,
    policy: str,
    generator: torch.Generator | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if policy == "magnitude":
        mask = topk_mask(delta, density)
    elif policy == "random":
        if generator is None:
            raise ValueError("random generator is required")
        mask = random_mask(delta, density, generator)
    else:
        raise ValueError(policy)
    quantized = fake_quantize_ste(delta, bits)
    return torch.where(mask, quantized, torch.zeros_like(delta)), mask


def layer_prefixes(block: Any, moe_input: torch.Tensor, max_rank: int) -> tuple[list[torch.Tensor], torch.Tensor]:
    outputs: list[torch.Tensor] = []
    aux: torch.Tensor | None = None
    for rank in range(max_rank + 1):
        block.moe.active_rank = rank
        output, rank_aux, _ = block.moe(moe_input)
        outputs.append(output)
        if rank == max_rank:
            aux = rank_aux
    block.moe.active_rank = max_rank
    if aux is None:
        raise AssertionError("missing router auxiliary loss")
    return outputs, aux


def forward_policy(
    model: Any,
    tokens: torch.Tensor,
    *,
    max_rank: int,
    policy: str,
    density: float,
    bits: int | None,
    random_seed: int,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    positions = torch.arange(tokens.shape[1], device=tokens.device)
    x = model.token_embedding(tokens) + model.position_embedding(positions)[None, :, :]
    mask = causal_mask(tokens.shape[1], tokens.device)
    aux_values: list[torch.Tensor] = []
    densities: list[torch.Tensor] = []
    generator = torch.Generator(device=tokens.device).manual_seed(random_seed)

    for block in model.blocks:
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
        prefixes, aux = layer_prefixes(block, moe_input, max_rank)
        aux_values.append(aux)
        if policy == "dense-k3":
            moe_output = prefixes[max_rank]
        elif policy == "dense-k1":
            moe_output = prefixes[1]
        else:
            moe_output = prefixes[0]
            for mode in range(1, max_rank + 1):
                delta = prefixes[mode] - prefixes[mode - 1]
                encoded, event_mask = encode_delta(
                    delta,
                    density=density,
                    bits=bits,
                    policy=policy,
                    generator=generator,
                )
                moe_output = moe_output + encoded
                densities.append(event_mask.float().mean(dim=-1))
        x = x + moe_output

    logits = model.output(model.norm(x))
    aux_total = torch.stack(aux_values).mean()
    if densities:
        density_values = torch.cat([value.reshape(-1) for value in densities])
        diagnostics = {
            "mean_density": float(density_values.detach().mean()),
            "p95_density": float(
                torch.quantile(density_values.detach(), 0.95)
            ),
        }
    else:
        diagnostics = {"mean_density": 0.0, "p95_density": 0.0}
    return logits, aux_total, diagnostics


def fine_tune_variant(
    model: Any,
    dataset: Any,
    cfg: Any,
    *,
    seed: int,
    steps: int,
    max_rank: int,
    label: str,
    density: float,
    bits: int | None,
) -> tuple[Any, dict[str, Any]]:
    set_seed(seed)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=2.5e-4, weight_decay=cfg.weight_decay
    )
    generator = torch.Generator().manual_seed(seed + 1)
    started = time.perf_counter()
    history: list[dict[str, float]] = []
    for step in range(1, steps + 1):
        tokens, targets = dataset.batch("train", cfg.batch_size, generator)
        dense_logits, dense_aux, _ = forward_policy(
            model,
            tokens,
            max_rank=max_rank,
            policy="dense-k3",
            density=1.0,
            bits=None,
            random_seed=seed + step,
        )
        dense_ce = F.cross_entropy(
            dense_logits.reshape(-1, dense_logits.shape[-1]),
            targets.reshape(-1),
        )
        if label == "full-control":
            event_ce = dense_ce
            event_aux = dense_aux
        else:
            event_logits, event_aux, diagnostics = forward_policy(
                model,
                tokens,
                max_rank=max_rank,
                policy="magnitude",
                density=density,
                bits=bits,
                random_seed=seed + step + 100000,
            )
            event_ce = F.cross_entropy(
                event_logits.reshape(-1, event_logits.shape[-1]),
                targets.reshape(-1),
            )
        loss = (
            0.80 * event_ce
            + 0.20 * dense_ce
            + cfg.aux_weight * 0.5 * (event_aux + dense_aux)
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()
        if step == 1 or step % 60 == 0 or step == steps:
            row = {
                "step": float(step),
                "event_ce": float(event_ce.detach()),
                "dense_ce": float(dense_ce.detach()),
                "objective": float(loss.detach()),
            }
            history.append(row)
            print(
                f"{label} step={step}/{steps} event={row['event_ce']:.4f} "
                f"dense={row['dense_ce']:.4f}",
                flush=True,
            )
    return model, {
        "label": label,
        "density": density,
        "bits": bits,
        "elapsed_seconds": time.perf_counter() - started,
        "history": history,
    }


def token_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        targets.reshape(-1),
        reduction="none",
    )


def traffic_ratio(density: float, bits: int | None) -> float:
    if bits is None:
        return 1.0
    # One bitmap bit per component plus value bits for active events.
    return (1.0 + density * bits) / 16.0


@torch.no_grad()
def evaluate(
    model: Any,
    batches: Sequence[tuple[torch.Tensor, torch.Tensor]],
    *,
    policy: str,
    density: float,
    bits: int | None,
    seed: int,
    max_rank: int,
) -> tuple[float, dict[str, float]]:
    model.eval()
    losses: list[float] = []
    mean_densities: list[float] = []
    p95_densities: list[float] = []
    for index, (tokens, targets) in enumerate(batches):
        logits, _, diagnostics = forward_policy(
            model,
            tokens,
            max_rank=max_rank,
            policy=policy,
            density=density,
            bits=bits,
            random_seed=seed + index,
        )
        losses.append(float(token_loss(logits, targets).mean()))
        mean_densities.append(diagnostics["mean_density"])
        p95_densities.append(diagnostics["p95_density"])
    return float(np.mean(losses)), {
        "mean_density": float(np.mean(mean_densities)),
        "p95_density": float(np.mean(p95_densities)),
    }


def evaluate_variant(
    model: Any,
    batches: Sequence[tuple[torch.Tensor, torch.Tensor]],
    *,
    training_variant: str,
    density: float,
    bits: int | None,
    seed: int,
    max_rank: int,
    control_dense_loss: float,
) -> list[PolicyResult]:
    started = time.perf_counter()
    dense_k3, _ = evaluate(
        model,
        batches,
        policy="dense-k3",
        density=1.0,
        bits=None,
        seed=seed,
        max_rank=max_rank,
    )
    dense_k1, _ = evaluate(
        model,
        batches,
        policy="dense-k1",
        density=1.0,
        bits=None,
        seed=seed,
        max_rank=max_rank,
    )
    rows = [
        PolicyResult(
            training_variant=training_variant,
            evaluation_policy="dense-k3",
            event_density=1.0,
            quantization_bits=None,
            validation_loss=dense_k3,
            validation_perplexity=float(math.exp(min(dense_k3, 20.0))),
            delta_vs_own_dense_k3=0.0,
            delta_vs_control_dense_k3=dense_k3 - control_dense_loss,
            random_mask_loss=None,
            magnitude_advantage_over_random=None,
            residual_bit_ratio_vs_dense_bf16=1.0,
            p95_token_event_density=1.0,
            dense_k3_loss=dense_k3,
            dense_k1_loss=dense_k1,
            elapsed_seconds=time.perf_counter() - started,
        ),
        PolicyResult(
            training_variant=training_variant,
            evaluation_policy="dense-k1",
            event_density=1.0,
            quantization_bits=None,
            validation_loss=dense_k1,
            validation_perplexity=float(math.exp(min(dense_k1, 20.0))),
            delta_vs_own_dense_k3=dense_k1 - dense_k3,
            delta_vs_control_dense_k3=dense_k1 - control_dense_loss,
            random_mask_loss=None,
            magnitude_advantage_over_random=None,
            residual_bit_ratio_vs_dense_bf16=1.0,
            p95_token_event_density=1.0,
            dense_k3_loss=dense_k3,
            dense_k1_loss=dense_k1,
            elapsed_seconds=time.perf_counter() - started,
        ),
    ]
    if training_variant == "full-control":
        evaluation_density = 0.10
        evaluation_bits = 4
    else:
        evaluation_density = density
        evaluation_bits = bits
    magnitude_loss, magnitude_diag = evaluate(
        model,
        batches,
        policy="magnitude",
        density=evaluation_density,
        bits=evaluation_bits,
        seed=seed + 2000,
        max_rank=max_rank,
    )
    random_loss, _ = evaluate(
        model,
        batches,
        policy="random",
        density=evaluation_density,
        bits=evaluation_bits,
        seed=seed + 3000,
        max_rank=max_rank,
    )
    rows.append(
        PolicyResult(
            training_variant=training_variant,
            evaluation_policy=f"topk-{1.0 - evaluation_density:.0%}-q{evaluation_bits}",
            event_density=magnitude_diag["mean_density"],
            quantization_bits=evaluation_bits,
            validation_loss=magnitude_loss,
            validation_perplexity=float(math.exp(min(magnitude_loss, 20.0))),
            delta_vs_own_dense_k3=magnitude_loss - dense_k3,
            delta_vs_control_dense_k3=magnitude_loss - control_dense_loss,
            random_mask_loss=random_loss,
            magnitude_advantage_over_random=random_loss - magnitude_loss,
            residual_bit_ratio_vs_dense_bf16=traffic_ratio(
                magnitude_diag["mean_density"], evaluation_bits
            ),
            p95_token_event_density=magnitude_diag["p95_density"],
            dense_k3_loss=dense_k3,
            dense_k1_loss=dense_k1,
            elapsed_seconds=time.perf_counter() - started,
        )
    )
    return rows


def make_decision(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    event_rows = [
        row for row in rows
        if row["evaluation_policy"].startswith("topk")
        and row["training_variant"] != "full-control"
    ]
    feasible = [
        row for row in event_rows
        if row["delta_vs_own_dense_k3"] <= 0.010
        and row["delta_vs_control_dense_k3"] <= 0.020
        and float(row["magnitude_advantage_over_random"]) >= 0.003
    ]
    if feasible:
        best = min(
            feasible,
            key=lambda row: (
                row["residual_bit_ratio_vs_dense_bf16"],
                row["delta_vs_own_dense_k3"],
            ),
        )
        verdict = "HARD_EVENT_BOTTLENECK_PASS"
    else:
        best = min(
            event_rows,
            key=lambda row: row["delta_vs_own_dense_k3"],
        )
        verdict = "HARD_EVENT_BOTTLENECK_FAIL"
    control = next(
        row for row in rows
        if row["training_variant"] == "full-control"
        and row["evaluation_policy"].startswith("topk")
    )
    return {
        "verdict": verdict,
        "best_training_variant": best["training_variant"],
        "best_policy": best["evaluation_policy"],
        "best_delta_vs_own_dense_k3": best["delta_vs_own_dense_k3"],
        "best_delta_vs_control_dense_k3": best["delta_vs_control_dense_k3"],
        "best_residual_bit_ratio": best["residual_bit_ratio_vs_dense_bf16"],
        "best_advantage_over_random": best["magnitude_advantage_over_random"],
        "control_posthoc_delta_vs_own_k3": control["delta_vs_own_dense_k3"],
        "improvement_over_posthoc_control": (
            control["delta_vs_own_dense_k3"] - best["delta_vs_own_dense_k3"]
        ),
        "rule": (
            "PASS requires event-forward loss within 0.010 nat of its dense K3, "
            "within 0.020 nat of the control K3, and >=0.003 nat advantage over "
            "an equal-density random event mask."
        ),
    }


def write_outputs(output_dir: Path, payload: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "hard_event_bottleneck.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with (output_dir / "results.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(payload["results"][0].keys())
        )
        writer.writeheader(); writer.writerows(payload["results"])
    d = payload["decision"]
    lines = [
        "# Test 6.2 — hard top-k event bottleneck",
        "",
        f"**Decision:** **{d['verdict']}**",
        "",
        "| Training | Evaluation | Loss | Δ own K3 | Δ control K3 | Density | Residual bits/BF16 | Advantage vs random |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["results"]:
        lines.append(
            f"| {row['training_variant']} | {row['evaluation_policy']} | "
            f"{row['validation_loss']:.4f} | {row['delta_vs_own_dense_k3']:+.4f} | "
            f"{row['delta_vs_control_dense_k3']:+.4f} | {row['event_density']:.3%} | "
            f"{row['residual_bit_ratio_vs_dense_bf16']:.3%} | "
            f"{('n/a' if row['magnitude_advantage_over_random'] is None else f'{row['magnitude_advantage_over_random']:+.4f}')} |"
        )
    lines += [
        "",
        f"- Best: `{d['best_training_variant']}` / `{d['best_policy']}`.",
        f"- Δ versus own dense K3: `{d['best_delta_vs_own_dense_k3']:+.4f}` nat.",
        f"- Δ versus control dense K3: `{d['best_delta_vs_control_dense_k3']:+.4f}` nat.",
        f"- Residual traffic: `{d['best_residual_bit_ratio']:.3%}` of dense BF16 deltas.",
        f"- Advantage over random: `{d['best_advantage_over_random']:+.4f}` nat.",
        f"- Improvement over post-hoc control: `{d['improvement_over_posthoc_control']:+.4f}` nat.",
        "",
        "The top-k mask and fake quantizer are in the fine-tuning forward pass. Sorting/index selection cost is not included in the residual traffic ratio.",
    ]
    (output_dir / "VERDICT.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def self_test() -> None:
    set_seed(23)
    delta = torch.randn(3, 5, 20, requires_grad=True)
    encoded, mask = encode_delta(
        delta,
        density=0.10,
        bits=4,
        policy="magnitude",
        generator=None,
    )
    if not torch.all(mask.sum(dim=-1) == 2):
        raise AssertionError(mask.sum(dim=-1))
    encoded.sum().backward()
    if delta.grad is None or not torch.isfinite(delta.grad).all():
        raise AssertionError("missing STE gradients")
    print("self-test passed")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=False)
    parser.add_argument("--progressive-source", type=Path, required=False)
    parser.add_argument("--text", type=Path, required=False)
    parser.add_argument("--output-dir", type=Path, required=False)
    parser.add_argument("--seed", type=int, default=62600)
    parser.add_argument("--initial-steps", type=int, default=400)
    parser.add_argument("--fine-tune-steps", type=int, default=220)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args(argv)

    if args.self_test:
        self_test(); return 0
    if any(
        value is None
        for value in (
            args.source,
            args.progressive_source,
            args.text,
            args.output_dir,
        )
    ):
        parser.error("source, progressive source, text, and output are required")

    source = load_module("hard_event_base", args.source)
    progressive = load_module("hard_event_progressive", args.progressive_source)
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)

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
    cfg = source.Config(
        steps=args.initial_steps,
        eval_interval=100,
        eval_batches=20,
    )
    dataset = source.CharDataset(
        args.text.read_text(encoding="utf-8"), cfg.seq_len
    )
    initial_model, initial_training_seconds = progressive.train_progressive(
        source, dataset, cfg, args.seed, 3
    )
    validation_generator = torch.Generator().manual_seed(args.seed + 900)
    validation_batches = [
        tuple(
            value.cpu()
            for value in dataset.batch(
                "validation", cfg.batch_size, validation_generator
            )
        )
        for _ in range(24)
    ]

    variants = [
        {"label": "full-control", "density": 1.0, "bits": None},
        {"label": "topk90-fp", "density": 0.10, "bits": None},
        {"label": "topk90-q4", "density": 0.10, "bits": 4},
        {"label": "topk95-q4", "density": 0.05, "bits": 4},
    ]
    trained: list[tuple[dict[str, Any], Any, dict[str, Any]]] = []
    for index, variant in enumerate(variants):
        model = copy.deepcopy(initial_model)
        model, training = fine_tune_variant(
            model,
            dataset,
            cfg,
            seed=args.seed + 1000 + index,
            steps=args.fine_tune_steps,
            max_rank=3,
            label=variant["label"],
            density=float(variant["density"]),
            bits=variant["bits"],
        )
        trained.append((variant, model, training))

    control_variant, control_model, _ = trained[0]
    control_dense_loss, _ = evaluate(
        control_model,
        validation_batches,
        policy="dense-k3",
        density=1.0,
        bits=None,
        seed=args.seed + 5000,
        max_rank=3,
    )
    rows: list[dict[str, Any]] = []
    training_records: list[dict[str, Any]] = []
    for index, (variant, model, training) in enumerate(trained):
        training_records.append(training)
        policy_rows = evaluate_variant(
            model,
            validation_batches,
            training_variant=variant["label"],
            density=float(variant["density"]),
            bits=variant["bits"],
            seed=args.seed + 6000 + index * 100,
            max_rank=3,
            control_dense_loss=control_dense_loss,
        )
        rows.extend(asdict(row) for row in policy_rows)

    payload = {
        "metadata": {
            "seed": args.seed,
            "initial_steps": args.initial_steps,
            "initial_training_seconds": initial_training_seconds,
            "fine_tune_steps": args.fine_tune_steps,
            "experts": cfg.n_experts,
            "top_k": cfg.top_k,
            "d_model": cfg.d_model,
            "d_ff": cfg.d_ff,
            "layers": cfg.n_layers,
            "validation_batches": len(validation_batches),
            "traffic_note": "bitmap plus nonzero value bits; top-k selection and index-compaction latency excluded",
        },
        "training": training_records,
        "results": rows,
    }
    payload["decision"] = make_decision(rows)
    write_outputs(args.output_dir, payload)
    print((args.output_dir / "VERDICT.md").read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
