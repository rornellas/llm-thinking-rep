#!/usr/bin/env python3
"""Test 6.0: can nested refinement cycles be encoded as sparse events?

A progressive Modal-MoE checkpoint exposes prefix outputs K=0..3.  At each MoE
layer we define refinement deltas

    delta_k(x) = moe_K(x) - moe_{K-1}(x),  k=1,2,3.

This experiment calibrates magnitude thresholds on train-split activations,
then transmits held-out deltas as a bitmap plus quantized non-zero values.  The
base K=0 output remains dense.  A random-mask control preserves the same event
rate but removes magnitude-based selection.

The experiment is a digital proxy for pulse/event hardware.  It tests whether
refinement *state traffic* can become sparse.  It does not claim that the
current implementation avoids computing dense prefixes: the reference Python
path evaluates all prefixes to measure the exact deltas.  A positive result
would authorize a directly event-producing architecture or hardware kernel.
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
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
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
class EventResult:
    label: str
    sparsity_target: float | None
    quantization_bits: int | None
    mask_policy: str
    validation_loss: float
    validation_perplexity: float
    loss_delta_vs_k3: float
    loss_delta_vs_k1: float
    mean_event_density: float
    p95_token_event_density: float
    residual_bit_ratio_vs_dense_bf16: float
    residual_bits_per_token_layer: float
    mean_delta_relative_error: float
    elapsed_seconds: float


def causal_mask(length: int, device: torch.device) -> torch.Tensor:
    mask = torch.full((length, length), float("-inf"), device=device)
    return torch.triu(mask, diagonal=1)


@torch.no_grad()
def layer_prefix_outputs(block: Any, moe_input: torch.Tensor, max_rank: int) -> list[torch.Tensor]:
    outputs: list[torch.Tensor] = []
    for rank in range(max_rank + 1):
        block.moe.active_rank = rank
        output, _, _ = block.moe(moe_input)
        outputs.append(output)
    block.moe.active_rank = max_rank
    return outputs


@torch.no_grad()
def collect_delta_samples(
    model: Any,
    dataset: Any,
    cfg: Any,
    *,
    batches: int,
    seed: int,
    max_rank: int,
    sample_per_delta: int = 131072,
) -> tuple[dict[tuple[int, int], np.ndarray], list[tuple[torch.Tensor, torch.Tensor]]]:
    generator = torch.Generator().manual_seed(seed)
    samples: dict[tuple[int, int], list[np.ndarray]] = {
        (layer, mode): []
        for layer in range(len(model.blocks))
        for mode in range(1, max_rank + 1)
    }
    stored: list[tuple[torch.Tensor, torch.Tensor]] = []
    model.eval()
    rng = np.random.default_rng(seed + 99)
    for batch_index in range(batches):
        tokens, targets = dataset.batch("train", cfg.batch_size, generator)
        stored.append((tokens.cpu(), targets.cpu()))
        positions = torch.arange(tokens.shape[1])
        x = model.token_embedding(tokens) + model.position_embedding(positions)[None, :, :]
        mask = causal_mask(tokens.shape[1], x.device)
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
            prefixes = layer_prefix_outputs(block, moe_input, max_rank)
            for mode in range(1, max_rank + 1):
                values = (prefixes[mode] - prefixes[mode - 1]).abs().reshape(-1).cpu().numpy()
                if values.size > sample_per_delta // batches:
                    indices = rng.choice(
                        values.size,
                        size=sample_per_delta // batches,
                        replace=False,
                    )
                    values = values[indices]
                samples[(layer_index, mode)].append(values.astype(np.float32, copy=False))
            x = x + prefixes[max_rank]
        if batch_index == 0 or (batch_index + 1) % 4 == 0:
            print(f"calibration batch {batch_index + 1}/{batches}", flush=True)
    merged = {
        key: np.concatenate(chunks) if chunks else np.zeros(1, dtype=np.float32)
        for key, chunks in samples.items()
    }
    return merged, stored


def calibration_table(
    samples: dict[tuple[int, int], np.ndarray],
    sparsity_targets: Sequence[float],
) -> dict[tuple[float, int, int], dict[str, float]]:
    table: dict[tuple[float, int, int], dict[str, float]] = {}
    for sparsity in sparsity_targets:
        for (layer, mode), values in samples.items():
            threshold = float(np.quantile(values, sparsity))
            retained = values[values > threshold]
            if retained.size == 0:
                retained = np.asarray([max(float(values.max()), EPS)], dtype=np.float32)
            table[(float(sparsity), layer, mode)] = {
                "threshold": threshold,
                "mean_retained": float(np.mean(retained)),
                "clip_995": float(max(np.quantile(retained, 0.995), EPS)),
                "calibration_density": float(np.mean(values > threshold)),
            }
    return table


def quantize_events(
    delta: torch.Tensor,
    *,
    threshold: float,
    mean_retained: float,
    clip_995: float,
    bits: int,
    mask_policy: str,
    target_density: float,
    random_generator: torch.Generator | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if mask_policy == "magnitude":
        mask = delta.abs() > threshold
    elif mask_policy == "random":
        if random_generator is None:
            raise ValueError("random generator required")
        mask = torch.rand(
            delta.shape,
            generator=random_generator,
            device=delta.device,
        ) < target_density
    else:
        raise ValueError(mask_policy)

    if bits == 2:
        quantized = torch.sign(delta) * float(mean_retained)
    else:
        levels = (1 << (bits - 1)) - 1
        clip = float(max(clip_995, EPS))
        scaled = torch.round(delta.clamp(-clip, clip) / clip * levels)
        nonzero = torch.sign(delta) * torch.clamp(scaled.abs(), min=1.0)
        quantized = nonzero / levels * clip
    return torch.where(mask, quantized, torch.zeros_like(delta)), mask


@torch.no_grad()
def forward_with_events(
    model: Any,
    tokens: torch.Tensor,
    *,
    max_rank: int,
    calibration: dict[tuple[float, int, int], dict[str, float]] | None,
    sparsity: float | None,
    bits: int | None,
    mask_policy: str,
    random_seed: int,
) -> tuple[torch.Tensor, dict[str, Any]]:
    positions = torch.arange(tokens.shape[1], device=tokens.device)
    x = model.token_embedding(tokens) + model.position_embedding(positions)[None, :, :]
    mask = causal_mask(tokens.shape[1], tokens.device)
    event_counts: list[float] = []
    token_densities: list[np.ndarray] = []
    relative_errors: list[float] = []
    random_generator = torch.Generator(device=tokens.device).manual_seed(random_seed)

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
        prefixes = layer_prefix_outputs(block, moe_input, max_rank)
        if mask_policy.startswith("static-k"):
            rank = int(mask_policy.split("k", 1)[1])
            moe_output = prefixes[rank]
        elif mask_policy == "full":
            moe_output = prefixes[max_rank]
        else:
            if calibration is None or sparsity is None or bits is None:
                raise ValueError("event calibration is required")
            moe_output = prefixes[0]
            for mode in range(1, max_rank + 1):
                delta = prefixes[mode] - prefixes[mode - 1]
                info = calibration[(float(sparsity), layer_index, mode)]
                encoded, event_mask = quantize_events(
                    delta,
                    threshold=info["threshold"],
                    mean_retained=info["mean_retained"],
                    clip_995=info["clip_995"],
                    bits=bits,
                    mask_policy=mask_policy,
                    target_density=info["calibration_density"],
                    random_generator=random_generator,
                )
                moe_output = moe_output + encoded
                density_by_token = event_mask.float().mean(dim=-1)
                token_densities.append(density_by_token.cpu().numpy().reshape(-1))
                event_counts.append(float(event_mask.float().mean()))
                error = torch.linalg.vector_norm(encoded - delta) / torch.clamp(
                    torch.linalg.vector_norm(delta), min=EPS
                )
                relative_errors.append(float(error))
        x = x + moe_output

    logits = model.output(model.norm(x))
    diagnostics = {
        "mean_event_density": float(np.mean(event_counts)) if event_counts else 0.0,
        "p95_token_event_density": float(
            np.percentile(np.concatenate(token_densities), 95)
        ) if token_densities else 0.0,
        "mean_delta_relative_error": float(np.mean(relative_errors)) if relative_errors else 0.0,
    }
    return logits, diagnostics


def token_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        targets.reshape(-1),
        reduction="none",
    )


def residual_bit_metrics(
    density: float,
    bits: int,
    *,
    d_model: int,
    modes: int,
) -> tuple[float, float]:
    # One bitmap bit per coordinate and mode plus value bits for active events.
    event_bits = modes * d_model * (1.0 + density * bits)
    dense_bits = modes * d_model * 16.0
    return event_bits / dense_bits, event_bits


@torch.no_grad()
def evaluate_policy(
    model: Any,
    batches: Sequence[tuple[torch.Tensor, torch.Tensor]],
    *,
    max_rank: int,
    calibration: dict[tuple[float, int, int], dict[str, float]] | None,
    sparsity: float | None,
    bits: int | None,
    mask_policy: str,
    reference_k3_loss: float,
    reference_k1_loss: float,
    seed: int,
    d_model: int,
) -> EventResult:
    started = time.perf_counter()
    losses: list[float] = []
    densities: list[float] = []
    p95_densities: list[float] = []
    errors: list[float] = []
    for batch_index, (tokens, targets) in enumerate(batches):
        logits, diagnostics = forward_with_events(
            model,
            tokens,
            max_rank=max_rank,
            calibration=calibration,
            sparsity=sparsity,
            bits=bits,
            mask_policy=mask_policy,
            random_seed=seed + batch_index,
        )
        losses.append(float(token_loss(logits, targets).mean()))
        densities.append(diagnostics["mean_event_density"])
        p95_densities.append(diagnostics["p95_token_event_density"])
        errors.append(diagnostics["mean_delta_relative_error"])
    loss = float(np.mean(losses))
    density = float(np.mean(densities))
    if bits is None:
        bit_ratio, bits_per_token = 0.0, 0.0
    else:
        bit_ratio, bits_per_token = residual_bit_metrics(
            density, bits, d_model=d_model, modes=max_rank
        )
    if mask_policy == "full":
        label = "static-k3"
    elif mask_policy.startswith("static-k"):
        label = mask_policy
    else:
        label = f"{mask_policy}-s{int(round(float(sparsity) * 100))}-q{bits}"
    return EventResult(
        label=label,
        sparsity_target=sparsity,
        quantization_bits=bits,
        mask_policy=mask_policy,
        validation_loss=loss,
        validation_perplexity=float(math.exp(min(loss, 20.0))),
        loss_delta_vs_k3=loss - reference_k3_loss,
        loss_delta_vs_k1=loss - reference_k1_loss,
        mean_event_density=density,
        p95_token_event_density=float(np.mean(p95_densities)),
        residual_bit_ratio_vs_dense_bf16=bit_ratio,
        residual_bits_per_token_layer=bits_per_token,
        mean_delta_relative_error=float(np.mean(errors)),
        elapsed_seconds=time.perf_counter() - started,
    )


def make_decision(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    magnitude = [
        row for row in rows
        if row["mask_policy"] == "magnitude"
        and float(row["sparsity_target"]) >= 0.90
        and int(row["quantization_bits"]) <= 4
    ]
    feasible = [
        row for row in magnitude
        if row["loss_delta_vs_k3"] <= 0.010
    ]
    random_control = next(
        row for row in rows if row["label"] == "random-s90-q4"
    )
    magnitude_control = next(
        row for row in rows if row["label"] == "magnitude-s90-q4"
    )
    control_advantage = (
        random_control["validation_loss"] - magnitude_control["validation_loss"]
    )
    if feasible and control_advantage >= 0.003:
        best = min(
            feasible,
            key=lambda row: (
                row["residual_bit_ratio_vs_dense_bf16"],
                row["loss_delta_vs_k3"],
            ),
        )
        verdict = "EVENT_REPRESENTATION_SIGNAL"
    elif feasible:
        best = min(
            feasible,
            key=lambda row: row["residual_bit_ratio_vs_dense_bf16"],
        )
        verdict = "EVENT_COMPRESSION_SIGNAL"
    else:
        best = min(magnitude, key=lambda row: row["loss_delta_vs_k3"])
        verdict = "FAIL"
    return {
        "verdict": verdict,
        "best_candidate": best["label"],
        "best_loss_delta_vs_k3": best["loss_delta_vs_k3"],
        "best_residual_bit_ratio": best["residual_bit_ratio_vs_dense_bf16"],
        "magnitude_advantage_over_random_s90_q4_nats": control_advantage,
        "rule": (
            "Strong signal requires at least 90% target sparsity, <=4-bit "
            "events, <=0.010 nat held-out loss increase versus K3, and at "
            "least 0.003 nat advantage over a random mask with the same event rate."
        ),
    }


def write_outputs(output_dir: Path, payload: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "event_delta_refinement.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with (output_dir / "results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(payload["results"][0].keys()))
        writer.writeheader(); writer.writerows(payload["results"])
    d = payload["decision"]
    lines = [
        "# Test 6.0 — sparse event-delta refinement",
        "",
        f"**Decision:** **{d['verdict']}**",
        "",
        "| Policy | Loss | Δ vs K3 | Δ vs K1 | Event density | p95 token density | Residual bits/BF16 | Delta error |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["results"]:
        lines.append(
            f"| {row['label']} | {row['validation_loss']:.4f} | "
            f"{row['loss_delta_vs_k3']:+.4f} | {row['loss_delta_vs_k1']:+.4f} | "
            f"{row['mean_event_density']:.3%} | {row['p95_token_event_density']:.3%} | "
            f"{row['residual_bit_ratio_vs_dense_bf16']:.3%} | "
            f"{row['mean_delta_relative_error']:.3f} |"
        )
    lines += [
        "",
        f"- Best candidate: `{d['best_candidate']}`.",
        f"- Best held-out Δ loss versus K3: `{d['best_loss_delta_vs_k3']:+.4f}` nat.",
        f"- Best residual traffic ratio: `{d['best_residual_bit_ratio']:.3%}` of dense BF16 deltas.",
        f"- Magnitude advantage over random at 90%/4-bit: `{d['magnitude_advantage_over_random_s90_q4_nats']:+.4f}` nat.",
        "",
        "Only refinement-state transport is modeled. The current measurement path still computes dense prefixes to obtain exact deltas; a positive result must be followed by a directly event-producing model or kernel.",
    ]
    (output_dir / "VERDICT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def self_test(source: Any, progressive: Any) -> None:
    set_seed(13)
    moe = progressive.ProgressiveNeuronwiseMoE(12, 20, 7, 3, 3)
    x = torch.randn(2, 5, 12)
    prefixes = []
    for rank in range(4):
        moe.active_rank = rank
        value, _, _ = moe(x)
        prefixes.append(value)
    reconstructed = prefixes[0]
    for mode in range(1, 4):
        reconstructed = reconstructed + prefixes[mode] - prefixes[mode - 1]
    error = float(torch.max(torch.abs(reconstructed - prefixes[3])))
    if error > 1e-6:
        raise AssertionError(error)
    delta = prefixes[1] - prefixes[0]
    encoded, mask = quantize_events(
        delta,
        threshold=float(torch.quantile(delta.abs(), 0.9)),
        mean_retained=float(delta.abs().mean()),
        clip_995=float(torch.quantile(delta.abs(), 0.995)),
        bits=4,
        mask_policy="magnitude",
        target_density=0.1,
        random_generator=None,
    )
    if encoded.shape != delta.shape or mask.shape != delta.shape:
        raise AssertionError((encoded.shape, mask.shape))
    print(f"self-test passed; telescoping error={error:.3e}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--progressive-source", type=Path, required=True)
    parser.add_argument("--text", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=False)
    parser.add_argument("--seed", type=int, default=60600)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--calibration-batches", type=int, default=16)
    parser.add_argument("--test-batches", type=int, default=24)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args(argv)

    source = load_module("event_base_source", args.source)
    progressive = load_module("event_progressive_source", args.progressive_source)
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    if args.self_test:
        self_test(source, progressive); return 0
    if args.output_dir is None:
        parser.error("--output-dir is required")

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
        steps=args.steps,
        eval_interval=100,
        eval_batches=20,
    )
    dataset = source.CharDataset(args.text.read_text(encoding="utf-8"), cfg.seq_len)
    model, training_seconds = progressive.train_progressive(
        source, dataset, cfg, args.seed, 3
    )
    samples, _ = collect_delta_samples(
        model,
        dataset,
        cfg,
        batches=args.calibration_batches,
        seed=args.seed + 100,
        max_rank=3,
    )
    sparsities = (0.75, 0.90, 0.95)
    calibration = calibration_table(samples, sparsities)

    validation_generator = torch.Generator().manual_seed(args.seed + 200)
    test_batches = [
        tuple(value.cpu() for value in dataset.batch("validation", cfg.batch_size, validation_generator))
        for _ in range(args.test_batches)
    ]
    # Reference policies first.
    placeholder = 0.0
    k3 = evaluate_policy(
        model,
        test_batches,
        max_rank=3,
        calibration=None,
        sparsity=None,
        bits=None,
        mask_policy="full",
        reference_k3_loss=placeholder,
        reference_k1_loss=placeholder,
        seed=args.seed,
        d_model=cfg.d_model,
    )
    k1 = evaluate_policy(
        model,
        test_batches,
        max_rank=3,
        calibration=None,
        sparsity=None,
        bits=None,
        mask_policy="static-k1",
        reference_k3_loss=k3.validation_loss,
        reference_k1_loss=placeholder,
        seed=args.seed,
        d_model=cfg.d_model,
    )
    k0 = evaluate_policy(
        model,
        test_batches,
        max_rank=3,
        calibration=None,
        sparsity=None,
        bits=None,
        mask_policy="static-k0",
        reference_k3_loss=k3.validation_loss,
        reference_k1_loss=k1.validation_loss,
        seed=args.seed,
        d_model=cfg.d_model,
    )
    k2 = evaluate_policy(
        model,
        test_batches,
        max_rank=3,
        calibration=None,
        sparsity=None,
        bits=None,
        mask_policy="static-k2",
        reference_k3_loss=k3.validation_loss,
        reference_k1_loss=k1.validation_loss,
        seed=args.seed,
        d_model=cfg.d_model,
    )
    # Correct reference deltas in rows created before K1 was known.
    k3.loss_delta_vs_k3 = 0.0
    k3.loss_delta_vs_k1 = k3.validation_loss - k1.validation_loss
    k1.loss_delta_vs_k3 = k1.validation_loss - k3.validation_loss
    k1.loss_delta_vs_k1 = 0.0
    rows: list[EventResult] = [k0, k1, k2, k3]

    for sparsity in sparsities:
        for bits in (8, 4, 2):
            rows.append(
                evaluate_policy(
                    model,
                    test_batches,
                    max_rank=3,
                    calibration=calibration,
                    sparsity=sparsity,
                    bits=bits,
                    mask_policy="magnitude",
                    reference_k3_loss=k3.validation_loss,
                    reference_k1_loss=k1.validation_loss,
                    seed=args.seed + int(sparsity * 1000) + bits,
                    d_model=cfg.d_model,
                )
            )
    rows.append(
        evaluate_policy(
            model,
            test_batches,
            max_rank=3,
            calibration=calibration,
            sparsity=0.90,
            bits=4,
            mask_policy="random",
            reference_k3_loss=k3.validation_loss,
            reference_k1_loss=k1.validation_loss,
            seed=args.seed + 9004,
            d_model=cfg.d_model,
        )
    )
    result_rows = [asdict(row) for row in rows]
    payload = {
        "metadata": {
            "seed": args.seed,
            "training_steps": args.steps,
            "training_seconds": training_seconds,
            "calibration_batches": args.calibration_batches,
            "test_batches": args.test_batches,
            "d_model": cfg.d_model,
            "d_ff": cfg.d_ff,
            "layers": cfg.n_layers,
            "experts": cfg.n_experts,
            "top_k": cfg.top_k,
            "max_rank": 3,
            "quantization_note": "2-bit uses ternary zero/sign with train-calibrated mean retained magnitude; 4/8-bit use symmetric uniform quantization clipped at train-calibrated p99.5.",
            "traffic_note": "Residual bit ratio includes one bitmap bit per component per refinement mode plus value bits for nonzero events.",
        },
        "calibration": {
            f"s{sparsity}-l{layer}-m{mode}": values
            for (sparsity, layer, mode), values in calibration.items()
        },
        "results": result_rows,
    }
    payload["decision"] = make_decision(result_rows)
    write_outputs(args.output_dir, payload)
    print((args.output_dir / "VERDICT.md").read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
