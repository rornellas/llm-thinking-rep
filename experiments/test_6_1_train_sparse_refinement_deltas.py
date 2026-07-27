#!/usr/bin/env python3
"""Test 6.1: train progressive Modal-MoE prefixes for sparse refinements.

Test 6.0 showed that post-training magnitude thresholding retained quality at
75% sparsity and narrowly missed the strict 90% sparsity gate.  This experiment
fine-tunes identical copies of one progressive checkpoint with a scale-invariant
Hoyer concentration penalty on per-token prefix deltas:

    R(delta) = ||delta||_1 / (sqrt(d) * ||delta||_2).

Minimizing R encourages each token's refinement to concentrate on fewer output
coordinates without rewarding the trivial all-zero solution through scale
shrinkage.  Full-rank and sampled-prefix language losses remain in the
objective.  A lambda=0 fine-tune controls for the extra optimization steps.

After fine-tuning, thresholds are calibrated on train activations and held-out
90%/95% sparse 4-bit event streams are evaluated exactly as in Test 6.0.  This
still measures a digital event-traffic proxy; the reference implementation
computes dense prefixes to observe their deltas.
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
from dataclasses import asdict
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


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def causal_mask(length: int, device: torch.device) -> torch.Tensor:
    mask = torch.full((length, length), float("-inf"), device=device)
    return torch.triu(mask, diagonal=1)


def hoyer_ratio(delta: torch.Tensor) -> torch.Tensor:
    flat = delta.reshape(-1, delta.shape[-1])
    l1 = torch.sum(torch.abs(flat), dim=-1)
    l2 = torch.sqrt(torch.sum(flat.square(), dim=-1) + 1e-8)
    return torch.mean(l1 / (math.sqrt(flat.shape[-1]) * l2 + 1e-8))


def forward_rank_with_sparsity(
    model: Any,
    tokens: torch.Tensor,
    *,
    requested_rank: int,
    max_rank: int,
    collect_penalty: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    positions = torch.arange(tokens.shape[1], device=tokens.device)
    x = model.token_embedding(tokens) + model.position_embedding(positions)[None, :, :]
    mask = causal_mask(tokens.shape[1], tokens.device)
    aux_values: list[torch.Tensor] = []
    penalties: list[torch.Tensor] = []

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
        prefixes: list[torch.Tensor] = []
        rank_aux: list[torch.Tensor] = []
        for rank in range(max_rank + 1):
            block.moe.active_rank = rank
            output, aux, _ = block.moe(moe_input)
            prefixes.append(output)
            rank_aux.append(aux)
        block.moe.active_rank = max_rank
        if collect_penalty:
            for mode in range(1, max_rank + 1):
                penalties.append(hoyer_ratio(prefixes[mode] - prefixes[mode - 1]))
        aux_values.append(rank_aux[max_rank])
        x = x + prefixes[requested_rank]

    logits = model.output(model.norm(x))
    aux = torch.stack(aux_values).mean()
    penalty = (
        torch.stack(penalties).mean()
        if penalties
        else torch.zeros((), dtype=logits.dtype, device=logits.device)
    )
    return logits, aux, penalty


def fine_tune(
    model: Any,
    dataset: Any,
    cfg: Any,
    *,
    seed: int,
    steps: int,
    max_rank: int,
    hoyer_lambda: float,
) -> tuple[Any, dict[str, Any]]:
    set_seed(seed)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=2.5e-4, weight_decay=cfg.weight_decay
    )
    generator = torch.Generator().manual_seed(seed + 1)
    rank_rng = random.Random(seed + 2)
    started = time.perf_counter()
    history: list[dict[str, float]] = []
    for step in range(1, steps + 1):
        tokens, targets = dataset.batch("train", cfg.batch_size, generator)
        sampled_rank = rank_rng.randrange(max_rank + 1)
        full_logits, full_aux, concentration = forward_rank_with_sparsity(
            model,
            tokens,
            requested_rank=max_rank,
            max_rank=max_rank,
            collect_penalty=True,
        )
        full_ce = F.cross_entropy(
            full_logits.reshape(-1, full_logits.shape[-1]), targets.reshape(-1)
        )
        if sampled_rank == max_rank:
            sampled_ce = full_ce
            sampled_aux = full_aux
        else:
            sampled_logits, sampled_aux, _ = forward_rank_with_sparsity(
                model,
                tokens,
                requested_rank=sampled_rank,
                max_rank=max_rank,
                collect_penalty=False,
            )
            sampled_ce = F.cross_entropy(
                sampled_logits.reshape(-1, sampled_logits.shape[-1]),
                targets.reshape(-1),
            )
        # Ramp concentration pressure in over the first third of fine-tuning.
        ramp = min(1.0, step / max(1.0, steps / 3.0))
        loss = (
            0.65 * full_ce
            + 0.35 * sampled_ce
            + cfg.aux_weight * 0.5 * (full_aux + sampled_aux)
            + float(hoyer_lambda) * ramp * concentration
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()
        if step == 1 or step % 60 == 0 or step == steps:
            row = {
                "step": float(step),
                "full_ce": float(full_ce.detach()),
                "sampled_ce": float(sampled_ce.detach()),
                "hoyer_ratio": float(concentration.detach()),
                "objective": float(loss.detach()),
                "sampled_rank": float(sampled_rank),
            }
            history.append(row)
            print(
                f"lambda={hoyer_lambda:.3f} step={step}/{steps} "
                f"full={row['full_ce']:.4f} sampled={row['sampled_ce']:.4f} "
                f"hoyer={row['hoyer_ratio']:.4f}",
                flush=True,
            )
    return model, {
        "lambda": float(hoyer_lambda),
        "elapsed_seconds": time.perf_counter() - started,
        "history": history,
    }


def evaluate_variant(
    event: Any,
    model: Any,
    dataset: Any,
    cfg: Any,
    *,
    label: str,
    calibration_seed: int,
    validation_batches: Sequence[tuple[torch.Tensor, torch.Tensor]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    samples, _ = event.collect_delta_samples(
        model,
        dataset,
        cfg,
        batches=16,
        seed=calibration_seed,
        max_rank=3,
    )
    calibration = event.calibration_table(samples, (0.90, 0.95))
    placeholder = 0.0
    k3 = event.evaluate_policy(
        model,
        validation_batches,
        max_rank=3,
        calibration=None,
        sparsity=None,
        bits=None,
        mask_policy="full",
        reference_k3_loss=placeholder,
        reference_k1_loss=placeholder,
        seed=calibration_seed,
        d_model=cfg.d_model,
    )
    k1 = event.evaluate_policy(
        model,
        validation_batches,
        max_rank=3,
        calibration=None,
        sparsity=None,
        bits=None,
        mask_policy="static-k1",
        reference_k3_loss=k3.validation_loss,
        reference_k1_loss=placeholder,
        seed=calibration_seed,
        d_model=cfg.d_model,
    )
    k3.loss_delta_vs_k3 = 0.0
    k3.loss_delta_vs_k1 = k3.validation_loss - k1.validation_loss
    k1.loss_delta_vs_k3 = k1.validation_loss - k3.validation_loss
    k1.loss_delta_vs_k1 = 0.0
    policies = [k1, k3]
    for sparsity in (0.90, 0.95):
        policies.append(
            event.evaluate_policy(
                model,
                validation_batches,
                max_rank=3,
                calibration=calibration,
                sparsity=sparsity,
                bits=4,
                mask_policy="magnitude",
                reference_k3_loss=k3.validation_loss,
                reference_k1_loss=k1.validation_loss,
                seed=calibration_seed + int(sparsity * 1000) + 4,
                d_model=cfg.d_model,
            )
        )
    policies.append(
        event.evaluate_policy(
            model,
            validation_batches,
            max_rank=3,
            calibration=calibration,
            sparsity=0.90,
            bits=4,
            mask_policy="random",
            reference_k3_loss=k3.validation_loss,
            reference_k1_loss=k1.validation_loss,
            seed=calibration_seed + 9004,
            d_model=cfg.d_model,
        )
    )
    rows: list[dict[str, Any]] = []
    for policy in policies:
        row = asdict(policy)
        row["training_variant"] = label
        rows.append(row)
    calibration_density = {
        f"s{sparsity}-l{layer}-m{mode}": values
        for (sparsity, layer, mode), values in calibration.items()
    }
    return rows, calibration_density


def make_decision(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    variants = sorted({row["training_variant"] for row in rows})
    summaries: list[dict[str, Any]] = []
    for variant in variants:
        selected = [row for row in rows if row["training_variant"] == variant]
        indexed = {row["label"]: row for row in selected}
        magnitude = indexed["magnitude-s90-q4"]
        random_control = indexed["random-s90-q4"]
        summaries.append(
            {
                "training_variant": variant,
                "k3_loss": indexed["static-k3"]["validation_loss"],
                "k1_loss": indexed["static-k1"]["validation_loss"],
                "event90_loss": magnitude["validation_loss"],
                "event90_delta_vs_k3": magnitude["loss_delta_vs_k3"],
                "event90_density": magnitude["mean_event_density"],
                "event90_bit_ratio": magnitude["residual_bit_ratio_vs_dense_bf16"],
                "event90_advantage_over_random": (
                    random_control["validation_loss"] - magnitude["validation_loss"]
                ),
                "event95_delta_vs_k3": indexed["magnitude-s95-q4"]["loss_delta_vs_k3"],
                "event95_bit_ratio": indexed["magnitude-s95-q4"]["residual_bit_ratio_vs_dense_bf16"],
            }
        )
    control = next(row for row in summaries if row["training_variant"] == "lambda-0.000")
    regularized = [row for row in summaries if row["training_variant"] != "lambda-0.000"]
    feasible = [
        row for row in regularized
        if row["event90_delta_vs_k3"] <= 0.010
        and row["event90_advantage_over_random"] >= 0.003
        and row["k3_loss"] <= control["k3_loss"] + 0.010
    ]
    if feasible:
        best = min(
            feasible,
            key=lambda row: (
                row["event90_bit_ratio"],
                row["event90_delta_vs_k3"],
            ),
        )
        verdict = "TRAINED_EVENT_SPARSITY_PASS"
    else:
        best = min(
            regularized,
            key=lambda row: row["event90_delta_vs_k3"],
        )
        verdict = "TRAINED_EVENT_SPARSITY_FAIL"
    return {
        "verdict": verdict,
        "best_variant": best["training_variant"],
        "best_event90_delta_vs_k3": best["event90_delta_vs_k3"],
        "best_event90_bit_ratio": best["event90_bit_ratio"],
        "best_event90_advantage_over_random": best["event90_advantage_over_random"],
        "control_event90_delta_vs_k3": control["event90_delta_vs_k3"],
        "event90_improvement_over_control": (
            control["event90_delta_vs_k3"] - best["event90_delta_vs_k3"]
        ),
        "summaries": summaries,
        "rule": (
            "PASS requires a regularized variant with 90%/4-bit held-out delta "
            "<=0.010 nat versus its K3 path, >=0.003 nat advantage over random "
            "events, and K3 quality within 0.010 nat of the lambda=0 fine-tune."
        ),
    }


def write_outputs(output_dir: Path, payload: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "trained_sparse_deltas.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with (output_dir / "policies.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(payload["policies"][0].keys()))
        writer.writeheader(); writer.writerows(payload["policies"])
    summaries = payload["decision"]["summaries"]
    with (output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summaries[0].keys()))
        writer.writeheader(); writer.writerows(summaries)

    d = payload["decision"]
    lines = [
        "# Test 6.1 — training for sparse refinement events",
        "",
        f"**Decision:** **{d['verdict']}**",
        "",
        "| Fine-tune | K3 loss | K1 loss | 90%/4-bit loss | Δ vs K3 | Event density | Residual bits/BF16 | Advantage vs random | 95% Δ |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summaries:
        lines.append(
            f"| {row['training_variant']} | {row['k3_loss']:.4f} | "
            f"{row['k1_loss']:.4f} | {row['event90_loss']:.4f} | "
            f"{row['event90_delta_vs_k3']:+.4f} | {row['event90_density']:.3%} | "
            f"{row['event90_bit_ratio']:.3%} | "
            f"{row['event90_advantage_over_random']:+.4f} | "
            f"{row['event95_delta_vs_k3']:+.4f} |"
        )
    lines += [
        "",
        f"- Best variant: `{d['best_variant']}`.",
        f"- Best 90%/4-bit Δ versus K3: `{d['best_event90_delta_vs_k3']:+.4f}` nat.",
        f"- Event traffic: `{d['best_event90_bit_ratio']:.3%}` of dense BF16 residual deltas.",
        f"- Improvement over lambda=0 fine-tune: `{d['event90_improvement_over_control']:+.4f}` nat.",
        f"- Magnitude advantage over random: `{d['best_event90_advantage_over_random']:+.4f}` nat.",
        "",
        "Hoyer regularization is scale invariant and operates on per-token MoE output deltas. This remains a state-traffic screen; dense prefix computation is still used by the reference path.",
    ]
    (output_dir / "VERDICT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def self_test() -> None:
    dense = torch.ones(4, 16)
    sparse = torch.zeros(4, 16)
    sparse[:, 0] = 4.0
    if not hoyer_ratio(sparse) < hoyer_ratio(dense):
        raise AssertionError((hoyer_ratio(sparse), hoyer_ratio(dense)))
    print("self-test passed")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=False)
    parser.add_argument("--progressive-source", type=Path, required=False)
    parser.add_argument("--event-source", type=Path, required=False)
    parser.add_argument("--text", type=Path, required=False)
    parser.add_argument("--output-dir", type=Path, required=False)
    parser.add_argument("--seed", type=int, default=61600)
    parser.add_argument("--initial-steps", type=int, default=400)
    parser.add_argument("--fine-tune-steps", type=int, default=180)
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
            args.event_source,
            args.text,
            args.output_dir,
        )
    ):
        parser.error("source, progressive source, event source, text, and output are required")

    source = load_module("sparse_train_base", args.source)
    progressive = load_module("sparse_train_progressive", args.progressive_source)
    event = load_module("sparse_train_event", args.event_source)
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
    dataset = source.CharDataset(args.text.read_text(encoding="utf-8"), cfg.seq_len)
    initial_model, initial_training_seconds = progressive.train_progressive(
        source, dataset, cfg, args.seed, 3
    )

    validation_generator = torch.Generator().manual_seed(args.seed + 900)
    validation_batches = [
        tuple(
            value.cpu()
            for value in dataset.batch("validation", cfg.batch_size, validation_generator)
        )
        for _ in range(24)
    ]

    lambdas = (0.0, 0.02, 0.05)
    policies: list[dict[str, Any]] = []
    training_records: list[dict[str, Any]] = []
    calibration_records: dict[str, Any] = {}
    for index, regularization in enumerate(lambdas):
        model = copy.deepcopy(initial_model)
        model, record = fine_tune(
            model,
            dataset,
            cfg,
            seed=args.seed + 1000 + index,
            steps=args.fine_tune_steps,
            max_rank=3,
            hoyer_lambda=regularization,
        )
        label = f"lambda-{regularization:.3f}"
        record["label"] = label
        training_records.append(record)
        rows, calibration = evaluate_variant(
            event,
            model,
            dataset,
            cfg,
            label=label,
            calibration_seed=args.seed + 2000 + index,
            validation_batches=validation_batches,
        )
        policies.extend(rows)
        calibration_records[label] = calibration
        del model

    decision = make_decision(policies)
    payload = {
        "metadata": {
            "seed": args.seed,
            "initial_steps": args.initial_steps,
            "initial_training_seconds": initial_training_seconds,
            "fine_tune_steps_per_variant": args.fine_tune_steps,
            "lambdas": list(lambdas),
            "experts": cfg.n_experts,
            "top_k": cfg.top_k,
            "d_model": cfg.d_model,
            "d_ff": cfg.d_ff,
            "layers": cfg.n_layers,
            "validation_batches": len(validation_batches),
        },
        "training": training_records,
        "calibration": calibration_records,
        "policies": policies,
        "decision": decision,
    }
    write_outputs(args.output_dir, payload)
    print((args.output_dir / "VERDICT.md").read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
