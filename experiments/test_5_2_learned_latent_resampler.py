#!/usr/bin/env python3
"""Test 5.2: replace discrete patch boundaries with learned latent resampling.

Test 5.1 showed that fixed, random, bigram, and neural-surprisal boundaries are
statistically indistinguishable when all compress a 128-byte context to roughly
32 units.  This experiment removes explicit segmentation and learns a continuous
compression map.

A common next-16-byte predictor receives latent context slots produced by one of:

* fixed mean pooling over four-byte blocks (32 slots);
* a learned GRU inside fixed four-byte blocks (32 slots);
* a learned stride-4 convolution (32 slots);
* 32 learned global queries cross-attending to all 128 bytes;
* the same 32-slot resampler with frozen random compressor parameters;
* 16 learned global queries, testing twice the compression.

Within each seed, every variant uses exactly the same raw contexts, targets,
shared predictor architecture, shared-module initialization order, optimization
steps, and evaluation examples.  The compressor parameter count and actual CPU
throughput are reported; the resampler is not assumed to be free.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


EPS = 1e-12


@dataclass
class Config:
    context_bytes: int = 128
    horizon_bytes: int = 16
    patch_bytes: int = 4
    slots: int = 32
    batch_size: int = 16
    steps: int = 700
    eval_batches: int = 40
    d_model: int = 64
    n_heads: int = 4
    n_layers: int = 2
    d_ff: int = 192
    lr: float = 5e-4
    weight_decay: float = 0.03
    grad_clip: float = 1.0


@dataclass
class Result:
    seed: int
    variant: str
    slots: int
    validation_bits_per_byte: float
    validation_loss_nats_per_byte: float
    best_validation_bits_per_byte: float
    final_train_bits_per_byte: float
    trainable_parameters: int
    total_parameters: int
    compressor_trainable_parameters: int
    attention_work_ratio_to_byte: float
    training_seconds: float
    context_bytes_per_second: float
    latent_effective_rank: float
    latent_mean_pairwise_cosine: float


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class FixedMeanCompressor(nn.Module):
    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.patch_bytes = cfg.patch_bytes
        self.slots = cfg.context_bytes // cfg.patch_bytes

    def forward(self, embedded: torch.Tensor) -> torch.Tensor:
        batch, length, width = embedded.shape
        return embedded.reshape(
            batch, self.slots, self.patch_bytes, width
        ).mean(dim=2)


class FixedGRUCompressor(nn.Module):
    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.patch_bytes = cfg.patch_bytes
        self.slots = cfg.context_bytes // cfg.patch_bytes
        self.gru = nn.GRU(cfg.d_model, cfg.d_model, batch_first=True)

    def forward(self, embedded: torch.Tensor) -> torch.Tensor:
        batch, _, width = embedded.shape
        patches = embedded.reshape(
            batch * self.slots, self.patch_bytes, width
        )
        encoded, _ = self.gru(patches)
        return encoded[:, -1].reshape(batch, self.slots, width)


class StridedConvCompressor(nn.Module):
    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.slots = cfg.context_bytes // cfg.patch_bytes
        self.conv = nn.Conv1d(
            cfg.d_model,
            cfg.d_model,
            kernel_size=cfg.patch_bytes,
            stride=cfg.patch_bytes,
        )

    def forward(self, embedded: torch.Tensor) -> torch.Tensor:
        return self.conv(embedded.transpose(1, 2)).transpose(1, 2)


class LatentResampler(nn.Module):
    def __init__(self, cfg: Config, slots: int, *, frozen: bool = False) -> None:
        super().__init__()
        self.slots = slots
        self.queries = nn.Parameter(torch.empty(slots, cfg.d_model))
        self.cross_attention = nn.MultiheadAttention(
            cfg.d_model,
            cfg.n_heads,
            batch_first=True,
            dropout=0.0,
        )
        self.norm1 = nn.LayerNorm(cfg.d_model)
        self.norm2 = nn.LayerNorm(cfg.d_model)
        self.ffn = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_model * 2),
            nn.GELU(),
            nn.Linear(cfg.d_model * 2, cfg.d_model),
        )
        nn.init.normal_(self.queries, std=0.02)
        if frozen:
            for parameter in self.parameters():
                parameter.requires_grad_(False)

    def forward(self, embedded: torch.Tensor) -> torch.Tensor:
        query = self.queries[None, :, :].expand(embedded.shape[0], -1, -1)
        attended, _ = self.cross_attention(
            self.norm1(query),
            embedded,
            embedded,
            need_weights=False,
        )
        latent = query + attended
        return latent + self.ffn(self.norm2(latent))


class ContextPredictor(nn.Module):
    def __init__(self, cfg: Config, variant: str) -> None:
        super().__init__()
        self.cfg = cfg
        self.variant = variant
        # Shared modules are constructed before variant-specific modules so a
        # common seed produces identical shared initialization across variants.
        self.byte_embedding = nn.Embedding(256, cfg.d_model)
        self.byte_position_embedding = nn.Embedding(cfg.context_bytes, cfg.d_model)
        self.latent_position_embedding = nn.Embedding(cfg.slots + 1, cfg.d_model)
        self.readout_query = nn.Parameter(torch.empty(1, 1, cfg.d_model))
        layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.n_heads,
            dim_feedforward=cfg.d_ff,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.context_transformer = nn.TransformerEncoder(
            layer, num_layers=cfg.n_layers
        )
        self.final_norm = nn.LayerNorm(cfg.d_model)
        self.output = nn.Linear(cfg.d_model, cfg.horizon_bytes * 256)
        self._reset_shared()

        if variant == "fixed-mean32":
            self.compressor = FixedMeanCompressor(cfg)
        elif variant == "fixed-gru32":
            self.compressor = FixedGRUCompressor(cfg)
        elif variant == "conv32":
            self.compressor = StridedConvCompressor(cfg)
        elif variant == "resampler32":
            self.compressor = LatentResampler(cfg, 32)
        elif variant == "resampler32-frozen":
            self.compressor = LatentResampler(cfg, 32, frozen=True)
        elif variant == "resampler16":
            self.compressor = LatentResampler(cfg, 16)
        else:
            raise ValueError(variant)

    def _reset_shared(self) -> None:
        nn.init.normal_(self.byte_embedding.weight, std=0.02)
        nn.init.normal_(self.byte_position_embedding.weight, std=0.02)
        nn.init.normal_(self.latent_position_embedding.weight, std=0.02)
        nn.init.normal_(self.readout_query, std=0.02)
        nn.init.normal_(self.output.weight, std=0.02)
        nn.init.zeros_(self.output.bias)

    @property
    def slots(self) -> int:
        return int(self.compressor.slots)

    def compress(self, values: torch.Tensor) -> torch.Tensor:
        positions = torch.arange(
            values.shape[1], device=values.device
        )
        embedded = (
            self.byte_embedding(values)
            + self.byte_position_embedding(positions)[None, :, :]
        )
        return self.compressor(embedded)

    def forward(
        self, values: torch.Tensor, *, return_latents: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        latents = self.compress(values)
        batch = values.shape[0]
        query = self.readout_query.expand(batch, -1, -1)
        sequence = torch.cat([query, latents], dim=1)
        positions = torch.arange(
            sequence.shape[1], device=values.device
        )
        sequence = (
            sequence
            + self.latent_position_embedding(positions)[None, :, :]
        )
        transformed = self.context_transformer(sequence)
        logits = self.output(self.final_norm(transformed[:, 0])).reshape(
            batch, self.cfg.horizon_bytes, 256
        )
        if return_latents:
            return logits, latents
        return logits


def build_windows(
    data: np.ndarray,
    starts: np.ndarray,
    cfg: Config,
) -> tuple[torch.Tensor, torch.Tensor]:
    windows = np.stack(
        [
            data[
                int(start):
                int(start) + cfg.context_bytes + cfg.horizon_bytes
            ]
            for start in starts
        ]
    ).astype(np.int64, copy=False)
    return (
        torch.from_numpy(windows[:, :cfg.context_bytes]),
        torch.from_numpy(windows[:, cfg.context_bytes:]),
    )


@torch.no_grad()
def evaluate(
    model: ContextPredictor,
    contexts: torch.Tensor,
    targets: torch.Tensor,
    cfg: Config,
) -> tuple[float, np.ndarray, dict[str, float]]:
    model.eval()
    losses: list[np.ndarray] = []
    latent_sample: torch.Tensor | None = None
    for start in range(0, len(contexts), cfg.batch_size):
        x = contexts[start:start + cfg.batch_size]
        y = targets[start:start + cfg.batch_size]
        logits, latents = model(x, return_latents=True)
        per_byte = F.cross_entropy(
            logits.reshape(-1, 256),
            y.reshape(-1),
            reduction="none",
        ).reshape(len(x), cfg.horizon_bytes)
        losses.append(per_byte.mean(dim=1).cpu().numpy())
        if latent_sample is None:
            latent_sample = latents.detach().cpu()
    values = np.concatenate(losses)
    if latent_sample is None:
        raise AssertionError("no evaluation latents")
    flat = latent_sample.reshape(-1, latent_sample.shape[-1]).double()
    centered = flat - flat.mean(dim=0, keepdim=True)
    singular = torch.linalg.svdvals(centered)
    energy = singular.square()
    effective_rank = float(
        energy.sum().square()
        / torch.clamp(energy.square().sum(), min=EPS)
    )
    normalized = F.normalize(latent_sample.double(), dim=-1)
    cosine = torch.einsum("bsd,btd->bst", normalized, normalized)
    slots = cosine.shape[-1]
    upper = cosine[:, torch.triu_indices(slots, slots, offset=1)[0], torch.triu_indices(slots, slots, offset=1)[1]]
    latent_stats = {
        "effective_rank": effective_rank,
        "mean_pairwise_cosine": float(upper.mean()),
    }
    return float(np.mean(values)), values, latent_stats


def train_variant(
    variant: str,
    seed: int,
    train_contexts: torch.Tensor,
    train_targets: torch.Tensor,
    validation_contexts: torch.Tensor,
    validation_targets: torch.Tensor,
    cfg: Config,
) -> tuple[Result, np.ndarray]:
    set_seed(seed)
    model = ContextPredictor(cfg, variant)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )
    best = math.inf
    final_train = math.nan
    started = time.perf_counter()
    for step in range(1, cfg.steps + 1):
        offset = (step - 1) * cfg.batch_size
        x = train_contexts[offset:offset + cfg.batch_size]
        y = train_targets[offset:offset + cfg.batch_size]
        model.train()
        logits = model(x)
        loss = F.cross_entropy(
            logits.reshape(-1, 256), y.reshape(-1)
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()
        final_train = float(loss.detach())
        if step == 1 or step % 200 == 0 or step == cfg.steps:
            validation_loss, _, _ = evaluate(
                model,
                validation_contexts,
                validation_targets,
                cfg,
            )
            best = min(best, validation_loss)
            print(
                f"seed={seed} {variant} step={step}/{cfg.steps} "
                f"train-bpb={final_train / math.log(2.0):.4f} "
                f"val-bpb={validation_loss / math.log(2.0):.4f}",
                flush=True,
            )
    training_seconds = time.perf_counter() - started
    validation_loss, per_example, latent_stats = evaluate(
        model,
        validation_contexts,
        validation_targets,
        cfg,
    )
    best = min(best, validation_loss)
    compressor_trainable = sum(
        parameter.numel()
        for parameter in model.compressor.parameters()
        if parameter.requires_grad
    )
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    context_seen = cfg.steps * cfg.batch_size * cfg.context_bytes
    result = Result(
        seed=seed,
        variant=variant,
        slots=model.slots,
        validation_bits_per_byte=validation_loss / math.log(2.0),
        validation_loss_nats_per_byte=validation_loss,
        best_validation_bits_per_byte=best / math.log(2.0),
        final_train_bits_per_byte=final_train / math.log(2.0),
        trainable_parameters=trainable_parameters,
        total_parameters=total_parameters,
        compressor_trainable_parameters=compressor_trainable,
        attention_work_ratio_to_byte=(model.slots / cfg.context_bytes) ** 2,
        training_seconds=training_seconds,
        context_bytes_per_second=context_seen / max(training_seconds, EPS),
        latent_effective_rank=latent_stats["effective_rank"],
        latent_mean_pairwise_cosine=latent_stats["mean_pairwise_cosine"],
    )
    return result, per_example


def bootstrap_lower_bound(
    advantages: np.ndarray,
    seed: int,
    samples: int = 5000,
    quantile: float = 0.05,
) -> float:
    rng = np.random.default_rng(seed)
    indices = rng.integers(
        0, len(advantages), size=(samples, len(advantages))
    )
    return float(np.quantile(advantages[indices].mean(axis=1), quantile))


def aggregate(
    rows: Sequence[dict[str, Any]],
    losses: dict[tuple[int, str], np.ndarray],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    variants = sorted({row["variant"] for row in rows})
    summary: list[dict[str, Any]] = []
    for variant in variants:
        selected = [row for row in rows if row["variant"] == variant]
        summary.append(
            {
                "variant": variant,
                "runs": len(selected),
                "slots": int(selected[0]["slots"]),
                "validation_bpb_mean": statistics.mean(
                    row["validation_bits_per_byte"] for row in selected
                ),
                "validation_bpb_std": statistics.pstdev(
                    row["validation_bits_per_byte"] for row in selected
                ),
                "trainable_parameters": int(selected[0]["trainable_parameters"]),
                "compressor_trainable_parameters": int(
                    selected[0]["compressor_trainable_parameters"]
                ),
                "attention_work_ratio_to_byte": float(
                    selected[0]["attention_work_ratio_to_byte"]
                ),
                "context_bytes_per_second_mean": statistics.mean(
                    row["context_bytes_per_second"] for row in selected
                ),
                "latent_effective_rank_mean": statistics.mean(
                    row["latent_effective_rank"] for row in selected
                ),
                "latent_pairwise_cosine_mean": statistics.mean(
                    row["latent_mean_pairwise_cosine"] for row in selected
                ),
            }
        )

    seeds = sorted({int(row["seed"]) for row in rows})
    fixed_variants = ("fixed-mean32", "fixed-gru32", "conv32")
    comparisons: list[dict[str, Any]] = []
    for seed in seeds:
        best_fixed_name = min(
            fixed_variants,
            key=lambda name: float(np.mean(losses[(seed, name)])),
        )
        best_fixed = losses[(seed, best_fixed_name)] / math.log(2.0)
        learned32 = losses[(seed, "resampler32")] / math.log(2.0)
        frozen32 = losses[(seed, "resampler32-frozen")] / math.log(2.0)
        learned16 = losses[(seed, "resampler16")] / math.log(2.0)
        comparisons.append(
            {
                "seed": seed,
                "best_fixed": best_fixed_name,
                "resampler32_advantage_over_best_fixed_bpb": float(
                    np.mean(best_fixed - learned32)
                ),
                "resampler32_advantage_lcb95_bpb": bootstrap_lower_bound(
                    best_fixed - learned32, seed + 1
                ),
                "resampler32_advantage_over_frozen_bpb": float(
                    np.mean(frozen32 - learned32)
                ),
                "resampler16_delta_vs_resampler32_bpb": float(
                    np.mean(learned16 - learned32)
                ),
            }
        )

    summary_by_name = {row["variant"]: row for row in summary}
    learned32 = summary_by_name["resampler32"]
    learned16 = summary_by_name["resampler16"]
    fixed_best_mean = min(
        (summary_by_name[name] for name in fixed_variants),
        key=lambda row: row["validation_bpb_mean"],
    )
    advantages = [
        row["resampler32_advantage_over_best_fixed_bpb"]
        for row in comparisons
    ]
    frozen_advantages = [
        row["resampler32_advantage_over_frozen_bpb"]
        for row in comparisons
    ]
    half_slot_deltas = [
        row["resampler16_delta_vs_resampler32_bpb"]
        for row in comparisons
    ]
    parameter_ratio = (
        learned32["trainable_parameters"]
        / fixed_best_mean["trainable_parameters"]
    )
    learned_every_seed = min(advantages) > 0.0
    learned_mean = statistics.mean(advantages)
    half_slot_ratio = (
        learned16["validation_bpb_mean"]
        / learned32["validation_bpb_mean"]
    )
    if (
        learned_every_seed
        and learned_mean >= 0.003
        and min(frozen_advantages) > 0.0
        and parameter_ratio <= 1.10
    ):
        verdict = "LEARNED_RESAMPLER_SIGNAL"
    elif half_slot_ratio <= 1.01:
        verdict = "HALF_SLOT_COMPRESSION_SIGNAL"
    else:
        verdict = "FIXED_PATCHES_REMAIN_BEST"
    decision = {
        "verdict": verdict,
        "best_fixed_variant": fixed_best_mean["variant"],
        "resampler32_advantages_bpb": advantages,
        "resampler32_advantage_mean_bpb": learned_mean,
        "resampler32_beats_fixed_every_seed": learned_every_seed,
        "resampler32_advantages_over_frozen_bpb": frozen_advantages,
        "resampler32_parameter_ratio_to_best_fixed": parameter_ratio,
        "resampler16_deltas_vs_resampler32_bpb": half_slot_deltas,
        "resampler16_to_resampler32_bpb_ratio": half_slot_ratio,
        "rule": (
            "Strong learned-resampler signal requires it to beat the best fixed "
            "compressor in every seed by >=0.003 mean BpB, beat its frozen "
            "control in every seed, and use <=10% more trainable parameters. "
            "A half-slot signal permits the 16-slot model within 1% BpB of 32 slots."
        ),
    }
    return summary, {"comparisons": comparisons, "decision": decision}


def write_outputs(output_dir: Path, payload: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "latent_resampler.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with (output_dir / "runs.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(payload["runs"][0].keys())
        )
        writer.writeheader(); writer.writerows(payload["runs"])
    with (output_dir / "summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(payload["summary"][0].keys())
        )
        writer.writeheader(); writer.writerows(payload["summary"])
    with (output_dir / "comparisons.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(payload["comparisons"][0].keys())
        )
        writer.writeheader(); writer.writerows(payload["comparisons"])

    d = payload["decision"]
    lines = [
        "# Test 5.2 — learned continuous latent resampler",
        "",
        f"**Decision:** **{d['verdict']}**",
        "",
        "| Compressor | Runs | Slots | BpB | Params | Compressor params | Attention work/byte | Context bytes/s | Effective rank | Mean slot cosine |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["summary"]:
        lines.append(
            f"| {row['variant']} | {row['runs']} | {row['slots']} | "
            f"{row['validation_bpb_mean']:.4f} ± {row['validation_bpb_std']:.4f} | "
            f"{row['trainable_parameters']:,} | {row['compressor_trainable_parameters']:,} | "
            f"{row['attention_work_ratio_to_byte']:.3%} | "
            f"{row['context_bytes_per_second_mean']:.0f} | "
            f"{row['latent_effective_rank_mean']:.1f} | "
            f"{row['latent_pairwise_cosine_mean']:.3f} |"
        )
    lines += [
        "",
        f"- Best fixed compressor: `{d['best_fixed_variant']}`.",
        f"- Resampler32 advantages over paired best-fixed controls: `{[round(value, 4) for value in d['resampler32_advantages_bpb']]}`; mean `{d['resampler32_advantage_mean_bpb']:+.4f}` BpB.",
        f"- Resampler32 advantages over frozen control: `{[round(value, 4) for value in d['resampler32_advantages_over_frozen_bpb']]}`.",
        f"- Resampler32 parameter ratio to best fixed: `{d['resampler32_parameter_ratio_to_best_fixed']:.3f}`.",
        f"- Resampler16/Resampler32 BpB ratio: `{d['resampler16_to_resampler32_bpb_ratio']:.4f}`; paired deltas `{[round(value, 4) for value in d['resampler16_deltas_vs_resampler32_bpb']]}`.",
        "",
        "Every model predicts the next 16 bytes from identical 128-byte contexts. Global resampling cost is included in measured model throughput; attention-work ratios describe only the downstream latent Transformer.",
    ]
    (output_dir / "VERDICT.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def self_test() -> None:
    cfg = Config(
        context_bytes=32,
        horizon_bytes=4,
        patch_bytes=4,
        slots=8,
        batch_size=2,
        steps=1,
        eval_batches=1,
        d_model=16,
        n_heads=4,
        n_layers=1,
        d_ff=32,
    )
    values = torch.randint(0, 256, (2, 32))
    for variant in (
        "fixed-mean32",
        "fixed-gru32",
        "conv32",
        "resampler32",
        "resampler32-frozen",
        "resampler16",
    ):
        # The production variant names use fixed slot counts; for the compact
        # self-test only verify the corresponding compressor classes directly.
        if variant.startswith("fixed-mean"):
            compressor = FixedMeanCompressor(cfg)
        elif variant.startswith("fixed-gru"):
            compressor = FixedGRUCompressor(cfg)
        elif variant == "conv32":
            compressor = StridedConvCompressor(cfg)
        else:
            compressor = LatentResampler(
                cfg,
                8 if variant != "resampler16" else 4,
                frozen=variant.endswith("frozen"),
            )
        embedded = torch.randn(2, 32, 16)
        output = compressor(embedded)
        if output.ndim != 3 or not torch.isfinite(output).all():
            raise AssertionError((variant, output.shape))
    print("self-test passed")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-bytes", type=Path)
    parser.add_argument("--validation-bytes", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--seeds", default="55201,56302,57403")
    parser.add_argument("--steps", type=int, default=700)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args(argv)

    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    if args.self_test:
        self_test(); return 0
    if args.train_bytes is None or args.validation_bytes is None or args.output_dir is None:
        parser.error("train, validation, and output paths are required")

    cfg = Config(steps=args.steps)
    train = np.frombuffer(args.train_bytes.read_bytes(), dtype=np.uint8)
    validation = np.frombuffer(args.validation_bytes.read_bytes(), dtype=np.uint8)
    variants = (
        "fixed-mean32",
        "fixed-gru32",
        "conv32",
        "resampler32",
        "resampler32-frozen",
        "resampler16",
    )
    seeds = [int(value) for value in args.seeds.split(",") if value.strip()]
    rows: list[dict[str, Any]] = []
    loss_arrays: dict[tuple[int, str], np.ndarray] = {}
    for seed in seeds:
        train_rng = np.random.default_rng(seed + 1)
        validation_rng = np.random.default_rng(seed + 2)
        train_starts = train_rng.integers(
            0,
            len(train) - cfg.context_bytes - cfg.horizon_bytes,
            size=cfg.steps * cfg.batch_size,
            endpoint=False,
            dtype=np.int64,
        )
        validation_starts = validation_rng.integers(
            0,
            len(validation) - cfg.context_bytes - cfg.horizon_bytes,
            size=cfg.eval_batches * cfg.batch_size,
            endpoint=False,
            dtype=np.int64,
        )
        train_contexts, train_targets = build_windows(train, train_starts, cfg)
        validation_contexts, validation_targets = build_windows(
            validation, validation_starts, cfg
        )
        for variant in variants:
            result, losses = train_variant(
                variant,
                seed,
                train_contexts,
                train_targets,
                validation_contexts,
                validation_targets,
                cfg,
            )
            rows.append(asdict(result))
            loss_arrays[(seed, variant)] = losses

    summary, aggregate_data = aggregate(rows, loss_arrays)
    payload = {
        "metadata": {
            "task": "predict next 16 raw UTF-8 bytes from a 128-byte context",
            "dataset": "WikiText-2 raw UTF-8 bytes",
            "seeds": seeds,
            "steps_per_variant": cfg.steps,
            "context_bytes": cfg.context_bytes,
            "horizon_bytes": cfg.horizon_bytes,
            "evaluation_examples_per_seed": cfg.eval_batches * cfg.batch_size,
            "train_corpus_bytes": int(len(train)),
            "validation_corpus_bytes": int(len(validation)),
        },
        "runs": rows,
        "summary": summary,
        "comparisons": aggregate_data["comparisons"],
        "decision": aggregate_data["decision"],
    }
    write_outputs(args.output_dir, payload)
    print((args.output_dir / "VERDICT.md").read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
