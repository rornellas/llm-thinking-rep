#!/usr/bin/env python3
"""Test 5.4: a proper autoregressive language-model test for compact byte patches.

Earlier context-to-vector screens supplied only 16 targets per 128-byte window
and did not beat an unconditional byte-frequency baseline.  This benchmark uses
a dense autoregressive objective: every sampled 144-byte window contributes the
same target span, bytes 16..143 (128 targets), for every representation.

Variants
--------
* byte: causal Transformer over individual bytes;
* patch4-gru: 4-byte local GRU encoder and autoregressive local decoder;
* patch8-gru: 8-byte patches;
* patch16-gru: 16-byte patches;
* patch8-mean: order-insensitive mean encoder control with the same decoder.

For patch size p, all patches preceding each target patch are processed by a
causal latent Transformer.  The hidden state for patch j-1 initializes a local
GRU decoder that predicts patch j byte by byte with teacher forcing.  All
variants predict the *same* 128 raw target bytes from the same raw windows.

Held-out evaluation also rolls input windows among examples while retaining the
targets.  This in-distribution intervention measures whether the model truly
uses the context-target relationship.  Bits per byte, actual CPU throughput,
parameter count, and downstream attention work are reported across three paired
seeds.
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
    window_bytes: int = 144
    target_start: int = 16
    target_bytes: int = 128
    batch_size: int = 8
    steps: int = 1000
    eval_batches: int = 40
    d_model: int = 64
    n_heads: int = 4
    n_layers: int = 2
    d_ff: int = 192
    lr: float = 5e-4
    weight_decay: float = 0.03
    grad_clip: float = 1.0


@dataclass
class RunResult:
    seed: int
    variant: str
    patch_size: int
    validation_bits_per_byte: float
    rolled_context_bits_per_byte: float
    rolled_context_delta_bpb: float
    rolled_context_lcb95_bpb: float
    final_train_bits_per_byte: float
    best_validation_bits_per_byte: float
    trainable_parameters: int
    model_bytes_per_second: float
    latent_positions: int
    downstream_attention_work_ratio_to_byte: float
    training_seconds: float


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def causal_mask(length: int, device: torch.device) -> torch.Tensor:
    mask = torch.full((length, length), float("-inf"), device=device)
    return torch.triu(mask, diagonal=1)


def build_windows(
    data: np.ndarray,
    starts: np.ndarray,
    cfg: Config,
) -> torch.Tensor:
    windows = np.stack(
        [
            data[int(start):int(start) + cfg.window_bytes]
            for start in starts
        ]
    ).astype(np.int64, copy=False)
    return torch.from_numpy(windows)


class ByteCausalLM(nn.Module):
    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.cfg = cfg
        self.patch_size = 1
        self.byte_embedding = nn.Embedding(256, cfg.d_model)
        self.position_embedding = nn.Embedding(cfg.window_bytes - 1, cfg.d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.n_heads,
            dim_feedforward=cfg.d_ff,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=cfg.n_layers)
        self.norm = nn.LayerNorm(cfg.d_model)
        self.output = nn.Linear(cfg.d_model, 256)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.byte_embedding.weight, std=0.02)
        nn.init.normal_(self.position_embedding.weight, std=0.02)
        nn.init.normal_(self.output.weight, std=0.02)
        nn.init.zeros_(self.output.bias)

    @property
    def latent_positions(self) -> int:
        return self.cfg.window_bytes - 1

    def forward(self, windows: torch.Tensor) -> torch.Tensor:
        inputs = windows[:, :-1]
        positions = torch.arange(inputs.shape[1], device=windows.device)
        hidden = (
            self.byte_embedding(inputs)
            + self.position_embedding(positions)[None, :, :]
        )
        hidden = self.transformer(
            hidden,
            mask=causal_mask(hidden.shape[1], hidden.device),
        )
        # Hidden at position t predicts byte t+1.  Select predictions for the
        # shared target span [target_start, window_bytes).
        selected = hidden[
            :,
            self.cfg.target_start - 1:self.cfg.window_bytes - 1,
        ]
        if selected.shape[1] != self.cfg.target_bytes:
            raise AssertionError(selected.shape)
        return self.output(self.norm(selected))


class PatchCausalLM(nn.Module):
    def __init__(self, cfg: Config, patch_size: int, encoder_type: str) -> None:
        super().__init__()
        if cfg.window_bytes % patch_size != 0:
            raise ValueError((cfg.window_bytes, patch_size))
        if cfg.target_start % patch_size != 0:
            raise ValueError((cfg.target_start, patch_size))
        self.cfg = cfg
        self.patch_size = patch_size
        self.encoder_type = encoder_type
        self.total_patches = cfg.window_bytes // patch_size
        self.input_patches = self.total_patches - 1
        self.target_patch_start = cfg.target_start // patch_size
        self.target_patches = cfg.target_bytes // patch_size
        self.byte_embedding = nn.Embedding(257, cfg.d_model)  # ID 256 = BOS
        self.patch_position_embedding = nn.Embedding(
            self.input_patches, cfg.d_model
        )
        if encoder_type == "gru":
            self.local_encoder = nn.GRU(
                cfg.d_model, cfg.d_model, batch_first=True
            )
        elif encoder_type == "mean":
            self.local_encoder = nn.Identity()
        else:
            raise ValueError(encoder_type)
        layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.n_heads,
            dim_feedforward=cfg.d_ff,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.latent_transformer = nn.TransformerEncoder(
            layer, num_layers=cfg.n_layers
        )
        self.context_to_decoder = nn.Linear(cfg.d_model, cfg.d_model)
        self.local_decoder = nn.GRU(
            cfg.d_model, cfg.d_model, batch_first=True
        )
        self.decoder_norm = nn.LayerNorm(cfg.d_model)
        self.output = nn.Linear(cfg.d_model, 256)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.byte_embedding.weight, std=0.02)
        nn.init.normal_(self.patch_position_embedding.weight, std=0.02)
        nn.init.normal_(self.output.weight, std=0.02)
        nn.init.zeros_(self.output.bias)

    @property
    def latent_positions(self) -> int:
        return self.input_patches

    def encode_patches(self, patches: torch.Tensor) -> torch.Tensor:
        batch, count, width = patches.shape
        embedded = self.byte_embedding(patches).reshape(
            batch * count, width, self.cfg.d_model
        )
        if self.encoder_type == "gru":
            encoded, _ = self.local_encoder(embedded)
            latent = encoded[:, -1]
        else:
            latent = embedded.mean(dim=1)
        return latent.reshape(batch, count, self.cfg.d_model)

    def forward(self, windows: torch.Tensor) -> torch.Tensor:
        batch = windows.shape[0]
        patches = windows.reshape(
            batch, self.total_patches, self.patch_size
        )
        input_patches = patches[:, :self.input_patches]
        latent = self.encode_patches(input_patches)
        positions = torch.arange(self.input_patches, device=windows.device)
        latent = (
            latent
            + self.patch_position_embedding(positions)[None, :, :]
        )
        latent = self.latent_transformer(
            latent,
            mask=causal_mask(self.input_patches, windows.device),
        )
        # Target patch j is predicted from latent patch j-1.  Shared raw target
        # span starts at target_patch_start and ends at total_patches-1.
        first_context_index = self.target_patch_start - 1
        context_states = latent[
            :,
            first_context_index:self.total_patches - 1,
        ]
        target_patches = patches[
            :,
            self.target_patch_start:self.total_patches,
        ]
        if context_states.shape[1] != self.target_patches:
            raise AssertionError((context_states.shape, target_patches.shape))

        flat_context = context_states.reshape(
            batch * self.target_patches, self.cfg.d_model
        )
        flat_targets = target_patches.reshape(
            batch * self.target_patches, self.patch_size
        )
        bos = torch.full(
            (flat_targets.shape[0], 1),
            256,
            dtype=torch.long,
            device=windows.device,
        )
        decoder_tokens = torch.cat(
            [bos, flat_targets[:, :-1]], dim=1
        )
        decoder_inputs = self.byte_embedding(decoder_tokens)
        initial = torch.tanh(
            self.context_to_decoder(flat_context)
        ).unsqueeze(0)
        decoded, _ = self.local_decoder(decoder_inputs, initial)
        logits = self.output(self.decoder_norm(decoded))
        return logits.reshape(
            batch, self.target_patches * self.patch_size, 256
        )


def make_model(cfg: Config, variant: str) -> nn.Module:
    if variant == "byte":
        return ByteCausalLM(cfg)
    if variant == "patch4-gru":
        return PatchCausalLM(cfg, 4, "gru")
    if variant == "patch8-gru":
        return PatchCausalLM(cfg, 8, "gru")
    if variant == "patch16-gru":
        return PatchCausalLM(cfg, 16, "gru")
    if variant == "patch8-mean":
        return PatchCausalLM(cfg, 8, "mean")
    raise ValueError(variant)


def targets_from_windows(windows: torch.Tensor, cfg: Config) -> torch.Tensor:
    targets = windows[:, cfg.target_start:cfg.window_bytes]
    if targets.shape[1] != cfg.target_bytes:
        raise AssertionError(targets.shape)
    return targets


@torch.no_grad()
def evaluate(
    model: nn.Module,
    windows: torch.Tensor,
    cfg: Config,
    *,
    rolled_context: bool,
) -> tuple[float, np.ndarray, float]:
    model.eval()
    losses: list[np.ndarray] = []
    total_target_bytes = 0
    started = time.perf_counter()
    for offset in range(0, len(windows), cfg.batch_size):
        original = windows[offset:offset + cfg.batch_size]
        targets = targets_from_windows(original, cfg)
        if rolled_context:
            inputs = torch.roll(original, shifts=1, dims=0)
            # Retain the original held-out target span while replacing all
            # preceding/context bytes with another in-distribution example.
            inputs = inputs.clone()
            inputs[:, cfg.target_start:] = original[:, cfg.target_start:]
        else:
            inputs = original
        logits = model(inputs)
        per_byte = F.cross_entropy(
            logits.reshape(-1, 256),
            targets.reshape(-1),
            reduction="none",
        ).reshape(len(original), cfg.target_bytes)
        losses.append(per_byte.mean(dim=1).cpu().numpy())
        total_target_bytes += int(targets.numel())
    elapsed = time.perf_counter() - started
    values = np.concatenate(losses)
    return (
        float(np.mean(values)),
        values,
        total_target_bytes / max(elapsed, EPS),
    )


def bootstrap_lcb(values: np.ndarray, seed: int, samples: int = 5000) -> float:
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(samples, len(values)))
    return float(np.quantile(values[indices].mean(axis=1), 0.05))


def train_variant(
    variant: str,
    seed: int,
    train_windows: torch.Tensor,
    validation_windows: torch.Tensor,
    cfg: Config,
) -> tuple[RunResult, np.ndarray]:
    set_seed(seed)
    model = make_model(cfg, variant)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    best = math.inf
    final_train = math.nan
    started = time.perf_counter()
    for step in range(1, cfg.steps + 1):
        offset = (step - 1) * cfg.batch_size
        windows = train_windows[offset:offset + cfg.batch_size]
        targets = targets_from_windows(windows, cfg)
        logits = model(windows)
        loss = F.cross_entropy(
            logits.reshape(-1, 256), targets.reshape(-1)
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()
        final_train = float(loss.detach())
        if step == 1 or step % 250 == 0 or step == cfg.steps:
            validation_loss, _, _ = evaluate(
                model, validation_windows, cfg, rolled_context=False
            )
            best = min(best, validation_loss)
            print(
                f"seed={seed} {variant} step={step}/{cfg.steps} "
                f"train-bpb={final_train / math.log(2.0):.4f} "
                f"val-bpb={validation_loss / math.log(2.0):.4f}",
                flush=True,
            )
    training_seconds = time.perf_counter() - started
    validation_loss, correct_values, throughput = evaluate(
        model, validation_windows, cfg, rolled_context=False
    )
    rolled_loss, rolled_values, _ = evaluate(
        model, validation_windows, cfg, rolled_context=True
    )
    best = min(best, validation_loss)
    delta_values = (rolled_values - correct_values) / math.log(2.0)
    latent_positions = int(model.latent_positions)
    byte_positions = cfg.window_bytes - 1
    result = RunResult(
        seed=seed,
        variant=variant,
        patch_size=int(model.patch_size),
        validation_bits_per_byte=validation_loss / math.log(2.0),
        rolled_context_bits_per_byte=rolled_loss / math.log(2.0),
        rolled_context_delta_bpb=float(np.mean(delta_values)),
        rolled_context_lcb95_bpb=bootstrap_lcb(
            delta_values, seed + sum(ord(char) for char in variant)
        ),
        final_train_bits_per_byte=final_train / math.log(2.0),
        best_validation_bits_per_byte=best / math.log(2.0),
        trainable_parameters=sum(
            parameter.numel() for parameter in model.parameters()
        ),
        model_bytes_per_second=throughput,
        latent_positions=latent_positions,
        downstream_attention_work_ratio_to_byte=(
            latent_positions / byte_positions
        ) ** 2,
        training_seconds=training_seconds,
    )
    return result, correct_values / math.log(2.0)


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
                "patch_size": int(selected[0]["patch_size"]),
                "validation_bpb_mean": statistics.mean(
                    row["validation_bits_per_byte"] for row in selected
                ),
                "validation_bpb_std": statistics.pstdev(
                    row["validation_bits_per_byte"] for row in selected
                ),
                "rolled_delta_bpb_mean": statistics.mean(
                    row["rolled_context_delta_bpb"] for row in selected
                ),
                "worst_rolled_lcb95_bpb": min(
                    row["rolled_context_lcb95_bpb"] for row in selected
                ),
                "trainable_parameters": int(selected[0]["trainable_parameters"]),
                "latent_positions": int(selected[0]["latent_positions"]),
                "attention_work_ratio_to_byte": float(
                    selected[0]["downstream_attention_work_ratio_to_byte"]
                ),
                "model_bytes_per_second_mean": statistics.mean(
                    row["model_bytes_per_second"] for row in selected
                ),
            }
        )

    indexed = {row["variant"]: row for row in summary}
    byte = indexed["byte"]
    comparisons: list[dict[str, Any]] = []
    seeds = sorted({int(row["seed"]) for row in rows})
    for seed in seeds:
        byte_losses = losses[(seed, "byte")]
        for variant in variants:
            if variant == "byte":
                continue
            patch_losses = losses[(seed, variant)]
            comparisons.append(
                {
                    "seed": seed,
                    "variant": variant,
                    "delta_vs_byte_bpb": float(
                        np.mean(patch_losses - byte_losses)
                    ),
                    "delta_vs_byte_lcb95_bpb": bootstrap_lcb(
                        patch_losses - byte_losses,
                        seed + sum(ord(char) for char in variant) + 10000,
                    ),
                }
            )

    def decision_for(variant: str, max_ratio: float) -> dict[str, Any]:
        row = indexed[variant]
        ratio = row["validation_bpb_mean"] / byte["validation_bpb_mean"]
        paired = [
            item for item in comparisons if item["variant"] == variant
        ]
        context_ok = (
            row["rolled_delta_bpb_mean"] >= 0.05
            and row["worst_rolled_lcb95_bpb"] > 0.0
        )
        return {
            "bpb_ratio_to_byte": ratio,
            "context_dependent": context_ok,
            "paired_deltas_vs_byte": [
                item["delta_vs_byte_bpb"] for item in paired
            ],
            "passes": ratio <= max_ratio and context_ok,
        }

    patch4 = decision_for("patch4-gru", 1.02)
    patch8 = decision_for("patch8-gru", 1.02)
    patch16 = decision_for("patch16-gru", 1.03)
    mean8 = indexed["patch8-mean"]
    order_advantage = (
        mean8["validation_bpb_mean"]
        - indexed["patch8-gru"]["validation_bpb_mean"]
    )
    if patch8["passes"] and order_advantage >= 0.005:
        verdict = "AUTOREGRESSIVE_PATCH8_SIGNAL"
    elif patch4["passes"]:
        verdict = "AUTOREGRESSIVE_PATCH4_SIGNAL"
    elif patch16["passes"]:
        verdict = "AUTOREGRESSIVE_PATCH16_SIGNAL"
    else:
        verdict = "AUTOREGRESSIVE_PATCH_FAIL"
    decision = {
        "verdict": verdict,
        "patch4": patch4,
        "patch8": patch8,
        "patch16": patch16,
        "patch8_gru_advantage_over_mean_bpb": order_advantage,
        "byte_validation_bpb": byte["validation_bpb_mean"],
        "rule": (
            "Patch4/8 require BpB <=1.02x byte and rolled-context cost >=0.05 "
            "BpB with positive LCB95 in every seed. Patch16 permits 1.03x. "
            "Strong patch8 signal additionally requires GRU order encoding to "
            "beat the mean encoder by >=0.005 BpB."
        ),
    }
    return summary, {"comparisons": comparisons, "decision": decision}


def write_outputs(output_dir: Path, payload: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "autoregressive_patch_lm.json").write_text(
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
        "# Test 5.4 — autoregressive byte-patch language model",
        "",
        f"**Decision:** **{d['verdict']}**",
        "",
        "| Model | Runs | Patch | BpB | Rolled Δ | Worst rolled LCB95 | Params | Latent positions | Attention work/byte | Target bytes/s |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["summary"]:
        lines.append(
            f"| {row['variant']} | {row['runs']} | {row['patch_size']} | "
            f"{row['validation_bpb_mean']:.4f} ± {row['validation_bpb_std']:.4f} | "
            f"{row['rolled_delta_bpb_mean']:+.4f} | "
            f"{row['worst_rolled_lcb95_bpb']:+.4f} | "
            f"{row['trainable_parameters']:,} | {row['latent_positions']} | "
            f"{row['attention_work_ratio_to_byte']:.3%} | "
            f"{row['model_bytes_per_second_mean']:.0f} |"
        )
    lines += [
        "",
        f"- Byte baseline: `{d['byte_validation_bpb']:.4f}` BpB.",
        f"- Patch4: ratio `{d['patch4']['bpb_ratio_to_byte']:.4f}`, context `{d['patch4']['context_dependent']}`.",
        f"- Patch8: ratio `{d['patch8']['bpb_ratio_to_byte']:.4f}`, context `{d['patch8']['context_dependent']}`.",
        f"- Patch16: ratio `{d['patch16']['bpb_ratio_to_byte']:.4f}`, context `{d['patch16']['context_dependent']}`.",
        f"- Patch8 GRU advantage over mean encoder: `{d['patch8_gru_advantage_over_mean_bpb']:+.4f}` BpB.",
        "",
        "Every model receives the same 144 raw bytes and predicts the same 128 raw target bytes. Patch models are autoregressive within each target patch; rolled-context evaluation keeps targets fixed and replaces preceding context with another held-out example.",
    ]
    (output_dir / "VERDICT.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def self_test() -> None:
    cfg = Config(
        window_bytes=48,
        target_start=16,
        target_bytes=32,
        batch_size=2,
        steps=1,
        eval_batches=1,
        d_model=16,
        n_heads=4,
        n_layers=1,
        d_ff=32,
    )
    windows = torch.randint(0, 256, (2, 48))
    for variant in (
        "byte",
        "patch4-gru",
        "patch8-gru",
        "patch16-gru",
        "patch8-mean",
    ):
        model = make_model(cfg, variant)
        logits = model(windows)
        if logits.shape != (2, 32, 256):
            raise AssertionError((variant, logits.shape))
        F.cross_entropy(
            logits.reshape(-1, 256),
            targets_from_windows(windows, cfg).reshape(-1),
        ).backward()
    print("self-test passed")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-bytes", type=Path, required=False)
    parser.add_argument("--validation-bytes", type=Path, required=False)
    parser.add_argument("--output-dir", type=Path, required=False)
    parser.add_argument("--seeds", default="61401,62502,63603")
    parser.add_argument("--steps", type=int, default=1000)
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
        "byte",
        "patch4-gru",
        "patch8-gru",
        "patch16-gru",
        "patch8-mean",
    )
    seeds = [int(value) for value in args.seeds.split(",") if value.strip()]
    rows: list[dict[str, Any]] = []
    loss_arrays: dict[tuple[int, str], np.ndarray] = {}
    for seed in seeds:
        train_rng = np.random.default_rng(seed + 1)
        validation_rng = np.random.default_rng(seed + 2)
        train_starts = train_rng.integers(
            0,
            len(train) - cfg.window_bytes,
            size=cfg.steps * cfg.batch_size,
            endpoint=False,
            dtype=np.int64,
        )
        validation_starts = validation_rng.integers(
            0,
            len(validation) - cfg.window_bytes,
            size=cfg.eval_batches * cfg.batch_size,
            endpoint=False,
            dtype=np.int64,
        )
        train_windows = build_windows(train, train_starts, cfg)
        validation_windows = build_windows(
            validation, validation_starts, cfg
        )
        for variant in variants:
            result, losses = train_variant(
                variant,
                seed,
                train_windows,
                validation_windows,
                cfg,
            )
            rows.append(asdict(result))
            loss_arrays[(seed, variant)] = losses

    summary, aggregate_data = aggregate(rows, loss_arrays)
    payload = {
        "metadata": {
            "task": "autoregressive prediction of bytes 16..143 from each 144-byte WikiText-2 window",
            "dataset": "WikiText-2 raw UTF-8 bytes",
            "seeds": seeds,
            "steps_per_variant": cfg.steps,
            "window_bytes": cfg.window_bytes,
            "target_start": cfg.target_start,
            "target_bytes": cfg.target_bytes,
            "evaluation_examples_per_seed": cfg.eval_batches * cfg.batch_size,
            "train_corpus_bytes": int(len(train)),
            "validation_corpus_bytes": int(len(validation)),
            "intervention": "roll context windows across held-out examples while retaining target span",
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
