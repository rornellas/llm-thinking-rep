#!/usr/bin/env python3
"""Test 5.0: causal adaptive byte patches with exact byte likelihood.

The experiment isolates sequence representation from Modal-MoE.  Every variant
uses the same trainable model and predicts the same UTF-8 byte stream.  Only the
segmentation and the number of global positions differ:

* ``byte-context``: one byte per patch, with a context-matched long sequence;
* ``fixed4``: deterministic four-byte patches;
* ``random-matched``: random patch lengths sampled from the adaptive length
  distribution but independent of content;
* ``entropy-adaptive``: a causal boundary rule based on the entropy of the next
  byte under a train-only smoothed byte bigram model.

For a patch p_t the global Transformer receives p_0...p_{t-1}.  A local GRU
then predicts every byte in p_t autoregressively.  Because segmentation is a
fixed causal rule, the summed byte cross-entropy is a valid likelihood and can
be compared as bits per byte across representations.
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
from statistics import mean
from typing import Any, Callable, Sequence

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


BYTE_VALUES = 256
LOCAL_BOS = 256
PAD_BYTE = 257
BYTE_EMBED_VOCAB = 258


@dataclass(frozen=True)
class Config:
    d_model: int = 96
    n_heads: int = 4
    n_layers: int = 2
    global_ff: int = 384
    max_patch_len: int = 8
    context_bytes: int = 192
    batch_size: int = 4
    steps: int = 400
    eval_interval: int = 100
    validation_windows: int = 32
    test_windows: int = 128
    learning_rate: float = 4e-4
    weight_decay: float = 0.02
    grad_clip: float = 1.0


@dataclass
class VariantResult:
    variant: str
    mean_patch_length_train: float
    mean_patch_length_test: float
    patches_per_kib: float
    global_positions: int
    context_bytes_mean: float
    train_bytes_seen: int
    global_positions_seen: int
    model_parameters: int
    best_validation_bits_per_byte: float
    test_bits_per_byte: float
    test_nats_per_byte: float
    test_bytes: int
    train_seconds: float
    test_seconds: float
    train_bytes_per_second: float
    test_bytes_per_second: float
    estimated_attention_pairs_seen: int
    attention_pairs_ratio_to_byte: float
    boundary_entropy_lift_bits: float


@dataclass
class PatchStream:
    patches: np.ndarray
    lengths: np.ndarray
    starts: np.ndarray
    byte_count: int
    mean_patch_length: float
    boundary_entropy_lift_bits: float

    def patch_index_for_offsets(self, offsets: np.ndarray, patch_count: int) -> np.ndarray:
        indices = np.searchsorted(self.starts, offsets, side="right") - 1
        max_start = max(0, len(self.lengths) - patch_count - 1)
        return np.clip(indices, 0, max_start).astype(np.int64, copy=False)

    def batch_from_indices(
        self,
        indices: np.ndarray,
        patch_count: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch = len(indices)
        max_len = self.patches.shape[1]
        input_patches = np.empty((batch, patch_count, max_len), dtype=np.int64)
        target_patches = np.empty((batch, patch_count, max_len), dtype=np.int64)
        input_lengths = np.empty((batch, patch_count), dtype=np.int64)
        target_lengths = np.empty((batch, patch_count), dtype=np.int64)
        for row, start in enumerate(indices.tolist()):
            stop = start + patch_count
            input_patches[row] = self.patches[start:stop]
            target_patches[row] = self.patches[start + 1:stop + 1]
            input_lengths[row] = self.lengths[start:stop]
            target_lengths[row] = self.lengths[start + 1:stop + 1]
        return (
            torch.from_numpy(input_patches).to(device),
            torch.from_numpy(input_lengths).to(device),
            torch.from_numpy(target_patches).to(device),
            torch.from_numpy(target_lengths).to(device),
        )


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_byte_manifest(path: Path, train_limit: int, validation_limit: int, test_limit: int) -> dict[str, np.ndarray]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    result: dict[str, np.ndarray] = {}
    for split, limit in (
        ("train", train_limit),
        ("validation", validation_limit),
        ("test", test_limit),
    ):
        split_path = Path(manifest[f"{split}_path"])
        values = np.load(split_path, mmap_mode="r")
        if values.dtype != np.uint8:
            values = np.asarray(values, dtype=np.uint8)
        result[split] = np.asarray(values[: min(limit, len(values))], dtype=np.uint8)
    return result


def bigram_entropy_bits(train_bytes: np.ndarray, smoothing: float = 0.25) -> tuple[np.ndarray, np.ndarray]:
    counts = np.full((BYTE_VALUES, BYTE_VALUES), smoothing, dtype=np.float64)
    previous = train_bytes[:-1].astype(np.int64, copy=False)
    following = train_bytes[1:].astype(np.int64, copy=False)
    np.add.at(counts, (previous, following), 1.0)
    probabilities = counts / counts.sum(axis=1, keepdims=True)
    entropy = -np.sum(probabilities * np.log2(probabilities), axis=1)
    return entropy.astype(np.float32), probabilities.astype(np.float32)


def segment_fixed(data: np.ndarray, patch_length: int, max_patch_len: int, entropy: np.ndarray) -> PatchStream:
    if patch_length < 1 or patch_length > max_patch_len:
        raise ValueError(patch_length)
    lengths = np.full(math.ceil(len(data) / patch_length), patch_length, dtype=np.int64)
    if lengths.size:
        lengths[-1] = len(data) - patch_length * (len(lengths) - 1)
    return materialize_stream(data, lengths, max_patch_len, entropy)


def adaptive_lengths(
    data: np.ndarray,
    entropy: np.ndarray,
    threshold: float,
    min_len: int,
    max_len: int,
) -> np.ndarray:
    lengths: list[int] = []
    index = 0
    total = len(data)
    while index < total:
        length = 0
        while index + length < total and length < max_len:
            current = int(data[index + length])
            length += 1
            if length >= min_len and float(entropy[current]) >= threshold:
                break
        lengths.append(length)
        index += length
    return np.asarray(lengths, dtype=np.int64)


def calibrate_adaptive_threshold(
    train_bytes: np.ndarray,
    entropy: np.ndarray,
    target_mean: float,
    min_len: int,
    max_len: int,
    calibration_bytes: int = 300_000,
) -> tuple[float, np.ndarray]:
    sample = train_bytes[: min(calibration_bytes, len(train_bytes))]
    low = float(np.min(entropy)) - 1e-5
    high = float(np.max(entropy)) + 1e-5
    best: tuple[float, float, np.ndarray] | None = None
    for _ in range(24):
        threshold = (low + high) * 0.5
        lengths = adaptive_lengths(sample, entropy, threshold, min_len, max_len)
        observed = float(np.mean(lengths))
        error = abs(observed - target_mean)
        if best is None or error < best[0]:
            best = (error, threshold, lengths)
        if observed < target_mean:
            low = threshold
        else:
            high = threshold
    assert best is not None
    return best[1], best[2]


def segment_adaptive(
    data: np.ndarray,
    entropy: np.ndarray,
    threshold: float,
    min_len: int,
    max_len: int,
) -> PatchStream:
    return materialize_stream(
        data,
        adaptive_lengths(data, entropy, threshold, min_len, max_len),
        max_len,
        entropy,
    )


def segment_random_matched(
    data: np.ndarray,
    reference_lengths: np.ndarray,
    max_patch_len: int,
    entropy: np.ndarray,
    seed: int,
) -> PatchStream:
    counts = np.bincount(reference_lengths, minlength=max_patch_len + 1)[1:max_patch_len + 1].astype(np.float64)
    probabilities = counts / counts.sum()
    rng = np.random.default_rng(seed)
    lengths: list[int] = []
    covered = 0
    choices = np.arange(1, max_patch_len + 1, dtype=np.int64)
    while covered < len(data):
        length = int(rng.choice(choices, p=probabilities))
        length = min(length, len(data) - covered)
        lengths.append(length)
        covered += length
    return materialize_stream(data, np.asarray(lengths, dtype=np.int64), max_patch_len, entropy)


def materialize_stream(
    data: np.ndarray,
    lengths: np.ndarray,
    max_patch_len: int,
    entropy: np.ndarray,
) -> PatchStream:
    if int(np.sum(lengths)) != len(data):
        raise ValueError((int(np.sum(lengths)), len(data)))
    patches = np.full((len(lengths), max_patch_len), PAD_BYTE, dtype=np.int64)
    starts = np.empty(len(lengths), dtype=np.int64)
    index = 0
    boundary_entropies: list[float] = []
    for row, length_value in enumerate(lengths.tolist()):
        length = int(length_value)
        starts[row] = index
        patches[row, :length] = data[index:index + length]
        boundary_entropies.append(float(entropy[int(data[index + length - 1])]))
        index += length
    all_entropy = entropy[data.astype(np.int64, copy=False)]
    lift = float(np.mean(boundary_entropies) - np.mean(all_entropy)) if len(lengths) else 0.0
    return PatchStream(
        patches=patches,
        lengths=lengths,
        starts=starts,
        byte_count=len(data),
        mean_patch_length=float(np.mean(lengths)),
        boundary_entropy_lift_bits=lift,
    )


class CausalPatchByteLM(nn.Module):
    def __init__(self, cfg: Config, max_global_positions: int) -> None:
        super().__init__()
        self.cfg = cfg
        self.byte_embedding = nn.Embedding(BYTE_EMBED_VOCAB, cfg.d_model)
        self.local_position = nn.Embedding(cfg.max_patch_len, cfg.d_model)
        self.length_embedding = nn.Embedding(cfg.max_patch_len + 1, cfg.d_model)
        self.patch_encoder = nn.Sequential(
            nn.LayerNorm(cfg.d_model),
            nn.Linear(cfg.d_model, cfg.d_model * 2),
            nn.GELU(),
            nn.Linear(cfg.d_model * 2, cfg.d_model),
        )
        self.global_position = nn.Embedding(max_global_positions, cfg.d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.n_heads,
            dim_feedforward=cfg.global_ff,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=cfg.n_layers, enable_nested_tensor=False)
        self.context_norm = nn.LayerNorm(cfg.d_model)
        self.context_to_hidden = nn.Linear(cfg.d_model, cfg.d_model)
        self.local_decoder = nn.GRU(cfg.d_model, cfg.d_model, batch_first=True)
        self.byte_head = nn.Linear(cfg.d_model, BYTE_VALUES, bias=False)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.byte_embedding.weight, std=0.02)
        nn.init.normal_(self.local_position.weight, std=0.02)
        nn.init.normal_(self.length_embedding.weight, std=0.02)
        nn.init.normal_(self.global_position.weight, std=0.02)
        nn.init.xavier_uniform_(self.context_to_hidden.weight)
        nn.init.zeros_(self.context_to_hidden.bias)
        nn.init.normal_(self.byte_head.weight, std=0.02)

    def encode_patches(self, patches: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        batch, positions, width = patches.shape
        local_positions = torch.arange(width, device=patches.device)
        embedded = self.byte_embedding(patches) + self.local_position(local_positions)[None, None, :, :]
        mask = local_positions[None, None, :] < lengths[:, :, None]
        pooled = torch.sum(embedded * mask[:, :, :, None], dim=2)
        pooled = pooled / torch.sqrt(lengths.clamp_min(1).to(embedded.dtype))[:, :, None]
        pooled = pooled + self.length_embedding(lengths)
        return self.patch_encoder(pooled)

    def forward_logits(
        self,
        input_patches: torch.Tensor,
        input_lengths: torch.Tensor,
        target_patches: torch.Tensor,
        target_lengths: torch.Tensor,
    ) -> torch.Tensor:
        batch, positions, width = target_patches.shape
        encoded = self.encode_patches(input_patches, input_lengths)
        global_positions = torch.arange(positions, device=encoded.device)
        encoded = encoded + self.global_position(global_positions)[None, :, :]
        causal_mask = torch.triu(
            torch.ones((positions, positions), dtype=torch.bool, device=encoded.device),
            diagonal=1,
        )
        context = self.context_norm(self.transformer(encoded, mask=causal_mask))

        decoder_tokens = torch.full_like(target_patches, LOCAL_BOS)
        decoder_tokens[:, :, 1:] = target_patches[:, :, :-1]
        local_positions = torch.arange(width, device=encoded.device)
        decoder_input = self.byte_embedding(decoder_tokens) + self.local_position(local_positions)[None, None, :, :]
        decoder_input = decoder_input.reshape(batch * positions, width, self.cfg.d_model)
        initial = torch.tanh(self.context_to_hidden(context)).reshape(1, batch * positions, self.cfg.d_model)
        decoded, _ = self.local_decoder(decoder_input, initial)
        return self.byte_head(decoded).reshape(batch, positions, width, BYTE_VALUES)

    def nll(
        self,
        input_patches: torch.Tensor,
        input_lengths: torch.Tensor,
        target_patches: torch.Tensor,
        target_lengths: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits = self.forward_logits(input_patches, input_lengths, target_patches, target_lengths)
        width = target_patches.shape[-1]
        positions = torch.arange(width, device=target_patches.device)
        mask = positions[None, None, :] < target_lengths[:, :, None]
        losses = F.cross_entropy(
            logits.reshape(-1, BYTE_VALUES),
            target_patches.clamp_max(BYTE_VALUES - 1).reshape(-1),
            reduction="none",
        ).reshape_as(target_patches)
        nll_sum = torch.sum(losses * mask)
        byte_count = torch.sum(mask)
        return nll_sum / byte_count.clamp_min(1), nll_sum.detach(), byte_count.detach()


def build_streams(
    data: dict[str, np.ndarray],
    cfg: Config,
    entropy: np.ndarray,
    threshold: float,
    adaptive_train_lengths: np.ndarray,
    variant: str,
    seed: int,
) -> dict[str, PatchStream]:
    streams: dict[str, PatchStream] = {}
    for split_index, split in enumerate(("train", "validation", "test")):
        values = data[split]
        if variant == "byte-context":
            stream = segment_fixed(values, 1, cfg.max_patch_len, entropy)
        elif variant == "fixed4":
            stream = segment_fixed(values, 4, cfg.max_patch_len, entropy)
        elif variant == "entropy-adaptive":
            stream = segment_adaptive(values, entropy, threshold, 1, cfg.max_patch_len)
        elif variant == "random-matched":
            stream = segment_random_matched(
                values,
                adaptive_train_lengths,
                cfg.max_patch_len,
                entropy,
                seed + 1000 * (split_index + 1),
            )
        else:
            raise ValueError(variant)
        streams[split] = stream
    return streams


def patch_count_for_variant(variant: str, cfg: Config, train_stream: PatchStream) -> int:
    if variant == "byte-context":
        return cfg.context_bytes
    return max(8, int(round(cfg.context_bytes / train_stream.mean_patch_length)))


def shared_offsets(byte_count: int, batch_size: int, steps: int, seed: int, margin: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    high = max(1, byte_count - margin)
    return rng.integers(0, high, size=(steps, batch_size), endpoint=False, dtype=np.int64)


@torch.no_grad()
def evaluate_windows(
    model: CausalPatchByteLM,
    stream: PatchStream,
    patch_count: int,
    windows: int,
    batch_size: int,
    device: torch.device,
) -> tuple[float, int, float]:
    model.eval()
    max_index = max(1, len(stream.lengths) - patch_count - 1)
    indices = np.linspace(0, max_index - 1, num=windows, dtype=np.int64)
    nll_total = 0.0
    bytes_total = 0
    started = time.perf_counter()
    for start in range(0, len(indices), batch_size):
        batch_indices = indices[start:start + batch_size]
        inputs, input_lengths, targets, target_lengths = stream.batch_from_indices(
            batch_indices, patch_count, device
        )
        _, nll_sum, byte_count = model.nll(inputs, input_lengths, targets, target_lengths)
        nll_total += float(nll_sum)
        bytes_total += int(byte_count)
    elapsed = time.perf_counter() - started
    return nll_total / max(1, bytes_total), bytes_total, elapsed


def train_variant(
    variant: str,
    streams: dict[str, PatchStream],
    cfg: Config,
    seed: int,
    device: torch.device,
    max_global_positions: int,
) -> tuple[VariantResult, dict[str, Any]]:
    set_seed(seed)
    patch_count = patch_count_for_variant(variant, cfg, streams["train"])
    model = CausalPatchByteLM(cfg, max_global_positions=max_global_positions).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )
    offsets = shared_offsets(
        streams["train"].byte_count,
        cfg.batch_size,
        cfg.steps,
        seed + 77,
        cfg.context_bytes * 3,
    )
    best_validation = math.inf
    best_state: dict[str, torch.Tensor] | None = None
    history: list[dict[str, float]] = []
    train_bytes_seen = 0
    started = time.perf_counter()

    for step in range(1, cfg.steps + 1):
        model.train()
        indices = streams["train"].patch_index_for_offsets(offsets[step - 1], patch_count)
        inputs, input_lengths, targets, target_lengths = streams["train"].batch_from_indices(
            indices, patch_count, device
        )
        loss, _, byte_count = model.nll(inputs, input_lengths, targets, target_lengths)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()
        train_bytes_seen += int(byte_count)

        if step == 1 or step % cfg.eval_interval == 0 or step == cfg.steps:
            val_nats, val_bytes, _ = evaluate_windows(
                model,
                streams["validation"],
                patch_count,
                cfg.validation_windows,
                cfg.batch_size,
                device,
            )
            val_bpb = val_nats / math.log(2.0)
            history.append(
                {
                    "step": float(step),
                    "train_nats_per_byte": float(loss.detach()),
                    "validation_bits_per_byte": val_bpb,
                    "validation_bytes": float(val_bytes),
                }
            )
            print(
                f"{variant} step={step}/{cfg.steps} train={float(loss.detach()):.4f} "
                f"val-bpb={val_bpb:.4f} bytes-seen={train_bytes_seen:,}",
                flush=True,
            )
            if val_bpb < best_validation:
                best_validation = val_bpb
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in model.state_dict().items()
                }

    train_seconds = time.perf_counter() - started
    if best_state is None:
        raise RuntimeError("No checkpoint was selected")
    model.load_state_dict(best_state)
    test_nats, test_bytes, test_seconds = evaluate_windows(
        model,
        streams["test"],
        patch_count,
        cfg.test_windows,
        cfg.batch_size,
        device,
    )
    train_stream = streams["train"]
    test_stream = streams["test"]
    attention_pairs = cfg.steps * cfg.batch_size * cfg.n_heads * patch_count * patch_count * cfg.n_layers
    byte_patch_count = cfg.context_bytes
    byte_attention_pairs = cfg.steps * cfg.batch_size * cfg.n_heads * byte_patch_count * byte_patch_count * cfg.n_layers
    result = VariantResult(
        variant=variant,
        mean_patch_length_train=train_stream.mean_patch_length,
        mean_patch_length_test=test_stream.mean_patch_length,
        patches_per_kib=1024.0 / test_stream.mean_patch_length,
        global_positions=patch_count,
        context_bytes_mean=patch_count * train_stream.mean_patch_length,
        train_bytes_seen=train_bytes_seen,
        global_positions_seen=cfg.steps * cfg.batch_size * patch_count,
        model_parameters=sum(parameter.numel() for parameter in model.parameters()),
        best_validation_bits_per_byte=best_validation,
        test_bits_per_byte=test_nats / math.log(2.0),
        test_nats_per_byte=test_nats,
        test_bytes=test_bytes,
        train_seconds=train_seconds,
        test_seconds=test_seconds,
        train_bytes_per_second=train_bytes_seen / max(train_seconds, 1e-9),
        test_bytes_per_second=test_bytes / max(test_seconds, 1e-9),
        estimated_attention_pairs_seen=attention_pairs,
        attention_pairs_ratio_to_byte=attention_pairs / byte_attention_pairs,
        boundary_entropy_lift_bits=test_stream.boundary_entropy_lift_bits,
    )
    return result, {"history": history}


def make_decision(results: Sequence[dict[str, Any]]) -> dict[str, Any]:
    indexed = {row["variant"]: row for row in results}
    adaptive = indexed["entropy-adaptive"]
    fixed = indexed["fixed4"]
    random_control = indexed["random-matched"]
    byte = indexed["byte-context"]
    quality_ratio_fixed = adaptive["test_bits_per_byte"] / fixed["test_bits_per_byte"]
    quality_ratio_random = adaptive["test_bits_per_byte"] / random_control["test_bits_per_byte"]
    speedup_vs_byte = adaptive["test_bytes_per_second"] / byte["test_bytes_per_second"]
    position_ratio_vs_byte = adaptive["global_positions"] / byte["global_positions"]
    if (
        quality_ratio_fixed <= 1.02
        and quality_ratio_random <= 0.995
        and speedup_vs_byte >= 1.5
        and position_ratio_vs_byte <= 0.35
    ):
        verdict = "ADAPTIVE_BYTE_PATCH_PASS"
    elif quality_ratio_fixed <= 1.02 and speedup_vs_byte >= 1.25:
        verdict = "ADAPTIVE_BYTE_PATCH_BORDERLINE"
    else:
        verdict = "ADAPTIVE_BYTE_PATCH_FAIL"
    return {
        "verdict": verdict,
        "adaptive_to_fixed_bpb_ratio": quality_ratio_fixed,
        "adaptive_to_random_bpb_ratio": quality_ratio_random,
        "adaptive_speedup_vs_context_matched_byte": speedup_vs_byte,
        "adaptive_global_position_ratio_vs_byte": position_ratio_vs_byte,
        "adaptive_boundary_entropy_lift_bits": adaptive["boundary_entropy_lift_bits"],
        "rule": "Strong PASS requires adaptive B/B <=1.02x fixed4, <=0.995x the length-matched random control, >=1.5x test byte throughput versus context-matched bytes, and <=35% as many global positions.",
    }


def write_outputs(output_dir: Path, payload: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "adaptive_byte_patches.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    rows = payload["results"]
    with (output_dir / "results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    d = payload["decision"]
    lines = [
        "# Test 5.0 — causal adaptive byte patches",
        "",
        f"**Decision:** **{d['verdict']}**",
        "",
        "| Variant | Mean bytes/patch | Global positions | Test bits/byte | Patches/KiB | Test bytes/s | Attention-pair ratio | Boundary entropy lift |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['variant']} | {row['mean_patch_length_test']:.3f} | "
            f"{row['global_positions']} | {row['test_bits_per_byte']:.4f} | "
            f"{row['patches_per_kib']:.1f} | {row['test_bytes_per_second']:.0f} | "
            f"{row['attention_pairs_ratio_to_byte']:.3%} | "
            f"{row['boundary_entropy_lift_bits']:+.3f} bits |"
        )
    lines += [
        "",
        "## Comparisons",
        f"- adaptive/fixed4 bits-per-byte ratio: `{d['adaptive_to_fixed_bpb_ratio']:.4f}×`.",
        f"- adaptive/random-matched bits-per-byte ratio: `{d['adaptive_to_random_bpb_ratio']:.4f}×`.",
        f"- adaptive throughput versus context-matched byte sequence: `{d['adaptive_speedup_vs_context_matched_byte']:.2f}×`.",
        f"- adaptive global-position ratio versus bytes: `{d['adaptive_global_position_ratio_vs_byte']:.3%}`.",
        "",
        "All variants share model parameters, optimizer budget, raw UTF-8 corpus, and approximately 192 bytes of context. The byte baseline therefore uses 192 global positions; patched variants use roughly 48. The local decoder contributes exact autoregressive likelihood for every byte.",
        "",
        d["rule"],
    ]
    (output_dir / "VERDICT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def self_test() -> None:
    set_seed(5)
    synthetic = np.asarray((b"abcd " * 400) + (b"xyz\n" * 200), dtype=np.uint8)
    entropy, _ = bigram_entropy_bits(synthetic)
    threshold, calibration = calibrate_adaptive_threshold(synthetic, entropy, 4.0, 1, 8, len(synthetic))
    streams = {
        "fixed": segment_fixed(synthetic, 4, 8, entropy),
        "adaptive": segment_adaptive(synthetic, entropy, threshold, 1, 8),
        "random": segment_random_matched(synthetic, calibration, 8, entropy, 9),
    }
    for stream in streams.values():
        if int(np.sum(stream.lengths)) != len(synthetic):
            raise AssertionError("stream does not cover bytes")
        if stream.patches.shape[1] != 8 or np.any(stream.lengths < 1) or np.any(stream.lengths > 8):
            raise AssertionError("invalid patch lengths")
    cfg = Config(d_model=24, n_heads=4, n_layers=1, global_ff=48, batch_size=2, steps=1)
    model = CausalPatchByteLM(cfg, max_global_positions=16)
    indices = np.asarray([0, 2], dtype=np.int64)
    batch = streams["fixed"].batch_from_indices(indices, 8, torch.device("cpu"))
    loss, _, count = model.nll(*batch)
    if not torch.isfinite(loss) or int(count) <= 0:
        raise AssertionError((loss, count))
    loss.backward()
    if any(parameter.grad is None for parameter in model.parameters()):
        raise AssertionError("missing gradient")
    print(
        f"self-test passed: threshold={threshold:.4f}, adaptive mean={streams['adaptive'].mean_patch_length:.3f}, loss={float(loss):.4f}"
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("results/test-5-0/latest"))
    parser.add_argument("--seed", type=int, default=50500)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--train-bytes", type=int, default=1_200_000)
    parser.add_argument("--validation-bytes", type=int, default=180_000)
    parser.add_argument("--test-bytes", type=int, default=180_000)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args(argv)

    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    if args.self_test:
        self_test()
        return 0
    if args.manifest is None:
        parser.error("--manifest is required unless --self-test is used")

    cfg = Config(steps=args.steps)
    device = torch.device("cpu")
    data = load_byte_manifest(
        args.manifest,
        args.train_bytes,
        args.validation_bytes,
        args.test_bytes,
    )
    entropy, _ = bigram_entropy_bits(data["train"])
    threshold, calibration_lengths = calibrate_adaptive_threshold(
        data["train"], entropy, 4.0, 1, cfg.max_patch_len
    )
    full_adaptive_train_lengths = adaptive_lengths(
        data["train"], entropy, threshold, 1, cfg.max_patch_len
    )
    print(
        f"adaptive threshold={threshold:.6f}; calibration mean={float(np.mean(calibration_lengths)):.4f}; "
        f"full train mean={float(np.mean(full_adaptive_train_lengths)):.4f}",
        flush=True,
    )

    variants = ("byte-context", "fixed4", "random-matched", "entropy-adaptive")
    results: list[dict[str, Any]] = []
    diagnostics: dict[str, Any] = {}
    for variant_index, variant in enumerate(variants):
        streams = build_streams(
            data,
            cfg,
            entropy,
            threshold,
            full_adaptive_train_lengths,
            variant,
            args.seed + 10_000 * variant_index,
        )
        result, detail = train_variant(
            variant,
            streams,
            cfg,
            args.seed,
            device,
            max_global_positions=cfg.context_bytes,
        )
        results.append(asdict(result))
        diagnostics[variant] = {
            **detail,
            "split_patch_statistics": {
                split: {
                    "patches": int(len(stream.lengths)),
                    "mean_length": stream.mean_patch_length,
                    "boundary_entropy_lift_bits": stream.boundary_entropy_lift_bits,
                }
                for split, stream in streams.items()
            },
        }
        print(
            f"completed {variant}: test={result.test_bits_per_byte:.4f} bits/byte, "
            f"positions={result.global_positions}, test-throughput={result.test_bytes_per_second:.0f} bytes/s",
            flush=True,
        )

    payload: dict[str, Any] = {
        "metadata": {
            "task": "WikiText-2 raw UTF-8 byte language modeling",
            "seed": args.seed,
            "steps_per_variant": cfg.steps,
            "train_bytes_available": int(len(data["train"])),
            "validation_bytes_available": int(len(data["validation"])),
            "test_bytes_available": int(len(data["test"])),
            "adaptive_target_mean_length": 4.0,
            "adaptive_threshold_entropy_bits": threshold,
            "segmentation_note": "A patch ends after byte b when train-only H(next_byte|b) exceeds the calibrated threshold, or at eight bytes. The rule uses no future byte.",
            "likelihood_note": "The global model is causal over preceding patches and a local GRU is causal within the target patch; reported NLL sums every target byte.",
        },
        "results": results,
        "diagnostics": diagnostics,
    }
    payload["decision"] = make_decision(results)
    write_outputs(args.output_dir, payload)
    print((args.output_dir / "VERDICT.md").read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
