#!/usr/bin/env python3
"""Test 5.0: compact language units built directly from bytes.

This experiment isolates the representation of the *context* while keeping the
predictor architecture, initialization, examples, parameter count, optimizer,
and output target identical. Every model predicts the next raw byte from the
same fixed-size byte context.

Variants
--------
* byte: one unit per byte;
* bpe512: deterministic byte-pair units learned only from the train split;
* fixed: fixed-size raw-byte patches;
* adaptive: causal patches closed by accumulated byte surprisal;
* random-matched: random boundaries drawn from the adaptive length histogram.

The BPE is trained over a reversible private-use Unicode encoding of bytes. It
therefore merges arbitrary byte strings without relying on natural-language
Unicode boundaries. Adaptive boundaries use a smoothed train-only byte bigram
model. No validation byte is used to train the tokenizer, boundary model, or to
select the adaptive information threshold.

The primary metric is bits per byte. Sequence compactness is measured as units
per KiB and estimated quadratic attention work. Model throughput excludes
one-time representation preprocessing and is reported separately from it.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import resource
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from tokenizers import Tokenizer, models, trainers


PRIVATE_BASE = 0xE000
EPS = 1e-12


@dataclass
class Config:
    context_bytes: int = 128
    batch_size: int = 16
    steps: int = 300
    eval_batches: int = 30
    d_model: int = 64
    n_heads: int = 4
    n_layers: int = 2
    d_ff: int = 192
    dropout: float = 0.0
    lr: float = 5e-4
    weight_decay: float = 0.03
    grad_clip: float = 1.0
    vocab_capacity: int = 512
    bpe_vocab_size: int = 512
    max_unit_bytes: int = 16
    target_patch_bytes: float = 4.0


@dataclass
class EncodedExample:
    unit_symbols: list[list[int]]
    unit_byte_lengths: list[int]
    target: int


@dataclass
class VariantResult:
    variant: str
    seed: int
    validation_loss_nats: float
    validation_bits_per_byte: float
    validation_perplexity_per_byte: float
    mean_units_per_context: float
    median_units_per_context: float
    p95_units_per_context: float
    units_per_kib: float
    mean_bytes_per_unit: float
    estimated_attention_work_ratio_to_byte: float
    preprocessing_seconds: float
    training_seconds: float
    model_bytes_per_second: float
    trainable_parameters: int
    final_train_loss_nats: float
    best_validation_loss_nats: float
    fixed_patch_size: int | None
    adaptive_information_threshold_bits: float | None
    peak_rss_mib: float


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def bytes_to_private(raw: bytes | np.ndarray) -> str:
    if isinstance(raw, np.ndarray):
        values = raw.tolist()
    else:
        values = raw
    return "".join(chr(PRIVATE_BASE + int(value)) for value in values)


def private_to_bytes(text: str) -> bytes:
    return bytes(ord(char) - PRIVATE_BASE for char in text)


def chunks(values: np.ndarray, size: int = 16384) -> Iterable[str]:
    for start in range(0, len(values), size):
        yield bytes_to_private(values[start:start + size])


def train_byte_bpe(train_bytes: np.ndarray, vocab_size: int, output_path: Path) -> Tokenizer:
    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    initial_alphabet = [chr(PRIVATE_BASE + value) for value in range(256)]
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=2,
        special_tokens=["<unk>"],
        initial_alphabet=initial_alphabet,
        show_progress=False,
    )
    tokenizer.train_from_iterator(chunks(train_bytes), trainer=trainer)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(output_path))

    probe = bytes(range(256)) + b"abracadabra\x00\xff"
    encoded = tokenizer.encode(bytes_to_private(probe)).ids
    recovered = "".join(tokenizer.id_to_token(index) or "" for index in encoded)
    if private_to_bytes(recovered) != probe:
        raise AssertionError("custom byte BPE is not exactly reversible")
    if tokenizer.get_vocab_size() > vocab_size:
        raise AssertionError(tokenizer.get_vocab_size())
    return tokenizer


def train_bigram_surprisal(train_bytes: np.ndarray, alpha: float = 0.25) -> np.ndarray:
    counts = np.full((257, 256), alpha, dtype=np.float64)
    if len(train_bytes) < 2:
        raise ValueError("training corpus is too small")
    previous = train_bytes[:-1].astype(np.int64, copy=False)
    current = train_bytes[1:].astype(np.int64, copy=False)
    np.add.at(counts, (previous, current), 1.0)
    global_counts = np.bincount(train_bytes.astype(np.int64), minlength=256).astype(np.float64)
    counts[256] += global_counts
    probabilities = counts / counts.sum(axis=1, keepdims=True)
    return -np.log2(np.maximum(probabilities, EPS)).astype(np.float32)


def corpus_surprisal(data: np.ndarray, table: np.ndarray) -> np.ndarray:
    previous = np.empty(len(data), dtype=np.int64)
    previous[0] = 256
    if len(data) > 1:
        previous[1:] = data[:-1]
    return table[previous, data.astype(np.int64, copy=False)]


def segment_fixed(raw: np.ndarray, size: int) -> list[list[int]]:
    return [raw[start:start + size].astype(np.int64).tolist() for start in range(0, len(raw), size)]


def segment_adaptive(
    raw: np.ndarray,
    surprisal: np.ndarray,
    threshold_bits: float,
    max_unit_bytes: int,
) -> list[list[int]]:
    units: list[list[int]] = []
    start = 0
    information = 0.0
    for index, value in enumerate(surprisal):
        information += float(value)
        length = index - start + 1
        if information >= threshold_bits or length >= max_unit_bytes:
            units.append(raw[start:index + 1].astype(np.int64).tolist())
            start = index + 1
            information = 0.0
    if start < len(raw):
        units.append(raw[start:].astype(np.int64).tolist())
    return units


def adaptive_mean_length(
    data: np.ndarray,
    surprisals: np.ndarray,
    starts: np.ndarray,
    context_bytes: int,
    threshold_bits: float,
    max_unit_bytes: int,
) -> float:
    lengths: list[int] = []
    for start in starts:
        raw = data[int(start):int(start) + context_bytes]
        score = surprisals[int(start):int(start) + context_bytes]
        lengths.extend(len(unit) for unit in segment_adaptive(raw, score, threshold_bits, max_unit_bytes))
    return float(np.mean(lengths))


def calibrate_adaptive_threshold(
    data: np.ndarray,
    surprisals: np.ndarray,
    starts: np.ndarray,
    context_bytes: int,
    target_length: float,
    max_unit_bytes: int,
) -> tuple[float, list[int]]:
    low, high = 0.1, 128.0
    for _ in range(24):
        midpoint = (low + high) * 0.5
        observed = adaptive_mean_length(
            data,
            surprisals,
            starts,
            context_bytes,
            midpoint,
            max_unit_bytes,
        )
        if observed < target_length:
            low = midpoint
        else:
            high = midpoint
    threshold = (low + high) * 0.5
    histogram: list[int] = []
    for start in starts:
        raw = data[int(start):int(start) + context_bytes]
        score = surprisals[int(start):int(start) + context_bytes]
        histogram.extend(
            len(unit) for unit in segment_adaptive(raw, score, threshold, max_unit_bytes)
        )
    return threshold, histogram


def segment_random_matched(
    raw: np.ndarray,
    empirical_lengths: Sequence[int],
    seed: int,
) -> list[list[int]]:
    rng = np.random.default_rng(seed)
    units: list[list[int]] = []
    position = 0
    while position < len(raw):
        sampled = int(empirical_lengths[int(rng.integers(0, len(empirical_lengths)))])
        length = max(1, min(sampled, len(raw) - position))
        units.append(raw[position:position + length].astype(np.int64).tolist())
        position += length
    return units


def bpe_token_lengths(tokenizer: Tokenizer) -> dict[int, int]:
    result: dict[int, int] = {}
    for index in range(tokenizer.get_vocab_size()):
        token = tokenizer.id_to_token(index) or ""
        if token == "<unk>":
            result[index] = 1
        else:
            result[index] = max(1, len(token))
    return result


def encode_context(
    variant: str,
    data: np.ndarray,
    surprisals: np.ndarray,
    start: int,
    cfg: Config,
    *,
    tokenizer: Tokenizer,
    token_lengths: dict[int, int],
    fixed_size: int,
    adaptive_threshold: float,
    empirical_lengths: Sequence[int],
    random_seed: int,
) -> EncodedExample:
    raw = data[start:start + cfg.context_bytes]
    target = int(data[start + cfg.context_bytes])
    if variant == "byte":
        units = [[int(value)] for value in raw]
        byte_lengths = [1] * len(units)
    elif variant == "bpe512":
        ids = tokenizer.encode(bytes_to_private(raw)).ids
        reconstructed = "".join(tokenizer.id_to_token(index) or "" for index in ids)
        if private_to_bytes(reconstructed) != bytes(raw):
            raise AssertionError("BPE context reconstruction mismatch")
        units = [[int(index)] for index in ids]
        byte_lengths = [token_lengths[int(index)] for index in ids]
    elif variant == "fixed":
        units = segment_fixed(raw, fixed_size)
        byte_lengths = [len(unit) for unit in units]
    elif variant == "adaptive":
        score = surprisals[start:start + cfg.context_bytes]
        units = segment_adaptive(raw, score, adaptive_threshold, cfg.max_unit_bytes)
        byte_lengths = [len(unit) for unit in units]
    elif variant == "random-matched":
        units = segment_random_matched(raw, empirical_lengths, random_seed ^ start)
        byte_lengths = [len(unit) for unit in units]
    else:
        raise ValueError(variant)
    if sum(byte_lengths) != cfg.context_bytes:
        raise AssertionError((variant, sum(byte_lengths), cfg.context_bytes))
    return EncodedExample(units, byte_lengths, target)


def precompute_examples(
    variant: str,
    data: np.ndarray,
    surprisals: np.ndarray,
    starts: np.ndarray,
    cfg: Config,
    **kwargs: Any,
) -> tuple[list[EncodedExample], float]:
    started = time.perf_counter()
    examples = [
        encode_context(
            variant,
            data,
            surprisals,
            int(start),
            cfg,
            random_seed=int(kwargs["random_seed"]),
            tokenizer=kwargs["tokenizer"],
            token_lengths=kwargs["token_lengths"],
            fixed_size=int(kwargs["fixed_size"]),
            adaptive_threshold=float(kwargs["adaptive_threshold"]),
            empirical_lengths=kwargs["empirical_lengths"],
        )
        for start in starts
    ]
    return examples, time.perf_counter() - started


def collate(examples: Sequence[EncodedExample], cfg: Config) -> tuple[torch.Tensor, ...]:
    batch = len(examples)
    max_units = max(len(example.unit_symbols) for example in examples)
    max_symbols = max(len(unit) for example in examples for unit in example.unit_symbols)
    if max_symbols > cfg.max_unit_bytes:
        # BPE units are represented by one symbol, so this applies to raw patches only.
        raise AssertionError(max_symbols)
    ids = torch.zeros(batch, max_units, max_symbols, dtype=torch.long)
    symbol_lengths = torch.ones(batch, max_units, dtype=torch.long)
    byte_lengths = torch.ones(batch, max_units, dtype=torch.long)
    padding_mask = torch.ones(batch, max_units, dtype=torch.bool)
    targets = torch.empty(batch, dtype=torch.long)
    for row, example in enumerate(examples):
        targets[row] = example.target
        for column, (unit, byte_length) in enumerate(
            zip(example.unit_symbols, example.unit_byte_lengths, strict=True)
        ):
            ids[row, column, :len(unit)] = torch.tensor(unit, dtype=torch.long)
            symbol_lengths[row, column] = len(unit)
            byte_lengths[row, column] = byte_length
            padding_mask[row, column] = False
    return ids, symbol_lengths, byte_lengths, padding_mask, targets


class UnitContextModel(nn.Module):
    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.cfg = cfg
        self.symbol_embedding = nn.Embedding(cfg.vocab_capacity, cfg.d_model)
        self.unit_encoder = nn.GRU(cfg.d_model, cfg.d_model, batch_first=True)
        self.length_embedding = nn.Embedding(cfg.max_unit_bytes + 1, cfg.d_model)
        self.position_embedding = nn.Embedding(cfg.context_bytes + 2, cfg.d_model)
        self.query = nn.Parameter(torch.zeros(1, 1, cfg.d_model))
        layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.n_heads,
            dim_feedforward=cfg.d_ff,
            dropout=cfg.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=cfg.n_layers)
        self.final_norm = nn.LayerNorm(cfg.d_model)
        self.output = nn.Linear(cfg.d_model, 256)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.symbol_embedding.weight, std=0.02)
        nn.init.normal_(self.position_embedding.weight, std=0.02)
        nn.init.normal_(self.length_embedding.weight, std=0.02)
        nn.init.normal_(self.query, std=0.02)
        nn.init.normal_(self.output.weight, std=0.02)
        nn.init.zeros_(self.output.bias)

    def forward(
        self,
        ids: torch.Tensor,
        symbol_lengths: torch.Tensor,
        byte_lengths: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch, units, symbols = ids.shape
        embedded = self.symbol_embedding(ids).reshape(batch * units, symbols, self.cfg.d_model)
        encoded, _ = self.unit_encoder(embedded)
        flat_lengths = symbol_lengths.reshape(-1).clamp_min(1)
        gather = (flat_lengths - 1).view(-1, 1, 1).expand(-1, 1, self.cfg.d_model)
        unit_states = torch.gather(encoded, 1, gather).squeeze(1).reshape(batch, units, self.cfg.d_model)
        unit_states = unit_states + self.length_embedding(
            byte_lengths.clamp(1, self.cfg.max_unit_bytes)
        )
        positions = torch.arange(units + 1, device=ids.device)
        query = self.query.expand(batch, -1, -1)
        sequence = torch.cat([unit_states, query], dim=1)
        sequence = sequence + self.position_embedding(positions)[None, :, :]
        full_mask = torch.cat(
            [padding_mask, torch.zeros(batch, 1, dtype=torch.bool, device=ids.device)],
            dim=1,
        )
        transformed = self.transformer(sequence, src_key_padding_mask=full_mask)
        return self.output(self.final_norm(transformed[:, -1]))


def evaluate(
    model: UnitContextModel,
    examples: Sequence[EncodedExample],
    cfg: Config,
) -> float:
    model.eval()
    losses: list[float] = []
    with torch.no_grad():
        for start in range(0, len(examples), cfg.batch_size):
            batch = examples[start:start + cfg.batch_size]
            ids, symbol_lengths, byte_lengths, padding_mask, targets = collate(batch, cfg)
            logits = model(ids, symbol_lengths, byte_lengths, padding_mask)
            losses.append(float(F.cross_entropy(logits, targets, reduction="sum")))
    return float(sum(losses) / len(examples))


def representation_statistics(examples: Sequence[EncodedExample], context_bytes: int) -> dict[str, float]:
    counts = np.asarray([len(example.unit_symbols) for example in examples], dtype=np.float64)
    return {
        "mean": float(np.mean(counts)),
        "median": float(np.median(counts)),
        "p95": float(np.percentile(counts, 95)),
        "units_per_kib": float(np.mean(counts) / context_bytes * 1024.0),
        "mean_bytes_per_unit": float(context_bytes / np.mean(counts)),
        "attention_work_ratio": float(np.mean(counts * counts) / (context_bytes * context_bytes)),
    }


def train_variant(
    variant: str,
    train_examples: Sequence[EncodedExample],
    validation_examples: Sequence[EncodedExample],
    cfg: Config,
    seed: int,
    preprocessing_seconds: float,
    *,
    fixed_size: int,
    adaptive_threshold: float,
) -> VariantResult:
    set_seed(seed)
    model = UnitContextModel(cfg)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    best_validation = math.inf
    final_train = math.nan
    started = time.perf_counter()
    for step in range(cfg.steps):
        offset = step * cfg.batch_size
        batch = train_examples[offset:offset + cfg.batch_size]
        ids, symbol_lengths, byte_lengths, padding_mask, targets = collate(batch, cfg)
        model.train()
        logits = model(ids, symbol_lengths, byte_lengths, padding_mask)
        loss = F.cross_entropy(logits, targets)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()
        final_train = float(loss.detach())
        if step == 0 or (step + 1) % 100 == 0 or step + 1 == cfg.steps:
            validation_loss = evaluate(model, validation_examples, cfg)
            best_validation = min(best_validation, validation_loss)
            print(
                f"{variant} step={step + 1}/{cfg.steps} "
                f"train={final_train:.4f} val={validation_loss:.4f} "
                f"bpb={validation_loss / math.log(2.0):.4f}",
                flush=True,
            )
    training_seconds = time.perf_counter() - started
    validation_loss = evaluate(model, validation_examples, cfg)
    best_validation = min(best_validation, validation_loss)
    stats = representation_statistics(validation_examples, cfg.context_bytes)
    bytes_seen = cfg.steps * cfg.batch_size * (cfg.context_bytes + 1)
    peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    return VariantResult(
        variant=variant,
        seed=seed,
        validation_loss_nats=validation_loss,
        validation_bits_per_byte=validation_loss / math.log(2.0),
        validation_perplexity_per_byte=float(math.exp(min(validation_loss, 20.0))),
        mean_units_per_context=stats["mean"],
        median_units_per_context=stats["median"],
        p95_units_per_context=stats["p95"],
        units_per_kib=stats["units_per_kib"],
        mean_bytes_per_unit=stats["mean_bytes_per_unit"],
        estimated_attention_work_ratio_to_byte=stats["attention_work_ratio"],
        preprocessing_seconds=preprocessing_seconds,
        training_seconds=training_seconds,
        model_bytes_per_second=bytes_seen / max(training_seconds, EPS),
        trainable_parameters=sum(parameter.numel() for parameter in model.parameters()),
        final_train_loss_nats=final_train,
        best_validation_loss_nats=best_validation,
        fixed_patch_size=fixed_size if variant == "fixed" else None,
        adaptive_information_threshold_bits=(
            adaptive_threshold if variant == "adaptive" else None
        ),
        peak_rss_mib=peak_rss,
    )


def make_decision(results: Sequence[dict[str, Any]]) -> dict[str, Any]:
    indexed = {row["variant"]: row for row in results}
    adaptive = indexed["adaptive"]
    fixed = indexed["fixed"]
    random_control = indexed["random-matched"]
    bpe = indexed["bpe512"]
    adaptive_vs_fixed = fixed["validation_bits_per_byte"] - adaptive["validation_bits_per_byte"]
    adaptive_vs_random = random_control["validation_bits_per_byte"] - adaptive["validation_bits_per_byte"]
    unit_reduction_vs_bpe = 1.0 - adaptive["mean_units_per_context"] / bpe["mean_units_per_context"]
    bpb_ratio_to_bpe = adaptive["validation_bits_per_byte"] / bpe["validation_bits_per_byte"]
    if (
        bpb_ratio_to_bpe <= 1.02
        and unit_reduction_vs_bpe >= 0.20
        and adaptive_vs_fixed >= 0.0
        and adaptive_vs_random >= 0.005
    ):
        verdict = "ADAPTIVE_REPRESENTATION_SIGNAL"
    elif bpb_ratio_to_bpe <= 1.03 and unit_reduction_vs_bpe > 0.0:
        verdict = "COMPRESSION_SIGNAL"
    else:
        verdict = "FAIL"
    return {
        "verdict": verdict,
        "adaptive_bpb_ratio_to_bpe": bpb_ratio_to_bpe,
        "adaptive_unit_reduction_vs_bpe": unit_reduction_vs_bpe,
        "adaptive_bpb_advantage_over_fixed": adaptive_vs_fixed,
        "adaptive_bpb_advantage_over_random_matched": adaptive_vs_random,
        "rule": (
            "Strong signal requires adaptive BpB within 2% of BPE, at least 20% "
            "fewer units than BPE, no regression versus fixed patches, and at "
            "least 0.005 BpB advantage over random matched boundaries."
        ),
    }


def write_outputs(output_dir: Path, payload: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "adaptive_byte_units.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with (output_dir / "results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(payload["results"][0].keys()))
        writer.writeheader()
        writer.writerows(payload["results"])
    decision = payload["decision"]
    lines = [
        "# Test 5.0 — adaptive byte-unit representation screen",
        "",
        f"**Decision:** **{decision['verdict']}**",
        "",
        "| Representation | Bits/byte | Units/context | Bytes/unit | Units/KiB | Attention work/byte | Model bytes/s |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["results"]:
        lines.append(
            f"| {row['variant']} | {row['validation_bits_per_byte']:.4f} | "
            f"{row['mean_units_per_context']:.2f} | {row['mean_bytes_per_unit']:.2f} | "
            f"{row['units_per_kib']:.1f} | {row['estimated_attention_work_ratio_to_byte']:.3%} | "
            f"{row['model_bytes_per_second']:.0f} |"
        )
    lines += [
        "",
        f"- Adaptive/BPE BpB ratio: `{decision['adaptive_bpb_ratio_to_bpe']:.4f}`.",
        f"- Adaptive unit reduction versus BPE: `{decision['adaptive_unit_reduction_vs_bpe']:+.2%}`.",
        f"- Adaptive BpB advantage versus fixed: `{decision['adaptive_bpb_advantage_over_fixed']:+.4f}`.",
        f"- Adaptive BpB advantage versus random matched boundaries: `{decision['adaptive_bpb_advantage_over_random_matched']:+.4f}`.",
        f"- Calibrated adaptive information budget: `{payload['metadata']['adaptive_threshold_bits']:.3f}` bits; fixed patch size `{payload['metadata']['fixed_patch_size']}`.",
        "",
        "All variants predict the same next raw byte from the same byte windows with identical model parameter count and initialization. This is a representation screen, not yet a full autoregressive patch decoder.",
    ]
    (output_dir / "VERDICT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def self_test() -> None:
    raw = np.frombuffer(("abracadabra π código " * 20).encode("utf-8"), dtype=np.uint8)
    with torch.no_grad():
        cfg = Config(context_bytes=32, batch_size=2, steps=1, eval_batches=1, d_model=16, n_heads=4, n_layers=1, d_ff=32, max_unit_bytes=8)
        temp = Path("/tmp/test-5-0-byte-bpe.json")
        tokenizer = train_byte_bpe(raw, 320, temp)
        table = train_bigram_surprisal(raw)
        surprise = corpus_surprisal(raw, table)
        starts = np.asarray([0, 5, 10], dtype=np.int64)
        threshold, histogram = calibrate_adaptive_threshold(raw, surprise, starts, cfg.context_bytes, 4.0, cfg.max_unit_bytes)
        lengths = bpe_token_lengths(tokenizer)
        example = encode_context(
            "adaptive", raw, surprise, 0, cfg,
            tokenizer=tokenizer, token_lengths=lengths, fixed_size=4,
            adaptive_threshold=threshold, empirical_lengths=histogram, random_seed=7,
        )
        batch = collate([example, example], cfg)
        model = UnitContextModel(cfg)
        logits = model(*batch[:-1])
        if logits.shape != (2, 256) or not torch.isfinite(logits).all():
            raise AssertionError(logits.shape)
    temp.unlink(missing_ok=True)
    print("self-test passed")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-bytes", type=Path, required=False)
    parser.add_argument("--validation-bytes", type=Path, required=False)
    parser.add_argument("--output-dir", type=Path, default=Path("results/test-5-0/latest"))
    parser.add_argument("--seed", type=int, default=50500)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--context-bytes", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--eval-batches", type=int, default=30)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args(argv)

    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    if args.self_test:
        self_test()
        return 0
    if args.train_bytes is None or args.validation_bytes is None:
        parser.error("--train-bytes and --validation-bytes are required")

    cfg = Config(
        context_bytes=args.context_bytes,
        batch_size=args.batch_size,
        steps=args.steps,
        eval_batches=args.eval_batches,
    )
    train = np.frombuffer(args.train_bytes.read_bytes(), dtype=np.uint8)
    validation = np.frombuffer(args.validation_bytes.read_bytes(), dtype=np.uint8)
    if len(train) <= cfg.context_bytes + 1 or len(validation) <= cfg.context_bytes + 1:
        raise ValueError((len(train), len(validation)))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = train_byte_bpe(train, cfg.bpe_vocab_size, args.output_dir / "byte_bpe_tokenizer.json")
    token_lengths = bpe_token_lengths(tokenizer)
    bigram_table = train_bigram_surprisal(train)
    train_surprise = corpus_surprisal(train, bigram_table)
    validation_surprise = corpus_surprisal(validation, bigram_table)

    rng_train = np.random.default_rng(args.seed + 1)
    rng_validation = np.random.default_rng(args.seed + 2)
    train_starts = rng_train.integers(
        0, len(train) - cfg.context_bytes - 1,
        size=cfg.steps * cfg.batch_size,
        endpoint=False,
        dtype=np.int64,
    )
    validation_starts = rng_validation.integers(
        0, len(validation) - cfg.context_bytes - 1,
        size=cfg.eval_batches * cfg.batch_size,
        endpoint=False,
        dtype=np.int64,
    )
    calibration_starts = rng_train.integers(
        0, len(train) - cfg.context_bytes - 1,
        size=384,
        endpoint=False,
        dtype=np.int64,
    )
    adaptive_threshold, empirical_lengths = calibrate_adaptive_threshold(
        train,
        train_surprise,
        calibration_starts,
        cfg.context_bytes,
        cfg.target_patch_bytes,
        cfg.max_unit_bytes,
    )
    fixed_size = max(1, int(round(statistics.mean(empirical_lengths))))
    print(
        f"adaptive threshold={adaptive_threshold:.4f} bits; "
        f"adaptive mean length={statistics.mean(empirical_lengths):.3f}; fixed={fixed_size}",
        flush=True,
    )

    variants = ["byte", "bpe512", "fixed", "adaptive", "random-matched"]
    results: list[dict[str, Any]] = []
    for variant in variants:
        kwargs = {
            "tokenizer": tokenizer,
            "token_lengths": token_lengths,
            "fixed_size": fixed_size,
            "adaptive_threshold": adaptive_threshold,
            "empirical_lengths": empirical_lengths,
            "random_seed": args.seed + 77,
        }
        train_examples, prep_train = precompute_examples(
            variant, train, train_surprise, train_starts, cfg, **kwargs
        )
        validation_examples, prep_validation = precompute_examples(
            variant, validation, validation_surprise, validation_starts, cfg, **kwargs
        )
        result = train_variant(
            variant,
            train_examples,
            validation_examples,
            cfg,
            args.seed,
            prep_train + prep_validation,
            fixed_size=fixed_size,
            adaptive_threshold=adaptive_threshold,
        )
        results.append(asdict(result))
        del train_examples, validation_examples

    payload = {
        "metadata": {
            "task": "next raw byte from a fixed raw-byte context",
            "dataset": "WikiText-2 raw UTF-8 bytes",
            "seed": args.seed,
            "training_steps": cfg.steps,
            "batch_size": cfg.batch_size,
            "context_bytes": cfg.context_bytes,
            "evaluation_examples": cfg.eval_batches * cfg.batch_size,
            "train_corpus_bytes": int(len(train)),
            "validation_corpus_bytes": int(len(validation)),
            "bpe_vocab_size": tokenizer.get_vocab_size(),
            "adaptive_threshold_bits": adaptive_threshold,
            "adaptive_calibration_mean_bytes": statistics.mean(empirical_lengths),
            "adaptive_calibration_p95_bytes": float(np.percentile(empirical_lengths, 95)),
            "fixed_patch_size": fixed_size,
            "model_note": "Identical GRU unit encoder and Transformer query encoder for every representation; one-step next-byte prediction.",
        },
        "results": results,
    }
    payload["decision"] = make_decision(results)
    write_outputs(args.output_dir, payload)
    print((args.output_dir / "VERDICT.md").read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
