#!/usr/bin/env python3
"""Test 5.1: do causal neural-surprisal boundaries carry useful information?

Test 5.0 established that a learned local encoder can compress a 128-byte
context into roughly 32 units without measurable next-byte degradation. It did
*not* establish that its bigram-surprisal boundaries were better than fixed or
random matched boundaries.

This experiment increases statistical power and boundary quality:

* every example predicts the next 16 raw bytes, not just one;
* three paired model-training seeds are used;
* a causal byte-GRU, trained only on the train split, estimates per-byte
  surprisal inside each context;
* neural-adaptive, fixed, and random-matched representations have the same
  average unit budget;
* byte, reversible byte-BPE, and bigram-adaptive controls are retained;
* every variant uses identical context windows, targets, model parameters,
  initialization, optimization steps, and evaluation examples within a seed.

The neural boundary model is an explicit preprocessing cost and is reported
separately. It never sees validation targets while training or calibrating the
information threshold. The first byte of every sampled context receives a fixed
neutral BOS surprisal, so no byte before the declared context can influence a
boundary.
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import random
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


NEUTRAL_BOS_BITS = 8.0
EPS = 1e-12


def load_representation_module():
    path = Path(__file__).with_name("test_5_0b_adaptive_byte_units_query_fix.py")
    spec = importlib.util.spec_from_file_location("neural_adaptive_base", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    wrapper = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = wrapper
    spec.loader.exec_module(wrapper)
    return wrapper.base


base = load_representation_module()


@dataclass
class ExperimentConfig:
    context_bytes: int = 128
    horizon_bytes: int = 16
    batch_size: int = 16
    steps: int = 700
    eval_batches: int = 40
    d_model: int = 64
    n_heads: int = 4
    n_layers: int = 2
    d_ff: int = 192
    max_unit_bytes: int = 16
    vocab_capacity: int = 512
    bpe_vocab_size: int = 512
    target_patch_bytes: float = 4.0
    lr: float = 5e-4
    weight_decay: float = 0.03
    grad_clip: float = 1.0


@dataclass
class HorizonExample:
    unit_symbols: list[list[int]]
    unit_byte_lengths: list[int]
    targets: list[int]


@dataclass
class RunResult:
    seed: int
    variant: str
    validation_bits_per_byte: float
    validation_loss_nats_per_byte: float
    best_validation_bits_per_byte: float
    final_train_bits_per_byte: float
    mean_units_per_context: float
    p95_units_per_context: float
    mean_bytes_per_unit: float
    units_per_kib: float
    attention_work_ratio_to_byte: float
    trainable_parameters: int
    training_seconds: float
    model_context_bytes_per_second: float
    preprocessing_seconds: float


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class ByteSurprisalTeacher(nn.Module):
    def __init__(self, embedding_dim: int = 48, hidden_dim: int = 128) -> None:
        super().__init__()
        self.embedding = nn.Embedding(256, embedding_dim)
        self.gru = nn.GRU(embedding_dim, hidden_dim, batch_first=True)
        self.output = nn.Linear(hidden_dim, 256)
        nn.init.normal_(self.embedding.weight, std=0.03)
        nn.init.normal_(self.output.weight, std=0.02)
        nn.init.zeros_(self.output.bias)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        hidden, _ = self.gru(self.embedding(values))
        return self.output(hidden)


def train_teacher(
    train: np.ndarray,
    validation: np.ndarray,
    *,
    seed: int,
    steps: int,
    sequence_length: int,
    batch_size: int,
) -> tuple[ByteSurprisalTeacher, dict[str, float]]:
    set_seed(seed)
    model = ByteSurprisalTeacher()
    optimizer = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=0.02)
    train_rng = np.random.default_rng(seed + 1)
    validation_rng = np.random.default_rng(seed + 2)
    started = time.perf_counter()
    final_loss = math.nan
    for step in range(1, steps + 1):
        starts = train_rng.integers(
            0,
            len(train) - sequence_length - 1,
            size=batch_size,
            endpoint=False,
        )
        windows = np.stack(
            [train[int(start):int(start) + sequence_length + 1] for start in starts]
        ).astype(np.int64, copy=False)
        x = torch.from_numpy(windows[:, :-1])
        y = torch.from_numpy(windows[:, 1:])
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, 256), y.reshape(-1))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        final_loss = float(loss.detach())
        if step == 1 or step % 200 == 0 or step == steps:
            print(
                f"boundary-teacher step={step}/{steps} "
                f"train-bpb={final_loss / math.log(2.0):.4f}",
                flush=True,
            )

    model.eval()
    losses: list[float] = []
    with torch.no_grad():
        for _ in range(20):
            starts = validation_rng.integers(
                0,
                len(validation) - sequence_length - 1,
                size=batch_size,
                endpoint=False,
            )
            windows = np.stack(
                [
                    validation[int(start):int(start) + sequence_length + 1]
                    for start in starts
                ]
            ).astype(np.int64, copy=False)
            x = torch.from_numpy(windows[:, :-1])
            y = torch.from_numpy(windows[:, 1:])
            logits = model(x)
            losses.append(
                float(F.cross_entropy(logits.reshape(-1, 256), y.reshape(-1)))
            )
    return model, {
        "parameters": float(sum(parameter.numel() for parameter in model.parameters())),
        "training_seconds": time.perf_counter() - started,
        "final_train_bits_per_byte": final_loss / math.log(2.0),
        "validation_bits_per_byte": statistics.mean(losses) / math.log(2.0),
    }


@torch.no_grad()
def neural_context_surprisals(
    teacher: ByteSurprisalTeacher,
    data: np.ndarray,
    starts: np.ndarray,
    context_bytes: int,
    batch_size: int = 64,
) -> np.ndarray:
    result = np.empty((len(starts), context_bytes), dtype=np.float32)
    teacher.eval()
    for offset in range(0, len(starts), batch_size):
        selected = starts[offset:offset + batch_size]
        contexts = np.stack(
            [data[int(start):int(start) + context_bytes] for start in selected]
        ).astype(np.int64, copy=False)
        score = np.full(contexts.shape, NEUTRAL_BOS_BITS, dtype=np.float32)
        if context_bytes > 1:
            x = torch.from_numpy(contexts[:, :-1])
            targets = torch.from_numpy(contexts[:, 1:])
            logits = teacher(x)
            losses = F.cross_entropy(
                logits.reshape(-1, 256),
                targets.reshape(-1),
                reduction="none",
            ).reshape(len(selected), context_bytes - 1)
            score[:, 1:] = (losses / math.log(2.0)).cpu().numpy()
        result[offset:offset + len(selected)] = score
    return result


def segment_adaptive(
    raw: np.ndarray,
    score: np.ndarray,
    threshold_bits: float,
    max_unit_bytes: int,
) -> list[list[int]]:
    units: list[list[int]] = []
    start = 0
    information = 0.0
    for index, value in enumerate(score):
        information += float(value)
        length = index - start + 1
        if information >= threshold_bits or length >= max_unit_bytes:
            units.append(raw[start:index + 1].astype(np.int64).tolist())
            start = index + 1
            information = 0.0
    if start < len(raw):
        units.append(raw[start:].astype(np.int64).tolist())
    return units


def calibrate_from_context_scores(
    data: np.ndarray,
    starts: np.ndarray,
    scores: np.ndarray,
    *,
    context_bytes: int,
    target_length: float,
    max_unit_bytes: int,
) -> tuple[float, list[int]]:
    def lengths_at(threshold: float) -> list[int]:
        lengths: list[int] = []
        for row, start in enumerate(starts):
            raw = data[int(start):int(start) + context_bytes]
            lengths.extend(
                len(unit)
                for unit in segment_adaptive(
                    raw, scores[row], threshold, max_unit_bytes
                )
            )
        return lengths

    low, high = 0.1, 160.0
    for _ in range(25):
        midpoint = (low + high) * 0.5
        observed = statistics.mean(lengths_at(midpoint))
        if observed < target_length:
            low = midpoint
        else:
            high = midpoint
    threshold = (low + high) * 0.5
    return threshold, lengths_at(threshold)


def collate(
    examples: Sequence[HorizonExample], cfg: ExperimentConfig
) -> tuple[torch.Tensor, ...]:
    batch = len(examples)
    max_units = max(len(example.unit_symbols) for example in examples)
    max_symbols = max(len(unit) for example in examples for unit in example.unit_symbols)
    ids = torch.zeros(batch, max_units, max_symbols, dtype=torch.long)
    symbol_lengths = torch.ones(batch, max_units, dtype=torch.long)
    byte_lengths = torch.ones(batch, max_units, dtype=torch.long)
    padding_mask = torch.ones(batch, max_units, dtype=torch.bool)
    targets = torch.empty(batch, cfg.horizon_bytes, dtype=torch.long)
    for row, example in enumerate(examples):
        targets[row] = torch.tensor(example.targets, dtype=torch.long)
        for column, (unit, byte_length) in enumerate(
            zip(example.unit_symbols, example.unit_byte_lengths, strict=True)
        ):
            ids[row, column, :len(unit)] = torch.tensor(unit, dtype=torch.long)
            symbol_lengths[row, column] = len(unit)
            byte_lengths[row, column] = byte_length
            padding_mask[row, column] = False
    return ids, symbol_lengths, byte_lengths, padding_mask, targets


class HorizonUnitModel(base.UnitContextModel):
    def __init__(self, cfg: ExperimentConfig) -> None:
        base_cfg = base.Config(
            context_bytes=cfg.context_bytes,
            batch_size=cfg.batch_size,
            steps=cfg.steps,
            eval_batches=cfg.eval_batches,
            d_model=cfg.d_model,
            n_heads=cfg.n_heads,
            n_layers=cfg.n_layers,
            d_ff=cfg.d_ff,
            max_unit_bytes=cfg.max_unit_bytes,
            vocab_capacity=cfg.vocab_capacity,
            bpe_vocab_size=cfg.bpe_vocab_size,
            target_patch_bytes=cfg.target_patch_bytes,
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
            grad_clip=cfg.grad_clip,
        )
        super().__init__(base_cfg)
        self.horizon_bytes = cfg.horizon_bytes
        self.output = nn.Linear(cfg.d_model, cfg.horizon_bytes * 256)
        nn.init.normal_(self.output.weight, std=0.02)
        nn.init.zeros_(self.output.bias)

    def forward(self, ids, symbol_lengths, byte_lengths, padding_mask):
        flat = super().forward(ids, symbol_lengths, byte_lengths, padding_mask)
        return flat.reshape(flat.shape[0], self.horizon_bytes, 256)


def make_example(
    variant: str,
    data: np.ndarray,
    start: int,
    cfg: ExperimentConfig,
    *,
    tokenizer: Any,
    token_lengths: dict[int, int],
    fixed_size: int,
    bigram_score: np.ndarray,
    bigram_threshold: float,
    neural_score: np.ndarray,
    neural_threshold: float,
    neural_length_histogram: Sequence[int],
    random_seed: int,
) -> HorizonExample:
    raw = data[start:start + cfg.context_bytes]
    if variant == "byte":
        units = [[int(value)] for value in raw]
        byte_lengths = [1] * len(units)
    elif variant == "bpe512":
        ids = tokenizer.encode(base.bytes_to_private(raw)).ids
        recovered = "".join(tokenizer.id_to_token(index) or "" for index in ids)
        if base.private_to_bytes(recovered) != bytes(raw):
            raise AssertionError("BPE reconstruction mismatch")
        units = [[int(index)] for index in ids]
        byte_lengths = [token_lengths[int(index)] for index in ids]
    elif variant == "fixed":
        units = base.segment_fixed(raw, fixed_size)
        byte_lengths = [len(unit) for unit in units]
    elif variant == "bigram-adaptive":
        units = segment_adaptive(
            raw, bigram_score, bigram_threshold, cfg.max_unit_bytes
        )
        byte_lengths = [len(unit) for unit in units]
    elif variant == "neural-adaptive":
        units = segment_adaptive(
            raw, neural_score, neural_threshold, cfg.max_unit_bytes
        )
        byte_lengths = [len(unit) for unit in units]
    elif variant == "random-matched":
        units = base.segment_random_matched(
            raw, neural_length_histogram, random_seed ^ start
        )
        byte_lengths = [len(unit) for unit in units]
    else:
        raise ValueError(variant)
    if sum(byte_lengths) != cfg.context_bytes:
        raise AssertionError((variant, sum(byte_lengths)))
    targets = data[
        start + cfg.context_bytes:
        start + cfg.context_bytes + cfg.horizon_bytes
    ].astype(np.int64).tolist()
    if len(targets) != cfg.horizon_bytes:
        raise AssertionError(len(targets))
    return HorizonExample(units, byte_lengths, targets)


def build_examples(
    variant: str,
    data: np.ndarray,
    starts: np.ndarray,
    cfg: ExperimentConfig,
    *,
    tokenizer: Any,
    token_lengths: dict[int, int],
    fixed_size: int,
    bigram_scores: np.ndarray,
    bigram_threshold: float,
    neural_scores: np.ndarray,
    neural_threshold: float,
    neural_length_histogram: Sequence[int],
    random_seed: int,
) -> tuple[list[HorizonExample], float]:
    started = time.perf_counter()
    examples = [
        make_example(
            variant,
            data,
            int(start),
            cfg,
            tokenizer=tokenizer,
            token_lengths=token_lengths,
            fixed_size=fixed_size,
            bigram_score=bigram_scores[row],
            bigram_threshold=bigram_threshold,
            neural_score=neural_scores[row],
            neural_threshold=neural_threshold,
            neural_length_histogram=neural_length_histogram,
            random_seed=random_seed,
        )
        for row, start in enumerate(starts)
    ]
    return examples, time.perf_counter() - started


@torch.no_grad()
def evaluate(
    model: HorizonUnitModel,
    examples: Sequence[HorizonExample],
    cfg: ExperimentConfig,
) -> tuple[float, np.ndarray]:
    model.eval()
    losses: list[np.ndarray] = []
    for start in range(0, len(examples), cfg.batch_size):
        batch = examples[start:start + cfg.batch_size]
        ids, symbol_lengths, byte_lengths, padding_mask, targets = collate(batch, cfg)
        logits = model(ids, symbol_lengths, byte_lengths, padding_mask)
        per_byte = F.cross_entropy(
            logits.reshape(-1, 256), targets.reshape(-1), reduction="none"
        ).reshape(len(batch), cfg.horizon_bytes)
        losses.append(per_byte.mean(dim=1).cpu().numpy())
    array = np.concatenate(losses)
    return float(np.mean(array)), array


def representation_stats(
    examples: Sequence[HorizonExample], cfg: ExperimentConfig
) -> dict[str, float]:
    counts = np.asarray(
        [len(example.unit_symbols) for example in examples], dtype=np.float64
    )
    return {
        "mean_units": float(np.mean(counts)),
        "p95_units": float(np.percentile(counts, 95)),
        "mean_bytes_per_unit": float(cfg.context_bytes / np.mean(counts)),
        "units_per_kib": float(np.mean(counts) / cfg.context_bytes * 1024.0),
        "attention_ratio": float(
            np.mean(counts * counts) / (cfg.context_bytes * cfg.context_bytes)
        ),
    }


def train_variant(
    variant: str,
    seed: int,
    train_examples: Sequence[HorizonExample],
    validation_examples: Sequence[HorizonExample],
    preprocessing_seconds: float,
    cfg: ExperimentConfig,
) -> tuple[RunResult, np.ndarray]:
    set_seed(seed)
    model = HorizonUnitModel(cfg)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    best = math.inf
    final_train = math.nan
    started = time.perf_counter()
    for step in range(1, cfg.steps + 1):
        offset = (step - 1) * cfg.batch_size
        batch = train_examples[offset:offset + cfg.batch_size]
        ids, symbol_lengths, byte_lengths, padding_mask, targets = collate(batch, cfg)
        model.train()
        logits = model(ids, symbol_lengths, byte_lengths, padding_mask)
        loss = F.cross_entropy(logits.reshape(-1, 256), targets.reshape(-1))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()
        final_train = float(loss.detach())
        if step == 1 or step % 200 == 0 or step == cfg.steps:
            validation_loss, _ = evaluate(model, validation_examples, cfg)
            best = min(best, validation_loss)
            print(
                f"seed={seed} {variant} step={step}/{cfg.steps} "
                f"train-bpb={final_train / math.log(2.0):.4f} "
                f"val-bpb={validation_loss / math.log(2.0):.4f}",
                flush=True,
            )
    training_seconds = time.perf_counter() - started
    validation_loss, per_example = evaluate(model, validation_examples, cfg)
    best = min(best, validation_loss)
    stats = representation_stats(validation_examples, cfg)
    context_bytes_seen = cfg.steps * cfg.batch_size * cfg.context_bytes
    result = RunResult(
        seed=seed,
        variant=variant,
        validation_bits_per_byte=validation_loss / math.log(2.0),
        validation_loss_nats_per_byte=validation_loss,
        best_validation_bits_per_byte=best / math.log(2.0),
        final_train_bits_per_byte=final_train / math.log(2.0),
        mean_units_per_context=stats["mean_units"],
        p95_units_per_context=stats["p95_units"],
        mean_bytes_per_unit=stats["mean_bytes_per_unit"],
        units_per_kib=stats["units_per_kib"],
        attention_work_ratio_to_byte=stats["attention_ratio"],
        trainable_parameters=sum(parameter.numel() for parameter in model.parameters()),
        training_seconds=training_seconds,
        model_context_bytes_per_second=context_bytes_seen / max(training_seconds, EPS),
        preprocessing_seconds=preprocessing_seconds,
    )
    return result, per_example


def paired_bootstrap_ucb(
    differences: np.ndarray,
    *,
    seed: int,
    quantile: float = 0.95,
    samples: int = 5000,
) -> float:
    values = np.asarray(differences, dtype=np.float64)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(samples, len(values)))
    return float(np.quantile(values[indices].mean(axis=1), quantile))


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
                "validation_bpb_mean": statistics.mean(
                    row["validation_bits_per_byte"] for row in selected
                ),
                "validation_bpb_std": statistics.pstdev(
                    row["validation_bits_per_byte"] for row in selected
                ),
                "mean_units_per_context": statistics.mean(
                    row["mean_units_per_context"] for row in selected
                ),
                "mean_bytes_per_unit": statistics.mean(
                    row["mean_bytes_per_unit"] for row in selected
                ),
                "attention_work_ratio_to_byte": statistics.mean(
                    row["attention_work_ratio_to_byte"] for row in selected
                ),
                "model_context_bytes_per_second_mean": statistics.mean(
                    row["model_context_bytes_per_second"] for row in selected
                ),
            }
        )

    seeds = sorted({int(row["seed"]) for row in rows})
    comparisons: list[dict[str, Any]] = []
    for seed in seeds:
        neural = losses[(seed, "neural-adaptive")] / math.log(2.0)
        for control in ("fixed", "random-matched", "bigram-adaptive"):
            control_loss = losses[(seed, control)] / math.log(2.0)
            # Positive means the neural representation has lower BpB.
            advantage = control_loss - neural
            comparisons.append(
                {
                    "seed": seed,
                    "control": control,
                    "neural_advantage_bpb": float(np.mean(advantage)),
                    "neural_advantage_lcb95_bpb": float(
                        -paired_bootstrap_ucb(
                            -advantage,
                            seed=seed + sum(ord(char) for char in control),
                        )
                    ),
                }
            )

    summary_by_variant = {row["variant"]: row for row in summary}
    neural_summary = summary_by_variant["neural-adaptive"]
    bpe_summary = summary_by_variant["bpe512"]
    fixed_comparisons = [
        row for row in comparisons if row["control"] == "fixed"
    ]
    random_comparisons = [
        row for row in comparisons if row["control"] == "random-matched"
    ]
    unit_reduction_vs_bpe = 1.0 - (
        neural_summary["mean_units_per_context"]
        / bpe_summary["mean_units_per_context"]
    )
    bpb_ratio_to_bpe = (
        neural_summary["validation_bpb_mean"]
        / bpe_summary["validation_bpb_mean"]
    )
    fixed_every_seed = all(
        row["neural_advantage_bpb"] > 0.0 for row in fixed_comparisons
    )
    random_every_seed = all(
        row["neural_advantage_bpb"] > 0.0 for row in random_comparisons
    )
    fixed_mean = statistics.mean(
        row["neural_advantage_bpb"] for row in fixed_comparisons
    )
    random_mean = statistics.mean(
        row["neural_advantage_bpb"] for row in random_comparisons
    )
    fixed_lcb_worst = min(
        row["neural_advantage_lcb95_bpb"] for row in fixed_comparisons
    )
    random_lcb_worst = min(
        row["neural_advantage_lcb95_bpb"] for row in random_comparisons
    )
    if (
        bpb_ratio_to_bpe <= 1.01
        and unit_reduction_vs_bpe >= 0.35
        and fixed_every_seed
        and random_every_seed
        and fixed_mean >= 0.002
        and random_mean >= 0.002
        and fixed_lcb_worst > -0.001
        and random_lcb_worst > -0.001
    ):
        verdict = "NEURAL_BOUNDARY_SIGNAL"
    elif bpb_ratio_to_bpe <= 1.02 and unit_reduction_vs_bpe >= 0.35:
        verdict = "PATCH_COMPRESSION_ONLY"
    else:
        verdict = "FAIL"
    decision = {
        "verdict": verdict,
        "neural_bpb_ratio_to_bpe": bpb_ratio_to_bpe,
        "neural_unit_reduction_vs_bpe": unit_reduction_vs_bpe,
        "neural_advantage_over_fixed_mean_bpb": fixed_mean,
        "neural_advantage_over_random_mean_bpb": random_mean,
        "neural_beats_fixed_every_seed": fixed_every_seed,
        "neural_beats_random_every_seed": random_every_seed,
        "worst_fixed_advantage_lcb95_bpb": fixed_lcb_worst,
        "worst_random_advantage_lcb95_bpb": random_lcb_worst,
        "rule": (
            "Strong signal requires neural adaptive BpB within 1% of BPE, "
            "at least 35% fewer units than BPE, positive advantage over fixed "
            "and random matched boundaries in every seed, mean advantages of "
            "at least 0.002 BpB, and no seed LCB95 below -0.001 BpB."
        ),
    }
    return summary, {"decision": decision, "comparisons": comparisons}


def write_outputs(output_dir: Path, payload: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "neural_adaptive_units.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with (output_dir / "runs.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(payload["runs"][0].keys()))
        writer.writeheader(); writer.writerows(payload["runs"])
    with (output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(payload["summary"][0].keys()))
        writer.writeheader(); writer.writerows(payload["summary"])
    with (output_dir / "comparisons.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(payload["comparisons"][0].keys())
        )
        writer.writeheader(); writer.writerows(payload["comparisons"])

    d = payload["decision"]
    lines = [
        "# Test 5.1 — neural adaptive byte-unit boundaries",
        "",
        f"**Decision:** **{d['verdict']}**",
        "",
        "| Representation | Runs | Bits/byte | Units/context | Bytes/unit | Attention work/byte | Context bytes/s |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["summary"]:
        lines.append(
            f"| {row['variant']} | {row['runs']} | "
            f"{row['validation_bpb_mean']:.4f} ± {row['validation_bpb_std']:.4f} | "
            f"{row['mean_units_per_context']:.2f} | {row['mean_bytes_per_unit']:.2f} | "
            f"{row['attention_work_ratio_to_byte']:.3%} | "
            f"{row['model_context_bytes_per_second_mean']:.0f} |"
        )
    lines += [
        "",
        f"- Neural/BPE BpB ratio: `{d['neural_bpb_ratio_to_bpe']:.4f}`.",
        f"- Neural unit reduction versus BPE: `{d['neural_unit_reduction_vs_bpe']:+.2%}`.",
        f"- Mean neural advantage over fixed: `{d['neural_advantage_over_fixed_mean_bpb']:+.4f}` BpB.",
        f"- Mean neural advantage over random matched: `{d['neural_advantage_over_random_mean_bpb']:+.4f}` BpB.",
        f"- Beats fixed in every seed: `{d['neural_beats_fixed_every_seed']}`; random in every seed: `{d['neural_beats_random_every_seed']}`.",
        f"- Boundary teacher: `{payload['metadata']['teacher_parameters']:.0f}` parameters, validation `{payload['metadata']['teacher_validation_bpb']:.4f}` BpB, training `{payload['metadata']['teacher_training_seconds']:.1f}` s.",
        f"- Neural threshold: `{payload['metadata']['neural_threshold_bits']:.3f}` bits; bigram threshold: `{payload['metadata']['bigram_threshold_bits']:.3f}` bits; fixed patch `{payload['metadata']['fixed_patch_size']}` bytes.",
        "",
        "The task predicts the next 16 raw bytes from the same 128-byte context. The boundary teacher and all thresholds are train-only; final comparisons use the held-out WikiText-2 validation split.",
    ]
    (output_dir / "VERDICT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def self_test() -> None:
    raw = np.frombuffer(("alpha β código data " * 80).encode("utf-8"), dtype=np.uint8)
    teacher, _ = train_teacher(
        raw,
        raw,
        seed=11,
        steps=2,
        sequence_length=32,
        batch_size=2,
    )
    starts = np.asarray([0, 4, 8, 12], dtype=np.int64)
    scores = neural_context_surprisals(teacher, raw, starts, 32, batch_size=2)
    threshold, histogram = calibrate_from_context_scores(
        raw,
        starts,
        scores,
        context_bytes=32,
        target_length=4.0,
        max_unit_bytes=8,
    )
    if not math.isfinite(threshold) or not histogram:
        raise AssertionError((threshold, histogram))
    cfg = ExperimentConfig(
        context_bytes=32,
        horizon_bytes=4,
        batch_size=2,
        steps=1,
        eval_batches=1,
        d_model=16,
        n_heads=4,
        n_layers=1,
        d_ff=32,
        max_unit_bytes=8,
    )
    model = HorizonUnitModel(cfg)
    example = HorizonExample(
        unit_symbols=segment_adaptive(raw[:32], scores[0], threshold, 8),
        unit_byte_lengths=[],
        targets=raw[32:36].astype(np.int64).tolist(),
    )
    example.unit_byte_lengths = [len(unit) for unit in example.unit_symbols]
    logits = model(*collate([example, example], cfg)[:-1])
    if logits.shape != (2, 4, 256) or not torch.isfinite(logits).all():
        raise AssertionError(logits.shape)
    print("self-test passed")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-bytes", type=Path)
    parser.add_argument("--validation-bytes", type=Path)
    parser.add_argument("--output-dir", type=Path, required=False)
    parser.add_argument("--seeds", default="51101,52202,53303")
    parser.add_argument("--steps", type=int, default=700)
    parser.add_argument("--teacher-steps", type=int, default=700)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args(argv)

    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    if args.self_test:
        self_test(); return 0
    if args.train_bytes is None or args.validation_bytes is None or args.output_dir is None:
        parser.error("train, validation, and output paths are required")

    cfg = ExperimentConfig(steps=args.steps)
    train = np.frombuffer(args.train_bytes.read_bytes(), dtype=np.uint8)
    validation = np.frombuffer(args.validation_bytes.read_bytes(), dtype=np.uint8)
    minimum = cfg.context_bytes + cfg.horizon_bytes + 1
    if len(train) <= minimum or len(validation) <= minimum:
        raise ValueError((len(train), len(validation)))
    args.output_dir.mkdir(parents=True, exist_ok=True)

    teacher, teacher_stats = train_teacher(
        train,
        validation,
        seed=51000,
        steps=args.teacher_steps,
        sequence_length=cfg.context_bytes,
        batch_size=32,
    )
    tokenizer = base.train_byte_bpe(
        train, cfg.bpe_vocab_size, args.output_dir / "byte_bpe_tokenizer.json"
    )
    token_lengths = base.bpe_token_lengths(tokenizer)
    bigram_table = base.train_bigram_surprisal(train)
    train_bigram_corpus = base.corpus_surprisal(train, bigram_table)
    validation_bigram_corpus = base.corpus_surprisal(validation, bigram_table)

    calibration_rng = np.random.default_rng(51001)
    calibration_starts = calibration_rng.integers(
        0,
        len(train) - cfg.context_bytes - cfg.horizon_bytes,
        size=512,
        endpoint=False,
        dtype=np.int64,
    )
    calibration_neural_scores = neural_context_surprisals(
        teacher, train, calibration_starts, cfg.context_bytes
    )
    neural_threshold, neural_histogram = calibrate_from_context_scores(
        train,
        calibration_starts,
        calibration_neural_scores,
        context_bytes=cfg.context_bytes,
        target_length=cfg.target_patch_bytes,
        max_unit_bytes=cfg.max_unit_bytes,
    )
    # Reuse the corrected train-only local-context bigram calibration.
    bigram_threshold, _ = base.calibrate_adaptive_threshold(
        train,
        train_bigram_corpus,
        calibration_starts,
        cfg.context_bytes,
        cfg.target_patch_bytes,
        cfg.max_unit_bytes,
    )
    fixed_size = max(1, int(round(statistics.mean(neural_histogram))))
    print(
        f"neural threshold={neural_threshold:.4f} bits; "
        f"bigram threshold={bigram_threshold:.4f}; "
        f"mean patch={statistics.mean(neural_histogram):.3f}; fixed={fixed_size}",
        flush=True,
    )

    variants = [
        "byte",
        "bpe512",
        "fixed",
        "bigram-adaptive",
        "neural-adaptive",
        "random-matched",
    ]
    rows: list[dict[str, Any]] = []
    loss_arrays: dict[tuple[int, str], np.ndarray] = {}
    seeds = [int(value) for value in args.seeds.split(",") if value.strip()]
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
        train_neural_scores = neural_context_surprisals(
            teacher, train, train_starts, cfg.context_bytes
        )
        validation_neural_scores = neural_context_surprisals(
            teacher, validation, validation_starts, cfg.context_bytes
        )

        def local_bigram_scores(data, corpus_scores, starts):
            result = np.empty((len(starts), cfg.context_bytes), dtype=np.float32)
            for row, start in enumerate(starts):
                score = np.asarray(
                    corpus_scores[int(start):int(start) + cfg.context_bytes],
                    dtype=np.float32,
                ).copy()
                score[0] = NEUTRAL_BOS_BITS
                result[row] = score
            return result

        train_bigram_scores = local_bigram_scores(
            train, train_bigram_corpus, train_starts
        )
        validation_bigram_scores = local_bigram_scores(
            validation, validation_bigram_corpus, validation_starts
        )
        for variant in variants:
            train_examples, prep_train = build_examples(
                variant,
                train,
                train_starts,
                cfg,
                tokenizer=tokenizer,
                token_lengths=token_lengths,
                fixed_size=fixed_size,
                bigram_scores=train_bigram_scores,
                bigram_threshold=bigram_threshold,
                neural_scores=train_neural_scores,
                neural_threshold=neural_threshold,
                neural_length_histogram=neural_histogram,
                random_seed=seed + 77,
            )
            validation_examples, prep_validation = build_examples(
                variant,
                validation,
                validation_starts,
                cfg,
                tokenizer=tokenizer,
                token_lengths=token_lengths,
                fixed_size=fixed_size,
                bigram_scores=validation_bigram_scores,
                bigram_threshold=bigram_threshold,
                neural_scores=validation_neural_scores,
                neural_threshold=neural_threshold,
                neural_length_histogram=neural_histogram,
                random_seed=seed + 77,
            )
            result, per_example = train_variant(
                variant,
                seed,
                train_examples,
                validation_examples,
                prep_train + prep_validation,
                cfg,
            )
            rows.append(asdict(result))
            loss_arrays[(seed, variant)] = per_example
            del train_examples, validation_examples

    summary, aggregate_data = aggregate(rows, loss_arrays)
    payload = {
        "metadata": {
            "task": "predict next 16 raw UTF-8 bytes from a 128-byte context",
            "dataset": "WikiText-2 raw UTF-8 bytes",
            "seeds": seeds,
            "steps_per_variant": cfg.steps,
            "teacher_steps": args.teacher_steps,
            "teacher_parameters": teacher_stats["parameters"],
            "teacher_training_seconds": teacher_stats["training_seconds"],
            "teacher_train_bpb": teacher_stats["final_train_bits_per_byte"],
            "teacher_validation_bpb": teacher_stats["validation_bits_per_byte"],
            "neural_threshold_bits": neural_threshold,
            "bigram_threshold_bits": bigram_threshold,
            "neural_calibration_mean_patch_bytes": statistics.mean(neural_histogram),
            "neural_calibration_p95_patch_bytes": float(
                np.percentile(neural_histogram, 95)
            ),
            "fixed_patch_size": fixed_size,
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
