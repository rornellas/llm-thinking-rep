#!/usr/bin/env python3
"""Methodological corrections for Test 5.0.

Corrections applied before the first real run:

1. The learned query is prepended at a constant position, so its positional
   embedding does not depend on another example's padding length.
2. The first byte's boundary surprisal is a fixed neutral BOS value. The base
   corpus surprisal array uses the byte immediately before a sampled window;
   allowing that value to affect the first boundary would expose information
   outside the declared context. Calibration and evaluation both use the
   corrected local-context rule.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import torch


NEUTRAL_BOS_SURPRISAL_BITS = 8.0


def load_base():
    path = Path(__file__).with_name("test_5_0_adaptive_byte_units.py")
    spec = importlib.util.spec_from_file_location("adaptive_byte_units_base", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


base = load_base()
original_encode_context = base.encode_context


def local_surprisal(surprisals, start: int, context_bytes: int):
    score = np.asarray(
        surprisals[start:start + context_bytes], dtype=np.float32
    ).copy()
    if score.size:
        score[0] = NEUTRAL_BOS_SURPRISAL_BITS
    return score


def local_adaptive_mean_length(
    data,
    surprisals,
    starts,
    context_bytes,
    threshold_bits,
    max_unit_bytes,
):
    lengths = []
    for start in starts:
        start = int(start)
        raw = data[start:start + context_bytes]
        score = local_surprisal(surprisals, start, context_bytes)
        lengths.extend(
            len(unit)
            for unit in base.segment_adaptive(
                raw, score, threshold_bits, max_unit_bytes
            )
        )
    return float(np.mean(lengths))


def local_calibrate_adaptive_threshold(
    data,
    surprisals,
    starts,
    context_bytes,
    target_length,
    max_unit_bytes,
):
    low, high = 0.1, 128.0
    for _ in range(24):
        midpoint = (low + high) * 0.5
        observed = local_adaptive_mean_length(
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
    histogram = []
    for start in starts:
        start = int(start)
        raw = data[start:start + context_bytes]
        score = local_surprisal(surprisals, start, context_bytes)
        histogram.extend(
            len(unit)
            for unit in base.segment_adaptive(
                raw, score, threshold, max_unit_bytes
            )
        )
    return threshold, histogram


def local_encode_context(
    variant,
    data,
    surprisals,
    start,
    cfg,
    *,
    tokenizer,
    token_lengths,
    fixed_size,
    adaptive_threshold,
    empirical_lengths,
    random_seed,
):
    if variant != "adaptive":
        return original_encode_context(
            variant,
            data,
            surprisals,
            start,
            cfg,
            tokenizer=tokenizer,
            token_lengths=token_lengths,
            fixed_size=fixed_size,
            adaptive_threshold=adaptive_threshold,
            empirical_lengths=empirical_lengths,
            random_seed=random_seed,
        )
    raw = data[start:start + cfg.context_bytes]
    score = local_surprisal(surprisals, start, cfg.context_bytes)
    units = base.segment_adaptive(
        raw, score, adaptive_threshold, cfg.max_unit_bytes
    )
    byte_lengths = [len(unit) for unit in units]
    if sum(byte_lengths) != cfg.context_bytes:
        raise AssertionError(sum(byte_lengths))
    return base.EncodedExample(
        units,
        byte_lengths,
        int(data[start + cfg.context_bytes]),
    )


def invariant_forward(self, ids, symbol_lengths, byte_lengths, padding_mask):
    batch, units, symbols = ids.shape
    embedded = self.symbol_embedding(ids).reshape(
        batch * units, symbols, self.cfg.d_model
    )
    encoded, _ = self.unit_encoder(embedded)
    flat_lengths = symbol_lengths.reshape(-1).clamp_min(1)
    gather = (flat_lengths - 1).view(-1, 1, 1).expand(
        -1, 1, self.cfg.d_model
    )
    unit_states = torch.gather(encoded, 1, gather).squeeze(1).reshape(
        batch, units, self.cfg.d_model
    )
    unit_states = unit_states + self.length_embedding(
        byte_lengths.clamp(1, self.cfg.max_unit_bytes)
    )
    query = self.query.expand(batch, -1, -1)
    sequence = torch.cat([query, unit_states], dim=1)
    positions = torch.arange(units + 1, device=ids.device)
    sequence = sequence + self.position_embedding(positions)[None, :, :]
    full_mask = torch.cat(
        [
            torch.zeros(batch, 1, dtype=torch.bool, device=ids.device),
            padding_mask,
        ],
        dim=1,
    )
    transformed = self.transformer(
        sequence, src_key_padding_mask=full_mask
    )
    return self.output(self.final_norm(transformed[:, 0]))


base.adaptive_mean_length = local_adaptive_mean_length
base.calibrate_adaptive_threshold = local_calibrate_adaptive_threshold
base.encode_context = local_encode_context
base.UnitContextModel.forward = invariant_forward


if __name__ == "__main__":
    raise SystemExit(base.main())
