#!/usr/bin/env python3
"""Run Test 5.0 with a batch-padding-invariant query position.

The original screen appended the learned query after the batch maximum number
of units. That is valid but makes the query positional embedding depend on the
longest example in the batch. This wrapper prepends the query at position zero,
so variable-length adaptive and random controls are not penalized by unrelated
examples in the same batch.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch


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


base.UnitContextModel.forward = invariant_forward


if __name__ == "__main__":
    raise SystemExit(base.main())
