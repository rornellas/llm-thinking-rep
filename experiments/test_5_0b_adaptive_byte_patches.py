#!/usr/bin/env python3
"""Corrected entrypoint for Test 5.0.

The original experiment remains the implementation source.  This entrypoint
replaces only the synthetic byte construction in its self-test, which must use
``np.frombuffer`` rather than treating a ``bytes`` object as a scalar array.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import torch


SOURCE = Path(__file__).with_name("test_5_0_adaptive_byte_patches.py")
spec = importlib.util.spec_from_file_location("adaptive_byte_patch_source", SOURCE)
if spec is None or spec.loader is None:
    raise RuntimeError(SOURCE)
base = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = base
spec.loader.exec_module(base)


def corrected_self_test() -> None:
    base.set_seed(5)
    synthetic = np.frombuffer(
        (b"abcd " * 400) + (b"xyz\n" * 200), dtype=np.uint8
    ).copy()
    entropy, _ = base.bigram_entropy_bits(synthetic)
    threshold, calibration = base.calibrate_adaptive_threshold(
        synthetic, entropy, 4.0, 1, 8, len(synthetic)
    )
    streams = {
        "fixed": base.segment_fixed(synthetic, 4, 8, entropy),
        "adaptive": base.segment_adaptive(synthetic, entropy, threshold, 1, 8),
        "random": base.segment_random_matched(
            synthetic, calibration, 8, entropy, 9
        ),
    }
    for stream in streams.values():
        if int(np.sum(stream.lengths)) != len(synthetic):
            raise AssertionError("stream does not cover bytes")
        if (
            stream.patches.shape[1] != 8
            or np.any(stream.lengths < 1)
            or np.any(stream.lengths > 8)
        ):
            raise AssertionError("invalid patch lengths")

    cfg = base.Config(
        d_model=24,
        n_heads=4,
        n_layers=1,
        global_ff=48,
        batch_size=2,
        steps=1,
    )
    model = base.CausalPatchByteLM(cfg, max_global_positions=16)
    indices = np.asarray([0, 2], dtype=np.int64)
    batch = streams["fixed"].batch_from_indices(
        indices, 8, torch.device("cpu")
    )
    loss, _, count = model.nll(*batch)
    if not torch.isfinite(loss) or int(count) <= 0:
        raise AssertionError((loss, count))
    loss.backward()
    if any(parameter.grad is None for parameter in model.parameters()):
        raise AssertionError("missing gradient")
    print(
        "self-test passed: "
        f"threshold={threshold:.4f}, "
        f"adaptive mean={streams['adaptive'].mean_patch_length:.3f}, "
        f"loss={float(loss.detach()):.4f}"
    )


base.self_test = corrected_self_test


if __name__ == "__main__":
    raise SystemExit(base.main())
