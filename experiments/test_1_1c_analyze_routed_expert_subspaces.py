#!/usr/bin/env python3
"""Robust wrapper for the routed-expert activation analysis.

The llama.cpp top-k node is a non-contiguous view. The preceding argsort node
is contiguous and contains all 64 expert IDs ordered by routing score. This
wrapper reconstructs top-8 from argsort and also removes singleton dimensions
left by the generic capture loader.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np


def load_base():
    path = Path(__file__).with_name("test_1_1_analyze_routed_expert_subspaces.py")
    spec = importlib.util.spec_from_file_location("routed_analysis_base", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


base = load_base()


def robust_find_tensor(captured: dict[str, np.ndarray], needle: str) -> np.ndarray:
    if needle == "ffn_moe_topk-7":
        if needle in captured:
            array = np.squeeze(captured[needle])
        elif "ffn_moe_argsort-7" in captured:
            ordered = np.squeeze(captured["ffn_moe_argsort-7"])
            if ordered.ndim != 2 or ordered.shape[1] < 8:
                raise ValueError(f"unexpected argsort shape: {ordered.shape}")
            array = ordered[:, :8]
        else:
            raise KeyError("neither top-k nor argsort routing IDs were captured")
        if array.ndim != 2 or array.shape[1] != 8:
            raise ValueError(f"unexpected top-k shape: {array.shape}")
        return array

    # Prefer exact graph-node names. The capture also contains a reshaped
    # ffn_norm view, so substring-only matching is ambiguous.
    if needle in captured:
        return np.squeeze(captured[needle])
    matches = [(name, np.squeeze(array)) for name, array in captured.items() if needle in name]
    if len(matches) != 1:
        raise KeyError(f"expected one tensor containing {needle!r}, found {[name for name, _ in matches]}")
    return matches[0][1]


base.find_tensor = robust_find_tensor


if __name__ == "__main__":
    raise SystemExit(base.main())
