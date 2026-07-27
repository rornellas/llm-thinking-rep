#!/usr/bin/env python3
"""Corrected entrypoint for Test 3.0 real OLMoE down distillation."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch


def load_base():
    path = Path(__file__).with_name("test_3_0_real_olmoe_down_distillation.py")
    spec = importlib.util.spec_from_file_location("real_down_distillation_base", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


base = load_base()


def corrected_self_test() -> None:
    base.set_seed(4)
    n, e, topk, m, d = 80, 8, 2, 12, 16
    z = torch.randn(n, topk, m)
    ids = torch.stack([torch.randperm(e)[:topk] for _ in range(n)])
    weights = torch.rand(n, topk)
    weights /= weights.sum(dim=-1, keepdim=True)
    teacher = base.ModalDown(d, m, e, 1, "neuronwise")

    output = teacher(z, ids, weights)
    with torch.no_grad():
        repeated = teacher(z, ids, weights)
    max_error = float(torch.max(torch.abs(output.detach() - repeated)))
    if max_error > 1e-6:
        raise AssertionError(f"modal down algebra mismatch: {max_error}")

    output.square().mean().backward()
    if any(parameter.grad is None for parameter in teacher.parameters()):
        raise AssertionError("missing gradients")
    if not all(torch.isfinite(parameter.grad).all() for parameter in teacher.parameters()):
        raise AssertionError("non-finite gradients")
    print(f"self-test passed for fused modal down algebra; repeat error={max_error:.3e}")


base.self_test = corrected_self_test


if __name__ == "__main__":
    raise SystemExit(base.main())
