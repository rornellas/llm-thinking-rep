#!/usr/bin/env python3
"""Balanced calibration for the learned Modal-MoE rank controller.

The deployment acceptance limit is +0.010 nat versus static K=1. This policy
uses only +0.003 nat of that budget on calibration, leaving +0.007 nat for
calibration-to-validation drift. With the larger 16-batch calibration set this
selects a materially cheaper policy without using validation labels.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np


def load_base():
    path = Path(__file__).with_name("test_2_7_learned_rank_controller.py")
    spec = importlib.util.spec_from_file_location("learned_controller_balanced_base", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


base = load_base()


def balanced_threshold(
    model: Any,
    controllers: Sequence[Any],
    tune_batches: Sequence[tuple[Any, Any]],
    static_rows: Sequence[dict[str, float]],
    controller_overhead: float,
) -> tuple[float, list[dict[str, Any]], float]:
    static_k1_loss = next(row["validation_loss"] for row in static_rows if int(row["rank"]) == 1)
    calibration_budget_nats = 0.003
    quality_limit = static_k1_loss + calibration_budget_nats
    candidates: list[dict[str, Any]] = []
    for threshold in np.linspace(0.10, 0.90, 17):
        row = base.evaluate_dynamic(model, controllers, tune_batches, float(threshold), controller_overhead)
        row["quality_limit"] = quality_limit
        row["calibration_budget_nats"] = calibration_budget_nats
        row["reserved_drift_budget_nats"] = 0.010 - calibration_budget_nats
        row["within_quality_limit"] = row["validation_loss"] <= quality_limit
        candidates.append(row)
        print(
            f"balanced-threshold={threshold:.2f} tune-loss={row['validation_loss']:.4f} "
            f"limit={quality_limit:.4f} mean-K={row['mean_rank']:.3f} "
            f"compute={row['projected_olmoe_compute_ratio']:.3%}",
            flush=True,
        )
    feasible = [row for row in candidates if row["within_quality_limit"]]
    if feasible:
        chosen = min(feasible, key=lambda row: (row["projected_olmoe_compute_ratio"], row["validation_loss"]))
    else:
        chosen = min(candidates, key=lambda row: (row["validation_loss"], row["projected_olmoe_compute_ratio"]))
    return float(chosen["threshold"]), candidates, float(quality_limit)


base.choose_threshold = balanced_threshold


if __name__ == "__main__":
    raise SystemExit(base.main())
