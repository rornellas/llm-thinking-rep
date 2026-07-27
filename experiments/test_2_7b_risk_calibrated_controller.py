#!/usr/bin/env python3
"""Risk-calibrated rerun of Test 2.7.

The first learned controller selected the cheapest threshold whose calibration
loss was allowed to exceed static K=1 by 0.005 nat. It missed the final PASS
criterion by only 0.00105 nat on held-out validation. This rerun removes that
optimistic slack and requires a small calibration safety margin before choosing
a lower-compute threshold.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np


def load_base():
    path = Path(__file__).with_name("test_2_7_learned_rank_controller.py")
    spec = importlib.util.spec_from_file_location("learned_controller_base", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


base = load_base()


def risk_calibrated_threshold(
    model: Any,
    controllers: Sequence[Any],
    tune_batches: Sequence[tuple[Any, Any]],
    static_rows: Sequence[dict[str, float]],
    controller_overhead: float,
) -> tuple[float, list[dict[str, Any]], float]:
    static_k1_loss = next(row["validation_loss"] for row in static_rows if int(row["rank"]) == 1)
    # The dynamic policy must beat static K=1 on calibration by at least this
    # amount. This protects against calibration-to-validation drift while still
    # permitting a substantial compute reduction.
    safety_margin_nats = 0.001
    quality_limit = static_k1_loss - safety_margin_nats
    candidates: list[dict[str, Any]] = []
    for threshold in np.linspace(0.10, 0.90, 17):
        row = base.evaluate_dynamic(model, controllers, tune_batches, float(threshold), controller_overhead)
        row["quality_limit"] = quality_limit
        row["calibration_safety_margin_nats"] = safety_margin_nats
        row["within_quality_limit"] = row["validation_loss"] <= quality_limit
        candidates.append(row)
        print(
            f"risk-threshold={threshold:.2f} tune-loss={row['validation_loss']:.4f} "
            f"limit={quality_limit:.4f} mean-K={row['mean_rank']:.3f} "
            f"compute={row['projected_olmoe_compute_ratio']:.3%}",
            flush=True,
        )
    feasible = [row for row in candidates if row["within_quality_limit"]]
    if feasible:
        chosen = min(feasible, key=lambda row: (row["projected_olmoe_compute_ratio"], row["validation_loss"]))
    else:
        # Fail closed: choose the most accurate policy rather than silently
        # spending less compute at the expense of the quality constraint.
        chosen = min(candidates, key=lambda row: (row["validation_loss"], row["projected_olmoe_compute_ratio"]))
    return float(chosen["threshold"]), candidates, float(quality_limit)


base.choose_threshold = risk_calibrated_threshold


if __name__ == "__main__":
    raise SystemExit(base.main())
