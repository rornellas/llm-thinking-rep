#!/usr/bin/env python3
"""Extend Test 2.10's utility-price search beyond its truncated boundary.

The first marginal-utility experiment selected lambda=0.020, exactly the
largest candidate.  This wrapper reuses the pre-registered training and
validation protocol but expands the train-calibration price grid through 0.060
nat per residual mode.  No validation statistic participates in selection.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np


def load_base():
    path = Path(__file__).with_name("test_2_10_marginal_utility_controller.py")
    spec = importlib.util.spec_from_file_location("extended_utility_base", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


base = load_base()
original_write_outputs = base.write_outputs


def extended_select_lambda(
    controller_module: Any,
    model: Any,
    controllers: Sequence[Any],
    tune_batches: Sequence[tuple[Any, Any]],
    static_k1_losses: Sequence[float],
    overhead: float,
    max_rank: int,
    seed: int,
):
    candidates: list[dict[str, Any]] = []
    lambdas = np.concatenate(
        [
            np.arange(0.0, 0.0401, 0.002),
            np.asarray([0.045, 0.050, 0.060]),
        ]
    )
    for index, value in enumerate(lambdas):
        row = base.evaluate_dynamic(
            controller_module,
            model,
            controllers,
            tune_batches,
            float(value),
            overhead,
            max_rank,
        )
        deltas = [
            dynamic - static
            for dynamic, static in zip(
                row["batch_losses"], static_k1_losses, strict=True
            )
        ]
        row["delta_mean"] = float(np.mean(deltas))
        row["delta_ucb95"] = base.paired_bootstrap_ucb(
            deltas, seed + index
        )
        row["guard_pass"] = row["delta_ucb95"] <= 0.005
        candidates.append(row)
        print(
            f"extended-lambda={value:.3f} delta={row['delta_mean']:+.4f} "
            f"ucb95={row['delta_ucb95']:+.4f} mean-K={row['mean_rank']:.3f} "
            f"bucket16={row['bucket16_compute_ratio']:.3%}",
            flush=True,
        )
    feasible = [row for row in candidates if row["guard_pass"]]
    if feasible:
        chosen = min(
            feasible,
            key=lambda row: (
                row["bucket16_compute_ratio"], row["delta_ucb95"]
            ),
        )
    else:
        chosen = min(
            candidates,
            key=lambda row: (
                row["delta_ucb95"], row["bucket16_compute_ratio"]
            ),
        )
    return float(chosen["lambda_nats"]), candidates


def extended_write_outputs(output_dir: Path, payload: dict[str, Any]) -> None:
    payload.setdefault("metadata", {}).update(
        {
            "experiment": "Test 2.10b extended utility-price search",
            "lambda_grid": [
                *[round(float(value), 6) for value in np.arange(0.0, 0.0401, 0.002)],
                0.045,
                0.050,
                0.060,
            ],
            "selection_note": "The first Test 2.10 selected the maximum lambda=0.020. This run extends the train-calibration grid without changing seeds, model training, guard, or validation protocol.",
        }
    )
    original_write_outputs(output_dir, payload)
    verdict = output_dir / "VERDICT.md"
    if verdict.exists():
        text = verdict.read_text(encoding="utf-8")
        text = text.replace(
            "# Test 2.10 — marginal-utility rank controller",
            "# Test 2.10b — extended marginal-utility price search",
            1,
        )
        verdict.write_text(text, encoding="utf-8")


base.select_lambda = extended_select_lambda
base.write_outputs = extended_write_outputs


if __name__ == "__main__":
    raise SystemExit(base.main())
