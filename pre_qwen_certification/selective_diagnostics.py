"""Training-only diagnostics for selective hot/cold expert compression.

No held-out language-model metric is consumed here.  The diagnostic asks two
questions before any student is trained:

1. Are expert-importance rankings stable enough to support selective capacity?
2. Does exact-hot plus shared-cold execution have a route-conditioned compute
   point that can beat a conventional narrow baseline?

A negative answer is a valid pre-training falsification of the proposed screen.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Iterable

import numpy as np
import torch

from .modal import ConventionalSwiGLUMoE
from .selective import (
    ExpertImportance,
    ImportanceMetric,
    SelectiveHotColdMoE,
    choose_hot_experts,
    score_expert_importance,
)
from .tiny_lm import CapturedLayerDataset

DETERMINISTIC_METRICS: tuple[ImportanceMetric, ...] = (
    "weighted-output-energy",
    "gate-mass",
    "routing-frequency",
)


def _average_ranks(values: np.ndarray) -> np.ndarray:
    """Return ascending average ranks, including exact ties."""
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    index = 0
    while index < len(values):
        end = index + 1
        while end < len(values) and values[order[end]] == values[order[index]]:
            end += 1
        average = 0.5 * (index + end - 1)
        ranks[order[index:end]] = average
        index = end
    return ranks


def spearman_rank_correlation(left: Iterable[float], right: Iterable[float]) -> float:
    a = np.asarray(tuple(left), dtype=np.float64)
    b = np.asarray(tuple(right), dtype=np.float64)
    if a.shape != b.shape or a.ndim != 1 or a.size < 2:
        raise ValueError("rank vectors must be one-dimensional and shape-matched")
    ra = _average_ranks(a)
    rb = _average_ranks(b)
    if np.std(ra) == 0.0 or np.std(rb) == 0.0:
        return 0.0
    return float(np.corrcoef(ra, rb)[0, 1])


def jaccard(left: Iterable[int], right: Iterable[int]) -> float:
    a, b = set(map(int, left)), set(map(int, right))
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def normalized_coverage(values: torch.Tensor, selected: torch.Tensor) -> float:
    total = float(values.sum())
    if total <= 0.0:
        return 0.0
    return float(values.index_select(0, selected).sum()) / total


def hot_slot_histogram(
    top_ids: torch.Tensor, hot_expert_ids: torch.Tensor
) -> dict[str, float]:
    hot_mask = torch.zeros(
        int(top_ids.max()) + 1 if top_ids.numel() else 0, dtype=torch.bool
    )
    needed = max(
        int(top_ids.max()) + 1 if top_ids.numel() else 0,
        int(hot_expert_ids.max()) + 1 if hot_expert_ids.numel() else 0,
    )
    if len(hot_mask) < needed:
        hot_mask = torch.zeros(needed, dtype=torch.bool)
    hot_mask[hot_expert_ids] = True
    counts = hot_mask.index_select(0, top_ids.reshape(-1)).reshape_as(top_ids).sum(-1)
    result = {
        f"hot_slots_{slot}": float((counts == slot).double().mean())
        for slot in range(top_ids.shape[1] + 1)
    }
    result.update(
        {
            "all_cold_fraction": float((counts == 0).double().mean()),
            "mixed_fraction": float(
                ((counts > 0) & (counts < top_ids.shape[1])).double().mean()
            ),
            "all_hot_fraction": float((counts == top_ids.shape[1]).double().mean()),
        }
    )
    return result


def _slice_capture(capture: CapturedLayerDataset, indices: slice) -> CapturedLayerDataset:
    return CapturedLayerDataset(
        inputs=capture.inputs[indices],
        outputs=capture.outputs[indices],
        top_ids=capture.top_ids[indices],
        route_weights=capture.route_weights[indices],
    )


def diagnose_teacher(
    teacher: ConventionalSwiGLUMoE,
    capture: CapturedLayerDataset,
    *,
    hot_counts: Iterable[int],
    cold_ranks: Iterable[int],
    random_trials: int,
    seed: int,
    chunk_size: int = 512,
) -> dict[str, Any]:
    """Produce a deterministic training-only feasibility diagnostic."""
    hot_counts = tuple(sorted(set(map(int, hot_counts))))
    cold_ranks = tuple(sorted(set(map(int, cold_ranks))))
    if not hot_counts or not cold_ranks:
        raise ValueError("hot_counts and cold_ranks must be non-empty")
    if max(hot_counts) >= teacher.geometry.n_experts:
        raise ValueError("hot count must leave at least one cold expert")
    if random_trials < 1:
        raise ValueError("random_trials must be positive")

    midpoint = len(capture.inputs) // 2
    first = _slice_capture(capture, slice(0, midpoint))
    second = _slice_capture(capture, slice(midpoint, None))
    importance = score_expert_importance(
        teacher,
        capture.inputs,
        capture.top_ids,
        capture.route_weights,
        chunk_size=chunk_size,
    )
    first_importance = score_expert_importance(
        teacher,
        first.inputs,
        first.top_ids,
        first.route_weights,
        chunk_size=chunk_size,
    )
    second_importance = score_expert_importance(
        teacher,
        second.inputs,
        second.top_ids,
        second.route_weights,
        chunk_size=chunk_size,
    )

    selections: list[dict[str, Any]] = []
    stability: dict[str, Any] = {}
    for metric in DETERMINISTIC_METRICS:
        values = importance.vector(metric)
        first_values = first_importance.vector(metric)
        second_values = second_importance.vector(metric)
        metric_stability: dict[str, Any] = {
            "spearman_halves": spearman_rank_correlation(first_values, second_values),
            "top_h_jaccard": {},
        }
        for hot_count in hot_counts:
            selected = choose_hot_experts(
                importance, hot_count=hot_count, metric=metric, seed=seed
            )
            selected_first = choose_hot_experts(
                first_importance, hot_count=hot_count, metric=metric, seed=seed
            )
            selected_second = choose_hot_experts(
                second_importance, hot_count=hot_count, metric=metric, seed=seed
            )
            metric_stability["top_h_jaccard"][str(hot_count)] = jaccard(
                selected_first.tolist(), selected_second.tolist()
            )
            for cold_rank in cold_ranks:
                module = SelectiveHotColdMoE(
                    teacher.geometry,
                    hot_expert_ids=selected,
                    cold_rank=cold_rank,
                )
                route_cost = module.route_cost_metrics(
                    capture.top_ids, capture.route_weights
                )
                selections.append(
                    {
                        "metric": metric,
                        "hot_count": hot_count,
                        "cold_rank": cold_rank,
                        "hot_expert_ids": selected.tolist(),
                        "selection_score_coverage": normalized_coverage(values, selected),
                        "routing_frequency_coverage": normalized_coverage(
                            importance.vector("routing-frequency"), selected
                        ),
                        "gate_mass_coverage": normalized_coverage(
                            importance.vector("gate-mass"), selected
                        ),
                        "weighted_output_energy_coverage": normalized_coverage(
                            importance.vector("weighted-output-energy"), selected
                        ),
                        "expert_parameter_ratio": (
                            module.expert_transform_parameter_count()
                            / (
                                teacher.gate.numel()
                                + teacher.up.numel()
                                + teacher.down.numel()
                            )
                        ),
                        **route_cost,
                        **hot_slot_histogram(capture.top_ids, selected),
                    }
                )
        stability[metric] = metric_stability

    random_rows: list[dict[str, Any]] = []
    for trial in range(random_trials):
        for hot_count in hot_counts:
            selected = choose_hot_experts(
                importance,
                hot_count=hot_count,
                metric="random",
                seed=seed + trial * 1009 + hot_count,
            )
            for cold_rank in cold_ranks:
                module = SelectiveHotColdMoE(
                    teacher.geometry,
                    hot_expert_ids=selected,
                    cold_rank=cold_rank,
                )
                random_rows.append(
                    {
                        "trial": trial,
                        "hot_count": hot_count,
                        "cold_rank": cold_rank,
                        "hot_expert_ids": selected.tolist(),
                        "routing_frequency_coverage": normalized_coverage(
                            importance.vector("routing-frequency"), selected
                        ),
                        "gate_mass_coverage": normalized_coverage(
                            importance.vector("gate-mass"), selected
                        ),
                        "weighted_output_energy_coverage": normalized_coverage(
                            importance.vector("weighted-output-energy"), selected
                        ),
                        **module.route_cost_metrics(
                            capture.top_ids, capture.route_weights
                        ),
                    }
                )

    return {
        "tokens": len(capture.inputs),
        "geometry": {
            "d_model": teacher.geometry.d_model,
            "d_ff": teacher.geometry.d_ff,
            "n_experts": teacher.geometry.n_experts,
            "top_k": teacher.geometry.top_k,
        },
        "importance": importance.as_dict(),
        "half_importance": {
            "first": first_importance.as_dict(),
            "second": second_importance.as_dict(),
        },
        "stability": stability,
        "selections": selections,
        "random_controls": random_rows,
        "methodological_note": (
            "All ranking and feasibility calculations use training captures only. "
            "Weighted-output energy is a diagonal contribution proxy, not a causal "
            "Shapley value; gate-mass, routing-frequency, and random controls are required."
        ),
    }
