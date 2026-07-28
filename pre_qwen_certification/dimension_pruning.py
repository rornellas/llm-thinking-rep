"""Minimal dimension-selection primitives used by conditional pruning tests.

The full research branch contains additional scorers and physical pruned-MoE
implementations.  This reconstructed module preserves the immutable selection
representation needed by the audited conditional-ablation artifacts.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib

import torch


@dataclass(frozen=True)
class DimensionSelection:
    indices: tuple[tuple[int, ...], ...]
    budget_ratio: float
    allocation: str
    score_name: str

    @property
    def total_dimensions(self) -> int:
        return sum(len(row) for row in self.indices)

    @property
    def widths(self) -> tuple[int, ...]:
        return tuple(len(row) for row in self.indices)

    def digest(self) -> str:
        payload = "|".join(",".join(str(value) for value in row) for row in self.indices)
        return hashlib.sha256(payload.encode("ascii")).hexdigest()

    def as_dict(self) -> dict[str, object]:
        return {
            "indices": [list(row) for row in self.indices],
            "widths": list(self.widths),
            "total_dimensions": self.total_dimensions,
            "budget_ratio": float(self.budget_ratio),
            "allocation": self.allocation,
            "score_name": self.score_name,
            "sha256": self.digest(),
        }


def select_per_expert_dimensions(
    scores: torch.Tensor, *, budget_ratio: float, score_name: str
) -> DimensionSelection:
    if scores.ndim != 2:
        raise ValueError("scores must have shape [experts, width]")
    if not 0.0 < budget_ratio <= 1.0:
        raise ValueError("budget_ratio must be in (0, 1]")
    retained = max(1, min(scores.shape[1], int(round(scores.shape[1] * budget_ratio))))
    rows = []
    for expert in range(scores.shape[0]):
        order = sorted(
            range(scores.shape[1]),
            key=lambda coordinate: (-float(scores[expert, coordinate]), coordinate),
        )
        rows.append(tuple(sorted(order[:retained])))
    return DimensionSelection(
        indices=tuple(rows),
        budget_ratio=retained / scores.shape[1],
        allocation="per-expert",
        score_name=score_name,
    )


def score_rank_correlation(left: torch.Tensor, right: torch.Tensor) -> float:
    if left.shape != right.shape:
        raise ValueError("score tensors must have identical shapes")
    a = left.detach().double().reshape(-1)
    b = right.detach().double().reshape(-1)
    ar = torch.argsort(torch.argsort(a)).double()
    br = torch.argsort(torch.argsort(b)).double()
    ar -= ar.mean(); br -= br.mean()
    denom = torch.sqrt(torch.sum(ar.square()) * torch.sum(br.square())).clamp_min(1e-12)
    return float(torch.sum(ar * br) / denom)
