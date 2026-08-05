from __future__ import annotations

import torch

from scripts.diagnose_native_compact_gate_2a import (
    centered_variance_ratio,
    pairwise_cosine,
    spectrum_metrics,
)


def test_spectrum_metrics_detect_rank_one_and_balanced_rank() -> None:
    rank_one = torch.diag(torch.tensor([3.0, 0.0, 0.0]))
    result = spectrum_metrics(rank_one)
    assert result["rank95"] == 1
    assert abs(float(result["top1_energy"]) - 1.0) < 1e-12
    assert abs(float(result["stable_rank"]) - 1.0) < 1e-12

    balanced = torch.eye(4)
    result = spectrum_metrics(balanced)
    assert result["rank95"] == 4
    assert abs(float(result["top1_energy"]) - 0.25) < 1e-12
    assert abs(float(result["stable_rank"]) - 4.0) < 1e-12


def test_expert_similarity_metrics_separate_identical_and_orthogonal_banks() -> None:
    identical = [torch.eye(2), torch.eye(2), torch.eye(2)]
    cosine = pairwise_cosine(identical)
    assert abs(cosine["mean"] - 1.0) < 1e-12
    assert centered_variance_ratio(identical) == 0.0

    orthogonal = [
        torch.tensor([[1.0, 0.0], [0.0, 0.0]]),
        torch.tensor([[0.0, 1.0], [0.0, 0.0]]),
        torch.tensor([[0.0, 0.0], [1.0, 0.0]]),
    ]
    cosine = pairwise_cosine(orthogonal)
    assert abs(cosine["mean"]) < 1e-12
    ratio = centered_variance_ratio(orthogonal)
    assert 0.6 < ratio < 0.7
