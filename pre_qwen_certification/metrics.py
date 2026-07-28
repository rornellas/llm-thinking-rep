"""Numerical and hierarchical statistical utilities for certification."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import numpy as np
import torch


EPS = 1e-12


@dataclass(frozen=True)
class TensorMetrics:
    mse: float
    rmse: float
    nrmse: float
    cosine: float
    max_absolute_error: float

    def as_dict(self) -> dict[str, float]:
        return {
            "mse": self.mse,
            "rmse": self.rmse,
            "nrmse": self.nrmse,
            "cosine": self.cosine,
            "max_absolute_error": self.max_absolute_error,
        }


def tensor_metrics(prediction: torch.Tensor, target: torch.Tensor) -> TensorMetrics:
    if prediction.shape != target.shape:
        raise ValueError(
            f"shape mismatch: prediction={tuple(prediction.shape)} target={tuple(target.shape)}"
        )
    p = prediction.detach().double().reshape(-1)
    t = target.detach().double().reshape(-1)
    difference = p - t
    mse = float(torch.mean(difference.square()))
    rmse = mse**0.5
    scale = float(torch.mean(t.square()).sqrt())
    nrmse = rmse / max(scale, EPS)
    cosine = float(
        torch.dot(p, t)
        / (
            torch.linalg.vector_norm(p).clamp_min(EPS)
            * torch.linalg.vector_norm(t).clamp_min(EPS)
        )
    )
    return TensorMetrics(
        mse=mse,
        rmse=rmse,
        nrmse=nrmse,
        cosine=cosine,
        max_absolute_error=float(torch.max(torch.abs(difference))),
    )


def rowwise_nrmse(prediction: torch.Tensor, target: torch.Tensor) -> np.ndarray:
    if prediction.shape != target.shape or prediction.ndim < 2:
        raise ValueError("rowwise_nrmse expects equally shaped tensors with ndim >= 2")
    p = prediction.detach().double().reshape(prediction.shape[0], -1)
    t = target.detach().double().reshape(target.shape[0], -1)
    numerator = torch.mean((p - t).square(), dim=-1).sqrt()
    denominator = torch.mean(t.square(), dim=-1).sqrt().clamp_min(EPS)
    return (numerator / denominator).cpu().numpy()


def paired_kl_from_logits(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
) -> torch.Tensor:
    """Return per-row KL(teacher || student) in nats."""
    if student_logits.shape != teacher_logits.shape:
        raise ValueError("student and teacher logits must have equal shape")
    teacher_log_prob = torch.log_softmax(teacher_logits.double(), dim=-1)
    student_log_prob = torch.log_softmax(student_logits.double(), dim=-1)
    teacher_prob = teacher_log_prob.exp()
    return torch.sum(
        teacher_prob * (teacher_log_prob - student_log_prob), dim=-1
    )


def _crossed_matrix(
    records: Sequence[Mapping[str, object]],
    *,
    value_key: str,
    seed_key: str,
    document_key: str,
) -> tuple[np.ndarray, list[object], list[object]]:
    seeds = sorted({row[seed_key] for row in records}, key=str)
    documents = sorted({row[document_key] for row in records}, key=str)
    seed_index = {value: index for index, value in enumerate(seeds)}
    document_index = {value: index for index, value in enumerate(documents)}
    buckets: list[list[list[float]]] = [
        [[] for _ in documents] for _ in seeds
    ]
    for row in records:
        value = float(row[value_key])
        if not np.isfinite(value):
            raise ValueError(f"non-finite value in record: {row}")
        buckets[seed_index[row[seed_key]]][document_index[row[document_key]]].append(
            value
        )
    matrix = np.full((len(seeds), len(documents)), np.nan, dtype=np.float64)
    for seed_id, seed_buckets in enumerate(buckets):
        for document_id, values in enumerate(seed_buckets):
            if values:
                matrix[seed_id, document_id] = float(np.mean(values))
    return matrix, seeds, documents


def crossed_hierarchical_bootstrap(
    records: Sequence[Mapping[str, object]],
    *,
    value_key: str,
    seed_key: str = "seed",
    document_key: str = "document_id",
    samples: int = 5000,
    random_seed: int = 0,
    confidence: float = 0.95,
) -> dict[str, object]:
    """Two-way cluster bootstrap over training seeds and documents.

    Tokens/windows within a seed-document cell are averaged first.  Bootstrap
    replicates independently resample seed rows and document columns.  This avoids
    treating correlated tokens or repeated evaluation documents as IID samples.
    """
    if samples < 100:
        raise ValueError("at least 100 bootstrap samples are required")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be in (0, 1)")
    matrix, seeds, documents = _crossed_matrix(
        records,
        value_key=value_key,
        seed_key=seed_key,
        document_key=document_key,
    )
    if matrix.size == 0 or np.all(np.isnan(matrix)):
        raise ValueError("no observations")
    observed = float(np.nanmean(matrix))
    rng = np.random.default_rng(random_seed)
    draws = np.empty(samples, dtype=np.float64)
    for index in range(samples):
        sampled_seed_ids = rng.integers(0, matrix.shape[0], size=matrix.shape[0])
        sampled_document_ids = rng.integers(
            0, matrix.shape[1], size=matrix.shape[1]
        )
        sampled = matrix[np.ix_(sampled_seed_ids, sampled_document_ids)]
        draws[index] = float(np.nanmean(sampled))
    alpha = 1.0 - confidence
    return {
        "mean": observed,
        "lcb": float(np.quantile(draws, alpha / 2.0)),
        "ucb": float(np.quantile(draws, 1.0 - alpha / 2.0)),
        "confidence": confidence,
        "bootstrap_samples": samples,
        "training_seeds": [str(value) for value in seeds],
        "documents": [str(value) for value in documents],
        "effective_cells": int(np.isfinite(matrix).sum()),
    }


def one_sided_crossed_bound(
    records: Sequence[Mapping[str, object]],
    *,
    value_key: str,
    quantile: float,
    seed_key: str = "seed",
    document_key: str = "document_id",
    samples: int = 5000,
    random_seed: int = 0,
) -> float:
    if not 0.0 < quantile < 1.0:
        raise ValueError("quantile must be in (0, 1)")
    matrix, _, _ = _crossed_matrix(
        records,
        value_key=value_key,
        seed_key=seed_key,
        document_key=document_key,
    )
    rng = np.random.default_rng(random_seed)
    draws = np.empty(samples, dtype=np.float64)
    for index in range(samples):
        seed_ids = rng.integers(0, matrix.shape[0], size=matrix.shape[0])
        doc_ids = rng.integers(0, matrix.shape[1], size=matrix.shape[1])
        draws[index] = float(np.nanmean(matrix[np.ix_(seed_ids, doc_ids)]))
    return float(np.quantile(draws, quantile))


def summarize_groups(
    records: Iterable[Mapping[str, object]],
    *,
    value_key: str,
    group_keys: Sequence[str],
) -> list[dict[str, object]]:
    buckets: dict[tuple[object, ...], list[float]] = {}
    for row in records:
        key = tuple(row[name] for name in group_keys)
        buckets.setdefault(key, []).append(float(row[value_key]))
    result: list[dict[str, object]] = []
    for key, values in sorted(buckets.items(), key=lambda item: tuple(map(str, item[0]))):
        array = np.asarray(values, dtype=np.float64)
        result.append(
            {
                **dict(zip(group_keys, key, strict=True)),
                "count": int(array.size),
                "mean": float(array.mean()),
                "std": float(array.std()),
                "p50": float(np.quantile(array, 0.50)),
                "p95": float(np.quantile(array, 0.95)),
                "max": float(array.max()),
            }
        )
    return result
