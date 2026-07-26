#!/usr/bin/env python3
"""Permutation/scale alignment diagnostic for OLMoE experts.

This is Test 0.5 after a raw expert-axis PCA failed.  It asks whether the
near-full-rank spectrum is mainly caused by exact hidden-neuron symmetries:

* permutation of the SwiGLU intermediate neurons;
* reciprocal scaling/sign freedom between ``up`` rows and ``down`` columns.

The matching coordinates and evaluation coordinates are disjoint.  The probe
runs two deterministic expert subsets and two independent reference starts.
It can automatically expand to all experts only when a conservative pilot gate
is met.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

from modal_moe_weight_probe_v2 import (
    HttpRangeSafeTensorSource,
    LocalSafeTensorSource,
    build_default_remote_url,
    discover_layer_keys,
)

EPS = 1e-12


@dataclass
class ExpertWeights:
    expert_id: int
    gate: np.ndarray
    up: np.ndarray
    down: np.ndarray


@dataclass
class Assignment:
    permutation: np.ndarray
    signs: np.ndarray
    mean_score_train: float
    mean_score_holdout: float
    mean_gate_cos_holdout: float
    mean_up_cos_holdout: float
    mean_down_cos_holdout: float


@dataclass
class Spectrum:
    label: str
    projection: str
    experts: int
    common_energy_fraction: float
    stable_rank: float
    participation_ratio: float
    rank90: int
    rank95: int
    rank99: int
    explained_k1: float
    explained_k2: float
    explained_k4: float
    explained_k8: float
    mean_pairwise_cosine: float
    median_pairwise_cosine: float


@dataclass
class AlignmentRun:
    subset_name: str
    expert_ids: list[int]
    start_expert: int
    iteration: int
    score_profile: str
    mean_train_score: float
    mean_holdout_score: float
    mean_gate_cos_holdout: float
    mean_up_cos_holdout: float
    mean_down_cos_holdout: float
    spectra: list[Spectrum]


def _row_normalize(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(norms, EPS)


def canonicalize_scale(expert: ExpertWeights) -> tuple[ExpertWeights, dict[str, float]]:
    """Use the exact U/D reciprocal scale symmetry to equalize paired norms."""
    u = np.asarray(expert.up, dtype=np.float32)
    d = np.asarray(expert.down, dtype=np.float32)
    norm_u = np.linalg.norm(u, axis=1).astype(np.float64)
    norm_d = np.linalg.norm(d, axis=0).astype(np.float64)
    scale = np.sqrt((norm_d + EPS) / (norm_u + EPS))
    scale32 = scale.astype(np.float32)
    up = u * scale32[:, None]
    down = d / scale32[None, :]
    log_scale = np.log10(np.maximum(scale, EPS))
    stats = {
        "log10_scale_min": float(np.min(log_scale)),
        "log10_scale_p01": float(np.percentile(log_scale, 1)),
        "log10_scale_p50": float(np.percentile(log_scale, 50)),
        "log10_scale_p99": float(np.percentile(log_scale, 99)),
        "log10_scale_max": float(np.max(log_scale)),
        "median_norm_ratio_after": float(np.median(
            np.linalg.norm(up, axis=1) / np.maximum(np.linalg.norm(down, axis=0), EPS)
        )),
    }
    return ExpertWeights(expert.expert_id, expert.gate, up, down), stats


def choose_disjoint_indices(total: int, train_n: int, holdout_n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    if train_n + holdout_n > total:
        raise ValueError("signature coordinate request exceeds available dimensions")
    rng = np.random.default_rng(seed)
    indices = rng.choice(total, size=train_n + holdout_n, replace=False)
    return np.sort(indices[:train_n]), np.sort(indices[train_n:])


def component_signatures(
    expert: ExpertWeights,
    input_cols: np.ndarray,
    output_rows: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    g = _row_normalize(np.asarray(expert.gate[:, input_cols], dtype=np.float32))
    u = _row_normalize(np.asarray(expert.up[:, input_cols], dtype=np.float32))
    d = _row_normalize(np.asarray(expert.down[output_rows, :].T, dtype=np.float32))
    return g, u, d


def pairwise_score_matrices(
    reference_sig: tuple[np.ndarray, np.ndarray, np.ndarray],
    expert_sig: tuple[np.ndarray, np.ndarray, np.ndarray],
    weights: tuple[float, float, float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rg, ru, rd = reference_sig
    eg, eu, ed = expert_sig
    sg = rg @ eg.T
    su = ru @ eu.T
    sd = rd @ ed.T
    wg, wu, wd = weights
    ud_signed = wu * su + wd * sd
    total = wg * sg + np.abs(ud_signed)
    return total, sg, su, sd


def solve_assignment(
    reference_train: tuple[np.ndarray, np.ndarray, np.ndarray],
    expert_train: tuple[np.ndarray, np.ndarray, np.ndarray],
    reference_holdout: tuple[np.ndarray, np.ndarray, np.ndarray],
    expert_holdout: tuple[np.ndarray, np.ndarray, np.ndarray],
    weights: tuple[float, float, float],
) -> Assignment:
    total, _, su, sd = pairwise_score_matrices(reference_train, expert_train, weights)
    rows, cols = linear_sum_assignment(total, maximize=True)
    if not np.array_equal(rows, np.arange(total.shape[0])):
        order = np.argsort(rows)
        rows, cols = rows[order], cols[order]
    permutation = cols.astype(np.int32, copy=False)
    _, hsg, hsu, hsd = pairwise_score_matrices(reference_holdout, expert_holdout, weights)

    wg, wu, wd = weights
    train_ud = wu * su[rows, cols] + wd * sd[rows, cols]
    signs = np.where(train_ud >= 0.0, 1.0, -1.0).astype(np.float32)

    gate_h = hsg[rows, cols]
    up_h = signs * hsu[rows, cols]
    down_h = signs * hsd[rows, cols]
    total_h = wg * gate_h + wu * up_h + wd * down_h
    train_score = total[rows, cols]
    return Assignment(
        permutation=permutation,
        signs=signs,
        mean_score_train=float(np.mean(train_score)),
        mean_score_holdout=float(np.mean(total_h)),
        mean_gate_cos_holdout=float(np.mean(gate_h)),
        mean_up_cos_holdout=float(np.mean(up_h)),
        mean_down_cos_holdout=float(np.mean(down_h)),
    )


def apply_assignment(expert: ExpertWeights, assignment: Assignment) -> ExpertWeights:
    p = assignment.permutation
    s = assignment.signs
    gate = np.ascontiguousarray(expert.gate[p, :])
    up = np.ascontiguousarray(expert.up[p, :] * s[:, None])
    down = np.ascontiguousarray(expert.down[:, p] * s[None, :])
    return ExpertWeights(expert.expert_id, gate, up, down)


def mean_reference_signature(
    aligned_signatures: Sequence[tuple[np.ndarray, np.ndarray, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    components: list[np.ndarray] = []
    for component_index in range(3):
        mean = np.mean(
            np.stack([sig[component_index] for sig in aligned_signatures], axis=0),
            axis=0,
            dtype=np.float64,
        ).astype(np.float32)
        components.append(_row_normalize(mean))
    return components[0], components[1], components[2]


def choose_eval_flat_indices(
    rows: int,
    cols: int,
    n: int,
    seed: int,
    *,
    excluded_columns: np.ndarray | None = None,
    excluded_rows: np.ndarray | None = None,
) -> np.ndarray:
    total = rows * cols
    rng = np.random.default_rng(seed)
    result: set[int] = set()
    excluded_c = set(map(int, excluded_columns)) if excluded_columns is not None else set()
    excluded_r = set(map(int, excluded_rows)) if excluded_rows is not None else set()
    while len(result) < min(n, total):
        batch = rng.integers(0, total, size=max(4096, 2 * (n - len(result))), endpoint=False)
        for flat in batch:
            r = int(flat) // cols
            c = int(flat) % cols
            if c in excluded_c or r in excluded_r:
                continue
            result.add(int(flat))
            if len(result) >= n:
                break
    return np.array(sorted(result), dtype=np.int64)


def _rank_for(eigenvalues: np.ndarray, fraction: float) -> int:
    total = float(eigenvalues.sum())
    if total <= 0:
        return 0
    return int(np.searchsorted(np.cumsum(eigenvalues), fraction * total, side="left") + 1)


def spectrum_from_matrix(label: str, projection: str, x: np.ndarray) -> Spectrum:
    x64 = np.asarray(x, dtype=np.float64)
    experts = x64.shape[0]
    original_energy = float(np.sum(x64 * x64))
    mean = np.mean(x64, axis=0)
    centered = x64 - mean
    gram = centered @ centered.T
    gram = (gram + gram.T) * 0.5
    eigenvalues = np.linalg.eigvalsh(gram)[::-1]
    eigenvalues = np.maximum(eigenvalues[: experts - 1], 0.0)
    total = float(eigenvalues.sum())
    common_energy = experts * float(np.dot(mean, mean))
    stable = total / float(eigenvalues[0]) if eigenvalues.size and eigenvalues[0] > 0 else 0.0
    sq = float(np.dot(eigenvalues, eigenvalues))
    participation = total * total / sq if sq > 0 else 0.0

    def explained(k: int) -> float:
        return float(eigenvalues[: min(k, len(eigenvalues))].sum() / total) if total > 0 else 1.0

    norms = np.linalg.norm(x64, axis=1)
    normed = x64 / np.maximum(norms[:, None], EPS)
    cosine = normed @ normed.T
    upper = cosine[np.triu_indices(experts, k=1)]
    return Spectrum(
        label=label,
        projection=projection,
        experts=experts,
        common_energy_fraction=common_energy / original_energy if original_energy > 0 else 0.0,
        stable_rank=stable,
        participation_ratio=participation,
        rank90=_rank_for(eigenvalues, 0.90),
        rank95=_rank_for(eigenvalues, 0.95),
        rank99=_rank_for(eigenvalues, 0.99),
        explained_k1=explained(1),
        explained_k2=explained(2),
        explained_k4=explained(4),
        explained_k8=explained(8),
        mean_pairwise_cosine=float(np.mean(upper)),
        median_pairwise_cosine=float(np.median(upper)),
    )


def sample_projection(experts: Sequence[ExpertWeights], projection: str, flat_indices: np.ndarray) -> np.ndarray:
    rows: list[np.ndarray] = []
    for expert in experts:
        matrix = getattr(expert, projection)
        rows.append(np.asarray(matrix.reshape(-1)[flat_indices], dtype=np.float32))
    return np.stack(rows, axis=0)


def evaluate_conditions(
    raw: Sequence[ExpertWeights],
    scaled: Sequence[ExpertWeights],
    raw_aligned: Sequence[ExpertWeights] | None,
    scaled_aligned: Sequence[ExpertWeights] | None,
    eval_indices: dict[str, np.ndarray],
) -> list[Spectrum]:
    conditions: list[tuple[str, Sequence[ExpertWeights]]] = [("raw", raw), ("scale", scaled)]
    if raw_aligned is not None:
        conditions.append(("perm", raw_aligned))
    if scaled_aligned is not None:
        conditions.append(("scale+perm", scaled_aligned))

    result: list[Spectrum] = []
    for label, collection in conditions:
        blocks: list[np.ndarray] = []
        for projection in ("gate", "up", "down"):
            x = sample_projection(collection, projection, eval_indices[projection])
            result.append(spectrum_from_matrix(label, projection, x))
            block_norm = float(np.linalg.norm(x))
            blocks.append(x / max(block_norm, EPS))
        joint = np.concatenate(blocks, axis=1)
        result.append(spectrum_from_matrix(label, "joint", joint))
    return result


def load_experts(source: Any, discovered: dict[str, list[tuple[int, str]]], expert_ids: Sequence[int]) -> list[ExpertWeights]:
    key_maps = {p: dict(entries) for p, entries in discovered.items()}
    result: list[ExpertWeights] = []
    for position, expert_id in enumerate(expert_ids, start=1):
        gate = source.get_tensor(key_maps["gate"][expert_id]).to(torch.float32).cpu().numpy().copy()
        up = source.get_tensor(key_maps["up"][expert_id]).to(torch.float32).cpu().numpy().copy()
        down = source.get_tensor(key_maps["down"][expert_id]).to(torch.float32).cpu().numpy().copy()
        result.append(ExpertWeights(int(expert_id), gate, up, down))
        print(f"loaded expert {expert_id} ({position}/{len(expert_ids)})", flush=True)
    return result


def run_alignment(
    subset_name: str,
    raw: Sequence[ExpertWeights],
    scaled: Sequence[ExpertWeights],
    start_index: int,
    train_coords: tuple[np.ndarray, np.ndarray],
    holdout_coords: tuple[np.ndarray, np.ndarray],
    eval_indices: dict[str, np.ndarray],
    weights: tuple[float, float, float],
    score_profile: str,
    iterations: int = 2,
) -> AlignmentRun:
    train_sigs = [component_signatures(e, train_coords[0], train_coords[1]) for e in scaled]
    hold_sigs = [component_signatures(e, holdout_coords[0], holdout_coords[1]) for e in scaled]
    reference_train = train_sigs[start_index]
    reference_hold = hold_sigs[start_index]
    assignments: list[Assignment] = []

    for iteration in range(1, iterations + 1):
        assignments = [
            solve_assignment(reference_train, ts, reference_hold, hs, weights)
            for ts, hs in zip(train_sigs, hold_sigs, strict=True)
        ]
        aligned_train: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        aligned_hold: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        for ts, hs, assignment in zip(train_sigs, hold_sigs, assignments, strict=True):
            p, s = assignment.permutation, assignment.signs
            aligned_train.append((ts[0][p], ts[1][p] * s[:, None], ts[2][p] * s[:, None]))
            aligned_hold.append((hs[0][p], hs[1][p] * s[:, None], hs[2][p] * s[:, None]))
        reference_train = mean_reference_signature(aligned_train)
        reference_hold = mean_reference_signature(aligned_hold)
        print(
            f"{subset_name} start={raw[start_index].expert_id} iteration={iteration}: "
            f"train={np.mean([a.mean_score_train for a in assignments]):.4f} "
            f"holdout={np.mean([a.mean_score_holdout for a in assignments]):.4f}",
            flush=True,
        )

    raw_aligned = [apply_assignment(e, a) for e, a in zip(raw, assignments, strict=True)]
    scaled_aligned = [apply_assignment(e, a) for e, a in zip(scaled, assignments, strict=True)]
    spectra = evaluate_conditions(raw, scaled, raw_aligned, scaled_aligned, eval_indices)
    return AlignmentRun(
        subset_name=subset_name,
        expert_ids=[e.expert_id for e in raw],
        start_expert=raw[start_index].expert_id,
        iteration=iterations,
        score_profile=score_profile,
        mean_train_score=float(np.mean([a.mean_score_train for a in assignments])),
        mean_holdout_score=float(np.mean([a.mean_score_holdout for a in assignments])),
        mean_gate_cos_holdout=float(np.mean([a.mean_gate_cos_holdout for a in assignments])),
        mean_up_cos_holdout=float(np.mean([a.mean_up_cos_holdout for a in assignments])),
        mean_down_cos_holdout=float(np.mean([a.mean_down_cos_holdout for a in assignments])),
        spectra=spectra,
    )


def metric(run: AlignmentRun, label: str, projection: str) -> Spectrum:
    for item in run.spectra:
        if item.label == label and item.projection == projection:
            return item
    raise KeyError((label, projection))


def pilot_gate(runs: Sequence[AlignmentRun]) -> dict[str, Any]:
    per_run: list[dict[str, Any]] = []
    for run in runs:
        raw_joint = metric(run, "scale", "joint")
        aligned_joint = metric(run, "scale+perm", "joint")
        gain_k4 = aligned_joint.explained_k4 - raw_joint.explained_k4
        gain_k8 = aligned_joint.explained_k8 - raw_joint.explained_k8
        per_run.append({
            "subset": run.subset_name,
            "start_expert": run.start_expert,
            "raw_joint_k4": raw_joint.explained_k4,
            "aligned_joint_k4": aligned_joint.explained_k4,
            "gain_k4": gain_k4,
            "gain_k8": gain_k8,
            "aligned_joint_rank95": aligned_joint.rank95,
            "holdout_score": run.mean_holdout_score,
        })
    subset_passes: dict[str, bool] = {}
    for subset in sorted({r.subset_name for r in runs}):
        candidates = [x for x in per_run if x["subset"] == subset]
        subset_passes[subset] = any(
            x["gain_k4"] >= 0.12 and x["aligned_joint_rank95"] <= 11
            for x in candidates
        )
    passed = bool(subset_passes) and all(subset_passes.values())
    return {"passed": passed, "subset_passes": subset_passes, "runs": per_run}


def write_outputs(output_dir: Path, payload: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "alignment_probe.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    runs: list[dict[str, Any]] = payload["runs"]
    rows: list[dict[str, Any]] = []
    for run in runs:
        for spectrum in run["spectra"]:
            rows.append({
                "subset": run["subset_name"],
                "start_expert": run["start_expert"],
                "score_profile": run["score_profile"],
                **spectrum,
            })
    with (output_dir / "spectra.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    gate = payload["pilot_gate"]
    lines = [
        "# Test 0.5 — scale and permutation alignment",
        "",
        f"**Pilot continuation gate:** **{'PASS' if gate['passed'] else 'FAIL'}**",
        "",
        "The matching coordinates and evaluation coordinates are disjoint.",
        "",
        "| Subset | Start expert | Joint K=4 before | Joint K=4 after | Gain | Rank95 after | Holdout score |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for item in gate["runs"]:
        lines.append(
            f"| {item['subset']} | {item['start_expert']} | "
            f"{item['raw_joint_k4']:.2%} | {item['aligned_joint_k4']:.2%} | "
            f"{item['gain_k4']:+.2%} | {item['aligned_joint_rank95']} | "
            f"{item['holdout_score']:.4f} |"
        )
    lines += [
        "",
        "Gate rule: each independent subset needs at least one start with a joint K=4 gain of 12 percentage points and aligned rank95 at most 11.",
        "",
        "A PASS authorizes all-64 alignment. A FAIL redirects the project toward permutation-invariant neuron dictionaries, clustering, or activation-aware factorization.",
    ]
    (output_dir / "VERDICT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def self_test() -> None:
    rng = np.random.default_rng(7)
    e, m, d = 8, 32, 64
    base_g = rng.normal(size=(m, d)).astype(np.float32)
    base_u = rng.normal(size=(m, d)).astype(np.float32)
    base_d = rng.normal(size=(d, m)).astype(np.float32)
    modes = []
    for _ in range(3):
        modes.append((
            rng.normal(scale=0.08, size=(m, d)).astype(np.float32),
            rng.normal(scale=0.08, size=(m, d)).astype(np.float32),
            rng.normal(scale=0.08, size=(d, m)).astype(np.float32),
        ))
    experts: list[ExpertWeights] = []
    for expert_id in range(e):
        coeff = rng.normal(size=3)
        g = base_g.copy(); u = base_u.copy(); down = base_d.copy()
        for a, (mg, mu, md) in zip(coeff, modes, strict=True):
            g += a * mg; u += a * mu; down += a * md
        p = rng.permutation(m)
        scales = np.exp(rng.normal(scale=0.7, size=m)).astype(np.float32)
        signs = rng.choice(np.array([-1.0, 1.0], dtype=np.float32), size=m)
        g = g[p]
        u = u[p] * (scales * signs)[:, None]
        down = down[:, p] / scales[None, :] * signs[None, :]
        experts.append(ExpertWeights(expert_id, g, u, down))

    scaled = [canonicalize_scale(x)[0] for x in experts]
    train_in, hold_in = choose_disjoint_indices(d, 24, 16, 1)
    train_out, hold_out = choose_disjoint_indices(d, 24, 16, 2)
    eval_indices = {
        "gate": choose_eval_flat_indices(m, d, 512, 11, excluded_columns=np.r_[train_in, hold_in]),
        "up": choose_eval_flat_indices(m, d, 512, 12, excluded_columns=np.r_[train_in, hold_in]),
        "down": choose_eval_flat_indices(d, m, 512, 13, excluded_rows=np.r_[train_out, hold_out]),
    }
    run = run_alignment(
        "synthetic", experts, scaled, 0,
        (train_in, train_out), (hold_in, hold_out), eval_indices,
        (1.0, 0.5, 0.5), "balanced", iterations=2,
    )
    before = metric(run, "scale", "joint").explained_k4
    after = metric(run, "scale+perm", "joint").explained_k4
    if after < 0.95 or after <= before + 0.20:
        raise AssertionError(f"synthetic alignment failed: before={before}, after={after}")

    x = rng.normal(size=(d, 7)).astype(np.float32)
    original = experts[0]
    canon = scaled[0]
    y0 = original.down @ ((1.0 / (1.0 + np.exp(-(original.gate @ x)))) * (original.gate @ x) * (original.up @ x))
    y1 = canon.down @ ((1.0 / (1.0 + np.exp(-(canon.gate @ x)))) * (canon.gate @ x) * (canon.up @ x))
    rel = np.linalg.norm(y0 - y1) / max(np.linalg.norm(y0), EPS)
    if rel > 2e-5:
        raise AssertionError(f"scale canonicalization changed function: {rel}")
    print(f"self-test passed: joint K4 {before:.2%} -> {after:.2%}; invariance error={rel:.3e}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--model-id", default="allenai/OLMoE-1B-7B-0924")
    parser.add_argument("--revision", default="3a970199d0f87db4e3e57275abb93812bf10fd83")
    parser.add_argument("--shard-name", default="model-00002-of-00003.safetensors")
    parser.add_argument("--shard-path", type=Path)
    parser.add_argument("--remote-shard-url")
    parser.add_argument("--cache-dir", type=Path, default=Path(".cache/olmoe-ranges"))
    parser.add_argument("--layer", type=int, default=7)
    parser.add_argument("--subset-size", type=int, default=16)
    parser.add_argument("--subset-count", type=int, default=2)
    parser.add_argument("--subset-seed", type=int, default=20260726)
    parser.add_argument("--signature-dims", type=int, default=64)
    parser.add_argument("--eval-dims", type=int, default=131072)
    parser.add_argument("--iterations", type=int, default=2)
    parser.add_argument("--output-dir", type=Path, default=Path("results/alignment_probe"))
    args = parser.parse_args(argv)

    if args.self_test:
        self_test()
        return 0

    started = time.perf_counter()
    if args.shard_path:
        source = LocalSafeTensorSource(args.shard_path)
        source_url = None
    else:
        source_url = args.remote_shard_url or build_default_remote_url(args.model_id, args.revision, args.shard_name)
        source = HttpRangeSafeTensorSource(source_url, cache_dir=args.cache_dir)
    discovered = discover_layer_keys(source.keys(), args.layer)
    total_experts = len(discovered["gate"])
    rng = np.random.default_rng(args.subset_seed)
    shuffled = rng.permutation(total_experts)
    requested = args.subset_size * args.subset_count
    if requested > total_experts:
        raise ValueError("subset_size * subset_count exceeds expert count")
    subsets = [
        sorted(map(int, shuffled[i * args.subset_size:(i + 1) * args.subset_size]))
        for i in range(args.subset_count)
    ]

    key0 = dict(discovered["gate"])[subsets[0][0]]
    gate_shape = source.descriptor(key0).shape
    m, d = gate_shape
    train_in, hold_in = choose_disjoint_indices(d, args.signature_dims, args.signature_dims, args.subset_seed + 1)
    train_out, hold_out = choose_disjoint_indices(d, args.signature_dims, args.signature_dims, args.subset_seed + 2)
    excluded_in = np.r_[train_in, hold_in]
    excluded_out = np.r_[train_out, hold_out]
    eval_indices = {
        "gate": choose_eval_flat_indices(m, d, args.eval_dims, args.subset_seed + 11, excluded_columns=excluded_in),
        "up": choose_eval_flat_indices(m, d, args.eval_dims, args.subset_seed + 12, excluded_columns=excluded_in),
        "down": choose_eval_flat_indices(d, m, args.eval_dims, args.subset_seed + 13, excluded_rows=excluded_out),
    }

    runs: list[AlignmentRun] = []
    scale_stats: dict[str, list[dict[str, float]]] = {}
    for subset_idx, expert_ids in enumerate(subsets, start=1):
        subset_name = f"subset-{subset_idx}"
        print(f"\n=== {subset_name}: {expert_ids} ===", flush=True)
        raw = load_experts(source, discovered, expert_ids)
        scaled_with_stats = [canonicalize_scale(e) for e in raw]
        scaled = [x[0] for x in scaled_with_stats]
        scale_stats[subset_name] = [x[1] for x in scaled_with_stats]
        starts = [0, len(raw) // 2]
        for start_index in starts:
            runs.append(run_alignment(
                subset_name, raw, scaled, start_index,
                (train_in, train_out), (hold_in, hold_out), eval_indices,
                (1.0, 0.5, 0.5), "balanced-gate1-up0.5-down0.5", args.iterations,
            ))
        del raw, scaled, scaled_with_stats

    gate = pilot_gate(runs)
    payload = {
        "metadata": {
            "model_id": args.model_id,
            "revision": args.revision,
            "layer": args.layer,
            "total_experts": total_experts,
            "subsets": subsets,
            "signature_dims_per_component_train": args.signature_dims,
            "signature_dims_per_component_holdout": args.signature_dims,
            "eval_dims_per_projection": args.eval_dims,
            "iterations": args.iterations,
            "elapsed_seconds": time.perf_counter() - started,
            "source": source.metadata(),
            "matching_and_evaluation_are_disjoint": True,
        },
        "scale_stats": scale_stats,
        "pilot_gate": gate,
        "runs": [asdict(run) for run in runs],
    }
    write_outputs(args.output_dir, payload)
    print((args.output_dir / "VERDICT.md").read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
