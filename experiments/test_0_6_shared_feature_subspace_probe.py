#!/usr/bin/env python3
"""Shared feature-subspace diagnostic for a trained SwiGLU MoE layer.

The failed expert-axis PCA asked whether entire expert matrices were linear
combinations of a few global matrices. This probe tests a different and more
hardware-compatible hypothesis that does not depend on neuron alignment:

    G_e ≈ A^G_e B_in
    U_e ≈ A^U_e B_in
    D_e ≈ B_out C^D_e

``B_in`` is a shared input projection for gate and up. ``B_out`` is a shared
output projection for down. The expert-specific factors are much smaller and
the factorization can be executed directly without reconstructing weights.

Training and validation neuron samples are disjoint for every expert. The
basis is fit on sampled rows/columns while all 2,048 feature coordinates are
retained, so the test measures generalization to unseen intermediate neurons.
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import scipy.linalg
import torch

from modal_moe_weight_probe_v2 import (
    HttpRangeSafeTensorSource,
    LocalSafeTensorSource,
    build_default_remote_url,
    discover_layer_keys,
)

EPS = 1e-12


@dataclass
class BasisCurvePoint:
    profile: str
    evaluated_block: str
    rank: int
    train_energy_captured: float
    validation_energy_captured: float
    validation_mean_expert_capture: float
    validation_p05_expert_capture: float
    validation_p50_expert_capture: float
    validation_p95_expert_capture: float
    isotropic_random_expectation: float


@dataclass
class BasisSummary:
    profile: str
    evaluated_block: str
    train_vectors: int
    validation_vectors: int
    feature_dimension: int
    max_basis_rank: int
    train_rank90: int | None
    train_rank95: int | None
    train_rank99: int | None
    validation_rank80: int | None
    validation_rank90: int | None
    validation_rank95: int | None
    validation_capture_at_128: float
    validation_capture_at_256: float
    validation_capture_at_512: float
    validation_capture_at_768: float
    validation_capture_at_1024: float


@dataclass
class FactorBudget:
    input_rank: int
    output_rank: int
    parameter_ratio: float
    idealized_compute_ratio: float
    parameter_compression_factor: float
    idealized_compute_reduction: float


def deterministic_split(total: int, train_n: int, validation_n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    if train_n + validation_n > total:
        raise ValueError("requested rows exceed intermediate dimension")
    rng = np.random.default_rng(seed)
    ids = rng.choice(total, size=train_n + validation_n, replace=False)
    return np.sort(ids[:train_n]), np.sort(ids[train_n:])


def load_sampled_vectors(
    source: Any,
    discovered: dict[str, list[tuple[int, str]]],
    rows_per_expert: int,
    validation_rows_per_expert: int,
    seed: int,
) -> dict[str, np.ndarray]:
    key_maps = {projection: dict(entries) for projection, entries in discovered.items()}
    expert_ids = sorted(key_maps["gate"])
    gate_train: list[np.ndarray] = []
    gate_val: list[np.ndarray] = []
    up_train: list[np.ndarray] = []
    up_val: list[np.ndarray] = []
    down_train: list[np.ndarray] = []
    down_val: list[np.ndarray] = []
    owner_train: list[int] = []
    owner_val: list[int] = []

    for position, expert_id in enumerate(expert_ids, start=1):
        gate = source.get_tensor(key_maps["gate"][expert_id]).to(torch.float32).cpu().numpy()
        up = source.get_tensor(key_maps["up"][expert_id]).to(torch.float32).cpu().numpy()
        down = source.get_tensor(key_maps["down"][expert_id]).to(torch.float32).cpu().numpy()
        intermediate, hidden = gate.shape
        train_ids, val_ids = deterministic_split(
            intermediate,
            rows_per_expert,
            validation_rows_per_expert,
            seed + 1009 * expert_id,
        )
        gate_train.append(np.asarray(gate[train_ids, :], dtype=np.float32).copy())
        gate_val.append(np.asarray(gate[val_ids, :], dtype=np.float32).copy())
        up_train.append(np.asarray(up[train_ids, :], dtype=np.float32).copy())
        up_val.append(np.asarray(up[val_ids, :], dtype=np.float32).copy())
        down_train.append(np.asarray(down[:, train_ids].T, dtype=np.float32).copy())
        down_val.append(np.asarray(down[:, val_ids].T, dtype=np.float32).copy())
        owner_train.extend([expert_id] * rows_per_expert)
        owner_val.extend([expert_id] * validation_rows_per_expert)
        print(f"sampled expert {expert_id} ({position}/{len(expert_ids)})", flush=True)
        del gate, up, down

    return {
        "gate_train": np.concatenate(gate_train, axis=0),
        "gate_val": np.concatenate(gate_val, axis=0),
        "up_train": np.concatenate(up_train, axis=0),
        "up_val": np.concatenate(up_val, axis=0),
        "down_train": np.concatenate(down_train, axis=0),
        "down_val": np.concatenate(down_val, axis=0),
        "owner_train": np.asarray(owner_train, dtype=np.int32),
        "owner_val": np.asarray(owner_val, dtype=np.int32),
        "expert_ids": np.asarray(expert_ids, dtype=np.int32),
        "hidden": np.asarray([hidden], dtype=np.int32),
        "intermediate": np.asarray([intermediate], dtype=np.int32),
    }


def rank_at(cumulative: np.ndarray, threshold: float) -> int | None:
    hits = np.flatnonzero(cumulative >= threshold)
    return int(hits[0] + 1) if hits.size else None


def capture_at(cumulative: np.ndarray, rank: int) -> float:
    if cumulative.size == 0:
        return 0.0
    return float(cumulative[min(rank, cumulative.size) - 1]) if rank > 0 else 0.0


def fit_basis(train: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(train, dtype=np.float32, order="C")
    _, singular_values, vt = scipy.linalg.svd(
        x,
        full_matrices=False,
        overwrite_a=False,
        check_finite=False,
        lapack_driver="gesdd",
    )
    return singular_values.astype(np.float64, copy=False), vt.astype(np.float32, copy=False)


def validation_statistics(
    validation: np.ndarray,
    vt: np.ndarray,
    owners: np.ndarray,
) -> tuple[np.ndarray, dict[int, np.ndarray]]:
    x = np.asarray(validation, dtype=np.float32, order="C")
    coefficients = x @ vt.T
    squared = np.asarray(coefficients, dtype=np.float64) ** 2
    total_energy = float(np.sum(np.asarray(x, dtype=np.float64) ** 2))
    cumulative = np.cumsum(np.sum(squared, axis=0)) / max(total_energy, EPS)

    per_expert: dict[int, np.ndarray] = {}
    for expert_id in np.unique(owners):
        mask = owners == expert_id
        expert_energy = float(np.sum(np.asarray(x[mask], dtype=np.float64) ** 2))
        per_expert[int(expert_id)] = np.cumsum(np.sum(squared[mask], axis=0)) / max(expert_energy, EPS)
    return cumulative, per_expert


def analyze_profile(
    profile: str,
    train: np.ndarray,
    validation_blocks: dict[str, np.ndarray],
    validation_owners: dict[str, np.ndarray],
    ranks: Sequence[int],
) -> tuple[list[BasisCurvePoint], list[BasisSummary], np.ndarray]:
    print(f"\nFitting shared feature basis: {profile} on {train.shape}", flush=True)
    singular_values, vt = fit_basis(train)
    train_energy = singular_values * singular_values
    train_cumulative = np.cumsum(train_energy) / max(float(train_energy.sum()), EPS)
    points: list[BasisCurvePoint] = []
    summaries: list[BasisSummary] = []

    for block_name, validation in validation_blocks.items():
        val_cumulative, per_expert = validation_statistics(
            validation,
            vt,
            validation_owners[block_name],
        )
        for rank in ranks:
            effective = min(rank, vt.shape[0])
            expert_values = np.array(
                [capture_at(curve, effective) for curve in per_expert.values()],
                dtype=np.float64,
            )
            points.append(BasisCurvePoint(
                profile=profile,
                evaluated_block=block_name,
                rank=rank,
                train_energy_captured=capture_at(train_cumulative, effective),
                validation_energy_captured=capture_at(val_cumulative, effective),
                validation_mean_expert_capture=float(np.mean(expert_values)),
                validation_p05_expert_capture=float(np.percentile(expert_values, 5)),
                validation_p50_expert_capture=float(np.percentile(expert_values, 50)),
                validation_p95_expert_capture=float(np.percentile(expert_values, 95)),
                isotropic_random_expectation=min(rank, validation.shape[1]) / validation.shape[1],
            ))
        summaries.append(BasisSummary(
            profile=profile,
            evaluated_block=block_name,
            train_vectors=int(train.shape[0]),
            validation_vectors=int(validation.shape[0]),
            feature_dimension=int(validation.shape[1]),
            max_basis_rank=int(vt.shape[0]),
            train_rank90=rank_at(train_cumulative, 0.90),
            train_rank95=rank_at(train_cumulative, 0.95),
            train_rank99=rank_at(train_cumulative, 0.99),
            validation_rank80=rank_at(val_cumulative, 0.80),
            validation_rank90=rank_at(val_cumulative, 0.90),
            validation_rank95=rank_at(val_cumulative, 0.95),
            validation_capture_at_128=capture_at(val_cumulative, 128),
            validation_capture_at_256=capture_at(val_cumulative, 256),
            validation_capture_at_512=capture_at(val_cumulative, 512),
            validation_capture_at_768=capture_at(val_cumulative, 768),
            validation_capture_at_1024=capture_at(val_cumulative, 1024),
        ))
    return points, summaries, vt


def factor_budget(hidden: int, intermediate: int, experts: int, top_k: int, r_in: int, r_out: int) -> FactorBudget:
    original_parameters = 3 * experts * intermediate * hidden
    factor_parameters = (
        hidden * r_in
        + 2 * experts * intermediate * r_in
        + hidden * r_out
        + experts * intermediate * r_out
    )
    original_compute = 3 * top_k * intermediate * hidden
    factor_compute = (
        hidden * r_in
        + 2 * top_k * intermediate * r_in
        + hidden * r_out
        + top_k * intermediate * r_out
    )
    parameter_ratio = factor_parameters / original_parameters
    compute_ratio = factor_compute / original_compute
    return FactorBudget(
        input_rank=r_in,
        output_rank=r_out,
        parameter_ratio=parameter_ratio,
        idealized_compute_ratio=compute_ratio,
        parameter_compression_factor=1.0 / parameter_ratio,
        idealized_compute_reduction=1.0 - compute_ratio,
    )


def get_point(points: Sequence[BasisCurvePoint], profile: str, block: str, rank: int) -> BasisCurvePoint:
    for point in points:
        if point.profile == profile and point.evaluated_block == block and point.rank == rank:
            return point
    raise KeyError((profile, block, rank))


def decision(points: Sequence[BasisCurvePoint]) -> dict[str, Any]:
    evaluations: list[dict[str, Any]] = []
    for rank in (128, 256, 384, 512, 768, 1024):
        joint = get_point(points, "gate_up_joint", "gate_up_joint", rank)
        down = get_point(points, "down_output", "down", rank)
        worst_p05 = min(joint.validation_p05_expert_capture, down.validation_p05_expert_capture)
        evaluations.append({
            "rank": rank,
            "gate_up_validation_capture": joint.validation_energy_captured,
            "down_validation_capture": down.validation_energy_captured,
            "worst_p05_expert_capture": worst_p05,
        })
    pass_rank = next((x["rank"] for x in evaluations if
                      x["rank"] <= 512
                      and x["gate_up_validation_capture"] >= 0.90
                      and x["down_validation_capture"] >= 0.90
                      and x["worst_p05_expert_capture"] >= 0.80), None)
    borderline_rank = next((x["rank"] for x in evaluations if
                            x["rank"] <= 1024
                            and x["gate_up_validation_capture"] >= 0.90
                            and x["down_validation_capture"] >= 0.90), None)
    verdict = "PASS" if pass_rank is not None else ("BORDERLINE" if borderline_rank is not None else "FAIL")
    return {
        "verdict": verdict,
        "pass_rank": pass_rank,
        "borderline_rank": borderline_rank,
        "evaluations": evaluations,
    }


def write_outputs(output_dir: Path, payload: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "shared_feature_subspace.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    points = payload["curve_points"]
    with (output_dir / "curve.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(points[0].keys()))
        writer.writeheader()
        writer.writerows(points)
    summaries = payload["summaries"]
    with (output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summaries[0].keys()))
        writer.writeheader()
        writer.writerows(summaries)

    verdict = payload["decision"]
    budgets = {item["input_rank"]: item for item in payload["budgets"] if item["input_rank"] == item["output_rank"]}
    lines = [
        "# Test 0.6 — shared feature subspaces",
        "",
        f"**Decision:** **{verdict['verdict']}**",
        "",
        "This tests a shared input basis for `gate`+`up` and a shared output basis for `down`; it does not require one-to-one neuron alignment.",
        "",
        "| Rank | Gate+up validation capture | Down validation capture | Worst expert p05 | Parameter ratio | Ideal compute ratio |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for item in verdict["evaluations"]:
        budget = budgets[item["rank"]]
        lines.append(
            f"| {item['rank']} | {item['gate_up_validation_capture']:.2%} | "
            f"{item['down_validation_capture']:.2%} | {item['worst_p05_expert_capture']:.2%} | "
            f"{budget['parameter_ratio']:.2%} | {budget['idealized_compute_ratio']:.2%} |"
        )
    lines += [
        "",
        "PASS requires at least 90% held-out weight energy in both branches by rank ≤512, with p05 expert capture ≥80%.",
        "BORDERLINE permits rank ≤1024 and authorizes an activation-aware test, but not an acceleration claim.",
    ]
    (output_dir / "VERDICT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def self_test() -> None:
    rng = np.random.default_rng(19)
    experts, intermediate, hidden, true_rank = 8, 48, 128, 24
    q_in, _ = np.linalg.qr(rng.normal(size=(hidden, true_rank)))
    b_in = q_in.T.astype(np.float32)
    q_out, _ = np.linalg.qr(rng.normal(size=(hidden, true_rank)))
    b_out = q_out.astype(np.float32)
    gate_train: list[np.ndarray] = []
    gate_val: list[np.ndarray] = []
    up_train: list[np.ndarray] = []
    up_val: list[np.ndarray] = []
    down_train: list[np.ndarray] = []
    down_val: list[np.ndarray] = []
    owners: list[int] = []
    for e in range(experts):
        ag = rng.normal(size=(intermediate, true_rank)).astype(np.float32)
        au = rng.normal(size=(intermediate, true_rank)).astype(np.float32)
        cd = rng.normal(size=(true_rank, intermediate)).astype(np.float32)
        g = ag @ b_in
        u = au @ b_in
        d = b_out @ cd
        gate_train.append(g[:24]); gate_val.append(g[24:])
        up_train.append(u[:24]); up_val.append(u[24:])
        down_train.append(d[:, :24].T); down_val.append(d[:, 24:].T)
        owners.extend([e] * 24)
    owner = np.asarray(owners, dtype=np.int32)
    points, _, _ = analyze_profile(
        "gate_up_joint",
        np.vstack([np.concatenate(gate_train), np.concatenate(up_train)]),
        {"gate_up_joint": np.vstack([np.concatenate(gate_val), np.concatenate(up_val)])},
        {"gate_up_joint": np.concatenate([owner, owner])},
        [true_rank, 32],
    )
    p = get_point(points, "gate_up_joint", "gate_up_joint", true_rank)
    if p.validation_energy_captured < 0.9999:
        raise AssertionError(p)

    x = rng.normal(size=hidden).astype(np.float32)
    ag = rng.normal(size=(intermediate, true_rank)).astype(np.float32)
    w = ag @ b_in
    direct = w @ x
    factored = ag @ (b_in @ x)
    if np.linalg.norm(direct - factored) / max(np.linalg.norm(direct), EPS) > 1e-5:
        raise AssertionError("input factor execution mismatch")
    z = [rng.normal(size=intermediate).astype(np.float32) for _ in range(3)]
    c = [rng.normal(size=(true_rank, intermediate)).astype(np.float32) for _ in range(3)]
    pi = rng.random(3).astype(np.float32); pi /= pi.sum()
    direct_down = sum(float(pi[i]) * (b_out @ c[i]) @ z[i] for i in range(3))
    fused_down = b_out @ sum(float(pi[i]) * (c[i] @ z[i]) for i in range(3))
    if np.linalg.norm(direct_down - fused_down) / max(np.linalg.norm(direct_down), EPS) > 1e-5:
        raise AssertionError("down aggregation mismatch")
    print(f"self-test passed: true rank {true_rank}, held-out capture={p.validation_energy_captured:.6%}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--model-id", default="allenai/OLMoE-1B-7B-0924")
    parser.add_argument("--revision", default="3a970199d0f87db4e3e57275abb93812bf10fd83")
    parser.add_argument("--shard-name", default="model-00002-of-00003.safetensors")
    parser.add_argument("--shard-path", type=Path)
    parser.add_argument("--remote-shard-url")
    parser.add_argument("--cache-dir", type=Path, default=Path(".cache/olmoe-feature-ranges"))
    parser.add_argument("--layer", type=int, default=7)
    parser.add_argument("--rows-per-expert", type=int, default=16)
    parser.add_argument("--validation-rows-per-expert", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260727)
    parser.add_argument("--ranks", default="32,64,128,256,384,512,768,1024")
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--output-dir", type=Path, default=Path("results/shared_feature_subspace"))
    args = parser.parse_args(argv)

    if args.self_test:
        self_test()
        return 0

    ranks = sorted({int(x) for x in args.ranks.split(",") if x.strip()})
    started = time.perf_counter()
    if args.shard_path:
        source = LocalSafeTensorSource(args.shard_path)
    else:
        url = args.remote_shard_url or build_default_remote_url(args.model_id, args.revision, args.shard_name)
        source = HttpRangeSafeTensorSource(url, cache_dir=args.cache_dir)
    discovered = discover_layer_keys(source.keys(), args.layer)
    data = load_sampled_vectors(
        source,
        discovered,
        args.rows_per_expert,
        args.validation_rows_per_expert,
        args.seed,
    )
    owners = data["owner_val"]
    all_points: list[BasisCurvePoint] = []
    all_summaries: list[BasisSummary] = []

    profiles = [
        (
            "gate_input",
            data["gate_train"],
            {"gate": data["gate_val"]},
            {"gate": owners},
        ),
        (
            "up_input",
            data["up_train"],
            {"up": data["up_val"]},
            {"up": owners},
        ),
        (
            "gate_up_joint",
            np.vstack([data["gate_train"], data["up_train"]]),
            {
                "gate": data["gate_val"],
                "up": data["up_val"],
                "gate_up_joint": np.vstack([data["gate_val"], data["up_val"]]),
            },
            {
                "gate": owners,
                "up": owners,
                "gate_up_joint": np.concatenate([owners, owners]),
            },
        ),
        (
            "down_output",
            data["down_train"],
            {"down": data["down_val"]},
            {"down": owners},
        ),
    ]
    for profile, train, blocks, block_owners in profiles:
        points, summaries, _ = analyze_profile(profile, train, blocks, block_owners, ranks)
        all_points.extend(points)
        all_summaries.extend(summaries)

    hidden = int(data["hidden"][0])
    intermediate = int(data["intermediate"][0])
    expert_count = int(len(data["expert_ids"]))
    budget_ranks = sorted(set(ranks) | {128, 256, 384, 512, 768, 1024})
    budgets = [factor_budget(hidden, intermediate, expert_count, args.top_k, r, r) for r in budget_ranks]
    payload = {
        "metadata": {
            "model_id": args.model_id,
            "revision": args.revision,
            "layer": args.layer,
            "experts": expert_count,
            "hidden": hidden,
            "intermediate": intermediate,
            "top_k": args.top_k,
            "rows_per_expert": args.rows_per_expert,
            "validation_rows_per_expert": args.validation_rows_per_expert,
            "seed": args.seed,
            "elapsed_seconds": time.perf_counter() - started,
            "source": source.metadata(),
            "training_and_validation_neurons_are_disjoint": True,
        },
        "decision": decision(all_points),
        "budgets": [asdict(x) for x in budgets],
        "summaries": [asdict(x) for x in all_summaries],
        "curve_points": [asdict(x) for x in all_points],
    }
    write_outputs(args.output_dir, payload)
    print((args.output_dir / "VERDICT.md").read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
