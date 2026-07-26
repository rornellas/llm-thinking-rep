#!/usr/bin/env python3
"""Analyze routed gate/up/nonlinear/down activations from one OLMoE layer.

The raw weights are geometrically high-rank. This probe asks a different
question: on the tokens actually routed to an expert, do its projection outputs
live in a small subspace that generalizes to held-out tokens?

An orthogonal output basis A implies the directly executable factorization
W x ~= A (A^T W x), so projecting a held-out true output is exactly the oracle
error of the corresponding activation-aware low-rank matrix factorization.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

EPS = 1e-12


@dataclass
class CurvePoint:
    scope: str
    component: str
    expert: int | None
    rank: int
    train_samples: int
    validation_samples: int
    validation_capture: float
    validation_relative_error: float
    validation_mean_sample_capture: float
    validation_p05_sample_capture: float
    nonlinear_z_relative_error: float | None
    aggregate_moe_relative_error: float | None
    idealized_parameter_ratio: float | None
    idealized_compute_ratio: float | None


@dataclass
class ExpertSummary:
    expert: int
    assignments: int
    train_assignments: int
    validation_assignments: int
    primary_route_fraction: float
    route_weight_mean: float
    route_weight_p50: float
    route_weight_p95: float
    gate_up_rank90_train: int | None
    down_rank90_train: int | None
    gate_up_capture_r16: float
    gate_up_capture_r32: float
    gate_up_capture_r64: float
    down_capture_r16: float
    down_capture_r32: float
    down_capture_r64: float
    z_error_r16: float
    z_error_r32: float
    z_error_r64: float


def safe_silu(x: np.ndarray) -> np.ndarray:
    x64 = np.asarray(x, dtype=np.float64)
    return (x64 / (1.0 + np.exp(-np.clip(x64, -60.0, 60.0)))).astype(np.float32)


def load_chunks(meta_path: Path) -> tuple[str, np.ndarray]:
    records = [json.loads(line) for line in meta_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not records:
        raise ValueError(f"empty metadata: {meta_path}")
    raw = np.fromfile(meta_path.with_suffix(".f32"), dtype=np.float32)
    offset = 0
    chunks: list[np.ndarray] = []
    for record in records:
        ne = [int(x) for x in record["ne"]]
        elements = int(record["elements"])
        chunk = raw[offset:offset + elements]
        offset += elements
        if chunk.size != elements:
            raise ValueError(f"truncated data for {record['name']}")
        shaped = chunk.reshape(ne[3], ne[2], ne[1], ne[0])
        if ne[3] == 1:
            shaped = shaped[0]
        if shaped.ndim >= 2 and shaped.shape[-1] == 1:
            shaped = shaped[..., 0]
        chunks.append(np.asarray(shaped))
    if offset != raw.size:
        raise ValueError(f"unused data values: {raw.size - offset}")
    return records[0]["name"], np.concatenate(chunks, axis=0)


def find_tensor(captured: dict[str, np.ndarray], needle: str) -> np.ndarray:
    matches = [(name, array) for name, array in captured.items() if needle in name]
    if len(matches) != 1:
        raise KeyError(f"expected one tensor containing {needle!r}, found {[m[0] for m in matches]}")
    return matches[0][1]


def split_indices(count: int, seed: int, train_fraction: float = 0.67) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    order = rng.permutation(count)
    train_count = max(1, int(round(count * train_fraction)))
    if count > 1:
        train_count = min(train_count, count - 1)
    return np.sort(order[:train_count]), np.sort(order[train_count:])


def fit_basis(train: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(train, dtype=np.float32)
    _, singular, vt = np.linalg.svd(x, full_matrices=False)
    energy = singular.astype(np.float64) ** 2
    curve = np.cumsum(energy) / max(float(energy.sum()), EPS)
    return vt.astype(np.float32, copy=False), curve


def rank_at(curve: np.ndarray, threshold: float) -> int | None:
    hits = np.flatnonzero(curve >= threshold)
    return int(hits[0] + 1) if hits.size else None


def project(x: np.ndarray, vt: np.ndarray, rank: int) -> np.ndarray:
    effective = min(rank, vt.shape[0])
    basis = vt[:effective]
    return (np.asarray(x, dtype=np.float32) @ basis.T) @ basis


def capture_metrics(x: np.ndarray, reconstructed: np.ndarray) -> tuple[float, float, float, float]:
    x64 = np.asarray(x, dtype=np.float64)
    r64 = np.asarray(reconstructed, dtype=np.float64)
    total = np.sum(x64 * x64, axis=1)
    residual = np.sum((x64 - r64) ** 2, axis=1)
    sample_capture = 1.0 - residual / np.maximum(total, EPS)
    global_capture = 1.0 - float(np.sum(residual)) / max(float(np.sum(total)), EPS)
    relative_error = math.sqrt(max(0.0, 1.0 - global_capture))
    return global_capture, relative_error, float(np.mean(sample_capture)), float(np.percentile(sample_capture, 5))


def factor_budget(rank: int, d: int = 2048, m: int = 1024) -> tuple[float, float]:
    return rank * (d + 2 * m) / (2 * d * m), rank * (d + 2 * m) / (2 * d * m)


def down_budget(rank: int, d: int = 2048, m: int = 1024) -> tuple[float, float]:
    return rank * (d + m) / (d * m), rank * (d + m) / (d * m)


def validate_capture(
    gate: np.ndarray,
    up: np.ndarray,
    z: np.ndarray,
    down: np.ndarray,
    weights: np.ndarray,
    moe_out: np.ndarray,
) -> dict[str, float]:
    z_from_gu = safe_silu(gate) * up
    z_error = float(np.linalg.norm(z_from_gu.astype(np.float64) - z.astype(np.float64)) / max(np.linalg.norm(z.astype(np.float64)), EPS))
    aggregate = np.sum(down * weights[..., None], axis=1)
    aggregate_error = float(np.linalg.norm(aggregate.astype(np.float64) - moe_out.astype(np.float64)) / max(np.linalg.norm(moe_out.astype(np.float64)), EPS))
    return {"swiglu_consistency_error": z_error, "weighted_down_consistency_error": aggregate_error}


def global_analysis(
    component: str,
    train: np.ndarray,
    validation: np.ndarray,
    ranks: Sequence[int],
    d: int,
    m: int,
) -> tuple[list[CurvePoint], np.ndarray]:
    vt, _ = fit_basis(train)
    points: list[CurvePoint] = []
    for rank in ranks:
        reconstructed = project(validation, vt, rank)
        capture, error, mean_capture, p05 = capture_metrics(validation, reconstructed)
        if component == "gate_up":
            parameter_ratio, compute_ratio = factor_budget(rank, d, m)
        elif component == "down":
            parameter_ratio, compute_ratio = down_budget(rank, d, m)
        else:
            parameter_ratio = compute_ratio = None
        points.append(CurvePoint(
            scope="global", component=component, expert=None, rank=rank,
            train_samples=len(train), validation_samples=len(validation),
            validation_capture=capture, validation_relative_error=error,
            validation_mean_sample_capture=mean_capture, validation_p05_sample_capture=p05,
            nonlinear_z_relative_error=None, aggregate_moe_relative_error=None,
            idealized_parameter_ratio=parameter_ratio, idealized_compute_ratio=compute_ratio,
        ))
    return points, vt


def analyze(
    x: np.ndarray,
    topk: np.ndarray,
    weights: np.ndarray,
    gate: np.ndarray,
    up: np.ndarray,
    z: np.ndarray,
    down: np.ndarray,
    moe_out: np.ndarray,
    ranks: Sequence[int],
    seed: int,
) -> dict[str, Any]:
    n_tokens, top_k, m = gate.shape
    d = down.shape[-1]
    if x.shape != (n_tokens, d):
        raise ValueError((x.shape, n_tokens, d))
    consistency = validate_capture(gate, up, z, down, weights, moe_out)

    token_train, token_val = split_indices(n_tokens, seed)
    assignment_train_mask = np.zeros((n_tokens, top_k), dtype=bool)
    assignment_train_mask[token_train] = True
    assignment_val_mask = ~assignment_train_mask
    gu = np.concatenate([gate, up], axis=-1)

    all_points: list[CurvePoint] = []
    for component, values in {"gate_up": gu, "z": z, "down": down}.items():
        points, _ = global_analysis(component, values[assignment_train_mask], values[assignment_val_mask], ranks, d, m)
        all_points.extend(points)

    expert_summaries: list[ExpertSummary] = []
    per_expert_bases: dict[int, dict[str, np.ndarray]] = {}
    for expert in range(int(topk.max()) + 1):
        positions = np.argwhere(topk == expert)
        if len(positions) < 12:
            continue
        train_ids, val_ids = split_indices(len(positions), seed + 1009 * expert)
        train_pos, val_pos = positions[train_ids], positions[val_ids]
        gt = gu[train_pos[:, 0], train_pos[:, 1]]
        gv = gu[val_pos[:, 0], val_pos[:, 1]]
        zv = z[val_pos[:, 0], val_pos[:, 1]]
        dt = down[train_pos[:, 0], train_pos[:, 1]]
        dv = down[val_pos[:, 0], val_pos[:, 1]]

        gu_vt, gu_train_curve = fit_basis(gt)
        down_vt, down_train_curve = fit_basis(dt)
        per_expert_bases[expert] = {"gate_up": gu_vt, "down": down_vt}
        captures_gu: dict[int, float] = {}
        captures_down: dict[int, float] = {}
        z_errors: dict[int, float] = {}

        for rank in ranks:
            gu_hat = project(gv, gu_vt, rank)
            gate_hat, up_hat = np.split(gu_hat, 2, axis=1)
            z_hat = safe_silu(gate_hat) * up_hat
            z_error = float(np.linalg.norm(z_hat.astype(np.float64) - zv.astype(np.float64)) / max(np.linalg.norm(zv.astype(np.float64)), EPS))
            gu_capture, gu_error, gu_mean, gu_p05 = capture_metrics(gv, gu_hat)
            down_hat = project(dv, down_vt, rank)
            down_capture, down_error, down_mean, down_p05 = capture_metrics(dv, down_hat)
            p_ratio, c_ratio = factor_budget(rank, d, m)
            all_points.append(CurvePoint(
                scope="per_expert", component="gate_up", expert=expert, rank=rank,
                train_samples=len(gt), validation_samples=len(gv),
                validation_capture=gu_capture, validation_relative_error=gu_error,
                validation_mean_sample_capture=gu_mean, validation_p05_sample_capture=gu_p05,
                nonlinear_z_relative_error=z_error, aggregate_moe_relative_error=None,
                idealized_parameter_ratio=p_ratio, idealized_compute_ratio=c_ratio,
            ))
            p_ratio, c_ratio = down_budget(rank, d, m)
            all_points.append(CurvePoint(
                scope="per_expert", component="down", expert=expert, rank=rank,
                train_samples=len(dt), validation_samples=len(dv),
                validation_capture=down_capture, validation_relative_error=down_error,
                validation_mean_sample_capture=down_mean, validation_p05_sample_capture=down_p05,
                nonlinear_z_relative_error=None, aggregate_moe_relative_error=None,
                idealized_parameter_ratio=p_ratio, idealized_compute_ratio=c_ratio,
            ))
            captures_gu[rank] = gu_capture
            captures_down[rank] = down_capture
            z_errors[rank] = z_error

        assigned_weights = weights[positions[:, 0], positions[:, 1]]
        expert_summaries.append(ExpertSummary(
            expert=expert, assignments=len(positions), train_assignments=len(train_pos), validation_assignments=len(val_pos),
            primary_route_fraction=float(np.mean(positions[:, 1] == 0)),
            route_weight_mean=float(np.mean(assigned_weights)),
            route_weight_p50=float(np.percentile(assigned_weights, 50)),
            route_weight_p95=float(np.percentile(assigned_weights, 95)),
            gate_up_rank90_train=rank_at(gu_train_curve, 0.90), down_rank90_train=rank_at(down_train_curve, 0.90),
            gate_up_capture_r16=captures_gu.get(16, float("nan")),
            gate_up_capture_r32=captures_gu.get(32, float("nan")),
            gate_up_capture_r64=captures_gu.get(64, float("nan")),
            down_capture_r16=captures_down.get(16, float("nan")),
            down_capture_r32=captures_down.get(32, float("nan")),
            down_capture_r64=captures_down.get(64, float("nan")),
            z_error_r16=z_errors.get(16, float("nan")),
            z_error_r32=z_errors.get(32, float("nan")),
            z_error_r64=z_errors.get(64, float("nan")),
        ))

    aggregate_rows: list[dict[str, float]] = []
    for rank in ranks:
        reconstructed_down = np.empty_like(down)
        for token in range(n_tokens):
            for slot in range(top_k):
                expert = int(topk[token, slot])
                basis = per_expert_bases.get(expert, {}).get("down")
                reconstructed_down[token, slot] = down[token, slot] if basis is None else project(down[token, slot:slot + 1], basis, rank)[0]
        aggregate_hat = np.sum(reconstructed_down * weights[..., None], axis=1)
        val_true, val_hat = moe_out[token_val], aggregate_hat[token_val]
        relative = float(np.linalg.norm(val_hat.astype(np.float64) - val_true.astype(np.float64)) / max(np.linalg.norm(val_true.astype(np.float64)), EPS))
        aggregate_rows.append({"rank": rank, "heldout_moe_relative_error_oracle_down_only": relative})
        all_points.append(CurvePoint(
            scope="aggregate", component="moe_out_oracle_down", expert=None, rank=rank,
            train_samples=len(token_train), validation_samples=len(token_val),
            validation_capture=1.0 - relative * relative, validation_relative_error=relative,
            validation_mean_sample_capture=float("nan"), validation_p05_sample_capture=float("nan"),
            nonlinear_z_relative_error=None, aggregate_moe_relative_error=relative,
            idealized_parameter_ratio=None, idealized_compute_ratio=None,
        ))

    return {
        "metadata": {"tokens": n_tokens, "top_k": top_k, "hidden": d, "intermediate": m, "seed": seed},
        "consistency": consistency,
        "decision": make_decision(all_points, expert_summaries),
        "aggregate_oracle": aggregate_rows,
        "expert_summaries": [asdict(x) for x in expert_summaries],
        "curve_points": [asdict(x) for x in all_points],
    }


def make_decision(points: Sequence[CurvePoint], experts: Sequence[ExpertSummary]) -> dict[str, Any]:
    def optional_global(component: str, rank: int) -> CurvePoint | None:
        return next((p for p in points if p.scope == "global" and p.component == component and p.rank == rank), None)
    gu512, down512 = optional_global("gate_up", 512), optional_global("down", 512)
    median_gu32 = float(np.nanmedian([e.gate_up_capture_r32 for e in experts]))
    median_down32 = float(np.nanmedian([e.down_capture_r32 for e in experts]))
    median_z32 = float(np.nanmedian([e.z_error_r32 for e in experts]))
    if gu512 and down512 and gu512.validation_capture >= 0.90 and down512.validation_capture >= 0.90:
        verdict = "GLOBAL_PASS"
    elif median_gu32 >= 0.90 and median_down32 >= 0.90 and median_z32 <= 0.20:
        verdict = "PER_EXPERT_PROMISING"
    elif median_gu32 >= 0.70 or median_down32 >= 0.70:
        verdict = "BORDERLINE"
    else:
        verdict = "FAIL"
    return {
        "verdict": verdict,
        "global_gate_up_capture_r512": gu512.validation_capture if gu512 else None,
        "global_down_capture_r512": down512.validation_capture if down512 else None,
        "median_expert_gate_up_capture_r32": median_gu32,
        "median_expert_down_capture_r32": median_down32,
        "median_expert_z_relative_error_r32": median_z32,
        "note": "Per-expert ranks are pilot estimates limited by assignments per expert; a positive signal requires a larger corpus before a compression claim.",
    }


def write_outputs(output_dir: Path, payload: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "routed_expert_subspaces.json").write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=True) + "\n", encoding="utf-8")
    curves = payload["curve_points"]
    with (output_dir / "curves.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(curves[0].keys()))
        writer.writeheader(); writer.writerows(curves)
    experts = payload["expert_summaries"]
    with (output_dir / "experts.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(experts[0].keys()))
        writer.writeheader(); writer.writerows(experts)

    d, c = payload["decision"], payload["consistency"]
    lines = [
        "# Test 1.1 — routed expert activation subspaces", "", f"**Decision:** **{d['verdict']}**", "",
        f"- SwiGLU capture consistency error: `{c['swiglu_consistency_error']:.3e}`",
        f"- Weighted down/output consistency error: `{c['weighted_down_consistency_error']:.3e}`", "",
        "| Metric | Value |", "|---|---:|",
        f"| Global gate+up held-out capture @512 | {d['global_gate_up_capture_r512']:.2%} |" if d['global_gate_up_capture_r512'] is not None else "| Global gate+up held-out capture @512 | n/a |",
        f"| Global down held-out capture @512 | {d['global_down_capture_r512']:.2%} |" if d['global_down_capture_r512'] is not None else "| Global down held-out capture @512 | n/a |",
        f"| Median per-expert gate+up capture @32 | {d['median_expert_gate_up_capture_r32']:.2%} |",
        f"| Median per-expert down capture @32 | {d['median_expert_down_capture_r32']:.2%} |",
        f"| Median per-expert nonlinear z error @32 | {d['median_expert_z_relative_error_r32']:.2%} |", "",
        "The per-expert oracle projection is directly realizable as a low-rank matrix factorization `A(AᵀW)x`; it does not reconstruct or compare raw weights. Aggregate down error is an optimistic lower bound because gate/up approximation is not propagated through `D_e` in this screen.",
    ]
    (output_dir / "VERDICT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def self_test() -> None:
    rng = np.random.default_rng(41)
    n, e, topk, m, d, local_rank = 1200, 8, 2, 32, 64, 4
    top_ids = np.empty((n, topk), dtype=np.int32)
    for i in range(n):
        top_ids[i] = rng.choice(e, size=topk, replace=False)
    weights = rng.random((n, topk), dtype=np.float32)
    weights /= weights.sum(axis=1, keepdims=True)
    x = rng.normal(size=(n, d)).astype(np.float32)
    gate_bases = [np.linalg.qr(rng.normal(size=(2 * m, local_rank)))[0].astype(np.float32) for _ in range(e)]
    down_bases = [np.linalg.qr(rng.normal(size=(d, local_rank)))[0].astype(np.float32) for _ in range(e)]
    gate = np.empty((n, topk, m), dtype=np.float32)
    up = np.empty_like(gate)
    down = np.empty((n, topk, d), dtype=np.float32)
    for i in range(n):
        for s in range(topk):
            expert = int(top_ids[i, s])
            gu = rng.normal(size=local_rank).astype(np.float32) @ gate_bases[expert].T
            gate[i, s], up[i, s] = np.split(gu, 2)
            down[i, s] = rng.normal(size=local_rank).astype(np.float32) @ down_bases[expert].T
    z = safe_silu(gate) * up
    moe = np.sum(down * weights[..., None], axis=1)
    payload = analyze(x, top_ids, weights, gate, up, z, down, moe, [4, 8, 16, 32, 64, 512], 7)
    decision = payload["decision"]
    if decision["median_expert_gate_up_capture_r32"] < 0.999 or decision["median_expert_down_capture_r32"] < 0.999:
        raise AssertionError(decision)
    if payload["consistency"]["weighted_down_consistency_error"] > 1e-6:
        raise AssertionError(payload["consistency"])
    print(f"self-test passed: {decision}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--capture-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("results/routed_expert_subspaces"))
    parser.add_argument("--ranks", default="4,8,16,24,32,48,64,128,256,512")
    parser.add_argument("--seed", type=int, default=20260731)
    args = parser.parse_args(argv)
    if args.self_test:
        self_test(); return 0
    if args.capture_dir is None:
        parser.error("--capture-dir is required")
    captured: dict[str, np.ndarray] = {}
    for meta in sorted(args.capture_dir.glob("*.jsonl")):
        name, array = load_chunks(meta)
        captured[name] = array
    x = find_tensor(captured, "ffn_norm-7")
    topk = np.rint(find_tensor(captured, "ffn_moe_topk-7")).astype(np.int32)
    weights = find_tensor(captured, "ffn_moe_weights-7")
    gate = find_tensor(captured, "ffn_moe_gate-7")
    up = find_tensor(captured, "ffn_moe_up-7")
    z = find_tensor(captured, "ffn_moe_swiglu-7")
    down = find_tensor(captured, "ffn_moe_down-7")
    moe_out = find_tensor(captured, "ffn_moe_out-7")
    ranks = sorted({int(x) for x in args.ranks.split(",") if x.strip()})
    payload = analyze(x, topk, weights, gate, up, z, down, moe_out, ranks, args.seed)
    payload["captured_tensor_names"] = sorted(captured)
    write_outputs(args.output_dir, payload)
    print((args.output_dir / "VERDICT.md").read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
