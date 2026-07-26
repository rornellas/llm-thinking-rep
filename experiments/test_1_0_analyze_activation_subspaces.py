#!/usr/bin/env python3
"""Analyze held-out activation subspaces captured from llama.cpp."""
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

EPS = 1e-12


@dataclass
class CurvePoint:
    tensor: str
    rank: int
    train_capture: float
    validation_capture: float
    validation_mean_token_capture: float
    validation_p05_token_capture: float
    isotropic_expectation: float
    input_branch_compute_ratio: float
    output_branch_compute_ratio: float
    combined_compute_ratio: float


@dataclass
class TensorSummary:
    tensor: str
    tokens: int
    features: int
    train_tokens: int
    validation_tokens: int
    train_rank80: int | None
    train_rank90: int | None
    train_rank95: int | None
    train_rank99: int | None
    validation_capture_64: float
    validation_capture_128: float
    validation_capture_256: float
    validation_capture_384: float
    validation_capture_512: float
    validation_capture_max: float
    validation_max_rank: int
    stable_rank_train: float
    participation_ratio_train: float


def load_tensor(meta_path: Path) -> tuple[str, np.ndarray]:
    records = [json.loads(line) for line in meta_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not records:
        raise ValueError(f"empty metadata: {meta_path}")
    data_path = meta_path.with_suffix(".f32")
    raw = np.fromfile(data_path, dtype=np.float32)
    offset = 0
    chunks: list[np.ndarray] = []
    feature_dim: int | None = None
    for record in records:
        ne = record["ne"]
        elements = int(record["elements"])
        chunk = raw[offset:offset + elements]
        offset += elements
        if chunk.size != elements:
            raise ValueError(f"truncated data in {data_path}")
        observations = int(np.prod(ne[1:]))
        if feature_dim is None:
            feature_dim = int(ne[0])
        if int(ne[0]) != feature_dim:
            raise ValueError("feature dimension changed across chunks")
        chunks.append(chunk.reshape(observations, feature_dim))
    if offset != raw.size:
        raise ValueError(f"unused values in {data_path}: {raw.size - offset}")
    return records[0]["name"], np.concatenate(chunks, axis=0)


def rank_at(cumulative: np.ndarray, threshold: float) -> int | None:
    hits = np.flatnonzero(cumulative >= threshold)
    return int(hits[0] + 1) if hits.size else None


def capture_at(curve: np.ndarray, rank: int) -> float:
    if rank <= 0 or curve.size == 0:
        return 0.0
    return float(curve[min(rank, curve.size) - 1])


def compute_budgets(rank: int, hidden: int = 2048, intermediate: int = 1024, top_k: int = 8) -> tuple[float, float, float]:
    input_original = 2 * top_k * intermediate * hidden
    input_factored = hidden * rank + 2 * top_k * intermediate * rank
    output_original = top_k * intermediate * hidden
    output_factored = top_k * intermediate * rank + rank * hidden
    combined_original = 3 * top_k * intermediate * hidden
    combined_factored = 2 * hidden * rank + 3 * top_k * intermediate * rank
    return (
        input_factored / input_original,
        output_factored / output_original,
        combined_factored / combined_original,
    )


def analyze_tensor(name: str, x: np.ndarray, ranks: Sequence[int], seed: int) -> tuple[list[CurvePoint], TensorSummary]:
    x = np.asarray(x, dtype=np.float32)
    rng = np.random.default_rng(seed)
    order = rng.permutation(x.shape[0])
    train_n = max(1, int(round(0.67 * x.shape[0])))
    train_n = min(train_n, x.shape[0] - 1) if x.shape[0] > 1 else 1
    train = x[order[:train_n]]
    validation = x[order[train_n:]] if train_n < x.shape[0] else x[order[:1]]

    _, singular_values, vt = np.linalg.svd(train, full_matrices=False)
    energy = singular_values.astype(np.float64) ** 2
    train_curve = np.cumsum(energy) / max(float(energy.sum()), EPS)
    stable_rank = float(energy.sum() / max(float(energy[0]), EPS))
    participation = float(energy.sum() ** 2 / max(float(np.dot(energy, energy)), EPS))

    coefficients = validation @ vt.T
    squared = coefficients.astype(np.float64) ** 2
    total_validation = float(np.sum(validation.astype(np.float64) ** 2))
    validation_curve = np.cumsum(np.sum(squared, axis=0)) / max(total_validation, EPS)
    token_total = np.sum(validation.astype(np.float64) ** 2, axis=1)
    token_cumulative = np.cumsum(squared, axis=1) / np.maximum(token_total[:, None], EPS)

    points: list[CurvePoint] = []
    for rank in ranks:
        effective = min(rank, vt.shape[0])
        input_ratio, output_ratio, combined_ratio = compute_budgets(rank)
        token_values = token_cumulative[:, effective - 1]
        points.append(CurvePoint(
            tensor=name,
            rank=rank,
            train_capture=capture_at(train_curve, effective),
            validation_capture=capture_at(validation_curve, effective),
            validation_mean_token_capture=float(np.mean(token_values)),
            validation_p05_token_capture=float(np.percentile(token_values, 5)),
            isotropic_expectation=min(rank, x.shape[1]) / x.shape[1],
            input_branch_compute_ratio=input_ratio,
            output_branch_compute_ratio=output_ratio,
            combined_compute_ratio=combined_ratio,
        ))

    summary = TensorSummary(
        tensor=name,
        tokens=int(x.shape[0]),
        features=int(x.shape[1]),
        train_tokens=int(train.shape[0]),
        validation_tokens=int(validation.shape[0]),
        train_rank80=rank_at(train_curve, 0.80),
        train_rank90=rank_at(train_curve, 0.90),
        train_rank95=rank_at(train_curve, 0.95),
        train_rank99=rank_at(train_curve, 0.99),
        validation_capture_64=capture_at(validation_curve, 64),
        validation_capture_128=capture_at(validation_curve, 128),
        validation_capture_256=capture_at(validation_curve, 256),
        validation_capture_384=capture_at(validation_curve, 384),
        validation_capture_512=capture_at(validation_curve, 512),
        validation_capture_max=float(validation_curve[-1]),
        validation_max_rank=int(vt.shape[0]),
        stable_rank_train=stable_rank,
        participation_ratio_train=participation,
    )
    return points, summary


def choose_verdict(summaries: Sequence[TensorSummary]) -> dict:
    input_item = next((s for s in summaries if "ffn_norm-7" in s.tensor), None)
    output_item = next((s for s in summaries if "ffn_moe_out-7" in s.tensor), None)
    if input_item is None:
        return {"verdict": "INVALID", "reason": "ffn_norm-7 was not captured"}
    input_capture = input_item.validation_capture_512
    output_capture = output_item.validation_capture_512 if output_item else None
    if input_capture >= 0.90 and (output_capture is None or output_capture >= 0.90):
        verdict = "PASS"
    elif input_capture >= 0.70:
        verdict = "BORDERLINE"
    else:
        verdict = "FAIL"
    return {
        "verdict": verdict,
        "input_validation_capture_r512": input_capture,
        "output_validation_capture_r512": output_capture,
        "reason": "PASS authorizes activation-aware expert-output factorization; BORDERLINE authorizes distillation/retraining tests.",
    }


def write_outputs(output_dir: Path, payload: dict) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "activation_subspaces.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    points = payload["curve_points"]
    with (output_dir / "curve.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(points[0].keys()))
        writer.writeheader(); writer.writerows(points)
    summaries = payload["summaries"]
    with (output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summaries[0].keys()))
        writer.writeheader(); writer.writerows(summaries)

    verdict = payload["decision"]
    lines = [
        "# Test 1.0 — real activation subspaces",
        "",
        f"**Decision:** **{verdict['verdict']}**",
        "",
        "A basis is fitted on 67% of the captured tokens and evaluated on disjoint held-out tokens.",
        "The PCA is uncentered because the target implementation is a linear shared projection without an additive bias.",
        "",
        "| Tensor | Tokens | Train rank90 | Validation @128 | @256 | @384 | @512 | Max held-out capture |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in summaries:
        lines.append(
            f"| {item['tensor']} | {item['tokens']} | {item['train_rank90']} | "
            f"{item['validation_capture_128']:.2%} | {item['validation_capture_256']:.2%} | "
            f"{item['validation_capture_384']:.2%} | {item['validation_capture_512']:.2%} | "
            f"{item['validation_capture_max']:.2%} |"
        )
    lines += [
        "",
        "At rank 512 the idealized combined gate/up/down factorized compute ratio is 29.17% of the original top-8 projections, before kernel overhead.",
        "This is an oracle subspace screen. It does not yet prove that expert-specific low-dimensional cores preserve logits.",
    ]
    (output_dir / "VERDICT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def self_test() -> None:
    rng = np.random.default_rng(91)
    tokens, features, true_rank = 600, 256, 32
    basis, _ = np.linalg.qr(rng.normal(size=(features, true_rank)))
    x = (rng.normal(size=(tokens, true_rank)) @ basis.T + rng.normal(scale=1e-3, size=(tokens, features))).astype(np.float32)
    points, summary = analyze_tensor("ffn_norm-7", x, [16, 32, 64], 1)
    p32 = next(p for p in points if p.rank == 32)
    if p32.validation_capture < 0.999:
        raise AssertionError(p32)
    if summary.train_rank90 > true_rank:
        raise AssertionError(summary)
    print(f"self-test passed: rank32 held-out capture={p32.validation_capture:.6%}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--capture-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("results/activation_subspaces"))
    parser.add_argument("--ranks", default="16,32,64,128,192,256,384,512")
    parser.add_argument("--seed", type=int, default=20260730)
    args = parser.parse_args(argv)
    if args.self_test:
        self_test(); return 0
    if args.capture_dir is None:
        parser.error("--capture-dir is required")
    ranks = sorted({int(x) for x in args.ranks.split(",") if x.strip()})
    points: list[CurvePoint] = []
    summaries: list[TensorSummary] = []
    for meta_path in sorted(args.capture_dir.glob("*.jsonl")):
        name, x = load_tensor(meta_path)
        if x.shape[1] != 2048:
            continue
        tensor_points, summary = analyze_tensor(name, x, ranks, args.seed + len(summaries))
        points.extend(tensor_points); summaries.append(summary)
    if not summaries:
        raise RuntimeError("no 2048-dimensional captured tensors found")
    payload = {
        "decision": choose_verdict(summaries),
        "summaries": [asdict(x) for x in summaries],
        "curve_points": [asdict(x) for x in points],
    }
    write_outputs(args.output_dir, payload)
    print((args.output_dir / "VERDICT.md").read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
