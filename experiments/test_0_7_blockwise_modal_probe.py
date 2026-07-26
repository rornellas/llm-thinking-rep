#!/usr/bin/env python3
"""Blockwise expert-axis modal decomposition probe for OLMoE.

A global expert-axis PCA can be full-rank even when every weight tile is driven
by only a few local expert modes, because each tile may use a different code
space. This probe tests that escape route directly.

For each sampled tile b and projection p:

    W[e, b] = mean[b] + sum_k a[e, b, k] * B[b, k]

The representation remains directly executable tile by tile. Its idealized
matmul ratio versus top-k MoE is approximately mean(K_b + 1) / top_k. Its
parameter ratio includes both tile modes and expert-specific scalar codes:

    (K_b + 1) / E + K_b / tile_elements.
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
import torch

from modal_moe_weight_probe_v2 import (
    HttpRangeSafeTensorSource,
    LocalSafeTensorSource,
    build_default_remote_url,
    discover_layer_keys,
)

EPS = 1e-12


@dataclass
class TileMetric:
    projection: str
    block_size: int
    tile_index: int
    row_start: int
    col_start: int
    common_energy_fraction: float
    rank80: int
    rank90: int
    rank95: int
    rank99: int
    explained_k1: float
    explained_k2: float
    explained_k3: float
    explained_k4: float
    explained_k5: float
    explained_k6: float
    explained_k7: float
    stable_rank: float
    participation_ratio: float


@dataclass
class BlockSummary:
    projection: str
    block_size: int
    sampled_tiles: int
    tile_elements: int
    mean_common_energy: float
    mean_explained_k4: float
    median_explained_k4: float
    p10_explained_k4: float
    mean_explained_k7: float
    median_rank90: float
    mean_rank90: float
    median_rank95: float
    mean_rank95: float
    p90_rank95: float
    variable_k90_parameter_ratio: float
    variable_k90_compute_ratio: float
    variable_k95_parameter_ratio: float
    variable_k95_compute_ratio: float
    fixed_k4_parameter_ratio: float
    fixed_k4_compute_ratio: float


def rank_for(eigenvalues: np.ndarray, fraction: float) -> int:
    total = float(eigenvalues.sum())
    if total <= 0:
        return 0
    return int(np.searchsorted(np.cumsum(eigenvalues), fraction * total, side="left") + 1)


def explained(eigenvalues: np.ndarray, rank: int) -> float:
    total = float(eigenvalues.sum())
    return float(eigenvalues[:rank].sum() / total) if total > 0 else 1.0


def tile_metric(projection: str, block_size: int, tile_index: int, row: int, col: int, x: np.ndarray) -> TileMetric:
    x64 = np.asarray(x, dtype=np.float64)
    experts = x64.shape[0]
    original_energy = float(np.sum(x64 * x64))
    mean = np.mean(x64, axis=0)
    centered = x64 - mean
    gram = centered @ centered.T
    gram = (gram + gram.T) * 0.5
    eigenvalues = np.maximum(np.linalg.eigvalsh(gram)[::-1][: experts - 1], 0.0)
    total = float(eigenvalues.sum())
    common = experts * float(np.dot(mean, mean))
    sq = float(np.dot(eigenvalues, eigenvalues))
    return TileMetric(
        projection=projection,
        block_size=block_size,
        tile_index=tile_index,
        row_start=row,
        col_start=col,
        common_energy_fraction=common / original_energy if original_energy > 0 else 0.0,
        rank80=rank_for(eigenvalues, 0.80),
        rank90=rank_for(eigenvalues, 0.90),
        rank95=rank_for(eigenvalues, 0.95),
        rank99=rank_for(eigenvalues, 0.99),
        explained_k1=explained(eigenvalues, 1),
        explained_k2=explained(eigenvalues, 2),
        explained_k3=explained(eigenvalues, 3),
        explained_k4=explained(eigenvalues, 4),
        explained_k5=explained(eigenvalues, 5),
        explained_k6=explained(eigenvalues, 6),
        explained_k7=explained(eigenvalues, 7),
        stable_rank=total / float(eigenvalues[0]) if eigenvalues.size and eigenvalues[0] > 0 else 0.0,
        participation_ratio=total * total / sq if sq > 0 else 0.0,
    )


def choose_tiles(rows: int, cols: int, block_size: int, count: int, seed: int) -> list[tuple[int, int]]:
    row_starts = np.arange(0, rows - block_size + 1, block_size, dtype=np.int32)
    col_starts = np.arange(0, cols - block_size + 1, block_size, dtype=np.int32)
    grid = [(int(r), int(c)) for r in row_starts for c in col_starts]
    rng = np.random.default_rng(seed)
    if count >= len(grid):
        order = rng.permutation(len(grid))
    else:
        order = rng.choice(len(grid), size=count, replace=False)
    return [grid[int(i)] for i in order]


def load_tiles(
    source: Any,
    projection_entries: list[tuple[int, str]],
    tile_locations: list[tuple[int, int]],
    block_size: int,
) -> np.ndarray:
    experts = len(projection_entries)
    elements = block_size * block_size
    values = np.empty((len(tile_locations), experts, elements), dtype=np.float32)
    for expert_position, (expert_id, key) in enumerate(projection_entries):
        matrix = source.get_tensor(key).to(torch.float32).cpu().numpy()
        for tile_index, (row, col) in enumerate(tile_locations):
            values[tile_index, expert_position] = matrix[
                row:row + block_size,
                col:col + block_size,
            ].reshape(-1)
        print(
            f"block={block_size} loaded expert {expert_id} "
            f"({expert_position + 1}/{experts})",
            flush=True,
        )
        del matrix
    return values


def ratio_for_rank(rank: np.ndarray | float, experts: int, tile_elements: int, top_k: int) -> tuple[float, float]:
    k = np.asarray(rank, dtype=np.float64)
    parameter_ratio = float(np.mean((k + 1.0) / experts + k / tile_elements))
    compute_ratio = float(np.mean((k + 1.0) / top_k))
    return parameter_ratio, compute_ratio


def summarize(metrics: Sequence[TileMetric], experts: int, top_k: int) -> BlockSummary:
    first = metrics[0]
    elements = first.block_size * first.block_size
    rank90 = np.array([m.rank90 for m in metrics], dtype=np.float64)
    rank95 = np.array([m.rank95 for m in metrics], dtype=np.float64)
    p90_param, p90_compute = ratio_for_rank(rank90, experts, elements, top_k)
    p95_param, p95_compute = ratio_for_rank(rank95, experts, elements, top_k)
    k4_param, k4_compute = ratio_for_rank(4.0, experts, elements, top_k)
    k4 = np.array([m.explained_k4 for m in metrics], dtype=np.float64)
    k7 = np.array([m.explained_k7 for m in metrics], dtype=np.float64)
    common = np.array([m.common_energy_fraction for m in metrics], dtype=np.float64)
    return BlockSummary(
        projection=first.projection,
        block_size=first.block_size,
        sampled_tiles=len(metrics),
        tile_elements=elements,
        mean_common_energy=float(np.mean(common)),
        mean_explained_k4=float(np.mean(k4)),
        median_explained_k4=float(np.median(k4)),
        p10_explained_k4=float(np.percentile(k4, 10)),
        mean_explained_k7=float(np.mean(k7)),
        median_rank90=float(np.median(rank90)),
        mean_rank90=float(np.mean(rank90)),
        median_rank95=float(np.median(rank95)),
        mean_rank95=float(np.mean(rank95)),
        p90_rank95=float(np.percentile(rank95, 90)),
        variable_k90_parameter_ratio=p90_param,
        variable_k90_compute_ratio=p90_compute,
        variable_k95_parameter_ratio=p95_param,
        variable_k95_compute_ratio=p95_compute,
        fixed_k4_parameter_ratio=k4_param,
        fixed_k4_compute_ratio=k4_compute,
    )


def decide(summaries: Sequence[BlockSummary]) -> dict[str, Any]:
    candidates: list[dict[str, Any]] = []
    for block_size in sorted({s.block_size for s in summaries}):
        group = [s for s in summaries if s.block_size == block_size]
        worst_k4 = min(s.p10_explained_k4 for s in group)
        mean_k4 = float(np.mean([s.mean_explained_k4 for s in group]))
        worst_mean_k90_compute = max(s.variable_k90_compute_ratio for s in group)
        worst_mean_k95_compute = max(s.variable_k95_compute_ratio for s in group)
        candidates.append({
            "block_size": block_size,
            "worst_projection_p10_k4": worst_k4,
            "mean_explained_k4": mean_k4,
            "worst_projection_variable_k90_compute_ratio": worst_mean_k90_compute,
            "worst_projection_variable_k95_compute_ratio": worst_mean_k95_compute,
        })
    pass_block = next((x["block_size"] for x in candidates if
                       x["mean_explained_k4"] >= 0.80
                       and x["worst_projection_p10_k4"] >= 0.60
                       and x["worst_projection_variable_k90_compute_ratio"] <= 0.90), None)
    borderline_block = next((x["block_size"] for x in candidates if
                             x["mean_explained_k4"] >= 0.50
                             and x["worst_projection_variable_k90_compute_ratio"] <= 1.25), None)
    verdict = "PASS" if pass_block is not None else ("BORDERLINE" if borderline_block is not None else "FAIL")
    return {
        "verdict": verdict,
        "pass_block": pass_block,
        "borderline_block": borderline_block,
        "candidates": candidates,
    }


def write_outputs(output_dir: Path, payload: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "blockwise_modal.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    metrics = payload["tile_metrics"]
    with (output_dir / "tiles.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metrics[0].keys()))
        writer.writeheader(); writer.writerows(metrics)
    summaries = payload["summaries"]
    with (output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summaries[0].keys()))
        writer.writeheader(); writer.writerows(summaries)

    decision = payload["decision"]
    lines = [
        "# Test 0.7 — blockwise expert modes",
        "",
        f"**Decision:** **{decision['verdict']}**",
        "",
        "| Projection | Block | Mean K=4 explained | p10 K=4 explained | Mean rank90 | Mean rank95 | K90 compute ratio | K95 compute ratio |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in summaries:
        lines.append(
            f"| {item['projection']} | {item['block_size']} | "
            f"{item['mean_explained_k4']:.2%} | {item['p10_explained_k4']:.2%} | "
            f"{item['mean_rank90']:.2f} | {item['mean_rank95']:.2f} | "
            f"{item['variable_k90_compute_ratio']:.2%} | {item['variable_k95_compute_ratio']:.2%} |"
        )
    lines += [
        "",
        "A compute ratio below 100% is necessary but not sufficient for a real kernel speedup.",
        "PASS requires strong K=4 locality and a variable-rank 90% reconstruction budget below 90% of top-8 compute.",
    ]
    (output_dir / "VERDICT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def self_test() -> None:
    rng = np.random.default_rng(31)
    experts, tiles, elements, local_rank = 64, 32, 256, 3
    blocks = np.empty((tiles, experts, elements), dtype=np.float32)
    for tile in range(tiles):
        mean = rng.normal(size=elements)
        modes = rng.normal(size=(local_rank, elements))
        coeff = rng.normal(size=(experts, local_rank))
        blocks[tile] = mean + coeff @ modes + rng.normal(scale=1e-3, size=(experts, elements))
    metrics = [tile_metric("synthetic", 16, i, 0, 0, blocks[i]) for i in range(tiles)]
    if max(m.rank95 for m in metrics) > local_rank:
        raise AssertionError("local modal rank was not recovered")
    global_x = blocks.transpose(1, 0, 2).reshape(experts, -1)
    global_metric = tile_metric("synthetic-global", 16, 0, 0, 0, global_x)
    if global_metric.rank95 < 40:
        raise AssertionError("synthetic construction did not separate local and global rank")
    print(
        f"self-test passed: local rank95≤{max(m.rank95 for m in metrics)}, "
        f"global rank95={global_metric.rank95}"
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--model-id", default="allenai/OLMoE-1B-7B-0924")
    parser.add_argument("--revision", default="3a970199d0f87db4e3e57275abb93812bf10fd83")
    parser.add_argument("--shard-name", default="model-00002-of-00003.safetensors")
    parser.add_argument("--shard-path", type=Path)
    parser.add_argument("--remote-shard-url")
    parser.add_argument("--cache-dir", type=Path, default=Path(".cache/olmoe-block-ranges"))
    parser.add_argument("--layer", type=int, default=7)
    parser.add_argument("--block-sizes", default="16,32,64,128")
    parser.add_argument("--tiles-per-size", default="256,128,64,32")
    parser.add_argument("--seed", type=int, default=20260728)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--output-dir", type=Path, default=Path("results/blockwise_modal"))
    args = parser.parse_args(argv)

    if args.self_test:
        self_test(); return 0

    block_sizes = [int(x) for x in args.block_sizes.split(",") if x.strip()]
    tile_counts = [int(x) for x in args.tiles_per_size.split(",") if x.strip()]
    if len(block_sizes) != len(tile_counts):
        raise ValueError("block-sizes and tiles-per-size must have equal lengths")

    started = time.perf_counter()
    if args.shard_path:
        source = LocalSafeTensorSource(args.shard_path)
    else:
        url = args.remote_shard_url or build_default_remote_url(args.model_id, args.revision, args.shard_name)
        source = HttpRangeSafeTensorSource(url, cache_dir=args.cache_dir)
    discovered = discover_layer_keys(source.keys(), args.layer)
    all_metrics: list[TileMetric] = []
    all_summaries: list[BlockSummary] = []

    for projection_index, projection in enumerate(("gate", "up", "down")):
        entries = discovered[projection]
        shape = source.descriptor(entries[0][1]).shape
        print(f"\n=== projection={projection}, shape={shape} ===", flush=True)
        for block_size, tile_count in zip(block_sizes, tile_counts, strict=True):
            locations = choose_tiles(
                shape[0], shape[1], block_size, tile_count,
                args.seed + projection_index * 10000 + block_size,
            )
            values = load_tiles(source, entries, locations, block_size)
            metrics = [
                tile_metric(projection, block_size, i, row, col, values[i])
                for i, (row, col) in enumerate(locations)
            ]
            summary = summarize(metrics, len(entries), args.top_k)
            all_metrics.extend(metrics); all_summaries.append(summary)
            print(
                f"{projection} block={block_size}: mean K4={summary.mean_explained_k4:.2%}, "
                f"mean rank90={summary.mean_rank90:.2f}, "
                f"K90 compute={summary.variable_k90_compute_ratio:.2%}",
                flush=True,
            )
            del values

    payload = {
        "metadata": {
            "model_id": args.model_id,
            "revision": args.revision,
            "layer": args.layer,
            "experts": len(discovered["gate"]),
            "top_k": args.top_k,
            "block_sizes": block_sizes,
            "tiles_per_size": tile_counts,
            "seed": args.seed,
            "elapsed_seconds": time.perf_counter() - started,
            "source": source.metadata(),
        },
        "decision": decide(all_summaries),
        "summaries": [asdict(x) for x in all_summaries],
        "tile_metrics": [asdict(x) for x in all_metrics],
    }
    write_outputs(args.output_dir, payload)
    print((args.output_dir / "VERDICT.md").read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
