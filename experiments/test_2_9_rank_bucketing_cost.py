#!/usr/bin/env python3
"""Hardware-aware cost bounds for per-token nested Modal-MoE ranks.

Two schedules are compared:

1. unbucketed tiles: every token in a tile executes the maximum rank present;
2. rank compaction: tokens are grouped by predicted rank and only tail padding
   within each bucket is paid.

The first is an analytical iid divergence estimate from the empirical rank
histogram. The second is an exact padded-work calculation for that histogram.
Neither includes measured scatter/prefix-sum latency; that is the next kernel
benchmark.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Sequence


def normalized_counts(raw: dict[str, int]) -> dict[int, int]:
    counts = {int(key): int(value) for key, value in raw.items()}
    if not counts or min(counts) < 0 or sum(counts.values()) <= 0:
        raise ValueError(raw)
    return counts


def compute_ratio(mean_rank: float, projected_modes_per_token: float, overhead: float) -> float:
    return projected_modes_per_token / 8.0 + mean_rank / 2048.0 + overhead


def expected_unbucketed_max(counts: dict[int, int], tile: int) -> float:
    total = sum(counts.values())
    max_rank = max(counts)
    cumulative = 0.0
    cdf: dict[int, float] = {}
    for rank in range(max_rank + 1):
        cumulative += counts.get(rank, 0) / total
        cdf[rank] = cumulative
    # E[max] = sum_{k=1}^R P(max >= k)
    return sum(1.0 - cdf[rank - 1] ** tile for rank in range(1, max_rank + 1))


def compacted_modes_per_token(counts: dict[int, int], tile: int) -> tuple[float, int]:
    total = sum(counts.values())
    padded_tokens = 0
    mode_work = 0
    for rank, count in counts.items():
        padded = math.ceil(count / tile) * tile
        padded_tokens += padded - count
        mode_work += padded * (rank + 1)
    return mode_work / total, padded_tokens


def analyze(payload: dict[str, Any], tiles: Sequence[int]) -> dict[str, Any]:
    dynamic = payload["dynamic_test"]
    counts = normalized_counts(dynamic["rank_counts"])
    total = sum(counts.values())
    mean_rank = sum(rank * count for rank, count in counts.items()) / total
    overhead = float(dynamic.get("controller_overhead_ratio", 0.0))
    rows: list[dict[str, Any]] = []
    for tile in tiles:
        expected_max = expected_unbucketed_max(counts, tile)
        unbucketed = compute_ratio(expected_max, expected_max + 1.0, overhead)
        compacted_modes, padded_tokens = compacted_modes_per_token(counts, tile)
        compacted = compute_ratio(mean_rank, compacted_modes, overhead)
        rows.append({
            "tile": tile,
            "tokens": total,
            "mean_rank": mean_rank,
            "expected_unbucketed_max_rank": expected_max,
            "unbucketed_compute_ratio": unbucketed,
            "compacted_compute_ratio": compacted,
            "compaction_padding_tokens": padded_tokens,
            "compaction_padding_fraction": padded_tokens / total,
            "static_k1_compute_ratio": 2.0 / 8.0 + 1.0 / 2048.0,
            "unbucketed_advantage_vs_k1": (2.0 / 8.0 + 1.0 / 2048.0) - unbucketed,
            "compacted_advantage_vs_k1": (2.0 / 8.0 + 1.0 / 2048.0) - compacted,
        })
    tile2 = next(row for row in rows if row["tile"] == 2)
    largest_compacted = rows[-1]
    if tile2["unbucketed_advantage_vs_k1"] <= 0 and largest_compacted["compacted_advantage_vs_k1"] > 0:
        verdict = "RANK_COMPACTION_REQUIRED"
    elif tile2["unbucketed_advantage_vs_k1"] > 0:
        verdict = "SMALL_UNBUCKETED_TILES_POSSIBLE"
    else:
        verdict = "DYNAMIC_SCHEDULING_NOT_USEFUL"
    return {
        "metadata": {
            "rank_counts": {str(key): value for key, value in counts.items()},
            "tokens_across_layers": total,
            "mean_rank": mean_rank,
            "controller_overhead_ratio": overhead,
            "source_dynamic_compute_ratio": dynamic["projected_olmoe_compute_ratio"],
            "assumption": "unbucketed ranks are iid within a tile; compacted cost uses exact observed counts",
        },
        "decision": {
            "verdict": verdict,
            "note": "Projected matrix work only. Scatter, scan, launch, and memory-layout costs require a kernel benchmark.",
        },
        "rows": rows,
    }


def self_test() -> None:
    counts = {0: 50, 1: 25, 2: 15, 3: 10}
    if abs(expected_unbucketed_max(counts, 1) - 0.85) > 1e-12:
        raise AssertionError(expected_unbucketed_max(counts, 1))
    modes, padding = compacted_modes_per_token(counts, 8)
    expected = (56 * 1 + 32 * 2 + 16 * 3 + 16 * 4) / 100
    if abs(modes - expected) > 1e-12 or padding != 20:
        raise AssertionError((modes, padding))
    print("self-test passed for unbucketed and compacted cost formulas")


def write_outputs(output_dir: Path, result: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "rank_bucketing.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    with (output_dir / "rank_bucketing.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(result["rows"][0].keys()))
        writer.writeheader(); writer.writerows(result["rows"])
    lines = [
        "# Test 2.9 — rank bucketing and tile-divergence cost", "",
        f"**Decision:** **{result['decision']['verdict']}**", "",
        "| Tile | Expected max K without bucketing | Compute without bucketing | Compute after rank compaction | Padding after compaction | Advantage vs static K1 |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in result["rows"]:
        lines.append(
            f"| {row['tile']} | {row['expected_unbucketed_max_rank']:.3f} | "
            f"{row['unbucketed_compute_ratio']:.3%} | {row['compacted_compute_ratio']:.3%} | "
            f"{row['compaction_padding_fraction']:.3%} | {row['compacted_advantage_vs_k1']:+.3%} |"
        )
    lines += [
        "",
        "Unbucketed estimates assume ranks are independently mixed inside a hardware tile. Compacted estimates sort tokens by predicted rank and include exact tail padding. The controller and elementwise code overhead are included; scan/scatter and kernel-launch latency are not.",
    ]
    (output_dir / "VERDICT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("results/test-2-9/latest"))
    parser.add_argument("--tiles", default="1,2,4,8,16,32,64,128,256,512")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args(argv)
    if args.self_test:
        self_test(); return 0
    if args.input is None:
        parser.error("--input is required")
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    tiles = sorted({int(value) for value in args.tiles.split(",") if value.strip()})
    result = analyze(payload, tiles)
    write_outputs(args.output_dir, result)
    print((args.output_dir / "VERDICT.md").read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
