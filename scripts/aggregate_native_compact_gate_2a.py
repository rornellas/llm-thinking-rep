#!/usr/bin/env python3
"""Aggregation entry point with article-level paired comparison semantics."""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import aggregate_native_compact_gate_2a_impl as runner


def paired_rows(
    left: Sequence[Mapping[str, Any]],
    right: Sequence[Mapping[str, Any]],
    key: str = "loss",
) -> list[dict[str, Any]]:
    """Average windows/chunks within the preregistered seed×article unit, then pair."""

    def aggregate(rows: Sequence[Mapping[str, Any]]) -> dict[tuple[int, str], float]:
        buckets: dict[tuple[int, str], list[float]] = defaultdict(list)
        for row in rows:
            buckets[(int(row["seed"]), str(row["document_id"]))].append(float(row[key]))
        return {cell: float(np.mean(values)) for cell, values in buckets.items()}

    left_index = aggregate(left)
    right_index = aggregate(right)
    if set(left_index) != set(right_index):
        missing_left = sorted(set(right_index) - set(left_index))
        missing_right = sorted(set(left_index) - set(right_index))
        raise ValueError(
            f"paired article mismatch: missing_left={missing_left[:5]}, missing_right={missing_right[:5]}"
        )
    return [
        {
            "seed": seed,
            "document_id": document_id,
            "start": 0,
            "difference": float(left_index[(seed, document_id)] - right_index[(seed, document_id)]),
        }
        for seed, document_id in sorted(left_index)
    ]


runner.paired_rows = paired_rows

if __name__ == "__main__":
    raise SystemExit(runner.main())
