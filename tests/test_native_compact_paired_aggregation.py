from __future__ import annotations

import pytest

from scripts import aggregate_native_compact_gate_2a as aggregate
from scripts import audit_native_compact_gate_2a as audit


def rows(values: list[tuple[int, str, int, float]]) -> list[dict[str, object]]:
    return [
        {
            "seed": seed,
            "document_id": document_id,
            "start": start,
            "loss": loss,
        }
        for seed, document_id, start, loss in values
    ]


def test_duplicate_chunk_windows_are_aggregated_at_article_level() -> None:
    left = rows(
        [
            (1, "article-a", 0, 1.0),
            (1, "article-a", 0, 3.0),
            (1, "article-a", 7, 5.0),
            (1, "article-b", 0, 8.0),
        ]
    )
    right = rows(
        [
            (1, "article-a", 0, 2.0),
            (1, "article-a", 0, 2.0),
            (1, "article-a", 7, 2.0),
            (1, "article-b", 0, 5.0),
        ]
    )
    expected = [
        {
            "seed": 1,
            "document_id": "article-a",
            "start": 0,
            "difference": 1.0,
        },
        {
            "seed": 1,
            "document_id": "article-b",
            "start": 0,
            "difference": 3.0,
        },
    ]
    assert aggregate.paired_rows(left, right) == expected
    assert audit.paired_rows(left, right) == expected


def test_article_pairing_rejects_missing_statistical_units() -> None:
    left = rows([(1, "article-a", 0, 1.0)])
    right = rows([(1, "article-b", 0, 1.0)])
    with pytest.raises(ValueError, match="paired article mismatch"):
        aggregate.paired_rows(left, right)
    with pytest.raises(ValueError, match="paired article mismatch"):
        audit.paired_rows(left, right)
