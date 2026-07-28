from __future__ import annotations

from pre_qwen_certification.metrics import crossed_hierarchical_bootstrap


def test_crossed_bootstrap_uses_seed_document_cells_and_is_reproducible() -> None:
    records = []
    for seed in (11, 22, 33):
        for document in ("a", "b", "c", "d"):
            base = seed / 1000.0 + (ord(document) - ord("a")) / 100.0
            for repeated_window in range(5):
                records.append(
                    {
                        "seed": seed,
                        "document_id": document,
                        "value": base + repeated_window / 10000.0,
                    }
                )
    first = crossed_hierarchical_bootstrap(
        records,
        value_key="value",
        samples=500,
        random_seed=991,
    )
    second = crossed_hierarchical_bootstrap(
        records,
        value_key="value",
        samples=500,
        random_seed=991,
    )
    assert first == second
    assert first["effective_cells"] == 12
    assert first["lcb"] <= first["mean"] <= first["ucb"]
