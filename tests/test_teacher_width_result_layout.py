from __future__ import annotations

from scripts.aggregate_teacher_width_fresh import seed_payload_path as aggregate_seed_payload_path
from scripts.audit_teacher_width_fresh import seed_payload_path as audit_seed_payload_path


def test_teacher_width_seed_payload_resolver_prefers_canonical_nested_layout(tmp_path):
    seed = 91121
    flat = tmp_path / f"seed-{seed}.json"
    flat.write_text("{}\n", encoding="utf-8")
    nested = tmp_path / f"seed-{seed}" / f"seed-{seed}.json"
    nested.parent.mkdir()
    nested.write_text("{}\n", encoding="utf-8")

    assert aggregate_seed_payload_path(tmp_path, seed) == nested
    assert audit_seed_payload_path(tmp_path, seed) == nested


def test_teacher_width_seed_payload_resolver_supports_legacy_flat_layout(tmp_path):
    seed = 92129
    flat = tmp_path / f"seed-{seed}.json"
    flat.write_text("{}\n", encoding="utf-8")

    assert aggregate_seed_payload_path(tmp_path, seed) == flat
    assert audit_seed_payload_path(tmp_path, seed) == flat
