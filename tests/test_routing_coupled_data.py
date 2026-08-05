from __future__ import annotations

from pre_qwen_certification.data import find_duplicates, jaccard, text_sha256, word_shingles
from pre_qwen_certification.routing_coupled_data import (
    generate_routing_coupled_hypothesis_documents,
    generate_routing_coupled_ood_documents,
)
from pre_qwen_certification.teacher_width_data import generate_width_documents


def _documents():
    train = generate_width_documents(
        split="teacher-width-train-v1", documents=28, seed=24017
    )
    hypothesis = generate_routing_coupled_hypothesis_documents(
        split="routing-coupled-hypothesis-v4-confirmation",
        documents=20,
        seed=611953,
    )
    ood = generate_routing_coupled_ood_documents(
        split="routing-coupled-ood-v4-confirmation"
    )
    return train, hypothesis, ood


def test_v4_heldouts_fit_the_frozen_training_vocabulary() -> None:
    train, hypothesis, ood = _documents()
    vocabulary = {character for document in train for character in document.text}
    missing = {
        character for document in hypothesis + ood for character in document.text
    } - vocabulary
    assert not missing, f"held-out characters absent from frozen vocabulary: {sorted(missing)!r}"


def test_v4_document_ids_are_unique_and_splits_are_disjoint() -> None:
    train, hypothesis, ood = _documents()
    groups = [
        {document.document_id for document in train},
        {document.document_id for document in hypothesis},
        {document.document_id for document in ood},
    ]
    assert all(groups)
    assert len(groups[0] | groups[1] | groups[2]) == sum(map(len, groups))


def test_v4_has_no_exact_or_threshold_near_duplicates() -> None:
    train, hypothesis, ood = _documents()
    documents = train + hypothesis + ood
    assert not find_duplicates(documents, near_duplicate_threshold=0.70)


def test_v4_cross_split_audit_is_below_threshold() -> None:
    train, hypothesis, ood = _documents()
    groups = {"train": train, "hypothesis": hypothesis, "ood": ood}
    labels = list(groups)
    maximum = 0.0
    for left_index, left_label in enumerate(labels):
        for right_label in labels[left_index + 1 :]:
            for left in groups[left_label]:
                left_hash = text_sha256(left.text)
                left_shingles = word_shingles(left.text)
                for right in groups[right_label]:
                    assert left_hash != text_sha256(right.text)
                    maximum = max(maximum, jaccard(left_shingles, word_shingles(right.text)))
    assert maximum < 0.70
