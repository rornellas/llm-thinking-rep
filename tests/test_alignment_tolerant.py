from __future__ import annotations

import torch

from pre_qwen_certification.alignment_data import (
    generate_alignment_hypothesis_documents,
    generate_alignment_ood_documents,
)
from pre_qwen_certification.alignment_tolerant import SharedLowRankResidualMoE
from pre_qwen_certification.modal import ConventionalSwiGLUMoE, MoEGeometry
from pre_qwen_certification.teacher_width_data import generate_width_documents
from pre_qwen_certification.tiny_lm import CharacterCorpus


def test_alignment_tolerant_direct_matches_reconstructed() -> None:
    torch.manual_seed(701)
    geometry = MoEGeometry(d_model=24, d_ff=40, n_experts=12, top_k=4)
    teacher = ConventionalSwiGLUMoE(geometry)
    student = SharedLowRankResidualMoE.from_teacher(teacher, rank=5)
    inputs = torch.randn(19, geometry.d_model)
    direct, direct_routing = student(inputs)
    reference, reference_routing = student.reference_reconstructed(inputs)
    torch.testing.assert_close(direct, reference, rtol=2e-5, atol=2e-6)
    assert torch.equal(direct_routing.top_ids, reference_routing.top_ids)
    assert student.router.weight.requires_grad is False


def test_alignment_tolerant_forced_routing_matches_reconstructed() -> None:
    torch.manual_seed(709)
    geometry = MoEGeometry(d_model=8, d_ff=13, n_experts=7, top_k=3)
    teacher = ConventionalSwiGLUMoE(geometry)
    student = SharedLowRankResidualMoE.from_teacher(teacher, rank=4)
    inputs = torch.randn(11, geometry.d_model)
    _, routing = teacher(inputs)
    direct, _ = student(
        inputs,
        forced_top_ids=routing.top_ids,
        forced_weights=routing.weights,
    )
    reference, _ = student.reference_reconstructed(
        inputs,
        forced_top_ids=routing.top_ids,
        forced_weights=routing.weights,
    )
    torch.testing.assert_close(direct, reference, rtol=2e-5, atol=2e-6)


def test_alignment_tolerant_accounting_matches_preregistered_rank5() -> None:
    geometry = MoEGeometry(d_model=24, d_ff=40, n_experts=12, top_k=4)
    accounting = SharedLowRankResidualMoE(geometry, rank=5).accounting()
    assert abs(accounting.parameter_ratio - 5.0 / 12.0) < 1e-12
    assert abs(accounting.compute_ratio - 7.0 / 12.0) < 1e-12


def test_svd_initialization_error_is_monotone_in_rank() -> None:
    torch.manual_seed(719)
    geometry = MoEGeometry(d_model=12, d_ff=18, n_experts=6, top_k=2)
    teacher = ConventionalSwiGLUMoE(geometry)
    errors: list[float] = []
    for rank in (2, 4, 6):
        student = SharedLowRankResidualMoE.from_teacher(teacher, rank=rank)
        reconstructed = student.reconstruct_weights()
        teacher_banks = (teacher.gate, teacher.up, teacher.down)
        error = sum(
            float(torch.mean((estimate - target).square()).detach())
            for estimate, target in zip(reconstructed, teacher_banks, strict=True)
        )
        errors.append(error)
    assert errors[2] <= errors[1] <= errors[0]


def test_fresh_alignment_holdouts_fit_frozen_training_vocabulary() -> None:
    train = generate_width_documents(
        split="teacher-width-train-v1", documents=28, seed=24017
    )
    training_corpus = CharacterCorpus({"train": train}, seq_len=48)
    hypothesis = generate_alignment_hypothesis_documents(
        split="alignment-hypothesis-v1", documents=20, seed=89177
    )
    ood = generate_alignment_ood_documents(split="alignment-ood-v1")
    CharacterCorpus(
        {"hypothesis": hypothesis, "ood": ood},
        seq_len=48,
        vocabulary=training_corpus.itos,
    )
    assert not ({doc.document_id for doc in train} & {doc.document_id for doc in hypothesis})
    assert not ({doc.document_id for doc in train} & {doc.document_id for doc in ood})
