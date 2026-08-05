from __future__ import annotations

import torch

from pre_qwen_certification.heterogeneous_rank import HeterogeneousSharedLowRankResidualMoE
from pre_qwen_certification.modal import ConventionalSwiGLUMoE
from pre_qwen_certification.native_compact import (
    CANDIDATES,
    FULL,
    NARROW,
    PRIMARY,
    NativeArchitectureSpec,
    build_paired_candidate_models,
    candidate_accounting,
    route_health,
)
from pre_qwen_certification.tiny_lm import TinyLMConfig


def tiny_config() -> TinyLMConfig:
    return TinyLMConfig(
        seq_len=8,
        batch_size=2,
        d_model=8,
        n_heads=2,
        n_layers=2,
        d_ff=12,
        n_experts=4,
        top_k=2,
        teacher_steps=2,
        student_steps=0,
        learning_rate=1e-3,
        student_learning_rate=1e-3,
        weight_decay=0.0,
        aux_weight=0.01,
        grad_clip=1.0,
    )


def test_paired_candidates_share_non_moe_initialization_and_router() -> None:
    config = tiny_config()
    spec = NativeArchitectureSpec(native_rank=2, narrow_d_ff=8)
    models = build_paired_candidate_models(32, config, spec, seed=7717)
    assert tuple(models) == CANDIDATES
    reference = models[FULL]
    for candidate in CANDIDATES:
        model = models[candidate]
        assert torch.equal(model.token_embedding.weight, reference.token_embedding.weight)
        assert torch.equal(model.position_embedding.weight, reference.position_embedding.weight)
        assert torch.equal(model.blocks[0].attention.in_proj_weight, reference.blocks[0].attention.in_proj_weight)
        assert torch.equal(model.blocks[0].moe.router.weight, reference.blocks[0].moe.router.weight)
    assert isinstance(models[FULL].blocks[0].moe, ConventionalSwiGLUMoE)
    assert isinstance(models[NARROW].blocks[0].moe, ConventionalSwiGLUMoE)
    assert isinstance(models[PRIMARY].blocks[0].moe, HeterogeneousSharedLowRankResidualMoE)


def test_candidates_execute_and_accounting_matches_closed_form() -> None:
    config = tiny_config()
    spec = NativeArchitectureSpec(native_rank=2, narrow_d_ff=8)
    models = build_paired_candidate_models(32, config, spec, seed=7723)
    tokens = torch.randint(0, 32, (2, config.seq_len))
    for model in models.values():
        logits, auxiliary, _ = model(tokens)
        assert logits.shape == (2, config.seq_len, 32)
        assert auxiliary.ndim == 0
        assert torch.isfinite(logits).all()
    accounting = candidate_accounting(models, config)
    assert accounting[FULL].expert_parameter_ratio == 1.0
    assert accounting[FULL].expert_compute_ratio == 1.0
    assert abs(accounting[NARROW].expert_parameter_ratio - 8 / 12) < 1e-12
    assert abs(accounting[NARROW].expert_compute_ratio - 8 / 12) < 1e-12
    expected_native_parameters = 1 / config.n_experts + (
        (config.d_ff + config.d_model) * spec.native_rank
        / (config.d_ff * config.d_model)
    )
    expected_native_compute = 1 / config.top_k + (
        (config.d_ff + config.d_model) * spec.native_rank
        / (config.d_ff * config.d_model)
    )
    assert abs(accounting[PRIMARY].expert_parameter_ratio - expected_native_parameters) < 1e-12
    assert abs(accounting[PRIMARY].expert_compute_ratio - expected_native_compute) < 1e-12


def test_route_health_detects_dead_experts() -> None:
    health = route_health([1.0, 1.0, 0.0, 0.0], top_k=2)
    assert health["dead_experts"] == 2
    assert health["maximum_frequency"] == 1.0
    assert health["minimum_frequency"] == 0.0
    assert 0.0 < float(health["normalized_entropy"]) < 1.0
