from __future__ import annotations

import torch

from pre_qwen_certification.metrics import tensor_metrics
from pre_qwen_certification.modal import (
    ConventionalSwiGLUMoE,
    MoEGeometry,
    NeuronwiseModalMoE,
    ScalarModalMoE,
    set_seed,
)
from pre_qwen_certification.synthetic import make_modal_teacher


def test_scalar_direct_execution_matches_explicit_reconstruction() -> None:
    set_seed(17)
    geometry = MoEGeometry(d_model=12, d_ff=20, n_experts=7, top_k=3)
    model = ScalarModalMoE(geometry, rank=2)
    inputs = torch.randn(23, geometry.d_model)
    direct, routing = model(inputs)
    reconstructed, _ = model.reference_reconstructed(
        inputs,
        forced_top_ids=routing.top_ids,
        forced_weights=routing.weights,
    )
    assert tensor_metrics(direct, reconstructed).nrmse < 1e-6


def test_neuronwise_direct_execution_matches_reconstructed_weights() -> None:
    set_seed(29)
    geometry = MoEGeometry(d_model=10, d_ff=16, n_experts=6, top_k=2)
    model = NeuronwiseModalMoE(geometry, rank=2)
    teacher = ConventionalSwiGLUMoE.from_modal(model)  # duck-typed reconstruction
    inputs = torch.randn(19, geometry.d_model)
    direct, routing = model(inputs)
    explicit, _ = teacher(
        inputs,
        forced_top_ids=routing.top_ids,
        forced_weights=routing.weights,
    )
    assert tensor_metrics(direct, explicit).nrmse < 1e-6


def test_svd_recovers_hidden_scalar_rank_and_rejects_lower_rank() -> None:
    geometry = MoEGeometry(d_model=10, d_ff=14, n_experts=8, top_k=2)
    hidden = make_modal_teacher(
        geometry,
        rank=2,
        seed=104729,
        residual_scale=1.0,
        code_scale=1.2,
    )
    conventional = ConventionalSwiGLUMoE.from_modal(hidden)
    exact, _ = ScalarModalMoE.from_conventional_svd(conventional, 2)
    lower, _ = ScalarModalMoE.from_conventional_svd(conventional, 1)
    inputs = torch.randn(97, geometry.d_model, generator=torch.Generator().manual_seed(91))
    target, routing = conventional(inputs)
    exact_output, _ = exact(
        inputs,
        forced_top_ids=routing.top_ids,
        forced_weights=routing.weights,
    )
    lower_output, _ = lower(
        inputs,
        forced_top_ids=routing.top_ids,
        forced_weights=routing.weights,
    )
    assert tensor_metrics(exact_output, target).nrmse < 1e-5
    assert tensor_metrics(lower_output, target).nrmse > 0.02


def test_asymmetric_scalar_modal_matches_reconstructed_reference() -> None:
    from pre_qwen_certification.modal import AsymmetricScalarModalMoE

    set_seed(73)
    geometry = MoEGeometry(d_model=12, d_ff=20, n_experts=7, top_k=3)
    module = AsymmetricScalarModalMoE(geometry, ranks=(1, 3, 2))
    inputs = torch.randn(11, geometry.d_model)
    direct, routing = module(inputs)
    reference, reference_routing = module.reference_reconstructed(inputs)
    assert torch.equal(routing.top_ids, reference_routing.top_ids)
    assert torch.allclose(routing.weights, reference_routing.weights)
    assert torch.allclose(direct, reference, atol=3e-5, rtol=3e-5)
    assert module.dominant_matrix_compute_ratio() == 1.0
    assert module.idealized_expert_compute_ratio() == 1.0 + 6 / (3 * 12)


def test_residual_scalar_modal_matches_reconstructed_reference() -> None:
    from pre_qwen_certification.modal import ResidualScalarModalMoE

    set_seed(101)
    geometry = MoEGeometry(d_model=12, d_ff=20, n_experts=7, top_k=3)
    module = ResidualScalarModalMoE(geometry, rank=1, residual_rank=2)
    with torch.no_grad():
        for parameter in (
            module.gate_residual_left,
            module.gate_residual_right,
            module.up_residual_left,
            module.up_residual_right,
            module.down_residual_left,
            module.down_residual_right,
        ):
            parameter.normal_(std=0.05)
    inputs = torch.randn(13, geometry.d_model)
    direct, routing = module(inputs)
    reference, reference_routing = module.reference_reconstructed(inputs)
    assert torch.equal(routing.top_ids, reference_routing.top_ids)
    assert torch.allclose(routing.weights, reference_routing.weights)
    assert torch.allclose(direct, reference, atol=4e-5, rtol=4e-5)


def test_clustered_residual_matches_reconstructed_reference() -> None:
    from pre_qwen_certification.modal import ClusteredResidualMoE

    set_seed(151)
    geometry = MoEGeometry(d_model=12, d_ff=20, n_experts=8, top_k=3)
    mapping = torch.tensor([0, 0, 1, 1, 2, 2, 1, 0])
    module = ClusteredResidualMoE(
        geometry, n_groups=3, residual_rank=2, expert_to_group=mapping
    )
    with torch.no_grad():
        for parameter in (
            module.gate_residual_left,
            module.gate_residual_right,
            module.up_residual_left,
            module.up_residual_right,
            module.down_residual_left,
            module.down_residual_right,
        ):
            parameter.normal_(std=0.04)
    inputs = torch.randn(17, geometry.d_model)
    direct, routing = module(inputs)
    reference, reference_routing = module.reference_reconstructed(inputs)
    assert torch.equal(routing.top_ids, reference_routing.top_ids)
    assert torch.allclose(routing.weights, reference_routing.weights)
    assert torch.allclose(direct, reference, atol=5e-5, rtol=5e-5)
    assert module.dominant_matrix_compute_ratio() == 1.0


def test_clustered_initialization_uses_all_groups() -> None:
    from pre_qwen_certification.modal import ClusteredResidualMoE

    set_seed(157)
    geometry = MoEGeometry(d_model=10, d_ff=14, n_experts=8, top_k=2)
    teacher = ConventionalSwiGLUMoE(geometry)
    calibration = torch.randn(128, geometry.d_model)
    student, metadata = ClusteredResidualMoE.from_conventional_grouped(
        teacher,
        n_groups=3,
        residual_rank=1,
        calibration_inputs=calibration,
    )
    assert torch.unique(student.expert_to_group).numel() == 3
    assert sum(metadata["group_sizes"]) == geometry.n_experts


def test_selective_hot_cold_matches_reconstructed_reference() -> None:
    from pre_qwen_certification.selective import SelectiveHotColdMoE

    set_seed(173)
    geometry = MoEGeometry(d_model=12, d_ff=18, n_experts=8, top_k=3)
    teacher = ConventionalSwiGLUMoE(geometry)
    module, metadata = SelectiveHotColdMoE.from_conventional(
        teacher,
        hot_expert_ids=torch.tensor([1, 4, 7]),
        cold_rank=1,
    )
    inputs = torch.randn(29, geometry.d_model)
    direct, routing = module(inputs)
    reference, reference_routing = module.reference_reconstructed(
        inputs,
        forced_top_ids=routing.top_ids,
        forced_weights=routing.weights,
    )
    assert torch.equal(routing.top_ids, reference_routing.top_ids)
    assert torch.allclose(routing.weights, reference_routing.weights)
    assert torch.allclose(direct, reference, atol=5e-5, rtol=5e-5)
    assert metadata["hot_expert_ids"] == [1, 4, 7]
    assert module.hot_gate.requires_grad is False
    assert module.expert_transform_parameter_count() < (
        teacher.gate.numel() + teacher.up.numel() + teacher.down.numel()
    )


def test_selective_route_cost_accounts_for_hot_and_cold_paths() -> None:
    from pre_qwen_certification.selective import SelectiveHotColdMoE

    geometry = MoEGeometry(d_model=8, d_ff=12, n_experts=6, top_k=2)
    module = SelectiveHotColdMoE(
        geometry,
        hot_expert_ids=torch.tensor([0, 1]),
        cold_rank=1,
    )
    top_ids = torch.tensor([[0, 1], [0, 2], [3, 4], [1, 5]])
    metrics = module.route_cost_metrics(top_ids)
    # Per-token dominant ratios: 1.0, 1.5, 1.0, 1.5.
    assert abs(metrics["dominant_matrix_compute_ratio"] - 1.25) < 1e-12
    assert abs(metrics["mean_hot_slots"] - 1.0) < 1e-12
    assert abs(metrics["mean_cold_slots"] - 1.0) < 1e-12


def test_weighted_output_energy_identifies_high_contribution_expert() -> None:
    from pre_qwen_certification.selective import (
        choose_hot_experts,
        score_expert_importance,
    )

    geometry = MoEGeometry(d_model=4, d_ff=6, n_experts=4, top_k=2)
    teacher = ConventionalSwiGLUMoE(geometry)
    with torch.no_grad():
        teacher.gate.fill_(0.05)
        teacher.up.fill_(0.05)
        teacher.down.fill_(0.05)
        teacher.gate[2].fill_(0.8)
        teacher.up[2].fill_(0.8)
        teacher.down[2].fill_(0.8)
    inputs = torch.ones(20, geometry.d_model)
    pattern = torch.tensor([[0, 1], [2, 3]]).repeat(10, 1)
    weights = torch.full((20, 2), 0.5)
    importance = score_expert_importance(teacher, inputs, pattern, weights)
    selected = choose_hot_experts(
        importance,
        hot_count=1,
        metric="weighted-output-energy",
        seed=0,
    )
    assert selected.tolist() == [2]
    assert max(importance.routing_frequency) - min(importance.routing_frequency) < 1e-12


def test_selective_importance_chunking_is_stable() -> None:
    from pre_qwen_certification.selective import score_expert_importance

    set_seed(181)
    geometry = MoEGeometry(d_model=6, d_ff=10, n_experts=5, top_k=2)
    teacher = ConventionalSwiGLUMoE(geometry)
    inputs = torch.randn(37, geometry.d_model)
    _, routing = teacher(inputs)
    one = score_expert_importance(teacher, inputs, routing.top_ids, routing.weights, chunk_size=37)
    many = score_expert_importance(teacher, inputs, routing.top_ids, routing.weights, chunk_size=7)
    for key in one.as_dict():
        assert torch.allclose(
            torch.tensor(one.as_dict()[key]),
            torch.tensor(many.as_dict()[key]),
            atol=1e-9,
            rtol=1e-6,
        )


def test_selective_diagnostic_rank_and_jaccard_helpers() -> None:
    from pre_qwen_certification.selective_diagnostics import jaccard, spearman_rank_correlation

    assert abs(spearman_rank_correlation([1, 2, 3], [10, 20, 30]) - 1.0) < 1e-12
    assert abs(spearman_rank_correlation([1, 2, 3], [30, 20, 10]) + 1.0) < 1e-12
    assert abs(jaccard([1, 2], [2, 3]) - 1 / 3) < 1e-12
