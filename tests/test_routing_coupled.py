from __future__ import annotations

import copy

import torch

from pre_qwen_certification.alignment_tolerant import SharedLowRankResidualMoE
from pre_qwen_certification.modal import ConventionalSwiGLUMoE, MoEGeometry
from pre_qwen_certification.routing_coupled import (
    RoutingCoupledResidualMoE,
    coupled_routing_set_loss,
)
from pre_qwen_certification.routing_set_distillation import (
    RoutingSetWeights,
    conventional_routed_expert_outputs,
)


def _fixture() -> tuple[ConventionalSwiGLUMoE, SharedLowRankResidualMoE, RoutingCoupledResidualMoE]:
    torch.manual_seed(17)
    geometry = MoEGeometry(d_model=8, d_ff=10, n_experts=6, top_k=3)
    teacher = ConventionalSwiGLUMoE(geometry)
    base = SharedLowRankResidualMoE.from_teacher(teacher, rank=3)
    coupled = RoutingCoupledResidualMoE.from_base(
        base, set_dim=4, hidden_dim=5, use_second_moment=True
    )
    return teacher, base, coupled


def test_zero_initialized_coupling_is_exactly_the_base() -> None:
    _, base, coupled = _fixture()
    x = torch.randn(13, 8)
    top_ids = torch.tensor([[0, 2, 4]] * len(x))
    weights = torch.softmax(torch.randn(len(x), 3), dim=-1)
    expected, _ = base(x, forced_top_ids=top_ids, forced_weights=weights)
    actual, _ = coupled(x, forced_top_ids=top_ids, forced_weights=weights)
    torch.testing.assert_close(actual, expected, atol=1e-7, rtol=1e-7)


def test_coupled_execution_is_invariant_to_consistent_slot_permutation() -> None:
    _, _, coupled = _fixture()
    torch.manual_seed(23)
    coupled.coupling_out.data.normal_(std=0.15)
    x = torch.randn(11, 8)
    top_ids = torch.tensor([[0, 2, 4]] * len(x))
    weights = torch.softmax(torch.randn(len(x), 3), dim=-1)
    original, _ = coupled(x, forced_top_ids=top_ids, forced_weights=weights)
    permutation = torch.tensor([2, 0, 1])
    permuted, _ = coupled(
        x,
        forced_top_ids=top_ids[:, permutation],
        forced_weights=weights[:, permutation],
    )
    torch.testing.assert_close(permuted, original, atol=2e-6, rtol=2e-6)


def test_second_moment_can_distinguish_sets_with_equal_mean() -> None:
    _, _, coupled = _fixture()
    coupled.use_second_moment = True
    coupled.expert_alignment.data.zero_()
    coupled.expert_embedding.data.zero_()
    coupled.route_direction.data.zero_()
    coupled.coupling_in.data.zero_()
    coupled.coupling_out.data.zero_()
    # Select the first second-moment coordinate and map it to output dimension 0.
    coupled.coupling_in.data[0, coupled.set_dim] = 1.0
    coupled.coupling_out.data[0, 0] = 1.0
    weights = torch.tensor([[0.5, 0.5, 0.0]])
    a = torch.zeros(1, 3, coupled.set_dim)
    b = torch.zeros_like(a)
    a[0, 0, 0], a[0, 1, 0] = 1.0, -1.0
    b[0, 0, 1], b[0, 1, 1] = 1.0, -1.0
    out_a = coupled.correction_from_aligned(a, weights)
    out_b = coupled.correction_from_aligned(b, weights)
    assert out_a[0, 0] > 0.1
    torch.testing.assert_close(out_b[0, 0], torch.tensor(0.0), atol=1e-7, rtol=0.0)


def test_coupled_loss_reaches_zero_initialized_output_projection() -> None:
    teacher, _, coupled = _fixture()
    x = torch.randn(19, 8)
    natural = teacher(x)[1]
    teacher_experts = conventional_routed_expert_outputs(teacher, x, natural.top_ids)
    weights = RoutingSetWeights(
        mixture=0.20,
        expert=0.20,
        counterfactual=0.40,
        geometry=0.10,
        cosine=0.10,
    )
    losses = coupled_routing_set_loss(
        coupled,
        x,
        natural.top_ids,
        natural.weights,
        teacher_experts,
        weights,
    )
    losses.objective.backward()
    assert coupled.coupling_out.grad is not None
    assert torch.isfinite(coupled.coupling_out.grad).all()
    assert float(coupled.coupling_out.grad.abs().sum()) > 0.0


def test_disabling_coupling_recovers_base_after_training_like_parameters() -> None:
    _, base, coupled = _fixture()
    coupled.coupling_out.data.normal_(std=0.1)
    coupled.coupling_enabled = False
    x = torch.randn(7, 8)
    top_ids = torch.tensor([[1, 3, 5]] * len(x))
    weights = torch.softmax(torch.randn(len(x), 3), dim=-1)
    actual, _ = coupled(x, forced_top_ids=top_ids, forced_weights=weights)
    expected, _ = base(x, forced_top_ids=top_ids, forced_weights=weights)
    torch.testing.assert_close(actual, expected, atol=1e-7, rtol=1e-7)


def test_q8_h8_budget_is_below_narrow65_for_current_geometry() -> None:
    torch.manual_seed(31)
    geometry = MoEGeometry(d_model=24, d_ff=40, n_experts=12, top_k=4)
    teacher = ConventionalSwiGLUMoE(geometry)
    base = SharedLowRankResidualMoE.from_teacher(teacher, rank=5)
    coupled = RoutingCoupledResidualMoE.from_base(base, set_dim=8, hidden_dim=8)
    accounting = coupled.accounting()
    assert abs(accounting.parameter_ratio - (15304 / 34560)) < 1e-12
    assert abs(accounting.compute_ratio - 0.625) < 1e-12
    assert accounting.parameter_ratio < 0.65
    assert accounting.compute_ratio < 0.65


def test_mean_only_and_second_moment_start_from_identical_function() -> None:
    _, _, coupled = _fixture()
    mean_only = copy.deepcopy(coupled)
    mean_only.use_second_moment = False
    x = torch.randn(9, 8)
    top_ids = torch.tensor([[0, 1, 2]] * len(x))
    weights = torch.softmax(torch.randn(len(x), 3), dim=-1)
    full, _ = coupled(x, forced_top_ids=top_ids, forced_weights=weights)
    mean, _ = mean_only(x, forced_top_ids=top_ids, forced_weights=weights)
    torch.testing.assert_close(full, mean, atol=0.0, rtol=0.0)
