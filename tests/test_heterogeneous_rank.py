from __future__ import annotations

import torch

from pre_qwen_certification.heterogeneous_rank import (
    HeterogeneousSharedLowRankResidualMoE,
    allocate_static_ranks,
    residual_mode_utilities,
    uniform_rank_vector,
)
from pre_qwen_certification.modal import ConventionalSwiGLUMoE, MoEGeometry


def test_full_rank_reconstructs_teacher_and_forward() -> None:
    torch.manual_seed(101)
    geometry = MoEGeometry(d_model=6, d_ff=8, n_experts=4, top_k=2)
    teacher = ConventionalSwiGLUMoE(geometry)
    ranks = uniform_rank_vector(geometry.n_experts, min(geometry.d_model, geometry.d_ff))
    student = HeterogeneousSharedLowRankResidualMoE.from_teacher(teacher, ranks=ranks)
    gate, up, down = student.reconstruct_weights()
    torch.testing.assert_close(gate, teacher.gate, atol=3e-5, rtol=3e-5)
    torch.testing.assert_close(up, teacher.up, atol=3e-5, rtol=3e-5)
    torch.testing.assert_close(down, teacher.down, atol=3e-5, rtol=3e-5)
    x = torch.randn(19, geometry.d_model)
    teacher_output, routing = teacher(x)
    student_output, _ = student(
        x, forced_top_ids=routing.top_ids, forced_weights=routing.weights
    )
    torch.testing.assert_close(student_output, teacher_output, atol=5e-5, rtol=5e-5)


def test_accounting_matches_uniform_formula() -> None:
    geometry = MoEGeometry(d_model=8, d_ff=12, n_experts=4, top_k=2)
    module = HeterogeneousSharedLowRankResidualMoE(geometry, ranks=[2, 2, 2, 2])
    frequencies = torch.tensor([0.5, 0.5, 0.5, 0.5], dtype=torch.float64)
    accounting = module.accounting(frequencies, uniform_reference_rank=2)
    expected_parameters = 3 * geometry.d_ff * geometry.d_model + 3 * (
        geometry.d_ff + geometry.d_model
    ) * 8
    expected_teacher_parameters = 3 * geometry.n_experts * geometry.d_ff * geometry.d_model
    expected_macs = 3 * geometry.d_ff * geometry.d_model + 3 * (
        geometry.d_ff + geometry.d_model
    ) * 4.0
    expected_teacher_macs = 3 * geometry.top_k * geometry.d_ff * geometry.d_model
    assert accounting.expert_parameters == expected_parameters
    assert accounting.teacher_expert_parameters == expected_teacher_parameters
    assert abs(accounting.expected_routed_matrix_macs - expected_macs) < 1e-10
    assert accounting.teacher_routed_matrix_macs == expected_teacher_macs


def test_allocator_respects_both_budgets_and_can_choose_nonuniform() -> None:
    utilities = torch.tensor(
        [
            [8.0, 7.0, 1.0],
            [1.0, 0.8, 0.4],
            [1.0, 4.0, 3.0],
        ],
        dtype=torch.float64,
    )
    frequencies = torch.tensor([1.0, 0.7, 0.3], dtype=torch.float64)
    ranks = allocate_static_ranks(
        utilities,
        frequencies,
        total_rank_budget=6,
        expected_active_rank_budget=4.0,
        min_rank=1,
        max_rank=3,
    )
    assert len(ranks) == 3
    assert sum(ranks) == 6
    assert float(torch.dot(frequencies, torch.tensor(ranks, dtype=torch.float64))) <= 4.0 + 1e-10
    assert len(set(ranks)) > 1


def test_residual_utilities_are_nonnegative_and_rank_ordered() -> None:
    torch.manual_seed(113)
    geometry = MoEGeometry(d_model=6, d_ff=9, n_experts=5, top_k=2)
    teacher = ConventionalSwiGLUMoE(geometry)
    utilities = residual_mode_utilities(teacher, max_rank=5)
    assert tuple(utilities.shape) == (geometry.n_experts, 5)
    assert torch.all(utilities >= 0)
    assert torch.all(utilities[:, 1:] <= utilities[:, :-1] + 1e-10)
