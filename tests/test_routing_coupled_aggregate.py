from __future__ import annotations

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


def test_routing_aggregate_error_matches_direct_mixture_error() -> None:
    torch.manual_seed(211)
    geometry = MoEGeometry(d_model=8, d_ff=10, n_experts=6, top_k=3)
    teacher = ConventionalSwiGLUMoE(geometry)
    base = SharedLowRankResidualMoE.from_teacher(teacher, rank=3)
    student = RoutingCoupledResidualMoE.from_base(base, set_dim=4, hidden_dim=5)
    student.coupling_out.data.normal_(std=0.1)
    x = torch.randn(23, geometry.d_model)
    _, routing = teacher(x)
    teacher_experts = conventional_routed_expert_outputs(
        teacher, x, routing.top_ids
    )
    student_experts, aligned = student.routed_components(
        x, forced_top_ids=routing.top_ids
    )
    student_mix, _ = student.mix_from_components(
        student_experts, aligned, routing.weights
    )
    teacher_mix = torch.einsum("nt,ntd->nd", routing.weights, teacher_experts)
    # The diagnostic normalizes each token by its teacher mixture energy before
    # averaging. This remains a direct, permutation-invariant function of the
    # final mixture and matches the historical routing-set definition.
    error_energy = (student_mix - teacher_mix).square().sum(dim=-1)
    teacher_energy = teacher_mix.square().sum(dim=-1).clamp_min(1e-8)
    direct = torch.mean(error_energy / teacher_energy)
    weights = RoutingSetWeights(
        mixture=0.2,
        expert=0.2,
        counterfactual=0.4,
        geometry=0.1,
        cosine=0.1,
    )
    losses = coupled_routing_set_loss(
        student,
        x,
        routing.top_ids,
        routing.weights,
        teacher_experts,
        weights,
    )
    torch.testing.assert_close(losses.aggregate_error, direct, atol=2e-6, rtol=2e-6)


def test_cross_component_is_not_needed_to_reconstruct_aggregate_gate() -> None:
    torch.manual_seed(223)
    geometry = MoEGeometry(d_model=8, d_ff=10, n_experts=6, top_k=3)
    teacher = ConventionalSwiGLUMoE(geometry)
    base = SharedLowRankResidualMoE.from_teacher(teacher, rank=3)
    student = RoutingCoupledResidualMoE.from_base(base, set_dim=4, hidden_dim=5)
    student.coupling_out.data.normal_(std=0.2)
    x = torch.randn(17, geometry.d_model)
    _, routing = teacher(x)
    teacher_experts = conventional_routed_expert_outputs(
        teacher, x, routing.top_ids
    )
    weights = RoutingSetWeights(
        mixture=0.2,
        expert=0.2,
        counterfactual=0.4,
        geometry=0.1,
        cosine=0.1,
    )
    losses = coupled_routing_set_loss(
        student,
        x,
        routing.top_ids,
        routing.weights,
        teacher_experts,
        weights,
    )
    torch.testing.assert_close(
        losses.self_error + losses.cross_error,
        losses.aggregate_error,
        atol=2e-6,
        rtol=2e-6,
    )
