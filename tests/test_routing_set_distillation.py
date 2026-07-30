from __future__ import annotations

import torch

from pre_qwen_certification.alignment_tolerant import SharedLowRankResidualMoE
from pre_qwen_certification.modal import ConventionalSwiGLUMoE, MoEGeometry, route_topk, set_seed
from pre_qwen_certification.routing_set_distillation import (
    RoutingSetWeights,
    build_counterfactual_weights,
    conventional_routed_expert_outputs,
    error_decomposition,
    routed_expert_outputs,
    routing_set_loss,
)


def _fixture():
    set_seed(123)
    geometry = MoEGeometry(d_model=12, d_ff=16, n_experts=7, top_k=3)
    teacher = ConventionalSwiGLUMoE(geometry)
    student = SharedLowRankResidualMoE.from_teacher(teacher, rank=3)
    x = torch.randn(11, geometry.d_model)
    routing = route_topk(x, teacher.router.weight, geometry.top_k)
    objective = RoutingSetWeights(
        mixture=0.25,
        expert=0.25,
        counterfactual=0.30,
        geometry=0.10,
        cosine=0.10,
    )
    return geometry, teacher, student, x, routing, objective


def test_factorized_expert_outputs_reconstruct_forward():
    _, _, student, x, routing, _ = _fixture()
    experts = student.routed_expert_outputs(x, forced_top_ids=routing.top_ids)
    direct, _ = student(
        x, forced_top_ids=routing.top_ids, forced_weights=routing.weights
    )
    mixed = torch.einsum("nt,ntd->nd", routing.weights, experts)
    assert torch.max(torch.abs(direct - mixed)).item() < 2e-5


def test_conventional_expert_outputs_reconstruct_forward():
    _, teacher, _, x, routing, _ = _fixture()
    experts = conventional_routed_expert_outputs(teacher, x, routing.top_ids)
    direct, _ = teacher(
        x, forced_top_ids=routing.top_ids, forced_weights=routing.weights
    )
    mixed = torch.einsum("nt,ntd->nd", routing.weights, experts)
    assert torch.max(torch.abs(direct - mixed)).item() < 2e-5


def test_counterfactual_probes_are_normalized_and_nontrivial():
    _, _, _, _, routing, _ = _fixture()
    probes = build_counterfactual_weights(routing.weights)
    assert probes.shape[0] == routing.weights.shape[0]
    assert probes.shape[2] == routing.weights.shape[1]
    assert probes.shape[1] == 1 + routing.weights.shape[1] + 3
    assert torch.allclose(probes.sum(dim=-1), torch.ones_like(probes[..., 0]))
    assert torch.all(probes >= 0.0)


def test_zero_error_zeroes_every_scientific_loss():
    _, teacher, _, x, routing, objective = _fixture()
    experts = routed_expert_outputs(teacher, x, routing.top_ids)
    losses = routing_set_loss(experts, experts.clone(), routing.weights, objective)
    assert losses.objective.item() < 1e-8
    assert losses.mixture.item() < 1e-8
    assert losses.expert.item() < 1e-8
    assert losses.counterfactual.item() < 1e-8
    assert losses.geometry.item() < 1e-8
    assert abs(losses.cross_error.item()) < 1e-8


def test_error_decomposition_identity():
    _, teacher, student, x, routing, _ = _fixture()
    t = routed_expert_outputs(teacher, x, routing.top_ids)
    s = routed_expert_outputs(student, x, routing.top_ids)
    self_error, cross_error, aggregate = error_decomposition(s, t, routing.weights)
    assert abs((self_error + cross_error - aggregate).item()) < 1e-6


def test_objective_is_invariant_to_consistent_slot_permutation():
    _, teacher, student, x, routing, objective = _fixture()
    t = routed_expert_outputs(teacher, x, routing.top_ids)
    s = routed_expert_outputs(student, x, routing.top_ids)
    original = routing_set_loss(s, t, routing.weights, objective)
    permutation = torch.tensor([2, 0, 1])
    permuted = routing_set_loss(
        s[:, permutation], t[:, permutation], routing.weights[:, permutation], objective
    )
    for key in ("objective", "mixture", "expert", "counterfactual", "geometry", "cosine"):
        left = getattr(original, key)
        right = getattr(permuted, key)
        assert torch.allclose(left, right, atol=2e-6), key


def test_counterfactual_term_detects_cancelling_natural_mixture_error():
    _, teacher, _, x, routing, objective = _fixture()
    t = routed_expert_outputs(teacher, x, routing.top_ids)
    s = t.clone()
    direction = torch.randn_like(s[:, 0])
    # For the first two expert slots, construct opposite errors that cancel under
    # the natural routing weights but are exposed by pair/leave-one-out probes.
    w0 = routing.weights[:, 0].clamp_min(1e-4)
    w1 = routing.weights[:, 1].clamp_min(1e-4)
    s[:, 0] += direction
    s[:, 1] -= (w0 / w1)[:, None] * direction
    losses = routing_set_loss(s, t, routing.weights, objective)
    assert losses.counterfactual.item() > losses.mixture.item() * 5.0


def test_fresh_routing_set_documents_fit_frozen_train_vocabulary():
    from pre_qwen_certification.routing_set_data import (
        generate_routing_set_hypothesis_documents,
        generate_routing_set_ood_documents,
    )
    from pre_qwen_certification.teacher_width_data import generate_width_documents
    from pre_qwen_certification.tiny_lm import CharacterCorpus

    train = generate_width_documents(
        split="teacher-width-train-v1", documents=28, seed=24017
    )
    train_corpus = CharacterCorpus({"train": train}, seq_len=48)
    hypothesis = generate_routing_set_hypothesis_documents(
        split="routing-set-hypothesis-v3-confirmation", documents=20, seed=286319
    )
    ood = generate_routing_set_ood_documents(split="routing-set-ood-v3-confirmation")
    CharacterCorpus(
        {"hypothesis": hypothesis, "ood": ood},
        seq_len=48,
        vocabulary=train_corpus.itos,
    )
