"""Coupled routing-set distillation for alignment-tolerant MoE students.

The v1 objective supervised only the naturally weighted MoE mixture. The v2
objective added explicit expert-wise fidelity. Both can still learn error fields
whose geometry changes under alternative expert mixtures. This module supervises
an entire deterministic set of counterfactual mixtures over each routed expert
set. The resulting quadratic form constrains both expert errors and their cross
terms without rewarding arbitrarily negative covariance.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Mapping

import torch
from torch import nn
import torch.nn.functional as F

from .alignment_tolerant import SharedLowRankResidualMoE
from .modal import ConventionalSwiGLUMoE
from .tiny_lm import (
    CapturedLayerDataset,
    CharacterCorpus,
    TinyMoELanguageModel,
    install_student,
)

EPS = 1e-8


@dataclass(frozen=True)
class RoutingSetWeights:
    mixture: float
    expert: float
    counterfactual: float
    geometry: float
    cosine: float

    @classmethod
    def from_mapping(cls, values: Mapping[str, float]) -> "RoutingSetWeights":
        required = ("mixture", "expert", "counterfactual", "geometry", "cosine")
        missing = [name for name in required if name not in values]
        if missing:
            raise ValueError(f"missing routing-set weights: {missing}")
        result = cls(**{name: float(values[name]) for name in required})
        if any(value < 0.0 for value in result.as_dict().values()):
            raise ValueError("routing-set weights must be non-negative")
        if abs(sum(result.as_dict().values()) - 1.0) > 1e-7:
            raise ValueError("routing-set weights must sum to one")
        return result

    def as_dict(self) -> dict[str, float]:
        return {
            "mixture": self.mixture,
            "expert": self.expert,
            "counterfactual": self.counterfactual,
            "geometry": self.geometry,
            "cosine": self.cosine,
        }


@dataclass(frozen=True)
class JointRoutingSetWeights:
    local: float
    kl: float
    ce: float

    @classmethod
    def from_mapping(cls, values: Mapping[str, float]) -> "JointRoutingSetWeights":
        result = cls(
            local=float(values["local"]),
            kl=float(values["kl"]),
            ce=float(values["ce"]),
        )
        if min(result.local, result.kl, result.ce) < 0.0:
            raise ValueError("joint weights must be non-negative")
        if abs(result.local + result.kl + result.ce - 1.0) > 1e-7:
            raise ValueError("joint weights must sum to one")
        return result


@dataclass
class RoutingSetLosses:
    objective: torch.Tensor
    mixture: torch.Tensor
    expert: torch.Tensor
    counterfactual: torch.Tensor
    geometry: torch.Tensor
    cosine: torch.Tensor
    self_error: torch.Tensor
    cross_error: torch.Tensor
    aggregate_error: torch.Tensor

    def detached(self) -> dict[str, float]:
        return {
            "objective": float(self.objective.detach()),
            "mixture": float(self.mixture.detach()),
            "expert": float(self.expert.detach()),
            "counterfactual": float(self.counterfactual.detach()),
            "geometry": float(self.geometry.detach()),
            "cosine": float(self.cosine.detach()),
            "self_error": float(self.self_error.detach()),
            "cross_error": float(self.cross_error.detach()),
            "aggregate_error": float(self.aggregate_error.detach()),
        }


def conventional_routed_expert_outputs(
    module: ConventionalSwiGLUMoE,
    inputs: torch.Tensor,
    top_ids: torch.Tensor,
) -> torch.Tensor:
    """Return unweighted output of every selected conventional expert."""
    g = module.geometry
    if tuple(top_ids.shape) != (inputs.shape[0], g.top_k):
        raise ValueError("top_ids shape does not match inputs/top_k")
    selected_gate = module.gate.index_select(0, top_ids.reshape(-1)).reshape(
        inputs.shape[0], g.top_k, g.d_ff, g.d_model
    )
    selected_up = module.up.index_select(0, top_ids.reshape(-1)).reshape(
        inputs.shape[0], g.top_k, g.d_ff, g.d_model
    )
    selected_down = module.down.index_select(0, top_ids.reshape(-1)).reshape(
        inputs.shape[0], g.top_k, g.d_model, g.d_ff
    )
    gate_values = torch.einsum("ntfd,nd->ntf", selected_gate, inputs)
    up_values = torch.einsum("ntfd,nd->ntf", selected_up, inputs)
    hidden = F.silu(gate_values) * up_values
    return torch.einsum("ntdf,ntf->ntd", selected_down, hidden)


def routed_expert_outputs(
    module: nn.Module,
    inputs: torch.Tensor,
    top_ids: torch.Tensor,
) -> torch.Tensor:
    if isinstance(module, ConventionalSwiGLUMoE):
        return conventional_routed_expert_outputs(module, inputs, top_ids)
    if isinstance(module, SharedLowRankResidualMoE):
        return module.routed_expert_outputs(inputs, forced_top_ids=top_ids)
    method = getattr(module, "routed_expert_outputs", None)
    if callable(method):
        return method(inputs, forced_top_ids=top_ids)
    raise TypeError(f"unsupported MoE type for expert outputs: {type(module).__name__}")


def build_counterfactual_weights(natural_weights: torch.Tensor) -> torch.Tensor:
    """Build deterministic alternative mixtures for each routed expert set.

    Probes include uniform routing, every leave-one-out mixture based on the
    natural weights, and every equal-weight pair. Natural routing and one-hot
    probes are intentionally excluded: they are covered separately by the
    mixture and expert losses.
    """
    if natural_weights.ndim != 2:
        raise ValueError("natural_weights must have shape [tokens, top_k]")
    n, top_k = natural_weights.shape
    if top_k < 2:
        raise ValueError("counterfactual routing requires top_k >= 2")
    dtype = natural_weights.dtype
    device = natural_weights.device
    probes: list[torch.Tensor] = []

    probes.append(torch.full_like(natural_weights, 1.0 / top_k))

    for omitted in range(top_k):
        current = natural_weights.clone()
        current[:, omitted] = 0.0
        current = current / current.sum(dim=-1, keepdim=True).clamp_min(EPS)
        probes.append(current)

    for left in range(top_k):
        for right in range(left + 1, top_k):
            current = torch.zeros((n, top_k), dtype=dtype, device=device)
            current[:, left] = 0.5
            current[:, right] = 0.5
            probes.append(current)

    result = torch.stack(probes, dim=1)
    if not torch.allclose(result.sum(dim=-1), torch.ones_like(result[..., 0]), atol=1e-6):
        raise AssertionError("counterfactual probes are not normalized")
    return result


def _weighted_scale(values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    return (
        torch.sum(weights[..., None] * values.square())
        / (values.shape[-1] * torch.sum(weights).clamp_min(EPS))
    ).detach().clamp_min(EPS)


def _geometry_loss(student: torch.Tensor, teacher: torch.Tensor) -> torch.Tensor:
    """Match the relative geometry of experts inside each routed set.

    Outputs are centered across expert slots and normalized per vector before
    comparing off-diagonal cosine Gram matrices. This term is invariant to a
    consistent permutation of routed slots.
    """
    s = student - student.mean(dim=1, keepdim=True)
    t = teacher - teacher.mean(dim=1, keepdim=True)
    s = F.normalize(s, dim=-1, eps=EPS)
    t = F.normalize(t, dim=-1, eps=EPS)
    gram_s = torch.einsum("ntd,nsd->nts", s, s)
    gram_t = torch.einsum("ntd,nsd->nts", t, t)
    top_k = student.shape[1]
    mask = ~torch.eye(top_k, dtype=torch.bool, device=student.device)[None, :, :]
    return torch.mean((gram_s - gram_t).square()[mask.expand_as(gram_s)])


def error_decomposition(
    student_experts: torch.Tensor,
    teacher_experts: torch.Tensor,
    weights: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return normalized self, cross, and aggregate routing-error energies."""
    errors = student_experts - teacher_experts
    weighted_errors = weights[..., None] * errors
    self_energy = weighted_errors.square().sum(dim=(1, 2))
    aggregate_energy = weighted_errors.sum(dim=1).square().sum(dim=-1)
    cross_energy = aggregate_energy - self_energy
    teacher_mix = torch.einsum("nt,ntd->nd", weights, teacher_experts)
    scale = teacher_mix.square().sum(dim=-1).detach().clamp_min(EPS)
    return (
        torch.mean(self_energy / scale),
        torch.mean(cross_energy / scale),
        torch.mean(aggregate_energy / scale),
    )


def routing_set_loss(
    student_experts: torch.Tensor,
    teacher_experts: torch.Tensor,
    natural_weights: torch.Tensor,
    objective_weights: RoutingSetWeights,
) -> RoutingSetLosses:
    if student_experts.shape != teacher_experts.shape:
        raise ValueError("student and teacher expert outputs must match")
    if student_experts.ndim != 3:
        raise ValueError("expert outputs must have shape [tokens, top_k, d_model]")
    if natural_weights.shape != student_experts.shape[:2]:
        raise ValueError("natural routing shape mismatch")

    teacher_experts = teacher_experts.detach()
    teacher_mix = torch.einsum("nt,ntd->nd", natural_weights, teacher_experts)
    student_mix = torch.einsum("nt,ntd->nd", natural_weights, student_experts)
    mix_scale = teacher_mix.square().mean().detach().clamp_min(EPS)
    mixture = (student_mix - teacher_mix).square().mean() / mix_scale

    expert_scale = _weighted_scale(teacher_experts, natural_weights)
    expert = (
        torch.sum(natural_weights[..., None] * (student_experts - teacher_experts).square())
        / (student_experts.shape[-1] * torch.sum(natural_weights).clamp_min(EPS))
        / expert_scale
    )

    probes = build_counterfactual_weights(natural_weights)
    student_probes = torch.einsum("npk,nkd->npd", probes, student_experts)
    teacher_probes = torch.einsum("npk,nkd->npd", probes, teacher_experts)
    probe_scale = teacher_probes.square().mean().detach().clamp_min(EPS)
    counterfactual = (student_probes - teacher_probes).square().mean() / probe_scale

    geometry = _geometry_loss(student_experts, teacher_experts)
    cosine_rows = 1.0 - F.cosine_similarity(student_experts, teacher_experts, dim=-1)
    cosine = torch.sum(natural_weights * cosine_rows) / natural_weights.sum().clamp_min(EPS)

    self_error, cross_error, aggregate_error = error_decomposition(
        student_experts, teacher_experts, natural_weights
    )
    objective = (
        objective_weights.mixture * mixture
        + objective_weights.expert * expert
        + objective_weights.counterfactual * counterfactual
        + objective_weights.geometry * geometry
        + objective_weights.cosine * cosine
    )
    return RoutingSetLosses(
        objective=objective,
        mixture=mixture,
        expert=expert,
        counterfactual=counterfactual,
        geometry=geometry,
        cosine=cosine,
        self_error=self_error,
        cross_error=cross_error,
        aggregate_error=aggregate_error,
    )


def distill_routing_set_student(
    student: nn.Module,
    teacher_moe: ConventionalSwiGLUMoE,
    captured: CapturedLayerDataset,
    *,
    steps: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
    objective_weights: RoutingSetWeights,
) -> list[dict[str, float]]:
    parameters = [parameter for parameter in student.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=learning_rate, weight_decay=0.0)
    generator = torch.Generator().manual_seed(seed)
    history: list[dict[str, float]] = []
    student.train(); teacher_moe.eval()
    for step in range(1, steps + 1):
        indices = torch.randint(
            0,
            len(captured.inputs),
            (min(batch_size, len(captured.inputs)),),
            generator=generator,
        )
        inputs = captured.inputs.index_select(0, indices)
        top_ids = captured.top_ids.index_select(0, indices)
        weights = captured.route_weights.index_select(0, indices)
        with torch.no_grad():
            teacher_experts = routed_expert_outputs(teacher_moe, inputs, top_ids)
        student_experts = routed_expert_outputs(student, inputs, top_ids)
        losses = routing_set_loss(
            student_experts, teacher_experts, weights, objective_weights
        )
        optimizer.zero_grad(set_to_none=True)
        losses.objective.backward()
        torch.nn.utils.clip_grad_norm_(parameters, 1.0)
        optimizer.step()
        if step == 1 or step == steps or step % max(steps // 5, 1) == 0:
            history.append({"step": float(step), **losses.detached()})
    return history


def joint_fine_tune_routing_set(
    teacher: TinyMoELanguageModel,
    student: nn.Module,
    corpus: CharacterCorpus,
    *,
    layer_id: int,
    steps: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
    local_weights: RoutingSetWeights,
    joint_weights: JointRoutingSetWeights,
) -> tuple[nn.Module, list[dict[str, float]]]:
    if steps <= 0:
        return student, []
    teacher.eval()
    candidate = install_student(teacher, student, layer_id=layer_id)
    for parameter in candidate.parameters():
        parameter.requires_grad_(False)
    transplanted = candidate.blocks[layer_id].moe
    for parameter in transplanted.parameters():
        parameter.requires_grad_(parameter is not transplanted.router.weight)
    trainable = [parameter for parameter in transplanted.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=learning_rate, weight_decay=0.0)
    generator = torch.Generator().manual_seed(seed)
    teacher_moe = teacher.blocks[layer_id].moe
    if not isinstance(teacher_moe, ConventionalSwiGLUMoE):
        raise TypeError("teacher layer must be ConventionalSwiGLUMoE")
    history: list[dict[str, float]] = []
    candidate.train()
    for step in range(1, steps + 1):
        tokens, targets = corpus.sample_batch("train", batch_size, generator)
        with torch.no_grad():
            teacher_logits, _, teacher_capture = teacher(tokens, collect_layer=layer_id)
        candidate_logits, _, candidate_capture = candidate(tokens, collect_layer=layer_id)
        if teacher_capture is None or candidate_capture is None:
            raise AssertionError("joint routing-set capture missing")
        top_ids = teacher_capture.routing.top_ids
        route_weights = teacher_capture.routing.weights
        with torch.no_grad():
            teacher_experts = routed_expert_outputs(
                teacher_moe, teacher_capture.moe_input, top_ids
            )
        student_experts = routed_expert_outputs(
            transplanted, teacher_capture.moe_input, top_ids
        )
        local = routing_set_loss(
            student_experts, teacher_experts, route_weights, local_weights
        )
        teacher_log_prob = torch.log_softmax(teacher_logits.detach(), dim=-1)
        candidate_log_prob = torch.log_softmax(candidate_logits, dim=-1)
        teacher_prob = teacher_log_prob.exp()
        kl = torch.sum(teacher_prob * (teacher_log_prob - candidate_log_prob), dim=-1).mean()
        ce = F.cross_entropy(
            candidate_logits.reshape(-1, candidate_logits.shape[-1]), targets.reshape(-1)
        )
        objective = joint_weights.local * local.objective + joint_weights.kl * kl + joint_weights.ce * ce
        optimizer.zero_grad(set_to_none=True)
        objective.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()
        if step == 1 or step == steps or step % max(steps // 4, 1) == 0:
            history.append(
                {
                    "step": float(step),
                    "objective": float(objective.detach()),
                    "logit_kl": float(kl.detach()),
                    "token_ce": float(ce.detach()),
                    **{f"local_{key}": value for key, value in local.detached().items()},
                }
            )
    return copy.deepcopy(transplanted), history


@torch.no_grad()
def evaluate_routing_set_fidelity(
    teacher_moe: ConventionalSwiGLUMoE,
    student: nn.Module,
    inputs: torch.Tensor,
    top_ids: torch.Tensor,
    weights: torch.Tensor,
    objective_weights: RoutingSetWeights,
) -> dict[str, float]:
    teacher_experts = routed_expert_outputs(teacher_moe, inputs, top_ids)
    student_experts = routed_expert_outputs(student, inputs, top_ids)
    return routing_set_loss(
        student_experts, teacher_experts, weights, objective_weights
    ).detached()
