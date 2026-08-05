"""Explicit route-set coupling for alignment-tolerant MoE students.

The v1-v3 students represent each expert independently and rely on the training
objective to induce favorable cross-expert error covariance.  This module makes
coordination representable: it reuses each routed expert's low-rank ``down``
latent, aligns those latents into a shared set space, pools weighted first and
second moments, and emits one zero-initialized correction for the routed set.

The correction is permutation-invariant in routed-slot order and is evaluated
for arbitrary normalized routing weights, so the same module supports natural
and counterfactual mixtures.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F

from .alignment_tolerant import SharedLowRankResidualMoE
from .modal import ConventionalSwiGLUMoE, Routing, _validate_forced_routing, route_topk
from .routing_set_distillation import (
    EPS,
    JointRoutingSetWeights,
    RoutingSetLosses,
    RoutingSetWeights,
    _geometry_loss,
    _weighted_scale,
    build_counterfactual_weights,
    conventional_routed_expert_outputs,
    error_decomposition,
)
from .tiny_lm import CapturedLayerDataset, CharacterCorpus, TinyMoELanguageModel, install_student


@dataclass(frozen=True)
class CoupledAccounting:
    expert_parameters: int
    teacher_expert_parameters: int
    parameter_ratio: float
    routed_matrix_macs_per_token: int
    teacher_routed_matrix_macs_per_token: int
    compute_ratio: float
    coupling_parameters: int
    coupling_matrix_macs_per_token: int

    def as_dict(self) -> dict[str, int | float]:
        return {
            "expert_parameters": self.expert_parameters,
            "teacher_expert_parameters": self.teacher_expert_parameters,
            "parameter_ratio": self.parameter_ratio,
            "routed_matrix_macs_per_token": self.routed_matrix_macs_per_token,
            "teacher_routed_matrix_macs_per_token": self.teacher_routed_matrix_macs_per_token,
            "compute_ratio": self.compute_ratio,
            "coupling_parameters": self.coupling_parameters,
            "coupling_matrix_macs_per_token": self.coupling_matrix_macs_per_token,
        }


class RoutingCoupledResidualMoE(nn.Module):
    """Shared-low-rank experts plus an explicit permutation-invariant set correction."""

    def __init__(
        self,
        base: SharedLowRankResidualMoE,
        *,
        set_dim: int = 8,
        hidden_dim: int = 8,
        use_second_moment: bool = True,
    ) -> None:
        super().__init__()
        if set_dim <= 0 or hidden_dim <= 0:
            raise ValueError("set_dim and hidden_dim must be positive")
        self.base = copy.deepcopy(base)
        self.set_dim = int(set_dim)
        self.hidden_dim = int(hidden_dim)
        self.use_second_moment = bool(use_second_moment)
        self.coupling_enabled = True

        g = self.base.geometry
        r = self.base.rank
        self.expert_alignment = nn.Parameter(torch.empty(g.n_experts, self.set_dim, r))
        self.expert_embedding = nn.Parameter(torch.zeros(g.n_experts, self.set_dim))
        self.route_direction = nn.Parameter(torch.zeros(self.set_dim))
        self.coupling_in = nn.Parameter(torch.empty(self.hidden_dim, 2 * self.set_dim))
        self.coupling_out = nn.Parameter(torch.zeros(g.d_model, self.hidden_dim))
        self.reset_coupling_parameters()

    @property
    def geometry(self):
        return self.base.geometry

    @property
    def rank(self) -> int:
        return self.base.rank

    @property
    def router(self) -> nn.Linear:
        return self.base.router

    def reset_coupling_parameters(self) -> None:
        nn.init.normal_(self.expert_alignment, std=1.0 / max(self.rank, 1) ** 0.5)
        nn.init.zeros_(self.expert_embedding)
        nn.init.zeros_(self.route_direction)
        nn.init.xavier_uniform_(self.coupling_in)
        # Zero initialization makes the deployed function exactly equal to the
        # frozen base before any coupling training.
        nn.init.zeros_(self.coupling_out)

    @classmethod
    def from_base(
        cls,
        base: SharedLowRankResidualMoE,
        *,
        set_dim: int = 8,
        hidden_dim: int = 8,
        use_second_moment: bool = True,
    ) -> "RoutingCoupledResidualMoE":
        return cls(
            base,
            set_dim=set_dim,
            hidden_dim=hidden_dim,
            use_second_moment=use_second_moment,
        )

    def freeze_base(self) -> None:
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)

    def unfreeze_all_except_router(self) -> None:
        for parameter in self.parameters():
            parameter.requires_grad_(True)
        self.router.weight.requires_grad_(False)

    def coupling_parameters(self) -> list[nn.Parameter]:
        return [
            self.expert_alignment,
            self.expert_embedding,
            self.route_direction,
            self.coupling_in,
            self.coupling_out,
        ]

    def accounting(self) -> CoupledAccounting:
        base = self.base.accounting()
        g = self.geometry
        q = self.set_dim
        h = self.hidden_dim
        r = self.rank
        coupling_parameters = (
            g.n_experts * q * r
            + g.n_experts * q
            + q
            + (2 * q) * h
            + h * g.d_model
        )
        coupling_macs = g.top_k * q * r + (2 * q) * h + h * g.d_model
        expert_parameters = base.expert_parameters + coupling_parameters
        routed_macs = base.routed_matrix_macs_per_token + coupling_macs
        return CoupledAccounting(
            expert_parameters=expert_parameters,
            teacher_expert_parameters=base.teacher_expert_parameters,
            parameter_ratio=expert_parameters / base.teacher_expert_parameters,
            routed_matrix_macs_per_token=routed_macs,
            teacher_routed_matrix_macs_per_token=base.teacher_routed_matrix_macs_per_token,
            compute_ratio=routed_macs / base.teacher_routed_matrix_macs_per_token,
            coupling_parameters=coupling_parameters,
            coupling_matrix_macs_per_token=coupling_macs,
        )

    def _base_outputs_and_latents(
        self,
        x: torch.Tensor,
        top_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        g = self.geometry
        if tuple(top_ids.shape) != (x.shape[0], g.top_k):
            raise ValueError("top_ids has invalid shape")
        flat_ids = top_ids.reshape(-1)
        n = x.shape[0]
        base = self.base

        common_gate = F.linear(x, base.common_gate)[:, None, :]
        gate_right = base.gate_right.index_select(0, flat_ids).reshape(
            n, g.top_k, self.rank, g.d_model
        )
        gate_left = base.gate_left.index_select(0, flat_ids).reshape(
            n, g.top_k, g.d_ff, self.rank
        )
        gate_latent = torch.einsum("ntrd,nd->ntr", gate_right, x)
        gate_values = common_gate + torch.einsum("nthr,ntr->nth", gate_left, gate_latent)

        common_up = F.linear(x, base.common_up)[:, None, :]
        up_right = base.up_right.index_select(0, flat_ids).reshape(
            n, g.top_k, self.rank, g.d_model
        )
        up_left = base.up_left.index_select(0, flat_ids).reshape(
            n, g.top_k, g.d_ff, self.rank
        )
        up_latent = torch.einsum("ntrd,nd->ntr", up_right, x)
        up_values = common_up + torch.einsum("nthr,ntr->nth", up_left, up_latent)
        hidden = F.silu(gate_values) * up_values

        common_output = F.linear(hidden, base.common_down)
        down_right = base.down_right.index_select(0, flat_ids).reshape(
            n, g.top_k, self.rank, g.d_ff
        )
        down_left = base.down_left.index_select(0, flat_ids).reshape(
            n, g.top_k, g.d_model, self.rank
        )
        down_latent = torch.einsum("ntrh,nth->ntr", down_right, hidden)
        residual_output = torch.einsum("ntdr,ntr->ntd", down_left, down_latent)
        return common_output + residual_output, down_latent

    def routed_components(
        self,
        x: torch.Tensor,
        *,
        forced_top_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        experts, latents = self._base_outputs_and_latents(x, forced_top_ids)
        n = x.shape[0]
        selected_alignment = self.expert_alignment.index_select(
            0, forced_top_ids.reshape(-1)
        ).reshape(n, self.geometry.top_k, self.set_dim, self.rank)
        aligned = torch.einsum("ntqr,ntr->ntq", selected_alignment, latents)
        embedding = self.expert_embedding.index_select(
            0, forced_top_ids.reshape(-1)
        ).reshape(n, self.geometry.top_k, self.set_dim)
        return experts, aligned + embedding

    def routed_expert_outputs(
        self,
        x: torch.Tensor,
        *,
        forced_top_ids: torch.Tensor,
    ) -> torch.Tensor:
        # Exposes the uncorrected expert functions for expert-wise diagnostics.
        return self.routed_components(x, forced_top_ids=forced_top_ids)[0]

    def correction_from_aligned(
        self,
        aligned: torch.Tensor,
        weights: torch.Tensor,
        *,
        use_second_moment: bool | None = None,
    ) -> torch.Tensor:
        if weights.shape[-1] != aligned.shape[1]:
            raise ValueError("routing weights do not match routed expert slots")
        if weights.ndim not in (2, 3):
            raise ValueError("weights must have shape [N,T] or [N,P,T]")
        second_enabled = self.use_second_moment if use_second_moment is None else bool(use_second_moment)
        if weights.ndim == 2:
            features = aligned + torch.log(weights.clamp_min(EPS))[..., None] * self.route_direction
            mean = torch.einsum("nt,ntq->nq", weights, features)
            second = torch.einsum("nt,ntq->nq", weights, features.square()) - mean.square()
        else:
            features = aligned[:, None, :, :] + torch.log(weights.clamp_min(EPS))[..., None] * self.route_direction
            mean = torch.einsum("npt,nptq->npq", weights, features)
            second = torch.einsum("npt,nptq->npq", weights, features.square()) - mean.square()
        if not second_enabled:
            second = torch.zeros_like(second)
        stats = torch.cat((mean, second), dim=-1)
        hidden = F.silu(F.linear(stats, self.coupling_in))
        correction = F.linear(hidden, self.coupling_out)
        if not self.coupling_enabled:
            correction = torch.zeros_like(correction)
        return correction

    def mix_from_components(
        self,
        experts: torch.Tensor,
        aligned: torch.Tensor,
        weights: torch.Tensor,
        *,
        use_second_moment: bool | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if weights.ndim == 2:
            base_mix = torch.einsum("nt,ntd->nd", weights, experts)
        elif weights.ndim == 3:
            base_mix = torch.einsum("npt,ntd->npd", weights, experts)
        else:
            raise ValueError("weights must have shape [N,T] or [N,P,T]")
        correction = self.correction_from_aligned(
            aligned, weights, use_second_moment=use_second_moment
        )
        return base_mix + correction, correction

    def forward(
        self,
        x: torch.Tensor,
        *,
        forced_top_ids: torch.Tensor | None = None,
        forced_weights: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, Routing]:
        _validate_forced_routing(x, forced_top_ids, forced_weights, self.geometry.top_k)
        natural = route_topk(x, self.router.weight, self.geometry.top_k)
        top_ids = natural.top_ids if forced_top_ids is None else forced_top_ids
        weights = natural.weights if forced_weights is None else forced_weights
        experts, aligned = self.routed_components(x, forced_top_ids=top_ids)
        output, _ = self.mix_from_components(experts, aligned, weights)
        return output, Routing(natural.logits, top_ids, weights)

    def clone(self) -> "RoutingCoupledResidualMoE":
        return copy.deepcopy(self)


def coupled_routing_set_loss(
    student: RoutingCoupledResidualMoE,
    inputs: torch.Tensor,
    top_ids: torch.Tensor,
    natural_weights: torch.Tensor,
    teacher_experts: torch.Tensor,
    objective_weights: RoutingSetWeights,
) -> RoutingSetLosses:
    teacher_experts = teacher_experts.detach()
    student_experts, aligned = student.routed_components(inputs, forced_top_ids=top_ids)
    teacher_mix = torch.einsum("nt,ntd->nd", natural_weights, teacher_experts)
    student_mix, correction = student.mix_from_components(
        student_experts, aligned, natural_weights
    )
    mix_scale = teacher_mix.square().mean().detach().clamp_min(EPS)
    mixture = (student_mix - teacher_mix).square().mean() / mix_scale

    expert_scale = _weighted_scale(teacher_experts, natural_weights)
    expert = (
        torch.sum(natural_weights[..., None] * (student_experts - teacher_experts).square())
        / (student_experts.shape[-1] * torch.sum(natural_weights).clamp_min(EPS))
        / expert_scale
    )

    probes = build_counterfactual_weights(natural_weights)
    student_probes, _ = student.mix_from_components(student_experts, aligned, probes)
    teacher_probes = torch.einsum("npt,ntd->npd", probes, teacher_experts)
    probe_scale = teacher_probes.square().mean().detach().clamp_min(EPS)
    counterfactual = (student_probes - teacher_probes).square().mean() / probe_scale

    geometry = _geometry_loss(student_experts, teacher_experts)
    cosine_rows = 1.0 - F.cosine_similarity(student_experts, teacher_experts, dim=-1)
    cosine = torch.sum(natural_weights * cosine_rows) / natural_weights.sum().clamp_min(EPS)

    # Broadcasting the natural correction to each slot yields an exact natural
    # mixture decomposition because routing weights sum to one.
    effective_experts = student_experts + correction[:, None, :]
    self_error, cross_error, aggregate_error = error_decomposition(
        effective_experts, teacher_experts, natural_weights
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


def distill_coupled_student(
    student: RoutingCoupledResidualMoE,
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
    if not parameters:
        raise ValueError("student has no trainable parameters")
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
            teacher_experts = conventional_routed_expert_outputs(teacher_moe, inputs, top_ids)
        losses = coupled_routing_set_loss(
            student, inputs, top_ids, weights, teacher_experts, objective_weights
        )
        optimizer.zero_grad(set_to_none=True)
        losses.objective.backward()
        torch.nn.utils.clip_grad_norm_(parameters, 1.0)
        optimizer.step()
        if step == 1 or step == steps or step % max(steps // 5, 1) == 0:
            history.append({"step": float(step), **losses.detached()})
    return history


def joint_fine_tune_coupled(
    teacher: TinyMoELanguageModel,
    student: RoutingCoupledResidualMoE,
    corpus: CharacterCorpus,
    *,
    layer_id: int,
    steps: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
    local_weights: RoutingSetWeights,
    joint_weights: JointRoutingSetWeights,
) -> tuple[RoutingCoupledResidualMoE, list[dict[str, float]]]:
    if steps <= 0:
        return student, []
    teacher.eval()
    candidate = install_student(teacher, student, layer_id=layer_id)
    for parameter in candidate.parameters():
        parameter.requires_grad_(False)
    transplanted = candidate.blocks[layer_id].moe
    if not isinstance(transplanted, RoutingCoupledResidualMoE):
        raise TypeError("transplanted module is not RoutingCoupledResidualMoE")
    transplanted.unfreeze_all_except_router()
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
            raise AssertionError("joint coupled capture missing")
        top_ids = teacher_capture.routing.top_ids
        route_weights = teacher_capture.routing.weights
        with torch.no_grad():
            teacher_experts = conventional_routed_expert_outputs(
                teacher_moe, teacher_capture.moe_input, top_ids
            )
        local = coupled_routing_set_loss(
            transplanted,
            teacher_capture.moe_input,
            top_ids,
            route_weights,
            teacher_experts,
            local_weights,
        )
        teacher_log_prob = torch.log_softmax(teacher_logits.detach(), dim=-1)
        candidate_log_prob = torch.log_softmax(candidate_logits, dim=-1)
        teacher_prob = teacher_log_prob.exp()
        kl = torch.sum(
            teacher_prob * (teacher_log_prob - candidate_log_prob), dim=-1
        ).mean()
        ce = F.cross_entropy(
            candidate_logits.reshape(-1, candidate_logits.shape[-1]), targets.reshape(-1)
        )
        objective = (
            joint_weights.local * local.objective
            + joint_weights.kl * kl
            + joint_weights.ce * ce
        )
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
def evaluate_coupled_fidelity(
    teacher_moe: ConventionalSwiGLUMoE,
    student: RoutingCoupledResidualMoE,
    inputs: torch.Tensor,
    top_ids: torch.Tensor,
    weights: torch.Tensor,
    objective_weights: RoutingSetWeights,
) -> dict[str, float]:
    teacher_experts = conventional_routed_expert_outputs(teacher_moe, inputs, top_ids)
    losses = coupled_routing_set_loss(
        student, inputs, top_ids, weights, teacher_experts, objective_weights
    )
    student_experts, aligned = student.routed_components(inputs, forced_top_ids=top_ids)
    _, correction = student.mix_from_components(student_experts, aligned, weights)
    teacher_mix = torch.einsum("nt,ntd->nd", weights, teacher_experts)
    correction_ratio = (
        correction.square().sum(dim=-1) / teacher_mix.square().sum(dim=-1).clamp_min(EPS)
    ).mean()
    return {**losses.detached(), "correction_energy_ratio": float(correction_ratio)}
