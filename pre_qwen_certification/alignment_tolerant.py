"""Alignment-tolerant expert compression for controlled MoE experiments.

The primary family keeps the full SwiGLU hidden width but represents each expert
matrix as a shared full-rank base plus an expert-specific low-rank residual. In
contrast to scalar Modal codes, every expert owns left and right side factors,
so rotations and permutations inside an expert need not align globally.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F

from .modal import (
    ConventionalSwiGLUMoE,
    MoEGeometry,
    Routing,
    _validate_forced_routing,
    route_topk,
)


@dataclass(frozen=True)
class CompressionAccounting:
    expert_parameters: int
    teacher_expert_parameters: int
    parameter_ratio: float
    routed_matrix_macs_per_token: int
    teacher_routed_matrix_macs_per_token: int
    compute_ratio: float

    def as_dict(self) -> dict[str, int | float]:
        return {
            "expert_parameters": self.expert_parameters,
            "teacher_expert_parameters": self.teacher_expert_parameters,
            "parameter_ratio": self.parameter_ratio,
            "routed_matrix_macs_per_token": self.routed_matrix_macs_per_token,
            "teacher_routed_matrix_macs_per_token": self.teacher_routed_matrix_macs_per_token,
            "compute_ratio": self.compute_ratio,
        }


def _svd_factors(matrix: torch.Tensor, rank: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Return balanced factors ``left @ right`` for a rank-truncated matrix."""
    if matrix.ndim != 2:
        raise ValueError("matrix must be two-dimensional")
    if rank <= 0:
        raise ValueError("rank must be positive")
    effective = min(rank, min(matrix.shape))
    u, singular, vh = torch.linalg.svd(matrix, full_matrices=False)
    root = singular[:effective].clamp_min(0.0).sqrt()
    left = u[:, :effective] * root[None, :]
    right = root[:, None] * vh[:effective, :]
    if effective == rank:
        return left, right
    padded_left = matrix.new_zeros((matrix.shape[0], rank))
    padded_right = matrix.new_zeros((rank, matrix.shape[1]))
    padded_left[:, :effective] = left
    padded_right[:effective, :] = right
    return padded_left, padded_right


class SharedLowRankResidualMoE(nn.Module):
    """Full-width SwiGLU experts with shared bases and bilateral side factors.

    For every expert and every SwiGLU matrix, the deployed weight is
    ``W_e = W_shared + L_e @ R_e``.

    The implementation executes the factorization directly. Shared projections
    are evaluated once per token; expert-specific factors are evaluated only for
    routed experts. The router is copied from the teacher and frozen.
    """

    def __init__(self, geometry: MoEGeometry, rank: int) -> None:
        super().__init__()
        geometry.validate()
        if rank <= 0:
            raise ValueError("rank must be positive")
        if rank > min(geometry.d_model, geometry.d_ff):
            raise ValueError("rank exceeds the smallest matrix dimension")
        self.geometry = geometry
        self.rank = int(rank)
        e, h, d, r = (
            geometry.n_experts,
            geometry.d_ff,
            geometry.d_model,
            self.rank,
        )
        self.router = nn.Linear(d, e, bias=False)

        self.common_gate = nn.Parameter(torch.empty(h, d))
        self.gate_left = nn.Parameter(torch.empty(e, h, r))
        self.gate_right = nn.Parameter(torch.empty(e, r, d))

        self.common_up = nn.Parameter(torch.empty(h, d))
        self.up_left = nn.Parameter(torch.empty(e, h, r))
        self.up_right = nn.Parameter(torch.empty(e, r, d))

        self.common_down = nn.Parameter(torch.empty(d, h))
        self.down_left = nn.Parameter(torch.empty(e, d, r))
        self.down_right = nn.Parameter(torch.empty(e, r, h))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.router.weight, std=0.02)
        for common in (self.common_gate, self.common_up, self.common_down):
            nn.init.xavier_uniform_(common)
        scale = 1.0 / max(self.rank, 1) ** 0.5
        for left in (self.gate_left, self.up_left, self.down_left):
            nn.init.normal_(left, std=0.02 * scale)
        for right in (self.gate_right, self.up_right, self.down_right):
            nn.init.normal_(right, std=0.02 * scale)

    @classmethod
    @torch.no_grad()
    def from_teacher(
        cls,
        teacher: ConventionalSwiGLUMoE,
        *,
        rank: int,
    ) -> "SharedLowRankResidualMoE":
        student = cls(teacher.geometry, rank)
        student.router.weight.copy_(teacher.router.weight)
        student.router.weight.requires_grad_(False)

        for teacher_bank, common, left, right in (
            (teacher.gate, student.common_gate, student.gate_left, student.gate_right),
            (teacher.up, student.common_up, student.up_left, student.up_right),
            (teacher.down, student.common_down, student.down_left, student.down_right),
        ):
            mean = teacher_bank.mean(dim=0)
            common.copy_(mean)
            for expert in range(teacher.geometry.n_experts):
                factor_left, factor_right = _svd_factors(
                    teacher_bank[expert] - mean,
                    rank,
                )
                left[expert].copy_(factor_left)
                right[expert].copy_(factor_right)
        return student

    def reconstruct_weights(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        gate = self.common_gate[None, :, :] + torch.einsum(
            "ehr,erd->ehd", self.gate_left, self.gate_right
        )
        up = self.common_up[None, :, :] + torch.einsum(
            "ehr,erd->ehd", self.up_left, self.up_right
        )
        down = self.common_down[None, :, :] + torch.einsum(
            "edr,erh->edh", self.down_left, self.down_right
        )
        return gate, up, down

    def accounting(self) -> CompressionAccounting:
        g = self.geometry
        shared_per_bank = g.d_ff * g.d_model
        factor_per_bank = g.n_experts * self.rank * (g.d_ff + g.d_model)
        expert_parameters = 3 * (shared_per_bank + factor_per_bank)
        teacher_parameters = 3 * g.n_experts * g.d_ff * g.d_model

        shared_macs = 3 * g.d_ff * g.d_model
        routed_factor_macs = 3 * g.top_k * self.rank * (g.d_ff + g.d_model)
        student_macs = shared_macs + routed_factor_macs
        teacher_macs = 3 * g.top_k * g.d_ff * g.d_model
        return CompressionAccounting(
            expert_parameters=expert_parameters,
            teacher_expert_parameters=teacher_parameters,
            parameter_ratio=expert_parameters / teacher_parameters,
            routed_matrix_macs_per_token=student_macs,
            teacher_routed_matrix_macs_per_token=teacher_macs,
            compute_ratio=student_macs / teacher_macs,
        )

    def routed_expert_outputs(
        self,
        x: torch.Tensor,
        *,
        forced_top_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Return unweighted outputs of the routed experts, shape ``[N,T,D]``."""
        g = self.geometry
        if tuple(forced_top_ids.shape) != (x.shape[0], g.top_k):
            raise ValueError("forced_top_ids has invalid shape")
        flat_ids = forced_top_ids.reshape(-1)
        n = x.shape[0]

        common_gate = F.linear(x, self.common_gate)[:, None, :]
        gate_right = self.gate_right.index_select(0, flat_ids).reshape(
            n, g.top_k, self.rank, g.d_model
        )
        gate_left = self.gate_left.index_select(0, flat_ids).reshape(
            n, g.top_k, g.d_ff, self.rank
        )
        gate_latent = torch.einsum("ntrd,nd->ntr", gate_right, x)
        gate_values = common_gate + torch.einsum("nthr,ntr->nth", gate_left, gate_latent)

        common_up = F.linear(x, self.common_up)[:, None, :]
        up_right = self.up_right.index_select(0, flat_ids).reshape(
            n, g.top_k, self.rank, g.d_model
        )
        up_left = self.up_left.index_select(0, flat_ids).reshape(
            n, g.top_k, g.d_ff, self.rank
        )
        up_latent = torch.einsum("ntrd,nd->ntr", up_right, x)
        up_values = common_up + torch.einsum("nthr,ntr->nth", up_left, up_latent)
        hidden = F.silu(gate_values) * up_values

        common_output = F.linear(hidden, self.common_down)
        down_right = self.down_right.index_select(0, flat_ids).reshape(
            n, g.top_k, self.rank, g.d_ff
        )
        down_left = self.down_left.index_select(0, flat_ids).reshape(
            n, g.top_k, g.d_model, self.rank
        )
        down_latent = torch.einsum("ntrh,nth->ntr", down_right, hidden)
        residual_output = torch.einsum("ntdr,ntr->ntd", down_left, down_latent)
        return common_output + residual_output

    def forward(
        self,
        x: torch.Tensor,
        *,
        forced_top_ids: torch.Tensor | None = None,
        forced_weights: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, Routing]:
        g = self.geometry
        _validate_forced_routing(x, forced_top_ids, forced_weights, g.top_k)
        natural = route_topk(x, self.router.weight, g.top_k)
        top_ids = natural.top_ids if forced_top_ids is None else forced_top_ids
        weights = natural.weights if forced_weights is None else forced_weights
        expert_outputs = self.routed_expert_outputs(x, forced_top_ids=top_ids)
        output = torch.einsum("nt,ntd->nd", weights, expert_outputs)
        return output, Routing(natural.logits, top_ids, weights)

    @torch.no_grad()
    def reference_reconstructed(
        self,
        x: torch.Tensor,
        *,
        forced_top_ids: torch.Tensor | None = None,
        forced_weights: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, Routing]:
        conventional = ConventionalSwiGLUMoE(self.geometry)
        gate, up, down = self.reconstruct_weights()
        conventional.router.weight.copy_(self.router.weight)
        conventional.gate.copy_(gate)
        conventional.up.copy_(up)
        conventional.down.copy_(down)
        return conventional(
            x,
            forced_top_ids=forced_top_ids,
            forced_weights=forced_weights,
        )

    def clone(self) -> "SharedLowRankResidualMoE":
        return copy.deepcopy(self)
