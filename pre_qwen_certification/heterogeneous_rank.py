"""Static heterogeneous-rank shared-base MoE for Reality Gate 1A.

The module keeps the shared full-rank gate/up/down projections used by the
alignment-tolerant family, but allows every expert to own a different residual
rank.  The implementation executes shared projections once per token and only
executes the active expert factors for selected experts.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

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

EPS = 1e-12


@dataclass(frozen=True)
class HeterogeneousAccounting:
    expert_parameters: int
    teacher_expert_parameters: int
    parameter_ratio: float
    expected_routed_matrix_macs: float
    teacher_routed_matrix_macs: int
    compute_ratio: float
    expected_active_rank: float
    uniform_reference_rank: float | None = None

    def as_dict(self) -> dict[str, int | float | None]:
        return {
            "expert_parameters": self.expert_parameters,
            "teacher_expert_parameters": self.teacher_expert_parameters,
            "parameter_ratio": self.parameter_ratio,
            "expected_routed_matrix_macs": self.expected_routed_matrix_macs,
            "teacher_routed_matrix_macs": self.teacher_routed_matrix_macs,
            "compute_ratio": self.compute_ratio,
            "expected_active_rank": self.expected_active_rank,
            "uniform_reference_rank": self.uniform_reference_rank,
        }


def _balanced_svd_factors(matrix: torch.Tensor, rank: int) -> tuple[torch.Tensor, torch.Tensor]:
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


def _validate_ranks(geometry: MoEGeometry, ranks: Sequence[int]) -> tuple[int, ...]:
    result = tuple(int(value) for value in ranks)
    if len(result) != geometry.n_experts:
        raise ValueError("one rank is required for every expert")
    maximum = min(geometry.d_model, geometry.d_ff)
    if any(value <= 0 or value > maximum for value in result):
        raise ValueError(f"expert ranks must be in [1, {maximum}]")
    return result


class HeterogeneousSharedLowRankResidualMoE(nn.Module):
    """Shared full-rank bases plus variable-rank expert residuals.

    Each bank follows ``W_e = W_shared + L_e @ R_e``.  The factor width may
    differ per expert.  Shared gate/up projections are evaluated once for every
    token; the shared down projection is evaluated once after routed hidden
    states have been aggregated.
    """

    def __init__(self, geometry: MoEGeometry, ranks: Sequence[int]) -> None:
        super().__init__()
        geometry.validate()
        self.geometry = geometry
        self.ranks = _validate_ranks(geometry, ranks)
        e, h, d = geometry.n_experts, geometry.d_ff, geometry.d_model
        self.router = nn.Linear(d, e, bias=False)
        self.common_gate = nn.Parameter(torch.empty(h, d))
        self.common_up = nn.Parameter(torch.empty(h, d))
        self.common_down = nn.Parameter(torch.empty(d, h))

        self.gate_left = nn.ParameterList(
            [nn.Parameter(torch.empty(h, rank)) for rank in self.ranks]
        )
        self.gate_right = nn.ParameterList(
            [nn.Parameter(torch.empty(rank, d)) for rank in self.ranks]
        )
        self.up_left = nn.ParameterList(
            [nn.Parameter(torch.empty(h, rank)) for rank in self.ranks]
        )
        self.up_right = nn.ParameterList(
            [nn.Parameter(torch.empty(rank, d)) for rank in self.ranks]
        )
        self.down_left = nn.ParameterList(
            [nn.Parameter(torch.empty(d, rank)) for rank in self.ranks]
        )
        self.down_right = nn.ParameterList(
            [nn.Parameter(torch.empty(rank, h)) for rank in self.ranks]
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.router.weight, std=0.02)
        for common in (self.common_gate, self.common_up, self.common_down):
            nn.init.xavier_uniform_(common)
        for lefts, rights in (
            (self.gate_left, self.gate_right),
            (self.up_left, self.up_right),
            (self.down_left, self.down_right),
        ):
            for rank, left, right in zip(self.ranks, lefts, rights, strict=True):
                scale = 1.0 / max(rank, 1) ** 0.5
                nn.init.normal_(left, std=0.02 * scale)
                nn.init.normal_(right, std=0.02 * scale)

    @classmethod
    @torch.no_grad()
    def from_teacher(
        cls,
        teacher: ConventionalSwiGLUMoE,
        *,
        ranks: Sequence[int],
    ) -> "HeterogeneousSharedLowRankResidualMoE":
        student = cls(teacher.geometry, ranks)
        student.router.weight.copy_(teacher.router.weight)
        student.router.weight.requires_grad_(False)
        for teacher_bank, common, lefts, rights in (
            (teacher.gate, student.common_gate, student.gate_left, student.gate_right),
            (teacher.up, student.common_up, student.up_left, student.up_right),
            (teacher.down, student.common_down, student.down_left, student.down_right),
        ):
            mean = teacher_bank.mean(dim=0)
            common.copy_(mean)
            for expert, rank in enumerate(student.ranks):
                left, right = _balanced_svd_factors(teacher_bank[expert] - mean, rank)
                lefts[expert].copy_(left)
                rights[expert].copy_(right)
        return student

    def _routing(
        self,
        x: torch.Tensor,
        forced_top_ids: torch.Tensor | None,
        forced_weights: torch.Tensor | None,
    ) -> Routing:
        _validate_forced_routing(
            x, forced_top_ids, forced_weights, self.geometry.top_k
        )
        if forced_top_ids is None:
            return route_topk(x, self.router.weight, self.geometry.top_k)
        logits = F.linear(x, self.router.weight)
        return Routing(logits=logits, top_ids=forced_top_ids, weights=forced_weights)

    def _expert_hidden(self, x: torch.Tensor, expert: int) -> torch.Tensor:
        gate = F.linear(x, self.common_gate)
        gate = gate + F.linear(F.linear(x, self.gate_right[expert]), self.gate_left[expert])
        up = F.linear(x, self.common_up)
        up = up + F.linear(F.linear(x, self.up_right[expert]), self.up_left[expert])
        return F.silu(gate) * up

    def _expert_output(self, x: torch.Tensor, expert: int) -> torch.Tensor:
        hidden = self._expert_hidden(x, expert)
        return F.linear(hidden, self.common_down) + F.linear(
            F.linear(hidden, self.down_right[expert]), self.down_left[expert]
        )

    def routed_expert_outputs(
        self,
        x: torch.Tensor,
        *,
        forced_top_ids: torch.Tensor,
    ) -> torch.Tensor:
        if tuple(forced_top_ids.shape) != (x.shape[0], self.geometry.top_k):
            raise ValueError("forced_top_ids shape mismatch")
        output = x.new_empty((x.shape[0], self.geometry.top_k, self.geometry.d_model))
        for expert in range(self.geometry.n_experts):
            token_ids, slot_ids = (forced_top_ids == expert).nonzero(as_tuple=True)
            if token_ids.numel() == 0:
                continue
            output[token_ids, slot_ids] = self._expert_output(x.index_select(0, token_ids), expert)
        return output

    def forward(
        self,
        x: torch.Tensor,
        forced_top_ids: torch.Tensor | None = None,
        forced_weights: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, Routing]:
        routing = self._routing(x, forced_top_ids, forced_weights)
        common_gate = F.linear(x, self.common_gate)
        common_up = F.linear(x, self.common_up)
        aggregated_hidden = x.new_zeros((x.shape[0], self.geometry.d_ff))
        residual_output = x.new_zeros((x.shape[0], self.geometry.d_model))
        for expert in range(self.geometry.n_experts):
            token_ids, slot_ids = (routing.top_ids == expert).nonzero(as_tuple=True)
            if token_ids.numel() == 0:
                continue
            selected = x.index_select(0, token_ids)
            gate = common_gate.index_select(0, token_ids) + F.linear(
                F.linear(selected, self.gate_right[expert]), self.gate_left[expert]
            )
            up = common_up.index_select(0, token_ids) + F.linear(
                F.linear(selected, self.up_right[expert]), self.up_left[expert]
            )
            hidden = F.silu(gate) * up
            weights = routing.weights[token_ids, slot_ids, None]
            aggregated_hidden.index_add_(0, token_ids, weights * hidden)
            residual = F.linear(
                F.linear(hidden, self.down_right[expert]), self.down_left[expert]
            )
            residual_output.index_add_(0, token_ids, weights * residual)
        return F.linear(aggregated_hidden, self.common_down) + residual_output, routing

    @torch.no_grad()
    def reconstruct_weights(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        gate: list[torch.Tensor] = []
        up: list[torch.Tensor] = []
        down: list[torch.Tensor] = []
        for expert in range(self.geometry.n_experts):
            gate.append(self.common_gate + self.gate_left[expert] @ self.gate_right[expert])
            up.append(self.common_up + self.up_left[expert] @ self.up_right[expert])
            down.append(self.common_down + self.down_left[expert] @ self.down_right[expert])
        return torch.stack(gate), torch.stack(up), torch.stack(down)

    def expert_transform_parameter_count(self) -> int:
        g = self.geometry
        shared = 3 * g.d_ff * g.d_model
        factors = 3 * (g.d_ff + g.d_model) * sum(self.ranks)
        return int(shared + factors)

    def active_rank_per_token(self, top_ids: torch.Tensor) -> torch.Tensor:
        ranks = torch.tensor(self.ranks, device=top_ids.device, dtype=torch.float32)
        return ranks.index_select(0, top_ids.reshape(-1)).reshape_as(top_ids).sum(dim=-1)

    def accounting(
        self,
        route_frequencies: Sequence[float] | torch.Tensor,
        *,
        uniform_reference_rank: float | None = None,
    ) -> HeterogeneousAccounting:
        frequencies = torch.as_tensor(route_frequencies, dtype=torch.float64)
        if tuple(frequencies.shape) != (self.geometry.n_experts,):
            raise ValueError("route_frequencies must have one value per expert")
        if torch.any(frequencies < 0):
            raise ValueError("route_frequencies must be non-negative")
        if abs(float(frequencies.sum()) - self.geometry.top_k) > 1e-5:
            raise ValueError("route frequencies must sum to top_k")
        g = self.geometry
        teacher_parameters = 3 * g.n_experts * g.d_ff * g.d_model
        parameters = self.expert_transform_parameter_count()
        expected_active_rank = float(
            torch.dot(frequencies, torch.tensor(self.ranks, dtype=torch.float64))
        )
        expected_macs = 3 * g.d_ff * g.d_model + 3 * (
            g.d_ff + g.d_model
        ) * expected_active_rank
        teacher_macs = 3 * g.top_k * g.d_ff * g.d_model
        return HeterogeneousAccounting(
            expert_parameters=parameters,
            teacher_expert_parameters=teacher_parameters,
            parameter_ratio=parameters / teacher_parameters,
            expected_routed_matrix_macs=float(expected_macs),
            teacher_routed_matrix_macs=teacher_macs,
            compute_ratio=float(expected_macs / teacher_macs),
            expected_active_rank=expected_active_rank,
            uniform_reference_rank=uniform_reference_rank,
        )


def route_frequencies(top_ids: torch.Tensor, n_experts: int) -> torch.Tensor:
    if top_ids.ndim != 2:
        raise ValueError("top_ids must have shape [tokens, top_k]")
    counts = torch.bincount(top_ids.reshape(-1), minlength=n_experts).double()
    return counts / top_ids.shape[0]


@torch.no_grad()
def residual_mode_utilities(
    teacher: ConventionalSwiGLUMoE,
    *,
    max_rank: int,
) -> torch.Tensor:
    """Return scale-normalized marginal residual energy per expert and mode."""
    if max_rank <= 0 or max_rank > min(teacher.geometry.d_model, teacher.geometry.d_ff):
        raise ValueError("invalid max_rank")
    utilities = torch.zeros((teacher.geometry.n_experts, max_rank), dtype=torch.float64)
    for bank in (teacher.gate, teacher.up, teacher.down):
        mean = bank.mean(dim=0)
        for expert in range(teacher.geometry.n_experts):
            singular = torch.linalg.svdvals(bank[expert] - mean).double()
            scale = bank[expert].double().square().sum().clamp_min(EPS)
            available = min(max_rank, singular.numel())
            utilities[expert, :available] += singular[:available].square() / scale
    return utilities


def effective_residual_rank(
    teacher: ConventionalSwiGLUMoE,
    *,
    retained_fraction: float = 0.95,
) -> dict[str, list[int]]:
    if not 0.0 < retained_fraction <= 1.0:
        raise ValueError("retained_fraction must be in (0, 1]")
    result: dict[str, list[int]] = {}
    for name, bank in (("gate", teacher.gate), ("up", teacher.up), ("down", teacher.down)):
        mean = bank.mean(dim=0)
        ranks: list[int] = []
        for expert in range(teacher.geometry.n_experts):
            singular = torch.linalg.svdvals(bank[expert] - mean).double().square()
            total = singular.sum().clamp_min(EPS)
            cumulative = torch.cumsum(singular, dim=0) / total
            index = int(torch.searchsorted(cumulative, torch.tensor(retained_fraction, dtype=cumulative.dtype, device=cumulative.device)).item())
            ranks.append(min(index + 1, singular.numel()))
        result[name] = ranks
    return result


def allocate_static_ranks(
    marginal_utilities: torch.Tensor,
    route_frequency: Sequence[float] | torch.Tensor,
    *,
    total_rank_budget: int,
    expected_active_rank_budget: float,
    min_rank: int,
    max_rank: int,
) -> tuple[int, ...]:
    """Solve the nested-rank allocation exactly as a small binary MILP.

    Binary variable ``x[e,j]`` states whether mode ``j`` of expert ``e`` is
    retained. Prefix constraints enforce nested ranks. The equality constraint
    matches the uniform parameter budget exactly; the weighted inequality
    matches or improves its expected active-rank budget.
    """
    import numpy as np
    from scipy.optimize import Bounds, LinearConstraint, milp
    from scipy.sparse import lil_matrix, vstack

    utilities = torch.as_tensor(marginal_utilities, dtype=torch.float64)
    frequencies = torch.as_tensor(route_frequency, dtype=torch.float64)
    if utilities.ndim != 2 or utilities.shape[0] != frequencies.numel():
        raise ValueError("utility/frequency shape mismatch")
    n_experts = int(utilities.shape[0])
    if utilities.shape[1] < max_rank:
        raise ValueError("utility table does not cover max_rank")
    if not 1 <= min_rank <= max_rank:
        raise ValueError("invalid rank range")
    if total_rank_budget < n_experts * min_rank or total_rank_budget > n_experts * max_rank:
        raise ValueError("total rank budget is infeasible")
    if float(frequencies.sum()) <= 0.0:
        raise ValueError("route frequencies must have positive mass")

    variables = n_experts * max_rank
    objective = -(
        frequencies[:, None] * utilities[:, :max_rank]
    ).cpu().numpy().reshape(-1)
    lower = np.zeros(variables, dtype=np.float64)
    upper = np.ones(variables, dtype=np.float64)
    for expert in range(n_experts):
        left = expert * max_rank
        lower[left : left + min_rank] = 1.0
        upper[left : left + min_rank] = 1.0

    equality = np.ones((1, variables), dtype=np.float64)
    compute = np.repeat(frequencies.cpu().numpy(), max_rank)[None, :]
    prefix = lil_matrix((n_experts * (max_rank - 1), variables), dtype=np.float64)
    row = 0
    for expert in range(n_experts):
        base = expert * max_rank
        for mode in range(max_rank - 1):
            # x[e, mode+1] - x[e, mode] <= 0
            prefix[row, base + mode] = -1.0
            prefix[row, base + mode + 1] = 1.0
            row += 1

    constraints = [
        LinearConstraint(equality, total_rank_budget, total_rank_budget),
        LinearConstraint(compute, -np.inf, expected_active_rank_budget),
        LinearConstraint(prefix.tocsr(), -np.inf, 0.0),
    ]
    result = milp(
        c=objective,
        integrality=np.ones(variables, dtype=np.int8),
        bounds=Bounds(lower, upper),
        constraints=constraints,
        options={"mip_rel_gap": 0.0, "time_limit": 60.0, "presolve": True},
    )
    if not result.success or result.x is None:
        raise RuntimeError(f"heterogeneous rank MILP failed: {result.message}")
    selected = (result.x.reshape(n_experts, max_rank) >= 0.5).astype(np.int64)
    ranks = tuple(int(row_values.sum()) for row_values in selected)
    if sum(ranks) != total_rank_budget:
        raise AssertionError("MILP did not match the exact parameter budget")
    active = float(torch.dot(frequencies, torch.tensor(ranks, dtype=torch.float64)))
    if active > expected_active_rank_budget + 1e-7:
        raise AssertionError("MILP exceeded the expected active-rank budget")
    if any(rank < min_rank or rank > max_rank for rank in ranks):
        raise AssertionError("MILP produced an invalid rank")
    return ranks


def uniform_rank_vector(n_experts: int, rank: int) -> tuple[int, ...]:
    if n_experts <= 0 or rank <= 0:
        raise ValueError("n_experts and rank must be positive")
    return (int(rank),) * int(n_experts)
