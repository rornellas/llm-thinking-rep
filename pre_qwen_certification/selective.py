"""Selective hot/cold expert compression for controlled MoE transplantation.

The module preserves a training-selected subset of experts exactly ("hot") and
represents the remaining experts ("cold") with a shared scalar Modal bank.  The
selection uses only training captures.  Evaluation routes remain the frozen
teacher routes, so the experiment isolates heterogeneous expert capacity from
router adaptation.

This is an exploratory architecture.  It does not alter the frozen pre-Qwen
NO-GO decision without a new preregistered sealed replication.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import torch
from torch import nn
import torch.nn.functional as F

from .modal import (
    ConventionalSwiGLUMoE,
    MoEGeometry,
    Routing,
    ScalarModalMoE,
    _validate_forced_routing,
    route_topk,
)

ImportanceMetric = Literal[
    "weighted-output-energy",
    "gate-mass",
    "routing-frequency",
    "random",
]


@dataclass(frozen=True)
class ExpertImportance:
    routing_frequency: tuple[float, ...]
    gate_mass: tuple[float, ...]
    weighted_output_energy: tuple[float, ...]

    def vector(self, metric: ImportanceMetric) -> torch.Tensor:
        if metric == "routing-frequency":
            values = self.routing_frequency
        elif metric == "gate-mass":
            values = self.gate_mass
        elif metric == "weighted-output-energy":
            values = self.weighted_output_energy
        else:
            raise ValueError(f"random has no deterministic importance vector: {metric}")
        return torch.tensor(values, dtype=torch.float64)

    def as_dict(self) -> dict[str, list[float]]:
        return {
            "routing_frequency": list(self.routing_frequency),
            "gate_mass": list(self.gate_mass),
            "weighted_output_energy": list(self.weighted_output_energy),
        }


@torch.no_grad()
def score_expert_importance(
    teacher: ConventionalSwiGLUMoE,
    inputs: torch.Tensor,
    top_ids: torch.Tensor,
    route_weights: torch.Tensor,
    *,
    chunk_size: int = 512,
) -> ExpertImportance:
    """Measure expert contribution using only captured training activations.

    The primary score is the diagonal contribution proxy

        E[ || pi_e(x) f_e(x) ||_2^2 ],

    which combines routing frequency, gate weight, and activation/output
    strength.  It is not asserted to equal causal importance because cross-expert
    terms are omitted; random and simpler scoring controls are mandatory.

    Computation is chunked because materializing every selected expert matrix at
    once scales as ``tokens * top_k * d_ff * d_model``.  The accumulators are
    float64 and the result is invariant to chunk boundaries up to floating-point
    summation order.
    """
    geometry = teacher.geometry
    expected = (inputs.shape[0], geometry.top_k)
    if tuple(top_ids.shape) != expected or tuple(route_weights.shape) != expected:
        raise ValueError("captured routing shape does not match inputs")
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")

    frequency = torch.zeros(geometry.n_experts, dtype=torch.float64)
    gate_mass = torch.zeros(geometry.n_experts, dtype=torch.float64)
    energy = torch.zeros(geometry.n_experts, dtype=torch.float64)

    for offset in range(0, inputs.shape[0], chunk_size):
        x = inputs[offset : offset + chunk_size]
        ids2 = top_ids[offset : offset + chunk_size]
        weights = route_weights[offset : offset + chunk_size]
        selected_gate = teacher.gate.index_select(0, ids2.reshape(-1)).reshape(
            x.shape[0], geometry.top_k, geometry.d_ff, geometry.d_model
        )
        selected_up = teacher.up.index_select(0, ids2.reshape(-1)).reshape(
            x.shape[0], geometry.top_k, geometry.d_ff, geometry.d_model
        )
        selected_down = teacher.down.index_select(0, ids2.reshape(-1)).reshape(
            x.shape[0], geometry.top_k, geometry.d_model, geometry.d_ff
        )
        gate = torch.einsum("ntfd,nd->ntf", selected_gate, x)
        up = torch.einsum("ntfd,nd->ntf", selected_up, x)
        hidden = F.silu(gate) * up
        expert_output = torch.einsum("ntdf,ntf->ntd", selected_down, hidden)

        ids = ids2.reshape(-1)
        frequency += torch.bincount(ids, minlength=geometry.n_experts).double()
        gate_mass.index_add_(0, ids, weights.reshape(-1).double())
        contribution = (weights[..., None] * expert_output).square().sum(dim=-1)
        energy.index_add_(0, ids, contribution.reshape(-1).double())

    frequency /= max(float(top_ids.numel()), 1.0)
    gate_mass /= max(float(inputs.shape[0]), 1.0)
    energy /= max(float(inputs.shape[0]), 1.0)

    return ExpertImportance(
        routing_frequency=tuple(float(value) for value in frequency),
        gate_mass=tuple(float(value) for value in gate_mass),
        weighted_output_energy=tuple(float(value) for value in energy),
    )


def choose_hot_experts(
    importance: ExpertImportance,
    *,
    hot_count: int,
    metric: ImportanceMetric,
    seed: int,
) -> torch.Tensor:
    experts = len(importance.routing_frequency)
    if not 1 <= hot_count < experts:
        raise ValueError("hot_count must be in [1, n_experts-1]")
    if metric == "random":
        generator = torch.Generator().manual_seed(seed)
        selected = torch.randperm(experts, generator=generator)[:hot_count]
    else:
        values = importance.vector(metric)
        # Stable secondary key favors lower expert ID under exact ties.
        tie_break = torch.arange(experts, dtype=torch.float64) * 1e-15
        selected = torch.argsort(values - tie_break, descending=True)[:hot_count]
    return torch.sort(selected.long()).values


class SelectiveHotColdMoE(nn.Module):
    """Exact hot experts plus a scalar-Modal bank for cold experts.

    Hot expert tensors are registered as buffers, not trainable parameters.  This
    enforces the experimental intervention: their teacher functions remain exact
    while only the cold bank is distilled.  They are still counted in deployment
    parameter metrics.
    """

    def __init__(
        self,
        geometry: MoEGeometry,
        *,
        hot_expert_ids: torch.Tensor,
        cold_rank: int,
    ) -> None:
        super().__init__()
        geometry.validate()
        if cold_rank < 0:
            raise ValueError("cold_rank must be non-negative")
        hot = torch.as_tensor(hot_expert_ids, dtype=torch.long)
        if hot.ndim != 1 or hot.numel() == 0 or hot.numel() >= geometry.n_experts:
            raise ValueError("hot_expert_ids must be a non-empty proper subset")
        hot = torch.unique(hot, sorted=True)
        if hot.min() < 0 or hot.max() >= geometry.n_experts:
            raise ValueError("hot expert ID is out of range")
        all_ids = torch.arange(geometry.n_experts, dtype=torch.long)
        hot_mask = torch.zeros(geometry.n_experts, dtype=torch.bool)
        hot_mask[hot] = True
        cold = all_ids[~hot_mask]

        self.geometry = geometry
        self.cold_rank = int(cold_rank)
        self.cold_modes = self.cold_rank + 1
        self.router = nn.Linear(geometry.d_model, geometry.n_experts, bias=False)
        self.register_buffer("hot_expert_ids", hot)
        self.register_buffer("cold_expert_ids", cold)
        hot_lookup = torch.full((geometry.n_experts,), -1, dtype=torch.long)
        cold_lookup = torch.full((geometry.n_experts,), -1, dtype=torch.long)
        hot_lookup[hot] = torch.arange(hot.numel())
        cold_lookup[cold] = torch.arange(cold.numel())
        self.register_buffer("hot_lookup", hot_lookup)
        self.register_buffer("cold_lookup", cold_lookup)

        h, c, d, f, k = (
            hot.numel(),
            cold.numel(),
            geometry.d_model,
            geometry.d_ff,
            self.cold_rank,
        )
        self.register_buffer("hot_gate", torch.empty(h, f, d))
        self.register_buffer("hot_up", torch.empty(h, f, d))
        self.register_buffer("hot_down", torch.empty(h, d, f))
        self.cold_gate_modes = nn.Parameter(torch.empty(k + 1, f, d))
        self.cold_up_modes = nn.Parameter(torch.empty(k + 1, f, d))
        self.cold_down_modes = nn.Parameter(torch.empty(k + 1, d, f))
        self.cold_gate_codes = nn.Parameter(torch.empty(c, k))
        self.cold_up_codes = nn.Parameter(torch.empty(c, k))
        self.cold_down_codes = nn.Parameter(torch.empty(c, k))
        self.reset_parameters()

    @property
    def hot_count(self) -> int:
        return int(self.hot_expert_ids.numel())

    @property
    def cold_count(self) -> int:
        return int(self.cold_expert_ids.numel())

    def reset_parameters(self) -> None:
        nn.init.normal_(self.router.weight, std=0.02)
        for bank in (
            self.hot_gate,
            self.hot_up,
            self.hot_down,
            self.cold_gate_modes,
            self.cold_up_modes,
            self.cold_down_modes,
        ):
            for matrix in bank:
                nn.init.xavier_uniform_(matrix)
        for codes in (
            self.cold_gate_codes,
            self.cold_up_codes,
            self.cold_down_codes,
        ):
            if codes.numel():
                nn.init.normal_(codes, std=0.7)

    def _full_codes(self, codes: torch.Tensor) -> torch.Tensor:
        common = torch.ones(
            codes.shape[0], 1, dtype=codes.dtype, device=codes.device
        )
        return torch.cat([common, codes], dim=-1)

    def reconstruct_weights(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        g = self.geometry
        gate = torch.empty(
            g.n_experts, g.d_ff, g.d_model,
            dtype=self.cold_gate_modes.dtype,
            device=self.cold_gate_modes.device,
        )
        up = torch.empty_like(gate)
        down = torch.empty(
            g.n_experts, g.d_model, g.d_ff,
            dtype=self.cold_down_modes.dtype,
            device=self.cold_down_modes.device,
        )
        gate.index_copy_(0, self.hot_expert_ids, self.hot_gate)
        up.index_copy_(0, self.hot_expert_ids, self.hot_up)
        down.index_copy_(0, self.hot_expert_ids, self.hot_down)
        cold_gate = torch.einsum(
            "em,mfd->efd",
            self._full_codes(self.cold_gate_codes),
            self.cold_gate_modes,
        )
        cold_up = torch.einsum(
            "em,mfd->efd",
            self._full_codes(self.cold_up_codes),
            self.cold_up_modes,
        )
        cold_down = torch.einsum(
            "em,mdf->edf",
            self._full_codes(self.cold_down_codes),
            self.cold_down_modes,
        )
        gate.index_copy_(0, self.cold_expert_ids, cold_gate)
        up.index_copy_(0, self.cold_expert_ids, cold_up)
        down.index_copy_(0, self.cold_expert_ids, cold_down)
        return gate, up, down

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

        hot_indices = self.hot_lookup.index_select(0, top_ids.reshape(-1)).reshape_as(top_ids)
        hot_mask = hot_indices >= 0
        output = torch.zeros_like(x)
        # Compute only actual hot routes.  The earlier vectorized reference
        # evaluated a hot matrix for every route slot and masked cold slots
        # afterward, which was functionally correct but did not match the
        # route-conditioned operation-count model used by this experiment.
        hot_where = torch.nonzero(hot_mask, as_tuple=False)
        if hot_where.numel():
            token_ids, slots = hot_where[:, 0], hot_where[:, 1]
            selected_x = x.index_select(0, token_ids)
            local_ids = hot_indices[token_ids, slots]
            selected_gate = self.hot_gate.index_select(0, local_ids)
            selected_up = self.hot_up.index_select(0, local_ids)
            selected_down = self.hot_down.index_select(0, local_ids)
            hot_gate = torch.einsum("nfd,nd->nf", selected_gate, selected_x)
            hot_up = torch.einsum("nfd,nd->nf", selected_up, selected_x)
            hot_hidden = F.silu(hot_gate) * hot_up
            hot_output = torch.einsum("ndf,nf->nd", selected_down, hot_hidden)
            weighted = hot_output * weights[token_ids, slots, None]
            output.index_add_(0, token_ids, weighted)

        cold_mask = ~hot_mask
        token_mask = cold_mask.any(dim=-1)
        if token_mask.any():
            token_ids = torch.nonzero(token_mask, as_tuple=False).reshape(-1)
            cold_x = x.index_select(0, token_ids)
            cold_top_ids = top_ids.index_select(0, token_ids)
            cold_weights = weights.index_select(0, token_ids)
            cold_slot_mask = cold_mask.index_select(0, token_ids)
            cold_indices = self.cold_lookup.index_select(
                0, cold_top_ids.reshape(-1)
            ).reshape_as(cold_top_ids).clamp_min(0)

            gate_projected = torch.einsum(
                "nd,mfd->nmf", cold_x, self.cold_gate_modes
            )
            up_projected = torch.einsum(
                "nd,mfd->nmf", cold_x, self.cold_up_modes
            )
            code_shape = (
                cold_x.shape[0], g.top_k, self.cold_modes
            )
            gate_codes = self._full_codes(self.cold_gate_codes).index_select(
                0, cold_indices.reshape(-1)
            ).reshape(code_shape)
            up_codes = self._full_codes(self.cold_up_codes).index_select(
                0, cold_indices.reshape(-1)
            ).reshape(code_shape)
            down_codes = self._full_codes(self.cold_down_codes).index_select(
                0, cold_indices.reshape(-1)
            ).reshape(code_shape)
            gate_values = torch.einsum(
                "nmf,ntm->ntf", gate_projected, gate_codes
            )
            up_values = torch.einsum(
                "nmf,ntm->ntf", up_projected, up_codes
            )
            hidden = F.silu(gate_values) * up_values
            effective_weights = cold_weights * cold_slot_mask.to(cold_weights.dtype)
            mode_inputs = torch.einsum(
                "nt,ntm,ntf->nmf", effective_weights, down_codes, hidden
            )
            cold_output = torch.einsum(
                "mdf,nmf->nd", self.cold_down_modes, mode_inputs
            )
            output = output.index_add(0, token_ids, cold_output)

        return output, Routing(natural.logits, top_ids, weights)

    @torch.no_grad()
    def reference_reconstructed(
        self,
        x: torch.Tensor,
        *,
        forced_top_ids: torch.Tensor | None = None,
        forced_weights: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, Routing]:
        conventional = ConventionalSwiGLUMoE(self.geometry).to(
            device=x.device, dtype=x.dtype
        )
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

    @classmethod
    def from_conventional(
        cls,
        teacher: ConventionalSwiGLUMoE,
        *,
        hot_expert_ids: torch.Tensor,
        cold_rank: int,
        freeze_router: bool = True,
    ) -> tuple["SelectiveHotColdMoE", dict[str, Any]]:
        student = cls(
            teacher.geometry,
            hot_expert_ids=hot_expert_ids,
            cold_rank=cold_rank,
        )
        cold = student.cold_expert_ids
        gate_modes, gate_codes, gate_error = ScalarModalMoE._svd_bank(
            teacher.gate.detach().index_select(0, cold), cold_rank
        )
        up_modes, up_codes, up_error = ScalarModalMoE._svd_bank(
            teacher.up.detach().index_select(0, cold), cold_rank
        )
        down_modes, down_codes, down_error = ScalarModalMoE._svd_bank(
            teacher.down.detach().index_select(0, cold), cold_rank
        )
        with torch.no_grad():
            student.router.weight.copy_(teacher.router.weight)
            student.hot_gate.copy_(
                teacher.gate.detach().index_select(0, student.hot_expert_ids)
            )
            student.hot_up.copy_(
                teacher.up.detach().index_select(0, student.hot_expert_ids)
            )
            student.hot_down.copy_(
                teacher.down.detach().index_select(0, student.hot_expert_ids)
            )
            student.cold_gate_modes.copy_(gate_modes)
            student.cold_up_modes.copy_(up_modes)
            student.cold_down_modes.copy_(down_modes)
            if cold_rank:
                student.cold_gate_codes.copy_(gate_codes)
                student.cold_up_codes.copy_(up_codes)
                student.cold_down_codes.copy_(down_codes)
        if freeze_router:
            student.router.weight.requires_grad_(False)
        return student, {
            "method": "exact-hot-plus-cold-scalar-modal-svd",
            "hot_expert_ids": student.hot_expert_ids.tolist(),
            "cold_expert_ids": student.cold_expert_ids.tolist(),
            "cold_rank": cold_rank,
            "gate_cold_relative_error": gate_error,
            "up_cold_relative_error": up_error,
            "down_cold_relative_error": down_error,
            "max_cold_relative_error": max(gate_error, up_error, down_error),
        }

    def expert_transform_parameter_count(self) -> int:
        return int(
            self.hot_gate.numel()
            + self.hot_up.numel()
            + self.hot_down.numel()
            + self.cold_gate_modes.numel()
            + self.cold_up_modes.numel()
            + self.cold_down_modes.numel()
            + self.cold_gate_codes.numel()
            + self.cold_up_codes.numel()
            + self.cold_down_codes.numel()
        )

    def route_cost_metrics(
        self,
        top_ids: torch.Tensor,
        route_weights: torch.Tensor | None = None,
    ) -> dict[str, float]:
        if top_ids.ndim != 2 or top_ids.shape[1] != self.geometry.top_k:
            raise ValueError("top_ids has incompatible shape")
        if route_weights is not None and route_weights.shape != top_ids.shape:
            raise ValueError("route_weights has incompatible shape")
        hot = self.hot_lookup.index_select(0, top_ids.reshape(-1)).reshape_as(top_ids) >= 0
        hot_slots = hot.sum(dim=-1).double()
        cold_slots = (~hot).sum(dim=-1).double()
        cold_any = cold_slots > 0
        dominant = (
            hot_slots + self.cold_modes * cold_any.double()
        ) / self.geometry.top_k
        cold_fraction = cold_slots / self.geometry.top_k
        code_overhead = cold_fraction * self.cold_rank / self.geometry.d_model
        result = {
            "mean_hot_slots": float(hot_slots.mean()),
            "mean_cold_slots": float(cold_slots.mean()),
            "hot_slot_fraction": float(hot.double().mean()),
            "cold_token_fraction": float(cold_any.double().mean()),
            "dominant_matrix_compute_ratio": float(dominant.mean()),
            "code_adjusted_compute_ratio": float((dominant + code_overhead).mean()),
            "p95_dominant_matrix_compute_ratio": float(
                torch.quantile(dominant, 0.95)
            ),
        }
        if route_weights is not None:
            hot_mass = (route_weights.double() * hot.double()).sum(dim=-1)
            result.update(
                {
                    "mean_hot_route_mass": float(hot_mass.mean()),
                    "p05_hot_route_mass": float(torch.quantile(hot_mass, 0.05)),
                }
            )
        return result
