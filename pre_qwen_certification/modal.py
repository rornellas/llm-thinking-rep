"""Core conventional and scalar Modal-MoE implementations used for certification.

The implementation deliberately keeps the router identical across teacher and
student.  This isolates expert-function representability from router learning.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import random
from typing import Iterable

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


@dataclass(frozen=True)
class MoEGeometry:
    d_model: int
    d_ff: int
    n_experts: int
    top_k: int

    def validate(self) -> None:
        if self.d_model <= 0 or self.d_ff <= 0:
            raise ValueError("model dimensions must be positive")
        if self.n_experts <= 1:
            raise ValueError("n_experts must be greater than one")
        if not 1 <= self.top_k <= self.n_experts:
            raise ValueError("top_k must be in [1, n_experts]")


@dataclass
class Routing:
    logits: torch.Tensor
    top_ids: torch.Tensor
    weights: torch.Tensor


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def route_topk(
    x: torch.Tensor,
    router_weight: torch.Tensor,
    top_k: int,
) -> Routing:
    if x.ndim != 2:
        raise ValueError(f"expected [tokens, d_model], got {tuple(x.shape)}")
    logits = F.linear(x, router_weight)
    values, top_ids = torch.topk(logits, top_k, dim=-1)
    weights = F.softmax(values, dim=-1)
    return Routing(logits=logits, top_ids=top_ids, weights=weights)


def _validate_forced_routing(
    x: torch.Tensor,
    forced_top_ids: torch.Tensor | None,
    forced_weights: torch.Tensor | None,
    top_k: int,
) -> None:
    if (forced_top_ids is None) != (forced_weights is None):
        raise ValueError("forced_top_ids and forced_weights must be provided together")
    if forced_top_ids is None:
        return
    expected = (x.shape[0], top_k)
    if tuple(forced_top_ids.shape) != expected or tuple(forced_weights.shape) != expected:
        raise ValueError(
            f"forced routing must have shape {expected}; got "
            f"{tuple(forced_top_ids.shape)} and {tuple(forced_weights.shape)}"
        )


class ConventionalSwiGLUMoE(nn.Module):
    """Vectorized conventional sparse SwiGLU MoE.

    Weight shapes follow the common convention:

    * gate/up: ``[experts, d_ff, d_model]``
    * down: ``[experts, d_model, d_ff]``
    """

    def __init__(self, geometry: MoEGeometry) -> None:
        super().__init__()
        geometry.validate()
        self.geometry = geometry
        self.router = nn.Linear(geometry.d_model, geometry.n_experts, bias=False)
        self.gate = nn.Parameter(
            torch.empty(geometry.n_experts, geometry.d_ff, geometry.d_model)
        )
        self.up = nn.Parameter(
            torch.empty(geometry.n_experts, geometry.d_ff, geometry.d_model)
        )
        self.down = nn.Parameter(
            torch.empty(geometry.n_experts, geometry.d_model, geometry.d_ff)
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.router.weight, std=0.02)
        for bank in (self.gate, self.up, self.down):
            for matrix in bank:
                nn.init.xavier_uniform_(matrix)

    def forward(
        self,
        x: torch.Tensor,
        *,
        forced_top_ids: torch.Tensor | None = None,
        forced_weights: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, Routing]:
        g = self.geometry
        _validate_forced_routing(
            x, forced_top_ids, forced_weights, g.top_k
        )
        natural = route_topk(x, self.router.weight, g.top_k)
        top_ids = natural.top_ids if forced_top_ids is None else forced_top_ids
        weights = natural.weights if forced_weights is None else forced_weights

        selected_gate = self.gate.index_select(0, top_ids.reshape(-1)).reshape(
            x.shape[0], g.top_k, g.d_ff, g.d_model
        )
        selected_up = self.up.index_select(0, top_ids.reshape(-1)).reshape(
            x.shape[0], g.top_k, g.d_ff, g.d_model
        )
        selected_down = self.down.index_select(0, top_ids.reshape(-1)).reshape(
            x.shape[0], g.top_k, g.d_model, g.d_ff
        )

        gate_values = torch.einsum("ntfd,nd->ntf", selected_gate, x)
        up_values = torch.einsum("ntfd,nd->ntf", selected_up, x)
        hidden = F.silu(gate_values) * up_values
        expert_outputs = torch.einsum("ntdf,ntf->ntd", selected_down, hidden)
        output = torch.einsum("nt,ntd->nd", weights, expert_outputs)
        return output, Routing(natural.logits, top_ids, weights)

    @classmethod
    def from_modal(cls, modal: "ScalarModalMoE") -> "ConventionalSwiGLUMoE":
        teacher = cls(modal.geometry)
        gate, up, down = modal.reconstruct_weights()
        with torch.no_grad():
            teacher.router.weight.copy_(modal.router.weight)
            teacher.gate.copy_(gate)
            teacher.up.copy_(up)
            teacher.down.copy_(down)
        return teacher


class ScalarModalMoE(nn.Module):
    """Scalar-code Modal-MoE with direct fused execution.

    ``rank`` is the number of residual modes.  The common mode is always present,
    therefore ``n_modes = rank + 1``.
    """

    def __init__(self, geometry: MoEGeometry, rank: int) -> None:
        super().__init__()
        geometry.validate()
        if rank < 0:
            raise ValueError("rank must be non-negative")
        self.geometry = geometry
        self.rank = int(rank)
        self.n_modes = self.rank + 1
        self.router = nn.Linear(geometry.d_model, geometry.n_experts, bias=False)
        self.gate_modes = nn.Parameter(
            torch.empty(self.n_modes, geometry.d_ff, geometry.d_model)
        )
        self.up_modes = nn.Parameter(
            torch.empty(self.n_modes, geometry.d_ff, geometry.d_model)
        )
        self.down_modes = nn.Parameter(
            torch.empty(self.n_modes, geometry.d_model, geometry.d_ff)
        )
        self.gate_codes = nn.Parameter(
            torch.empty(geometry.n_experts, self.rank)
        )
        self.up_codes = nn.Parameter(
            torch.empty(geometry.n_experts, self.rank)
        )
        self.down_codes = nn.Parameter(
            torch.empty(geometry.n_experts, self.rank)
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.router.weight, std=0.02)
        for bank in (self.gate_modes, self.up_modes, self.down_modes):
            for index, matrix in enumerate(bank):
                nn.init.xavier_uniform_(matrix)
                if index > 0:
                    matrix.data.mul_(0.5 / math.sqrt(max(self.rank, 1)))
        for codes in (self.gate_codes, self.up_codes, self.down_codes):
            if codes.numel():
                nn.init.normal_(codes, std=0.7)

    def full_codes(self, codes: torch.Tensor) -> torch.Tensor:
        common = torch.ones(
            codes.shape[0], 1, dtype=codes.dtype, device=codes.device
        )
        return torch.cat([common, codes], dim=-1)

    def reconstruct_weights(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        cg = self.full_codes(self.gate_codes)
        cu = self.full_codes(self.up_codes)
        cd = self.full_codes(self.down_codes)
        gate = torch.einsum("em,mfd->efd", cg, self.gate_modes)
        up = torch.einsum("em,mfd->efd", cu, self.up_modes)
        down = torch.einsum("em,mdf->edf", cd, self.down_modes)
        return gate, up, down

    def forward(
        self,
        x: torch.Tensor,
        *,
        forced_top_ids: torch.Tensor | None = None,
        forced_weights: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, Routing]:
        g = self.geometry
        _validate_forced_routing(
            x, forced_top_ids, forced_weights, g.top_k
        )
        natural = route_topk(x, self.router.weight, g.top_k)
        top_ids = natural.top_ids if forced_top_ids is None else forced_top_ids
        weights = natural.weights if forced_weights is None else forced_weights

        gate_projected = torch.einsum("nd,mfd->nmf", x, self.gate_modes)
        up_projected = torch.einsum("nd,mfd->nmf", x, self.up_modes)
        gate_codes = self.full_codes(self.gate_codes).index_select(
            0, top_ids.reshape(-1)
        ).reshape(x.shape[0], g.top_k, self.n_modes)
        up_codes = self.full_codes(self.up_codes).index_select(
            0, top_ids.reshape(-1)
        ).reshape(x.shape[0], g.top_k, self.n_modes)
        down_codes = self.full_codes(self.down_codes).index_select(
            0, top_ids.reshape(-1)
        ).reshape(x.shape[0], g.top_k, self.n_modes)

        gate_values = torch.einsum("nmf,ntm->ntf", gate_projected, gate_codes)
        up_values = torch.einsum("nmf,ntm->ntf", up_projected, up_codes)
        hidden = F.silu(gate_values) * up_values

        mode_inputs = torch.einsum(
            "nt,ntm,ntf->nmf", weights, down_codes, hidden
        )
        output = torch.einsum("mdf,nmf->nd", self.down_modes, mode_inputs)
        return output, Routing(natural.logits, top_ids, weights)

    @torch.no_grad()
    def reference_reconstructed(
        self,
        x: torch.Tensor,
        *,
        forced_top_ids: torch.Tensor | None = None,
        forced_weights: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, Routing]:
        conventional = ConventionalSwiGLUMoE.from_modal(self)
        conventional = conventional.to(device=x.device, dtype=x.dtype)
        return conventional(
            x,
            forced_top_ids=forced_top_ids,
            forced_weights=forced_weights,
        )

    @staticmethod
    def _svd_bank(
        weights: torch.Tensor,
        rank: int,
    ) -> tuple[torch.Tensor, torch.Tensor, float]:
        experts = weights.shape[0]
        flat = weights.reshape(experts, -1).double()
        mean = flat.mean(dim=0, keepdim=True)
        centered = flat - mean
        if rank == 0:
            reconstructed = mean.expand_as(flat)
            denominator = torch.linalg.vector_norm(flat).clamp_min(1e-12)
            error = float(
                torch.linalg.vector_norm(flat - reconstructed) / denominator
            )
            modes = mean.reshape(1, *weights.shape[1:])
            codes = torch.empty(experts, 0, dtype=weights.dtype)
            return modes.to(weights.dtype), codes, error

        _, singular_values, vh = torch.linalg.svd(centered, full_matrices=False)
        usable = min(rank, vh.shape[0])
        residual_modes = vh[:usable]
        # U*S is recovered without retaining U explicitly: centered @ V^T.
        residual_codes = centered @ residual_modes.T
        if usable < rank:
            residual_modes = torch.cat(
                [
                    residual_modes,
                    torch.zeros(
                        rank - usable,
                        residual_modes.shape[1],
                        dtype=residual_modes.dtype,
                    ),
                ],
                dim=0,
            )
            residual_codes = torch.cat(
                [
                    residual_codes,
                    torch.zeros(
                        experts,
                        rank - usable,
                        dtype=residual_codes.dtype,
                    ),
                ],
                dim=1,
            )
        reconstructed = mean + residual_codes[:, :usable] @ vh[:usable]
        denominator = torch.linalg.vector_norm(flat).clamp_min(1e-12)
        error = float(torch.linalg.vector_norm(flat - reconstructed) / denominator)
        modes = torch.cat([mean, residual_modes], dim=0).reshape(
            rank + 1, *weights.shape[1:]
        )
        return (
            modes.to(dtype=weights.dtype),
            residual_codes.to(dtype=weights.dtype),
            error,
        )

    @classmethod
    def from_conventional_svd(
        cls,
        teacher: ConventionalSwiGLUMoE,
        rank: int,
        *,
        freeze_router: bool = True,
    ) -> tuple["ScalarModalMoE", dict[str, float]]:
        student = cls(teacher.geometry, rank)
        gate_modes, gate_codes, gate_error = cls._svd_bank(teacher.gate.detach(), rank)
        up_modes, up_codes, up_error = cls._svd_bank(teacher.up.detach(), rank)
        down_modes, down_codes, down_error = cls._svd_bank(teacher.down.detach(), rank)
        with torch.no_grad():
            student.router.weight.copy_(teacher.router.weight)
            student.gate_modes.copy_(gate_modes)
            student.up_modes.copy_(up_modes)
            student.down_modes.copy_(down_modes)
            if rank:
                student.gate_codes.copy_(gate_codes)
                student.up_codes.copy_(up_codes)
                student.down_codes.copy_(down_codes)
        if freeze_router:
            student.router.weight.requires_grad_(False)
        return student, {
            "gate_weight_relative_error": gate_error,
            "up_weight_relative_error": up_error,
            "down_weight_relative_error": down_error,
            "max_weight_relative_error": max(gate_error, up_error, down_error),
        }

    def expert_transform_parameter_count(self) -> int:
        tensors: Iterable[torch.Tensor] = (
            self.gate_modes,
            self.up_modes,
            self.down_modes,
            self.gate_codes,
            self.up_codes,
            self.down_codes,
        )
        return sum(t.numel() for t in tensors)

    def idealized_expert_compute_ratio(self) -> float:
        return self.n_modes / self.geometry.top_k + self.rank / self.geometry.d_model


class NeuronwiseModalMoE(nn.Module):
    """Modal-MoE with expert/mode coefficients per SwiGLU hidden neuron.

    The shared matrices remain full rank.  Additional capacity is carried by
    diagonal coefficients ``[expert, residual_mode, d_ff]``.  This matches the
    project's neuron-wise formulation while preserving fused down aggregation.
    """

    def __init__(self, geometry: MoEGeometry, rank: int) -> None:
        super().__init__()
        geometry.validate()
        if rank < 0:
            raise ValueError("rank must be non-negative")
        self.geometry = geometry
        self.rank = int(rank)
        self.n_modes = self.rank + 1
        self.router = nn.Linear(geometry.d_model, geometry.n_experts, bias=False)
        self.gate_modes = nn.Parameter(
            torch.empty(self.n_modes, geometry.d_ff, geometry.d_model)
        )
        self.up_modes = nn.Parameter(
            torch.empty(self.n_modes, geometry.d_ff, geometry.d_model)
        )
        self.down_modes = nn.Parameter(
            torch.empty(self.n_modes, geometry.d_model, geometry.d_ff)
        )
        self.gate_codes = nn.Parameter(
            torch.empty(geometry.n_experts, self.rank, geometry.d_ff)
        )
        self.up_codes = nn.Parameter(
            torch.empty(geometry.n_experts, self.rank, geometry.d_ff)
        )
        self.down_codes = nn.Parameter(
            torch.empty(geometry.n_experts, self.rank, geometry.d_ff)
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.router.weight, std=0.02)
        for bank in (self.gate_modes, self.up_modes, self.down_modes):
            for index, matrix in enumerate(bank):
                nn.init.xavier_uniform_(matrix)
                if index > 0:
                    matrix.data.mul_(0.5 / math.sqrt(max(self.rank, 1)))
        for codes in (self.gate_codes, self.up_codes, self.down_codes):
            if codes.numel():
                nn.init.normal_(codes, std=0.7)

    def full_codes(self, codes: torch.Tensor) -> torch.Tensor:
        common = torch.ones(
            codes.shape[0],
            1,
            self.geometry.d_ff,
            dtype=codes.dtype,
            device=codes.device,
        )
        return torch.cat([common, codes], dim=1)

    def reconstruct_weights(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        gate = torch.einsum(
            "emf,mfd->efd", self.full_codes(self.gate_codes), self.gate_modes
        )
        up = torch.einsum(
            "emf,mfd->efd", self.full_codes(self.up_codes), self.up_modes
        )
        down = torch.einsum(
            "emf,mdf->edf", self.full_codes(self.down_codes), self.down_modes
        )
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

        gate_projected = torch.einsum("nd,mfd->nmf", x, self.gate_modes)
        up_projected = torch.einsum("nd,mfd->nmf", x, self.up_modes)
        selected_shape = (x.shape[0], g.top_k, self.n_modes, g.d_ff)
        gate_codes = self.full_codes(self.gate_codes).index_select(
            0, top_ids.reshape(-1)
        ).reshape(selected_shape)
        up_codes = self.full_codes(self.up_codes).index_select(
            0, top_ids.reshape(-1)
        ).reshape(selected_shape)
        down_codes = self.full_codes(self.down_codes).index_select(
            0, top_ids.reshape(-1)
        ).reshape(selected_shape)
        gate_values = torch.einsum("nmf,ntmf->ntf", gate_projected, gate_codes)
        up_values = torch.einsum("nmf,ntmf->ntf", up_projected, up_codes)
        hidden = F.silu(gate_values) * up_values
        mode_inputs = torch.einsum(
            "nt,ntmf,ntf->nmf", weights, down_codes, hidden
        )
        output = torch.einsum("mdf,nmf->nd", self.down_modes, mode_inputs)
        return output, Routing(natural.logits, top_ids, weights)

    @staticmethod
    def _svd_rows(
        weights: torch.Tensor,
        rank: int,
        *,
        down: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, float]:
        # gate/up: [E, F, D]; down: [E, D, F]
        experts = weights.shape[0]
        d_ff = weights.shape[-1] if down else weights.shape[1]
        d_model = weights.shape[1] if down else weights.shape[2]
        if down:
            modes = torch.zeros(
                rank + 1, d_model, d_ff, dtype=torch.float64
            )
        else:
            modes = torch.zeros(
                rank + 1, d_ff, d_model, dtype=torch.float64
            )
        codes = torch.zeros(experts, rank, d_ff, dtype=torch.float64)
        reconstructed = torch.zeros_like(weights, dtype=torch.float64)
        source = weights.detach().double()
        for neuron in range(d_ff):
            matrix = source[:, :, neuron] if down else source[:, neuron, :]
            mean = matrix.mean(dim=0, keepdim=True)
            centered = matrix - mean
            if rank:
                _, _, vh = torch.linalg.svd(centered, full_matrices=False)
                usable = min(rank, vh.shape[0])
                residual_modes = vh[:usable]
                residual_codes = centered @ residual_modes.T
            else:
                usable = 0
                residual_modes = torch.empty(0, d_model, dtype=torch.float64)
                residual_codes = torch.empty(experts, 0, dtype=torch.float64)
            if down:
                modes[0, :, neuron] = mean[0]
                if usable:
                    modes[1 : usable + 1, :, neuron] = residual_modes
                    codes[:, :usable, neuron] = residual_codes
                reconstructed[:, :, neuron] = mean + residual_codes @ residual_modes
            else:
                modes[0, neuron, :] = mean[0]
                if usable:
                    modes[1 : usable + 1, neuron, :] = residual_modes
                    codes[:, :usable, neuron] = residual_codes
                reconstructed[:, neuron, :] = mean + residual_codes @ residual_modes
        error = float(
            torch.linalg.vector_norm(source - reconstructed)
            / torch.linalg.vector_norm(source).clamp_min(1e-12)
        )
        return modes.to(weights.dtype), codes.to(weights.dtype), error

    @classmethod
    def from_conventional_svd(
        cls,
        teacher: ConventionalSwiGLUMoE,
        rank: int,
        *,
        freeze_router: bool = True,
    ) -> tuple["NeuronwiseModalMoE", dict[str, float]]:
        student = cls(teacher.geometry, rank)
        gate_modes, gate_codes, gate_error = cls._svd_rows(
            teacher.gate, rank, down=False
        )
        up_modes, up_codes, up_error = cls._svd_rows(
            teacher.up, rank, down=False
        )
        down_modes, down_codes, down_error = cls._svd_rows(
            teacher.down, rank, down=True
        )
        with torch.no_grad():
            student.router.weight.copy_(teacher.router.weight)
            student.gate_modes.copy_(gate_modes)
            student.up_modes.copy_(up_modes)
            student.down_modes.copy_(down_modes)
            if rank:
                student.gate_codes.copy_(gate_codes)
                student.up_codes.copy_(up_codes)
                student.down_codes.copy_(down_codes)
        if freeze_router:
            student.router.weight.requires_grad_(False)
        return student, {
            "gate_weight_relative_error": gate_error,
            "up_weight_relative_error": up_error,
            "down_weight_relative_error": down_error,
            "max_weight_relative_error": max(gate_error, up_error, down_error),
        }

    def expert_transform_parameter_count(self) -> int:
        return sum(
            tensor.numel()
            for tensor in (
                self.gate_modes,
                self.up_modes,
                self.down_modes,
                self.gate_codes,
                self.up_codes,
                self.down_codes,
            )
        )

    def idealized_expert_compute_ratio(self) -> float:
        return self.n_modes / self.geometry.top_k + self.rank / self.geometry.d_model


class AsymmetricScalarModalMoE(nn.Module):
    """Scalar Modal-MoE with separate residual ranks for gate, up, and down.

    The symmetric scalar model uses the same residual rank ``K`` for all three
    SwiGLU projections.  This class preserves the same fused execution algebra
    while allowing a fixed total mode budget to be allocated according to
    functional sensitivity.  For ranks ``(K_g, K_u, K_d)`` the dominant matrix
    compute ratio relative to a conventional top-T expert path is

    ``((K_g + 1) + (K_u + 1) + (K_d + 1)) / (3 T)``.

    The elementwise expert-code modulation adds the proxy
    ``(K_g + K_u + K_d) / (3 d_model)``.  These are analytic operation-count
    proxies, not measured runtime claims.
    """

    def __init__(
        self,
        geometry: MoEGeometry,
        ranks: tuple[int, int, int],
    ) -> None:
        super().__init__()
        geometry.validate()
        if len(ranks) != 3 or any(int(rank) < 0 for rank in ranks):
            raise ValueError("ranks must be a non-negative (gate, up, down) tuple")
        self.geometry = geometry
        self.gate_rank, self.up_rank, self.down_rank = (
            int(value) for value in ranks
        )
        self.ranks = (self.gate_rank, self.up_rank, self.down_rank)
        self.router = nn.Linear(geometry.d_model, geometry.n_experts, bias=False)
        self.gate_modes = nn.Parameter(
            torch.empty(self.gate_rank + 1, geometry.d_ff, geometry.d_model)
        )
        self.up_modes = nn.Parameter(
            torch.empty(self.up_rank + 1, geometry.d_ff, geometry.d_model)
        )
        self.down_modes = nn.Parameter(
            torch.empty(self.down_rank + 1, geometry.d_model, geometry.d_ff)
        )
        self.gate_codes = nn.Parameter(
            torch.empty(geometry.n_experts, self.gate_rank)
        )
        self.up_codes = nn.Parameter(
            torch.empty(geometry.n_experts, self.up_rank)
        )
        self.down_codes = nn.Parameter(
            torch.empty(geometry.n_experts, self.down_rank)
        )
        self.reset_parameters()

    @staticmethod
    def _full_codes(codes: torch.Tensor) -> torch.Tensor:
        common = torch.ones(
            codes.shape[0], 1, dtype=codes.dtype, device=codes.device
        )
        return torch.cat([common, codes], dim=-1)

    def reset_parameters(self) -> None:
        nn.init.normal_(self.router.weight, std=0.02)
        for bank, rank in (
            (self.gate_modes, self.gate_rank),
            (self.up_modes, self.up_rank),
            (self.down_modes, self.down_rank),
        ):
            for index, matrix in enumerate(bank):
                nn.init.xavier_uniform_(matrix)
                if index > 0:
                    matrix.data.mul_(0.5 / math.sqrt(max(rank, 1)))
        for codes in (self.gate_codes, self.up_codes, self.down_codes):
            if codes.numel():
                nn.init.normal_(codes, std=0.7)

    def reconstruct_weights(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        gate = torch.einsum(
            "em,mfd->efd", self._full_codes(self.gate_codes), self.gate_modes
        )
        up = torch.einsum(
            "em,mfd->efd", self._full_codes(self.up_codes), self.up_modes
        )
        down = torch.einsum(
            "em,mdf->edf", self._full_codes(self.down_codes), self.down_modes
        )
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

        gate_projected = torch.einsum("nd,mfd->nmf", x, self.gate_modes)
        up_projected = torch.einsum("nd,mfd->nmf", x, self.up_modes)
        gate_codes = self._full_codes(self.gate_codes).index_select(
            0, top_ids.reshape(-1)
        ).reshape(x.shape[0], g.top_k, self.gate_rank + 1)
        up_codes = self._full_codes(self.up_codes).index_select(
            0, top_ids.reshape(-1)
        ).reshape(x.shape[0], g.top_k, self.up_rank + 1)
        down_codes = self._full_codes(self.down_codes).index_select(
            0, top_ids.reshape(-1)
        ).reshape(x.shape[0], g.top_k, self.down_rank + 1)

        gate_values = torch.einsum(
            "nmf,ntm->ntf", gate_projected, gate_codes
        )
        up_values = torch.einsum("nmf,ntm->ntf", up_projected, up_codes)
        hidden = F.silu(gate_values) * up_values
        mode_inputs = torch.einsum(
            "nt,ntm,ntf->nmf", weights, down_codes, hidden
        )
        output = torch.einsum("mdf,nmf->nd", self.down_modes, mode_inputs)
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

    @classmethod
    def from_conventional_svd(
        cls,
        teacher: ConventionalSwiGLUMoE,
        ranks: tuple[int, int, int],
        *,
        freeze_router: bool = True,
    ) -> tuple["AsymmetricScalarModalMoE", dict[str, float]]:
        student = cls(teacher.geometry, ranks)
        gate_modes, gate_codes, gate_error = ScalarModalMoE._svd_bank(
            teacher.gate.detach(), student.gate_rank
        )
        up_modes, up_codes, up_error = ScalarModalMoE._svd_bank(
            teacher.up.detach(), student.up_rank
        )
        down_modes, down_codes, down_error = ScalarModalMoE._svd_bank(
            teacher.down.detach(), student.down_rank
        )
        with torch.no_grad():
            student.router.weight.copy_(teacher.router.weight)
            student.gate_modes.copy_(gate_modes)
            student.up_modes.copy_(up_modes)
            student.down_modes.copy_(down_modes)
            if student.gate_rank:
                student.gate_codes.copy_(gate_codes)
            if student.up_rank:
                student.up_codes.copy_(up_codes)
            if student.down_rank:
                student.down_codes.copy_(down_codes)
        if freeze_router:
            student.router.weight.requires_grad_(False)
        return student, {
            "gate_weight_relative_error": gate_error,
            "up_weight_relative_error": up_error,
            "down_weight_relative_error": down_error,
            "max_weight_relative_error": max(gate_error, up_error, down_error),
        }

    def expert_transform_parameter_count(self) -> int:
        return sum(
            tensor.numel()
            for tensor in (
                self.gate_modes,
                self.up_modes,
                self.down_modes,
                self.gate_codes,
                self.up_codes,
                self.down_codes,
            )
        )

    def dominant_matrix_compute_ratio(self) -> float:
        mode_count = sum(rank + 1 for rank in self.ranks)
        return mode_count / (3.0 * self.geometry.top_k)

    def idealized_expert_compute_ratio(self) -> float:
        code_ratio = sum(self.ranks) / (3.0 * self.geometry.d_model)
        return self.dominant_matrix_compute_ratio() + code_ratio


class ResidualScalarModalMoE(ScalarModalMoE):
    """Scalar Modal-MoE plus small expert-specific low-rank residuals.

    For each projection, the teacher weight is represented as

    ``W_e = W_modal,e + A_e B_e``

    where ``A_e B_e`` has rank at most ``residual_rank``.  The shared Modal
    path preserves cross-expert reuse, while the residual path captures
    directions that are not well described by a low-dimensional expert-axis
    basis.  Unlike increasing Modal K, residual capacity does not add another
    full-rank shared matrix; it adds selected-expert rank-r products.

    The idealized projection-compute ratio is

    ``(K+1)/T + K/D + r(D+F)/(DF)``

    for hidden width D, expert width F, top-T routing, scalar residual modes K,
    and expert-residual rank r.  This is an operation-count proxy, not a runtime
    claim.
    """

    def __init__(
        self,
        geometry: MoEGeometry,
        rank: int,
        residual_rank: int,
    ) -> None:
        if residual_rank < 0:
            raise ValueError("residual_rank must be non-negative")
        self.residual_rank = int(residual_rank)
        super().__init__(geometry, rank)
        e, d, f, r = (
            geometry.n_experts,
            geometry.d_model,
            geometry.d_ff,
            self.residual_rank,
        )
        self.gate_residual_left = nn.Parameter(torch.zeros(e, f, r))
        self.gate_residual_right = nn.Parameter(torch.zeros(e, r, d))
        self.up_residual_left = nn.Parameter(torch.zeros(e, f, r))
        self.up_residual_right = nn.Parameter(torch.zeros(e, r, d))
        self.down_residual_left = nn.Parameter(torch.zeros(e, d, r))
        self.down_residual_right = nn.Parameter(torch.zeros(e, r, f))

    def reconstruct_weights(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        gate, up, down = super().reconstruct_weights()
        if self.residual_rank:
            gate = gate + torch.einsum(
                "efr,erd->efd",
                self.gate_residual_left,
                self.gate_residual_right,
            )
            up = up + torch.einsum(
                "efr,erd->efd",
                self.up_residual_left,
                self.up_residual_right,
            )
            down = down + torch.einsum(
                "edr,erf->edf",
                self.down_residual_left,
                self.down_residual_right,
            )
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

        gate_projected = torch.einsum("nd,mfd->nmf", x, self.gate_modes)
        up_projected = torch.einsum("nd,mfd->nmf", x, self.up_modes)
        selected_shape = (x.shape[0], g.top_k, self.n_modes)
        gate_codes = self.full_codes(self.gate_codes).index_select(
            0, top_ids.reshape(-1)
        ).reshape(selected_shape)
        up_codes = self.full_codes(self.up_codes).index_select(
            0, top_ids.reshape(-1)
        ).reshape(selected_shape)
        down_codes = self.full_codes(self.down_codes).index_select(
            0, top_ids.reshape(-1)
        ).reshape(selected_shape)
        gate_values = torch.einsum("nmf,ntm->ntf", gate_projected, gate_codes)
        up_values = torch.einsum("nmf,ntm->ntf", up_projected, up_codes)

        if self.residual_rank:
            shape_left = (x.shape[0], g.top_k, g.d_ff, self.residual_rank)
            shape_right = (x.shape[0], g.top_k, self.residual_rank, g.d_model)
            gate_left = self.gate_residual_left.index_select(
                0, top_ids.reshape(-1)
            ).reshape(shape_left)
            gate_right = self.gate_residual_right.index_select(
                0, top_ids.reshape(-1)
            ).reshape(shape_right)
            up_left = self.up_residual_left.index_select(
                0, top_ids.reshape(-1)
            ).reshape(shape_left)
            up_right = self.up_residual_right.index_select(
                0, top_ids.reshape(-1)
            ).reshape(shape_right)
            gate_latent = torch.einsum("ntrd,nd->ntr", gate_right, x)
            up_latent = torch.einsum("ntrd,nd->ntr", up_right, x)
            gate_values = gate_values + torch.einsum(
                "ntfr,ntr->ntf", gate_left, gate_latent
            )
            up_values = up_values + torch.einsum(
                "ntfr,ntr->ntf", up_left, up_latent
            )

        hidden = F.silu(gate_values) * up_values
        mode_inputs = torch.einsum(
            "nt,ntm,ntf->nmf", weights, down_codes, hidden
        )
        output = torch.einsum("mdf,nmf->nd", self.down_modes, mode_inputs)

        if self.residual_rank:
            down_left = self.down_residual_left.index_select(
                0, top_ids.reshape(-1)
            ).reshape(x.shape[0], g.top_k, g.d_model, self.residual_rank)
            down_right = self.down_residual_right.index_select(
                0, top_ids.reshape(-1)
            ).reshape(x.shape[0], g.top_k, self.residual_rank, g.d_ff)
            down_latent = torch.einsum("ntrf,ntf->ntr", down_right, hidden)
            expert_residual = torch.einsum(
                "ntdr,ntr->ntd", down_left, down_latent
            )
            output = output + torch.einsum(
                "nt,ntd->nd", weights, expert_residual
            )
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

    @staticmethod
    def _factor_residual(
        residual: torch.Tensor,
        rank: int,
    ) -> tuple[torch.Tensor, torch.Tensor, float]:
        """Per-expert truncated SVD with balanced square-root factors."""
        experts, out_features, in_features = residual.shape
        left = torch.zeros(
            experts, out_features, rank, dtype=residual.dtype
        )
        right = torch.zeros(
            experts, rank, in_features, dtype=residual.dtype
        )
        reconstructed = torch.zeros_like(residual)
        if rank:
            for expert in range(experts):
                u, singular, vh = torch.linalg.svd(
                    residual[expert].double(), full_matrices=False
                )
                usable = min(rank, singular.numel())
                root = singular[:usable].sqrt()
                left[expert, :, :usable] = (
                    u[:, :usable] * root[None, :]
                ).to(residual.dtype)
                right[expert, :usable, :] = (
                    root[:, None] * vh[:usable]
                ).to(residual.dtype)
                reconstructed[expert] = left[expert] @ right[expert]
        denominator = torch.linalg.vector_norm(residual.double()).clamp_min(1e-12)
        error = float(
            torch.linalg.vector_norm(
                residual.double() - reconstructed.double()
            )
            / denominator
        )
        return left, right, error

    @classmethod
    def from_conventional_svd(
        cls,
        teacher: ConventionalSwiGLUMoE,
        rank: int,
        residual_rank: int,
        *,
        freeze_router: bool = True,
    ) -> tuple["ResidualScalarModalMoE", dict[str, float]]:
        scalar, scalar_metrics = ScalarModalMoE.from_conventional_svd(
            teacher, rank, freeze_router=False
        )
        student = cls(teacher.geometry, rank, residual_rank)
        with torch.no_grad():
            student.router.weight.copy_(teacher.router.weight)
            student.gate_modes.copy_(scalar.gate_modes)
            student.up_modes.copy_(scalar.up_modes)
            student.down_modes.copy_(scalar.down_modes)
            if rank:
                student.gate_codes.copy_(scalar.gate_codes)
                student.up_codes.copy_(scalar.up_codes)
                student.down_codes.copy_(scalar.down_codes)
            modal_gate, modal_up, modal_down = scalar.reconstruct_weights()
            factors = []
            for teacher_bank, modal_bank in (
                (teacher.gate, modal_gate),
                (teacher.up, modal_up),
                (teacher.down, modal_down),
            ):
                factors.append(
                    cls._factor_residual(
                        (teacher_bank.detach() - modal_bank.detach()),
                        residual_rank,
                    )
                )
            (gl, gr, gate_error), (ul, ur, up_error), (
                dl,
                dr,
                down_error,
            ) = factors
            student.gate_residual_left.copy_(gl)
            student.gate_residual_right.copy_(gr)
            student.up_residual_left.copy_(ul)
            student.up_residual_right.copy_(ur)
            student.down_residual_left.copy_(dl)
            student.down_residual_right.copy_(dr)
        if freeze_router:
            student.router.weight.requires_grad_(False)
        return student, {
            **{f"scalar_{key}": value for key, value in scalar_metrics.items()},
            "gate_remaining_relative_error": gate_error,
            "up_remaining_relative_error": up_error,
            "down_remaining_relative_error": down_error,
            "max_remaining_relative_error": max(
                gate_error, up_error, down_error
            ),
        }

    def residual_compute_ratio(self) -> float:
        g = self.geometry
        return self.residual_rank * (g.d_model + g.d_ff) / (
            g.d_model * g.d_ff
        )

    def idealized_expert_compute_ratio(self) -> float:
        return super().idealized_expert_compute_ratio() + self.residual_compute_ratio()

    def expert_transform_parameter_count(self) -> int:
        base = super().expert_transform_parameter_count()
        residual = sum(
            tensor.numel()
            for tensor in (
                self.gate_residual_left,
                self.gate_residual_right,
                self.up_residual_left,
                self.up_residual_right,
                self.down_residual_left,
                self.down_residual_right,
            )
        )
        return base + residual


def deterministic_cosine_clusters(
    features: torch.Tensor,
    n_groups: int,
    *,
    iterations: int = 32,
) -> torch.Tensor:
    """Cluster expert feature rows with deterministic spherical k-means.

    Farthest-point initialization avoids RNG dependence.  Empty groups are
    repaired by moving the currently worst represented expert.  The function is
    intentionally small and auditable; it is not a claim of optimal clustering.
    """
    if features.ndim != 2:
        raise ValueError("features must be [experts, observations]")
    experts = features.shape[0]
    if not 1 <= n_groups <= experts:
        raise ValueError("n_groups must be in [1, experts]")
    normalized = F.normalize(features.double(), dim=-1, eps=1e-12)
    # Start with the expert whose feature energy is largest before normalization.
    first = int(torch.argmax(torch.linalg.vector_norm(features.double(), dim=-1)))
    centers = [normalized[first]]
    selected = {first}
    while len(centers) < n_groups:
        similarities = torch.stack(
            [normalized @ center for center in centers], dim=-1
        )
        best_similarity = similarities.max(dim=-1).values
        for index in selected:
            best_similarity[index] = float("inf")
        next_index = int(torch.argmin(best_similarity))
        centers.append(normalized[next_index])
        selected.add(next_index)
    centroids = torch.stack(centers)
    assignments = torch.zeros(experts, dtype=torch.long)
    for _ in range(iterations):
        similarity = normalized @ centroids.T
        updated = torch.argmax(similarity, dim=-1)
        counts = torch.bincount(updated, minlength=n_groups)
        for empty in torch.nonzero(counts == 0, as_tuple=False).reshape(-1):
            current_best = similarity.max(dim=-1).values
            donor = int(torch.argmin(current_best))
            updated[donor] = int(empty)
            counts = torch.bincount(updated, minlength=n_groups)
        new_centroids = []
        for group in range(n_groups):
            members = normalized[updated == group]
            new_centroids.append(F.normalize(members.mean(dim=0), dim=0, eps=1e-12))
        new_centroids_tensor = torch.stack(new_centroids)
        if torch.equal(updated, assignments):
            centroids = new_centroids_tensor
            break
        assignments = updated
        centroids = new_centroids_tensor
    return assignments


class ClusteredResidualMoE(nn.Module):
    """Group-shared full-rank bases plus expert-specific low-rank residuals.

    Expert ``e`` belongs to group ``g(e)`` and each projection is

    ``W_e = B_{g(e)} + A_e C_e``.

    All group bases are evaluated once per token.  Routed experts then select
    their group's gate/up activations and add a rank-r residual.  Down bases are
    applied after aggregation by group, preserving a direct execution graph.
    This is a hierarchical alternative to one global expert-axis basis.
    """

    def __init__(
        self,
        geometry: MoEGeometry,
        n_groups: int,
        residual_rank: int,
        expert_to_group: torch.Tensor,
    ) -> None:
        super().__init__()
        geometry.validate()
        if not 1 <= n_groups <= geometry.n_experts:
            raise ValueError("invalid n_groups")
        if residual_rank < 0:
            raise ValueError("residual_rank must be non-negative")
        mapping = torch.as_tensor(expert_to_group, dtype=torch.long)
        if mapping.shape != (geometry.n_experts,):
            raise ValueError("expert_to_group must have one entry per expert")
        if mapping.min() < 0 or mapping.max() >= n_groups:
            raise ValueError("expert_to_group contains invalid group IDs")
        if torch.unique(mapping).numel() != n_groups:
            raise ValueError("every group must contain at least one expert")
        self.geometry = geometry
        self.n_groups = int(n_groups)
        self.residual_rank = int(residual_rank)
        self.register_buffer("expert_to_group", mapping.clone())
        self.router = nn.Linear(geometry.d_model, geometry.n_experts, bias=False)
        g, d, f, e, r = (
            self.n_groups,
            geometry.d_model,
            geometry.d_ff,
            geometry.n_experts,
            self.residual_rank,
        )
        self.gate_bases = nn.Parameter(torch.empty(g, f, d))
        self.up_bases = nn.Parameter(torch.empty(g, f, d))
        self.down_bases = nn.Parameter(torch.empty(g, d, f))
        self.gate_residual_left = nn.Parameter(torch.zeros(e, f, r))
        self.gate_residual_right = nn.Parameter(torch.zeros(e, r, d))
        self.up_residual_left = nn.Parameter(torch.zeros(e, f, r))
        self.up_residual_right = nn.Parameter(torch.zeros(e, r, d))
        self.down_residual_left = nn.Parameter(torch.zeros(e, d, r))
        self.down_residual_right = nn.Parameter(torch.zeros(e, r, f))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.router.weight, std=0.02)
        for bank in (self.gate_bases, self.up_bases, self.down_bases):
            for matrix in bank:
                nn.init.xavier_uniform_(matrix)

    def reconstruct_weights(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        gate = self.gate_bases.index_select(0, self.expert_to_group)
        up = self.up_bases.index_select(0, self.expert_to_group)
        down = self.down_bases.index_select(0, self.expert_to_group)
        if self.residual_rank:
            gate = gate + torch.einsum(
                "efr,erd->efd", self.gate_residual_left, self.gate_residual_right
            )
            up = up + torch.einsum(
                "efr,erd->efd", self.up_residual_left, self.up_residual_right
            )
            down = down + torch.einsum(
                "edr,erf->edf", self.down_residual_left, self.down_residual_right
            )
        return gate, up, down

    def forward(
        self,
        x: torch.Tensor,
        *,
        forced_top_ids: torch.Tensor | None = None,
        forced_weights: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, Routing]:
        geometry = self.geometry
        _validate_forced_routing(
            x, forced_top_ids, forced_weights, geometry.top_k
        )
        natural = route_topk(x, self.router.weight, geometry.top_k)
        top_ids = natural.top_ids if forced_top_ids is None else forced_top_ids
        weights = natural.weights if forced_weights is None else forced_weights
        selected_groups = self.expert_to_group.index_select(
            0, top_ids.reshape(-1)
        ).reshape_as(top_ids)

        gate_projected = torch.einsum("nd,gfd->ngf", x, self.gate_bases)
        up_projected = torch.einsum("nd,gfd->ngf", x, self.up_bases)
        gather_index = selected_groups[..., None].expand(
            x.shape[0], geometry.top_k, geometry.d_ff
        )
        gate_values = gate_projected.gather(1, gather_index)
        up_values = up_projected.gather(1, gather_index)

        if self.residual_rank:
            left_shape = (
                x.shape[0],
                geometry.top_k,
                geometry.d_ff,
                self.residual_rank,
            )
            right_shape = (
                x.shape[0],
                geometry.top_k,
                self.residual_rank,
                geometry.d_model,
            )
            gate_left = self.gate_residual_left.index_select(
                0, top_ids.reshape(-1)
            ).reshape(left_shape)
            gate_right = self.gate_residual_right.index_select(
                0, top_ids.reshape(-1)
            ).reshape(right_shape)
            up_left = self.up_residual_left.index_select(
                0, top_ids.reshape(-1)
            ).reshape(left_shape)
            up_right = self.up_residual_right.index_select(
                0, top_ids.reshape(-1)
            ).reshape(right_shape)
            gate_latent = torch.einsum("ntrd,nd->ntr", gate_right, x)
            up_latent = torch.einsum("ntrd,nd->ntr", up_right, x)
            gate_values = gate_values + torch.einsum(
                "ntfr,ntr->ntf", gate_left, gate_latent
            )
            up_values = up_values + torch.einsum(
                "ntfr,ntr->ntf", up_left, up_latent
            )

        hidden = F.silu(gate_values) * up_values
        group_routes = F.one_hot(
            selected_groups, self.n_groups
        ).to(dtype=hidden.dtype) * weights[..., None]
        group_inputs = torch.einsum("ntg,ntf->ngf", group_routes, hidden)
        output = torch.einsum("gdf,ngf->nd", self.down_bases, group_inputs)

        if self.residual_rank:
            down_left = self.down_residual_left.index_select(
                0, top_ids.reshape(-1)
            ).reshape(
                x.shape[0],
                geometry.top_k,
                geometry.d_model,
                self.residual_rank,
            )
            down_right = self.down_residual_right.index_select(
                0, top_ids.reshape(-1)
            ).reshape(
                x.shape[0],
                geometry.top_k,
                self.residual_rank,
                geometry.d_ff,
            )
            latent = torch.einsum("ntrf,ntf->ntr", down_right, hidden)
            residual_output = torch.einsum("ntdr,ntr->ntd", down_left, latent)
            output = output + torch.einsum("nt,ntd->nd", weights, residual_output)
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

    @classmethod
    def from_conventional_grouped(
        cls,
        teacher: ConventionalSwiGLUMoE,
        n_groups: int,
        residual_rank: int,
        *,
        calibration_inputs: torch.Tensor | None = None,
        freeze_router: bool = True,
    ) -> tuple["ClusteredResidualMoE", dict[str, object]]:
        if calibration_inputs is None:
            features = teacher.router.weight.detach()
            probabilities = torch.ones(
                teacher.geometry.n_experts, dtype=teacher.gate.dtype
            ) / teacher.geometry.n_experts
            feature_source = "router-weight-cosine"
        else:
            with torch.no_grad():
                logits = F.linear(
                    calibration_inputs.to(teacher.router.weight),
                    teacher.router.weight,
                )
                features = logits.T.detach()
                probabilities = F.softmax(logits, dim=-1).mean(dim=0).detach()
            feature_source = "calibration-router-logits"
        assignments = deterministic_cosine_clusters(features, n_groups)
        student = cls(
            teacher.geometry, n_groups, residual_rank, assignments
        )
        with torch.no_grad():
            student.router.weight.copy_(teacher.router.weight)
            remaining_errors: dict[str, float] = {}
            for label, teacher_bank, base_bank, left_bank, right_bank in (
                (
                    "gate",
                    teacher.gate,
                    student.gate_bases,
                    student.gate_residual_left,
                    student.gate_residual_right,
                ),
                (
                    "up",
                    teacher.up,
                    student.up_bases,
                    student.up_residual_left,
                    student.up_residual_right,
                ),
                (
                    "down",
                    teacher.down,
                    student.down_bases,
                    student.down_residual_left,
                    student.down_residual_right,
                ),
            ):
                for group in range(n_groups):
                    members = torch.nonzero(
                        assignments == group, as_tuple=False
                    ).reshape(-1)
                    group_weights = probabilities.index_select(0, members)
                    group_weights = group_weights / group_weights.sum().clamp_min(1e-12)
                    base_bank[group].copy_(
                        torch.einsum(
                            "e,e...->...",
                            group_weights.to(teacher_bank),
                            teacher_bank.index_select(0, members),
                        )
                    )
                bases_for_experts = base_bank.index_select(0, assignments)
                left, right, error = ResidualScalarModalMoE._factor_residual(
                    teacher_bank.detach() - bases_for_experts.detach(),
                    residual_rank,
                )
                left_bank.copy_(left)
                right_bank.copy_(right)
                remaining_errors[label] = error
        if freeze_router:
            student.router.weight.requires_grad_(False)
        counts = torch.bincount(assignments, minlength=n_groups)
        return student, {
            "method": "router-semantic-clusters-weighted-base-plus-low-rank-residual",
            "feature_source": feature_source,
            "n_groups": n_groups,
            "residual_rank": residual_rank,
            "expert_to_group": assignments.tolist(),
            "group_sizes": counts.tolist(),
            "gate_remaining_relative_error": remaining_errors["gate"],
            "up_remaining_relative_error": remaining_errors["up"],
            "down_remaining_relative_error": remaining_errors["down"],
            "max_remaining_relative_error": max(remaining_errors.values()),
        }

    def residual_compute_ratio(self) -> float:
        geometry = self.geometry
        return self.residual_rank * (geometry.d_model + geometry.d_ff) / (
            geometry.d_model * geometry.d_ff
        )

    def dominant_matrix_compute_ratio(self) -> float:
        return self.n_groups / self.geometry.top_k

    def idealized_expert_compute_ratio(self) -> float:
        return self.dominant_matrix_compute_ratio() + self.residual_compute_ratio()

    def expert_transform_parameter_count(self) -> int:
        return sum(
            tensor.numel()
            for tensor in (
                self.gate_bases,
                self.up_bases,
                self.down_bases,
                self.gate_residual_left,
                self.gate_residual_right,
                self.up_residual_left,
                self.up_residual_right,
                self.down_residual_left,
                self.down_residual_right,
            )
        )
