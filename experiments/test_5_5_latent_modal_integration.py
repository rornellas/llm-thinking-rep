#!/usr/bin/env python3
"""Test 5.5: integrate causal byte patches with Modal-MoE.

This is a paired factorial experiment. It asks whether two mechanisms that
worked separately remain compatible in one autoregressive byte language model:

1. causal local patch encoding, which reduces the global sequence length; and
2. Modal-MoE, which replaces independent expert matrices with shared full-rank
   modes plus small expert codes.

The six primary variants cross two representations (byte, patch8-GRU) with
three expert parameterizations (full conventional, 25%-width conventional,
Modal K=1). Every variant sees the same raw windows and predicts the same raw
bytes. Evaluation reports bits per byte, paired non-inferiority, the factorial
interaction, causal context interventions, expert-code interventions, routing
utilization, parameter counts, idealized global work, and measured CPU
throughput.

The result is intentionally a small-model mechanism test, not a production
speed claim. Matrix-work ratios exclude local encoder/decoder cost; measured
end-to-end throughput includes it.
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import json
import math
import random
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

EPS = 1e-12
LN2 = math.log(2.0)


@dataclass
class Config:
    window_bytes: int = 144
    target_start: int = 16
    target_bytes: int = 128
    patch_size: int = 8
    batch_size: int = 8
    steps: int = 700
    eval_batches: int = 40
    d_model: int = 64
    n_heads: int = 4
    n_layers: int = 2
    d_ff: int = 96
    n_experts: int = 64
    top_k: int = 8
    modal_rank: int = 1
    lr: float = 4e-4
    weight_decay: float = 0.04
    aux_weight: float = 0.01
    grad_clip: float = 1.0

    def validate(self) -> None:
        if self.window_bytes % self.patch_size:
            raise ValueError("window_bytes must be divisible by patch_size")
        if self.target_start % self.patch_size:
            raise ValueError("target_start must be divisible by patch_size")
        if self.target_bytes != self.window_bytes - self.target_start:
            raise ValueError("target span must end at window end")
        if self.target_bytes % self.patch_size:
            raise ValueError("target_bytes must be divisible by patch_size")
        if self.d_ff % 4:
            raise ValueError("d_ff must be divisible by four")
        if not (1 <= self.top_k <= self.n_experts):
            raise ValueError("invalid top_k")


@dataclass
class Evaluation:
    full_bpb: float
    first_patch_bpb: float
    full_values: np.ndarray
    first_values: np.ndarray
    target_bytes_per_second: float
    utilization_entropy: float
    min_expert_fraction: float
    max_expert_fraction: float


@dataclass
class RunResult:
    seed: int
    variant: str
    representation: str
    experts: str
    validation_bpb: float
    rolled_validation_bpb: float
    rolled_delta_bpb: float
    rolled_lcb95_bpb: float
    first_patch_bpb: float
    rolled_first_patch_bpb: float
    rolled_first_patch_delta_bpb: float
    rolled_first_patch_lcb95_bpb: float
    trainable_parameters: int
    expert_transform_parameters: int
    expert_parameter_ratio_to_full: float
    expert_matrix_compute_ratio_to_full: float
    expert_code_adjusted_compute_ratio_to_full: float
    global_positions: int
    global_position_ratio_to_byte: float
    attention_work_ratio_to_byte: float
    joint_expert_compute_ratio_to_byte_full: float
    target_bytes_per_second: float
    final_train_bpb: float
    best_validation_bpb: float
    training_seconds: float
    utilization_entropy: float
    min_expert_fraction: float
    max_expert_fraction: float


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def causal_mask(length: int, device: torch.device) -> torch.Tensor:
    mask = torch.full((length, length), float("-inf"), device=device)
    return torch.triu(mask, diagonal=1)


def router_auxiliary_loss(
    logits: torch.Tensor, top_ids: torch.Tensor, n_experts: int
) -> torch.Tensor:
    probabilities = F.softmax(logits, dim=-1)
    importance = probabilities.mean(dim=0)
    assignments = F.one_hot(top_ids, n_experts).float().mean(dim=(0, 1))
    balance = n_experts * torch.sum(importance * assignments)
    z_loss = torch.mean(torch.logsumexp(logits, dim=-1).square())
    return balance + 0.1 * z_loss


class SparseBaselineMoE(nn.Module):
    def __init__(
        self, d_model: int, d_ff: int, n_experts: int, top_k: int
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.n_experts = n_experts
        self.top_k = top_k
        self.router = nn.Linear(d_model, n_experts, bias=False)
        self.gate = nn.Parameter(torch.empty(n_experts, d_ff, d_model))
        self.up = nn.Parameter(torch.empty(n_experts, d_ff, d_model))
        self.down = nn.Parameter(torch.empty(n_experts, d_model, d_ff))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.router.weight, std=0.02)
        for weights in (self.gate, self.up, self.down):
            for expert in weights:
                nn.init.xavier_uniform_(expert)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        shape = x.shape
        flat = x.reshape(-1, self.d_model)
        logits = self.router(flat)
        top_values, top_ids = torch.topk(logits, self.top_k, dim=-1)
        route_weights = F.softmax(top_values, dim=-1)
        output = torch.zeros_like(flat)
        for expert_id in range(self.n_experts):
            where = torch.nonzero(top_ids == expert_id, as_tuple=False)
            if where.numel() == 0:
                continue
            token_ids, slots = where[:, 0], where[:, 1]
            selected = flat.index_select(0, token_ids)
            gate = F.linear(selected, self.gate[expert_id])
            up = F.linear(selected, self.up[expert_id])
            hidden = F.silu(gate) * up
            expert_output = F.linear(hidden, self.down[expert_id])
            expert_output = (
                expert_output
                * route_weights[token_ids, slots].unsqueeze(-1)
            )
            output.index_add_(0, token_ids, expert_output)
        aux = router_auxiliary_loss(logits, top_ids, self.n_experts)
        return output.reshape(shape), aux, top_ids


class ModalMoE(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        n_experts: int,
        top_k: int,
        modal_rank: int,
    ) -> None:
        super().__init__()
        if modal_rank < 1:
            raise ValueError("modal_rank must be positive")
        self.d_model = d_model
        self.d_ff = d_ff
        self.n_experts = n_experts
        self.top_k = top_k
        self.modal_rank = modal_rank
        self.n_modes = modal_rank + 1
        self.router = nn.Linear(d_model, n_experts, bias=False)
        self.gate_modes = nn.Parameter(
            torch.empty(self.n_modes, d_ff, d_model)
        )
        self.up_modes = nn.Parameter(
            torch.empty(self.n_modes, d_ff, d_model)
        )
        self.down_modes = nn.Parameter(
            torch.empty(self.n_modes, d_model, d_ff)
        )
        self.gate_codes = nn.Parameter(torch.empty(n_experts, modal_rank))
        self.up_codes = nn.Parameter(torch.empty(n_experts, modal_rank))
        self.down_codes = nn.Parameter(torch.empty(n_experts, modal_rank))
        self.reset_parameters()

    @staticmethod
    def _full_codes(codes: torch.Tensor) -> torch.Tensor:
        return torch.cat(
            [
                torch.ones(
                    codes.shape[0],
                    1,
                    device=codes.device,
                    dtype=codes.dtype,
                ),
                codes,
            ],
            dim=1,
        )

    def reset_parameters(self) -> None:
        nn.init.normal_(self.router.weight, std=0.02)
        for bank in (self.gate_modes, self.up_modes, self.down_modes):
            for index, mode in enumerate(bank):
                nn.init.xavier_uniform_(mode)
                if index > 0:
                    mode.data.mul_(0.5 / math.sqrt(self.modal_rank))
        for codes in (self.gate_codes, self.up_codes, self.down_codes):
            nn.init.normal_(codes, std=0.7)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        shape = x.shape
        flat = x.reshape(-1, self.d_model)
        logits = self.router(flat)
        top_values, top_ids = torch.topk(logits, self.top_k, dim=-1)
        route_weights = F.softmax(top_values, dim=-1)

        gate_projected = torch.einsum(
            "nd,kmd->nkm", flat, self.gate_modes
        )
        up_projected = torch.einsum(
            "nd,kmd->nkm", flat, self.up_modes
        )
        selected_shape = (flat.shape[0], self.top_k, self.n_modes)
        gate_codes = self._full_codes(self.gate_codes).index_select(
            0, top_ids.reshape(-1)
        ).reshape(selected_shape)
        up_codes = self._full_codes(self.up_codes).index_select(
            0, top_ids.reshape(-1)
        ).reshape(selected_shape)
        gate = torch.einsum("nkm,ntk->ntm", gate_projected, gate_codes)
        up = torch.einsum("nkm,ntk->ntm", up_projected, up_codes)
        hidden = F.silu(gate) * up

        down_codes = self._full_codes(self.down_codes).index_select(
            0, top_ids.reshape(-1)
        ).reshape(selected_shape)
        mode_inputs = torch.einsum(
            "nt,ntk,ntm->nkm", route_weights, down_codes, hidden
        )
        output = torch.einsum("kdm,nkm->nd", self.down_modes, mode_inputs)
        aux = router_auxiliary_loss(logits, top_ids, self.n_experts)
        return output.reshape(shape), aux, top_ids

    @torch.no_grad()
    def reference_reconstructed(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        flat = x.reshape(-1, self.d_model)
        logits = self.router(flat)
        top_values, top_ids = torch.topk(logits, self.top_k, dim=-1)
        route_weights = F.softmax(top_values, dim=-1)
        cg = self._full_codes(self.gate_codes)
        cu = self._full_codes(self.up_codes)
        cd = self._full_codes(self.down_codes)
        output = torch.zeros_like(flat)
        for expert_id in range(self.n_experts):
            where = torch.nonzero(top_ids == expert_id, as_tuple=False)
            if where.numel() == 0:
                continue
            token_ids, slots = where[:, 0], where[:, 1]
            gate_weight = torch.einsum(
                "k,kmd->md", cg[expert_id], self.gate_modes
            )
            up_weight = torch.einsum(
                "k,kmd->md", cu[expert_id], self.up_modes
            )
            down_weight = torch.einsum(
                "k,kdm->dm", cd[expert_id], self.down_modes
            )
            selected = flat.index_select(0, token_ids)
            hidden = F.silu(F.linear(selected, gate_weight)) * F.linear(
                selected, up_weight
            )
            expert_output = F.linear(hidden, down_weight)
            expert_output = (
                expert_output
                * route_weights[token_ids, slots].unsqueeze(-1)
            )
            output.index_add_(0, token_ids, expert_output)
        return output.reshape(shape)


class GlobalMoEBlock(nn.Module):
    def __init__(self, cfg: Config, expert_variant: str) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(cfg.d_model)
        self.attention = nn.MultiheadAttention(
            cfg.d_model,
            cfg.n_heads,
            batch_first=True,
            dropout=0.0,
        )
        self.norm2 = nn.LayerNorm(cfg.d_model)
        if expert_variant == "full":
            self.moe: nn.Module = SparseBaselineMoE(
                cfg.d_model, cfg.d_ff, cfg.n_experts, cfg.top_k
            )
        elif expert_variant == "narrow":
            self.moe = SparseBaselineMoE(
                cfg.d_model, cfg.d_ff // 4, cfg.n_experts, cfg.top_k
            )
        elif expert_variant == "modal":
            self.moe = ModalMoE(
                cfg.d_model,
                cfg.d_ff,
                cfg.n_experts,
                cfg.top_k,
                cfg.modal_rank,
            )
        else:
            raise ValueError(expert_variant)

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        normalized = self.norm1(x)
        attended, _ = self.attention(
            normalized,
            normalized,
            normalized,
            attn_mask=mask,
            need_weights=False,
        )
        x = x + attended
        moe_output, aux, routes = self.moe(self.norm2(x))
        return x + moe_output, aux, routes


class GlobalMoEStack(nn.Module):
    def __init__(self, cfg: Config, expert_variant: str) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                GlobalMoEBlock(cfg, expert_variant)
                for _ in range(cfg.n_layers)
            ]
        )

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]:
        mask = causal_mask(x.shape[1], x.device)
        aux_total = torch.zeros((), device=x.device)
        routes: list[torch.Tensor] = []
        for block in self.blocks:
            x, aux, layer_routes = block(x, mask)
            aux_total = aux_total + aux
            routes.append(layer_routes)
        return x, aux_total / len(self.blocks), routes


class ByteMoELM(nn.Module):
    def __init__(self, cfg: Config, expert_variant: str) -> None:
        super().__init__()
        self.cfg = cfg
        self.representation = "byte"
        self.expert_variant = expert_variant
        self.byte_embedding = nn.Embedding(256, cfg.d_model)
        self.position_embedding = nn.Embedding(
            cfg.window_bytes - 1, cfg.d_model
        )
        self.global_stack = GlobalMoEStack(cfg, expert_variant)
        self.norm = nn.LayerNorm(cfg.d_model)
        self.output = nn.Linear(cfg.d_model, 256)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.byte_embedding.weight, std=0.02)
        nn.init.normal_(self.position_embedding.weight, std=0.02)
        nn.init.normal_(self.output.weight, std=0.02)
        nn.init.zeros_(self.output.bias)

    @property
    def global_positions(self) -> int:
        return self.cfg.window_bytes - 1

    def forward(
        self, windows: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]:
        inputs = windows[:, :-1]
        positions = torch.arange(inputs.shape[1], device=windows.device)
        hidden = self.byte_embedding(inputs) + self.position_embedding(
            positions
        )[None, :, :]
        hidden, aux, routes = self.global_stack(hidden)
        selected = hidden[
            :,
            self.cfg.target_start - 1 : self.cfg.window_bytes - 1,
        ]
        if selected.shape[1] != self.cfg.target_bytes:
            raise AssertionError(selected.shape)
        return self.output(self.norm(selected)), aux, routes


class PatchMoELM(nn.Module):
    def __init__(self, cfg: Config, expert_variant: str) -> None:
        super().__init__()
        self.cfg = cfg
        self.representation = "patch8"
        self.expert_variant = expert_variant
        self.patch_size = cfg.patch_size
        self.total_patches = cfg.window_bytes // cfg.patch_size
        self.input_patches = self.total_patches - 1
        self.target_patch_start = cfg.target_start // cfg.patch_size
        self.target_patches = cfg.target_bytes // cfg.patch_size
        self.byte_embedding = nn.Embedding(257, cfg.d_model)
        self.local_encoder = nn.GRU(
            cfg.d_model, cfg.d_model, batch_first=True
        )
        self.patch_position_embedding = nn.Embedding(
            self.input_patches, cfg.d_model
        )
        self.global_stack = GlobalMoEStack(cfg, expert_variant)
        self.context_to_decoder = nn.Linear(cfg.d_model, cfg.d_model)
        self.local_decoder = nn.GRU(
            cfg.d_model, cfg.d_model, batch_first=True
        )
        self.decoder_norm = nn.LayerNorm(cfg.d_model)
        self.output = nn.Linear(cfg.d_model, 256)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.byte_embedding.weight, std=0.02)
        nn.init.normal_(self.patch_position_embedding.weight, std=0.02)
        nn.init.normal_(self.output.weight, std=0.02)
        nn.init.zeros_(self.output.bias)

    @property
    def global_positions(self) -> int:
        return self.input_patches

    def encode_patches(self, patches: torch.Tensor) -> torch.Tensor:
        batch, count, width = patches.shape
        embedded = self.byte_embedding(patches).reshape(
            batch * count, width, self.cfg.d_model
        )
        encoded, _ = self.local_encoder(embedded)
        return encoded[:, -1].reshape(batch, count, self.cfg.d_model)

    def forward(
        self, windows: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]:
        batch = windows.shape[0]
        patches = windows.reshape(
            batch, self.total_patches, self.patch_size
        )
        input_patches = patches[:, : self.input_patches]
        latent = self.encode_patches(input_patches)
        positions = torch.arange(self.input_patches, device=windows.device)
        latent = latent + self.patch_position_embedding(positions)[None, :, :]
        latent, aux, routes = self.global_stack(latent)

        first_context_index = self.target_patch_start - 1
        context_states = latent[
            :, first_context_index : self.total_patches - 1
        ]
        target_patches = patches[
            :, self.target_patch_start : self.total_patches
        ]
        if context_states.shape[1] != self.target_patches:
            raise AssertionError((context_states.shape, target_patches.shape))

        flat_context = context_states.reshape(
            batch * self.target_patches, self.cfg.d_model
        )
        flat_targets = target_patches.reshape(
            batch * self.target_patches, self.patch_size
        )
        bos = torch.full(
            (flat_targets.shape[0], 1),
            256,
            dtype=torch.long,
            device=windows.device,
        )
        decoder_tokens = torch.cat([bos, flat_targets[:, :-1]], dim=1)
        decoder_inputs = self.byte_embedding(decoder_tokens)
        initial = torch.tanh(
            self.context_to_decoder(flat_context)
        ).unsqueeze(0)
        decoded, _ = self.local_decoder(decoder_inputs, initial)
        logits = self.output(self.decoder_norm(decoded))
        return (
            logits.reshape(batch, self.cfg.target_bytes, 256),
            aux,
            routes,
        )


def make_model(cfg: Config, representation: str, experts: str) -> nn.Module:
    if representation == "byte":
        return ByteMoELM(cfg, experts)
    if representation == "patch8":
        return PatchMoELM(cfg, experts)
    raise ValueError(representation)


def copy_matching_state(source: nn.Module, target: nn.Module) -> None:
    source_state = source.state_dict()
    target_state = target.state_dict()
    with torch.no_grad():
        for key, value in target_state.items():
            if key in source_state and source_state[key].shape == value.shape:
                value.copy_(source_state[key])
    target.load_state_dict(target_state)


def initialization_states(
    cfg: Config, seed: int
) -> dict[tuple[str, str], dict[str, torch.Tensor]]:
    set_seed(seed)
    byte_full = make_model(cfg, "byte", "full")
    set_seed(seed + 1)
    patch_full = make_model(cfg, "patch8", "full")
    copy_matching_state(byte_full, patch_full)

    result: dict[tuple[str, str], dict[str, torch.Tensor]] = {}
    for representation, template in (
        ("byte", byte_full),
        ("patch8", patch_full),
    ):
        result[(representation, "full")] = {
            key: value.detach().clone()
            for key, value in template.state_dict().items()
        }
        for offset, experts in enumerate(("narrow", "modal"), start=2):
            set_seed(seed + offset + (0 if representation == "byte" else 20))
            model = make_model(cfg, representation, experts)
            copy_matching_state(template, model)
            result[(representation, experts)] = {
                key: value.detach().clone()
                for key, value in model.state_dict().items()
            }
    return result


def build_windows(
    data: np.ndarray, starts: np.ndarray, cfg: Config
) -> torch.Tensor:
    windows = np.stack(
        [
            data[int(start) : int(start) + cfg.window_bytes]
            for start in starts
        ]
    ).astype(np.int64, copy=False)
    return torch.from_numpy(windows)


def targets_from_windows(windows: torch.Tensor, cfg: Config) -> torch.Tensor:
    targets = windows[:, cfg.target_start : cfg.window_bytes]
    if targets.shape[1] != cfg.target_bytes:
        raise AssertionError(targets.shape)
    return targets


def rolled_context_windows(windows: torch.Tensor, cfg: Config) -> torch.Tensor:
    rolled = windows.clone()
    replacement = torch.roll(windows[:, : cfg.target_start], shifts=1, dims=0)
    rolled[:, : cfg.target_start] = replacement
    return rolled


def bootstrap_bound(
    values: np.ndarray,
    seed: int,
    quantile: float,
    samples: int = 5000,
) -> float:
    values = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, values.size, size=(samples, values.size))
    return float(np.quantile(values[indices].mean(axis=1), quantile))


def route_statistics(
    route_tensors: Sequence[torch.Tensor], n_experts: int
) -> tuple[float, float, float]:
    counts = torch.zeros(n_experts, dtype=torch.float64)
    for routes in route_tensors:
        counts += torch.bincount(
            routes.reshape(-1).cpu(), minlength=n_experts
        ).double()
    fractions = counts / torch.clamp(counts.sum(), min=1.0)
    nonzero = fractions[fractions > 0]
    entropy = -torch.sum(nonzero * torch.log(nonzero)) / math.log(n_experts)
    return (
        float(entropy),
        float(fractions.min()),
        float(fractions.max()),
    )


@torch.no_grad()
def evaluate(
    model: nn.Module,
    windows: torch.Tensor,
    cfg: Config,
    *,
    rolled: bool = False,
) -> Evaluation:
    model.eval()
    inputs = rolled_context_windows(windows, cfg) if rolled else windows
    full_values: list[np.ndarray] = []
    first_values: list[np.ndarray] = []
    routes: list[torch.Tensor] = []
    total_target_bytes = 0
    started = time.perf_counter()
    for offset in range(0, len(inputs), cfg.batch_size):
        batch_inputs = inputs[offset : offset + cfg.batch_size]
        original = windows[offset : offset + cfg.batch_size]
        targets = targets_from_windows(original, cfg)
        logits, _, batch_routes = model(batch_inputs)
        per_byte = F.cross_entropy(
            logits.reshape(-1, 256),
            targets.reshape(-1),
            reduction="none",
        ).reshape(len(original), cfg.target_bytes)
        full_values.append((per_byte.mean(dim=1) / LN2).cpu().numpy())
        first_values.append(
            (per_byte[:, : cfg.patch_size].mean(dim=1) / LN2)
            .cpu()
            .numpy()
        )
        routes.extend(route.detach().cpu() for route in batch_routes)
        total_target_bytes += int(targets.numel())
    elapsed = time.perf_counter() - started
    full = np.concatenate(full_values)
    first = np.concatenate(first_values)
    entropy, minimum, maximum = route_statistics(routes, cfg.n_experts)
    return Evaluation(
        full_bpb=float(np.mean(full)),
        first_patch_bpb=float(np.mean(first)),
        full_values=full,
        first_values=first,
        target_bytes_per_second=total_target_bytes / max(elapsed, EPS),
        utilization_entropy=entropy,
        min_expert_fraction=minimum,
        max_expert_fraction=maximum,
    )


def iter_modal_moes(model: nn.Module) -> Iterator[ModalMoE]:
    for module in model.modules():
        if isinstance(module, ModalMoE):
            yield module


@contextlib.contextmanager
def modal_code_policy(
    model: nn.Module, policy: str, seed: int
) -> Iterator[None]:
    modules = list(iter_modal_moes(model))
    backups = [
        (
            module.gate_codes.detach().clone(),
            module.up_codes.detach().clone(),
            module.down_codes.detach().clone(),
        )
        for module in modules
    ]
    with torch.no_grad():
        for layer_index, module in enumerate(modules):
            code_tensors = (
                module.gate_codes,
                module.up_codes,
                module.down_codes,
            )
            if policy == "mean-code":
                for codes in code_tensors:
                    codes.copy_(codes.mean(dim=0, keepdim=True).expand_as(codes))
            elif policy == "zero-residual":
                for codes in code_tensors:
                    codes.zero_()
            elif policy == "shuffle-code":
                generator = torch.Generator().manual_seed(seed + layer_index)
                permutation = torch.randperm(
                    module.n_experts, generator=generator
                )
                for codes in code_tensors:
                    codes.copy_(codes.index_select(0, permutation))
            else:
                raise ValueError(policy)
    try:
        yield
    finally:
        with torch.no_grad():
            for module, saved in zip(modules, backups, strict=True):
                module.gate_codes.copy_(saved[0])
                module.up_codes.copy_(saved[1])
                module.down_codes.copy_(saved[2])


def expert_transform_parameter_count(model: nn.Module) -> int:
    total = 0
    for module in model.modules():
        if isinstance(module, SparseBaselineMoE):
            total += module.gate.numel() + module.up.numel() + module.down.numel()
        elif isinstance(module, ModalMoE):
            total += (
                module.gate_modes.numel()
                + module.up_modes.numel()
                + module.down_modes.numel()
                + module.gate_codes.numel()
                + module.up_codes.numel()
                + module.down_codes.numel()
            )
    return total


def expert_compute_ratios(cfg: Config, experts: str) -> tuple[float, float]:
    if experts == "full":
        return 1.0, 1.0
    if experts == "narrow":
        return 0.25, 0.25
    if experts == "modal":
        matrix = (cfg.modal_rank + 1) / cfg.top_k
        code_adjusted = matrix + cfg.modal_rank / cfg.d_model
        return matrix, code_adjusted
    raise ValueError(experts)


def train_variant(
    cfg: Config,
    representation: str,
    experts: str,
    seed: int,
    initial_state: dict[str, torch.Tensor],
    train_windows: torch.Tensor,
    validation_windows: torch.Tensor,
) -> tuple[nn.Module, RunResult, Evaluation, Evaluation, dict[str, Evaluation]]:
    model = make_model(cfg, representation, experts)
    model.load_state_dict(initial_state)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    best = math.inf
    final_train = math.nan
    started = time.perf_counter()
    for step in range(1, cfg.steps + 1):
        offset = (step - 1) * cfg.batch_size
        windows = train_windows[offset : offset + cfg.batch_size]
        targets = targets_from_windows(windows, cfg)
        model.train()
        logits, aux, _ = model(windows)
        language_loss = F.cross_entropy(
            logits.reshape(-1, 256), targets.reshape(-1)
        )
        loss = language_loss + cfg.aux_weight * aux
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()
        final_train = float(language_loss.detach() / LN2)
        if step == 1 or step % 175 == 0 or step == cfg.steps:
            validation = evaluate(model, validation_windows, cfg)
            best = min(best, validation.full_bpb)
            print(
                f"seed={seed} {representation}-{experts} step={step}/{cfg.steps} "
                f"train-bpb={final_train:.4f} val-bpb={validation.full_bpb:.4f}",
                flush=True,
            )
    training_seconds = time.perf_counter() - started
    correct = evaluate(model, validation_windows, cfg)
    rolled = evaluate(model, validation_windows, cfg, rolled=True)
    best = min(best, correct.full_bpb)

    ablations: dict[str, Evaluation] = {}
    if experts == "modal" and representation == "patch8":
        for index, policy in enumerate(
            ("mean-code", "shuffle-code", "zero-residual")
        ):
            with modal_code_policy(model, policy, seed + 1000 + index * 100):
                ablations[policy] = evaluate(model, validation_windows, cfg)

    byte_positions = cfg.window_bytes - 1
    position_ratio = model.global_positions / byte_positions
    attention_ratio = position_ratio**2
    matrix_ratio, adjusted_ratio = expert_compute_ratios(cfg, experts)
    full_transform_parameters = (
        cfg.n_layers
        * cfg.n_experts
        * 3
        * cfg.d_model
        * cfg.d_ff
    )
    transform_parameters = expert_transform_parameter_count(model)
    rolled_deltas = rolled.full_values - correct.full_values
    first_deltas = rolled.first_values - correct.first_values
    result = RunResult(
        seed=seed,
        variant=f"{representation}-{experts}",
        representation=representation,
        experts=experts,
        validation_bpb=correct.full_bpb,
        rolled_validation_bpb=rolled.full_bpb,
        rolled_delta_bpb=float(np.mean(rolled_deltas)),
        rolled_lcb95_bpb=bootstrap_bound(
            rolled_deltas, seed + 10, 0.05
        ),
        first_patch_bpb=correct.first_patch_bpb,
        rolled_first_patch_bpb=rolled.first_patch_bpb,
        rolled_first_patch_delta_bpb=float(np.mean(first_deltas)),
        rolled_first_patch_lcb95_bpb=bootstrap_bound(
            first_deltas, seed + 20, 0.05
        ),
        trainable_parameters=sum(
            parameter.numel() for parameter in model.parameters()
        ),
        expert_transform_parameters=transform_parameters,
        expert_parameter_ratio_to_full=(
            transform_parameters / full_transform_parameters
        ),
        expert_matrix_compute_ratio_to_full=matrix_ratio,
        expert_code_adjusted_compute_ratio_to_full=adjusted_ratio,
        global_positions=int(model.global_positions),
        global_position_ratio_to_byte=position_ratio,
        attention_work_ratio_to_byte=attention_ratio,
        joint_expert_compute_ratio_to_byte_full=(
            position_ratio * adjusted_ratio
        ),
        target_bytes_per_second=correct.target_bytes_per_second,
        final_train_bpb=final_train,
        best_validation_bpb=best,
        training_seconds=training_seconds,
        utilization_entropy=correct.utilization_entropy,
        min_expert_fraction=correct.min_expert_fraction,
        max_expert_fraction=correct.max_expert_fraction,
    )
    return model, result, correct, rolled, ablations


def summarize(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    variants = sorted({row["variant"] for row in rows})
    result: list[dict[str, Any]] = []
    for variant in variants:
        selected = [row for row in rows if row["variant"] == variant]
        result.append(
            {
                "variant": variant,
                "runs": len(selected),
                "validation_bpb_mean": statistics.mean(
                    row["validation_bpb"] for row in selected
                ),
                "validation_bpb_std": statistics.pstdev(
                    row["validation_bpb"] for row in selected
                ),
                "first_patch_bpb_mean": statistics.mean(
                    row["first_patch_bpb"] for row in selected
                ),
                "rolled_first_delta_mean": statistics.mean(
                    row["rolled_first_patch_delta_bpb"] for row in selected
                ),
                "worst_rolled_first_lcb95": min(
                    row["rolled_first_patch_lcb95_bpb"] for row in selected
                ),
                "trainable_parameters": int(
                    selected[0]["trainable_parameters"]
                ),
                "expert_parameter_ratio_to_full": float(
                    selected[0]["expert_parameter_ratio_to_full"]
                ),
                "joint_expert_compute_ratio_to_byte_full": float(
                    selected[0]["joint_expert_compute_ratio_to_byte_full"]
                ),
                "global_positions": int(selected[0]["global_positions"]),
                "attention_work_ratio_to_byte": float(
                    selected[0]["attention_work_ratio_to_byte"]
                ),
                "target_bytes_per_second_mean": statistics.mean(
                    row["target_bytes_per_second"] for row in selected
                ),
                "utilization_entropy_mean": statistics.mean(
                    row["utilization_entropy"] for row in selected
                ),
            }
        )
    return result


def make_decision(
    rows: Sequence[dict[str, Any]],
    evaluations: dict[tuple[int, str], Evaluation],
    ablations: dict[tuple[int, str], Evaluation],
) -> dict[str, Any]:
    seeds = sorted({int(row["seed"]) for row in rows})
    per_seed: list[dict[str, Any]] = []
    pooled_noninferiority: list[np.ndarray] = []
    pooled_interaction: list[np.ndarray] = []
    pooled_modal_delta: list[np.ndarray] = []
    ablation_pooled: dict[str, list[np.ndarray]] = {
        "mean-code": [],
        "shuffle-code": [],
        "zero-residual": [],
    }

    for seed in seeds:
        byte_full = evaluations[(seed, "byte-full")]
        byte_narrow = evaluations[(seed, "byte-narrow")]
        byte_modal = evaluations[(seed, "byte-modal")]
        patch_full = evaluations[(seed, "patch8-full")]
        patch_narrow = evaluations[(seed, "patch8-narrow")]
        integrated = evaluations[(seed, "patch8-modal")]

        noninferiority = integrated.full_values - byte_full.full_values
        modal_delta = integrated.full_values - patch_narrow.full_values
        interaction = (
            integrated.full_values
            - patch_narrow.full_values
            - byte_modal.full_values
            + byte_narrow.full_values
        )
        pooled_noninferiority.append(noninferiority)
        pooled_modal_delta.append(modal_delta)
        pooled_interaction.append(interaction)

        integrated_context = (
            evaluations[(seed, "patch8-modal-rolled")].first_values
            - integrated.first_values
        )
        full_context = (
            evaluations[(seed, "patch8-full-rolled")].first_values
            - patch_full.first_values
        )
        context_ratio = float(np.mean(integrated_context)) / max(
            float(np.mean(full_context)), EPS
        )

        seed_ablations: dict[str, Any] = {}
        for index, policy in enumerate(ablation_pooled):
            values = (
                ablations[(seed, policy)].full_values
                - integrated.full_values
            )
            ablation_pooled[policy].append(values)
            seed_ablations[policy] = {
                "delta_bpb": float(np.mean(values)),
                "lcb95_bpb": bootstrap_bound(
                    values, seed + 300 + index, 0.05
                ),
            }

        per_seed.append(
            {
                "seed": seed,
                "integrated_minus_byte_full_bpb": float(
                    np.mean(noninferiority)
                ),
                "integrated_minus_byte_full_ucb95": bootstrap_bound(
                    noninferiority, seed + 100, 0.95
                ),
                "integrated_minus_patch_narrow_bpb": float(
                    np.mean(modal_delta)
                ),
                "integrated_minus_patch_narrow_ucb95": bootstrap_bound(
                    modal_delta, seed + 110, 0.95
                ),
                "factorial_interaction_bpb": float(np.mean(interaction)),
                "factorial_interaction_ucb95": bootstrap_bound(
                    interaction, seed + 120, 0.95
                ),
                "integrated_first_context_delta_bpb": float(
                    np.mean(integrated_context)
                ),
                "integrated_first_context_lcb95": bootstrap_bound(
                    integrated_context, seed + 130, 0.05
                ),
                "patch_full_first_context_delta_bpb": float(
                    np.mean(full_context)
                ),
                "context_retention_ratio": context_ratio,
                "code_ablations": seed_ablations,
            }
        )

    pooled_noninferiority_array = np.concatenate(pooled_noninferiority)
    pooled_modal_array = np.concatenate(pooled_modal_delta)
    pooled_interaction_array = np.concatenate(pooled_interaction)
    pooled_code: dict[str, dict[str, float | bool]] = {}
    code_pass_count = 0
    for index, (policy, chunks) in enumerate(ablation_pooled.items()):
        values = np.concatenate(chunks)
        mean = float(np.mean(values))
        lcb = bootstrap_bound(values, 9000 + index, 0.05)
        passed = mean >= 0.002 and lcb > 0.0
        code_pass_count += int(passed)
        pooled_code[policy] = {
            "delta_bpb": mean,
            "lcb95_bpb": lcb,
            "passes": passed,
        }

    indexed_rows = {
        (int(row["seed"]), row["variant"]): row for row in rows
    }
    throughput_ratios = [
        indexed_rows[(seed, "patch8-modal")]["target_bytes_per_second"]
        / indexed_rows[(seed, "byte-full")]["target_bytes_per_second"]
        for seed in seeds
    ]

    quality_pass = all(
        item["integrated_minus_byte_full_ucb95"] <= 0.01
        and item["integrated_minus_patch_narrow_ucb95"] <= 0.01
        for item in per_seed
    )
    interaction_pass = all(
        item["factorial_interaction_ucb95"] <= 0.01
        for item in per_seed
    )
    context_pass = all(
        item["integrated_first_context_lcb95"] > 0.0
        and item["context_retention_ratio"] >= 0.80
        for item in per_seed
    )
    codes_pass = code_pass_count >= 2
    efficiency_pass = min(throughput_ratios) >= 1.20

    if quality_pass and interaction_pass and context_pass and codes_pass:
        verdict = (
            "INTEGRATED_PASS_WITH_THROUGHPUT"
            if efficiency_pass
            else "INTEGRATED_FUNCTIONAL_PASS"
        )
    elif quality_pass and interaction_pass:
        verdict = "INTEGRATED_BORDERLINE_CAUSAL"
    else:
        verdict = "INTEGRATED_FAIL"

    return {
        "verdict": verdict,
        "quality_pass": quality_pass,
        "interaction_pass": interaction_pass,
        "context_pass": context_pass,
        "codes_pass": codes_pass,
        "efficiency_pass": efficiency_pass,
        "pooled_integrated_minus_byte_full_bpb": float(
            np.mean(pooled_noninferiority_array)
        ),
        "pooled_integrated_minus_byte_full_ucb95": bootstrap_bound(
            pooled_noninferiority_array, 8101, 0.95
        ),
        "pooled_integrated_minus_patch_narrow_bpb": float(
            np.mean(pooled_modal_array)
        ),
        "pooled_integrated_minus_patch_narrow_ucb95": bootstrap_bound(
            pooled_modal_array, 8102, 0.95
        ),
        "pooled_factorial_interaction_bpb": float(
            np.mean(pooled_interaction_array)
        ),
        "pooled_factorial_interaction_ucb95": bootstrap_bound(
            pooled_interaction_array, 8103, 0.95
        ),
        "throughput_ratios_integrated_to_byte_full": throughput_ratios,
        "throughput_ratio_mean": statistics.mean(throughput_ratios),
        "code_ablations": pooled_code,
        "per_seed": per_seed,
        "rule": (
            "Functional PASS requires, in every seed, integrated Patch8+Modal K1 "
            "within +0.01 BpB of byte-full and patch8-narrow, factorial interaction "
            "UCB95 <= +0.01 BpB, first-patch rolled-context LCB95 > 0 with at "
            "least 80% retention versus patch8-full, and at least two code "
            "ablations with pooled mean >=0.002 BpB and positive LCB95. The "
            "throughput suffix additionally requires >=1.2x byte-full in every seed."
        ),
    }


def write_outputs(output_dir: Path, payload: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "latent_modal_integration.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with (output_dir / "runs.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(payload["runs"][0].keys())
        )
        writer.writeheader()
        writer.writerows(payload["runs"])
    with (output_dir / "summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(payload["summary"][0].keys())
        )
        writer.writeheader()
        writer.writerows(payload["summary"])

    d = payload["decision"]
    lines = [
        "# Test 5.5 — causal byte patches + Modal-MoE integration",
        "",
        f"**Decision:** **{d['verdict']}**",
        "",
        "| Variant | Runs | BpB | First-patch BpB | First rolled Δ | Params | Expert params/full | Global positions | Attention work/byte | Joint expert compute/byte-full | Target bytes/s |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["summary"]:
        lines.append(
            f"| {row['variant']} | {row['runs']} | "
            f"{row['validation_bpb_mean']:.4f} ± {row['validation_bpb_std']:.4f} | "
            f"{row['first_patch_bpb_mean']:.4f} | "
            f"{row['rolled_first_delta_mean']:+.4f} | "
            f"{row['trainable_parameters']:,} | "
            f"{row['expert_parameter_ratio_to_full']:.3%} | "
            f"{row['global_positions']} | "
            f"{row['attention_work_ratio_to_byte']:.3%} | "
            f"{row['joint_expert_compute_ratio_to_byte_full']:.3%} | "
            f"{row['target_bytes_per_second_mean']:.0f} |"
        )
    lines += [
        "",
        "## Paired gates",
        "",
        f"- Integrated minus byte-full: `{d['pooled_integrated_minus_byte_full_bpb']:+.4f}` BpB; UCB95 `{d['pooled_integrated_minus_byte_full_ucb95']:+.4f}`.",
        f"- Integrated minus patch8-narrow: `{d['pooled_integrated_minus_patch_narrow_bpb']:+.4f}` BpB; UCB95 `{d['pooled_integrated_minus_patch_narrow_ucb95']:+.4f}`.",
        f"- Factorial interaction: `{d['pooled_factorial_interaction_bpb']:+.4f}` BpB; UCB95 `{d['pooled_factorial_interaction_ucb95']:+.4f}`.",
        f"- Integrated/byte-full throughput ratios: `{[round(x, 3) for x in d['throughput_ratios_integrated_to_byte_full']]}`; mean `{d['throughput_ratio_mean']:.3f}x`.",
        "",
        "## Modal code interventions",
        "",
        "| Policy | Δ BpB | LCB95 | Pass |",
        "|---|---:|---:|---|",
    ]
    for policy, row in d["code_ablations"].items():
        lines.append(
            f"| {policy} | {row['delta_bpb']:+.4f} | "
            f"{row['lcb95_bpb']:+.4f} | {'yes' if row['passes'] else 'no'} |"
        )
    lines += [
        "",
        f"- Quality gate: `{d['quality_pass']}`.",
        f"- Factorial interaction gate: `{d['interaction_pass']}`.",
        f"- Causal context gate: `{d['context_pass']}`.",
        f"- Expert-code gate: `{d['codes_pass']}`.",
        f"- Measured throughput gate: `{d['efficiency_pass']}`.",
        "",
        "All models predict identical raw-byte targets from identical windows. "
        "The first-target-patch intervention avoids dilution by later teacher-forced "
        "target bytes. Idealized work ratios exclude local patch encoding/decoding; "
        "measured target bytes/s includes the entire forward path.",
    ]
    (output_dir / "VERDICT.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def self_test() -> None:
    cfg = Config(
        window_bytes=32,
        target_start=8,
        target_bytes=24,
        patch_size=8,
        batch_size=2,
        steps=2,
        eval_batches=2,
        d_model=16,
        n_heads=4,
        n_layers=1,
        d_ff=20,
        n_experts=8,
        top_k=2,
        modal_rank=1,
    )
    cfg.validate()
    set_seed(17)
    windows = torch.randint(0, 256, (4, cfg.window_bytes))
    for representation in ("byte", "patch8"):
        for experts in ("full", "narrow", "modal"):
            model = make_model(cfg, representation, experts)
            logits, aux, routes = model(windows[:2])
            if logits.shape != (2, cfg.target_bytes, 256):
                raise AssertionError((representation, experts, logits.shape))
            loss = F.cross_entropy(
                logits.reshape(-1, 256),
                targets_from_windows(windows[:2], cfg).reshape(-1),
            ) + 0.01 * aux
            loss.backward()
            if not routes or not torch.isfinite(loss):
                raise AssertionError((representation, experts))

    modal = make_model(cfg, "patch8", "modal")
    moe = next(iter_modal_moes(modal))
    probe = torch.randn(2, 3, cfg.d_model)
    direct, _, _ = moe(probe)
    reconstructed = moe.reference_reconstructed(probe)
    error = float(
        torch.max(torch.abs(direct.detach() - reconstructed.detach()))
    )
    if error > 3e-5:
        raise AssertionError(f"modal algebra mismatch: {error}")

    original = {
        key: value.clone() for key, value in modal.state_dict().items()
    }
    with modal_code_policy(modal, "shuffle-code", 99):
        _ = modal(windows[:2])
    for key, value in modal.state_dict().items():
        if not torch.equal(value, original[key]):
            raise AssertionError(f"code policy failed to restore {key}")

    rolled = rolled_context_windows(windows, cfg)
    if not torch.equal(
        rolled[:, cfg.target_start :], windows[:, cfg.target_start :]
    ):
        raise AssertionError("rolled intervention changed targets")
    print(f"self-test passed; modal reconstruction max error={error:.3e}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-bytes", type=Path)
    parser.add_argument("--validation-bytes", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--seeds", default="65501,66602,67703")
    parser.add_argument("--steps", type=int, default=700)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args(argv)

    if args.self_test:
        self_test()
        return 0
    if args.train_bytes is None or args.validation_bytes is None or args.output_dir is None:
        parser.error("train, validation, and output paths are required")

    cfg = Config(steps=args.steps)
    cfg.validate()
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    train_data = np.frombuffer(
        args.train_bytes.read_bytes(), dtype=np.uint8
    )
    validation_data = np.frombuffer(
        args.validation_bytes.read_bytes(), dtype=np.uint8
    )
    seeds = [int(value) for value in args.seeds.split(",") if value.strip()]
    variants = [
        ("byte", "full"),
        ("byte", "narrow"),
        ("byte", "modal"),
        ("patch8", "full"),
        ("patch8", "narrow"),
        ("patch8", "modal"),
    ]
    validation_rng = np.random.default_rng(55123)
    validation_starts = validation_rng.integers(
        0,
        len(validation_data) - cfg.window_bytes,
        size=cfg.eval_batches * cfg.batch_size,
    )
    validation_windows = build_windows(
        validation_data, validation_starts, cfg
    )

    started = time.perf_counter()
    rows: list[dict[str, Any]] = []
    evaluations: dict[tuple[int, str], Evaluation] = {}
    code_ablations: dict[tuple[int, str], Evaluation] = {}
    histories: dict[str, Any] = {}

    for seed in seeds:
        train_rng = np.random.default_rng(seed + 1)
        train_starts = train_rng.integers(
            0,
            len(train_data) - cfg.window_bytes,
            size=cfg.steps * cfg.batch_size,
        )
        train_windows = build_windows(train_data, train_starts, cfg)
        initial_states = initialization_states(cfg, seed)
        for representation, experts in variants:
            variant = f"{representation}-{experts}"
            model, result, correct, rolled, ablations = train_variant(
                cfg,
                representation,
                experts,
                seed,
                initial_states[(representation, experts)],
                train_windows,
                validation_windows,
            )
            rows.append(asdict(result))
            evaluations[(seed, variant)] = correct
            evaluations[(seed, f"{variant}-rolled")] = rolled
            for policy, evaluation in ablations.items():
                code_ablations[(seed, policy)] = evaluation
            histories[variant + f"-seed{seed}"] = {
                "final_train_bpb": result.final_train_bpb,
                "best_validation_bpb": result.best_validation_bpb,
                "training_seconds": result.training_seconds,
            }
            del model
            print(
                f"completed seed={seed} variant={variant} "
                f"bpb={result.validation_bpb:.4f}",
                flush=True,
            )

    summary = summarize(rows)
    decision = make_decision(rows, evaluations, code_ablations)
    payload = {
        "metadata": {
            "task": "autoregressive raw-byte LM over bytes 16..143 of 144-byte windows",
            "dataset": "WikiText-2 raw UTF-8 bytes",
            "seeds": seeds,
            "steps_per_variant": cfg.steps,
            "evaluation_examples_per_seed": len(validation_windows),
            "window_bytes": cfg.window_bytes,
            "target_start": cfg.target_start,
            "target_bytes": cfg.target_bytes,
            "patch_size": cfg.patch_size,
            "d_model": cfg.d_model,
            "d_ff_full": cfg.d_ff,
            "d_ff_narrow": cfg.d_ff // 4,
            "n_layers": cfg.n_layers,
            "n_experts": cfg.n_experts,
            "top_k": cfg.top_k,
            "modal_rank": cfg.modal_rank,
            "elapsed_seconds": time.perf_counter() - started,
            "methodological_note": (
                "Within each seed, all variants use identical train windows, "
                "validation windows, optimizer budget, and shape-compatible "
                "shared initialization. Every model predicts the same raw bytes."
            ),
        },
        "decision": decision,
        "summary": summary,
        "runs": rows,
        "history": histories,
    }
    write_outputs(args.output_dir, payload)
    print((args.output_dir / "VERDICT.md").read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
