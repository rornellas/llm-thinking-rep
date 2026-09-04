"""FA-1: frozen functional interventions and inference-only model exports.

No training, fitting, or calibration-dependent selection takes place here.
"""
from __future__ import annotations
import copy
from typing import Sequence
import torch
from torch import nn
import torch.nn.functional as F
from .heterogeneous_rank import HeterogeneousSharedLowRankResidualMoE
from .modal import ConventionalSwiGLUMoE, route_topk

BASE_INTERVENTIONS = ('original', 'mean-matrices', 'permute-1', 'permute-5',
                      'permute-7', 'uniform-selected')
COMPACT_INTERVENTIONS = BASE_INTERVENTIONS + ('common-only', 'rank1')


class DenseSwiGLU(nn.Module):
    """Actually discards router, factors, and unused expert banks."""
    def __init__(self, gate: torch.Tensor, up: torch.Tensor, down: torch.Tensor):
        super().__init__()
        self.gate = nn.Parameter(gate.detach().clone())
        self.up = nn.Parameter(up.detach().clone())
        self.down = nn.Parameter(down.detach().clone())

    def forward(self, x: torch.Tensor):
        return F.linear(F.silu(F.linear(x, self.gate)) * F.linear(x, self.up), self.down), None


class RouteIntervention(nn.Module):
    def __init__(self, source: nn.Module, kind: str):
        super().__init__()
        if kind not in ('permute-1', 'permute-5', 'permute-7', 'uniform-selected'):
            raise ValueError(kind)
        self.source = copy.deepcopy(source)
        self.kind = kind

    def forward(self, x: torch.Tensor):
        m = self.source
        routing = route_topk(x, m.router.weight, m.geometry.top_k)
        ids, weights = routing.top_ids, routing.weights
        if self.kind == 'uniform-selected':
            weights = torch.full_like(weights, 1.0 / m.geometry.top_k)
        else:
            ids = (ids + int(self.kind.split('-')[1])) % m.geometry.n_experts
        return m(x, forced_top_ids=ids, forced_weights=weights)


@torch.no_grad()
def effective_banks(m: nn.Module) -> tuple[torch.Tensor, ...]:
    if isinstance(m, HeterogeneousSharedLowRankResidualMoE):
        return m.reconstruct_weights()
    if isinstance(m, ConventionalSwiGLUMoE):
        return m.gate.detach(), m.up.detach(), m.down.detach()
    raise TypeError(type(m))


@torch.no_grad()
def transform(m: nn.Module, kind: str) -> nn.Module:
    compact = isinstance(m, HeterogeneousSharedLowRankResidualMoE)
    if kind not in (COMPACT_INTERVENTIONS if compact else BASE_INTERVENTIONS):
        raise ValueError(f'{kind} unsupported for {type(m)}')
    if kind == 'original':
        return copy.deepcopy(m)
    if kind == 'mean-matrices':
        banks = effective_banks(m)
        return DenseSwiGLU(*(bank.double().mean(0).to(bank.dtype) for bank in banks))
    if kind == 'common-only':
        return DenseSwiGLU(m.common_gate, m.common_up, m.common_down)
    if kind == 'rank1':
        out = HeterogeneousSharedLowRankResidualMoE(m.geometry, (1,) * m.geometry.n_experts)
        out.router.weight.copy_(m.router.weight)
        for bank in ('gate', 'up', 'down'):
            getattr(out, 'common_' + bank).copy_(getattr(m, 'common_' + bank))
            for i, (left, right) in enumerate(zip(getattr(m, bank+'_left'), getattr(m, bank+'_right'), strict=True)):
                residual = left.double() @ right.double()
                u, s, vh = torch.linalg.svd(residual, full_matrices=False)
                root = s[0].clamp_min(0).sqrt()
                getattr(out, bank+'_left')[i].copy_(u[:, :1] * root)
                getattr(out, bank+'_right')[i].copy_(vh[:1, :] * root)
        return out
    return RouteIntervention(m, kind)


class InferenceBlock(nn.Module):
    def __init__(self, block: nn.Module, kind: str):
        super().__init__()
        self.norm1 = copy.deepcopy(block.norm1)
        self.attention = copy.deepcopy(block.attention)
        self.norm2 = copy.deepcopy(block.norm2)
        self.moe = transform(block.moe, kind)

    def forward(self, x: torch.Tensor, mask: torch.Tensor):
        n = self.norm1(x)
        a, _ = self.attention(n, n, n, attn_mask=mask, need_weights=False)
        x = x + a
        y, _ = self.moe(self.norm2(x).flatten(0, 1))
        return x + y.reshape_as(x)


class InferenceLM(nn.Module):
    """Shares only tied embedding/output, never registers the source model."""
    def __init__(self, model: nn.Module, kind: str, layers: Sequence[int] | None = None):
        super().__init__()
        self.config = model.config
        self.token_embedding = copy.deepcopy(model.token_embedding)
        self.position_embedding = copy.deepcopy(model.position_embedding)
        selected = set(range(len(model.blocks))) if layers is None else set(layers)
        self.blocks = nn.ModuleList(InferenceBlock(b, kind if i in selected else 'original')
                                    for i, b in enumerate(model.blocks))
        self.norm = copy.deepcopy(model.norm)
        self.output = nn.Linear(model.config.d_model, model.token_embedding.num_embeddings, bias=False)
        self.output.weight = self.token_embedding.weight
        self.eval()

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.ndim != 2 or tokens.shape[1] > self.config.seq_len:
            raise ValueError('tokens must be batch x sequence within configured context')
        positions = torch.arange(tokens.shape[1], device=tokens.device)
        x = self.token_embedding(tokens) + self.position_embedding(positions)[None, :, :]
        mask = torch.triu(torch.full((tokens.shape[1], tokens.shape[1]), -float('inf'),
                                    device=tokens.device), diagonal=1)
        for block in self.blocks:
            x = block(x, mask)
        return self.output(self.norm(x))


def accounting(model: InferenceLM) -> dict[str, int]:
    total = sum(p.numel() for p in model.parameters())
    bytes_ = sum(p.numel() * p.element_size() for p in model.parameters())
    macs, expert_parameters, router_parameters = 0, 0, 0
    for block in model.blocks:
        m = block.moe
        if isinstance(m, RouteIntervention):
            m = m.source
        if isinstance(m, DenseSwiGLU):
            p = sum(x.numel() for x in m.parameters())
            expert_parameters += p
            macs += p
        elif isinstance(m, ConventionalSwiGLUMoE):
            g = m.geometry
            expert_parameters += 3*g.n_experts*g.d_ff*g.d_model
            router_parameters += g.n_experts*g.d_model
            macs += 3*g.top_k*g.d_ff*g.d_model
        elif isinstance(m, HeterogeneousSharedLowRankResidualMoE):
            g = m.geometry
            assert len(set(m.ranks)) == 1
            expert_parameters += m.expert_transform_parameter_count()
            router_parameters += g.n_experts*g.d_model
            macs += 3*g.d_ff*g.d_model + 3*(g.d_ff+g.d_model)*g.top_k*m.ranks[0]
        else:
            raise TypeError(type(m))
    return {'total_parameters':total, 'parameter_bytes':bytes_, 'expert_parameters':expert_parameters,
            'router_parameters':router_parameters, 'expert_matrix_macs_per_token':macs}
