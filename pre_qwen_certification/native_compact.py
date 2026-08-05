"""Native compact-MoE training helpers for Native Compact Gate 2A.

The gate compares architectures trained from initialization under identical token
batches and update counts.  It deliberately avoids teacher transplantation: the
question is whether a shared full-rank base plus expert low-rank residuals becomes
competitive when the representation is learned natively rather than imposed
post hoc.
"""
from __future__ import annotations

from dataclasses import dataclass
import copy
import hashlib
import math
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from .heterogeneous_rank import HeterogeneousSharedLowRankResidualMoE
from .modal import ConventionalSwiGLUMoE, MoEGeometry, set_seed
from .reality_gate import evaluate_language_loss, routing_distribution
from .tiny_lm import TinyLMConfig, TinyMoELanguageModel, expert_parameter_count

FULL = "conventional-full"
NARROW = "conventional-narrow65"
PRIMARY = "native-shared-rank"
CANDIDATES = (FULL, NARROW, PRIMARY)


@dataclass(frozen=True)
class NativeArchitectureSpec:
    native_rank: int
    narrow_d_ff: int

    def validate(self, config: TinyLMConfig) -> None:
        if not 1 <= self.native_rank <= min(config.d_model, config.d_ff):
            raise ValueError("native_rank is outside the matrix rank range")
        if not 1 <= self.narrow_d_ff < config.d_ff:
            raise ValueError("narrow_d_ff must be smaller than the full width")


@dataclass(frozen=True)
class NativeAccounting:
    candidate: str
    expert_parameters: int
    full_expert_parameters: int
    expert_parameter_ratio: float
    total_parameters: int
    full_total_parameters: int
    total_parameter_ratio: float
    expert_macs_per_token: float
    full_expert_macs_per_token: float
    expert_compute_ratio: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "candidate": self.candidate,
            "expert_parameters": self.expert_parameters,
            "full_expert_parameters": self.full_expert_parameters,
            "expert_parameter_ratio": self.expert_parameter_ratio,
            "total_parameters": self.total_parameters,
            "full_total_parameters": self.full_total_parameters,
            "total_parameter_ratio": self.total_parameter_ratio,
            "expert_macs_per_token": self.expert_macs_per_token,
            "full_expert_macs_per_token": self.full_expert_macs_per_token,
            "expert_compute_ratio": self.expert_compute_ratio,
        }


def stable_seed(*parts: object) -> int:
    payload = "::".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**31 - 1)


def _copy_router(source: nn.Module, target: nn.Module) -> None:
    with torch.no_grad():
        target.router.weight.copy_(source.router.weight)


def build_paired_candidate_models(
    vocab_size: int,
    config: TinyLMConfig,
    spec: NativeArchitectureSpec,
    *,
    seed: int,
) -> dict[str, TinyMoELanguageModel]:
    """Build candidates with identical non-MoE initialization and router weights."""
    spec.validate(config)
    set_seed(seed)
    base = TinyMoELanguageModel(vocab_size, config)
    models: dict[str, TinyMoELanguageModel] = {}
    for candidate in CANDIDATES:
        model = copy.deepcopy(base)
        for layer_id, block in enumerate(model.blocks):
            source = block.moe
            torch.manual_seed(stable_seed(seed, candidate, layer_id, "expert-init"))
            if candidate == FULL:
                replacement: nn.Module = ConventionalSwiGLUMoE(config.geometry)
            elif candidate == NARROW:
                replacement = ConventionalSwiGLUMoE(
                    MoEGeometry(
                        config.d_model,
                        spec.narrow_d_ff,
                        config.n_experts,
                        config.top_k,
                    )
                )
            elif candidate == PRIMARY:
                replacement = HeterogeneousSharedLowRankResidualMoE(
                    config.geometry,
                    ranks=(spec.native_rank,) * config.n_experts,
                )
            else:  # pragma: no cover - guarded by CANDIDATES
                raise ValueError(candidate)
            _copy_router(source, replacement)
            block.moe = replacement
        models[candidate] = model
    return models


def _expert_macs(module: nn.Module, config: TinyLMConfig) -> float:
    if isinstance(module, ConventionalSwiGLUMoE):
        return float(3 * config.top_k * module.geometry.d_ff * config.d_model)
    if isinstance(module, HeterogeneousSharedLowRankResidualMoE):
        if len(set(module.ranks)) != 1:
            raise ValueError("Native Gate 2A expects a uniform native rank")
        rank = int(module.ranks[0])
        return float(
            3 * config.d_ff * config.d_model
            + 3 * (config.d_ff + config.d_model) * config.top_k * rank
        )
    raise TypeError(f"unsupported candidate module: {type(module)!r}")


def candidate_accounting(
    models: Mapping[str, TinyMoELanguageModel],
    config: TinyLMConfig,
) -> dict[str, NativeAccounting]:
    if set(models) != set(CANDIDATES):
        raise ValueError("all frozen candidates are required for accounting")
    full_model = models[FULL]
    full_expert_parameters = sum(
        expert_parameter_count(block.moe) for block in full_model.blocks
    )
    full_total_parameters = sum(parameter.numel() for parameter in full_model.parameters())
    full_macs = sum(_expert_macs(block.moe, config) for block in full_model.blocks)
    result: dict[str, NativeAccounting] = {}
    for candidate, model in models.items():
        expert_parameters = sum(expert_parameter_count(block.moe) for block in model.blocks)
        total_parameters = sum(parameter.numel() for parameter in model.parameters())
        macs = sum(_expert_macs(block.moe, config) for block in model.blocks)
        result[candidate] = NativeAccounting(
            candidate=candidate,
            expert_parameters=int(expert_parameters),
            full_expert_parameters=int(full_expert_parameters),
            expert_parameter_ratio=float(expert_parameters / full_expert_parameters),
            total_parameters=int(total_parameters),
            full_total_parameters=int(full_total_parameters),
            total_parameter_ratio=float(total_parameters / full_total_parameters),
            expert_macs_per_token=float(macs),
            full_expert_macs_per_token=float(full_macs),
            expert_compute_ratio=float(macs / full_macs),
        )
    return result


def _cpu_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
    }


def train_matched_candidates(
    models: Mapping[str, TinyMoELanguageModel],
    corpus,
    config: TinyLMConfig,
    *,
    steps: int,
    evaluation_interval: int,
    calibration_windows_per_document: int,
    calibration_seed: int,
    seed: int,
) -> dict[str, Any]:
    """Train all candidates on exactly the same sampled batches and updates."""
    if set(models) != set(CANDIDATES):
        raise ValueError("candidate set is incomplete")
    if steps <= 0 or evaluation_interval <= 0:
        raise ValueError("training steps and evaluation interval must be positive")
    optimizers = {
        name: torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        for name, model in models.items()
    }
    generator = torch.Generator().manual_seed(seed + 101)
    histories: dict[str, list[dict[str, Any]]] = {name: [] for name in CANDIDATES}
    best_loss = {name: float("inf") for name in CANDIDATES}
    best_step = {name: 0 for name in CANDIDATES}
    best_states: dict[str, dict[str, torch.Tensor]] = {}
    rolling: dict[str, list[float]] = {name: [] for name in CANDIDATES}

    for model in models.values():
        model.train()
    for step in range(1, steps + 1):
        tokens, targets = corpus.sample_batch("train", config.batch_size, generator)
        for name in CANDIDATES:
            model = models[name]
            logits, auxiliary, _ = model(tokens)
            language_loss = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                targets.reshape(-1),
            )
            objective = language_loss + config.aux_weight * auxiliary
            optimizer = optimizers[name]
            optimizer.zero_grad(set_to_none=True)
            objective.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()
            rolling[name].append(float(language_loss.detach()))

        if step % evaluation_interval != 0 and step != steps:
            continue
        for name in CANDIDATES:
            model = models[name]
            validation_loss = evaluate_language_loss(
                model,
                corpus,
                split="calibration",
                windows_per_document=calibration_windows_per_document,
                seed=calibration_seed,
            )
            route = routing_distribution(
                model,
                corpus,
                split="calibration",
                layer_id=0,
                windows_per_document=calibration_windows_per_document,
                seed=calibration_seed,
            )
            recent = rolling[name][-evaluation_interval:]
            histories[name].append(
                {
                    "step": int(step),
                    "training_loss": float(np.mean(recent)),
                    "validation_loss": float(validation_loss),
                    "route_distribution": [float(value) for value in route],
                }
            )
            if validation_loss < best_loss[name]:
                best_loss[name] = float(validation_loss)
                best_step[name] = int(step)
                best_states[name] = _cpu_state_dict(model)
            model.train()

    final_states = {name: _cpu_state_dict(models[name]) for name in CANDIDATES}
    if set(best_states) != set(CANDIDATES):
        raise AssertionError("best state was not captured for every candidate")
    return {
        "histories": histories,
        "best_calibration_loss": best_loss,
        "best_step": best_step,
        "best_states": best_states,
        "final_states": final_states,
        "steps": int(steps),
        "batch_size": int(config.batch_size),
        "training_tokens_per_candidate": int(steps * config.batch_size * config.seq_len),
    }


def restore_phase_models(
    templates: Mapping[str, TinyMoELanguageModel],
    states: Mapping[str, Mapping[str, torch.Tensor]],
) -> dict[str, TinyMoELanguageModel]:
    if set(templates) != set(CANDIDATES) or set(states) != set(CANDIDATES):
        raise ValueError("candidate templates/states are incomplete")
    result: dict[str, TinyMoELanguageModel] = {}
    for name in CANDIDATES:
        model = copy.deepcopy(templates[name])
        model.load_state_dict(states[name])
        model.eval()
        result[name] = model
    return result


@torch.no_grad()
def evaluate_native_model(
    model: TinyMoELanguageModel,
    corpus,
    *,
    split: str,
    windows_per_document: int,
    evaluation_seed: int,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    windows = corpus.fixed_windows(
        split,
        windows_per_document=windows_per_document,
        seed=evaluation_seed,
    )
    model.eval()
    losses: list[float] = []
    entropies: list[float] = []
    confidences: list[float] = []
    records: list[dict[str, Any]] = []
    for window in windows:
        logits, _, capture = model(window.inputs[None, :], collect_layer=0)
        if capture is None:
            raise AssertionError("routing capture missing during native evaluation")
        token_loss = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            window.targets.reshape(-1),
            reduction="none",
        )
        probabilities = torch.softmax(logits.double(), dim=-1)
        entropy = -torch.sum(
            probabilities * torch.log(probabilities.clamp_min(1e-12)), dim=-1
        ).mean()
        confidence = probabilities.max(dim=-1).values.mean()
        loss = float(token_loss.mean())
        losses.append(loss)
        entropies.append(float(entropy))
        confidences.append(float(confidence))
        records.append(
            {
                "document_id": str(window.document_id),
                "domain": str(window.domain),
                "start": int(window.start),
                "loss": loss,
                "entropy": float(entropy),
                "confidence": float(confidence),
            }
        )
    mean_loss = float(np.mean(losses))
    return {
        "loss": mean_loss,
        "perplexity": float(math.exp(min(mean_loss, 20.0))),
        "entropy": float(np.mean(entropies)),
        "confidence": float(np.mean(confidences)),
        "windows": int(len(windows)),
    }, records


def route_health(distribution: Sequence[float], top_k: int) -> dict[str, float | int]:
    values = np.asarray(distribution, dtype=np.float64)
    if values.ndim != 1 or np.any(values < 0):
        raise ValueError("invalid route distribution")
    probabilities = values / max(float(values.sum()), 1e-12)
    nonzero = probabilities[probabilities > 0]
    entropy = -float(np.sum(nonzero * np.log(nonzero)))
    normalized = entropy / math.log(len(values)) if len(values) > 1 else 0.0
    return {
        "entropy": entropy,
        "normalized_entropy": normalized,
        "dead_experts": int(np.sum(values <= 1e-12)),
        "maximum_frequency": float(values.max()),
        "minimum_frequency": float(values.min()),
        "top_k": int(top_k),
    }
