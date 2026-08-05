"""Teacher plateau and compressibility-trajectory helpers for Reality Gate 1A."""
from __future__ import annotations

from dataclasses import dataclass
import copy
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from .heterogeneous_rank import route_frequencies
from .modal import ConventionalSwiGLUMoE, set_seed
from .tiny_lm import TinyLMConfig, TinyMoELanguageModel


@dataclass(frozen=True)
class PlateauRule:
    evaluation_interval: int
    minimum_steps: int
    maximum_steps: int
    window: int
    patience: int
    maximum_negative_slope_per_step: float
    maximum_positive_slope_per_step: float
    maximum_window_improvement: float
    maximum_window_range: float
    maximum_route_l1_change: float
    validation_windows_per_document: int
    validation_seed: int

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "PlateauRule":
        result = cls(
            evaluation_interval=int(values["evaluation_interval"]),
            minimum_steps=int(values["minimum_steps"]),
            maximum_steps=int(values["maximum_steps"]),
            window=int(values["window"]),
            patience=int(values["patience"]),
            maximum_negative_slope_per_step=float(values["maximum_negative_slope_per_step"]),
            maximum_positive_slope_per_step=float(values["maximum_positive_slope_per_step"]),
            maximum_window_improvement=float(values["maximum_window_improvement"]),
            maximum_window_range=float(values["maximum_window_range"]),
            maximum_route_l1_change=float(values["maximum_route_l1_change"]),
            validation_windows_per_document=int(values["validation_windows_per_document"]),
            validation_seed=int(values["validation_seed"]),
        )
        if result.evaluation_interval <= 0 or result.minimum_steps <= 0:
            raise ValueError("plateau step thresholds must be positive")
        if result.maximum_steps < result.minimum_steps:
            raise ValueError("maximum_steps must be >= minimum_steps")
        if result.window < 3 or result.patience <= 0:
            raise ValueError("plateau window/patience are invalid")
        return result


@torch.no_grad()
def evaluate_language_loss(
    model: TinyMoELanguageModel,
    corpus,
    *,
    split: str,
    windows_per_document: int,
    seed: int,
) -> float:
    windows = corpus.fixed_windows(
        split, windows_per_document=windows_per_document, seed=seed
    )
    model.eval()
    values: list[float] = []
    for window in windows:
        logits, _, _ = model(window.inputs[None, :])
        loss = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            window.targets.reshape(-1),
        )
        values.append(float(loss))
    return float(np.mean(values))


@torch.no_grad()
def routing_distribution(
    model: TinyMoELanguageModel,
    corpus,
    *,
    split: str,
    layer_id: int,
    windows_per_document: int,
    seed: int,
) -> torch.Tensor:
    windows = corpus.fixed_windows(
        split, windows_per_document=windows_per_document, seed=seed
    )
    ids: list[torch.Tensor] = []
    model.eval()
    for window in windows:
        _, _, capture = model(window.inputs[None, :], collect_layer=layer_id)
        if capture is None:
            raise AssertionError("missing routing capture")
        ids.append(capture.routing.top_ids.cpu())
    n_experts = model.config.n_experts
    return route_frequencies(torch.cat(ids), n_experts)


def plateau_window_status(
    history: Sequence[Mapping[str, Any]],
    rule: PlateauRule,
) -> dict[str, float | bool]:
    if len(history) < rule.window:
        return {
            "eligible": False,
            "plateau": False,
            "slope_per_step": float("nan"),
            "window_improvement": float("nan"),
            "window_range": float("nan"),
            "route_l1_change": float("nan"),
        }
    recent = list(history[-rule.window :])
    steps = np.asarray([float(row["step"]) for row in recent], dtype=np.float64)
    losses = np.asarray([float(row["validation_loss"]) for row in recent], dtype=np.float64)
    slope = float(np.polyfit(steps - steps.mean(), losses, deg=1)[0])
    improvement = float(losses[0] - losses.min())
    window_range = float(losses.max() - losses.min())
    route_values = [
        np.asarray(row["route_distribution"], dtype=np.float64) for row in recent
    ]
    route_l1 = max(
        float(np.abs(current - previous).sum())
        for previous, current in zip(route_values[:-1], route_values[1:], strict=True)
    )
    eligible = int(recent[-1]["step"]) >= rule.minimum_steps
    plateau = (
        eligible
        and slope >= -rule.maximum_negative_slope_per_step
        and slope <= rule.maximum_positive_slope_per_step
        and improvement <= rule.maximum_window_improvement
        and window_range <= rule.maximum_window_range
        and route_l1 <= rule.maximum_route_l1_change
    )
    return {
        "eligible": bool(eligible),
        "plateau": bool(plateau),
        "slope_per_step": slope,
        "window_improvement": improvement,
        "window_range": window_range,
        "route_l1_change": route_l1,
    }


def _cpu_state_dict(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def train_teacher_to_plateau(
    corpus,
    config: TinyLMConfig,
    *,
    seed: int,
    rule: PlateauRule,
    trajectory_fractions: Sequence[float],
    layer_id: int = 0,
) -> tuple[TinyMoELanguageModel, list[dict[str, Any]], dict[str, dict[str, torch.Tensor]], dict[str, Any]]:
    """Train a conventional teacher and stop only under the frozen plateau rule."""
    set_seed(seed)
    model = TinyMoELanguageModel(corpus.vocab_size, config)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    generator = torch.Generator().manual_seed(seed + 101)
    history: list[dict[str, Any]] = []
    snapshots: dict[str, dict[str, torch.Tensor]] = {}
    targets = sorted(
        {
            max(1, min(rule.maximum_steps, int(round(rule.maximum_steps * float(fraction)))))
            for fraction in trajectory_fractions
            if 0.0 < float(fraction) <= 1.0
        }
    )
    next_target = 0
    plateau_streak = 0
    train_losses: list[float] = []
    model.train()
    for step in range(1, rule.maximum_steps + 1):
        tokens, targets_batch = corpus.sample_batch("train", config.batch_size, generator)
        logits, aux, _ = model(tokens)
        language_loss = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), targets_batch.reshape(-1)
        )
        loss = language_loss + config.aux_weight * aux
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        optimizer.step()
        train_losses.append(float(language_loss.detach()))

        while next_target < len(targets) and step >= targets[next_target]:
            label = f"fraction-{targets[next_target] / rule.maximum_steps:.3f}"
            snapshots[label] = _cpu_state_dict(model)
            next_target += 1

        if step % rule.evaluation_interval != 0 and step != rule.maximum_steps:
            continue
        validation_loss = evaluate_language_loss(
            model,
            corpus,
            split="calibration",
            windows_per_document=rule.validation_windows_per_document,
            seed=rule.validation_seed,
        )
        frequencies = routing_distribution(
            model,
            corpus,
            split="calibration",
            layer_id=layer_id,
            windows_per_document=rule.validation_windows_per_document,
            seed=rule.validation_seed,
        )
        row: dict[str, Any] = {
            "step": int(step),
            "training_loss": float(np.mean(train_losses[-rule.evaluation_interval :])),
            "validation_loss": validation_loss,
            "route_distribution": [float(value) for value in frequencies],
        }
        history.append(row)
        status = plateau_window_status(history, rule)
        row["plateau_status"] = status
        plateau_streak = plateau_streak + 1 if status["plateau"] else 0
        row["plateau_streak"] = plateau_streak
        if plateau_streak >= rule.patience:
            break
        model.train()

    snapshots["final"] = _cpu_state_dict(model)
    final_status = plateau_window_status(history, rule)
    metadata = {
        "plateau_reached": bool(plateau_streak >= rule.patience),
        "final_step": int(history[-1]["step"] if history else rule.maximum_steps),
        "plateau_streak": int(plateau_streak),
        "final_plateau_window": final_status,
        "snapshot_labels": list(snapshots),
    }
    model.eval()
    return model, history, snapshots, metadata


def restore_teacher(
    state: Mapping[str, torch.Tensor],
    *,
    vocab_size: int,
    config: TinyLMConfig,
) -> TinyMoELanguageModel:
    model = TinyMoELanguageModel(vocab_size, config)
    model.load_state_dict(state)
    model.eval()
    if not isinstance(model.blocks[0].moe, ConventionalSwiGLUMoE):
        raise TypeError("restored teacher is not conventional")
    return model
