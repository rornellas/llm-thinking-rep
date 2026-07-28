"""Known-truth synthetic teachers and controlled functional distillation."""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from .metrics import rowwise_nrmse, tensor_metrics
from .modal import (
    ConventionalSwiGLUMoE,
    MoEGeometry,
    ScalarModalMoE,
    set_seed,
)


@dataclass
class ActivationSet:
    inputs: torch.Tensor
    document_ids: list[str]
    sequence_ids: list[str]
    token_positions: torch.Tensor
    domains: list[str]

    def validate(self) -> None:
        tokens = self.inputs.shape[0]
        if self.inputs.ndim != 2:
            raise ValueError("inputs must have shape [tokens, d_model]")
        if not (
            len(self.document_ids)
            == len(self.sequence_ids)
            == len(self.domains)
            == tokens
        ):
            raise ValueError("activation metadata length mismatch")
        if tuple(self.token_positions.shape) != (tokens,):
            raise ValueError("token_positions must have shape [tokens]")


def sample_document_activations(
    *,
    d_model: int,
    documents: int,
    tokens_per_document: int,
    seed: int,
    prefix: str,
    domain_shift: float = 0.0,
) -> ActivationSet:
    if documents <= 0 or tokens_per_document <= 0:
        raise ValueError("documents and tokens_per_document must be positive")
    generator = torch.Generator().manual_seed(seed)
    domains = ("general", "code", "math", "portuguese")
    chunks: list[torch.Tensor] = []
    document_ids: list[str] = []
    sequence_ids: list[str] = []
    token_positions: list[int] = []
    domain_rows: list[str] = []
    for document_index in range(documents):
        domain = domains[document_index % len(domains)]
        mean = torch.randn(d_model, generator=generator) * 0.65
        if domain == "code":
            mean[: max(1, d_model // 4)] += 0.8 + domain_shift
        elif domain == "math":
            mean[d_model // 4 : d_model // 2] -= 0.7 + domain_shift
        elif domain == "portuguese":
            mean[d_model // 2 : 3 * d_model // 4] += 0.55 + domain_shift
        else:
            mean[-max(1, d_model // 4) :] -= domain_shift
        diagonal = 0.55 + torch.rand(d_model, generator=generator) * 0.85
        values = (
            torch.randn(tokens_per_document, d_model, generator=generator)
            * diagonal
            + mean
        )
        # Add a weak causal-like drift so a one-token alignment fault is observable.
        positions = torch.linspace(-0.25, 0.25, tokens_per_document)[:, None]
        direction = torch.randn(d_model, generator=generator)
        direction = direction / torch.linalg.vector_norm(direction).clamp_min(1e-12)
        values = values + positions * direction[None, :]
        chunks.append(values)
        document_id = f"{prefix}-doc-{document_index:04d}"
        sequence_id = f"{document_id}-seq-0000"
        document_ids.extend([document_id] * tokens_per_document)
        sequence_ids.extend([sequence_id] * tokens_per_document)
        token_positions.extend(range(tokens_per_document))
        domain_rows.extend([domain] * tokens_per_document)
    result = ActivationSet(
        inputs=torch.cat(chunks, dim=0),
        document_ids=document_ids,
        sequence_ids=sequence_ids,
        token_positions=torch.tensor(token_positions, dtype=torch.long),
        domains=domain_rows,
    )
    result.validate()
    return result


def make_modal_teacher(
    geometry: MoEGeometry,
    *,
    rank: int,
    seed: int,
    residual_scale: float = 0.8,
    code_scale: float = 1.0,
) -> ScalarModalMoE:
    set_seed(seed)
    model = ScalarModalMoE(geometry, rank)
    with torch.no_grad():
        for bank in (model.gate_modes, model.up_modes, model.down_modes):
            if rank:
                bank[1:].mul_(residual_scale)
        for codes in (model.gate_codes, model.up_codes, model.down_codes):
            if codes.numel():
                codes.mul_(code_scale)
    return model


def make_negative_teacher(
    geometry: MoEGeometry,
    *,
    seed: int,
) -> ConventionalSwiGLUMoE:
    set_seed(seed)
    teacher = ConventionalSwiGLUMoE(geometry)
    # Increase expert-axis diversity while retaining a realistic common component.
    with torch.no_grad():
        for bank in (teacher.gate, teacher.up, teacher.down):
            common = bank.mean(dim=0, keepdim=True)
            centered = bank - common
            bank.copy_(common + 1.35 * centered)
    return teacher


def add_independent_expert_noise(
    teacher: ConventionalSwiGLUMoE,
    *,
    epsilon: float,
    seed: int,
) -> ConventionalSwiGLUMoE:
    if epsilon < 0.0:
        raise ValueError("epsilon must be non-negative")
    set_seed(seed)
    result = ConventionalSwiGLUMoE(teacher.geometry)
    with torch.no_grad():
        result.router.weight.copy_(teacher.router.weight)
        for source, target in (
            (teacher.gate, result.gate),
            (teacher.up, result.up),
            (teacher.down, result.down),
        ):
            noise = torch.randn_like(source)
            # Normalize globally so epsilon has a comparable interpretation across banks.
            noise.mul_(
                torch.linalg.vector_norm(source)
                / torch.linalg.vector_norm(noise).clamp_min(1e-12)
            )
            target.copy_(source + epsilon * noise)
    return result


def evaluate_student(
    teacher: ConventionalSwiGLUMoE,
    student: nn.Module,
    activations: ActivationSet,
    *,
    seed: int,
    label: str,
) -> tuple[dict[str, float], list[dict[str, object]]]:
    activations.validate()
    teacher.eval()
    student.eval()
    with torch.no_grad():
        target, routing = teacher(activations.inputs)
        prediction, student_routing = student(
            activations.inputs,
            forced_top_ids=routing.top_ids,
            forced_weights=routing.weights,
        )
    aggregate = tensor_metrics(prediction, target).as_dict()
    aggregate["router_topk_agreement"] = float(
        torch.mean(
            (student_routing.top_ids == routing.top_ids).all(dim=-1).float()
        )
    )
    errors = rowwise_nrmse(prediction, target)
    records = [
        {
            "seed": int(seed),
            "document_id": activations.document_ids[index],
            "sequence_id": activations.sequence_ids[index],
            "token_position": int(activations.token_positions[index]),
            "domain": activations.domains[index],
            "variant": label,
            "nrmse": float(errors[index]),
        }
        for index in range(len(errors))
    ]
    return aggregate, records


def fine_tune_student(
    teacher: ConventionalSwiGLUMoE,
    student: ScalarModalMoE,
    train: ActivationSet,
    *,
    steps: int,
    batch_size: int,
    lr: float,
    seed: int,
    shuffle_targets: bool = False,
) -> list[dict[str, float]]:
    if steps < 0:
        raise ValueError("steps must be non-negative")
    train.validate()
    if steps == 0:
        return []
    teacher.eval()
    student.train()
    with torch.no_grad():
        targets, routing = teacher(train.inputs)
        if shuffle_targets:
            permutation = torch.randperm(
                len(targets), generator=torch.Generator().manual_seed(seed + 91)
            )
            targets = targets.index_select(0, permutation)
    # An exact known-truth recovery must not be moved away from the solution by
    # optimizer noise or regularization.  This check uses training data only.
    with torch.no_grad():
        initial_prediction, _ = student(
            train.inputs,
            forced_top_ids=routing.top_ids,
            forced_weights=routing.weights,
        )
        initial_scale = torch.mean(targets.square()).clamp_min(1e-8)
        initial_normalized_mse = float(
            torch.mean((initial_prediction - targets).square()) / initial_scale
        )
    if initial_normalized_mse <= 1e-12:
        return [
            {
                "step": 0.0,
                "objective": initial_normalized_mse,
                "normalized_mse": initial_normalized_mse,
                "cosine_penalty": 0.0,
                "skipped_exact_solution": 1.0,
            }
        ]

    parameters = [parameter for parameter in student.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=lr, weight_decay=0.0)
    generator = torch.Generator().manual_seed(seed)
    history: list[dict[str, float]] = []
    for step in range(1, steps + 1):
        indices = torch.randint(
            0,
            len(train.inputs),
            (min(batch_size, len(train.inputs)),),
            generator=generator,
        )
        x = train.inputs.index_select(0, indices)
        y = targets.index_select(0, indices)
        top_ids = routing.top_ids.index_select(0, indices)
        route_weights = routing.weights.index_select(0, indices)
        prediction, _ = student(
            x,
            forced_top_ids=top_ids,
            forced_weights=route_weights,
        )
        scale = torch.mean(y.square()).detach().clamp_min(1e-8)
        normalized_mse = torch.mean((prediction - y).square()) / scale
        cosine = 1.0 - F.cosine_similarity(prediction, y, dim=-1).mean()
        loss = normalized_mse + 0.1 * cosine
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, 1.0)
        optimizer.step()
        if step == 1 or step == steps or step % max(steps // 4, 1) == 0:
            history.append(
                {
                    "step": float(step),
                    "objective": float(loss.detach()),
                    "normalized_mse": float(normalized_mse.detach()),
                    "cosine_penalty": float(cosine.detach()),
                }
            )
    return history


def rank_sweep(
    teacher: ConventionalSwiGLUMoE,
    train: ActivationSet,
    validation: ActivationSet,
    *,
    ranks: Sequence[int],
    seed: int,
    fine_tune_steps: int,
    batch_size: int,
    lr: float,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    summaries: list[dict[str, object]] = []
    records: list[dict[str, object]] = []
    for rank in ranks:
        student, decomposition = ScalarModalMoE.from_conventional_svd(
            teacher, int(rank), freeze_router=True
        )
        before, before_records = evaluate_student(
            teacher,
            student,
            validation,
            seed=seed,
            label=f"k{rank}-before",
        )
        history = fine_tune_student(
            teacher,
            student,
            train,
            steps=fine_tune_steps,
            batch_size=batch_size,
            lr=lr,
            seed=seed + rank * 101,
        )
        after, after_records = evaluate_student(
            teacher,
            student,
            validation,
            seed=seed,
            label=f"k{rank}-after",
        )
        summaries.append(
            {
                "rank": int(rank),
                "decomposition": decomposition,
                "before": before,
                "after": after,
                "history": history,
                "expert_parameters": student.expert_transform_parameter_count(),
                "idealized_expert_compute_ratio": student.idealized_expert_compute_ratio(),
            }
        )
        records.extend(before_records)
        records.extend(after_records)
    return summaries, records


def input_blind_baseline(
    teacher: ConventionalSwiGLUMoE,
    train: ActivationSet,
    validation: ActivationSet,
) -> dict[str, float]:
    teacher.eval()
    with torch.no_grad():
        train_targets, _ = teacher(train.inputs)
        validation_targets, _ = teacher(validation.inputs)
        mean = train_targets.mean(dim=0, keepdim=True)
        prediction = mean.expand_as(validation_targets)
    return tensor_metrics(prediction, validation_targets).as_dict()


def shuffled_target_control(
    teacher: ConventionalSwiGLUMoE,
    train: ActivationSet,
    validation: ActivationSet,
    *,
    rank: int,
    seed: int,
    steps: int,
    batch_size: int,
    lr: float,
) -> tuple[dict[str, float], list[dict[str, float]]]:
    set_seed(seed)
    student = ScalarModalMoE(teacher.geometry, rank)
    with torch.no_grad():
        student.router.weight.copy_(teacher.router.weight)
    student.router.weight.requires_grad_(False)
    history = fine_tune_student(
        teacher,
        student,
        train,
        steps=steps,
        batch_size=batch_size,
        lr=lr,
        seed=seed + 1,
        shuffle_targets=True,
    )
    metrics, _ = evaluate_student(
        teacher, student, validation, seed=seed, label="shuffled-target"
    )
    return metrics, history


def monotonicity(values: Sequence[float], outcomes: Sequence[float]) -> dict[str, float | bool]:
    if len(values) != len(outcomes) or len(values) < 2:
        raise ValueError("at least two paired values are required")
    x = np.asarray(values, dtype=np.float64)
    y = np.asarray(outcomes, dtype=np.float64)
    correlation = float(np.corrcoef(x, y)[0, 1]) if np.std(y) > 0 else 0.0
    nondecreasing = bool(np.all(np.diff(y) >= -1e-8))
    return {"pearson": correlation, "nondecreasing": nondecreasing}
