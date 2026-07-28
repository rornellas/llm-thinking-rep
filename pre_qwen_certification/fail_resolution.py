"""Exploratory experiments that target the failed pre-Qwen comparison.

This module does not alter the frozen v1.3 decision.  It provides a development
screen on fresh, non-sealed documents to separate four hypotheses that the
original experiment confounded:

1. insufficient optimization budget;
2. mismatch between local layer regression and closed-loop language loss;
3. insufficient code granularity or factor-specific capacity;
4. an unfairly one-dimensional baseline comparison rather than a parameter /
   arithmetic / quality Pareto frontier.

Any candidate selected here requires a new preregistered protocol and a new
sealed holdout before it may change the GO/NO-GO decision.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import copy
import hashlib
import math
from typing import Any, Literal, Sequence

import torch
from torch import nn
import torch.nn.functional as F

from .modal import (
    AsymmetricScalarModalMoE,
    ClusteredResidualMoE,
    ConventionalSwiGLUMoE,
    NeuronwiseModalMoE,
    ResidualScalarModalMoE,
    ScalarModalMoE,
)
from .metrics import paired_kl_from_logits, tensor_metrics
from .tiny_lm import (
    CapturedLayerDataset,
    CharacterCorpus,
    TinyMoELanguageModel,
    distill_layer_student,
    evaluate_closed_loop,
    evaluate_local_student,
    expert_parameter_count,
    install_student,
    joint_fine_tune_transplant,
    make_narrow_student,
)


Objective = Literal["aggregate", "expert-output", "factor-resolved"]


@dataclass(frozen=True)
class CandidateSpec:
    name: str
    family: Literal["scalar", "neuronwise", "asymmetric", "residual", "clustered", "narrow"]
    local_steps: int
    joint_steps: int = 0
    objective: Objective = "aggregate"
    rank: int | None = None
    ranks: tuple[int, int, int] | None = None
    narrow_d_ff: int | None = None
    residual_rank: int | None = None
    n_groups: int | None = None
    notes: str = ""

    def validate(self) -> None:
        if self.local_steps <= 0 or self.joint_steps < 0:
            raise ValueError("local_steps must be positive and joint_steps non-negative")
        if self.family in {"scalar", "neuronwise"}:
            if self.rank is None or self.rank < 0:
                raise ValueError(f"{self.family} requires rank")
        elif self.family == "asymmetric":
            if self.ranks is None or len(self.ranks) != 3:
                raise ValueError("asymmetric requires (gate, up, down) ranks")
            if any(value < 0 for value in self.ranks):
                raise ValueError("asymmetric ranks must be non-negative")
        elif self.family == "residual":
            if self.rank is None or self.rank < 0:
                raise ValueError("residual requires non-negative scalar rank")
            if self.residual_rank is None or self.residual_rank < 0:
                raise ValueError("residual requires non-negative residual_rank")
        elif self.family == "clustered":
            if self.n_groups is None or self.n_groups <= 0:
                raise ValueError("clustered requires positive n_groups")
            if self.residual_rank is None or self.residual_rank < 0:
                raise ValueError("clustered requires non-negative residual_rank")
        elif self.family == "narrow":
            if self.narrow_d_ff is None or self.narrow_d_ff <= 0:
                raise ValueError("narrow requires positive narrow_d_ff")
            if self.objective != "aggregate":
                raise ValueError("factor-resolved objectives require equal hidden width")
        else:
            raise ValueError(self.family)


def stable_seed(base_seed: int, name: str, phase: str) -> int:
    """Derive process-independent experiment seeds from explicit labels."""
    digest = hashlib.sha256(
        f"{base_seed}\0{name}\0{phase}".encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], "big") % (2**63 - 1)


def build_candidate(
    teacher_moe: ConventionalSwiGLUMoE,
    spec: CandidateSpec,
    *,
    calibration_inputs: torch.Tensor | None = None,
) -> tuple[nn.Module, dict[str, Any]]:
    spec.validate()
    if spec.family == "scalar":
        student, initialization = ScalarModalMoE.from_conventional_svd(
            teacher_moe, int(spec.rank), freeze_router=True
        )
    elif spec.family == "neuronwise":
        student, initialization = NeuronwiseModalMoE.from_conventional_svd(
            teacher_moe, int(spec.rank), freeze_router=True
        )
    elif spec.family == "asymmetric":
        student, initialization = AsymmetricScalarModalMoE.from_conventional_svd(
            teacher_moe, tuple(int(value) for value in spec.ranks or ()), freeze_router=True
        )
    elif spec.family == "residual":
        student, initialization = ResidualScalarModalMoE.from_conventional_svd(
            teacher_moe,
            int(spec.rank),
            int(spec.residual_rank),
            freeze_router=True,
        )
    elif spec.family == "clustered":
        student, initialization = ClusteredResidualMoE.from_conventional_grouped(
            teacher_moe,
            int(spec.n_groups),
            int(spec.residual_rank),
            calibration_inputs=calibration_inputs,
            freeze_router=True,
        )
    elif spec.family == "narrow":
        student = make_narrow_student(
            teacher_moe, d_ff=int(spec.narrow_d_ff)
        )
        initialization = {
            "method": "teacher-neuron-importance-topk",
            "narrow_d_ff": int(spec.narrow_d_ff),
        }
    else:  # pragma: no cover - guarded by validate
        raise ValueError(spec.family)
    return student, initialization


def _selected_teacher_states(
    teacher: ConventionalSwiGLUMoE,
    inputs: torch.Tensor,
    top_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    geometry = teacher.geometry
    selected_gate = teacher.gate.index_select(0, top_ids.reshape(-1)).reshape(
        inputs.shape[0], geometry.top_k, geometry.d_ff, geometry.d_model
    )
    selected_up = teacher.up.index_select(0, top_ids.reshape(-1)).reshape(
        inputs.shape[0], geometry.top_k, geometry.d_ff, geometry.d_model
    )
    selected_down = teacher.down.index_select(0, top_ids.reshape(-1)).reshape(
        inputs.shape[0], geometry.top_k, geometry.d_model, geometry.d_ff
    )
    gate = torch.einsum("ntfd,nd->ntf", selected_gate, inputs)
    up = torch.einsum("ntfd,nd->ntf", selected_up, inputs)
    hidden = F.silu(gate) * up
    expert_output = torch.einsum("ntdf,ntf->ntd", selected_down, hidden)
    return gate, up, hidden, expert_output


def _selected_student_states(
    student: nn.Module,
    inputs: torch.Tensor,
    top_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    reconstruct = getattr(student, "reconstruct_weights", None)
    if not callable(reconstruct):
        raise TypeError("factor-resolved distillation requires reconstruct_weights")
    gate_weights, up_weights, down_weights = reconstruct()
    geometry = student.geometry
    selected_gate = gate_weights.index_select(0, top_ids.reshape(-1)).reshape(
        inputs.shape[0], geometry.top_k, geometry.d_ff, geometry.d_model
    )
    selected_up = up_weights.index_select(0, top_ids.reshape(-1)).reshape(
        inputs.shape[0], geometry.top_k, geometry.d_ff, geometry.d_model
    )
    selected_down = down_weights.index_select(0, top_ids.reshape(-1)).reshape(
        inputs.shape[0], geometry.top_k, geometry.d_model, geometry.d_ff
    )
    gate = torch.einsum("ntfd,nd->ntf", selected_gate, inputs)
    up = torch.einsum("ntfd,nd->ntf", selected_up, inputs)
    hidden = F.silu(gate) * up
    expert_output = torch.einsum("ntdf,ntf->ntd", selected_down, hidden)
    return gate, up, hidden, expert_output


def _normalized_mse(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    scale = torch.mean(target.square()).detach().clamp_min(1e-8)
    return torch.mean((prediction - target).square()) / scale


def distill_modal_student_resolved(
    student: nn.Module,
    teacher_moe: ConventionalSwiGLUMoE,
    captured: CapturedLayerDataset,
    *,
    objective: Objective,
    steps: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
) -> list[dict[str, float]]:
    """Distill aggregate output with optional expert/factor supervision.

    The deployment graph remains fused.  Reconstructed selected-expert weights
    are used only to define richer training losses.  This tests whether the
    aggregate-only objective was underdetermined rather than changing inference
    arithmetic.
    """
    if objective == "aggregate":
        return distill_layer_student(
            student,
            captured,
            steps=steps,
            batch_size=batch_size,
            learning_rate=learning_rate,
            seed=seed,
        )
    if objective not in {"expert-output", "factor-resolved"}:
        raise ValueError(objective)
    parameters = [p for p in student.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=learning_rate, weight_decay=0.0)
    generator = torch.Generator().manual_seed(seed)
    history: list[dict[str, float]] = []
    student.train()
    teacher_moe.eval()
    for step in range(1, steps + 1):
        indices = torch.randint(
            0,
            len(captured.inputs),
            (min(batch_size, len(captured.inputs)),),
            generator=generator,
        )
        inputs = captured.inputs.index_select(0, indices)
        targets = captured.outputs.index_select(0, indices)
        top_ids = captured.top_ids.index_select(0, indices)
        weights = captured.route_weights.index_select(0, indices)
        prediction, _ = student(
            inputs, forced_top_ids=top_ids, forced_weights=weights
        )
        aggregate_loss = _normalized_mse(prediction, targets)
        cosine = 1.0 - F.cosine_similarity(
            prediction, targets, dim=-1
        ).mean()
        with torch.no_grad():
            teacher_gate, teacher_up, teacher_hidden, teacher_expert = (
                _selected_teacher_states(teacher_moe, inputs, top_ids)
            )
        student_gate, student_up, student_hidden, student_expert = (
            _selected_student_states(student, inputs, top_ids)
        )
        expert_loss = _normalized_mse(student_expert, teacher_expert)
        gate_loss = _normalized_mse(student_gate, teacher_gate)
        up_loss = _normalized_mse(student_up, teacher_up)
        hidden_loss = _normalized_mse(student_hidden, teacher_hidden)
        if objective == "expert-output":
            loss = aggregate_loss + 0.1 * cosine + 0.50 * expert_loss
        else:
            loss = (
                aggregate_loss
                + 0.1 * cosine
                + 0.10 * gate_loss
                + 0.10 * up_loss
                + 0.15 * hidden_loss
                + 0.25 * expert_loss
            )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, 1.0)
        optimizer.step()
        if step == 1 or step == steps or step % max(steps // 5, 1) == 0:
            history.append(
                {
                    "step": float(step),
                    "objective": float(loss.detach()),
                    "aggregate_normalized_mse": float(aggregate_loss.detach()),
                    "cosine_penalty": float(cosine.detach()),
                    "expert_output_normalized_mse": float(expert_loss.detach()),
                    "gate_normalized_mse": float(gate_loss.detach()),
                    "up_normalized_mse": float(up_loss.detach()),
                    "hidden_normalized_mse": float(hidden_loss.detach()),
                }
            )
    return history


def dominant_matrix_compute_ratio(
    student: nn.Module,
    *,
    full_d_ff: int,
) -> float:
    method = getattr(student, "dominant_matrix_compute_ratio", None)
    if callable(method):
        return float(method())
    if isinstance(student, AsymmetricScalarModalMoE):
        return student.dominant_matrix_compute_ratio()
    if isinstance(student, (ScalarModalMoE, NeuronwiseModalMoE)):
        return student.n_modes / student.geometry.top_k
    if isinstance(student, ConventionalSwiGLUMoE):
        return student.geometry.d_ff / full_d_ff
    raise TypeError(type(student))


def code_adjusted_compute_ratio(
    student: nn.Module,
    *,
    full_d_ff: int,
) -> float:
    method = getattr(student, "idealized_expert_compute_ratio", None)
    if callable(method):
        return float(method())
    return dominant_matrix_compute_ratio(student, full_d_ff=full_d_ff)



def evaluate_local_student_batched(
    teacher_model: TinyMoELanguageModel,
    student: nn.Module,
    corpus: CharacterCorpus,
    *,
    split: str,
    layer_id: int,
    windows_per_document: int,
    seed: int,
    evaluation_batch_size: int = 16,
) -> tuple[dict[str, float], list[dict[str, object]]]:
    windows = corpus.fixed_windows(
        split, windows_per_document=windows_per_document, seed=seed
    )
    teacher_model.eval(); student.eval()
    predictions: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    records: list[dict[str, object]] = []
    with torch.no_grad():
        for offset in range(0, len(windows), evaluation_batch_size):
            batch = windows[offset : offset + evaluation_batch_size]
            tokens = torch.stack([window.inputs for window in batch])
            _, _, capture = teacher_model(tokens, collect_layer=layer_id)
            if capture is None:
                raise AssertionError("missing layer capture")
            prediction, _ = student(
                capture.moe_input,
                forced_top_ids=capture.routing.top_ids,
                forced_weights=capture.routing.weights,
            )
            target = capture.moe_output
            length = tokens.shape[1]
            for index, window in enumerate(batch):
                left, right = index * length, (index + 1) * length
                metric = tensor_metrics(prediction[left:right], target[left:right])
                records.append({
                    "document_id": window.document_id,
                    "domain": window.domain,
                    "start": window.start,
                    "nrmse": metric.nrmse,
                })
            predictions.append(prediction)
            targets.append(target)
    return tensor_metrics(torch.cat(predictions), torch.cat(targets)).as_dict(), records


def evaluate_closed_loop_batched(
    teacher: TinyMoELanguageModel,
    candidate: TinyMoELanguageModel,
    corpus: CharacterCorpus,
    *,
    split: str,
    windows_per_document: int,
    seed: int,
    evaluation_batch_size: int = 16,
) -> tuple[dict[str, float], list[dict[str, object]]]:
    windows = corpus.fixed_windows(
        split, windows_per_document=windows_per_document, seed=seed
    )
    teacher.eval(); candidate.eval()
    teacher_losses: list[float] = []
    candidate_losses: list[float] = []
    kls: list[float] = []
    agreements: list[float] = []
    records: list[dict[str, object]] = []
    with torch.no_grad():
        for offset in range(0, len(windows), evaluation_batch_size):
            batch = windows[offset : offset + evaluation_batch_size]
            tokens = torch.stack([window.inputs for window in batch])
            targets = torch.stack([window.targets for window in batch])
            teacher_logits, _, _ = teacher(tokens)
            candidate_logits, _, _ = candidate(tokens)
            teacher_token = F.cross_entropy(
                teacher_logits.reshape(-1, teacher_logits.shape[-1]),
                targets.reshape(-1), reduction="none"
            ).reshape(len(batch), -1)
            candidate_token = F.cross_entropy(
                candidate_logits.reshape(-1, candidate_logits.shape[-1]),
                targets.reshape(-1), reduction="none"
            ).reshape(len(batch), -1)
            kl = paired_kl_from_logits(
                candidate_logits.reshape(-1, candidate_logits.shape[-1]),
                teacher_logits.reshape(-1, teacher_logits.shape[-1]),
            ).reshape(len(batch), -1)
            agreement = (candidate_logits.argmax(-1) == teacher_logits.argmax(-1)).float()
            for index, window in enumerate(batch):
                teacher_mean = float(teacher_token[index].mean())
                candidate_mean = float(candidate_token[index].mean())
                kl_mean = float(kl[index].mean())
                agreement_mean = float(agreement[index].mean())
                teacher_losses.append(teacher_mean)
                candidate_losses.append(candidate_mean)
                kls.append(kl_mean)
                agreements.append(agreement_mean)
                records.append({
                    "document_id": window.document_id,
                    "domain": window.domain,
                    "start": window.start,
                    "teacher_loss": teacher_mean,
                    "candidate_loss": candidate_mean,
                    "loss_delta": candidate_mean - teacher_mean,
                    "kl_teacher_to_candidate": kl_mean,
                    "top1_agreement": agreement_mean,
                })
    teacher_loss = float(sum(teacher_losses) / len(teacher_losses))
    candidate_loss = float(sum(candidate_losses) / len(candidate_losses))
    return {
        "teacher_loss": teacher_loss,
        "candidate_loss": candidate_loss,
        "loss_delta": candidate_loss - teacher_loss,
        "teacher_perplexity": math.exp(min(teacher_loss, 20.0)),
        "candidate_perplexity": math.exp(min(candidate_loss, 20.0)),
        "perplexity_ratio": math.exp(min(candidate_loss - teacher_loss, 20.0)),
        "kl_teacher_to_candidate": float(sum(kls) / len(kls)),
        "top1_agreement": float(sum(agreements) / len(agreements)),
        "windows": len(windows),
    }, records

def run_candidate(
    teacher: TinyMoELanguageModel,
    teacher_moe: ConventionalSwiGLUMoE,
    captured: CapturedLayerDataset,
    corpus: CharacterCorpus,
    spec: CandidateSpec,
    *,
    layer_id: int,
    evaluation_split: str,
    evaluation_windows_per_document: int,
    evaluation_seed: int,
    local_batch_size: int,
    local_learning_rate: float,
    joint_learning_rate: float,
    base_seed: int,
) -> tuple[dict[str, Any], list[dict[str, object]]]:
    student, initialization = build_candidate(teacher_moe, spec)
    local_history = distill_modal_student_resolved(
        student,
        teacher_moe,
        captured,
        objective=spec.objective,
        steps=spec.local_steps,
        batch_size=local_batch_size,
        learning_rate=local_learning_rate,
        seed=stable_seed(base_seed, spec.name, "local"),
    )
    joint_history: list[dict[str, float]] = []
    if spec.joint_steps:
        student, joint_history = joint_fine_tune_transplant(
            teacher,
            student,
            corpus,
            layer_id=layer_id,
            steps=spec.joint_steps,
            batch_size=teacher.config.batch_size,
            learning_rate=joint_learning_rate,
            seed=stable_seed(base_seed, spec.name, "joint"),
        )
    local_metrics, local_records = evaluate_local_student_batched(
        teacher, student, corpus, split=evaluation_split, layer_id=layer_id,
        windows_per_document=evaluation_windows_per_document, seed=evaluation_seed
    )
    candidate_model = install_student(teacher, student, layer_id=layer_id)
    closed_metrics, records = evaluate_closed_loop_batched(
        teacher, candidate_model, corpus, split=evaluation_split,
        windows_per_document=evaluation_windows_per_document, seed=evaluation_seed
    )
    full_parameters = expert_parameter_count(teacher_moe)
    for row in records:
        row["candidate"] = spec.name
        row["family"] = spec.family
    result = {
        "candidate": spec.name,
        "spec": asdict(spec),
        "initialization": initialization,
        "local_metrics": local_metrics,
        "closed_loop": closed_metrics,
        "expert_parameters": expert_parameter_count(student),
        "expert_parameter_ratio": expert_parameter_count(student) / full_parameters,
        "dominant_matrix_compute_ratio": dominant_matrix_compute_ratio(
            student, full_d_ff=teacher_moe.geometry.d_ff
        ),
        "code_adjusted_compute_ratio": code_adjusted_compute_ratio(
            student, full_d_ff=teacher_moe.geometry.d_ff
        ),
        "local_history": local_history,
        "joint_history": joint_history,
        "local_records": local_records,
    }
    return result, records


def pareto_front(
    summaries: Sequence[dict[str, Any]],
    *,
    loss_key: str = "loss_delta_mean",
) -> list[str]:
    """Return candidates not dominated in parameters, compute, and loss."""
    frontier: list[str] = []
    for candidate in summaries:
        dominated = False
        for other in summaries:
            if other is candidate:
                continue
            no_worse = (
                float(other["expert_parameter_ratio"])
                <= float(candidate["expert_parameter_ratio"]) + 1e-12
                and float(other["code_adjusted_compute_ratio"])
                <= float(candidate["code_adjusted_compute_ratio"]) + 1e-12
                and float(other[loss_key]) <= float(candidate[loss_key]) + 1e-12
            )
            strictly_better = (
                float(other["expert_parameter_ratio"])
                < float(candidate["expert_parameter_ratio"]) - 1e-12
                or float(other["code_adjusted_compute_ratio"])
                < float(candidate["code_adjusted_compute_ratio"]) - 1e-12
                or float(other[loss_key]) < float(candidate[loss_key]) - 1e-12
            )
            if no_worse and strictly_better:
                dominated = True
                break
        if not dominated:
            frontier.append(str(candidate["candidate"]))
    return sorted(frontier)


def paired_candidate_records(
    left_records: Sequence[dict[str, object]],
    right_records: Sequence[dict[str, object]],
    *,
    value_key: str = "loss_delta",
    output_key: str = "left_minus_right",
) -> list[dict[str, object]]:
    index = {
        (int(row["seed"]), str(row["document_id"]), int(row["start"])): row
        for row in right_records
    }
    result: list[dict[str, object]] = []
    for row in left_records:
        key = (int(row["seed"]), str(row["document_id"]), int(row["start"]))
        other = index[key]
        result.append(
            {
                "seed": key[0],
                "document_id": key[1],
                "start": key[2],
                output_key: float(row[value_key]) - float(other[value_key]),
            }
        )
    return result
