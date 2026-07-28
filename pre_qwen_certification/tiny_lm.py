"""Controlled conventional-MoE language model for the pre-real-checkpoint gate.

The model is intentionally small enough for a CPU CI runner, but the transplant
is closed-loop: a conventional MoE is trained first, a Modal student sees only
captured layer inputs/outputs and frozen teacher routes, and the student is then
installed back into the language model for held-out evaluation.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import copy
import math
import random
from typing import Any, Iterable, Sequence

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from .data import Document
from .metrics import paired_kl_from_logits, tensor_metrics
from .modal import ConventionalSwiGLUMoE, MoEGeometry, Routing, ScalarModalMoE, set_seed


@dataclass(frozen=True)
class TinyLMConfig:
    seq_len: int = 64
    batch_size: int = 12
    d_model: int = 32
    n_heads: int = 4
    n_layers: int = 2
    d_ff: int = 64
    n_experts: int = 16
    top_k: int = 4
    teacher_steps: int = 500
    student_steps: int = 350
    learning_rate: float = 4e-4
    student_learning_rate: float = 8e-4
    weight_decay: float = 0.02
    aux_weight: float = 0.01
    grad_clip: float = 1.0

    @property
    def geometry(self) -> MoEGeometry:
        return MoEGeometry(
            self.d_model, self.d_ff, self.n_experts, self.top_k
        )


@dataclass
class LayerCapture:
    moe_input: torch.Tensor
    moe_output: torch.Tensor
    routing: Routing


@dataclass
class EvaluationWindow:
    inputs: torch.Tensor
    targets: torch.Tensor
    document_id: str
    domain: str
    start: int


class CharacterCorpus:
    def __init__(
        self,
        splits: dict[str, Sequence[Document]],
        seq_len: int,
        *,
        vocabulary: Sequence[str] | None = None,
    ) -> None:
        self.seq_len = int(seq_len)
        characters = list(vocabulary) if vocabulary is not None else sorted(
            {
                character
                for documents in splits.values()
                for document in documents
                for character in document.text
            }
        )
        if len(characters) < 8:
            raise ValueError("corpus vocabulary is unexpectedly small")
        self.itos = characters
        self.stoi = {character: index for index, character in enumerate(characters)}
        self.splits: dict[str, list[tuple[Document, torch.Tensor]]] = {}
        for split, documents in splits.items():
            encoded: list[tuple[Document, torch.Tensor]] = []
            for document in documents:
                unknown = sorted(set(document.text) - self.stoi.keys())
                if unknown:
                    raise ValueError(
                        f"document {document.document_id} contains characters outside the frozen vocabulary: {unknown}"
                    )
                values = torch.tensor(
                    [self.stoi[character] for character in document.text],
                    dtype=torch.long,
                )
                if len(values) <= self.seq_len + 1:
                    raise ValueError(
                        f"document {document.document_id} is too short for seq_len={self.seq_len}"
                    )
                encoded.append((document, values))
            self.splits[split] = encoded

    @property
    def vocab_size(self) -> int:
        return len(self.itos)

    def sample_batch(
        self,
        split: str,
        batch_size: int,
        generator: torch.Generator,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        documents = self.splits[split]
        selected_documents = torch.randint(
            0, len(documents), (batch_size,), generator=generator
        )
        inputs: list[torch.Tensor] = []
        targets: list[torch.Tensor] = []
        for document_index in selected_documents.tolist():
            _, data = documents[document_index]
            start = int(
                torch.randint(
                    0,
                    len(data) - self.seq_len - 1,
                    (1,),
                    generator=generator,
                )
            )
            inputs.append(data[start : start + self.seq_len])
            targets.append(data[start + 1 : start + self.seq_len + 1])
        return torch.stack(inputs), torch.stack(targets)

    def fixed_windows(
        self,
        split: str,
        *,
        windows_per_document: int,
        seed: int,
    ) -> list[EvaluationWindow]:
        if windows_per_document <= 0:
            raise ValueError("windows_per_document must be positive")
        result: list[EvaluationWindow] = []
        generator = torch.Generator().manual_seed(seed)
        for document, data in self.splits[split]:
            available = len(data) - self.seq_len - 1
            if available <= 0:
                continue
            if windows_per_document == 1:
                starts = [available // 2]
            else:
                # Stratify positions, then jitter within each stratum. Windows may
                # touch but never cross document boundaries.
                edges = np.linspace(0, available, windows_per_document + 1, dtype=int)
                starts = []
                for left, right in zip(edges[:-1], edges[1:], strict=True):
                    upper = max(left + 1, right)
                    starts.append(
                        int(torch.randint(left, upper, (1,), generator=generator))
                    )
            for start in starts:
                result.append(
                    EvaluationWindow(
                        inputs=data[start : start + self.seq_len],
                        targets=data[start + 1 : start + self.seq_len + 1],
                        document_id=document.document_id,
                        domain=document.domain,
                        start=start,
                    )
                )
        return result


class TinyMoEBlock(nn.Module):
    def __init__(self, config: TinyLMConfig, moe: nn.Module | None = None) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(config.d_model)
        self.attention = nn.MultiheadAttention(
            config.d_model,
            config.n_heads,
            batch_first=True,
            dropout=0.0,
        )
        self.norm2 = nn.LayerNorm(config.d_model)
        self.moe = moe or ConventionalSwiGLUMoE(config.geometry)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        *,
        collect: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, LayerCapture | None]:
        normalized = self.norm1(x)
        attended, _ = self.attention(
            normalized,
            normalized,
            normalized,
            attn_mask=mask,
            need_weights=False,
        )
        x = x + attended
        moe_input = self.norm2(x)
        flat_input = moe_input.reshape(-1, moe_input.shape[-1])
        flat_output, routing = self.moe(flat_input)
        moe_output = flat_output.reshape_as(x)
        capture = None
        if collect:
            capture = LayerCapture(
                moe_input=flat_input,
                moe_output=flat_output,
                routing=routing,
            )
        probabilities = F.softmax(routing.logits, dim=-1)
        importance = probabilities.mean(dim=0)
        assignments = F.one_hot(
            routing.top_ids, probabilities.shape[-1]
        ).float().mean(dim=(0, 1))
        balance = probabilities.shape[-1] * torch.sum(importance * assignments)
        z_loss = torch.mean(torch.logsumexp(routing.logits, dim=-1).square())
        return x + moe_output, balance + 0.1 * z_loss, capture


class TinyMoELanguageModel(nn.Module):
    def __init__(self, vocab_size: int, config: TinyLMConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(vocab_size, config.d_model)
        self.position_embedding = nn.Embedding(config.seq_len, config.d_model)
        self.blocks = nn.ModuleList(
            [TinyMoEBlock(config) for _ in range(config.n_layers)]
        )
        self.norm = nn.LayerNorm(config.d_model)
        self.output = nn.Linear(config.d_model, vocab_size, bias=False)
        self.output.weight = self.token_embedding.weight
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.position_embedding.weight, std=0.02)

    def forward(
        self,
        tokens: torch.Tensor,
        *,
        collect_layer: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, LayerCapture | None]:
        if tokens.ndim != 2:
            raise ValueError("tokens must have shape [batch, sequence]")
        positions = torch.arange(tokens.shape[1], device=tokens.device)
        x = self.token_embedding(tokens) + self.position_embedding(positions)[None, :, :]
        mask = torch.full(
            (tokens.shape[1], tokens.shape[1]),
            float("-inf"),
            device=tokens.device,
        )
        mask = torch.triu(mask, diagonal=1)
        aux_values: list[torch.Tensor] = []
        selected_capture: LayerCapture | None = None
        for layer_index, block in enumerate(self.blocks):
            x, aux, capture = block(
                x, mask, collect=collect_layer == layer_index
            )
            aux_values.append(aux)
            if capture is not None:
                selected_capture = capture
        logits = self.output(self.norm(x))
        return logits, torch.stack(aux_values).mean(), selected_capture


def generate_multidomain_documents(
    *,
    split: str,
    documents: int,
    seed: int,
) -> list[Document]:
    rng = random.Random(seed)
    domains = ("general", "code", "math", "portuguese")
    result: list[Document] = []
    for document_index in range(documents):
        domain = domains[document_index % len(domains)]
        blocks: list[str] = []
        for paragraph in range(28):
            a = rng.randint(2, 999)
            b = rng.randint(2, 999)
            c = rng.randint(2, 999)
            name = f"item_{split}_{document_index}_{paragraph}"
            if domain == "code":
                blocks.append(
                    f"def {name}(x):\n    total = x * {a} + {b}\n    return total % {max(c, 3)}\n"
                    f"// java check: int value = ({a} * input + {b}) % {max(c, 3)};\n"
                )
            elif domain == "math":
                blocks.append(
                    f"Problem {paragraph}: compute ({a} + {b}) mod {max(c, 3)}. "
                    f"Identity: {a}*({b}+{c}) = {a*b}+{a*c}. "
                    f"Sequence {name}: {a}, {a+b}, {a+2*b}, {a+3*b}.\n"
                )
            elif domain == "portuguese":
                blocks.append(
                    f"O registro {name} contém os números {a}, {b} e {c}. "
                    f"Para validar o relatório, some {a} com {b} e compare com {c}. "
                    "A resposta deve preservar contexto, ordem e precisão.\n"
                )
            else:
                blocks.append(
                    f"Record {name} links key K{a} to value V{b} under group G{c}. "
                    f"When asked for K{a}, return V{b}; the checksum is {a+b+c}. "
                    "The surrounding sentence is deliberately repetitive but causal.\n"
                )
        text = "\n".join(blocks)
        result.append(
            Document(
                document_id=f"{split}-doc-{document_index:04d}",
                text=text,
                source="deterministic-multidomain-generator-v1",
                domain=domain,
            )
        )
    return result


def train_teacher(
    corpus: CharacterCorpus,
    config: TinyLMConfig,
    *,
    seed: int,
) -> tuple[TinyMoELanguageModel, list[dict[str, float]]]:
    set_seed(seed)
    model = TinyMoELanguageModel(corpus.vocab_size, config)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    generator = torch.Generator().manual_seed(seed + 101)
    history: list[dict[str, float]] = []
    model.train()
    for step in range(1, config.teacher_steps + 1):
        tokens, targets = corpus.sample_batch(
            "train", config.batch_size, generator
        )
        logits, aux, _ = model(tokens)
        language_loss = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), targets.reshape(-1)
        )
        loss = language_loss + config.aux_weight * aux
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        optimizer.step()
        if step == 1 or step == config.teacher_steps or step % max(config.teacher_steps // 5, 1) == 0:
            history.append(
                {
                    "step": float(step),
                    "language_loss": float(language_loss.detach()),
                    "auxiliary_loss": float(aux.detach()),
                }
            )
    return model, history


@dataclass
class CapturedLayerDataset:
    inputs: torch.Tensor
    outputs: torch.Tensor
    top_ids: torch.Tensor
    route_weights: torch.Tensor


def capture_training_layer(
    model: TinyMoELanguageModel,
    corpus: CharacterCorpus,
    *,
    split: str,
    layer_id: int,
    batches: int,
    batch_size: int,
    seed: int,
) -> CapturedLayerDataset:
    model.eval()
    generator = torch.Generator().manual_seed(seed)
    inputs: list[torch.Tensor] = []
    outputs: list[torch.Tensor] = []
    top_ids: list[torch.Tensor] = []
    route_weights: list[torch.Tensor] = []
    with torch.no_grad():
        for _ in range(batches):
            tokens, _ = corpus.sample_batch(split, batch_size, generator)
            _, _, capture = model(tokens, collect_layer=layer_id)
            if capture is None:
                raise AssertionError("requested layer capture was not returned")
            inputs.append(capture.moe_input.cpu())
            outputs.append(capture.moe_output.cpu())
            top_ids.append(capture.routing.top_ids.cpu())
            route_weights.append(capture.routing.weights.cpu())
    return CapturedLayerDataset(
        inputs=torch.cat(inputs),
        outputs=torch.cat(outputs),
        top_ids=torch.cat(top_ids),
        route_weights=torch.cat(route_weights),
    )


def make_narrow_student(
    teacher: ConventionalSwiGLUMoE,
    *,
    d_ff: int,
) -> ConventionalSwiGLUMoE:
    if not 1 <= d_ff <= teacher.geometry.d_ff:
        raise ValueError("narrow d_ff must be between 1 and teacher d_ff")
    geometry = MoEGeometry(
        teacher.geometry.d_model,
        d_ff,
        teacher.geometry.n_experts,
        teacher.geometry.top_k,
    )
    student = ConventionalSwiGLUMoE(geometry)
    with torch.no_grad():
        student.router.weight.copy_(teacher.router.weight)
        for expert in range(teacher.geometry.n_experts):
            importance = (
                teacher.gate[expert].square().sum(dim=-1)
                + teacher.up[expert].square().sum(dim=-1)
                + teacher.down[expert].square().sum(dim=0)
            )
            chosen = torch.topk(importance, k=d_ff, sorted=True).indices
            student.gate[expert].copy_(teacher.gate[expert].index_select(0, chosen))
            student.up[expert].copy_(teacher.up[expert].index_select(0, chosen))
            student.down[expert].copy_(teacher.down[expert].index_select(1, chosen))
    student.router.weight.requires_grad_(False)
    return student


def distill_layer_student(
    student: nn.Module,
    captured: CapturedLayerDataset,
    *,
    steps: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
) -> list[dict[str, float]]:
    parameters = [parameter for parameter in student.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=learning_rate, weight_decay=0.0)
    generator = torch.Generator().manual_seed(seed)
    history: list[dict[str, float]] = []
    student.train()
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
        scale = torch.mean(targets.square()).detach().clamp_min(1e-8)
        normalized_mse = torch.mean((prediction - targets).square()) / scale
        cosine = 1.0 - F.cosine_similarity(prediction, targets, dim=-1).mean()
        loss = normalized_mse + 0.1 * cosine
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, 1.0)
        optimizer.step()
        if step == 1 or step == steps or step % max(steps // 5, 1) == 0:
            history.append(
                {
                    "step": float(step),
                    "objective": float(loss.detach()),
                    "normalized_mse": float(normalized_mse.detach()),
                    "cosine_penalty": float(cosine.detach()),
                }
            )
    return history


def evaluate_local_student(
    teacher_model: TinyMoELanguageModel,
    student: nn.Module,
    corpus: CharacterCorpus,
    *,
    split: str,
    layer_id: int,
    windows_per_document: int,
    seed: int,
) -> tuple[dict[str, float], list[dict[str, object]]]:
    windows = corpus.fixed_windows(
        split, windows_per_document=windows_per_document, seed=seed
    )
    teacher_model.eval()
    predictions: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    records: list[dict[str, object]] = []
    with torch.no_grad():
        for window in windows:
            _, _, capture = teacher_model(
                window.inputs[None, :], collect_layer=layer_id
            )
            if capture is None:
                raise AssertionError("missing layer capture")
            prediction, _ = student(
                capture.moe_input,
                forced_top_ids=capture.routing.top_ids,
                forced_weights=capture.routing.weights,
            )
            metric = tensor_metrics(prediction, capture.moe_output)
            records.append(
                {
                    "document_id": window.document_id,
                    "domain": window.domain,
                    "start": window.start,
                    "nrmse": metric.nrmse,
                }
            )
            predictions.append(prediction)
            targets.append(capture.moe_output)
    return tensor_metrics(torch.cat(predictions), torch.cat(targets)).as_dict(), records


def install_student(
    teacher_model: TinyMoELanguageModel,
    student: nn.Module,
    *,
    layer_id: int,
) -> TinyMoELanguageModel:
    model = copy.deepcopy(teacher_model)
    model.blocks[layer_id].moe = copy.deepcopy(student)
    return model


def evaluate_closed_loop(
    teacher: TinyMoELanguageModel,
    candidate: TinyMoELanguageModel,
    corpus: CharacterCorpus,
    *,
    split: str,
    windows_per_document: int,
    seed: int,
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
        for window in windows:
            tokens = window.inputs[None, :]
            targets = window.targets[None, :]
            teacher_logits, _, _ = teacher(tokens)
            candidate_logits, _, _ = candidate(tokens)
            teacher_token_loss = F.cross_entropy(
                teacher_logits.reshape(-1, teacher_logits.shape[-1]),
                targets.reshape(-1),
                reduction="none",
            )
            candidate_token_loss = F.cross_entropy(
                candidate_logits.reshape(-1, candidate_logits.shape[-1]),
                targets.reshape(-1),
                reduction="none",
            )
            kl = paired_kl_from_logits(
                candidate_logits.reshape(-1, candidate_logits.shape[-1]),
                teacher_logits.reshape(-1, teacher_logits.shape[-1]),
            )
            agreement = (
                candidate_logits.argmax(dim=-1) == teacher_logits.argmax(dim=-1)
            ).float()
            teacher_mean = float(teacher_token_loss.mean())
            candidate_mean = float(candidate_token_loss.mean())
            kl_mean = float(kl.mean())
            agreement_mean = float(agreement.mean())
            teacher_losses.append(teacher_mean)
            candidate_losses.append(candidate_mean)
            kls.append(kl_mean)
            agreements.append(agreement_mean)
            records.append(
                {
                    "document_id": window.document_id,
                    "domain": window.domain,
                    "start": window.start,
                    "teacher_loss": teacher_mean,
                    "candidate_loss": candidate_mean,
                    "loss_delta": candidate_mean - teacher_mean,
                    "kl_teacher_to_candidate": kl_mean,
                    "top1_agreement": agreement_mean,
                }
            )
    teacher_loss = float(np.mean(teacher_losses))
    candidate_loss = float(np.mean(candidate_losses))
    return {
        "teacher_loss": teacher_loss,
        "candidate_loss": candidate_loss,
        "loss_delta": candidate_loss - teacher_loss,
        "teacher_perplexity": math.exp(min(teacher_loss, 20.0)),
        "candidate_perplexity": math.exp(min(candidate_loss, 20.0)),
        "perplexity_ratio": math.exp(min(candidate_loss - teacher_loss, 20.0)),
        "kl_teacher_to_candidate": float(np.mean(kls)),
        "top1_agreement": float(np.mean(agreements)),
        "windows": len(windows),
    }, records


def expert_parameter_count(module: nn.Module) -> int:
    counter = getattr(module, "expert_transform_parameter_count", None)
    if callable(counter):
        return int(counter())
    if isinstance(module, ConventionalSwiGLUMoE):
        return module.gate.numel() + module.up.numel() + module.down.numel()
    return sum(parameter.numel() for parameter in module.parameters())


def joint_fine_tune_transplant(
    teacher: TinyMoELanguageModel,
    student: nn.Module,
    corpus: CharacterCorpus,
    *,
    layer_id: int,
    steps: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
    local_weight: float = 0.65,
    kl_weight: float = 0.30,
    ce_weight: float = 0.05,
) -> tuple[nn.Module, list[dict[str, float]]]:
    """Fine-tune only the transplanted layer through the frozen full model.

    The teacher and every non-student candidate parameter remain frozen.  The
    objective combines the captured layer function, final-logit KL, and a small
    token CE term.  This calibrates local fidelity against actual closed-loop harm.
    """
    if steps <= 0:
        return student, []
    if abs(local_weight + kl_weight + ce_weight - 1.0) > 1e-6:
        raise ValueError("joint fine-tuning weights must sum to one")
    teacher.eval()
    candidate = install_student(teacher, student, layer_id=layer_id)
    for parameter in candidate.parameters():
        parameter.requires_grad_(False)
    transplanted = candidate.blocks[layer_id].moe
    for parameter in transplanted.parameters():
        # The router remains frozen when it was frozen in the source student.
        if parameter is transplanted.router.weight:
            parameter.requires_grad_(False)
        else:
            parameter.requires_grad_(True)
    trainable = [parameter for parameter in transplanted.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=learning_rate, weight_decay=0.0)
    generator = torch.Generator().manual_seed(seed)
    history: list[dict[str, float]] = []
    candidate.train()
    for step in range(1, steps + 1):
        tokens, targets = corpus.sample_batch("train", batch_size, generator)
        with torch.no_grad():
            teacher_logits, _, teacher_capture = teacher(
                tokens, collect_layer=layer_id
            )
        candidate_logits, _, candidate_capture = candidate(
            tokens, collect_layer=layer_id
        )
        if teacher_capture is None or candidate_capture is None:
            raise AssertionError("joint fine-tuning capture missing")
        local_scale = torch.mean(teacher_capture.moe_output.square()).detach().clamp_min(1e-8)
        local_loss = torch.mean(
            (candidate_capture.moe_output - teacher_capture.moe_output).square()
        ) / local_scale
        teacher_log_prob = torch.log_softmax(teacher_logits.detach(), dim=-1)
        candidate_log_prob = torch.log_softmax(candidate_logits, dim=-1)
        teacher_prob = teacher_log_prob.exp()
        kl_loss = torch.sum(
            teacher_prob * (teacher_log_prob - candidate_log_prob), dim=-1
        ).mean()
        ce_loss = F.cross_entropy(
            candidate_logits.reshape(-1, candidate_logits.shape[-1]),
            targets.reshape(-1),
        )
        loss = local_weight * local_loss + kl_weight * kl_loss + ce_weight * ce_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()
        if step == 1 or step == steps or step % max(steps // 4, 1) == 0:
            history.append(
                {
                    "step": float(step),
                    "objective": float(loss.detach()),
                    "local_normalized_mse": float(local_loss.detach()),
                    "logit_kl": float(kl_loss.detach()),
                    "token_ce": float(ce_loss.detach()),
                }
            )
    return copy.deepcopy(transplanted), history


class OutputPerturbationMoE(nn.Module):
    """Deterministically inject output noise at a requested layer-level NRMSE."""

    def __init__(self, base: nn.Module, target_nrmse: float, seed: int) -> None:
        super().__init__()
        if target_nrmse < 0.0:
            raise ValueError("target_nrmse must be non-negative")
        self.base = copy.deepcopy(base)
        self.target_nrmse = float(target_nrmse)
        self.seed = int(seed)
        self._calls = 0

    def forward(self, x: torch.Tensor, **kwargs):
        output, routing = self.base(x, **kwargs)
        if self.target_nrmse == 0.0:
            return output, routing
        generator = torch.Generator(device=output.device).manual_seed(
            self.seed + self._calls
        )
        self._calls += 1
        noise = torch.randn(
            output.shape,
            dtype=output.dtype,
            device=output.device,
            generator=generator,
        )
        output_rms = torch.mean(output.square()).sqrt().detach()
        noise_rms = torch.mean(noise.square()).sqrt().detach().clamp_min(1e-12)
        noise = noise * (self.target_nrmse * output_rms / noise_rms)
        return output + noise, routing


def install_output_perturbation(
    teacher: TinyMoELanguageModel,
    *,
    layer_id: int,
    target_nrmse: float,
    seed: int,
) -> TinyMoELanguageModel:
    model = copy.deepcopy(teacher)
    model.blocks[layer_id].moe = OutputPerturbationMoE(
        teacher.blocks[layer_id].moe,
        target_nrmse=target_nrmse,
        seed=seed,
    )
    return model

class AnnealedBlendMoE(nn.Module):
    """Blend a frozen conventional teacher MoE into a trainable student.

    The wrapper is used only during recovery training.  The teacher routing is
    reused for the student so that the interpolation changes the expert
    function rather than the routing policy.  ``beta=1`` is the teacher and
    ``beta=0`` is the deployable student.
    """

    def __init__(self, teacher_moe: nn.Module, student: nn.Module) -> None:
        super().__init__()
        self.teacher_moe = copy.deepcopy(teacher_moe)
        self.student = student
        self.beta = 1.0
        for parameter in self.teacher_moe.parameters():
            parameter.requires_grad_(False)

    def forward(self, x: torch.Tensor, **kwargs):
        with torch.no_grad():
            teacher_output, routing = self.teacher_moe(x, **kwargs)
        student_output, _ = self.student(
            x,
            forced_top_ids=routing.top_ids,
            forced_weights=routing.weights,
        )
        beta = float(self.beta)
        return beta * teacher_output + (1.0 - beta) * student_output, routing


def joint_fine_tune_transplant_annealed(
    teacher: TinyMoELanguageModel,
    student: nn.Module,
    corpus: CharacterCorpus,
    *,
    layer_id: int,
    steps: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
    anneal_end_fraction: float = 0.5,
    local_weight: float = 0.65,
    kl_weight: float = 0.30,
    ce_weight: float = 0.05,
) -> tuple[nn.Module, list[dict[str, float]]]:
    """Recover a transplant while continuously annealing teacher contribution.

    This is an exploratory implementation of continuation/homotopy training:
    the full teacher expert path is gradually removed, ending with an ordinary
    deployable student.  The final evaluation never contains the teacher path.
    """
    if steps <= 0:
        return student, []
    if not 0.0 < anneal_end_fraction <= 1.0:
        raise ValueError("anneal_end_fraction must be in (0, 1]")
    if abs(local_weight + kl_weight + ce_weight - 1.0) > 1e-6:
        raise ValueError("joint fine-tuning weights must sum to one")

    teacher.eval()
    candidate = copy.deepcopy(teacher)
    wrapper = AnnealedBlendMoE(teacher.blocks[layer_id].moe, copy.deepcopy(student))
    candidate.blocks[layer_id].moe = wrapper
    for parameter in candidate.parameters():
        parameter.requires_grad_(False)
    for parameter in wrapper.student.parameters():
        if parameter is wrapper.student.router.weight:
            parameter.requires_grad_(False)
        else:
            parameter.requires_grad_(True)
    trainable = [p for p in wrapper.student.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=learning_rate, weight_decay=0.0)
    generator = torch.Generator().manual_seed(seed)
    history: list[dict[str, float]] = []
    candidate.train()
    anneal_steps = max(1, int(math.ceil(steps * anneal_end_fraction)))

    for step in range(1, steps + 1):
        wrapper.beta = max(1.0 - step / anneal_steps, 0.0)
        tokens, targets = corpus.sample_batch("train", batch_size, generator)
        with torch.no_grad():
            teacher_logits, _, teacher_capture = teacher(tokens, collect_layer=layer_id)
        candidate_logits, _, candidate_capture = candidate(tokens, collect_layer=layer_id)
        if teacher_capture is None or candidate_capture is None:
            raise AssertionError("annealed joint fine-tuning capture missing")
        local_scale = torch.mean(teacher_capture.moe_output.square()).detach().clamp_min(1e-8)
        local_loss = torch.mean(
            (candidate_capture.moe_output - teacher_capture.moe_output).square()
        ) / local_scale
        teacher_log_prob = torch.log_softmax(teacher_logits.detach(), dim=-1)
        candidate_log_prob = torch.log_softmax(candidate_logits, dim=-1)
        teacher_prob = teacher_log_prob.exp()
        kl_loss = torch.sum(
            teacher_prob * (teacher_log_prob - candidate_log_prob), dim=-1
        ).mean()
        ce_loss = F.cross_entropy(
            candidate_logits.reshape(-1, candidate_logits.shape[-1]),
            targets.reshape(-1),
        )
        loss = local_weight * local_loss + kl_weight * kl_loss + ce_weight * ce_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()
        if step == 1 or step == steps or step % max(steps // 4, 1) == 0:
            history.append(
                {
                    "step": float(step),
                    "beta": float(wrapper.beta),
                    "objective": float(loss.detach()),
                    "local_normalized_mse": float(local_loss.detach()),
                    "logit_kl": float(kl_loss.detach()),
                    "token_ce": float(ce_loss.detach()),
                }
            )
    wrapper.beta = 0.0
    return copy.deepcopy(wrapper.student), history


def joint_fine_tune_transplant_with_scope(
    teacher: TinyMoELanguageModel,
    student: nn.Module,
    corpus: CharacterCorpus,
    *,
    layer_id: int,
    steps: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
    recovery_scope: str,
    local_weight: float = 0.0,
    kl_weight: float = 0.9,
    ce_weight: float = 0.1,
) -> tuple[nn.Module, TinyMoELanguageModel, list[dict[str, float]], dict[str, int]]:
    """Recover a compressed layer with progressively broader trainable context.

    Supported scopes:
    - ``student-only``: only the compressed MoE;
    - ``next-block``: student plus the immediately following Transformer block;
    - ``all-transformer``: student, all Transformer blocks, and final norm, while
      token/position embeddings and all routers remain frozen.

    No parameters are added by widening the recovery scope.  The returned model
    contains ordinary updated weights and the deployable student, not a teacher
    blend.  This tests whether a locally non-isometric representation can be
    absorbed by neighboring layers.
    """
    valid_scopes = {"student-only", "next-block", "all-transformer"}
    if recovery_scope not in valid_scopes:
        raise ValueError(f"unknown recovery_scope: {recovery_scope}")
    if steps <= 0:
        candidate = install_student(teacher, student, layer_id=layer_id)
        return copy.deepcopy(candidate.blocks[layer_id].moe), candidate, [], {
            "trainable_parameters": 0,
            "total_parameters": sum(p.numel() for p in candidate.parameters()),
        }
    if abs(local_weight + kl_weight + ce_weight - 1.0) > 1e-6:
        raise ValueError("joint fine-tuning weights must sum to one")

    teacher.eval()
    candidate = install_student(teacher, student, layer_id=layer_id)
    for parameter in candidate.parameters():
        parameter.requires_grad_(False)

    transplanted = candidate.blocks[layer_id].moe
    for parameter in transplanted.parameters():
        parameter.requires_grad_(parameter is not transplanted.router.weight)

    if recovery_scope == "next-block":
        if layer_id + 1 >= len(candidate.blocks):
            raise ValueError("next-block scope requires a following block")
        for parameter in candidate.blocks[layer_id + 1].parameters():
            parameter.requires_grad_(True)
        candidate.blocks[layer_id + 1].moe.router.weight.requires_grad_(False)
    elif recovery_scope == "all-transformer":
        for block in candidate.blocks:
            for parameter in block.parameters():
                parameter.requires_grad_(True)
            block.moe.router.weight.requires_grad_(False)
        for parameter in candidate.norm.parameters():
            parameter.requires_grad_(True)
        # Preserve embedding/head semantics and position encoding.
        candidate.token_embedding.weight.requires_grad_(False)
        candidate.position_embedding.weight.requires_grad_(False)

    trainable = [parameter for parameter in candidate.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=learning_rate, weight_decay=0.0)
    generator = torch.Generator().manual_seed(seed)
    history: list[dict[str, float]] = []
    candidate.train()
    for step in range(1, steps + 1):
        tokens, targets = corpus.sample_batch("train", batch_size, generator)
        with torch.no_grad():
            teacher_logits, _, teacher_capture = teacher(tokens, collect_layer=layer_id)
        candidate_logits, _, candidate_capture = candidate(tokens, collect_layer=layer_id)
        if teacher_capture is None or candidate_capture is None:
            raise AssertionError("scope recovery capture missing")
        local_scale = torch.mean(teacher_capture.moe_output.square()).detach().clamp_min(1e-8)
        local_loss = torch.mean(
            (candidate_capture.moe_output - teacher_capture.moe_output).square()
        ) / local_scale
        teacher_log_prob = torch.log_softmax(teacher_logits.detach(), dim=-1)
        candidate_log_prob = torch.log_softmax(candidate_logits, dim=-1)
        teacher_prob = teacher_log_prob.exp()
        kl_loss = torch.sum(
            teacher_prob * (teacher_log_prob - candidate_log_prob), dim=-1
        ).mean()
        ce_loss = F.cross_entropy(
            candidate_logits.reshape(-1, candidate_logits.shape[-1]),
            targets.reshape(-1),
        )
        loss = local_weight * local_loss + kl_weight * kl_loss + ce_weight * ce_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()
        if step == 1 or step == steps or step % max(steps // 4, 1) == 0:
            history.append(
                {
                    "step": float(step),
                    "objective": float(loss.detach()),
                    "local_normalized_mse": float(local_loss.detach()),
                    "logit_kl": float(kl_loss.detach()),
                    "token_ce": float(ce_loss.detach()),
                }
            )
    diagnostics = {
        "trainable_parameters": sum(p.numel() for p in trainable),
        "total_parameters": sum(p.numel() for p in candidate.parameters()),
    }
    return (
        copy.deepcopy(candidate.blocks[layer_id].moe),
        copy.deepcopy(candidate),
        history,
        diagnostics,
    )
