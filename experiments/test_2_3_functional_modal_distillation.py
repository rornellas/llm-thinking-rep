#!/usr/bin/env python3
"""Distill a trained conventional MoE into scalar modal experts.

The teacher is trained normally. A modal student receives all compatible
non-expert parameters and router weights from the teacher; those parameters are
then frozen. Only shared modal matrices and expert codes are optimized against:

1. teacher MoE outputs at every layer;
2. teacher next-token distribution;
3. ground-truth language-model loss.

This is a direct test of post-training functional conversion after raw-weight
factorizations failed.
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


def load_source(path: Path):
    spec = importlib.util.spec_from_file_location("modal_distillation_source", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@dataclass
class StudentResult:
    modal_rank: int
    expert_parameter_ratio: float
    idealized_expert_compute_ratio: float
    trainable_modal_parameters: int
    validation_loss_before: float
    validation_loss_after: float
    validation_loss_teacher: float
    loss_ratio_to_teacher_before: float
    loss_ratio_to_teacher_after: float
    teacher_student_kl_before: float
    teacher_student_kl_after: float
    moe_relative_error_before: float
    moe_relative_error_after: float
    elapsed_seconds: float


def copy_compatible_teacher_state(teacher: nn.Module, student: nn.Module) -> list[str]:
    teacher_state = teacher.state_dict()
    student_state = student.state_dict()
    copied: list[str] = []
    for name, value in teacher_state.items():
        if name not in student_state or student_state[name].shape != value.shape:
            continue
        if name.endswith(".moe.gate") or name.endswith(".moe.up") or name.endswith(".moe.down"):
            continue
        student_state[name].copy_(value)
        copied.append(name)
    student.load_state_dict(student_state)
    return copied


def freeze_non_modal(student: nn.Module) -> int:
    trainable_tokens = ("gate_modes", "up_modes", "down_modes", "gate_codes", "up_codes", "down_codes")
    count = 0
    for name, parameter in student.named_parameters():
        parameter.requires_grad = any(token in name for token in trainable_tokens)
        if parameter.requires_grad:
            count += parameter.numel()
    return count


class MoECapture:
    def __init__(self, model: nn.Module, detach: bool) -> None:
        self.outputs: list[torch.Tensor | None] = [None] * len(model.blocks)
        self.handles = []
        for index, block in enumerate(model.blocks):
            self.handles.append(block.moe.register_forward_hook(self._hook(index, detach)))

    def _hook(self, index: int, detach: bool):
        def hook(_module, _inputs, output):
            value = output[0]
            self.outputs[index] = value.detach() if detach else value
        return hook

    def clear(self) -> None:
        for index in range(len(self.outputs)):
            self.outputs[index] = None

    def values(self) -> list[torch.Tensor]:
        if any(value is None for value in self.outputs):
            raise RuntimeError("missing MoE hook output")
        return [value for value in self.outputs if value is not None]

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()


def train_teacher(source, dataset, cfg, seed: int):
    source.set_seed(seed)
    model = source.LanguageModel(len(dataset.vocab), cfg, "baseline", None)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    generator = torch.Generator().manual_seed(seed + 701)
    model.train()
    for step in range(1, cfg.steps + 1):
        x, y = dataset.batch("train", cfg.batch_size, generator)
        logits, aux, _ = model(x)
        ce = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))
        loss = ce + cfg.aux_weight * aux
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()
        if step == 1 or step % cfg.eval_interval == 0 or step == cfg.steps:
            print(f"teacher step={step}/{cfg.steps} ce={float(ce):.4f} aux={float(aux):.4f}", flush=True)
    return model


@torch.no_grad()
def evaluate_pair(source, teacher, student, dataset, cfg, generator, batches: int = 20) -> dict[str, float]:
    teacher.eval(); student.eval()
    teacher_capture = MoECapture(teacher, detach=True)
    student_capture = MoECapture(student, detach=True)
    teacher_losses, student_losses, kls = [], [], []
    moe_residual, moe_energy = 0.0, 0.0
    for _ in range(batches):
        x, y = dataset.batch("validation", cfg.batch_size, generator)
        teacher_capture.clear(); student_capture.clear()
        teacher_logits, _, _ = teacher(x)
        student_logits, _, _ = student(x)
        teacher_losses.append(float(F.cross_entropy(teacher_logits.reshape(-1, teacher_logits.shape[-1]), y.reshape(-1))))
        student_losses.append(float(F.cross_entropy(student_logits.reshape(-1, student_logits.shape[-1]), y.reshape(-1))))
        teacher_prob = F.softmax(teacher_logits, dim=-1)
        kl = F.kl_div(F.log_softmax(student_logits, dim=-1), teacher_prob, reduction="none").sum(dim=-1).mean()
        kls.append(float(kl))
        for teacher_out, student_out in zip(teacher_capture.values(), student_capture.values(), strict=True):
            moe_residual += float(torch.sum((student_out - teacher_out) ** 2))
            moe_energy += float(torch.sum(teacher_out ** 2))
    teacher_capture.close(); student_capture.close()
    teacher.train(); student.train()
    return {
        "teacher_loss": float(np.mean(teacher_losses)),
        "student_loss": float(np.mean(student_losses)),
        "kl": float(np.mean(kls)),
        "moe_relative_error": math.sqrt(moe_residual / max(moe_energy, 1e-12)),
    }


def distill(source, teacher, student, dataset, cfg, seed: int, steps: int):
    trainable = [parameter for parameter in student.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=6e-4, weight_decay=0.02)
    generator = torch.Generator().manual_seed(seed + 1701)
    teacher.eval(); student.train()
    teacher_capture = MoECapture(teacher, detach=True)
    student_capture = MoECapture(student, detach=False)
    history: list[dict[str, float]] = []
    for step in range(1, steps + 1):
        x, y = dataset.batch("train", cfg.batch_size, generator)
        teacher_capture.clear(); student_capture.clear()
        with torch.no_grad():
            teacher_logits, _, _ = teacher(x)
        student_logits, aux, _ = student(x)
        ce = F.cross_entropy(student_logits.reshape(-1, student_logits.shape[-1]), y.reshape(-1))
        teacher_prob = F.softmax(teacher_logits, dim=-1)
        kl = F.kl_div(F.log_softmax(student_logits, dim=-1), teacher_prob, reduction="none").sum(dim=-1).mean()
        feature_terms = []
        for teacher_out, student_out in zip(teacher_capture.values(), student_capture.values(), strict=True):
            feature_terms.append(torch.mean((student_out - teacher_out) ** 2) / (torch.mean(teacher_out ** 2) + 1e-8))
        feature = torch.stack(feature_terms).mean()
        loss = 0.35 * ce + 0.45 * kl + 0.75 * feature + cfg.aux_weight * aux
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, cfg.grad_clip)
        optimizer.step()
        if step == 1 or step % cfg.eval_interval == 0 or step == steps:
            row = {"step": float(step), "ce": float(ce.detach()), "kl": float(kl.detach()), "feature": float(feature.detach()), "aux": float(aux.detach()), "total": float(loss.detach())}
            history.append(row)
            print(f"distill rank={student.blocks[0].moe.modal_rank} step={step}/{steps} ce={row['ce']:.4f} kl={row['kl']:.4f} feature={row['feature']:.4f}", flush=True)
    teacher_capture.close(); student_capture.close()
    return history


def exact_ratios(rank: int, n_experts: int, top_k: int, d_model: int, d_ff: int) -> tuple[float, float]:
    parameters = (rank + 1) / n_experts + rank / (d_model * d_ff)
    compute = (rank + 1) / top_k + rank / (d_model * d_ff)
    return parameters, compute


def make_decision(results: list[dict[str, Any]]) -> dict[str, Any]:
    indexed = {int(row["modal_rank"]): row for row in results}
    k1, k2 = indexed[1], indexed[2]
    if k1["loss_ratio_to_teacher_after"] <= 1.05 and k1["teacher_student_kl_after"] <= 0.10 and k1["moe_relative_error_after"] <= 0.35:
        verdict = "PASS_K1"
    elif k2["loss_ratio_to_teacher_after"] <= 1.05 and k2["teacher_student_kl_after"] <= 0.10 and k2["moe_relative_error_after"] <= 0.35:
        verdict = "PASS_K2"
    elif k2["loss_ratio_to_teacher_after"] <= 1.10:
        verdict = "BORDERLINE"
    else:
        verdict = "FAIL"
    return {"verdict": verdict, "rule": "PASS requires <=5% validation-loss penalty, KL <=0.10 nat/token, and layerwise MoE relative error <=35% after replacing only expert matrices."}


def self_test(source) -> None:
    source.set_seed(101)
    cfg = source.Config(steps=1, eval_batches=1, batch_size=2, seq_len=8, d_model=24, d_ff=32, n_experts=4, top_k=2, n_layers=1)
    teacher = source.LanguageModel(20, cfg, "baseline", None)
    student = source.LanguageModel(20, cfg, "modal", 1)
    copied = copy_compatible_teacher_state(teacher, student)
    trainable = freeze_non_modal(student)
    if not copied or trainable <= 0:
        raise AssertionError((copied, trainable))
    if any(parameter.requires_grad for name, parameter in student.named_parameters() if ".moe.router." in name):
        raise AssertionError("router must be frozen")
    x = torch.randint(0, 20, (2, 8))
    logits, _, _ = student(x)
    logits.sum().backward()
    if any(parameter.grad is None for name, parameter in student.named_parameters() if parameter.requires_grad):
        raise AssertionError("missing modal gradient")
    print(f"self-test passed: copied={len(copied)} tensors, trainable modal parameters={trainable}")


def write_outputs(output_dir: Path, payload: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "functional_distillation.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with (output_dir / "results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(payload["results"][0].keys()))
        writer.writeheader(); writer.writerows(payload["results"])
    rows = []
    for rank, history in payload["history"].items():
        for item in history:
            rows.append({"modal_rank": rank, **item})
    with (output_dir / "history.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader(); writer.writerows(rows)
    lines = [
        "# Test 2.3 — functional distillation of trained experts", "", f"**Decision:** **{payload['decision']['verdict']}**", "",
        "All compatible non-expert weights and router weights are copied from the conventional teacher and frozen. Only modal matrices and codes are trained.", "",
        "| Rank | Expert params | Ideal expert compute | Loss before | Loss after | Loss/teacher | KL after | MoE error before | MoE error after |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["results"]:
        lines.append(f"| {row['modal_rank']} | {row['expert_parameter_ratio']:.2%} | {row['idealized_expert_compute_ratio']:.2%} | {row['validation_loss_before']:.4f} | {row['validation_loss_after']:.4f} | {row['loss_ratio_to_teacher_after']:.3f}× | {row['teacher_student_kl_after']:.4f} | {row['moe_relative_error_before']:.2%} | {row['moe_relative_error_after']:.2%} |")
    lines += ["", "A positive result would demonstrate functional conversion by distillation, not analytic weight decomposition. The small model remains a mechanism test; scaling and multi-domain validation remain required."]
    (output_dir / "VERDICT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--text", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("results/functional_distillation"))
    parser.add_argument("--teacher-steps", type=int, default=400)
    parser.add_argument("--distill-steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args(argv)
    source = load_source(args.source)
    source.torch.set_num_threads(args.threads)
    source.torch.set_num_interop_threads(1)
    if args.self_test:
        self_test(source); return 0
    dataset = source.CharDataset(args.text.read_text(encoding="utf-8"), 64)
    cfg = source.Config(steps=args.teacher_steps, eval_batches=20, eval_interval=100, batch_size=16, seq_len=64)
    teacher = train_teacher(source, dataset, cfg, args.seed)
    eval_gen = torch.Generator().manual_seed(args.seed + 2901)
    results: list[StudentResult] = []
    histories: dict[str, list[dict[str, float]]] = {}
    for rank in (1, 2):
        source.set_seed(args.seed + rank)
        student = source.LanguageModel(len(dataset.vocab), cfg, "modal", rank)
        copied = copy_compatible_teacher_state(teacher, student)
        trainable = freeze_non_modal(student)
        before = evaluate_pair(source, teacher, student, dataset, cfg, eval_gen)
        started = time.perf_counter()
        history = distill(source, teacher, student, dataset, cfg, args.seed + 5000 * rank, args.distill_steps)
        elapsed = time.perf_counter() - started
        after = evaluate_pair(source, teacher, student, dataset, cfg, eval_gen)
        p_ratio, c_ratio = exact_ratios(rank, cfg.n_experts, cfg.top_k, cfg.d_model, cfg.d_ff)
        results.append(StudentResult(
            modal_rank=rank, expert_parameter_ratio=p_ratio, idealized_expert_compute_ratio=c_ratio,
            trainable_modal_parameters=trainable, validation_loss_before=before["student_loss"],
            validation_loss_after=after["student_loss"], validation_loss_teacher=after["teacher_loss"],
            loss_ratio_to_teacher_before=before["student_loss"] / before["teacher_loss"],
            loss_ratio_to_teacher_after=after["student_loss"] / after["teacher_loss"],
            teacher_student_kl_before=before["kl"], teacher_student_kl_after=after["kl"],
            moe_relative_error_before=before["moe_relative_error"], moe_relative_error_after=after["moe_relative_error"],
            elapsed_seconds=elapsed,
        ))
        histories[str(rank)] = history
        print(f"rank={rank} copied={len(copied)} trainable={trainable} before={before} after={after}", flush=True)
    payload = {"metadata": {"task": "Tiny Shakespeare character LM", "teacher_steps": args.teacher_steps, "distill_steps": args.distill_steps, "seed": args.seed}, "results": [asdict(x) for x in results], "history": histories}
    payload["decision"] = make_decision(payload["results"])
    write_outputs(args.output_dir, payload)
    print((args.output_dir / "VERDICT.md").read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
