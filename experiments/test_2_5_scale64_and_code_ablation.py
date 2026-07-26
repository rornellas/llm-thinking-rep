#!/usr/bin/env python3
"""Scale the modal architecture to 64 experts/top-8 and ablate expert codes."""
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


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@dataclass
class RunResult:
    variant: str
    validation_loss: float
    validation_perplexity: float
    trainable_parameters: int
    expert_parameter_ratio: float
    idealized_expert_compute_ratio: float
    utilization_entropy: float
    min_expert_fraction: float
    max_expert_fraction: float
    elapsed_seconds: float
    mean_code_std: float | None
    effective_code_rank: float | None
    mean_codes_loss: float | None
    shuffled_codes_loss: float | None
    zero_residual_codes_loss: float | None
    mean_codes_loss_ratio: float | None
    shuffled_codes_loss_ratio: float | None
    zero_residual_codes_loss_ratio: float | None


def exact_ratio(kind: str, rank: int, cfg) -> tuple[float, float]:
    original_params = 3 * cfg.n_experts * cfg.d_ff * cfg.d_model
    original_compute = 3 * cfg.top_k * cfg.d_ff * cfg.d_model
    shared = 3 * (rank + 1) * cfg.d_ff * cfg.d_model
    if kind == "neuronwise":
        codes_params = 3 * cfg.n_experts * rank * cfg.d_ff
        codes_compute = 3 * cfg.top_k * rank * cfg.d_ff
    else:
        codes_params = 3 * cfg.n_experts * rank
        codes_compute = 3 * cfg.top_k * rank
    return (shared + codes_params) / original_params, (shared + codes_compute) / original_compute


@torch.no_grad()
def evaluate_fixed(source, model, dataset, cfg, seed: int, batches: int = 20) -> tuple[float, np.ndarray]:
    generator = torch.Generator().manual_seed(seed)
    model.eval()
    losses: list[float] = []
    counts = np.zeros(cfg.n_experts, dtype=np.float64)
    for _ in range(batches):
        x, y = dataset.batch("validation", cfg.batch_size, generator)
        logits, _, routes = model(x)
        losses.append(float(F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))))
        for ids in routes:
            counts += np.bincount(ids.cpu().numpy().reshape(-1), minlength=cfg.n_experts)
    model.train()
    return float(np.mean(losses)), counts / max(float(counts.sum()), 1.0)


def train_model(source, model, dataset, cfg, seed: int):
    source.set_seed(seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    generator = torch.Generator().manual_seed(seed + 101)
    started = time.perf_counter()
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
            print(f"step={step}/{cfg.steps} ce={float(ce):.4f} aux={float(aux):.4f}", flush=True)
    return time.perf_counter() - started


def code_parameters(model) -> list[torch.nn.Parameter]:
    result = []
    for block in model.blocks:
        moe = block.moe
        for name in ("gate_codes", "up_codes", "down_codes"):
            if hasattr(moe, name):
                result.append(getattr(moe, name))
    return result


def code_diagnostics(model) -> tuple[float, float]:
    matrices = []
    stds = []
    for parameter in code_parameters(model):
        array = parameter.detach().float().cpu().numpy()
        stds.append(float(np.std(array, axis=0).mean()))
        matrices.append(array.reshape(array.shape[0], -1))
    if not matrices:
        return float("nan"), float("nan")
    joined = np.concatenate(matrices, axis=1)
    centered = joined - joined.mean(axis=0, keepdims=True)
    singular = np.linalg.svd(centered, compute_uv=False)
    energy = singular * singular
    participation = float(energy.sum() ** 2 / max(float(np.dot(energy, energy)), 1e-12))
    return float(np.mean(stds)), participation


def evaluate_ablations(source, model, dataset, cfg, seed: int) -> dict[str, float]:
    parameters = code_parameters(model)
    originals = [parameter.detach().clone() for parameter in parameters]
    normal, _ = evaluate_fixed(source, model, dataset, cfg, seed)

    with torch.no_grad():
        for parameter in parameters:
            parameter.copy_(parameter.mean(dim=0, keepdim=True).expand_as(parameter))
    mean_loss, _ = evaluate_fixed(source, model, dataset, cfg, seed)

    with torch.no_grad():
        for parameter, original in zip(parameters, originals, strict=True):
            parameter.copy_(original)
        generator = torch.Generator().manual_seed(seed + 77)
        permutation = torch.randperm(parameters[0].shape[0], generator=generator)
        for parameter, original in zip(parameters, originals, strict=True):
            parameter.copy_(original[permutation])
    shuffled_loss, _ = evaluate_fixed(source, model, dataset, cfg, seed)

    with torch.no_grad():
        for parameter in parameters:
            parameter.zero_()
    zero_loss, _ = evaluate_fixed(source, model, dataset, cfg, seed)

    with torch.no_grad():
        for parameter, original in zip(parameters, originals, strict=True):
            parameter.copy_(original)
    return {
        "normal": normal,
        "mean": mean_loss,
        "shuffled": shuffled_loss,
        "zero": zero_loss,
    }


def run_variant(source, neuronwise, dataset, cfg, seed: int, label: str, kind: str, rank: int | None) -> RunResult:
    source.set_seed(seed)
    if kind == "baseline":
        model = source.LanguageModel(len(dataset.vocab), cfg, "baseline", None)
        param_ratio = compute_ratio = 1.0
    else:
        source.ModalMoE = source._scalar_modal_class if kind == "scalar" else neuronwise.NeuronwiseModalMoE
        model = source.LanguageModel(len(dataset.vocab), cfg, "modal", rank)
        param_ratio, compute_ratio = exact_ratio(kind, int(rank), cfg)
    elapsed = train_model(source, model, dataset, cfg, seed)
    validation_loss, utilization = evaluate_fixed(source, model, dataset, cfg, seed + 999)
    nonzero = utilization[utilization > 0]
    entropy = float(-np.sum(nonzero * np.log(nonzero)) / math.log(cfg.n_experts)) if nonzero.size else 0.0
    mean_std = effective_rank = None
    mean_loss = shuffled_loss = zero_loss = None
    mean_ratio = shuffled_ratio = zero_ratio = None
    if kind != "baseline":
        mean_std, effective_rank = code_diagnostics(model)
        ablations = evaluate_ablations(source, model, dataset, cfg, seed + 999)
        validation_loss = ablations["normal"]
        mean_loss, shuffled_loss, zero_loss = ablations["mean"], ablations["shuffled"], ablations["zero"]
        mean_ratio = mean_loss / validation_loss
        shuffled_ratio = shuffled_loss / validation_loss
        zero_ratio = zero_loss / validation_loss
    return RunResult(
        variant=label,
        validation_loss=validation_loss,
        validation_perplexity=float(math.exp(min(validation_loss, 20.0))),
        trainable_parameters=sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
        expert_parameter_ratio=param_ratio,
        idealized_expert_compute_ratio=compute_ratio,
        utilization_entropy=entropy,
        min_expert_fraction=float(utilization.min()),
        max_expert_fraction=float(utilization.max()),
        elapsed_seconds=elapsed,
        mean_code_std=mean_std,
        effective_code_rank=effective_rank,
        mean_codes_loss=mean_loss,
        shuffled_codes_loss=shuffled_loss,
        zero_residual_codes_loss=zero_loss,
        mean_codes_loss_ratio=mean_ratio,
        shuffled_codes_loss_ratio=shuffled_ratio,
        zero_residual_codes_loss_ratio=zero_ratio,
    )


def make_decision(results: list[dict[str, Any]]) -> dict[str, Any]:
    indexed = {row["variant"]: row for row in results}
    baseline = indexed["baseline"]["validation_loss"]
    ratios = {name: row["validation_loss"] / baseline for name, row in indexed.items()}
    k1 = indexed["neuronwise-k1"]
    code_use = min(k1["mean_codes_loss_ratio"], k1["shuffled_codes_loss_ratio"], k1["zero_residual_codes_loss_ratio"])
    if ratios["neuronwise-k1"] <= 1.05 and code_use >= 1.01:
        verdict = "SCALE64_AND_CODES_USED"
    elif ratios["neuronwise-k2"] <= 1.05 and indexed["neuronwise-k2"]["zero_residual_codes_loss_ratio"] >= 1.01:
        verdict = "SCALE64_K2"
    elif ratios["neuronwise-k1"] <= 1.05:
        verdict = "QUALITY_PASS_CODES_NOT_IDENTIFIED"
    else:
        verdict = "FAIL"
    return {"verdict": verdict, "loss_ratios": ratios, "minimum_k1_ablation_penalty": code_use}


def self_test(source, neuronwise) -> None:
    source._scalar_modal_class = source.ModalMoE
    cfg = source.Config(d_model=24, d_ff=32, n_experts=8, top_k=2, n_layers=1, seq_len=8, batch_size=2, steps=1, eval_batches=1)
    text = "the modal test repeats enough characters to create a tiny dataset. " * 20
    dataset = source.CharDataset(text, cfg.seq_len)
    result = run_variant(source, neuronwise, dataset, cfg, 3, "neuronwise-k1", "neuronwise", 1)
    if not math.isfinite(result.validation_loss):
        raise AssertionError(result)
    print(f"self-test passed: loss={result.validation_loss:.4f}, code std={result.mean_code_std:.4f}")


def write_outputs(output_dir: Path, payload: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "scale64_code_ablation.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with (output_dir / "results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(payload["results"][0].keys()))
        writer.writeheader(); writer.writerows(payload["results"])
    baseline = next(row for row in payload["results"] if row["variant"] == "baseline")["validation_loss"]
    lines = [
        "# Test 2.5 — 64-expert/top-8 scaling and code ablations",
        "",
        f"**Decision:** **{payload['decision']['verdict']}**",
        "",
        "| Variant | Expert params | Ideal expert compute | Validation loss | Loss/full | Router entropy | Mean-code ablation | Shuffle ablation | Zero-code ablation |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["results"]:
        def fmt(value):
            return "n/a" if value is None else f"{value:.3f}×"
        lines.append(
            f"| {row['variant']} | {row['expert_parameter_ratio']:.3%} | {row['idealized_expert_compute_ratio']:.3%} | "
            f"{row['validation_loss']:.4f} | {row['validation_loss']/baseline:.3f}× | {row['utilization_entropy']:.3f} | "
            f"{fmt(row['mean_codes_loss_ratio'])} | {fmt(row['shuffled_codes_loss_ratio'])} | {fmt(row['zero_residual_codes_loss_ratio'])} |"
        )
    lines += [
        "",
        "An ablation penalty above 1.0 means expert-specific codes carry information beyond the common mode. This test matches OLMoE's 64-expert/top-8 geometry but remains a small character language model.",
    ]
    (output_dir / "VERDICT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--neuronwise-source", type=Path, required=True)
    parser.add_argument("--text", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("results/scale64"))
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--seed", type=int, default=4242)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args(argv)
    source = load_module("scale64_source", args.source)
    neuronwise = load_module("scale64_neuronwise", args.neuronwise_source)
    source._scalar_modal_class = source.ModalMoE
    source.torch.set_num_threads(args.threads)
    source.torch.set_num_interop_threads(1)
    if args.self_test:
        self_test(source, neuronwise); return 0
    cfg = source.Config(
        d_model=96, d_ff=128, n_experts=64, top_k=8, n_layers=2,
        seq_len=64, batch_size=16, steps=args.steps, eval_interval=100, eval_batches=20,
    )
    dataset = source.CharDataset(args.text.read_text(encoding="utf-8"), cfg.seq_len)
    variants = [
        ("baseline", "baseline", None),
        ("scalar-k1", "scalar", 1),
        ("neuronwise-k1", "neuronwise", 1),
        ("scalar-k2", "scalar", 2),
        ("neuronwise-k2", "neuronwise", 2),
    ]
    results = [asdict(run_variant(source, neuronwise, dataset, cfg, args.seed, label, kind, rank)) for label, kind, rank in variants]
    payload = {"metadata": {"experts": 64, "top_k": 8, "d_model": 96, "d_ff": 128, "steps": args.steps, "seed": args.seed}, "results": results}
    payload["decision"] = make_decision(results)
    write_outputs(args.output_dir, payload)
    print((args.output_dir / "VERDICT.md").read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
