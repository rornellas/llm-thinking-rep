#!/usr/bin/env python3
"""Assignment-supervised modal distillation of the real OLMoE down branch.

Test 3.0 optimized only the aggregate token output from 515 training tokens.
Because a 1024-dimensional shared matrix could interpolate those examples, the
common mode overfit and starved expert codes of useful gradient. This test uses
all eight captured expert outputs as supervision, giving 4,120 training
assignments, samples output coordinates during optimization, and selects the
best checkpoint on held-out tokens.
"""
from __future__ import annotations

import argparse
import copy
import csv
import importlib.util
import json
import math
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn.functional as F

EPS = 1e-12


def load_base():
    path = Path(__file__).with_name("test_3_0_real_olmoe_down_distillation.py")
    spec = importlib.util.spec_from_file_location("real_down_assignment_base", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


base = load_base()
ModalDown = base.ModalDown


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def assignment_predict_rows(
    model: ModalDown,
    z: torch.Tensor,
    experts: torch.Tensor,
    rows: torch.Tensor,
    *,
    code_mode: str = "normal",
    permutation: torch.Tensor | None = None,
) -> torch.Tensor:
    selected_modes = model.modes.index_select(1, rows)
    codes = model.full_codes(code_mode, permutation).index_select(0, experts)
    if model.code_type == "scalar":
        mode_outputs = torch.einsum("kqm,bm->bkq", selected_modes, z)
        return torch.einsum("bkq,bk->bq", mode_outputs, codes)
    mode_inputs = codes * z[:, None, :]
    return torch.einsum("kqm,bkm->bq", selected_modes, mode_inputs)


def aggregate_predict_rows(
    model: ModalDown,
    z: torch.Tensor,
    topk: torch.Tensor,
    weights: torch.Tensor,
    rows: torch.Tensor,
    *,
    code_mode: str = "normal",
    permutation: torch.Tensor | None = None,
) -> torch.Tensor:
    selected_modes = model.modes.index_select(1, rows)
    codes = model.full_codes(code_mode, permutation)[topk]
    if model.code_type == "scalar":
        mode_inputs = torch.einsum("bt,btk,btm->bkm", weights, codes, z)
    else:
        mode_inputs = torch.einsum("bt,btkm,btm->bkm", weights, codes, z)
    return torch.einsum("kqm,bkm->bq", selected_modes, mode_inputs)


def assignment_predict_full(
    model: ModalDown,
    z: torch.Tensor,
    experts: torch.Tensor,
    *,
    code_mode: str = "normal",
    permutation: torch.Tensor | None = None,
    chunk: int = 32,
) -> torch.Tensor:
    outputs: list[torch.Tensor] = []
    rows = torch.arange(model.d_model)
    with torch.no_grad():
        for start in range(0, len(z), chunk):
            outputs.append(assignment_predict_rows(
                model, z[start:start + chunk], experts[start:start + chunk], rows,
                code_mode=code_mode, permutation=permutation,
            ))
    return torch.cat(outputs, dim=0)


def normalized_mse(prediction: torch.Tensor, target: torch.Tensor, sample_weight: torch.Tensor | None = None) -> torch.Tensor:
    squared = (prediction - target).square()
    target_squared = target.square()
    if sample_weight is not None:
        weight = sample_weight[:, None]
        squared = squared * weight
        target_squared = target_squared * weight
    return squared.mean() / target_squared.mean().clamp_min(1e-8)


@torch.no_grad()
def evaluate(
    model: ModalDown,
    arrays: dict[str, torch.Tensor],
    token_ids: torch.Tensor,
    *,
    code_mode: str = "normal",
    permutation: torch.Tensor | None = None,
) -> dict[str, float]:
    z = arrays["z"].index_select(0, token_ids)
    topk = arrays["topk"].index_select(0, token_ids)
    weights = arrays["weights"].index_select(0, token_ids)
    teacher_down = arrays["down"].index_select(0, token_ids)
    teacher_out = arrays["moe_out"].index_select(0, token_ids)
    flat_prediction = assignment_predict_full(
        model,
        z.reshape(-1, z.shape[-1]),
        topk.reshape(-1),
        code_mode=code_mode,
        permutation=permutation,
    )
    predicted_down = flat_prediction.reshape_as(teacher_down)
    predicted_out = torch.sum(predicted_down * weights[..., None], dim=1)

    def metrics(prediction: torch.Tensor, target: torch.Tensor, prefix: str) -> dict[str, float]:
        p = prediction.to(torch.float64)
        t = target.to(torch.float64)
        relative = torch.linalg.vector_norm(p - t) / torch.linalg.vector_norm(t).clamp_min(EPS)
        flat_p = p.reshape(-1, p.shape[-1])
        flat_t = t.reshape(-1, t.shape[-1])
        cosine = F.cosine_similarity(flat_p, flat_t, dim=-1)
        per_relative = torch.linalg.vector_norm(flat_p - flat_t, dim=-1) / torch.linalg.vector_norm(flat_t, dim=-1).clamp_min(EPS)
        return {
            f"{prefix}_relative_error": float(relative),
            f"{prefix}_mean_cosine": float(cosine.mean()),
            f"{prefix}_p05_cosine": float(torch.quantile(cosine, 0.05)),
            f"{prefix}_mean_sample_relative_error": float(per_relative.mean()),
            f"{prefix}_p95_sample_relative_error": float(torch.quantile(per_relative, 0.95)),
        }

    result = metrics(predicted_down, teacher_down, "assignment")
    result.update(metrics(predicted_out, teacher_out, "aggregate"))
    return result


@dataclass
class Result:
    variant: str
    rank: int
    code_type: str
    parameter_ratio: float
    compression_factor: float
    idealized_compute_ratio: float
    best_step: int
    train_assignment_relative_error: float
    validation_assignment_relative_error: float
    validation_assignment_mean_cosine: float
    validation_aggregate_relative_error: float
    validation_aggregate_mean_cosine: float
    validation_aggregate_p05_cosine: float
    mean_code_ablation_ratio: float | None
    shuffled_code_ablation_ratio: float | None
    zero_code_ablation_ratio: float | None
    elapsed_seconds: float


def train_variant(
    arrays: dict[str, torch.Tensor],
    train_ids: np.ndarray,
    validation_ids: np.ndarray,
    *,
    rank: int,
    code_type: str,
    warmup_steps: int,
    steps: int,
    assignment_batch: int,
    token_batch: int,
    output_rows: int,
    seed: int,
) -> Result:
    set_seed(seed)
    z, topk, weights = arrays["z"], arrays["topk"], arrays["weights"]
    teacher_down, teacher_out = arrays["down"], arrays["moe_out"]
    d_ff, d_model = z.shape[-1], teacher_out.shape[-1]
    model = ModalDown(d_model, d_ff, 64, rank, code_type)
    # The original initializer is intended for full training. Use a smaller
    # scale here because output-coordinate SGD gives direct supervision.
    with torch.no_grad():
        for mode in model.modes:
            mode.normal_(std=0.01 / math.sqrt(d_ff))
        if model.codes.numel():
            model.codes.normal_(std=0.20)

    train_tokens = torch.as_tensor(train_ids, dtype=torch.long)
    validation_tokens = torch.as_tensor(validation_ids, dtype=torch.long)
    train_z = z.index_select(0, train_tokens)
    train_topk = topk.index_select(0, train_tokens)
    train_weights = weights.index_select(0, train_tokens)
    train_down = teacher_down.index_select(0, train_tokens)
    train_out = teacher_out.index_select(0, train_tokens)
    flat_z = train_z.reshape(-1, d_ff)
    flat_experts = train_topk.reshape(-1)
    flat_down = train_down.reshape(-1, d_model)
    flat_route = train_weights.reshape(-1)
    route_importance = (0.5 + 0.5 * flat_route / flat_route.mean().clamp_min(1e-8)).clamp(0.25, 4.0)
    generator = torch.Generator().manual_seed(seed + 77)

    def one_step(optimizer: torch.optim.Optimizer, common_only: bool) -> float:
        assignment_ids = torch.randint(0, len(flat_z), (min(assignment_batch, len(flat_z)),), generator=generator)
        token_positions = torch.randint(0, len(train_z), (min(token_batch, len(train_z)),), generator=generator)
        rows = torch.randperm(d_model, generator=generator)[:min(output_rows, d_model)]
        predicted_assignment = assignment_predict_rows(
            model,
            flat_z.index_select(0, assignment_ids),
            flat_experts.index_select(0, assignment_ids),
            rows,
            code_mode="zero" if common_only else "normal",
        )
        target_assignment = flat_down.index_select(0, assignment_ids).index_select(1, rows)
        assignment_loss = normalized_mse(
            predicted_assignment,
            target_assignment,
            route_importance.index_select(0, assignment_ids),
        )
        predicted_aggregate = aggregate_predict_rows(
            model,
            train_z.index_select(0, token_positions),
            train_topk.index_select(0, token_positions),
            train_weights.index_select(0, token_positions),
            rows,
            code_mode="zero" if common_only else "normal",
        )
        target_aggregate = train_out.index_select(0, token_positions).index_select(1, rows)
        aggregate_loss = normalized_mse(predicted_aggregate, target_aggregate)
        code_penalty = model.codes.square().mean() if model.codes.numel() else torch.zeros(())
        loss = 0.55 * assignment_loss + 0.45 * aggregate_loss + 1e-5 * code_penalty
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if common_only and model.codes.numel() and model.codes.grad is not None:
            model.codes.grad.zero_()
        if common_only and model.modes.grad is not None and model.modes.shape[0] > 1:
            model.modes.grad[1:].zero_()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        return float(loss.detach())

    warmup_optimizer = torch.optim.AdamW([model.modes], lr=3e-3, weight_decay=1e-4)
    for step in range(1, warmup_steps + 1):
        loss = one_step(warmup_optimizer, common_only=True)
        if step == 1 or step % 50 == 0 or step == warmup_steps:
            print(f"{code_type}-k{rank} warmup={step}/{warmup_steps} loss={loss:.5f}", flush=True)

    parameter_groups: list[dict[str, Any]] = [{"params": [model.modes], "lr": 1.5e-3}]
    if model.codes.numel():
        parameter_groups.append({"params": [model.codes], "lr": 6e-3})
    optimizer = torch.optim.AdamW(parameter_groups, weight_decay=1e-4)
    best_state = copy.deepcopy(model.state_dict())
    best_step = 0
    best_error = float("inf")
    started = time.perf_counter()
    for step in range(1, steps + 1):
        loss = one_step(optimizer, common_only=False)
        if step == 1 or step % 100 == 0 or step == steps:
            validation = evaluate(model, arrays, validation_tokens)
            error = validation["aggregate_relative_error"]
            print(
                f"{code_type}-k{rank} step={step}/{steps} loss={loss:.5f} "
                f"val-aggregate-error={error:.4%}",
                flush=True,
            )
            if error < best_error:
                best_error = error
                best_step = step
                best_state = copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    train_metrics = evaluate(model, arrays, train_tokens)
    validation_metrics = evaluate(model, arrays, validation_tokens)

    if rank > 0:
        permutation = torch.randperm(64, generator=torch.Generator().manual_seed(seed + 999))
        normal_error = validation_metrics["aggregate_relative_error"]
        mean_error = evaluate(model, arrays, validation_tokens, code_mode="mean")["aggregate_relative_error"]
        shuffled_error = evaluate(model, arrays, validation_tokens, code_mode="shuffle", permutation=permutation)["aggregate_relative_error"]
        zero_error = evaluate(model, arrays, validation_tokens, code_mode="zero")["aggregate_relative_error"]
        mean_ratio = mean_error / max(normal_error, EPS)
        shuffled_ratio = shuffled_error / max(normal_error, EPS)
        zero_ratio = zero_error / max(normal_error, EPS)
    else:
        mean_ratio = shuffled_ratio = zero_ratio = None

    parameter_ratio, compute_ratio = model.parameter_ratio(top_k=8)
    return Result(
        variant=f"{code_type}-k{rank}", rank=rank, code_type=code_type,
        parameter_ratio=parameter_ratio, compression_factor=1.0 / parameter_ratio,
        idealized_compute_ratio=compute_ratio, best_step=best_step,
        train_assignment_relative_error=train_metrics["assignment_relative_error"],
        validation_assignment_relative_error=validation_metrics["assignment_relative_error"],
        validation_assignment_mean_cosine=validation_metrics["assignment_mean_cosine"],
        validation_aggregate_relative_error=validation_metrics["aggregate_relative_error"],
        validation_aggregate_mean_cosine=validation_metrics["aggregate_mean_cosine"],
        validation_aggregate_p05_cosine=validation_metrics["aggregate_p05_cosine"],
        mean_code_ablation_ratio=mean_ratio,
        shuffled_code_ablation_ratio=shuffled_ratio,
        zero_code_ablation_ratio=zero_ratio,
        elapsed_seconds=time.perf_counter() - started,
    )


def make_decision(results: Sequence[Result]) -> dict[str, Any]:
    best = min(results, key=lambda row: row.validation_aggregate_relative_error)
    if best.validation_aggregate_relative_error <= 0.10 and best.validation_aggregate_mean_cosine >= 0.995:
        verdict = "REAL_ASSIGNMENT_DISTILLATION_PASS"
    elif best.validation_aggregate_relative_error <= 0.30 and best.validation_aggregate_mean_cosine >= 0.95:
        verdict = "BORDERLINE"
    else:
        verdict = "FAIL"
    return {
        "verdict": verdict,
        "best_variant": best.variant,
        "best_validation_aggregate_relative_error": best.validation_aggregate_relative_error,
        "best_validation_aggregate_mean_cosine": best.validation_aggregate_mean_cosine,
        "rule": "PASS requires <=10% held-out aggregate error and cosine >=0.995; BORDERLINE permits <=30% and cosine >=0.95.",
    }


def self_test() -> None:
    set_seed(5)
    model = ModalDown(16, 12, 8, 2, "neuronwise")
    z = torch.randn(9, 12)
    experts = torch.randint(0, 8, (9,))
    rows = torch.arange(16)
    direct = assignment_predict_rows(model, z, experts, rows)
    full_codes = model.full_codes()[experts]
    explicit = torch.zeros_like(direct)
    for sample in range(len(z)):
        for mode in range(model.n_modes):
            explicit[sample] += model.modes[mode] @ (full_codes[sample, mode] * z[sample])
    error = float(torch.max(torch.abs(direct - explicit)))
    if error > 2e-5:
        raise AssertionError(error)
    direct.square().mean().backward()
    if any(parameter.grad is None for parameter in model.parameters()):
        raise AssertionError("missing gradients")
    print(f"self-test passed; assignment algebra error={error:.3e}")


def write_outputs(output_dir: Path, payload: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "assignment_distillation.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    with (output_dir / "results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(payload["results"][0].keys()))
        writer.writeheader(); writer.writerows(payload["results"])
    d = payload["decision"]
    lines = [
        "# Test 3.1 — assignment-supervised real OLMoE down distillation", "",
        f"**Decision:** **{d['verdict']}**", "",
        "| Variant | Params | Compute | Best step | Train assignment error | Val assignment error | Val aggregate error | Aggregate cosine | Mean-code ablation | Shuffle ablation | Zero ablation |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["results"]:
        def ratio(value: Any) -> str:
            return "n/a" if value is None else f"{value:.3f}×"
        lines.append(
            f"| {row['variant']} | {row['parameter_ratio']:.3%} | {row['idealized_compute_ratio']:.3%} | "
            f"{row['best_step']} | {row['train_assignment_relative_error']:.2%} | "
            f"{row['validation_assignment_relative_error']:.2%} | {row['validation_aggregate_relative_error']:.2%} | "
            f"{row['validation_aggregate_mean_cosine']:.5f} | {ratio(row['mean_code_ablation_ratio'])} | "
            f"{ratio(row['shuffled_code_ablation_ratio'])} | {ratio(row['zero_code_ablation_ratio'])} |"
        )
    lines += [
        "",
        "Training uses the real per-assignment `D_e z_e` tensors as well as the weighted aggregate output. Output coordinates are randomly sampled during optimization; all 2,048 dimensions are used for held-out evaluation. The split is by token, so no assignment from a validation token appears in training.",
    ]
    (output_dir / "VERDICT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("results/test-3-1/latest"))
    parser.add_argument("--warmup-steps", type=int, default=180)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--assignment-batch", type=int, default=24)
    parser.add_argument("--token-batch", type=int, default=6)
    parser.add_argument("--output-rows", type=int, default=128)
    parser.add_argument("--seed", type=int, default=31001)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args(argv)
    torch.set_num_threads(args.threads); torch.set_num_interop_threads(1)
    if args.self_test:
        self_test(); return 0
    if args.capture_dir is None:
        parser.error("--capture-dir is required")

    capture = base.load_capture(args.capture_dir)
    arrays = {
        "z": torch.from_numpy(capture["z"]),
        "topk": torch.from_numpy(capture["topk"]).long(),
        "weights": torch.from_numpy(capture["weights"]),
        "down": torch.from_numpy(capture["down"]),
        "moe_out": torch.from_numpy(capture["moe_out"]),
    }
    train_ids, validation_ids = base.split_tokens(len(capture["z"]), args.seed)
    variants = [(0, "scalar"), (1, "scalar"), (1, "neuronwise"), (2, "neuronwise")]
    results: list[Result] = []
    for index, (rank, code_type) in enumerate(variants):
        results.append(train_variant(
            arrays, train_ids, validation_ids,
            rank=rank, code_type=code_type,
            warmup_steps=args.warmup_steps,
            steps=args.steps,
            assignment_batch=args.assignment_batch,
            token_batch=args.token_batch,
            output_rows=args.output_rows,
            seed=args.seed + index * 1000,
        ))
    payload = {
        "metadata": {
            "tokens": int(len(capture["z"])),
            "train_tokens": int(len(train_ids)),
            "validation_tokens": int(len(validation_ids)),
            "train_assignments": int(len(train_ids) * 8),
            "validation_assignments": int(len(validation_ids) * 8),
            "experts": 64, "top_k": 8,
            "d_model": int(capture["moe_out"].shape[-1]),
            "d_ff": int(capture["z"].shape[-1]),
            "warmup_steps": args.warmup_steps,
            "steps": args.steps,
            "output_rows_per_step": args.output_rows,
            "seed": args.seed,
        },
        "decision": make_decision(results),
        "results": [asdict(row) for row in results],
    }
    write_outputs(args.output_dir, payload)
    print((args.output_dir / "VERDICT.md").read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
