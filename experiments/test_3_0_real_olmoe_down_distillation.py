#!/usr/bin/env python3
"""Distill the real OLMoE layer-7 down branch into directly executed modes.

Inputs are captured from the official OLMoE Q4_K_M graph:
- selected expert IDs and router weights;
- true SwiGLU intermediate vectors z_e;
- true per-expert down outputs and aggregate MoE output.

The student never reconstructs D_e. It evaluates

    s_k = sum_e pi_e * (a^D_{e,k} odot z_e)
    y   = sum_k D_k s_k

with either scalar or neuron-wise expert codes. Train/validation splits are by
token, so all eight assignments of a validation token remain held out.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

EPS = 1e-12


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_chunks(meta_path: Path) -> tuple[str, np.ndarray]:
    records = [json.loads(line) for line in meta_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    raw = np.fromfile(meta_path.with_suffix(".f32"), dtype=np.float32)
    offset = 0
    chunks: list[np.ndarray] = []
    for record in records:
        ne = [int(value) for value in record["ne"]]
        elements = int(record["elements"])
        chunk = raw[offset:offset + elements]
        offset += elements
        if chunk.size != elements:
            raise ValueError(f"truncated {record['name']}")
        shaped = chunk.reshape(ne[3], ne[2], ne[1], ne[0])
        chunks.append(np.squeeze(shaped))
    if offset != raw.size:
        raise ValueError(f"unused values in {meta_path}: {raw.size - offset}")
    return records[0]["name"], np.concatenate([np.atleast_2d(chunk) for chunk in chunks], axis=0)


def load_capture(capture_dir: Path) -> dict[str, np.ndarray]:
    captured: dict[str, np.ndarray] = {}
    for meta in sorted(capture_dir.glob("*.jsonl")):
        name, array = load_chunks(meta)
        captured[name] = np.squeeze(array)

    def exact(name: str) -> np.ndarray:
        if name not in captured:
            raise KeyError(f"missing {name}; available={sorted(captured)}")
        return np.asarray(captured[name])

    ordered = exact("ffn_moe_argsort-7")
    if ordered.ndim != 2 or ordered.shape[1] < 8:
        raise ValueError(("argsort", ordered.shape))
    topk = np.rint(ordered[:, :8]).astype(np.int64)
    weights = exact("ffn_moe_weights-7").astype(np.float32)
    z = exact("ffn_moe_swiglu-7").astype(np.float32)
    down = exact("ffn_moe_down-7").astype(np.float32)
    moe_out = exact("ffn_moe_out-7").astype(np.float32)
    x = exact("ffn_norm-7").astype(np.float32)

    n = z.shape[0]
    if topk.shape != (n, 8) or weights.shape != (n, 8):
        raise ValueError((topk.shape, weights.shape, z.shape))
    if z.ndim != 3 or down.ndim != 3 or moe_out.ndim != 2:
        raise ValueError((z.shape, down.shape, moe_out.shape))
    aggregate = np.sum(down * weights[..., None], axis=1)
    consistency = np.linalg.norm(aggregate.astype(np.float64) - moe_out.astype(np.float64)) / max(
        np.linalg.norm(moe_out.astype(np.float64)), EPS
    )
    if consistency > 1e-5:
        raise ValueError(f"captured down/output inconsistency: {consistency}")
    return {
        "x": x,
        "topk": topk,
        "weights": weights,
        "z": z,
        "down": down,
        "moe_out": moe_out,
        "consistency": np.array(consistency),
    }


class ModalDown(nn.Module):
    def __init__(self, d_model: int, d_ff: int, n_experts: int, rank: int, code_type: str) -> None:
        super().__init__()
        if rank < 0 or code_type not in {"scalar", "neuronwise"}:
            raise ValueError((rank, code_type))
        self.d_model = d_model
        self.d_ff = d_ff
        self.n_experts = n_experts
        self.rank = rank
        self.n_modes = rank + 1
        self.code_type = code_type
        self.modes = nn.Parameter(torch.empty(self.n_modes, d_model, d_ff))
        if code_type == "scalar":
            self.codes = nn.Parameter(torch.empty(n_experts, rank))
        else:
            self.codes = nn.Parameter(torch.empty(n_experts, rank, d_ff))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for mode_index, mode in enumerate(self.modes):
            nn.init.xavier_uniform_(mode)
            if mode_index > 0:
                mode.data.mul_(0.05 / math.sqrt(max(1, self.rank)))
        if self.codes.numel():
            nn.init.normal_(self.codes, std=0.10)

    def full_codes(self, mode: str = "normal", permutation: torch.Tensor | None = None) -> torch.Tensor:
        if self.code_type == "scalar":
            common = torch.ones(self.n_experts, 1, dtype=self.modes.dtype, device=self.modes.device)
        else:
            common = torch.ones(self.n_experts, 1, self.d_ff, dtype=self.modes.dtype, device=self.modes.device)
        if self.rank == 0:
            return common
        residual = self.codes
        if mode == "mean":
            residual = residual.mean(dim=0, keepdim=True).expand_as(residual)
        elif mode == "shuffle":
            if permutation is None:
                raise ValueError("permutation required")
            residual = residual.index_select(0, permutation)
        elif mode == "zero":
            residual = torch.zeros_like(residual)
        elif mode != "normal":
            raise ValueError(mode)
        return torch.cat([common, residual], dim=1)

    def forward(
        self,
        z: torch.Tensor,
        topk: torch.Tensor,
        weights: torch.Tensor,
        *,
        code_mode: str = "normal",
        permutation: torch.Tensor | None = None,
    ) -> torch.Tensor:
        selected = self.full_codes(code_mode, permutation)[topk]
        if self.code_type == "scalar":
            mode_inputs = torch.einsum("bt,btk,btm->bkm", weights, selected, z)
        else:
            mode_inputs = torch.einsum("bt,btkm,btm->bkm", weights, selected, z)
        return torch.einsum("kdm,bkm->bd", self.modes, mode_inputs)

    def parameter_ratio(self, top_k: int) -> tuple[float, float]:
        original = self.n_experts * self.d_model * self.d_ff
        student = self.n_modes * self.d_model * self.d_ff + self.codes.numel()
        combine_overhead = self.rank / self.d_model
        compute = self.n_modes / top_k + combine_overhead
        return student / original, compute


@torch.no_grad()
def ridge_initialize_common(
    model: ModalDown,
    z: torch.Tensor,
    weights: torch.Tensor,
    target: torch.Tensor,
    ridge_fraction: float = 1e-3,
) -> None:
    aggregate_z = torch.sum(z * weights[..., None], dim=1).to(torch.float64)
    y = target.to(torch.float64)
    gram = aggregate_z @ aggregate_z.T
    ridge = ridge_fraction * float(torch.trace(gram)) / max(1, gram.shape[0])
    gram = gram + ridge * torch.eye(gram.shape[0], dtype=gram.dtype)
    alpha = torch.linalg.solve(gram, y)
    coefficients = aggregate_z.T @ alpha
    model.modes[0].copy_(coefficients.T.to(torch.float32))


def split_tokens(count: int, seed: int, fraction: float = 0.67) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    order = rng.permutation(count)
    train_count = int(round(count * fraction))
    return np.sort(order[:train_count]), np.sort(order[train_count:])


def relative_mse(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.mean((prediction - target).square()) / target.square().mean().clamp_min(1e-8)


@torch.no_grad()
def metrics(prediction: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    p = prediction.to(torch.float64)
    t = target.to(torch.float64)
    residual = torch.linalg.vector_norm(p - t) / torch.linalg.vector_norm(t).clamp_min(EPS)
    per_cos = F.cosine_similarity(p, t, dim=-1)
    per_rel = torch.linalg.vector_norm(p - t, dim=-1) / torch.linalg.vector_norm(t, dim=-1).clamp_min(EPS)
    return {
        "relative_error": float(residual),
        "mean_cosine": float(per_cos.mean()),
        "p05_cosine": float(torch.quantile(per_cos, 0.05)),
        "mean_token_relative_error": float(per_rel.mean()),
        "p95_token_relative_error": float(torch.quantile(per_rel, 0.95)),
        "relative_mse": float(torch.mean((p - t).square()) / torch.mean(t.square()).clamp_min(EPS)),
    }


@dataclass
class VariantResult:
    variant: str
    rank: int
    code_type: str
    parameter_ratio: float
    compression_factor: float
    idealized_compute_ratio: float
    initial_validation_relative_error: float
    validation_relative_error: float
    validation_mean_cosine: float
    validation_p05_cosine: float
    validation_mean_token_relative_error: float
    validation_p95_token_relative_error: float
    train_relative_error: float
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
    steps: int,
    batch_size: int,
    seed: int,
) -> VariantResult:
    set_seed(seed)
    z, topk, weights, target = arrays["z"], arrays["topk"], arrays["weights"], arrays["moe_out"]
    d_ff, d_model = z.shape[-1], target.shape[-1]
    model = ModalDown(d_model, d_ff, 64, rank, code_type)
    train_tensor = torch.as_tensor(train_ids, dtype=torch.long)
    validation_tensor = torch.as_tensor(validation_ids, dtype=torch.long)
    ridge_initialize_common(
        model,
        z.index_select(0, train_tensor),
        weights.index_select(0, train_tensor),
        target.index_select(0, train_tensor),
    )
    with torch.no_grad():
        initial = model(
            z.index_select(0, validation_tensor),
            topk.index_select(0, validation_tensor),
            weights.index_select(0, validation_tensor),
        )
        initial_metrics = metrics(initial, target.index_select(0, validation_tensor))

    parameter_groups = [{"params": [model.modes], "lr": 2e-4}]
    if model.codes.numel():
        parameter_groups.append({"params": [model.codes], "lr": 2e-3})
    optimizer = torch.optim.AdamW(parameter_groups, weight_decay=1e-4)
    generator = torch.Generator().manual_seed(seed + 99)
    started = time.perf_counter()
    model.train()
    for step in range(1, steps + 1):
        positions = torch.randint(0, len(train_tensor), (min(batch_size, len(train_tensor)),), generator=generator)
        ids = train_tensor.index_select(0, positions)
        prediction = model(z.index_select(0, ids), topk.index_select(0, ids), weights.index_select(0, ids))
        batch_target = target.index_select(0, ids)
        fit = relative_mse(prediction, batch_target)
        code_regularizer = model.codes.square().mean() if model.codes.numel() else torch.zeros(())
        loss = fit + 1e-5 * code_regularizer
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step == 1 or step % 50 == 0 or step == steps:
            print(f"{code_type}-k{rank} step={step}/{steps} rel-mse={float(fit.detach()):.5f}", flush=True)

    model.eval()
    with torch.no_grad():
        train_prediction = model(
            z.index_select(0, train_tensor), topk.index_select(0, train_tensor), weights.index_select(0, train_tensor)
        )
        validation_prediction = model(
            z.index_select(0, validation_tensor), topk.index_select(0, validation_tensor), weights.index_select(0, validation_tensor)
        )
        train_metrics = metrics(train_prediction, target.index_select(0, train_tensor))
        validation_metrics = metrics(validation_prediction, target.index_select(0, validation_tensor))
        normal_error = validation_metrics["relative_error"]
        if rank > 0:
            permutation = torch.randperm(model.n_experts, generator=torch.Generator().manual_seed(seed + 1234))
            mean_pred = model(
                z.index_select(0, validation_tensor), topk.index_select(0, validation_tensor), weights.index_select(0, validation_tensor), code_mode="mean"
            )
            shuffled_pred = model(
                z.index_select(0, validation_tensor), topk.index_select(0, validation_tensor), weights.index_select(0, validation_tensor), code_mode="shuffle", permutation=permutation
            )
            zero_pred = model(
                z.index_select(0, validation_tensor), topk.index_select(0, validation_tensor), weights.index_select(0, validation_tensor), code_mode="zero"
            )
            mean_ratio = metrics(mean_pred, target.index_select(0, validation_tensor))["relative_error"] / max(normal_error, EPS)
            shuffle_ratio = metrics(shuffled_pred, target.index_select(0, validation_tensor))["relative_error"] / max(normal_error, EPS)
            zero_ratio = metrics(zero_pred, target.index_select(0, validation_tensor))["relative_error"] / max(normal_error, EPS)
        else:
            mean_ratio = shuffle_ratio = zero_ratio = None
    parameter_ratio, compute_ratio = model.parameter_ratio(top_k=8)
    return VariantResult(
        variant=f"{code_type}-k{rank}", rank=rank, code_type=code_type,
        parameter_ratio=parameter_ratio, compression_factor=1.0 / parameter_ratio,
        idealized_compute_ratio=compute_ratio,
        initial_validation_relative_error=initial_metrics["relative_error"],
        validation_relative_error=validation_metrics["relative_error"],
        validation_mean_cosine=validation_metrics["mean_cosine"],
        validation_p05_cosine=validation_metrics["p05_cosine"],
        validation_mean_token_relative_error=validation_metrics["mean_token_relative_error"],
        validation_p95_token_relative_error=validation_metrics["p95_token_relative_error"],
        train_relative_error=train_metrics["relative_error"],
        mean_code_ablation_ratio=mean_ratio,
        shuffled_code_ablation_ratio=shuffle_ratio,
        zero_code_ablation_ratio=zero_ratio,
        elapsed_seconds=time.perf_counter() - started,
    )


def make_decision(results: Sequence[VariantResult]) -> dict[str, Any]:
    best = min(results, key=lambda row: row.validation_relative_error)
    if best.validation_relative_error <= 0.10 and best.validation_mean_cosine >= 0.995:
        verdict = "REAL_DOWN_DISTILLATION_PASS"
    elif best.validation_relative_error <= 0.30 and best.validation_mean_cosine >= 0.95:
        verdict = "BORDERLINE"
    else:
        verdict = "FAIL"
    return {
        "verdict": verdict,
        "best_variant": best.variant,
        "best_validation_relative_error": best.validation_relative_error,
        "best_validation_mean_cosine": best.validation_mean_cosine,
        "rule": "PASS requires <=10% held-out aggregate-output relative error and cosine >=0.995; BORDERLINE permits <=30% and cosine >=0.95.",
    }


def self_test() -> None:
    set_seed(4)
    n, e, topk, m, d = 80, 8, 2, 12, 16
    z = torch.randn(n, topk, m)
    ids = torch.stack([torch.randperm(e)[:topk] for _ in range(n)])
    weights = torch.rand(n, topk); weights /= weights.sum(dim=-1, keepdim=True)
    teacher = ModalDown(d, m, e, 1, "neuronwise")
    with torch.no_grad():
        y = teacher(z, ids, weights)
        reconstructed = teacher(z, ids, weights)
    if torch.max(torch.abs(y - reconstructed)) > 1e-6:
        raise AssertionError("modal down algebra mismatch")
    y.sum().backward()
    if any(parameter.grad is None for parameter in teacher.parameters()):
        raise AssertionError("missing gradients")
    print("self-test passed for fused modal down algebra")


def write_outputs(output_dir: Path, payload: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "real_down_distillation.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    rows = payload["results"]
    with (output_dir / "results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader(); writer.writerows(rows)
    decision = payload["decision"]
    lines = [
        "# Test 3.0 — real OLMoE layer-7 down-branch modal distillation", "",
        f"**Decision:** **{decision['verdict']}**", "",
        "| Variant | Params | Compression | Ideal compute | Initial error | Final val error | Mean cosine | p05 cosine | Mean-code ablation | Shuffle ablation | Zero-code ablation |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        def fmt(value: Any) -> str:
            return "n/a" if value is None else f"{value:.3f}×"
        lines.append(
            f"| {row['variant']} | {row['parameter_ratio']:.3%} | {row['compression_factor']:.2f}× | "
            f"{row['idealized_compute_ratio']:.3%} | {row['initial_validation_relative_error']:.2%} | "
            f"{row['validation_relative_error']:.2%} | {row['validation_mean_cosine']:.5f} | "
            f"{row['validation_p05_cosine']:.5f} | {fmt(row['mean_code_ablation_ratio'])} | "
            f"{fmt(row['shuffled_code_ablation_ratio'])} | {fmt(row['zero_code_ablation_ratio'])} |"
        )
    lines += [
        "",
        "The target consists of real layer-7 Q4_K_M expert activations from the official OLMoE graph. Only tokens in the training split update the modal matrices and codes; validation tokens are disjoint. This test isolates `down`: teacher SwiGLU states and original router decisions are supplied to the student.",
    ]
    (output_dir / "VERDICT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("results/test-3-0/latest"))
    parser.add_argument("--steps", type=int, default=220)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=30001)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args(argv)
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    if args.self_test:
        self_test(); return 0
    if args.capture_dir is None:
        parser.error("--capture-dir is required")

    capture = load_capture(args.capture_dir)
    arrays = {
        "z": torch.from_numpy(capture["z"]),
        "topk": torch.from_numpy(capture["topk"]).long(),
        "weights": torch.from_numpy(capture["weights"]),
        "moe_out": torch.from_numpy(capture["moe_out"]),
    }
    train_ids, validation_ids = split_tokens(len(capture["z"]), args.seed)
    variants = [(0, "scalar"), (1, "scalar"), (1, "neuronwise"), (2, "neuronwise")]
    results: list[VariantResult] = []
    for index, (rank, code_type) in enumerate(variants):
        results.append(train_variant(
            arrays, train_ids, validation_ids,
            rank=rank, code_type=code_type,
            steps=args.steps, batch_size=args.batch_size,
            seed=args.seed + 1000 * index,
        ))
    payload = {
        "metadata": {
            "tokens": int(len(capture["z"])),
            "train_tokens": int(len(train_ids)),
            "validation_tokens": int(len(validation_ids)),
            "experts": 64,
            "top_k": 8,
            "d_model": int(capture["moe_out"].shape[-1]),
            "d_ff": int(capture["z"].shape[-1]),
            "capture_consistency_error": float(capture["consistency"]),
            "steps": args.steps,
            "batch_size": args.batch_size,
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
