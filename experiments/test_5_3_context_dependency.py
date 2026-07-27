#!/usr/bin/env python3
"""Test 5.3: verify that compact context models actually use their context.

The close BpB values across Test 5.1/5.2 representations could mean either:

1. all compact representations preserve the useful context; or
2. the predictor mostly learns marginal next-byte frequencies and ignores its
   input.

This experiment trains fixed-mean32 and learned-resampler16 models on the same
next-16-byte task, then evaluates held-out contexts under interventions:

* correct context;
* context rolled from another example (primary causal control);
* bytes shuffled inside each example;
* zero context;
* only the final 64 or 32 bytes retained;
* reversed context.

It also reports the train-frequency unconditional byte baseline.  Three paired
seeds are used.  A context representation is informative only when mismatching
or destroying context causes a statistically material BpB increase.
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn.functional as F


def load_resampler(path: Path):
    spec = importlib.util.spec_from_file_location("context_dependency_resampler", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@dataclass
class Row:
    seed: int
    variant: str
    intervention: str
    bits_per_byte: float
    delta_vs_correct_bpb: float
    paired_lcb95_delta_bpb: float
    context_bytes_per_second: float


def build_windows(data: np.ndarray, starts: np.ndarray, cfg: Any):
    windows = np.stack(
        [
            data[
                int(start):int(start) + cfg.context_bytes + cfg.horizon_bytes
            ]
            for start in starts
        ]
    ).astype(np.int64, copy=False)
    return (
        torch.from_numpy(windows[:, :cfg.context_bytes]),
        torch.from_numpy(windows[:, cfg.context_bytes:]),
    )


def bootstrap_lcb(values: np.ndarray, seed: int, samples: int = 5000) -> float:
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(samples, len(values)))
    means = values[indices].mean(axis=1)
    return float(np.quantile(means, 0.05))


def intervene(
    contexts: torch.Tensor,
    intervention: str,
    *,
    seed: int,
) -> torch.Tensor:
    if intervention == "correct":
        return contexts
    if intervention == "rolled":
        return torch.roll(contexts, shifts=1, dims=0)
    if intervention == "zero":
        return torch.zeros_like(contexts)
    if intervention == "reverse":
        return torch.flip(contexts, dims=(1,))
    if intervention == "last64":
        modified = torch.zeros_like(contexts)
        modified[:, -64:] = contexts[:, -64:]
        return modified
    if intervention == "last32":
        modified = torch.zeros_like(contexts)
        modified[:, -32:] = contexts[:, -32:]
        return modified
    if intervention == "shuffle":
        generator = torch.Generator().manual_seed(seed)
        # Each example gets an independent permutation, preserving its byte
        # histogram while destroying order and local structure.
        permutations = torch.stack(
            [
                torch.randperm(contexts.shape[1], generator=generator)
                for _ in range(contexts.shape[0])
            ]
        )
        return torch.gather(contexts, 1, permutations)
    raise ValueError(intervention)


@torch.no_grad()
def evaluate_intervention(
    model: Any,
    contexts: torch.Tensor,
    targets: torch.Tensor,
    cfg: Any,
    intervention: str,
    *,
    seed: int,
) -> tuple[float, np.ndarray, float]:
    model.eval()
    started = time.perf_counter()
    losses: list[np.ndarray] = []
    total_context_bytes = 0
    for offset in range(0, len(contexts), cfg.batch_size):
        x = contexts[offset:offset + cfg.batch_size]
        y = targets[offset:offset + cfg.batch_size]
        x = intervene(x, intervention, seed=seed + offset)
        logits = model(x)
        per_byte = F.cross_entropy(
            logits.reshape(-1, 256),
            y.reshape(-1),
            reduction="none",
        ).reshape(len(x), cfg.horizon_bytes)
        losses.append(per_byte.mean(dim=1).cpu().numpy())
        total_context_bytes += int(x.numel())
    elapsed = time.perf_counter() - started
    values = np.concatenate(losses)
    return (
        float(np.mean(values) / math.log(2.0)),
        values / math.log(2.0),
        total_context_bytes / max(elapsed, 1e-12),
    )


def unconditional_bpb(
    train: np.ndarray,
    targets: torch.Tensor,
) -> float:
    counts = np.bincount(train.astype(np.int64), minlength=256).astype(np.float64)
    probabilities = (counts + 0.5) / (counts.sum() + 0.5 * 256)
    values = targets.numpy().reshape(-1)
    return float(np.mean(-np.log2(probabilities[values])))


def train_model(
    module: Any,
    variant: str,
    seed: int,
    train_contexts: torch.Tensor,
    train_targets: torch.Tensor,
    cfg: Any,
):
    module.set_seed(seed)
    model = module.ContextPredictor(cfg, variant)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )
    for step in range(1, cfg.steps + 1):
        offset = (step - 1) * cfg.batch_size
        x = train_contexts[offset:offset + cfg.batch_size]
        y = train_targets[offset:offset + cfg.batch_size]
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, 256), y.reshape(-1))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()
        if step == 1 or step % 300 == 0 or step == cfg.steps:
            print(
                f"seed={seed} variant={variant} step={step}/{cfg.steps} "
                f"train-bpb={float(loss.detach()) / math.log(2.0):.4f}",
                flush=True,
            )
    return model


def aggregate(rows: Sequence[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    variants = sorted({row["variant"] for row in rows})
    interventions = [
        "correct",
        "rolled",
        "shuffle",
        "zero",
        "last64",
        "last32",
        "reverse",
    ]
    summary: list[dict[str, Any]] = []
    for variant in variants:
        for intervention in interventions:
            selected = [
                row for row in rows
                if row["variant"] == variant
                and row["intervention"] == intervention
            ]
            summary.append(
                {
                    "variant": variant,
                    "intervention": intervention,
                    "runs": len(selected),
                    "bits_per_byte_mean": statistics.mean(
                        row["bits_per_byte"] for row in selected
                    ),
                    "bits_per_byte_std": statistics.pstdev(
                        row["bits_per_byte"] for row in selected
                    ),
                    "delta_vs_correct_mean": statistics.mean(
                        row["delta_vs_correct_bpb"] for row in selected
                    ),
                    "worst_paired_lcb95": min(
                        row["paired_lcb95_delta_bpb"] for row in selected
                    ),
                    "context_bytes_per_second_mean": statistics.mean(
                        row["context_bytes_per_second"] for row in selected
                    ),
                }
            )

    indexed = {
        (row["variant"], row["intervention"]): row
        for row in summary
    }
    decisions: dict[str, Any] = {}
    for variant in variants:
        rolled = indexed[(variant, "rolled")]
        shuffle = indexed[(variant, "shuffle")]
        last32 = indexed[(variant, "last32")]
        if (
            rolled["delta_vs_correct_mean"] >= 0.03
            and rolled["worst_paired_lcb95"] > 0.0
            and shuffle["delta_vs_correct_mean"] >= 0.02
        ):
            verdict = "CONTEXT_DEPENDENT"
        elif rolled["delta_vs_correct_mean"] >= 0.01:
            verdict = "WEAK_CONTEXT_SIGNAL"
        else:
            verdict = "CONTEXT_NOT_ESTABLISHED"
        decisions[variant] = {
            "verdict": verdict,
            "rolled_delta_mean_bpb": rolled["delta_vs_correct_mean"],
            "rolled_worst_lcb95_bpb": rolled["worst_paired_lcb95"],
            "shuffle_delta_mean_bpb": shuffle["delta_vs_correct_mean"],
            "last32_delta_mean_bpb": last32["delta_vs_correct_mean"],
        }
    overall = (
        "COMPACT_CONTEXT_VALIDATED"
        if all(value["verdict"] == "CONTEXT_DEPENDENT" for value in decisions.values())
        else "COMPACT_CONTEXT_INCONCLUSIVE"
    )
    return summary, {
        "verdict": overall,
        "variants": decisions,
        "rule": (
            "A representation is context-dependent when rolled context costs "
            ">=0.03 BpB with positive paired LCB95 in every seed and shuffled "
            "context costs >=0.02 BpB on average."
        ),
    }


def write_outputs(output_dir: Path, payload: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "context_dependency.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with (output_dir / "runs.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(payload["runs"][0].keys()))
        writer.writeheader(); writer.writerows(payload["runs"])
    with (output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(payload["summary"][0].keys()))
        writer.writeheader(); writer.writerows(payload["summary"])

    lines = [
        "# Test 5.3 — causal context dependency of compact units",
        "",
        f"**Decision:** **{payload['decision']['verdict']}**",
        "",
        f"Unconditional train-frequency baseline: `{payload['metadata']['unconditional_validation_bpb']:.4f}` BpB.",
        "",
        "| Representation | Intervention | BpB | Δ vs correct | Worst paired LCB95 | Context bytes/s |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in payload["summary"]:
        lines.append(
            f"| {row['variant']} | {row['intervention']} | "
            f"{row['bits_per_byte_mean']:.4f} ± {row['bits_per_byte_std']:.4f} | "
            f"{row['delta_vs_correct_mean']:+.4f} | "
            f"{row['worst_paired_lcb95']:+.4f} | "
            f"{row['context_bytes_per_second_mean']:.0f} |"
        )
    lines += ["", "## Per-representation decision"]
    for variant, decision in payload["decision"]["variants"].items():
        lines.append(
            f"- **{variant}: {decision['verdict']}** — rolled `{decision['rolled_delta_mean_bpb']:+.4f}` BpB, "
            f"shuffle `{decision['shuffle_delta_mean_bpb']:+.4f}`, last32 `{decision['last32_delta_mean_bpb']:+.4f}`, "
            f"rolled worst LCB95 `{decision['rolled_worst_lcb95_bpb']:+.4f}`."
        )
    lines += [
        "",
        "All interventions preserve the held-out targets. Rolled context is the primary causal control because it remains in-distribution while breaking the context-target relationship.",
    ]
    (output_dir / "VERDICT.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def self_test(module: Any) -> None:
    values = torch.arange(24).reshape(3, 8)
    rolled = intervene(values, "rolled", seed=1)
    if torch.equal(values, rolled) or not torch.equal(rolled[0], values[-1]):
        raise AssertionError(rolled)
    shuffled = intervene(values, "shuffle", seed=2)
    if not torch.equal(torch.sort(values, dim=1).values, torch.sort(shuffled, dim=1).values):
        raise AssertionError(shuffled)
    print("self-test passed")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resampler-source", type=Path, required=True)
    parser.add_argument("--train-bytes", type=Path, required=False)
    parser.add_argument("--validation-bytes", type=Path, required=False)
    parser.add_argument("--output-dir", type=Path, required=False)
    parser.add_argument("--seeds", default="58301,59402,60503")
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args(argv)

    module = load_resampler(args.resampler_source)
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    if args.self_test:
        self_test(module); return 0
    if args.train_bytes is None or args.validation_bytes is None or args.output_dir is None:
        parser.error("train, validation, and output paths are required")

    cfg = module.Config(steps=args.steps)
    train = np.frombuffer(args.train_bytes.read_bytes(), dtype=np.uint8)
    validation = np.frombuffer(args.validation_bytes.read_bytes(), dtype=np.uint8)
    seeds = [int(value) for value in args.seeds.split(",") if value.strip()]
    interventions = (
        "correct",
        "rolled",
        "shuffle",
        "zero",
        "last64",
        "last32",
        "reverse",
    )
    variants = ("fixed-mean32", "resampler16")
    rows: list[dict[str, Any]] = []
    unconditional_values: list[float] = []

    for seed in seeds:
        train_rng = np.random.default_rng(seed + 1)
        validation_rng = np.random.default_rng(seed + 2)
        train_starts = train_rng.integers(
            0,
            len(train) - cfg.context_bytes - cfg.horizon_bytes,
            size=cfg.steps * cfg.batch_size,
            endpoint=False,
            dtype=np.int64,
        )
        validation_starts = validation_rng.integers(
            0,
            len(validation) - cfg.context_bytes - cfg.horizon_bytes,
            size=cfg.eval_batches * cfg.batch_size,
            endpoint=False,
            dtype=np.int64,
        )
        train_contexts, train_targets = build_windows(train, train_starts, cfg)
        validation_contexts, validation_targets = build_windows(
            validation, validation_starts, cfg
        )
        unconditional_values.append(
            unconditional_bpb(train, validation_targets)
        )
        for variant in variants:
            model = train_model(
                module,
                variant,
                seed,
                train_contexts,
                train_targets,
                cfg,
            )
            correct_bpb, correct_values, correct_speed = evaluate_intervention(
                model,
                validation_contexts,
                validation_targets,
                cfg,
                "correct",
                seed=seed,
            )
            rows.append(
                asdict(
                    Row(
                        seed=seed,
                        variant=variant,
                        intervention="correct",
                        bits_per_byte=correct_bpb,
                        delta_vs_correct_bpb=0.0,
                        paired_lcb95_delta_bpb=0.0,
                        context_bytes_per_second=correct_speed,
                    )
                )
            )
            for index, intervention_name in enumerate(interventions[1:], start=1):
                bpb, values, speed = evaluate_intervention(
                    model,
                    validation_contexts,
                    validation_targets,
                    cfg,
                    intervention_name,
                    seed=seed + index * 100,
                )
                differences = values - correct_values
                rows.append(
                    asdict(
                        Row(
                            seed=seed,
                            variant=variant,
                            intervention=intervention_name,
                            bits_per_byte=bpb,
                            delta_vs_correct_bpb=float(np.mean(differences)),
                            paired_lcb95_delta_bpb=bootstrap_lcb(
                                differences,
                                seed + index * 1000,
                            ),
                            context_bytes_per_second=speed,
                        )
                    )
                )
            del model

    summary, decision = aggregate(rows)
    payload = {
        "metadata": {
            "task": "predict next 16 raw UTF-8 bytes from a 128-byte context",
            "dataset": "WikiText-2 raw UTF-8 bytes",
            "seeds": seeds,
            "steps_per_model": cfg.steps,
            "evaluation_examples_per_seed": cfg.eval_batches * cfg.batch_size,
            "unconditional_validation_bpb": statistics.mean(unconditional_values),
            "unconditional_validation_bpb_by_seed": unconditional_values,
            "primary_control": "roll contexts by one example while retaining targets",
        },
        "runs": rows,
        "summary": summary,
        "decision": decision,
    }
    write_outputs(args.output_dir, payload)
    print((args.output_dir / "VERDICT.md").read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
