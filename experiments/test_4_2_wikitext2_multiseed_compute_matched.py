#!/usr/bin/env python3
"""Three-seed WikiText-2 compute-matched Modal-MoE replication.

For each seed the following models use the same tokenizer, corpus, transformer
width, router geometry, number of optimization steps, and random seed:

* conventional full-width experts (d_ff=128);
* conventional d_ff=32 experts (25% expert arithmetic);
* conventional d_ff=48 experts (37.5% expert arithmetic);
* Modal K=1 (25% ideal expert arithmetic);
* Modal K=2 (37.5% ideal expert arithmetic).

The comparison asks whether sharing full-rank matrices across experts is more
parameter- and compute-efficient than simply narrowing every conventional
expert.
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
from dataclasses import asdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch


def load_source(path: Path):
    spec = importlib.util.spec_from_file_location("wikitext_multiseed_source", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class TokenArrayDataset:
    def __init__(self, manifest_path: Path, seq_len: int) -> None:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.train = torch.from_numpy(
            np.load(manifest["train_path"]).astype(np.int64, copy=False)
        )
        self.validation = torch.from_numpy(
            np.load(manifest["validation_path"]).astype(np.int64, copy=False)
        )
        self.vocab = range(int(manifest["vocab_size"]))
        self.seq_len = seq_len
        self.manifest = manifest

    def batch(self, split: str, batch_size: int, generator: torch.Generator):
        data = self.train if split == "train" else self.validation
        starts = torch.randint(
            0,
            len(data) - self.seq_len - 1,
            (batch_size,),
            generator=generator,
        )
        x = torch.stack([data[i:i + self.seq_len] for i in starts])
        y = torch.stack([data[i + 1:i + self.seq_len + 1] for i in starts])
        return x, y


def train_one(
    source: Any,
    dataset: TokenArrayDataset,
    *,
    seed: int,
    steps: int,
    variant: str,
    d_ff: int,
    rank: int | None,
):
    cfg = source.Config(
        seq_len=128,
        batch_size=8,
        d_model=96,
        n_heads=4,
        n_layers=2,
        d_ff=d_ff,
        n_experts=64,
        top_k=8,
        steps=steps,
        eval_interval=100,
        eval_batches=20,
    )
    result, history = source.train_variant(
        variant, rank, seed, dataset, cfg
    )
    if variant == "baseline":
        result.variant = (
            "baseline-full" if d_ff == 128 else f"baseline-dff{d_ff}"
        )
        result.expert_parameter_ratio = d_ff / 128.0
        result.idealized_expert_compute_ratio = d_ff / 128.0
    else:
        result.variant = f"modal-k{rank}"
    return result, history


def aggregate(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    variants = sorted({row["variant"] for row in rows})
    result: list[dict[str, Any]] = []
    for variant in variants:
        selected = [row for row in rows if row["variant"] == variant]
        losses = [float(row["final_validation_loss"]) for row in selected]
        result.append(
            {
                "variant": variant,
                "runs": len(selected),
                "expert_parameter_ratio": float(selected[0]["expert_parameter_ratio"]),
                "idealized_expert_compute_ratio": float(selected[0]["idealized_expert_compute_ratio"]),
                "validation_loss_mean": statistics.mean(losses),
                "validation_loss_std": statistics.pstdev(losses),
                "validation_loss_min": min(losses),
                "validation_loss_max": max(losses),
                "elapsed_seconds_mean": statistics.mean(
                    float(row["elapsed_seconds"]) for row in selected
                ),
            }
        )
    return result


def paired_differences(
    rows: Sequence[dict[str, Any]], left: str, right: str
) -> list[float]:
    by_key = {
        (int(row["seed"]), row["variant"]): float(
            row["final_validation_loss"]
        )
        for row in rows
    }
    seeds = sorted({int(row["seed"]) for row in rows})
    return [by_key[(seed, left)] - by_key[(seed, right)] for seed in seeds]


def make_decision(
    rows: Sequence[dict[str, Any]], summary: Sequence[dict[str, Any]]
) -> dict[str, Any]:
    k1_advantages = paired_differences(rows, "baseline-dff32", "modal-k1")
    k2_advantages = paired_differences(rows, "baseline-dff48", "modal-k2")
    full_losses = {
        int(row["seed"]): float(row["final_validation_loss"])
        for row in rows
        if row["variant"] == "baseline-full"
    }
    k1_ratios = [
        float(row["final_validation_loss"]) / full_losses[int(row["seed"])]
        for row in rows
        if row["variant"] == "modal-k1"
    ]
    if (
        min(k1_advantages) > 0.0
        and min(k2_advantages) > 0.0
        and statistics.mean(k1_advantages) >= 0.03
        and statistics.mean(k2_advantages) >= 0.03
        and max(k1_ratios) <= 1.02
    ):
        verdict = "MULTISEED_MODAL_ADVANTAGE"
    elif (
        statistics.mean(k1_advantages) > 0.0
        and statistics.mean(k2_advantages) > 0.0
    ):
        verdict = "MULTISEED_BORDERLINE"
    else:
        verdict = "MULTISEED_FAIL"
    return {
        "verdict": verdict,
        "k1_advantages_nats": k1_advantages,
        "k2_advantages_nats": k2_advantages,
        "k1_advantage_mean": statistics.mean(k1_advantages),
        "k1_advantage_min": min(k1_advantages),
        "k2_advantage_mean": statistics.mean(k2_advantages),
        "k2_advantage_min": min(k2_advantages),
        "modal_k1_loss_to_full_ratios": k1_ratios,
        "rule": "PASS requires Modal K1 and K2 to beat their compute-matched narrow conventional baseline in every seed, mean advantage >=0.03 nat for both, and worst K1/full loss ratio <=1.02.",
    }


def write_outputs(output_dir: Path, payload: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "multiseed_compute_matched.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with (output_dir / "runs.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(payload["runs"][0].keys())
        )
        writer.writeheader(); writer.writerows(payload["runs"])
    with (output_dir / "summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(payload["summary"][0].keys())
        )
        writer.writeheader(); writer.writerows(payload["summary"])

    d = payload["decision"]
    lines = [
        "# Test 4.2 — WikiText-2 multi-seed compute-matched replication",
        "",
        f"**Decision:** **{d['verdict']}**",
        "",
        "| Variant | Runs | Expert params | Ideal expert compute | Validation loss | Range |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in payload["summary"]:
        lines.append(
            f"| {row['variant']} | {row['runs']} | "
            f"{row['expert_parameter_ratio']:.3%} | "
            f"{row['idealized_expert_compute_ratio']:.3%} | "
            f"{row['validation_loss_mean']:.4f} ± {row['validation_loss_std']:.4f} | "
            f"{row['validation_loss_min']:.4f}–{row['validation_loss_max']:.4f} |"
        )
    lines += [
        "",
        f"- K1 paired advantages over d_ff=32: `{[round(x, 4) for x in d['k1_advantages_nats']]}`; mean `{d['k1_advantage_mean']:+.4f}` nat; worst `{d['k1_advantage_min']:+.4f}`.",
        f"- K2 paired advantages over d_ff=48: `{[round(x, 4) for x in d['k2_advantages_nats']]}`; mean `{d['k2_advantage_mean']:+.4f}` nat; worst `{d['k2_advantage_min']:+.4f}`.",
        f"- Modal K1/full loss ratios: `{[round(x, 4) for x in d['modal_k1_loss_to_full_ratios']]}`.",
        "",
        "Each paired comparison shares seed, tokenizer, batches, model width, 64-expert/top-8 geometry, and optimization budget. Only the expert parametrization and matched intermediate width differ.",
    ]
    (output_dir / "VERDICT.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seeds", default="42420,43430,44440")
    parser.add_argument("--steps", type=int, default=350)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args(argv)

    source = load_source(args.source)
    source.torch.set_num_threads(args.threads)
    source.torch.set_num_interop_threads(1)
    dataset = TokenArrayDataset(args.manifest, 128)
    seeds = [int(value) for value in args.seeds.split(",") if value.strip()]
    variants = [
        ("baseline", 128, None),
        ("baseline", 32, None),
        ("baseline", 48, None),
        ("modal", 128, 1),
        ("modal", 128, 2),
    ]
    started = time.perf_counter()
    rows: list[dict[str, Any]] = []
    histories: dict[str, Any] = {}
    for seed in seeds:
        for variant, d_ff, rank in variants:
            result, history = train_one(
                source,
                dataset,
                seed=seed,
                steps=args.steps,
                variant=variant,
                d_ff=d_ff,
                rank=rank,
            )
            row = asdict(result)
            rows.append(row)
            histories[f"{result.variant}-seed{seed}"] = history
            print(
                f"completed seed={seed} variant={result.variant} "
                f"validation={result.final_validation_loss:.4f}",
                flush=True,
            )
    summary = aggregate(rows)
    payload = {
        "metadata": {
            "task": "WikiText-2 raw byte-level BPE next-token LM",
            "seeds": seeds,
            "steps": args.steps,
            "experts": 64,
            "top_k": 8,
            "d_model": 96,
            "sequence_length": 128,
            "elapsed_seconds": time.perf_counter() - started,
        },
        "decision": make_decision(rows, summary),
        "runs": rows,
        "summary": summary,
        "history": histories,
    }
    write_outputs(args.output_dir, payload)
    print((args.output_dir / "VERDICT.md").read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
