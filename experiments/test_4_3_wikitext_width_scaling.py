#!/usr/bin/env python3
"""Test 4.3: width-scaling curve for compute-matched Modal-MoE experts.

At four Transformer widths, compare:

* conventional full-width experts;
* conventional experts narrowed to 25% arithmetic;
* Modal K=1 at 25% ideal expert arithmetic;
* conventional experts narrowed to 37.5% arithmetic;
* Modal K=2 at 37.5% ideal expert arithmetic.

All comparisons within a scale point share tokenizer, corpus, sampled batches,
seed, Transformer width/depth, router geometry (64 experts, top-8), optimizer,
and token budget.  Only expert parametrization or matched intermediate width
changes.  The test asks whether the Modal advantage is a one-width accident or
persists as dense matrix dimensions grow.
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import statistics
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch


def load_source(path: Path):
    spec = importlib.util.spec_from_file_location("modal_width_scaling_source", path)
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
        x = torch.stack([data[index:index + self.seq_len] for index in starts])
        y = torch.stack([data[index + 1:index + self.seq_len + 1] for index in starts])
        return x, y


SCALES = [
    {"name": "w64", "d_model": 64, "n_heads": 4, "d_ff": 96},
    {"name": "w96", "d_model": 96, "n_heads": 4, "d_ff": 128},
    {"name": "w128", "d_model": 128, "n_heads": 4, "d_ff": 192},
    {"name": "w160", "d_model": 160, "n_heads": 5, "d_ff": 256},
]


def train_one(
    source: Any,
    dataset: TokenArrayDataset,
    scale: dict[str, Any],
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
        d_model=int(scale["d_model"]),
        n_heads=int(scale["n_heads"]),
        n_layers=2,
        d_ff=int(d_ff),
        n_experts=64,
        top_k=8,
        steps=steps,
        eval_interval=100,
        eval_batches=20,
    )
    result, history = source.train_variant(variant, rank, seed, dataset, cfg)
    if variant == "baseline":
        if d_ff == int(scale["d_ff"]):
            label = "baseline-full"
            ratio = 1.0
        else:
            ratio = d_ff / float(scale["d_ff"])
            label = f"baseline-{ratio:.3f}x"
        result.expert_parameter_ratio = ratio
        result.idealized_expert_compute_ratio = ratio
    else:
        label = f"modal-k{rank}"
    result.variant = label
    row = asdict(result)
    row.update(
        {
            "scale": scale["name"],
            "d_model": int(scale["d_model"]),
            "n_heads": int(scale["n_heads"]),
            "n_layers": 2,
            "full_d_ff": int(scale["d_ff"]),
            "actual_d_ff": int(d_ff),
        }
    )
    return row, history


def paired(rows: Sequence[dict[str, Any]], scale: str, left: str, right: str) -> float:
    indexed = {
        (row["scale"], row["variant"]): float(row["final_validation_loss"])
        for row in rows
    }
    return indexed[(scale, left)] - indexed[(scale, right)]


def make_summary(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for scale in SCALES:
        name = scale["name"]
        selected = [row for row in rows if row["scale"] == name]
        indexed = {row["variant"]: row for row in selected}
        output.append(
            {
                "scale": name,
                "d_model": int(scale["d_model"]),
                "full_d_ff": int(scale["d_ff"]),
                "full_loss": indexed["baseline-full"]["final_validation_loss"],
                "narrow25_loss": indexed["baseline-0.250x"]["final_validation_loss"],
                "modal_k1_loss": indexed["modal-k1"]["final_validation_loss"],
                "k1_advantage_nats": (
                    indexed["baseline-0.250x"]["final_validation_loss"]
                    - indexed["modal-k1"]["final_validation_loss"]
                ),
                "modal_k1_to_full_ratio": (
                    indexed["modal-k1"]["final_validation_loss"]
                    / indexed["baseline-full"]["final_validation_loss"]
                ),
                "narrow375_loss": indexed["baseline-0.375x"]["final_validation_loss"],
                "modal_k2_loss": indexed["modal-k2"]["final_validation_loss"],
                "k2_advantage_nats": (
                    indexed["baseline-0.375x"]["final_validation_loss"]
                    - indexed["modal-k2"]["final_validation_loss"]
                ),
                "modal_k2_to_full_ratio": (
                    indexed["modal-k2"]["final_validation_loss"]
                    / indexed["baseline-full"]["final_validation_loss"]
                ),
                "full_parameters": indexed["baseline-full"]["trainable_parameters"],
                "modal_k1_parameters": indexed["modal-k1"]["trainable_parameters"],
                "modal_k2_parameters": indexed["modal-k2"]["trainable_parameters"],
            }
        )
    return output


def make_decision(summary: Sequence[dict[str, Any]]) -> dict[str, Any]:
    k1_advantages = [float(row["k1_advantage_nats"]) for row in summary]
    k2_advantages = [float(row["k2_advantage_nats"]) for row in summary]
    k1_ratios = [float(row["modal_k1_to_full_ratio"]) for row in summary]
    k2_ratios = [float(row["modal_k2_to_full_ratio"]) for row in summary]
    if (
        min(k1_advantages) > 0.0
        and min(k2_advantages) > 0.0
        and statistics.mean(k1_advantages) >= 0.03
        and statistics.mean(k2_advantages) >= 0.03
        and max(k1_ratios) <= 1.02
        and max(k2_ratios) <= 1.02
    ):
        verdict = "WIDTH_SCALING_MODAL_ADVANTAGE"
    elif statistics.mean(k1_advantages) > 0.0 and statistics.mean(k2_advantages) > 0.0:
        verdict = "WIDTH_SCALING_BORDERLINE"
    else:
        verdict = "WIDTH_SCALING_FAIL"
    return {
        "verdict": verdict,
        "k1_advantages_nats": k1_advantages,
        "k2_advantages_nats": k2_advantages,
        "k1_advantage_mean": statistics.mean(k1_advantages),
        "k1_advantage_min": min(k1_advantages),
        "k2_advantage_mean": statistics.mean(k2_advantages),
        "k2_advantage_min": min(k2_advantages),
        "worst_k1_to_full_ratio": max(k1_ratios),
        "worst_k2_to_full_ratio": max(k2_ratios),
        "rule": (
            "PASS requires K1 and K2 to beat compute-matched narrow experts at "
            "every width, mean advantage >=0.03 nat for both, and every Modal/full "
            "loss ratio <=1.02."
        ),
    }


def write_outputs(output_dir: Path, payload: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "width_scaling.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with (output_dir / "runs.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(payload["runs"][0].keys()))
        writer.writeheader(); writer.writerows(payload["runs"])
    with (output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(payload["summary"][0].keys()))
        writer.writeheader(); writer.writerows(payload["summary"])

    d = payload["decision"]
    lines = [
        "# Test 4.3 — WikiText-2 Modal-MoE width scaling",
        "",
        f"**Decision:** **{d['verdict']}**",
        "",
        "| Scale | d_model | d_ff | Full loss | Narrow25 | Modal K1 | K1 advantage | Narrow37.5 | Modal K2 | K2 advantage |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["summary"]:
        lines.append(
            f"| {row['scale']} | {row['d_model']} | {row['full_d_ff']} | "
            f"{row['full_loss']:.4f} | {row['narrow25_loss']:.4f} | "
            f"{row['modal_k1_loss']:.4f} | {row['k1_advantage_nats']:+.4f} | "
            f"{row['narrow375_loss']:.4f} | {row['modal_k2_loss']:.4f} | "
            f"{row['k2_advantage_nats']:+.4f} |"
        )
    lines += [
        "",
        f"- K1 advantage mean/worst: `{d['k1_advantage_mean']:+.4f}` / `{d['k1_advantage_min']:+.4f}` nat.",
        f"- K2 advantage mean/worst: `{d['k2_advantage_mean']:+.4f}` / `{d['k2_advantage_min']:+.4f}` nat.",
        f"- Worst Modal K1/full ratio: `{d['worst_k1_to_full_ratio']:.4f}`; K2/full: `{d['worst_k2_to_full_ratio']:.4f}`.",
        "",
        "This is a controlled small-model scaling screen. It increases dense matrix dimensions while holding 64-expert/top-8 routing, depth, corpus, token budget, and paired initialization policy fixed.",
    ]
    (output_dir / "VERDICT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=44300)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args(argv)

    source = load_source(args.source)
    source.torch.set_num_threads(args.threads)
    source.torch.set_num_interop_threads(1)
    dataset = TokenArrayDataset(args.manifest, 128)
    rows: list[dict[str, Any]] = []
    histories: dict[str, Any] = {}
    started = time.perf_counter()
    for scale_index, scale in enumerate(SCALES):
        seed = args.seed + scale_index * 100
        full = int(scale["d_ff"])
        narrow25 = max(8, int(round(full * 0.25)))
        narrow375 = max(8, int(round(full * 0.375)))
        variants = [
            ("baseline", full, None),
            ("baseline", narrow25, None),
            ("modal", full, 1),
            ("baseline", narrow375, None),
            ("modal", full, 2),
        ]
        for variant, d_ff, rank in variants:
            row, history = train_one(
                source,
                dataset,
                scale,
                seed=seed,
                steps=args.steps,
                variant=variant,
                d_ff=d_ff,
                rank=rank,
            )
            rows.append(row)
            histories[f"{scale['name']}-{row['variant']}"] = history
            print(
                f"completed {scale['name']} {row['variant']} "
                f"loss={row['final_validation_loss']:.4f}",
                flush=True,
            )
    summary = make_summary(rows)
    payload = {
        "metadata": {
            "task": "WikiText-2 raw byte-level BPE next-token LM",
            "base_seed": args.seed,
            "steps_per_variant": args.steps,
            "scales": SCALES,
            "experts": 64,
            "top_k": 8,
            "layers": 2,
            "sequence_length": 128,
            "batch_size": 8,
            "elapsed_seconds": time.perf_counter() - started,
        },
        "runs": rows,
        "summary": summary,
        "history": histories,
    }
    payload["decision"] = make_decision(summary)
    write_outputs(args.output_dir, payload)
    print((args.output_dir / "VERDICT.md").read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
