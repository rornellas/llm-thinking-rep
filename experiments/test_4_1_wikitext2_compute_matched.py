#!/usr/bin/env python3
"""WikiText-2 compute-matched baselines for the 64-expert/top-8 Modal MoE.

This separates the value of shared modal matrices from the simpler alternative
of shrinking every conventional expert until its arithmetic matches K=1/K=2.
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch


def load_source(path: Path):
    spec = importlib.util.spec_from_file_location("wikitext_compute_source", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class TokenArrayDataset:
    def __init__(self, manifest_path: Path, seq_len: int) -> None:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.train = torch.from_numpy(np.load(manifest["train_path"]).astype(np.int64, copy=False))
        self.validation = torch.from_numpy(np.load(manifest["validation_path"]).astype(np.int64, copy=False))
        self.vocab = range(int(manifest["vocab_size"]))
        self.seq_len = seq_len

    def batch(self, split: str, batch_size: int, generator: torch.Generator):
        data = self.train if split == "train" else self.validation
        starts = torch.randint(0, len(data) - self.seq_len - 1, (batch_size,), generator=generator)
        x = torch.stack([data[i:i + self.seq_len] for i in starts])
        y = torch.stack([data[i + 1:i + self.seq_len + 1] for i in starts])
        return x, y


def train(source: Any, dataset: TokenArrayDataset, *, seed: int, steps: int, variant: str, d_ff: int, rank: int | None):
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
    result, history = source.train_variant(variant, rank, seed, dataset, cfg)
    if variant == "baseline":
        result.variant = "baseline-full" if d_ff == 128 else f"baseline-dff{d_ff}"
        result.expert_parameter_ratio = d_ff / 128.0
        result.idealized_expert_compute_ratio = d_ff / 128.0
    else:
        result.variant = f"modal-k{rank}"
    return result, history


def write_outputs(output_dir: Path, payload: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "compute_matched.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    with (output_dir / "results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(payload["results"][0].keys()))
        writer.writeheader(); writer.writerows(payload["results"])
    indexed = {row["variant"]: row for row in payload["results"]}
    full = indexed["baseline-full"]["final_validation_loss"]
    lines = [
        "# Test 4.1 — WikiText-2 compute-matched conventional experts",
        "",
        f"**Decision:** **{payload['decision']['verdict']}**",
        "",
        "| Variant | Expert params | Ideal expert compute | Validation loss | Loss/full |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in payload["results"]:
        lines.append(
            f"| {row['variant']} | {row['expert_parameter_ratio']:.3%} | "
            f"{row['idealized_expert_compute_ratio']:.3%} | {row['final_validation_loss']:.4f} | "
            f"{row['final_validation_loss']/full:.3f}× |"
        )
    lines += [
        "",
        f"- Modal K1 advantage over conventional d_ff=32 at matched 25% expert arithmetic: `{payload['decision']['k1_loss_advantage']:+.4f}` nat.",
        f"- Modal K2 advantage over conventional d_ff=48 at matched 37.5% expert arithmetic: `{payload['decision']['k2_loss_advantage']:+.4f}` nat.",
        "",
        "All variants use the same tokenizer, corpus, transformer width, 64 experts, top-8 router geometry, training steps, and seed.",
    ]
    (output_dir / "VERDICT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=350)
    parser.add_argument("--seed", type=int, default=41410)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()

    source = load_source(args.source)
    source.torch.set_num_threads(args.threads)
    source.torch.set_num_interop_threads(1)
    dataset = TokenArrayDataset(args.manifest, 128)
    variants = [
        ("baseline", 128, None),
        ("baseline", 32, None),
        ("baseline", 48, None),
        ("modal", 128, 1),
        ("modal", 128, 2),
    ]
    results = []
    histories = {}
    for variant, d_ff, rank in variants:
        result, history = train(
            source, dataset, seed=args.seed, steps=args.steps,
            variant=variant, d_ff=d_ff, rank=rank,
        )
        results.append(asdict(result))
        histories[result.variant] = history

    indexed = {row["variant"]: row for row in results}
    full = indexed["baseline-full"]["final_validation_loss"]
    k1_adv = indexed["baseline-dff32"]["final_validation_loss"] - indexed["modal-k1"]["final_validation_loss"]
    k2_adv = indexed["baseline-dff48"]["final_validation_loss"] - indexed["modal-k2"]["final_validation_loss"]
    modal_close = indexed["modal-k1"]["final_validation_loss"] / full <= 1.06
    if k1_adv >= 0.0 and k2_adv >= 0.0 and modal_close:
        verdict = "MODAL_BEATS_MATCHED_NARROW"
    elif k1_adv >= -0.02 and k2_adv >= -0.02:
        verdict = "BORDERLINE"
    else:
        verdict = "FAIL"
    payload = {
        "metadata": {
            "task": "WikiText-2 raw byte-level BPE next-token LM",
            "seed": args.seed,
            "steps": args.steps,
            "experts": 64,
            "top_k": 8,
            "d_model": 96,
            "sequence_length": 128,
        },
        "decision": {
            "verdict": verdict,
            "k1_loss_advantage": k1_adv,
            "k2_loss_advantage": k2_adv,
        },
        "results": results,
        "history": histories,
    }
    write_outputs(args.output_dir, payload)
    print((args.output_dir / "VERDICT.md").read_text())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
