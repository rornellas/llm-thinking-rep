#!/usr/bin/env python3
"""Prepare a deterministic byte-level BPE view of WikiText-2."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from datasets import load_dataset
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--stats", type=Path, required=True)
    parser.add_argument("--vocab-size", type=int, default=2048)
    args = parser.parse_args()

    args.root.mkdir(parents=True, exist_ok=True)
    args.stats.parent.mkdir(parents=True, exist_ok=True)
    dataset = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1")
    train_lines = [row["text"] for row in dataset["train"] if row["text"].strip()]
    validation_lines = [row["text"] for row in dataset["validation"] if row["text"].strip()]

    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.train_from_iterator(
        train_lines,
        trainer=trainers.BpeTrainer(
            vocab_size=args.vocab_size,
            min_frequency=2,
            special_tokens=["<unk>"],
            show_progress=False,
        ),
    )
    tokenizer.save(str(args.root / "tokenizer.json"))

    def encode(lines: list[str]) -> np.ndarray:
        ids: list[int] = []
        for line in lines:
            ids.extend(tokenizer.encode(line + "\n").ids)
        return np.asarray(ids, dtype=np.int32)

    train = encode(train_lines)
    validation = encode(validation_lines)
    np.save(args.root / "train.npy", train)
    np.save(args.root / "validation.npy", validation)
    manifest = {
        "train_path": str(args.root / "train.npy"),
        "validation_path": str(args.root / "validation.npy"),
        "vocab_size": tokenizer.get_vocab_size(),
        "dataset": "Salesforce/wikitext",
        "subset": "wikitext-2-raw-v1",
    }
    (args.root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    stats = {
        **manifest,
        "train_documents": len(train_lines),
        "validation_documents": len(validation_lines),
        "train_tokens": int(train.size),
        "validation_tokens": int(validation.size),
    }
    args.stats.write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(stats, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
