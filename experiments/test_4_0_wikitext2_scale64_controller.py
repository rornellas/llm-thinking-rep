#!/usr/bin/env python3
"""Token-level WikiText-2 replication of the scale64 dynamic controller.

This public-run trigger preserves the experiment exactly while ensuring the
workflow is exercised on the repository's free GitHub-hosted runner.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch


def load_balanced():
    path = Path(__file__).with_name("test_2_7c_balanced_calibrated_controller.py")
    spec = importlib.util.spec_from_file_location("wikitext_balanced_controller", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


balanced = load_balanced()
base = balanced.base
original_load_module = base.load_module
original_write_outputs = base.write_outputs


class TokenArrayDataset:
    def __init__(self, manifest_text: str, seq_len: int) -> None:
        manifest = json.loads(manifest_text)
        self.train = torch.from_numpy(np.load(manifest["train_path"]).astype(np.int64, copy=False))
        self.validation = torch.from_numpy(np.load(manifest["validation_path"]).astype(np.int64, copy=False))
        self.seq_len = seq_len
        self.vocab = range(int(manifest["vocab_size"]))
        self.manifest = manifest
        if len(self.train) <= seq_len + 1 or len(self.validation) <= seq_len + 1:
            raise ValueError((len(self.train), len(self.validation), seq_len))

    def batch(self, split: str, batch_size: int, generator: torch.Generator):
        data = self.train if split == "train" else self.validation
        starts = torch.randint(0, len(data) - self.seq_len - 1, (batch_size,), generator=generator)
        x = torch.stack([data[index:index + self.seq_len] for index in starts])
        y = torch.stack([data[index + 1:index + self.seq_len + 1] for index in starts])
        return x, y


def token_load_module(name: str, path: Path):
    module = original_load_module(name, path)
    if name == "controller_base_source":
        original_config = module.Config

        class WikiScale64Config(original_config):
            def __init__(self, *args, **kwargs):
                kwargs["seq_len"] = 128
                kwargs["batch_size"] = 8
                kwargs["d_model"] = 96
                kwargs["n_heads"] = 4
                kwargs["n_layers"] = 2
                kwargs["d_ff"] = 128
                kwargs["n_experts"] = 64
                kwargs["top_k"] = 8
                super().__init__(*args, **kwargs)

        module.Config = WikiScale64Config
        module.CharDataset = TokenArrayDataset
    return module


def token_write_outputs(output_dir: Path, payload: dict):
    payload.setdefault("metadata", {}).update({
        "task": "WikiText-2 raw BPE next-token language modeling",
        "tokenizer": "byte-level BPE trained on the WikiText-2 training split",
        "vocab_size": 2048,
        "sequence_length": 128,
        "batch_size": 8,
        "n_experts": 64,
        "top_k": 8,
        "domain_note": "Official WikiText-2 train and validation splits; threshold selection remains confined to training calibration batches.",
    })
    original_write_outputs(output_dir, payload)


base.load_module = token_load_module
base.write_outputs = token_write_outputs


if __name__ == "__main__":
    raise SystemExit(base.main())
