from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from pre_qwen_certification.reality_gate_data import (
    ArrayTokenCorpus,
    PreparedTokenSplit,
    documents_from_prepared_split,
    load_prepared_arrays,
)


def prepared_fixture() -> PreparedTokenSplit:
    first = np.arange(700, dtype=np.int32) % 127
    second = (np.arange(900, dtype=np.int32) + 17) % 127
    return PreparedTokenSplit(
        tokens=np.concatenate((first, second)),
        offsets=np.asarray([0, len(first), len(first) + len(second)], dtype=np.int64),
    )


def test_array_token_corpus_preserves_article_boundaries() -> None:
    documents = documents_from_prepared_split(
        prepared_fixture(),
        prefix="unit",
        domain="unit",
        maximum_document_tokens=512,
        minimum_document_tokens=34,
    )
    assert len(documents) == 4
    assert documents[0].document_id == documents[1].document_id == "unit-article-0000"
    assert documents[2].document_id == documents[3].document_id == "unit-article-0001"
    assert documents[1].source_offset == 512
    corpus = ArrayTokenCorpus(
        {"train": documents, "calibration": documents[:2]},
        seq_len=32,
        vocab_size=127,
    )
    tokens, targets = corpus.sample_batch(
        "train", 5, torch.Generator().manual_seed(17)
    )
    assert tuple(tokens.shape) == (5, 32)
    assert tuple(targets.shape) == (5, 32)
    assert torch.equal(tokens[:, 1:], targets[:, :-1])
    windows = corpus.fixed_windows("calibration", windows_per_document=3, seed=19)
    assert len(windows) == 6
    assert all(len(window.inputs) == 32 for window in windows)


def test_prepared_loader_opens_only_requested_splits(tmp_path: Path) -> None:
    sha: dict[str, str] = {}
    for split in ("train", "validation", "test", "ood"):
        token_path = tmp_path / f"{split}-tokens.npy"
        offset_path = tmp_path / f"{split}-offsets.npy"
        np.save(token_path, np.arange(128, dtype=np.int32) % 64)
        np.save(offset_path, np.asarray([0, 64, 128], dtype=np.int64))
        for path in (token_path, offset_path):
            sha[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    tokenizer = tmp_path / "tokenizer.json"
    tokenizer.write_text("{}", encoding="utf-8")
    sha[tokenizer.name] = hashlib.sha256(tokenizer.read_bytes()).hexdigest()
    (tmp_path / "manifest.json").write_text(
        json.dumps({"vocab_size": 64, "sha256": sha}), encoding="utf-8"
    )
    arrays, _ = load_prepared_arrays(tmp_path, splits=("train", "validation"))
    assert set(arrays) == {"train", "validation"}
    arrays["train"].validate()
