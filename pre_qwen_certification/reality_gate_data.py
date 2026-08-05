"""Prepared article-preserving token corpus helpers for Reality Gate 1A."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch

from .tiny_lm import EvaluationWindow


@dataclass(frozen=True)
class TokenDocument:
    document_id: str
    domain: str
    tokens: torch.Tensor
    source_offset: int = 0


@dataclass(frozen=True)
class PreparedTokenSplit:
    """Packed token ids plus article offsets from the source dataset."""

    tokens: np.ndarray
    offsets: np.ndarray

    def validate(self) -> None:
        if self.tokens.ndim != 1 or self.offsets.ndim != 1:
            raise ValueError("prepared tokens and offsets must be one-dimensional")
        if len(self.offsets) < 2 or int(self.offsets[0]) != 0:
            raise ValueError("prepared offsets must start at zero and contain documents")
        if int(self.offsets[-1]) != len(self.tokens):
            raise ValueError("prepared offsets do not cover the packed token array")
        if np.any(np.diff(self.offsets) < 0):
            raise ValueError("prepared offsets are not monotonic")


class ArrayTokenCorpus:
    """Article-bounded token documents with the TinyLM corpus interface."""

    def __init__(
        self,
        splits: Mapping[str, Sequence[TokenDocument]],
        *,
        seq_len: int,
        vocab_size: int,
    ) -> None:
        if seq_len <= 0 or vocab_size <= 8:
            raise ValueError("invalid corpus dimensions")
        self.seq_len = int(seq_len)
        self._vocab_size = int(vocab_size)
        self.splits: dict[str, list[TokenDocument]] = {}
        for split, documents in splits.items():
            current: list[TokenDocument] = []
            for document in documents:
                tokens = document.tokens.detach().clone().long().cpu()
                if tokens.ndim != 1:
                    raise ValueError("document tokens must be one-dimensional")
                if len(tokens) <= self.seq_len + 1:
                    raise ValueError(f"document {document.document_id} is too short")
                if int(tokens.min()) < 0 or int(tokens.max()) >= self._vocab_size:
                    raise ValueError(f"document {document.document_id} contains invalid token ids")
                current.append(
                    TokenDocument(
                        document.document_id, document.domain, tokens, int(document.source_offset)
                    )
                )
            if not current:
                raise ValueError(f"split {split} is empty")
            self.splits[str(split)] = current

    @property
    def vocab_size(self) -> int:
        return self._vocab_size

    @property
    def itos(self) -> list[str]:
        return [f"<tok-{index}>" for index in range(self._vocab_size)]

    def sample_batch(
        self,
        split: str,
        batch_size: int,
        generator: torch.Generator,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        documents = self.splits[split]
        selected = torch.randint(0, len(documents), (batch_size,), generator=generator)
        inputs: list[torch.Tensor] = []
        targets: list[torch.Tensor] = []
        for document_index in selected.tolist():
            data = documents[document_index].tokens
            start = int(
                torch.randint(
                    0,
                    len(data) - self.seq_len - 1,
                    (1,),
                    generator=generator,
                )
            )
            inputs.append(data[start : start + self.seq_len])
            targets.append(data[start + 1 : start + self.seq_len + 1])
        return torch.stack(inputs), torch.stack(targets)

    def fixed_windows(
        self,
        split: str,
        *,
        windows_per_document: int,
        seed: int,
    ) -> list[EvaluationWindow]:
        if windows_per_document <= 0:
            raise ValueError("windows_per_document must be positive")
        generator = torch.Generator().manual_seed(seed)
        result: list[EvaluationWindow] = []
        for document in self.splits[split]:
            data = document.tokens
            available = len(data) - self.seq_len - 1
            edges = np.linspace(0, available, windows_per_document + 1, dtype=int)
            starts: list[int] = []
            for left, right in zip(edges[:-1], edges[1:], strict=True):
                upper = max(left + 1, right)
                starts.append(int(torch.randint(left, upper, (1,), generator=generator)))
            for start in starts:
                result.append(
                    EvaluationWindow(
                        inputs=data[start : start + self.seq_len],
                        targets=data[start + 1 : start + self.seq_len + 1],
                        document_id=document.document_id,
                        domain=document.domain,
                        start=int(document.source_offset) + start,
                    )
                )
        return result


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def documents_from_prepared_split(
    prepared: PreparedTokenSplit,
    *,
    prefix: str,
    domain: str,
    maximum_document_tokens: int,
    minimum_document_tokens: int,
    maximum_total_tokens: int | None = None,
) -> list[TokenDocument]:
    """Create bounded chunks without ever crossing an original article boundary."""
    prepared.validate()
    if maximum_document_tokens < minimum_document_tokens or minimum_document_tokens <= 0:
        raise ValueError("invalid document token bounds")
    result: list[TokenDocument] = []
    consumed = 0
    for article_index, (left, right) in enumerate(
        zip(prepared.offsets[:-1], prepared.offsets[1:], strict=True)
    ):
        article = torch.as_tensor(
            prepared.tokens[int(left) : int(right)], dtype=torch.long
        ).clone()
        if maximum_total_tokens is not None:
            remaining = int(maximum_total_tokens) - consumed
            if remaining < minimum_document_tokens:
                break
            article = article[:remaining]
        for chunk_index, start in enumerate(range(0, len(article), maximum_document_tokens)):
            chunk = article[start : start + maximum_document_tokens]
            if len(chunk) < minimum_document_tokens:
                continue
            result.append(
                TokenDocument(
                    document_id=f"{prefix}-article-{article_index:04d}",
                    domain=domain,
                    tokens=chunk,
                    source_offset=start,
                )
            )
            consumed += len(chunk)
            if maximum_total_tokens is not None and consumed >= int(maximum_total_tokens):
                break
        if maximum_total_tokens is not None and consumed >= int(maximum_total_tokens):
            break
    if not result:
        raise ValueError(f"no eligible documents for {prefix}")
    return result


def load_prepared_arrays(
    root: Path, *, splits: Sequence[str] = ("train", "validation", "test", "ood")
) -> tuple[dict[str, PreparedTokenSplit], dict]:
    import json

    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    arrays: dict[str, PreparedTokenSplit] = {}
    allowed = {"train", "validation", "test", "ood"}
    requested = tuple(str(split) for split in splits)
    if not requested or any(split not in allowed for split in requested):
        raise ValueError("invalid prepared split request")
    for split in requested:
        token_path = root / f"{split}-tokens.npy"
        offset_path = root / f"{split}-offsets.npy"
        for path in (token_path, offset_path):
            if sha256_file(path) != str(manifest["sha256"][path.name]):
                raise RuntimeError(f"prepared array hash mismatch: {path.name}")
        current = PreparedTokenSplit(
            tokens=np.load(token_path, mmap_mode="r"),
            offsets=np.load(offset_path, mmap_mode="r"),
        )
        current.validate()
        arrays[split] = current
    tokenizer = root / "tokenizer.json"
    if sha256_file(tokenizer) != str(manifest["sha256"]["tokenizer.json"]):
        raise RuntimeError("tokenizer hash mismatch")
    return arrays, manifest
