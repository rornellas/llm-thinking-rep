#!/usr/bin/env python3
"""Prepare a fresh, pinned WikiText-103 corpus for Native Compact Gate 2A."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import sys

import datasets
from datasets import load_dataset
from huggingface_hub import HfApi
import numpy as np
import tokenizers
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers


OOD_TEMPLATES = (
    "A ledger receives revisions 113, 197, 269 and 347. The third changes the payload and the fourth restores only the label. Reconstruct the final state.",
    "WITH ranked AS (SELECT key,value,ROW_NUMBER() OVER (PARTITION BY key ORDER BY stamp DESC) rn FROM events) SELECT key,value FROM ranked WHERE rn=1;",
    "Let a0=23, a1=41, and a(n+2)=7*a(n+1)-6*a(n)+17. Compute ten terms and verify the sequence modulo 29.",
    "Os lotes 211, 307, 401 e 503 possuem dependências ordenadas. A confirmação 601 corrige o terceiro sem apagar as relações anteriores.",
    "<message id='M809'><source>S43</source><target>T71</target><value>887</value></message><rule>rotate-then-check</rule>",
    "def bounded_merge(left, right):\n    state = 19\n    for index, pair in enumerate(zip(left, right)):\n        state = (state * 43 + pair[0] * 5 - pair[1] * 3 + index) % 991\n    return state\n",
    "A weighted directed graph has edges 37, 53, 79 and 101. Reverse exactly two edges and compare the path cost modulo 47.",
    "Duas filas recebem registros 223, 311, 419 e 509. O evento 617 combina as filas preservando as duas últimas correções.",
    "FLOW ID=F811 FROM=N43 TO=P73 VALUE=919; ACTION=swap-two; CHECK=997; CLOSE FLOW.",
    "Four reports share three identifiers. Revision 83 changes one payload, 127 changes a score, and 173 merges references while preserving both edits.",
    "class Fold { long apply(long[] x,long[] y){ long s=17; for(int i=0;i<x.length;i++) s=(s*47+x[i]*7-y[i]*3)%997; return s; } }",
    "Factor x to the eighth minus y to the eighth into coupled factors and verify signs for x=19 and y=5 before reducing modulo 53.",
    "SELECT customer_id, SUM(amount) total FROM orders WHERE status='closed' GROUP BY customer_id HAVING SUM(amount) > 701 ORDER BY total DESC;",
    "Em uma auditoria, os identificadores A31, B47 e C73 recebem versões 2, 4 e 7. A versão 11 desfaz somente a alteração de B47.",
    "A protocol sends packets 61, 89, 137 and 191. Packet 223 acknowledges the first and third, while 251 replaces only the checksum.",
    "function checksum(xs){let s=29; for(let i=0;i<xs.length;i++){s=(s*59+xs[i]*11+i)%1009;} return s;}"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def group_articles(lines: list[str]) -> list[str]:
    articles: list[str] = []
    current: list[str] = []
    for raw in lines:
        text = str(raw).strip()
        if not text:
            continue
        heading = text.startswith("=") and text.endswith("=")
        if heading and current:
            articles.append("\n".join(current) + "\n")
            current = []
        current.append(text)
    if current:
        articles.append("\n".join(current) + "\n")
    return [article for article in articles if article.strip()]


def encode_documents(
    tokenizer: Tokenizer,
    documents: list[str],
    maximum_tokens: int,
) -> tuple[np.ndarray, np.ndarray]:
    packed: list[int] = []
    offsets = [0]
    for document in documents:
        ids = tokenizer.encode(document).ids
        if not ids:
            continue
        remaining = maximum_tokens - len(packed)
        if remaining <= 0:
            break
        ids = ids[:remaining]
        if not ids:
            break
        packed.extend(ids)
        offsets.append(len(packed))
        if len(packed) >= maximum_tokens:
            break
    if len(offsets) < 2:
        raise RuntimeError("prepared split contains no tokenized documents")
    return np.asarray(packed, dtype=np.int32), np.asarray(offsets, dtype=np.int64)


def build_ood_documents(blocks_per_document: int) -> list[str]:
    return [
        "".join(
            f"Native compact OOD article {index}, block {block}. {template}\n"
            for block in range(blocks_per_document)
        )
        for index, template in enumerate(OOD_TEMPLATES)
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--vocab-size", type=int, default=512)
    parser.add_argument("--tokenizer-documents", type=int, default=2048)
    parser.add_argument("--train-tokens", type=int, default=800000)
    parser.add_argument("--calibration-tokens", type=int, default=140000)
    parser.add_argument("--hypothesis-tokens", type=int, default=140000)
    parser.add_argument("--ood-tokens", type=int, default=140000)
    parser.add_argument("--ood-blocks-per-document", type=int, default=75)
    args = parser.parse_args()

    args.root.mkdir(parents=True, exist_ok=True)
    source = "Salesforce/wikitext"
    subset = "wikitext-103-raw-v1"
    revision = HfApi().dataset_info(source).sha
    dataset = load_dataset(source, subset, revision=revision)
    source_articles = {
        split: group_articles([row["text"] for row in dataset[split]])
        for split in ("train", "validation", "test")
    }
    tokenizer_training = source_articles["train"][: int(args.tokenizer_documents)]
    if len(tokenizer_training) < 100:
        raise RuntimeError("too few train articles for tokenizer fitting")

    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.train_from_iterator(
        tokenizer_training,
        trainer=trainers.BpeTrainer(
            vocab_size=args.vocab_size,
            min_frequency=2,
            special_tokens=["<unk>"],
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
            show_progress=False,
        ),
    )
    tokenizer_path = args.root / "tokenizer.json"
    tokenizer.save(str(tokenizer_path))

    documents = {
        "train": source_articles["train"],
        "validation": source_articles["validation"],
        "test": source_articles["test"],
        "ood": build_ood_documents(args.ood_blocks_per_document),
    }
    limits = {
        "train": args.train_tokens,
        "validation": args.calibration_tokens,
        "test": args.hypothesis_tokens,
        "ood": args.ood_tokens,
    }
    split_metadata: dict[str, dict[str, int]] = {}
    data_files: list[Path] = [tokenizer_path]
    for split in ("train", "validation", "test", "ood"):
        tokens, offsets = encode_documents(tokenizer, documents[split], int(limits[split]))
        token_path = args.root / f"{split}-tokens.npy"
        offset_path = args.root / f"{split}-offsets.npy"
        np.save(token_path, tokens)
        np.save(offset_path, offsets)
        data_files.extend((token_path, offset_path))
        split_metadata[split] = {
            "tokens": int(tokens.size),
            "documents": int(len(offsets) - 1),
            "source_documents": int(len(documents[split])),
        }

    manifest = {
        "schema_version": "native-compact-wikitext103-article-v1",
        "source": source,
        "subset": subset,
        "revision": revision,
        "dataset_fingerprints": {
            split: str(dataset[split]._fingerprint)
            for split in ("train", "validation", "test")
        },
        "tokenizer_training_documents": int(len(tokenizer_training)),
        "vocab_size": tokenizer.get_vocab_size(),
        "splits": split_metadata,
        "sha256": {path.name: sha256_file(path) for path in data_files},
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "datasets": datasets.__version__,
            "tokenizers": tokenizers.__version__,
        },
    }
    (args.root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
