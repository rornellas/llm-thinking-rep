#!/usr/bin/env python3
"""Prepare pinned article-preserving WikiText-2 BPE data for Reality Gate 1A."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import sys

import numpy as np
import datasets
from datasets import load_dataset
import tokenizers
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers


DATASET_SOURCE = "Salesforce/wikitext"
DATASET_SUBSET = "wikitext-2-raw-v1"
DATASET_REVISION = "b08601e04326c79dfdd32d625aee71d232d685c3"


OOD_TEMPLATES = (
    "A ledger receives revisions 31, 47, 73 and 109. The second changes the label, the third changes the payload, and the fourth restores only the label. Determine the final pair.",
    "WITH ordered AS (SELECT id,value,ROW_NUMBER() OVER (PARTITION BY id ORDER BY stamp) rn FROM events) SELECT id,value FROM ordered WHERE rn=1 ORDER BY id;",
    "Let x0=17, x1=29, and x(n+2)=5*x(n+1)-4*x(n)+13. Compute eight terms and verify the invariant modulo 19.",
    "Os registros 181, 271, 367 e 461 possuem dependências ordenadas. O último corrige somente o valor de 367 sem apagar a relação anterior. Reconstrua o estado.",
    "<packet id='P701'><left>L37</left><right>R61</right><score>733</score></packet><rule>swap-once-then-validate</rule>",
    "def stable_fold(left, right):\n    total = 11\n    for index, pair in enumerate(zip(left, right)):\n        total = (total * 37 + pair[0] - pair[1] + index) % 977\n    return total\n",
    "A weighted bipartite graph has left weights 31, 47, 71 and right weights 43, 59, 83. Swap two matched edges and compare total cost modulo 37.",
    "Duas filas recebem os lotes 193, 277, 359 e 467. A confirmação 557 combina as filas sem apagar as correções anteriores. Explique o estado conjunto.",
    "FLOW ID=F607 FROM=K31 TO=M59 VALUE=701; ACTION=rotate-two; CHECK=907; CLOSE FLOW.",
    "Four reports share two identifiers. Revision 67 changes one payload, 97 changes the other score, and 131 merges references while preserving both edits.",
    "class PairFold { long apply(long[] x,long[] y){ long s=13; for(int i=0;i<x.length;i++) s=(s*41+x[i]*3-y[i]*5)%983; return s; } }",
    "Factor a sixth minus b sixth into coupled factors and verify every sign with a=17 and b=7 before reducing modulo 43.",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def group_wikitext_articles(lines: list[str]) -> list[str]:
    """Recover article units from WikiText headings without merging across them."""
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


def build_ood_documents(repetitions: int) -> list[str]:
    documents: list[str] = []
    for index, template in enumerate(OOD_TEMPLATES):
        blocks = [
            f"OOD article {index} block {block}. {template}\n"
            for block in range(repetitions)
        ]
        documents.append("".join(blocks))
    return documents


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--vocab-size", type=int, default=512)
    parser.add_argument("--train-tokens", type=int, default=1200000)
    parser.add_argument("--validation-tokens", type=int, default=220000)
    parser.add_argument("--test-tokens", type=int, default=220000)
    parser.add_argument("--ood-tokens", type=int, default=220000)
    parser.add_argument("--ood-blocks-per-document", type=int, default=180)
    args = parser.parse_args()

    args.root.mkdir(parents=True, exist_ok=True)
    source = DATASET_SOURCE
    subset = DATASET_SUBSET
    revision = DATASET_REVISION
    dataset = load_dataset(source, subset, revision=revision)
    source_articles = {
        split: group_wikitext_articles([row["text"] for row in dataset[split]])
        for split in ("train", "validation", "test")
    }

    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.train_from_iterator(
        source_articles["train"],
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
        **source_articles,
        "ood": build_ood_documents(args.ood_blocks_per_document),
    }
    limits = {
        "train": args.train_tokens,
        "validation": args.validation_tokens,
        "test": args.test_tokens,
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
        "schema_version": "reality-gate-wikitext-article-v2",
        "source": source,
        "subset": subset,
        "revision": revision,
        "dataset_fingerprints": {
            split: str(dataset[split]._fingerprint)
            for split in ("train", "validation", "test")
        },
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
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
