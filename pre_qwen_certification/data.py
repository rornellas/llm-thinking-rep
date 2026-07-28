"""Document-level split, deduplication, and sealed-manifest helpers."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import hmac
import json
import re
import unicodedata
from pathlib import Path
from typing import Iterable, Sequence


_SPACE = re.compile(r"\s+")
_WORD = re.compile(r"\w+", flags=re.UNICODE)


@dataclass(frozen=True)
class Document:
    document_id: str
    text: str
    source: str = "unknown"
    domain: str = "general"

    @classmethod
    def from_mapping(cls, value: dict[str, object]) -> "Document":
        document_id = str(value.get("document_id") or value.get("id") or "").strip()
        text = str(value.get("text") or "")
        if not document_id:
            raise ValueError("document_id is required")
        if not text.strip():
            raise ValueError(f"document {document_id!r} has empty text")
        return cls(
            document_id=document_id,
            text=text,
            source=str(value.get("source") or "unknown"),
            domain=str(value.get("domain") or "general"),
        )


@dataclass(frozen=True)
class DuplicateFinding:
    left_id: str
    right_id: str
    kind: str
    score: float


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).casefold()
    return _SPACE.sub(" ", text).strip()


def text_sha256(text: str) -> str:
    return hashlib.sha256(normalize_text(text).encode("utf-8")).hexdigest()


def word_shingles(text: str, width: int = 5) -> frozenset[tuple[str, ...]]:
    if width <= 0:
        raise ValueError("shingle width must be positive")
    words = _WORD.findall(normalize_text(text))
    if not words:
        return frozenset()
    if len(words) < width:
        return frozenset({tuple(words)})
    return frozenset(tuple(words[index : index + width]) for index in range(len(words) - width + 1))


def jaccard(left: frozenset[object], right: frozenset[object]) -> float:
    if not left and not right:
        return 1.0
    union = len(left | right)
    return len(left & right) / max(union, 1)


def find_duplicates(
    documents: Sequence[Document],
    *,
    near_duplicate_threshold: float = 0.85,
    shingle_width: int = 5,
) -> list[DuplicateFinding]:
    if not 0.0 <= near_duplicate_threshold <= 1.0:
        raise ValueError("near_duplicate_threshold must be in [0, 1]")
    hashes: dict[str, list[Document]] = {}
    shingles: dict[str, frozenset[tuple[str, ...]]] = {}
    for document in documents:
        hashes.setdefault(text_sha256(document.text), []).append(document)
        shingles[document.document_id] = word_shingles(document.text, shingle_width)

    findings: list[DuplicateFinding] = []
    exact_pairs: set[frozenset[str]] = set()
    for group in hashes.values():
        for left_index in range(len(group)):
            for right_index in range(left_index + 1, len(group)):
                left, right = group[left_index], group[right_index]
                exact_pairs.add(frozenset((left.document_id, right.document_id)))
                findings.append(
                    DuplicateFinding(left.document_id, right.document_id, "exact", 1.0)
                )

    # The exact O(n^2) detector is intentional for certification manifests.  For
    # very large corpora, callers should pre-cluster with MinHash and feed only
    # candidate pairs into this exact verifier.
    for left_index, left in enumerate(documents):
        for right in documents[left_index + 1 :]:
            pair = frozenset((left.document_id, right.document_id))
            if pair in exact_pairs:
                continue
            score = jaccard(shingles[left.document_id], shingles[right.document_id])
            if score >= near_duplicate_threshold:
                findings.append(
                    DuplicateFinding(
                        left.document_id, right.document_id, "near", float(score)
                    )
                )
    return sorted(findings, key=lambda row: (row.kind, row.left_id, row.right_id))


def _split_score(document_id: str, secret_seed: str) -> float:
    digest = hmac.new(
        secret_seed.encode("utf-8"),
        document_id.encode("utf-8"),
        hashlib.sha256,
    ).digest()
    integer = int.from_bytes(digest[:8], "big", signed=False)
    return integer / float(1 << 64)


def split_documents(
    documents: Sequence[Document],
    *,
    secret_seed: str,
    fractions: dict[str, float],
    reject_duplicates: bool = True,
    near_duplicate_threshold: float = 0.85,
) -> tuple[dict[str, list[Document]], list[DuplicateFinding]]:
    if not secret_seed:
        raise ValueError("a non-empty secret_seed is required")
    if not fractions:
        raise ValueError("at least one split is required")
    if any(value <= 0.0 for value in fractions.values()):
        raise ValueError("split fractions must be positive")
    total = sum(fractions.values())
    if abs(total - 1.0) > 1e-9:
        raise ValueError(f"split fractions must sum to 1.0, got {total}")
    identifiers = [document.document_id for document in documents]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("document IDs must be unique")

    duplicates = find_duplicates(
        documents, near_duplicate_threshold=near_duplicate_threshold
    )
    if reject_duplicates and duplicates:
        sample = duplicates[0]
        raise ValueError(
            f"duplicate documents detected: {sample.left_id!r} and "
            f"{sample.right_id!r} ({sample.kind}, score={sample.score:.3f})"
        )

    boundaries: list[tuple[str, float]] = []
    cumulative = 0.0
    for name, fraction in fractions.items():
        cumulative += float(fraction)
        boundaries.append((name, cumulative))
    boundaries[-1] = (boundaries[-1][0], 1.0)

    result = {name: [] for name in fractions}
    for document in documents:
        score = _split_score(document.document_id, secret_seed)
        for name, boundary in boundaries:
            if score < boundary:
                result[name].append(document)
                break
    return result, duplicates


def _document_manifest_row(document: Document) -> dict[str, object]:
    normalized = normalize_text(document.text)
    return {
        "document_id": document.document_id,
        "source": document.source,
        "domain": document.domain,
        "normalized_sha256": hashlib.sha256(normalized.encode("utf-8")).hexdigest(),
        "utf8_bytes": len(document.text.encode("utf-8")),
        "normalized_characters": len(normalized),
    }


def write_split_bundle(
    output_dir: Path,
    splits: dict[str, list[Document]],
    *,
    source_path: Path | None,
    protocol_version: str,
    seed_commitment: str,
    duplicate_findings: Sequence[DuplicateFinding] = (),
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, object] = {
        "protocol_version": protocol_version,
        "source_path": str(source_path) if source_path else None,
        "seed_commitment_sha256": seed_commitment,
        "splits": {},
        "duplicate_findings": [asdict(item) for item in duplicate_findings],
    }
    for split_name, documents in splits.items():
        path = output_dir / f"{split_name}.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            for document in sorted(documents, key=lambda item: item.document_id):
                handle.write(json.dumps(asdict(document), ensure_ascii=False) + "\n")
        file_digest = hashlib.sha256(path.read_bytes()).hexdigest()
        rows = [_document_manifest_row(document) for document in documents]
        manifest["splits"][split_name] = {
            "documents": len(documents),
            "jsonl_sha256": file_digest,
            "records": sorted(rows, key=lambda row: str(row["document_id"])),
        }

    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "sha256sums.txt").write_text(
        "\n".join(
            f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.name}"
            for path in sorted(output_dir.glob("*.jsonl"))
        )
        + "\n",
        encoding="utf-8",
    )
    return manifest


def load_jsonl_documents(path: Path) -> list[Document]:
    documents: list[Document] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
                documents.append(Document.from_mapping(payload))
            except Exception as error:  # noqa: BLE001 - add source line context
                raise ValueError(f"failed to parse {path}:{line_number}: {error}") from error
    if not documents:
        raise ValueError(f"no documents found in {path}")
    return documents


def seed_commitment(secret_seed: str) -> str:
    if not secret_seed:
        raise ValueError("secret_seed must not be empty")
    return hashlib.sha256(secret_seed.encode("utf-8")).hexdigest()


def verify_split_bundle(output_dir: Path) -> list[str]:
    manifest_path = output_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    errors: list[str] = []
    seen_documents: set[str] = set()
    seen_hashes: dict[str, str] = {}
    for split_name, split in manifest["splits"].items():
        path = output_dir / f"{split_name}.jsonl"
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != split["jsonl_sha256"]:
            errors.append(f"digest mismatch for {path.name}")
        for row in split["records"]:
            document_id = str(row["document_id"])
            normalized_hash = str(row["normalized_sha256"])
            if document_id in seen_documents:
                errors.append(f"document appears in multiple splits: {document_id}")
            seen_documents.add(document_id)
            if normalized_hash in seen_hashes:
                errors.append(
                    f"exact normalized duplicate across splits: {document_id} and "
                    f"{seen_hashes[normalized_hash]}"
                )
            seen_hashes[normalized_hash] = document_id
    return errors
