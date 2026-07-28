"""Small hand-authored OOD corpus for controlled transplantation diagnostics.

The texts are original to this repository and intentionally differ from the
repetitive key/value templates used by ``generate_multidomain_documents``.  The
corpus is still a small diagnostic set, not a substitute for a natural-language
benchmark suite.
"""
from __future__ import annotations

from .data import Document


def generate_hand_authored_ood_documents(split: str = "hand-ood") -> list[Document]:
    texts = [
        (
            "general",
            "A field notebook describes a quiet observatory after a storm. The operator checks the roof, "
            "compares two clocks, records a faint signal, and explains why a delayed measurement can still be useful. "
            "The account moves from cause to consequence without repeating a fixed key or checksum. "
            "Later, a second observer questions the interpretation and proposes an ordinary mechanical explanation. "
        ),
        (
            "portuguese",
            "Uma equipe pequena revisa um sistema de relatorios antes de apresentar o resultado. Primeiro confere os dados, "
            "depois reproduz o calculo em outro programa e por fim procura uma explicacao alternativa para cada melhora. "
            "O texto distingue evidencia exploratoria de confirmacao e registra tambem os resultados negativos. "
            "Nenhuma decisao depende de uma unica medida ou de um exemplo escolhido depois do teste. "
        ),
        (
            "code",
            "A parser reads a stream one record at a time. It validates the header, stores a checksum, and rejects a record "
            "when the declared length differs from the bytes consumed. A separate test replaces one byte, replays the stream, "
            "and expects a deterministic failure. The implementation avoids shared mutable state so two workers can run safely. "
        ),
        (
            "math",
            "Consider a sequence whose first term is three and whose next term equals twice the previous term minus one. "
            "The first values are three, five, nine, and seventeen. A proof by induction shows that the nth term is two to the n "
            "plus one. The useful point is the invariant, not memorizing the displayed examples. "
        ),
        (
            "science",
            "A laboratory compares two sensors under the same temperature ramp. One sensor has lower average error but a long tail, "
            "while the other is slightly noisier and never saturates. The report therefore presents mean error, worst-case error, "
            "calibration drift, and uncertainty across repeated runs instead of selecting a single favorable statistic. "
        ),
        (
            "reasoning",
            "Four boxes are labeled red, blue, green, and white, but every label is wrong. Opening the box labeled red reveals a blue "
            "object. This observation constrains the remaining assignments, yet it does not identify all of them without another clue. "
            "The solution must separate what follows logically from what is merely plausible. "
        ),
        (
            "systems",
            "During deployment, a service receives a request, consults a cache, and calls a slower database only on a miss. "
            "A benchmark that preloads every cache entry measures a different workload from production. The engineer reports both "
            "warm-cache throughput and cold-start latency, including serialization, queueing, and data movement. "
        ),
        (
            "narrative",
            "At dawn, Lina found an unsigned map beneath the workshop door. She recognized the river but not the marked bridge. "
            "Instead of following it immediately, she compared the ink with old plans, asked who had access to the room, and noticed "
            "that the newest line crossed a road built only last winter. That detail changed the order of her questions. "
        ),
    ]
    result = []
    for index, (domain, paragraph) in enumerate(texts):
        # Repeat with connective variation to ensure enough non-template context
        # for several fixed windows while keeping every document distinct.
        text = "\n".join(
            [
                paragraph,
                "A second paragraph adds context and tests whether earlier statements remain available when the subject changes. "
                + paragraph.lower(),
                "The final paragraph summarizes the claim, its limitation, and the observation that would falsify it. "
                + paragraph,
            ]
        )
        result.append(
            Document(
                document_id=f"{split}-doc-{index:04d}",
                text=text,
                source="hand-authored-ood-v1",
                domain=domain,
            )
        )
    return result
