"""Fresh deterministic documents for the teacher-informed width replication."""
from __future__ import annotations

import random
from typing import Sequence

from .data import Document


_DOMAINS = ("general", "code", "math", "portuguese", "structured")


def _paragraph(domain: str, split: str, document: int, paragraph: int, rng: random.Random) -> str:
    a, b, c, d = [rng.randint(3, 997) for _ in range(4)]
    key = f"tw_{split}_{document}_{paragraph}"
    if domain == "code":
        return (
            f"def {key}(x, y):\n"
            f"    value = (x * {a} + y * {b} + {c}) % {d}\n"
            f"    return value\n"
            f"// Java: int value = (x * {a} + y * {b} + {c}) % {d};\n"
        )
    if domain == "math":
        return (
            f"Exercise {key}. Let p={a}, q={b}, r={c}. "
            f"Compute (p*q+r) mod {d}. The sequence is {a}, {a+b}, {a+2*b}, {a+3*b}. "
            f"Check {a}*({b}+{c})={a*b}+{a*c}.\n"
        )
    if domain == "portuguese":
        return (
            f"O relatório {key} registra {a}, {b}, {c} e {d}. "
            f"A análise deve preservar contexto, precisão, ordem e causalidade. "
            f"Some {a} com {b}, compare com {c} e descreva a conclusão.\n"
        )
    if domain == "structured":
        return (
            f"BEGIN {key}\nGROUP=G{a}; ITEM=I{b}; SCORE={c}; LIMIT={d};\n"
            f"LINK G{a}->I{b}; CHECK ({c}+{a})%{d}; END {key}\n"
        )
    return (
        f"Record {key} maps key K{a} to value V{b} inside group G{c}. "
        f"The verifier combines {a}, {b}, and {c}, then compares the result with {d}. "
        "Context and ordering determine the correct continuation.\n"
    )


def generate_width_documents(*, split: str, documents: int, seed: int) -> list[Document]:
    rng = random.Random(seed)
    result: list[Document] = []
    for document_index in range(documents):
        domain = _DOMAINS[document_index % len(_DOMAINS)]
        paragraphs = [
            _paragraph(domain, split, document_index, paragraph, rng)
            for paragraph in range(32)
        ]
        if "train" in split.lower():
            # Freeze the character vocabulary on train data while covering every
            # symbol used by the preregistered OOD family. This line carries no
            # evaluation content or labels.
            paragraphs.append("Charset coverage: ' < > [ ] { } F Q W Y z ç é ê õ ú.\n")
        result.append(
            Document(
                document_id=f"{split}-doc-{document_index:04d}",
                source=split,
                domain=domain,
                text="".join(paragraphs),
            )
        )
    return result


def generate_width_ood_documents(*, split: str) -> list[Document]:
    templates: Sequence[tuple[str, str]] = (
        (
            "code",
            "class Ledger { int fold(int[] xs) { int s = 7; for (int x : xs) s = (s * 31 + x) % 997; return s; } }\n"
            "def recursive_checksum(values):\n    return 0 if not values else (values[0] + 17 * recursive_checksum(values[1:])) % 991\n",
        ),
        (
            "math",
            "Proof task. For integers a and b, analyze when a squared minus b squared is divisible by a minus b. "
            "Use the factorization (a-b)(a+b), identify boundary cases, and verify with 17 and 5.\n",
        ),
        (
            "portuguese",
            "Uma equipe avaliou séries temporais, exceções e dependências. O parecer deve distinguir correlação de causalidade, "
            "registrar incerteza e preservar os números 137, 281, 419 e 563.\n",
        ),
        (
            "structured",
            "TRACE root=A17 child=B29 weight=43; TRACE root=B29 child=C53 weight=71; "
            "QUERY path A17 C53; VERIFY checksum 887; CLOSE TRACE.\n",
        ),
        (
            "general",
            "A remote station received alternating packets from north and south. Packet 73 updates register 211, "
            "packet 89 reverses the queue, and packet 107 validates the final state. Explain the order-sensitive result.\n",
        ),
        (
            "code",
            "SELECT group_id, SUM(amount) FROM events WHERE status = 'valid' GROUP BY group_id HAVING SUM(amount) > 257;\n"
            "// Preserve null handling, stable ordering, and exact aggregation semantics.\n",
        ),
        (
            "math",
            "A recurrence starts with x0=11 and x1=19, then x(n+2)=3*x(n+1)-2*x(n)+7. "
            "Compute the next terms and discuss the invariant modulo 13.\n",
        ),
        (
            "portuguese",
            "No experimento, o lote 149 precede o lote 233, mas a confirmação 317 chega depois do alerta 271. "
            "Reconstrua a sequência sem trocar causa e consequência.\n",
        ),
        (
            "structured",
            "<entry id='E401'><source>S17</source><target>T23</target><value>619</value></entry> "
            "<rule>retain-order-and-validate</rule>\n",
        ),
        (
            "general",
            "The archive contains three conflicting revisions. Revision 43 changes the label, revision 61 changes the value, "
            "and revision 79 restores the label without restoring the value. Determine the final record.\n",
        ),
        (
            "code",
            "fn merge(left: Vec<i32>, right: Vec<i32>) -> Vec<i32> { /* stable merge; retain duplicates; validate 673 */ }\n",
        ),
        (
            "math",
            "Given a graph with vertices P, Q, R, S and weighted edges 13, 29, 47, 83, compare the shortest path and the minimum spanning tree.\n",
        ),
    )
    result: list[Document] = []
    for index, (domain, base) in enumerate(templates):
        text = "".join(f"OOD block {index}-{repeat}. {base}" for repeat in range(20))
        result.append(
            Document(
                document_id=f"{split}-doc-{index:04d}",
                source=split,
                domain=domain,
                text=text,
            )
        )
    return result
