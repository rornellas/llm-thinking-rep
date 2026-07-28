"""Fresh held-out documents for the alignment-tolerant shared-factor screen."""
from __future__ import annotations

import random
from typing import Sequence

from .data import Document

_DOMAINS = ("general", "code", "math", "portuguese", "structured")


def _hypothesis_paragraph(
    domain: str,
    split: str,
    document: int,
    paragraph: int,
    rng: random.Random,
) -> str:
    a, b, c, d, e = [rng.randint(11, 991) for _ in range(5)]
    key = f"al_{split}_{document}_{paragraph}"
    if domain == "code":
        return (
            f"def {key}(items):\n"
            f"    acc = {a}\n"
            f"    for item in items: acc = (acc * {b} + item + {c}) % {d}\n"
            f"    return acc == {e}\n"
            f"// Java check uses multiplier {b}, offset {c}, modulus {d}.\n"
        )
    if domain == "math":
        return (
            f"Task {key}. Start with x={a} and y={b}. Apply x=(x+y+{c}) mod {d}, "
            f"then y=(2*x+y+{e}) mod {d}. Preserve the update order and report both values.\n"
        )
    if domain == "portuguese":
        return (
            f"O caso {key} começa em {a}, recebe a alteração {b}, depois a mudanca {c}. "
            f"A regra {d} é aplicada antes da confirmação {e}. Explique a ordem sem inverter causa e efeito.\n"
        )
    if domain == "structured":
        return (
            f"BEGIN {key}\nNODE=A{a}; THEN=B{b}; COST={c}; LIMIT={d}; CHECK={e};\n"
            f"APPLY A{a}->B{b}; RETAIN ORDER; END {key}\n"
        )
    return (
        f"Case {key} assigns ticket T{a} to queue Q{b}. Event E{c} changes the value to {d}, "
        f"and confirmation C{e} closes the record. The final answer depends on temporal order.\n"
    )


def generate_alignment_hypothesis_documents(
    *, split: str, documents: int, seed: int
) -> list[Document]:
    rng = random.Random(seed)
    result: list[Document] = []
    for document_index in range(documents):
        domain = _DOMAINS[document_index % len(_DOMAINS)]
        text = "".join(
            _hypothesis_paragraph(domain, split, document_index, paragraph, rng)
            for paragraph in range(34)
        )
        result.append(
            Document(
                document_id=f"{split}-doc-{document_index:04d}",
                source="deterministic-alignment-hypothesis-v1",
                domain=domain,
                text=text,
            )
        )
    return result


def generate_alignment_ood_documents(*, split: str) -> list[Document]:
    templates: Sequence[tuple[str, str]] = (
        (
            "code",
            "function fold(xs) { let s = 19; for (const x of xs) s = (s * 37 + x) % 983; return s; }\n"
            "SQL: SELECT key_id, COUNT(*) FROM audit_log WHERE valid = 1 GROUP BY key_id ORDER BY key_id;\n",
        ),
        (
            "math",
            "Analyze the recurrence a0=7, a1=23, and a(n+2)=4*a(n+1)-3*a(n)+11. "
            "Compute four terms and test the result modulo 17.\n",
        ),
        (
            "portuguese",
            "O lote 163 foi aberto antes do alerta 227. A revisão 349 mudou o valor, e a confirmação 421 "
            "restaurou apenas o rótulo. Determine o estado final com incerteza clara.\n",
        ),
        (
            "structured",
            "<trace id='A31'><next>B47</next><weight>59</weight><limit>971</limit></trace> "
            "<rule>keep-order-then-validate</rule>\n",
        ),
        (
            "general",
            "A sensor network receives north packet 83, south packet 109, and correction packet 151. "
            "The correction changes the value but not the timestamp. Reconstruct the final record.\n",
        ),
        (
            "code",
            "fn stable_merge(left, right) keeps duplicates, preserves input order, and validates checksum 677. "
            "Explain which comparison is performed first when values are equal.\n",
        ),
        (
            "math",
            "For integers p and q, compare (p+q)*(p-q) with p*p-q*q. Include the boundary p=q and verify 29 and 13.\n",
        ),
        (
            "portuguese",
            "Uma série temporal registra 181, 263, 367 e 487. Diferencie correlação, precedência e causalidade "
            "antes de resumir a conclusão.\n",
        ),
        (
            "structured",
            "TRACE root=R41 child=S73 score=127; TRACE root=S73 child=T89 score=211; "
            "QUERY path R41 T89; VERIFY 881; CLOSE TRACE.\n",
        ),
        (
            "general",
            "Three revisions conflict. Revision 53 changes the owner, revision 71 changes the amount, "
            "and revision 97 restores the owner without restoring the amount. State the final pair.\n",
        ),
        (
            "code",
            "class Counter { int step(int x) { return (x * 43 + 17) % 977; } } "
            "Apply step three times starting from 31 and preserve every intermediate value.\n",
        ),
        (
            "math",
            "A weighted graph has vertices A, B, C, D and edges 17, 31, 61, 101. "
            "Contrast a shortest path with a minimum spanning tree without treating them as equivalent.\n",
        ),
    )
    result: list[Document] = []
    for index, (domain, base) in enumerate(templates):
        text = "".join(f"Alignment OOD block {index}-{repeat}. {base}" for repeat in range(20))
        result.append(
            Document(
                document_id=f"{split}-doc-{index:04d}",
                source="deterministic-alignment-ood-v1",
                domain=domain,
                text=text,
            )
        )
    return result
