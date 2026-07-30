"""Fresh deterministic held-out documents for routing-set distillation v3."""
from __future__ import annotations

import random

from .data import Document

_DOMAINS = ("general", "code", "math", "portuguese", "structured")


def _block(domain: str, split: str, doc: int, block: int, rng: random.Random) -> str:
    a, b, c, d, e = [rng.randint(11, 983) for _ in range(5)]
    key = f"rs_{split}_{doc}_{block}"
    if domain == "code":
        return (
            f"def {key}(items):\n"
            f"    total = {a}\n"
            f"    for index, value in enumerate(items):\n"
            f"        total = (total + value * {b} + index * {c}) % {d}\n"
            f"    return total == {e}\n"
            f"// preserve stable order and duplicate values for {key}\n"
        )
    if domain == "math":
        return (
            f"Problem {key}. Define x0={a}, x1={b}, and x(n+2)=({c}*x(n+1)+{d}*x(n)+{e}) mod 997. "
            f"Compute four terms, compare parity, and verify the recurrence without changing order.\n"
        )
    if domain == "portuguese":
        return (
            f"O ensaio {key} recebeu os eventos {a}, {b}, {c}, {d} e {e}. "
            "A conclusão deve distinguir sequência, associação e causalidade, mantendo os valores exatos e o contexto.\n"
        )
    if domain == "structured":
        return (
            f"BEGIN {key}\nNODE=N{a}; EDGE=N{a}->N{b}; VALUE={c}; LIMIT={d}; CHECK={e};\n"
            f"APPLY ORDERED MERGE; VERIFY ({a}+{b}+{c})%{d}; END {key}\n"
        )
    return (
        f"Record {key} links register R{a} to packet P{b} before revision V{c}. "
        f"A later update {d} changes the value but not the label, while audit {e} checks the final ordered state.\n"
    )


def generate_routing_set_hypothesis_documents(
    *, split: str, documents: int, seed: int
) -> list[Document]:
    rng = random.Random(seed)
    result: list[Document] = []
    for document_index in range(documents):
        domain = _DOMAINS[document_index % len(_DOMAINS)]
        text = "".join(
            _block(domain, split, document_index, block, rng)
            for block in range(28)
        )
        result.append(
            Document(
                document_id=f"{split}-doc-{document_index:04d}",
                source=split,
                domain=domain,
                text=text,
            )
        )
    return result


def generate_routing_set_ood_documents(*, split: str) -> list[Document]:
    if "confirmation" in split:
        templates = (
            ("code", "fn paired_fold(xs: Vec<i64>) -> i64 { retain duplicates, swap neighbor pairs, fold modulo 983, and verify 449. }\n"),
            ("math", "A directed hexagon has weights 19, 37, 53, 79, 113, 157. Reverse two separated edges and compare the least path before and after.\n"),
            ("portuguese", "O registro 173 antecede a revisão 257; o alerta 349 depende de 257, e a confirmação 443 corrige somente o valor. Reconstrua o estado final.\n"),
            ("structured", "FLOW ID=F607 FROM=K31 TO=M59 VALUE=701; ACTION=rotate-two; CHECK=907; CLOSE FLOW.\n"),
            ("general", "Six reports arrive from three channels. Report 67 changes owner, 97 changes score, and 131 restores owner while preserving the changed score. Determine the final tuple.\n"),
            ("code", "WITH ranked AS (SELECT key, value, ROW_NUMBER() OVER (PARTITION BY key ORDER BY stamp) AS rn FROM events) SELECT key,value FROM ranked WHERE rn=1;\n"),
            ("math", "Let z0=17, z1=29, and z(n+2)=5*z(n+1)-4*z(n)+13. Compute six terms and verify an invariant modulo 19.\n"),
            ("portuguese", "Os lotes 181, 271, 367 e 461 têm dependências ordenadas. O lote 461 corrige 367 sem apagar o vinculo de 271 com 181. Explique a sequência.\n"),
            ("structured", "<packet id='P701'><left>L37</left><right>R61</right><score>733</score></packet><rule>swap-once-then-validate</rule>\n"),
            ("general", "A ledger has revisions 41, 71, 103 and 137. The second changes payload, the third changes label, and the fourth restores payload only. State the final label and payload.\n"),
            ("code", "class StableMix { long apply(long[] a,long[] b){ long s=11; for(int i=0;i<a.length;i++) s=(s*37+a[i]-b[i])%977; return s; } }\n"),
            ("math", "Factor p to the fourth minus q to the fourth and verify every factor with p=23 and q=11, preserving the signs and order.\n"),
        )
    else:
        # Retired engineering-smoke family. Kept only so the smoke remains exactly
        # reproducible; it must never be used for final v3 evidence.
        templates = (
            ("code", "fn rotate_checksum(values: Vec<i32>) -> i32 { preserve order, rotate by 7, and fold modulo 977; verify 431. }\n"),
            ("math", "Consider a weighted directed graph with nodes A, B, C, D, E and edge weights 17, 31, 47, 73, 109. Compare path cost after reversing exactly one edge.\n"),
            ("portuguese", "O arquivo recebeu a revisão 127 antes do alerta 211, mas a confirmação 307 chegou depois da correção 263. Reconstrua o estado final sem trocar causa e consequência.\n"),
            ("structured", "TRACE ID=T401 SOURCE=S19 TARGET=Q23 VALUE=617; STEP=append; STEP=reverse-last; VERIFY=883; CLOSE.\n"),
            ("general", "A laboratory received five messages from alternating channels. Message 59 changes the key, 83 changes the value, and 101 restores only the key. Determine the final record.\n"),
            ("code", "SELECT account, SUM(delta) FROM ledger WHERE valid = 1 GROUP BY account HAVING SUM(delta) > 251 ORDER BY account; preserve null semantics.\n"),
            ("math", "Let y0=13, y1=23, and y(n+2)=4*y(n+1)-3*y(n)+11. Compute five terms and identify the invariant modulo 17.\n"),
            ("portuguese", "A análise compara os lotes 151, 239, 331 e 419. O lote 239 depende de 151; 419 corrige 331 sem apagar 239. Descreva a ordem correta.\n"),
            ("structured", "<entry id='E503'><from>R29</from><to>R47</to><score>659</score></entry><rule>retain-order-and-check</rule>\n"),
            ("general", "Three revisions disagree. Revision 37 changes priority, revision 61 changes payload, and revision 89 restores priority without restoring payload. State the final pair.\n"),
            ("code", "class QueueFold { int apply(int[] xs) { int s=5; for(int x:xs) s=(s*29+x)%991; return s; } }\n"),
            ("math", "For integers m and n, factor m cubed minus n cubed and test the identity with 19 and 7 while preserving every intermediate sign.\n"),
        )
    result: list[Document] = []
    for index, (domain, template) in enumerate(templates):
        text = "".join(
            f"Routing-set OOD block {split}-{index}-{repeat}. {template}"
            for repeat in range(22)
        )
        result.append(
            Document(
                document_id=f"{split}-doc-{index:04d}",
                source=split,
                domain=domain,
                text=text,
            )
        )
    return result
