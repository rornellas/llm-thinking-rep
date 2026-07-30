"""Fresh deterministic held-outs for route-set-coupled residual v4."""
from __future__ import annotations

import random

from .data import Document

_DOMAINS = ("general", "code", "math", "portuguese", "structured")


def _block(domain: str, split: str, doc: int, block: int, rng: random.Random) -> str:
    a, b, c, d, e, f = [rng.randint(17, 991) for _ in range(6)]
    key = f"rc_{split}_{doc}_{block}"
    if domain == "code":
        return (
            f"def {key}(left, right):\n"
            f"    state = {a}\n"
            f"    for i, pair in enumerate(zip(left, right)):\n"
            f"        state = (state * {b} + pair[0] * {c} - pair[1] * {d} + i * {e}) % {f}\n"
            f"    return state\n"
            f"# retain pairing, duplicates, and order for {key}\n"
        )
    if domain == "math":
        return (
            f"Exercise {key}. Let u0={a}, u1={b}, and u(n+2)=({c}*u(n+1)+{d}*u(n)+{e}) mod {f}. "
            "Compute five terms, compare two grouped sums, and preserve every sign and index.\n"
        )
    if domain == "portuguese":
        return (
            f"No cenário {key}, o evento {a} inicia duas trilhas; {b} altera a primeira, {c} altera a segunda, "
            f"{d} combina ambas, {e} corrige somente a segunda e {f} confirma o estado conjunto. "
            "Reconstrua as dependências sem trocar associação por causalidade.\n"
        )
    if domain == "structured":
        return (
            f"SET {key}\nLEFT=L{a}; RIGHT=R{b}; LINK={c}; WEIGHT={d}; REVISION={e}; CHECK={f};\n"
            "MERGE PAIRWISE; RETAIN ORDER; VERIFY JOINT STATE; CLOSE SET.\n"
        )
    return (
        f"Case {key} joins channel C{a} with channel C{b}. Update {c} changes the left payload, update {d} "
        f"changes the right label, update {e} joins the records, and audit {f} verifies the coupled result.\n"
    )


def generate_routing_coupled_hypothesis_documents(
    *, split: str, documents: int, seed: int
) -> list[Document]:
    rng = random.Random(seed)
    result: list[Document] = []
    for document_index in range(documents):
        domain = _DOMAINS[document_index % len(_DOMAINS)]
        text = "".join(
            _block(domain, split, document_index, block, rng)
            for block in range(30)
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


def generate_routing_coupled_ood_documents(*, split: str) -> list[Document]:
    templates = (
        ("code", "fn coupled_scan(a: Vec<i64>, b: Vec<i64>) -> i64 { zip, rotate alternating pairs, retain duplicates, and fold modulo 991. }\n"),
        ("math", "Two recurrences share every third term. Given initial values 23, 41, 67 and multipliers 5 and 7, compute the coupled state modulo 29.\n"),
        ("portuguese", "Duas filas recebem os lotes 193, 277, 359 e 467. A confirmação 557 combina as filas sem apagar as correções anteriores. Explique o estado conjunto.\n"),
        ("structured", "PAIRSET ID=PS719 LEFT=K43 RIGHT=M71 EDGE=887; ACTION=interleave; REVISION=929; VERIFY=967; CLOSE.\n"),
        ("general", "Four ledgers share two identifiers. Revision 73 changes one payload, 109 changes the other label, and 149 merges the references while keeping both edits.\n"),
        ("code", "WITH paired AS (SELECT l.id,l.value AS lv,r.value AS rv FROM left_log l JOIN right_log r ON l.id=r.id) SELECT id,lv-rv FROM paired ORDER BY id;\n"),
        ("math", "A weighted bipartite graph has left weights 31, 47, 71 and right weights 43, 59, 83. Swap two matched edges and compare total cost modulo 37.\n"),
        ("portuguese", "Os registros 211 e 313 seguem trilhas separadas; 421 associa as trilhas e 523 corrige apenas o valor vindo de 313. Determine a dupla final.\n"),
        ("structured", "<coupled id='C811'><a>A53</a><b>B79</b><join>857</join><check>947</check></coupled><rule>preserve-both-then-validate</rule>\n"),
        ("general", "Two timelines contain updates 61, 101, 139 and 181. The final reconciliation keeps the newer label from one and the newer payload from the other.\n"),
        ("code", "class PairFold { long apply(long[] x,long[] y){ long s=13; for(int i=0;i<x.length;i++) s=(s*41+x[i]*3-y[i]*5)%983; return s; } }\n"),
        ("math", "Factor a sixth minus b sixth into coupled factors and verify the signs with a=17 and b=7 before reducing modulo 43.\n"),
    )
    result: list[Document] = []
    for index, (domain, template) in enumerate(templates):
        text = "".join(
            f"Routing-coupled OOD block {split}-{index}-{repeat}. {template}"
            for repeat in range(24)
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
