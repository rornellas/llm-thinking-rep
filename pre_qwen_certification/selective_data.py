"""Fresh development documents for the selective hot/cold expert screen.

The in-distribution set deliberately keeps the character-level domains used to
train the tiny teachers while changing wording and record structure.  The OOD
set is original to this experiment and does not reuse the hand-authored v1
paragraphs consumed by earlier screens.  These are development data, not a
sealed confirmation set.
"""
from __future__ import annotations

import random

from .data import Document


def generate_selective_hypothesis_documents(
    *,
    documents: int,
    seed: int,
    split: str = "selective-hypothesis-v1",
) -> list[Document]:
    if documents < 4:
        raise ValueError("at least four documents are required")
    rng = random.Random(seed)
    domains = ("general", "code", "math", "portuguese")
    output: list[Document] = []
    for document_index in range(documents):
        domain = domains[document_index % len(domains)]
        paragraphs: list[str] = []
        for paragraph_index in range(24):
            a = rng.randint(11, 997)
            b = rng.randint(7, 991)
            c = rng.randint(5, 983)
            label = f"case_{document_index}_{paragraph_index}"
            if domain == "general":
                paragraphs.append(
                    f"The analyst opens {label}, reads values {a}, {b}, and {c}, then checks two independent rules. "
                    f"Rule one maps A{a} to V{b}; rule two reports the total {a+b+c}. "
                    "A later sentence asks which observation came first and which conclusion remains uncertain.\n"
                )
            elif domain == "code":
                paragraphs.append(
                    f"def {label}(input):\n    base = input + {a}\n    scaled = base * {b}\n"
                    f"    return scaled % {max(c, 7)}\n"
                    f"invariant: ({a} + x) * {b} is reduced only after multiplication.\n"
                )
            elif domain == "math":
                paragraphs.append(
                    f"Problem {paragraph_index}. Let q={a}, r={b}, and m={max(c, 7)}. "
                    f"Compare (q+r) mod m with ((q mod m)+(r mod m)) mod m. "
                    f"The check values are {a+b}, {(a+b) % max(c, 7)}, and {a*b}.\n"
                )
            else:
                paragraphs.append(
                    f"No caso {label}, a equipe recebe {a}, {b} e {c}. "
                    f"Primeiro verifica a ordem, depois calcula {a+b} e por fim registra {a+b+c}. "
                    "A conclusao separa o que foi observado do que ainda precisa de confirmacao.\n"
                )
        output.append(
            Document(
                document_id=f"{split}-doc-{document_index:04d}",
                text="\n".join(paragraphs),
                source="deterministic-selective-generator-v1",
                domain=domain,
            )
        )
    return output


def generate_selective_ood_documents(
    split: str = "selective-ood-v2",
) -> list[Document]:
    rows = [
        (
            "science",
            "A marine station compares salinity readings before and after a pump repair. The average moves toward the reference, "
            "but one probe still drifts whenever the tide changes. The report preserves the time order, shows every replicate, "
            "and states that a lower mean error does not by itself identify the cause.",
        ),
        (
            "portuguese",
            "Um arquivo antigo descreve tres etapas de uma auditoria. A primeira confere a origem dos numeros, a segunda repete "
            "o calculo sem compartilhar o mesmo codigo, e a terceira procura um caso em que a conclusao falharia. O resumo nao "
            "transforma uma melhora pequena em prova geral.",
        ),
        (
            "code",
            "A queue worker reads an identifier, reserves a task, and writes the result only after a durable checkpoint. A fault "
            "test stops the process between reservation and commit, restarts it, and verifies that the task is neither lost nor "
            "applied twice. The benchmark includes retry and serialization costs.",
        ),
        (
            "math",
            "Start with two fractions that have different denominators. Convert both to a common denominator, add the numerators, "
            "and reduce the result by their greatest common divisor. A worked example is useful only if the same rule explains a "
            "new pair that was not used to choose the method.",
        ),
        (
            "reasoning",
            "Three envelopes contain a key, a receipt, and an empty card. One statement says the key is not in the first envelope; "
            "another says the receipt is beside the key. Learning that the second statement is false removes possibilities, but it "
            "does not justify choosing a unique arrangement without checking the remaining constraint.",
        ),
        (
            "systems",
            "A storage service reports high throughput while every request hits memory. A production trace contains cache misses, "
            "small writes, and bursts that force compaction. The corrected experiment replays the trace and separates CPU time, "
            "bytes moved, queue delay, and tail latency instead of reporting one favorable average.",
        ),
        (
            "narrative",
            "Lina heard the workshop bell after the lights had gone out. She wrote down the sequence, compared it with the guard log, "
            "and noticed that the clock in the hall had stopped earlier. That detail weakened the first explanation and suggested a "
            "test that could distinguish a late visitor from a faulty timer.",
        ),
        (
            "general",
            "A committee ranks four proposals using cost, reliability, and reversibility. The cheapest option performs poorly under "
            "a rare failure, while the most reliable option is difficult to undo. The final note presents the tradeoff and the "
            "assumption behind each score rather than declaring a universal winner.",
        ),
    ]
    output: list[Document] = []
    for index, (domain, paragraph) in enumerate(rows):
        text = "\n".join(
            [
                paragraph,
                "A second passage changes the subject briefly, then asks whether the earlier evidence still supports the same conclusion. "
                + paragraph.lower(),
                "The closing passage records the strongest counterexample and the measurement that would resolve it. "
                + paragraph,
            ]
        )
        output.append(
            Document(
                document_id=f"{split}-doc-{index:04d}",
                text=text,
                source="hand-authored-selective-ood-v2",
                domain=domain,
            )
        )
    return output
