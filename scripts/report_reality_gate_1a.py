#!/usr/bin/env python3
"""Write a concise factual and adversarial Reality Gate 1A report."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

PRIMARY = "heterogeneous-spectral"
UNIFORM = "uniform-rank"
NARROW = "narrow65"


def interval(value: dict) -> str:
    return f"{value['mean']:+.6f} [{value['lcb']:+.6f}, {value['ucb']:+.6f}]"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    metrics = json.loads(args.metrics.read_text(encoding="utf-8"))
    audit = json.loads(args.audit.read_text(encoding="utf-8"))
    lines = [
        "# Reality Gate 1A — análise factual e adversarial",
        "",
        f"**Protocolo:** `{metrics['metadata']['protocol_version']}`  ",
        f"**Veredito:** `{metrics['decision']['verdict']}`  ",
        f"**Auditoria independente:** `{'PASS' if audit['audit_passed'] else 'FAIL'}`  ",
        "**Decisão global:** `NO_GO_FOR_OLMOE_OR_QWEN`",
        "",
        "## Pergunta",
        "",
        "Ranks heterogêneos estáticos, alocados apenas com informação de treino e sob os mesmos orçamentos de parâmetros e compute esperado do rank uniforme, melhoram a fidelidade em teachers com plateau explícito e em duas escalas?",
        "",
    ]
    for scale, data in metrics["scales"].items():
        lines.extend(
            [
                f"## Escala `{scale}`",
                "",
                "| Candidate | Δ loss | KL | Top-1 | Local NRMSE | Params | Train compute | Hyp compute |",
                "|---|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for candidate in (PRIMARY, UNIFORM, "heterogeneous-routing", NARROW, "full-identity-control"):
            current = data["candidates"][candidate]
            hyp = current["hypothesis"]
            acc = current["accounting"]
            lines.append(
                "| {name} | {loss:+.5f} | {kl:.5f} | {top:.2%} | {local:.5f} | {p:.2%} | {tc:.2%} | {hc:.2%} |".format(
                    name=candidate,
                    loss=float(hyp["loss_delta"]["mean"]),
                    kl=float(hyp["kl_teacher_to_candidate"]["mean"]),
                    top=float(hyp["top1_agreement"]["mean"]),
                    local=float(hyp["local_nrmse"]["mean"]),
                    p=float(acc["parameter_ratio_max"]),
                    tc=float(acc["train_compute_ratio_max"]),
                    hc=float(acc["hypothesis_compute_ratio_max"]),
                )
            )
        lines.extend(
            [
                "",
                "### Comparações load-bearing",
                "",
                f"- spectral − uniform loss: `{interval(data['comparisons']['primary_minus_uniform_loss'])}`.",
                f"- spectral − uniform KL: `{interval(data['comparisons']['primary_minus_uniform_kl'])}`.",
                f"- spectral − uniform top-1: `{interval(data['comparisons']['primary_minus_uniform_top1'])}`.",
                f"- spectral − narrow65 loss: `{interval(data['comparisons']['primary_minus_narrow_loss'])}`.",
                "",
                "### Plateau",
                "",
            ]
        )
        for seed, plateau in data["plateau"].items():
            lines.append(
                f"- seed `{seed}`: plateau=`{plateau['plateau_reached']}`, final_step=`{plateau['final_step']}`."
            )
        lines.append("")
    lines.extend(
        [
            "## Leitura adversarial obrigatória",
            "",
            "- Um ganho contra rank uniforme não basta se o teacher não atingiu plateau.",
            "- Alocação espectral deve superar o controle baseado apenas em frequência; caso contrário, a complexidade estrutural não está justificada.",
            "- Compute é um proxy analítico esperado. Nenhuma aceleração de runtime é inferida.",
            "- Resultado em WikiText-2 pequeno não autoriza transplante para checkpoint real.",
            "- Se heterogeneidade falhar nas duas escalas, o controlador dinâmico não será implementado; a linha pós-hoc será despriorizada.",
            "",
            "## Integridade",
            "",
            f"- audit passed: `{audit['audit_passed']}`;",
            f"- mismatches: `{len(audit['mismatches'])}`;",
            f"- provenance: `{audit['provenance_pass']}`;",
            f"- checkpoint hashes: `{audit['checkpoint_pass']}`;",
            f"- data isolation: `{audit['data_pass']}`;",
            "- bootstrap cruzado por seed e documento;",
            "- resultados, checkpoints, dados tokenizados, tokenizer, logs, ambiente e hashes versionados.",
            "",
        ]
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
