#!/usr/bin/env python3
"""Generate a deterministic technical report from audited v4 metrics."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PRIMARY = "rank5-coupled-q8-h8-v4"
MEAN_ONLY = "rank5-coupled-q8-h8-mean-only-control"
CAPACITY = "rank5-coupled-q12-h8-v4"
V3 = "rank5-v3-frozen-baseline"
RANK6 = "rank6-v3-frozen-capacity"
NARROW = "narrow65-frozen-baseline"
FULL = "full-continuation-control"
DISABLED = "rank5-coupled-q8-h8-v4__coupling-disabled"


def interval(row: dict) -> str:
    return f"{row['mean']:+.6f} [{row['lcb']:+.6f}, {row['ucb']:+.6f}]"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--metrics",
        type=Path,
        default=ROOT / "results/pre-qwen-routing-coupled/v4/metrics.json",
    )
    parser.add_argument(
        "--audit",
        type=Path,
        default=ROOT / "results/pre-qwen-routing-coupled/v4/adversarial-audit/audit.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "docs/results/2026-07-30-routing-coupled-v4-analysis.md",
    )
    args = parser.parse_args()
    metrics = json.loads(args.metrics.read_text(encoding="utf-8"))
    audit = json.loads(args.audit.read_text(encoding="utf-8"))
    candidates = metrics["candidates"]
    comparisons = metrics["comparisons"]
    decision = metrics["decision"]

    lines = [
        "# Routing-coupled residual v4 — análise factual e adversarial",
        "",
        f"**Protocolo:** `{metrics['metadata']['protocol_version']}`  ",
        f"**Veredito automático auditado:** `{decision['verdict']}`  ",
        f"**Auditoria independente:** `{'PASS' if audit['audit_passed'] else 'FAIL'}`  ",
        "**Decisão global:** `NO_GO_FOR_OLMOE_OR_QWEN`",
        "",
        "## Hipótese",
        "",
        "A v4 adiciona uma correção permutation-invariant condicionada ao conjunto "
        "roteado, reutilizando os latentes low-rank do `down`. O objetivo é tornar "
        "a coordenação entre experts representável pela arquitetura, em vez de "
        "esperar que a loss induza covariância favorável indiretamente.",
        "",
        "## Resultados",
        "",
        "| Candidate | Params | Compute | Hyp delta | Hyp UCB95 | KL | Top-1 | Local | CF | Aggregate error | Correction energy |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name in (PRIMARY, MEAN_ONLY, CAPACITY, V3, RANK6, NARROW, FULL):
        row = candidates[name]
        hyp = row["hypothesis"]
        lines.append(
            f"| {name} | {row['parameter_ratio']:.2%} | {row['compute_ratio']:.2%} | "
            f"{hyp['loss_delta']['mean']:+.5f} | {hyp['loss_delta']['ucb']:+.5f} | "
            f"{hyp['kl_teacher_to_candidate']['mean']:.5f} | "
            f"{hyp['top1_agreement']['mean']:.2%} | "
            f"{hyp['local_nrmse']['mean']:.5f} | "
            f"{hyp['counterfactual_nrmse']['mean']:.5f} | "
            f"{hyp['routing_aggregate_error']['mean']:.5f} | "
            f"{hyp['correction_energy_ratio']['mean']:.5f} |"
        )

    lines.extend(["", "## Comparações load-bearing", ""])
    for name in (
        "primary_minus_narrow_loss",
        "primary_minus_rank6_loss",
        "primary_minus_v3_loss",
        "primary_minus_v3_kl",
        "primary_minus_v3_top1",
        "primary_minus_v3_local",
        "primary_minus_v3_counterfactual",
        "primary_minus_mean_only_kl",
        "disabled_minus_primary_kl",
        "disabled_minus_primary_loss",
        "primary_minus_narrow_aggregate_error",
    ):
        lines.append(f"- `{name}`: `{interval(comparisons[name])}`.")

    lines.extend(["", "## Gates", ""])
    for name, value in decision["gates"].items():
        lines.append(f"- `{name}`: `{value}`.")
    lines.extend(
        [
            "",
            f"Behavior-improvement votes versus v3: `{decision['improvement_votes']}`.",
            "",
            "## Causalidade do acoplador",
            "",
            "A candidata `coupling-disabled` usa os mesmos pesos treinados, mas zera "
            "a correção de conjunto. Diferenças positivas de KL/loss em "
            "`disabled-primary` são necessárias para atribuir o resultado ao "
            "acoplador, e não apenas ao refinamento da base rank-5.",
            "",
            f"- KL disabled-primary: `{interval(comparisons['disabled_minus_primary_kl'])}`.",
            f"- loss disabled-primary: `{interval(comparisons['disabled_minus_primary_loss'])}`.",
            "",
            "## Leitura adversarial obrigatória",
            "",
            "- CE favorável sem KL, top-1 e fidelidade local não constitui preservação geral.",
            "- Cross-error é diagnóstico, não gate, pois uma correção de conjunto não possui alocação única por expert.",
            "- O controle mean-only separa o valor do segundo momento do simples aumento de parâmetros.",
            "- O controle q12 testa capacidade, mas não pode resgatar ausência de causalidade do q8 primário.",
            "- Os teachers são checkpoints fixos herdados e não foram demonstrados como plateaued.",
            "- Nenhuma razão analítica é um claim de speedup real.",
            "",
            "## Integridade",
            "",
            f"- auditoria independente: `{'PASS' if audit['audit_passed'] else 'FAIL'}`;",
            f"- divergências: `{len(audit['mismatches'])}`;",
            f"- cobertura: `{audit['coverage_pass']}`;",
            f"- dados: `{audit['data_pass']}`;",
            f"- aritmética: `{audit['arithmetic_pass']}`;",
            f"- source checkpoints: `{audit['source_file_pass']}`;",
            "- bootstrap cruzado por seed e documento;",
            "- leave-one-seed-out persistido no JSON da auditoria;",
            "- checkpoints, registros por janela, logs, ambiente e hashes versionados.",
            "",
        ]
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines), encoding="utf-8")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
