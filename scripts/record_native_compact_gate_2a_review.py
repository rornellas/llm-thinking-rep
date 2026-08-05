#!/usr/bin/env python3
"""Record the adversarial disposition of Native Compact Gate 2A.

This script reads only committed Gate 2A results, its independent audit, and the
explicitly post-hoc rank-utilization diagnostic. It preserves the frozen automatic
verdict and updates canonical research state without promoting post-hoc evidence to
confirmatory status.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from typing import Any, Mapping, MutableMapping, Sequence


FULL = "conventional-full"
NARROW = "conventional-narrow65"
PRIMARY = "native-shared-rank"


def pct(value: float) -> str:
    return f"{100.0 * value:.2f}%"


def ci(value: Mapping[str, Any]) -> str:
    return f"{float(value['mean']):+.6f} [{float(value['lcb']):+.6f}, {float(value['ucb']):+.6f}]"


def unique_append(values: list[str], additions: Sequence[str]) -> list[str]:
    result = list(values)
    for value in additions:
        if value not in result:
            result.append(value)
    return result


def upsert_claim(ledger: MutableMapping[str, Any], claim: Mapping[str, Any]) -> None:
    claims = ledger["claims"]
    for index, current in enumerate(claims):
        if current["id"] == claim["id"]:
            claims[index] = dict(claim)
            return
    claims.append(dict(claim))


def write_review(path: Path, metrics: Mapping[str, Any], diagnostic: Mapping[str, Any], *, results_commit: str) -> None:
    small = metrics["scales"]["small"]
    medium = metrics["scales"]["medium"]
    small_primary = small["accounting"][PRIMARY]
    small_narrow = small["accounting"][NARROW]
    medium_primary = medium["accounting"][PRIMARY]
    medium_narrow = medium["accounting"][NARROW]
    small_expert_saving = 1.0 - float(small_primary["expert_parameter_ratio"]) / float(small_narrow["expert_parameter_ratio"])
    medium_expert_saving = 1.0 - float(medium_primary["expert_parameter_ratio"]) / float(medium_narrow["expert_parameter_ratio"])
    small_total_saving = 1.0 - float(small_primary["total_parameter_ratio"]) / float(small_narrow["total_parameter_ratio"])
    medium_total_saving = 1.0 - float(medium_primary["total_parameter_ratio"]) / float(medium_narrow["total_parameter_ratio"])
    small_compute_saving = 1.0 - float(small_primary["expert_compute_ratio"]) / float(small_narrow["expert_compute_ratio"])
    medium_compute_saving = 1.0 - float(medium_primary["expert_compute_ratio"]) / float(medium_narrow["expert_compute_ratio"])

    lines = [
        "# Revisão factual adversarial multilentes — Native Compact Gate 2A",
        "",
        "**Data:** 5 de agosto de 2026  ",
        f"**Protocolo:** `{metrics['protocol_version']}`  ",
        f"**Veredito automático preservado:** `{metrics['decision']['verdict']}`  ",
        "**Auditoria independente:** `PASS`, zero divergências  ",
        "**Disposição científica adversarial:** `NATIVE_SHARED_RANK_UNDERUTILIZED__CURRENT_PARAMETERIZATION_FAILS__MODE_UTILIZATION_CAUSAL_TEST_REQUIRED`  ",
        "**Decisão global:** `NO_GO_FOR_OLMOE_OR_QWEN`",
        "",
        "## Veredito executivo",
        "",
        "O treinamento nativo não resgatou a formulação shared-base + resíduos bilaterais low-rank sob o protocolo congelado. Nas duas escalas, o candidato preservou a vantagem de parâmetros e um pequeno benefício analítico de compute, mas falhou a não-inferioridade de loss contra `narrow65`, falhou OOD e não apresentou robustez por seed.",
        "",
        "A leitura pós-hoc dos checkpoints identifica uma limitação mais específica: os resíduos de rank nominal 8 e 10 terminaram com stable rank médio próximo de 1 e quase toda a energia no primeiro modo. Os experts efetivos ficaram quase idênticos. Portanto, o Gate 2A refuta a parametrização e a dinâmica de otimização testadas; não refuta universalmente uma base compartilhada cuja capacidade local seja de fato utilizada.",
        "",
        "Consequências:",
        "",
        "- Gate 2B com nested prefixes não está autorizado;",
        "- rank dinâmico, checkpoint real e runtime continuam bloqueados;",
        "- aumentar apenas o rank nominal não é aprovado;",
        "- uma única intervenção causal mínima sobre utilização de modos é aprovada.",
        "",
        "## Integridade",
        "",
        f"- source científico: `{metrics['source_commit']}`;",
        "- workflow científico: `31021913119`;",
        "- workflow de finalização/auditoria: `31024180611`;",
        f"- commit consolidado: `{results_commit}`;",
        "- WikiText-103 oficial, revisão imutável e unidade estatística seed × artigo;",
        "- mesmos batches, updates, optimizer, routers e pesos não-MoE;",
        "- holdout aberto somente após congelamento dos checkpoints;",
        "- auditoria independente: `PASS`, zero mismatches.",
        "",
        "A falha inicial de agregação decorreu de coordenadas locais duplicadas em artigos OOD subdivididos. A correção agregou primeiro dentro da unidade pré-registrada seed × artigo e reutilizou os oito checkpoints congelados sem retreinamento.",
        "",
        "## Resultado quantitativo",
        "",
        "### Small",
        "",
        "| Item | Native | Narrow65 | Efeito |",
        "|---|---:|---:|---:|",
        f"| Expert params/full | {pct(float(small_primary['expert_parameter_ratio']))} | {pct(float(small_narrow['expert_parameter_ratio']))} | {pct(small_expert_saving)} menos |",
        f"| Total params/full | {pct(float(small_primary['total_parameter_ratio']))} | {pct(float(small_narrow['total_parameter_ratio']))} | {pct(small_total_saving)} menos |",
        f"| Expert compute/full | {pct(float(small_primary['expert_compute_ratio']))} | {pct(float(small_narrow['expert_compute_ratio']))} | {pct(small_compute_saving)} menos |",
        f"| Hypothesis loss | {float(small['candidates'][PRIMARY]['final']['hypothesis']['loss']['mean']):.5f} | {float(small['candidates'][NARROW]['final']['hypothesis']['loss']['mean']):.5f} | `{ci(small['comparisons']['primary_minus_narrow_hypothesis'])}` |",
        f"| OOD loss | {float(small['candidates'][PRIMARY]['final']['ood']['loss']['mean']):.5f} | {float(small['candidates'][NARROW]['final']['ood']['loss']['mean']):.5f} | `{ci(small['comparisons']['primary_minus_narrow_ood'])}` |",
        "",
        "Diferença hypothesis por seed (`native − narrow65`):",
    ]
    for seed, value in small["comparisons"]["primary_minus_narrow_hypothesis"]["per_seed"].items():
        lines.append(f"- `{seed}`: `{float(value):+.6f}`.")
    lines.extend([
        "",
        "Duas seeds favoreceram o candidato e duas favoreceram fortemente `narrow65`; não há robustez.",
        "",
        "### Medium",
        "",
        "| Item | Native | Narrow65 | Efeito |",
        "|---|---:|---:|---:|",
        f"| Expert params/full | {pct(float(medium_primary['expert_parameter_ratio']))} | {pct(float(medium_narrow['expert_parameter_ratio']))} | {pct(medium_expert_saving)} menos |",
        f"| Total params/full | {pct(float(medium_primary['total_parameter_ratio']))} | {pct(float(medium_narrow['total_parameter_ratio']))} | {pct(medium_total_saving)} menos |",
        f"| Expert compute/full | {pct(float(medium_primary['expert_compute_ratio']))} | {pct(float(medium_narrow['expert_compute_ratio']))} | {pct(medium_compute_saving)} menos |",
        f"| Hypothesis loss | {float(medium['candidates'][PRIMARY]['final']['hypothesis']['loss']['mean']):.5f} | {float(medium['candidates'][NARROW]['final']['hypothesis']['loss']['mean']):.5f} | `{ci(medium['comparisons']['primary_minus_narrow_hypothesis'])}` |",
        f"| OOD loss | {float(medium['candidates'][PRIMARY]['final']['ood']['loss']['mean']):.5f} | {float(medium['candidates'][NARROW]['final']['ood']['loss']['mean']):.5f} | `{ci(medium['comparisons']['primary_minus_narrow_ood'])}` |",
        "",
        "Diferença hypothesis por seed (`native − narrow65`):",
    ])
    for seed, value in medium["comparisons"]["primary_minus_narrow_hypothesis"]["per_seed"].items():
        lines.append(f"- `{seed}`: `{float(value):+.6f}`.")
    lines.extend([
        "",
        "Todas as quatro seeds medium foram piores; hypothesis e OOD ficaram significativamente acima de zero.",
        "",
        f"A penalidade média equivale a aproximadamente `{100.0 * (math.exp(float(small['comparisons']['primary_minus_narrow_hypothesis']['mean'])) - 1.0):.2f}%` de perplexidade em small e `{100.0 * (math.exp(float(medium['comparisons']['primary_minus_narrow_hypothesis']['mean'])) - 1.0):.2f}%` em medium.",
        "",
        "## Maturidade e routing",
        "",
        "O melhor checkpoint de calibração foi o final em todas as células. Não há claim de plateau. Os slopes terminais do `narrow65` eram mais negativos que os do candidato, portanto prolongar o mesmo treino não é um resgate sustentado pelos dados. Os gates de routing passaram e não houve expert morto; a falha não é explicada por colapso do router.",
        "",
        "## Diagnóstico pós-hoc de rank",
        "",
        "Este diagnóstico não foi pré-registrado e não altera o `FAIL`.",
        "",
        "| Escala | Estado | Cosine experts | Variância centrada | Residual/common | Top-1 energia | Stable rank | Uso nominal |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ])
    for scale in ("small", "medium"):
        values = diagnostic["scales"][scale]
        for phase in ("initial", "final"):
            current = values[phase][PRIMARY]
            lines.append(
                f"| {scale} | {phase} | {float(current['pairwise_cosine_mean']):.5f} | {float(current['centered_expert_variance_ratio_mean']):.5f} | {float(current['residual_common_norm_ratio_mean']):.5f} | {float(current['residual_top1_energy_mean']):.5f} | {float(current['residual_stable_rank_mean']):.3f} | {pct(float(current['stable_rank_utilization_ratio']))} |"
            )
    lines.extend([
        "",
        "Após o treinamento, 97,8–98,8% da energia residual estava no primeiro modo; somente 10–13% do rank nominal foi usado; o cosine entre experts ficou em 0,988–0,991 e a variância entre experts em aproximadamente 1%. O modelo se comportou quase como uma FFN compartilhada com correções expert-local rank-1.",
        "",
        "## Lentes adversariais",
        "",
        "- **Treinamento nativo falhou:** correto para esta parametrização; não é refutação universal.",
        "- **Aumentar rank:** bloqueado, pois o rank já disponível não foi usado.",
        "- **Treinar mais:** não ataca o sinal estrutural e o baseline ainda melhorava mais rápido.",
        "- **Economia de parâmetros basta:** não; o gate exigia qualidade e estabilidade.",
        "- **Rank collapse prova causalidade:** não; exige intervenção confirmatória.",
        "",
        "## Decisão",
        "",
        "```text",
        "NATIVE_COMPACT_GATE_2A_FAIL",
        "NO_GATE_2B",
        "NO_DYNAMIC_RANK",
        "NO_REAL_CHECKPOINT",
        "```",
        "",
        "A parametrização tiny-random bilinear torna-se baseline negativo. A classe shared-base não é encerrada ainda porque sua capacidade nominal não foi utilizada.",
        "",
        "## Próximo experimento autorizado",
        "",
        "`Native Mode Utilization Gate 2A.1` comparará tiny-random, fatores ortogonais balanceados com residual/common inicial de 10% e 25%, `narrow65` e full, usando artigos frescos. O endpoint causal primário será stable-rank utilization e loss em holdout será co-endpoint obrigatório.",
        "",
        "Somente melhora conjunta de utilização e loss autoriza uma replicação medium. Falha encerra a linha de resgate por inicialização. Nested prefixes, heterogeneidade e controle dinâmico permanecem bloqueados.",
        "",
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def write_canonical_verdict(path: Path, metrics: Mapping[str, Any], diagnostic: Mapping[str, Any], *, results_commit: str) -> None:
    small = metrics["scales"]["small"]
    medium = metrics["scales"]["medium"]
    lines = [
        "# Pre-Qwen gate verdict — audited status through Native Compact Gate 2A",
        "",
        "**Frozen decision:** **`NO_GO_FOR_OLMOE_OR_QWEN`**",
        "",
        "## Current conclusions",
        "",
        "1. the methodological harness and independent audits work in the controlled scope;",
        "2. scalar Modal, post-hoc residual, clustered, routing-coupled, and static heterogeneous variants do not move the required quality-compute frontier;",
        "3. native shared-base training preserves a large parameter advantage but fails loss non-inferiority to `narrow65` in two scales;",
        "4. the nominal rank-8/10 residuals collapsed to effective stable rank near 1 and left experts nearly identical;",
        "5. only a causal mode-utilization test is authorized; nested/dynamic rank and real checkpoints remain blocked.",
        "",
        "## Native Compact Gate 2A",
        "",
        f"Protocol: `{metrics['protocol_version']}`  ",
        f"Scientific source: `{metrics['source_commit']}`  ",
        "Scientific workflow: `31021913119`  ",
        "Finalization/audit workflow: `31024180611`  ",
        f"Results commit: `{results_commit}`  ",
        "Independent audit: `PASS`, zero mismatches",
        "",
        "```text",
        f"{metrics['decision']['verdict']}",
        "```",
        "",
        f"- small native − narrow65 hypothesis: `{ci(small['comparisons']['primary_minus_narrow_hypothesis'])}`;",
        f"- medium native − narrow65 hypothesis: `{ci(medium['comparisons']['primary_minus_narrow_hypothesis'])}`;",
        f"- small native budget: expert params `{pct(float(small['accounting'][PRIMARY]['expert_parameter_ratio']))}`, total params `{pct(float(small['accounting'][PRIMARY]['total_parameter_ratio']))}`, compute `{pct(float(small['accounting'][PRIMARY]['expert_compute_ratio']))}`;",
        f"- medium native budget: expert params `{pct(float(medium['accounting'][PRIMARY]['expert_parameter_ratio']))}`, total params `{pct(float(medium['accounting'][PRIMARY]['total_parameter_ratio']))}`, compute `{pct(float(medium['accounting'][PRIMARY]['expert_compute_ratio']))}`.",
        "",
        "The parameter signal is real; the quality-preservation claim failed.",
        "",
        "### Post-hoc diagnosis",
        "",
        f"- small residual stable rank `{float(diagnostic['scales']['small']['final'][PRIMARY]['residual_stable_rank_mean']):.3f}` of nominal 8;",
        f"- medium residual stable rank `{float(diagnostic['scales']['medium']['final'][PRIMARY]['residual_stable_rank_mean']):.3f}` of nominal 10;",
        f"- final expert cosine `{float(diagnostic['scales']['small']['final'][PRIMARY]['pairwise_cosine_mean']):.5f}` small and `{float(diagnostic['scales']['medium']['final'][PRIMARY]['pairwise_cosine_mean']):.5f}` medium.",
        "",
        "This diagnostic does not reverse the failure; it narrows the next causal question.",
        "",
        "## Next action",
        "",
        "```text",
        "native-mode-utilization-gate-2a1-v1",
        "```",
        "",
        "It changes only factor initialization and tests whether balanced multi-mode residuals prevent collapse and improve fresh-heldout loss. Failure closes the initialization-rescue line; success permits one medium replication only.",
        "",
        "```text",
        "NO_GO_FOR_OLMOE_OR_QWEN",
        "NO_GATE_2B",
        "NO_DYNAMIC_RANK",
        "MODE_UTILIZATION_CAUSAL_TEST_NEXT",
        "```",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def update_status(path: Path, metrics: Mapping[str, Any], diagnostic: Mapping[str, Any], *, results_commit: str, now: str) -> None:
    status = json.loads(path.read_text(encoding="utf-8"))
    status["schema_version"] = "pre-qwen-gate-status-v6"
    status["generated_at"] = now
    status["primary_research_line"] = "native_mode_utilization_causal_test"
    status["native_compact_gate_2a"] = {
        "protocol": metrics["protocol_version"],
        "automatic_verdict": metrics["decision"]["verdict"],
        "adversarial_disposition": "NATIVE_SHARED_RANK_UNDERUTILIZED__CURRENT_PARAMETERIZATION_FAILS__MODE_UTILIZATION_CAUSAL_TEST_REQUIRED",
        "independent_audit_passed": bool(metrics["decision"]["audit"]["audit_passed"]),
        "audit_mismatches": len(metrics["decision"]["audit"]["mismatches"]),
        "scientific_source_commit": metrics["source_commit"],
        "results_commit": results_commit,
        "scientific_workflow_run_id": 31021913119,
        "finalize_workflow_run_id": 31024180611,
        "seeds": [202781, 212789, 222793, 232801],
        "small": {
            "primary_expert_parameter_ratio": metrics["scales"]["small"]["accounting"][PRIMARY]["expert_parameter_ratio"],
            "primary_total_parameter_ratio": metrics["scales"]["small"]["accounting"][PRIMARY]["total_parameter_ratio"],
            "primary_expert_compute_ratio": metrics["scales"]["small"]["accounting"][PRIMARY]["expert_compute_ratio"],
            "primary_minus_narrow_hypothesis": metrics["scales"]["small"]["comparisons"]["primary_minus_narrow_hypothesis"],
            "primary_minus_narrow_ood": metrics["scales"]["small"]["comparisons"]["primary_minus_narrow_ood"],
        },
        "medium": {
            "primary_expert_parameter_ratio": metrics["scales"]["medium"]["accounting"][PRIMARY]["expert_parameter_ratio"],
            "primary_total_parameter_ratio": metrics["scales"]["medium"]["accounting"][PRIMARY]["total_parameter_ratio"],
            "primary_expert_compute_ratio": metrics["scales"]["medium"]["accounting"][PRIMARY]["expert_compute_ratio"],
            "primary_minus_narrow_hypothesis": metrics["scales"]["medium"]["comparisons"]["primary_minus_narrow_hypothesis"],
            "primary_minus_narrow_ood": metrics["scales"]["medium"]["comparisons"]["primary_minus_narrow_ood"],
        },
        "gates": metrics["decision"]["gates"],
        "posthoc_rank_utilization": {
            "status": "diagnostic_not_preregistered",
            "cannot_change_frozen_verdict": True,
            "small": diagnostic["scales"]["small"]["final"][PRIMARY],
            "medium": diagnostic["scales"]["medium"]["final"][PRIMARY],
            "rank_collapse_signal_both_scales": True,
            "effective_expert_similarity_signal_both_scales": True,
        },
        "conclusions": {
            "parameter_advantage_verified": True,
            "quality_noninferiority_to_narrow65": False,
            "routing_collapse_explanation": False,
            "nominal_rank_fully_utilized": False,
            "universal_native_shared_rank_refutation": False,
            "gate_2b_authorized": False,
            "mode_utilization_causal_test_authorized": True,
        },
    }
    status["artifact_integrity"].update({
        "native_gate_metrics": "results/native-compact-gate-2a/metrics.json",
        "native_gate_audit": "results/native-compact-gate-2a/adversarial-audit/audit.json",
        "native_gate_multilens_review": "docs/audits/2026-08-05-native-compact-gate-2a-multilens-review.md",
        "native_gate_posthoc_diagnostic": "results/native-compact-gate-2a/posthoc-diagnostics.json",
        "native_gate_posthoc_report": "docs/audits/2026-08-05-native-compact-gate-2a-posthoc-diagnostics.md",
    })
    status["closed_or_blocked_lines"] = unique_append(status["closed_or_blocked_lines"], [
        "native_shared_rank_tiny_random_bilinear_v1_as_sufficient_solution",
        "native_nested_prefix_gate_2b_before_mode_utilization_signal",
        "nominal_rank_increase_before_utilization_fix",
    ])
    status["next_action"] = {
        "protocol_family": "native-mode-utilization-gate-2a1-v1",
        "question": "does balanced multi-mode residual initialization causally prevent effective-rank collapse and improve fresh-heldout loss",
        "candidates": [
            "native_tiny_random_frozen_baseline",
            "native_balanced_orthogonal_residual_ratio_010",
            "native_balanced_orthogonal_residual_ratio_025",
            "conventional_narrow65",
            "conventional_full",
        ],
        "fresh_data_required": True,
        "primary_mechanism_endpoint": "residual_stable_rank_utilization",
        "quality_coendpoint": "paired_fresh_hypothesis_loss",
        "medium_replication_requires_joint_mechanism_and_quality_signal": True,
        "gate_2b_blocked": True,
        "dynamic_rank_blocked": True,
        "scale_up_blocked": True,
    }
    status["limitations"] = [
        "controlled_small_models",
        "fixed_training_budget_without_plateau_claim",
        "wikitext103_screen",
        "small_deterministic_ood_set",
        "posthoc_rank_diagnostic_is_not_confirmatory",
        "no_measured_runtime",
        "no_real_checkpoint",
    ]
    path.write_text(json.dumps(status, indent=2, sort_keys=False) + "\n", encoding="utf-8")


def update_ledger(path: Path, *, now: str) -> None:
    ledger = json.loads(path.read_text(encoding="utf-8"))
    claims = [
        {"id": "C-NATIVE-DATA-001", "claim": "Native Compact Gate 2A used a pinned official WikiText-103 revision, paired from-scratch training, article-level statistics, and heldout arrays opened only after candidate checkpoint freeze.", "grade": "VERIFIED", "independent_check": "results/native-compact-gate-2a/adversarial-audit/audit.json", "limitations": ["controlled small models", "deterministic OOD set"], "primary_evidence": ["data/native-compact-gate-2a/manifest.json", "results/native-compact-gate-2a/adversarial-audit/audit.json", "runs/native-compact-gate-2a/environment.json"]},
        {"id": "C-NATIVE-PARAMETER-001", "claim": "The native shared-rank candidate uses materially fewer expert and total parameters than conventional narrow65, with slightly lower analytical expert compute, in both Gate 2A scales.", "grade": "VERIFIED_WITHIN_GATE_2A", "independent_check": "independent parameter and matrix-operation reconstruction", "limitations": ["quality non-inferiority failed", "analytical compute is not runtime"], "primary_evidence": ["results/native-compact-gate-2a/metrics.json", "docs/audits/2026-08-05-native-compact-gate-2a-multilens-review.md"]},
        {"id": "C-NATIVE-NARROW65-001", "claim": "Native shared-base plus expert low-rank residual training is non-inferior to conventional narrow65 under the frozen Gate 2A hypothesis and OOD loss margins.", "grade": "REFUTED_IN_GATE_2A", "independent_check": "results/native-compact-gate-2a/adversarial-audit/audit.json", "limitations": ["one parameterization and initialization", "two controlled scales", "fixed training budget"], "primary_evidence": ["results/native-compact-gate-2a/metrics.json", "results/native-compact-gate-2a/VERDICT.md", "docs/audits/2026-08-05-native-compact-gate-2a-multilens-review.md"]},
        {"id": "C-NATIVE-ROUTING-COLLAPSE-001", "claim": "The Native Compact Gate 2A quality failure is explained by dead experts or router collapse.", "grade": "REFUTED_IN_GATE_2A", "independent_check": "heldout layer-wise routing health reconstruction", "limitations": ["routing health metrics do not cover every specialization pathology"], "primary_evidence": ["results/native-compact-gate-2a/metrics.json", "results/native-compact-gate-2a/adversarial-audit/audit.json"]},
        {"id": "C-NATIVE-RANK-UTILIZATION-001", "claim": "The nominal rank-8 and rank-10 native expert residuals remain materially multi-mode after training.", "grade": "REFUTED_BY_POSTHOC_DIAGNOSTIC", "independent_check": "reconstruction from frozen checkpoints with exact initial-state replay", "limitations": ["post-hoc and diagnostic", "does not establish causal source of collapse"], "primary_evidence": ["results/native-compact-gate-2a/posthoc-diagnostics.json", "docs/audits/2026-08-05-native-compact-gate-2a-posthoc-diagnostics.md"]},
        {"id": "C-NATIVE-UNIVERSAL-001", "claim": "All native shared-base low-rank MoE training is incapable of approaching conventional narrowing.", "grade": "NOT_ESTABLISHED", "independent_check": "scope and confound review", "limitations": ["Gate 2A residual factors collapsed to effective stable rank near one", "balanced mode utilization has not yet been causally tested"], "primary_evidence": ["results/native-compact-gate-2a/metrics.json", "results/native-compact-gate-2a/posthoc-diagnostics.json", "docs/audits/2026-08-05-native-compact-gate-2a-multilens-review.md"]},
        {"id": "C-GATE-2B-001", "claim": "Native Compact Gate 2A authorizes nested-prefix Gate 2B or dynamic rank as the next primary experiment.", "grade": "REFUTED__BLOCKED", "independent_check": "frozen Gate 2A decision policy plus adversarial review", "limitations": ["a fresh causal mode-utilization signal could later authorize one replication"], "primary_evidence": ["docs/prereg/NATIVE_COMPACT_GATE_2A.md", "results/native-compact-gate-2a/VERDICT.md", "docs/audits/2026-08-05-native-compact-gate-2a-multilens-review.md"]},
    ]
    for claim in claims:
        upsert_claim(ledger, claim)
    for claim in ledger["claims"]:
        if claim["id"] == "C-GO-001":
            claim["primary_evidence"] = unique_append(claim.get("primary_evidence", []), ["results/native-compact-gate-2a/VERDICT.md", "docs/audits/2026-08-05-native-compact-gate-2a-multilens-review.md"])
            claim["limitations"] = ["a materially different architecture plus a passed sequence of controlled gates may change the conclusion"]
        if claim["id"] == "C-RUNTIME-001":
            claim["claim"] = "The current Modal, residual, coupling, heterogeneous, or native shared-rank candidates provide measured runtime speedup."
            claim["limitations"] = ["only analytical operation-count proxies exist"]
    ids = [claim["id"] for claim in ledger["claims"]]
    if len(ids) != len(set(ids)):
        raise RuntimeError("duplicate claim IDs")
    ledger["schema_version"] = "important-claim-ledger-v3"
    ledger["updated_at"] = now
    path.write_text(json.dumps(ledger, indent=2, sort_keys=False) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--diagnostic", type=Path, required=True)
    parser.add_argument("--audit-review", type=Path, required=True)
    parser.add_argument("--canonical-verdict", type=Path, required=True)
    parser.add_argument("--status", type=Path, required=True)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--results-commit", required=True)
    args = parser.parse_args()
    metrics = json.loads(args.metrics.read_text(encoding="utf-8"))
    diagnostic = json.loads(args.diagnostic.read_text(encoding="utf-8"))
    if metrics["decision"]["verdict"] != "NATIVE_COMPACT_GATE_2A_FAIL" or not bool(metrics["decision"]["audit"]["audit_passed"]):
        raise RuntimeError("frozen Gate 2A evidence invalid")
    if not bool(diagnostic["posthoc_not_preregistered"]) or not bool(diagnostic["cannot_change_frozen_verdict"]):
        raise RuntimeError("post-hoc diagnostic guards invalid")
    now = datetime.now(timezone.utc).isoformat()
    write_review(args.audit_review, metrics, diagnostic, results_commit=args.results_commit)
    write_canonical_verdict(args.canonical_verdict, metrics, diagnostic, results_commit=args.results_commit)
    update_status(args.status, metrics, diagnostic, results_commit=args.results_commit, now=now)
    update_ledger(args.ledger, now=now)
    print(args.audit_review)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
