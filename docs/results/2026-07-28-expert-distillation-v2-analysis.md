# Expert-distillation v2 — resultado e análise

**Protocolo:** `pre-qwen-alignment-tolerant-expert-distillation-v2`  
**Branch:** `agent/alignment-tolerant-expert-distill-v2`  
**Estado:** `EXPERIMENT_COMPLETED__PARTIAL_EVIDENCE_CHECKPOINT`  
**Decisão:** `ALIGNMENT_TOLERANT_EXPERT_DISTILL_V2_FAIL`  
**Decisão global:** `NO_GO_FOR_OLMOE_OR_QWEN`

## Hipótese

Manter a arquitetura alignment-tolerant

```text
W_e = W_shared + L_e R_e
```

e substituir o objetivo do v1 por supervisão explícita de cada especialista roteado, combinando:

- erro normalizado por especialista;
- cosine loss por especialista;
- erro da mistura MoE;
- KL dos logits finais;
- cross-entropy de linguagem.

A hipótese era que o v1 falhava porque erros de especialistas diferentes se cancelavam no conjunto de rotas observado durante o treino, sem que cada especialista preservasse sua função.

## Candidata primária

```text
shared-lora-r5
expert parameter ratio: 41.6667%
routed compute proxy:   58.3333%
```

## Resultado agregado

| Métrica | Média | LCB95 | UCB95 | Estado |
|---|---:|---:|---:|---|
| Hypothesis Δ loss | -0.01735 | -0.05138 | +0.01366 | absoluto passa |
| OOD Δ loss | -0.06017 | -0.11420 | -0.00518 | absoluto passa |
| KL hypothesis | 0.36538 | 0.34437 | 0.38879 | falha |
| Top-1 agreement | 68.34% | 66.88% | 69.85% | falha |
| NRMSE local | 0.33787 | 0.31743 | 0.35550 | falha |
| NRMSE por especialista | 0.42030 | 0.39980 | 0.43701 | falha |

Comparação pareada contra o baseline convencional teacher-informed de 65%:

```text
rank5-v2 - narrow65, hypothesis Δ loss
mean: -0.01274 nat
95%:  [-0.05144, +0.02407]
```

O ponto estimado favorece o v2, mas o intervalo ainda permite inferioridade; a não-inferioridade não foi estabelecida.

## Comparação com o objetivo v1

```text
rank5-v2 - rank5-v1, Δ loss
mean: +0.00219 nat
95%:  [-0.02431, +0.03287]
```

Não houve melhora estabelecida de cross-entropy, KL, top-1 ou NRMSE agregado.

A melhora robusta ocorreu no erro individual dos especialistas:

```text
NRMSE por especialista, v2 - v1
mean: -0.30858
95%:  [-0.32293, -0.29416]
```

## Diagnóstico de covariância

A decomposição do erro da mistura foi analisada como:

```text
||Σ_e π_e ε_e||²
= Σ_e π_e² ||ε_e||²
+ 2 Σ_{i<j} π_i π_j <ε_i, ε_j>
```

### Rank-5 v2 versus rank-5 v1

| Componente normalizado | v2 | v1 |
|---|---:|---:|
| erro próprio | 0.15131 | 0.45331 |
| termo cruzado | -0.03467 | -0.33423 |
| erro agregado | 0.11664 | 0.11908 |

Comparações crossed:

```text
v2 - v1, erro próprio:  -0.30200  95% [-0.32618, -0.27668]
v2 - v1, termo cruzado: +0.29956  95% [+0.27477, +0.32372]
v2 - v1, agregado:      -0.00244  95% [-0.00608, +0.00037]
```

O v2 reduziu fortemente os erros individuais, mas perdeu o cancelamento favorável entre especialistas. O erro final da mistura praticamente não mudou.

### Rank-5 v2 versus narrow65

| Componente normalizado | v2 | narrow65 |
|---|---:|---:|
| erro próprio | 0.15131 | 0.15324 |
| termo cruzado | -0.03467 | -0.12413 |
| erro agregado | 0.11664 | 0.02911 |

```text
v2 - narrow65, erro próprio:  -0.00193  95% [-0.01319, +0.00929]
v2 - narrow65, termo cruzado: +0.08946  95% [+0.08232, +0.09677]
v2 - narrow65, agregado:      +0.08753  95% [+0.07720, +0.09694]
```

## Conclusão científica estreita

O experimento refuta, neste escopo, a hipótese de que a falta de supervisão individual por especialista era o único gargalo do v1.

A supervisão por especialista reduz o erro individual de forma forte e reproduzível, mas não reproduz a estrutura de coordenação/covariância entre os especialistas. O narrow65 obtém parte substancial de sua fidelidade agregada por cancelamento favorável entre erros roteados.

O próximo protocolo deve testar distilação acoplada por conjunto de routing, preservando simultaneamente:

1. erro individual;
2. erro da mistura;
3. termos cruzados/covariância dos erros;
4. KL e top-1 finais.

## Integridade e limitações deste checkpoint

- O código-fonte do v2 está preservado na branch como bundle Base64 fragmentado em `.github/import/expert-distill-v2-source/`.
- Este commit preserva o veredito, as métricas load-bearing e o diagnóstico adversarial.
- Checkpoints binários e JSONs completos por seed ainda não estão incluídos neste checkpoint remoto.
- Portanto, o estado correto é `PARTIAL_EVIDENCE_CHECKPOINT`, não `FULL_REPRODUCIBILITY_PASS`.
- Nenhum PR foi criado.
