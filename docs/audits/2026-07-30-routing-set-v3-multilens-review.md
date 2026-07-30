# Revisão factual adversarial multilentes — routing-set distillation v3.1

**Data:** 30 de julho de 2026  
**Protocolo:** `pre-qwen-alignment-tolerant-routing-set-distillation-v3.1`  
**Veredito mecânico pré-registrado preservado:** `ALIGNMENT_TOLERANT_ROUTING_SET_V3_MECHANISM_SIGNAL`  
**Disposição científica adversarial:** `OBJECTIVE_ONLY_LINE_CLOSED__ARCHITECTURE_CHANGE_REQUIRED`  
**Decisão global:** `NO_GO_FOR_OLMOE_OR_QWEN`

## 1. O que foi validado

A execução é utilizável como evidência no escopo declarado:

- cinco seeds e vinte documentos de hipótese;
- holdouts de confirmação materializados apenas após congelar os candidatos;
- hashes únicos de checkpoints;
- uma única configuração e um único commit de origem;
- bootstrap cruzado por seed e documento;
- auditor independente sem importar o agregador;
- zero divergências no recálculo dos gates;
- controles full-width, aritmética e auditoria de dados aprovados.

O rótulo automático `MECHANISM_SIGNAL` segue corretamente a regra implementada: dois dos quatro votos de mecanismo passaram.

## 2. Sinais positivos reais

Contra o controle expert-wise v2, a v3 melhorou dois componentes de forma estatisticamente estabelecida:

```text
KL v3 - v2:
  média  -0.01419
  IC95   [-0.01929, -0.00908]

cross-error v3 - v2:
  média  -0.01970
  IC95   [-0.02233, -0.01704]
```

O ponto estimado de loss também favoreceu a v3:

```text
v3 - v2 loss:
  média  -0.00345 nat
  IC95   [-0.01307, +0.00602]
```

A v3, portanto, alterou o mecanismo na direção pretendida: recuperou parte do termo cruzado favorável e reduziu KL relativamente ao v2.

## 3. Por que a revisão não aprova uma réplica idêntica

A descrição pré-registrada de `mechanism_signal` dizia que uma réplica fresca somente seria justificada se nenhum gate absoluto falhasse catastroficamente. Essa condição textual não foi codificada no agregador. O agregador exigiu apenas CE absoluto, orçamento, dados e auditoria para emitir `MECHANISM_SIGNAL`.

Quatro gates comportamentais falharam por margens grandes:

| Métrica primária | Resultado | Gate |
|---|---:|---:|
| KL UCB95 | `0.46201` | `<= 0.20` |
| Top-1 LCB95 | `0.64076` | `>= 0.78` |
| local NRMSE UCB95 | `0.38774` | `<= 0.24` |
| counterfactual NRMSE UCB95 | `0.47629` | `<= 0.28` |

Além disso:

```text
v3 - narrow65 cross-error:
  média  +0.07314
  IC95   [+0.06462, +0.08419]
```

A lacuna de coordenação para o narrow65 permaneceu clara.

O próprio counterfactual NRMSE piorou levemente contra v2:

```text
v3 - v2 counterfactual NRMSE:
  média  +0.00189
  IC95   [+0.00122, +0.00257]
```

Top-1 melhorou apenas pontualmente, sem confirmação:

```text
v3 - v2 top-1:
  média  +0.00778
  IC95   [-0.00118, +0.01639]
```

Consequentemente, a v3 não satisfez a condição semântica necessária para repetir a mesma hipótese. Preservamos o veredito automático, mas não aprovamos sua recomendação operacional.

## 4. Lente estatística

A CE foi favorável em quatro seeds e desfavorável na seed `93133`:

```text
91121  -0.07493
92129  -0.07638
93133  +0.03338
94151  -0.07349
95153  -0.05571
```

Ao retirar `93133`, o UCB95 fica abaixo de zero. Ao retirar qualquer outra seed, o intervalo volta a cruzar zero. Isso mostra heterogeneidade seed-específica, não uma vantagem uniforme.

A não-inferioridade contra narrow65 permaneceu inconclusiva:

```text
v3 - narrow65 loss:
  média  -0.03976 nat
  IC95   [-0.08813, +0.00787]
```

A evidência não autoriza afirmar domínio da v3 sobre o baseline.

## 5. Lente de otimização

As curvas da seed `91121` caem fortemente até aproximadamente o passo local 480 e pioram parcialmente no passo 600. No estágio joint, KL cai de `0.28023` para `0.09936`, mas oscila com os minibatches. Isso não demonstra saturação completa.

Entretanto, simplesmente aumentar passos não é a ação de maior valor porque:

- rank-6, a 65% do compute, continua com KL `0.37295`, top-1 `67.64%`, local NRMSE `0.34074` e counterfactual NRMSE `0.42323`;
- quatro gates absolutos falham simultaneamente;
- a v3 melhora alguns termos, mas não corrige a classe funcional da saída conjunta.

Mais otimização pode mover os números, porém não há evidência suficiente de que remova o gargalo estrutural.

## 6. Lente representacional

As v1–v3 tentam resolver coordenação alterando a loss de uma família na qual cada expert é representado por uma base compartilhada mais residual bilateral low-rank. A mistura final continua sendo apenas uma soma ponderada das saídas dos experts.

A v3 consegue escolher um campo de erros com termo cruzado mais favorável que o v2, mas ainda não possui um mecanismo explícito para modelar interação entre os experts selecionados. O objetivo tenta induzir coordenação indiretamente; a arquitetura não a representa diretamente.

## 7. Próxima hipótese aprovada

A próxima classe adicionará uma correção pequena, permutation-invariant e condicionada ao conjunto roteado:

```text
y = sum_e pi_e f_e(x) + C({pi_e, e, z_e}_{e in top-k})
```

onde `z_e` reutiliza os latentes low-rank já calculados no `down`. A correção usará alinhamento por expert, primeiro e segundo momentos ponderados e uma MLP compartilhada zero-inicializada.

Na geometria atual, rank-5 + acoplador `q=8, hidden=8` permanece aproximadamente em:

```text
expert parameters: ~44.3%
dominant matrix compute proxy: 62.5%
```

Portanto, a hipótese adiciona coordenação explícita sem ultrapassar o baseline narrow65 de 65%.

## 8. Decisão

- Aceitar os números e a auditoria da v3: **sim**.
- Aceitar que há dois sinais de mecanismo: **sim**.
- Afirmar não-inferioridade a narrow65: **não**.
- Repetir a mesma v3 sem mudança material: **não aprovado**.
- Continuar ajustando apenas pesos da loss: **linha encerrada**.
- Testar correção arquitetural condicionada ao conjunto roteado: **aprovado**.
- Alterar `NO_GO_FOR_OLMOE_OR_QWEN`: **não**.
