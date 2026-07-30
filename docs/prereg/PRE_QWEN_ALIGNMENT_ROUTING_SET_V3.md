# Pré-registro — distilação acoplada por conjunto roteado v3

**Protocolo:** `pre-qwen-alignment-tolerant-routing-set-distillation-v3.1`  
**Estado:** congelado antes do primeiro treinamento v3 e antes da criação dos novos holdouts.  
**Decisão global preservada:** `NO_GO_FOR_OLMOE_OR_QWEN`.

## Hipótese

O v1 preservou apenas a mistura natural e permitiu cancelamento frágil entre erros.
O v2 reduziu fortemente o erro de cada expert, mas perdeu a covariância favorável
entre experts. A v3 supervisiona um conjunto determinístico de misturas
contrafactuais sobre os mesmos experts roteados.

Para outputs de experts `f_e`, student `s_e`, erro `epsilon_e=s_e-f_e` e pesos
naturais `pi`, o erro da mistura é:

```text
||sum_e pi_e epsilon_e||²
= sum_e pi_e² ||epsilon_e||²
+ 2 sum_{i<j} pi_i pi_j <epsilon_i, epsilon_j>.
```

Em vez de premiar diretamente um termo cruzado negativo — que poderia induzir
anticorrelação artificial — a v3 compara teacher e student sob:

1. pesos naturais;
2. cada mistura leave-one-out;
3. todas as misturas uniformes de pares;
4. uma mistura uniforme sobre o top-k;
5. a geometria relativa dos outputs dos experts no conjunto roteado.

Essas projeções formam um objetivo quadrático positivo sobre o campo de erros e
restringem self-error e interações sem escolher o sinal da covariância como alvo.

## Candidatos

- rank-5 routing-set v3 — primário;
- rank-5 expert-wise v2 — controle contemporâneo;
- rank-5 aggregate-only v1 — controle contemporâneo;
- rank-6 routing-set v3 — controle de capacidade em 65% do compute proxy;
- narrow65 congelado;
- full continuation congelado.

Todos os candidatos rank-5 partem da mesma inicialização SVD, recebem os mesmos
inputs, rotas, batches, updates e orçamento de otimização. Apenas a loss local
muda. O router permanece congelado.

## Dados e isolamento

- teachers e treino são os mesmos do screen teacher-width, explicitamente sem claim
  de plateau;
- hypothesis e OOD são famílias novas `routing-set-*-v3`;
- candidatos são congelados antes da materialização held-out;
- splits são por documento;
- duplicatas exatas e near-duplicates são auditadas;
- unidade estatística: célula seed x documento;
- bootstrap crossed seed/documento com 20.000 draws;
- auditor independente recalcula resultados e gates.

## Gates load-bearing

O rank-5 v3 deve:

- hypothesis UCB95 `<= +0.030 nat`;
- OOD UCB95 `<= +0.050 nat`;
- pior seed hypothesis `<= +0.060 nat`;
- UCB95 de `v3 - narrow65 <= 0`;
- UCB95 de `v3 - expert-v2 <= 0`;
- KL UCB95 `<= 0.200`;
- top-1 LCB95 `>= 0.780`;
- local NRMSE UCB95 `<= 0.240`;
- counterfactual NRMSE UCB95 `<= 0.280`;
- gap do cross-error contra narrow65 UCB95 `<= 0.040`;
- parâmetros e compute estritamente abaixo de 65%;
- dados limpos e auditoria independente PASS.

## Regra de encerramento

Se a v3 não reduzir de forma consistente a divergência comportamental e o gap de
covariância em relação à v2, encerra-se a linha de ajustes apenas na função de
loss. A próxima hipótese deverá alterar a arquitetura ou impor a estrutura
compacta durante o treinamento nativo, antes da especialização dos experts.


## Correção de isolamento após smoke técnico

O smoke de engenharia usou passos drasticamente reduzidos e comprovou que o
pipeline completo executa. Ele também materializou a primeira família de holdout
v3. Por isso, esses documentos foram aposentados antes da execução científica.
A revisão v3.1 mantém candidatos, losses, budgets, gates e seeds de treinamento,
mas substitui hypothesis e OOD por famílias novas e uma nova seed. Nenhuma
métrica do smoke pode ser usada como evidência ou para ajustar thresholds.
