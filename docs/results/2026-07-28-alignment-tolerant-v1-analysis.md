# Análise factual e adversarial — alignment-tolerant shared low-rank v1

**Data:** 28 de julho de 2026  
**Protocolo:** `pre-qwen-alignment-tolerant-shared-low-rank-v1`  
**Veredito pré-registrado:** `ALIGNMENT_TOLERANT_SHARED_LORA_FAIL`  
**Auditoria independente:** `PASS`, zero divergências  
**Decisão global:** `NO_GO_FOR_OLMOE_OR_QWEN` preservada

## 1. Claim estrito testado

O screen perguntou se uma parametrização full-width alignment-tolerant,

\[
W_e = W_{shared} + L_eR_e,
\]

poderia igualar ou superar o baseline convencional teacher-informed narrow65 com
**menos de 65%** dos parâmetros e do proxy de compute dos experts.

O candidato primário foi `shared-lora-r5`:

```text
expert parameters/full: 41.6667%
routed matrix proxy/full: 58.3333%
```

O gate load-bearing exigia UCB95 da diferença pareada
`shared-lora-r5 - narrow65 <= 0`.

## 2. Resultado primário

| Métrica | shared-lora-r5 | narrow65 |
|---|---:|---:|
| Parâmetros dos experts | 41,67% | 65,00% |
| Proxy de compute | 58,33% | 65,00% |
| Hypothesis Δ loss médio | -0,00764 | -0,00388 |
| Hypothesis UCB95 | +0,02589 | +0,01539 |
| Pior seed hypothesis | +0,02640 | +0,02090 |
| OOD Δ loss médio | -0,04516 | +0,00244 |
| OOD UCB95 | +0,02483 | +0,03220 |
| KL hypothesis | 0,37658 | 0,11450 |
| Top-1 hypothesis | 68,44% | 83,34% |
| NRMSE local hypothesis | 0,34904 | 0,17163 |

Comparação pareada:

```text
rank5 - narrow65:
mean:  -0.00376 nat
IC95:  [-0.03058, +0.02035]
```

O efeito pontual favoreceu rank-5, mas o intervalo contém inferioridade possível.
Logo, a não-inferioridade não foi estabelecida.

O controle de capacidade `shared-lora-r6`, com 48,33% dos parâmetros e 65% do
compute, também favoreceu numericamente a arquitetura:

```text
rank6 - narrow65:
mean:  -0.00932 nat
IC95:  [-0.03889, +0.01778]
```

Seu intervalo também cruzou zero.

## 3. O que passou

O rank-5 passou todos os gates abaixo:

- fidelity absoluta em hypothesis;
- fidelity absoluta em OOD;
- limite por seed;
- budget estritamente abaixo de 65% em parâmetros;
- budget estritamente abaixo de 65% em compute;
- controles full hypothesis/OOD;
- fórmulas aritméticas;
- auditoria de overlap dos dados;
- auditoria estatística independente.

O único gate primário falho foi:

```text
primary_vs_narrow65_pass = false
```

Portanto, não é correto resumir o resultado como falha de fidelity absoluta,
aritmética, dados ou auditoria. O `FAIL` é causado pela comparação pareada
pré-registrada contra o baseline forte.

## 4. Leitura adversarial

### 4.1 CE não prova preservação comportamental

Apesar do bom cross-entropy, o rank-5 divergiu muito mais da distribuição do
teacher que narrow65:

```text
KL hypothesis:    0.37658 vs 0.11450
Top-1 agreement:  68.44% vs 83.34%
Local NRMSE:      0.34904 vs 0.17163
```

Assim, o resultado pode combinar:

- capacidade preditiva suficiente para o corpus pequeno;
- alterações funcionais grandes na camada transplantada;
- melhorias fortuitas de CE sem preservação das decisões do teacher.

Isso bloqueia qualquer interpretação forte de preservação geral de capacidade.

### 4.2 Variabilidade por seed

Diferença rank-5 menos narrow65 por seed:

```text
91121: +0.00432
92129: -0.00341
93133: -0.01416
94151: -0.02593
95153: +0.02039
```

Três seeds favorecem rank-5 e duas favorecem narrow65. A heterogeneidade é grande
demais para resolver o gate apenas pelo valor médio.

### 4.3 Mais amostras, isoladamente, não é a melhor próxima ação

O efeito médio pareado do rank-5 é pequeno frente à largura do intervalo. Sob
variabilidade semelhante, apenas multiplicar documentos/seeds seria caro e não
resolveria a divergência de KL/top-1. O próximo teste deve reduzir a heterogeneidade
e aumentar fidelidade funcional, não apenas aumentar poder estatístico.

### 4.4 A otimização não estava claramente saturada

Nas histórias de treino, a loss local dos ranks 4–6 ainda caía até o último ponto
registrado. Para rank-5 na seed 91121, a normalized MSE local passou de `0,11377`
no início do estágio local para `0,06920` no passo 600. Isso não prova que mais
passos resolverão o problema, mas mostra que a hipótese de supervisão/otimização
insuficiente ainda é falsificável.

## 5. Diagnóstico de mecanismo

A distilação v1 supervisiona principalmente a **saída agregada** da camada MoE:

\[
y = \sum_e \pi_e f_e(x).
\]

Esse objetivo permite que erros de especialistas diferentes se cancelem para uma
mistura de routing observada. Em novos inputs, quando pesos e combinações de experts
mudam, o cancelamento pode desaparecer. Isso é consistente com:

- CE agregado aceitável;
- erro local e KL altos;
- top-1 baixo;
- grande interação seed × documento.

A próxima hipótese materialmente nova é supervisionar também cada expert roteado:

\[
\mathcal L_{expert} =
\frac{\sum_{x,e\in topk(x)}\pi_e(x)
\|\hat f_e(x)-f_e(x)\|^2}
{\sum_{x,e\in topk(x)}\pi_e(x)\|f_e(x)\|^2+\epsilon}.
\]

Esse termo impede que a arquitetura obtenha boa loss agregada apenas por
cancelamento entre experts.

## 6. Próximo protocolo

Criar `pre-qwen-alignment-tolerant-expert-distillation-v2` com:

1. novos documentos hypothesis/OOD;
2. os mesmos teachers congelados, explicitamente como screen de mecanismo;
3. rank-5 com o mesmo budget de inferência da v1;
4. perda local agregada + perda por expert roteado + cosine;
5. closed-loop com peso maior de KL;
6. baseline rank-5 v1 congelado e narrow65 congelado;
7. gates de CE, comparação pareada, KL, top-1, budgets e auditoria independente;
8. nenhuma reutilização dos holdouts v1 para seleção.

## 7. Grau das afirmações

| Afirmação | Grau |
|---|---|
| Rank-5 shared low-rank passa os gates absolutos de CE neste screen | `VERIFIED_WITHIN_SCREEN` |
| Rank-5 iguala/supera narrow65 | `NOT_ESTABLISHED` |
| Rank-5 preserva distribuição/comportamento como narrow65 | `REFUTED_IN_V1_SCREEN` |
| A família alignment-tolerant deve ser abandonada | `NOT_SUPPORTED` |
| Supervisão por expert resolverá a falha | `UNTESTED_NEXT_HYPOTHESIS` |
| Há speedup real | `UNTESTED` |
| OLMoE/Qwen está autorizado | `REFUTED__NO_GO` |
