# Handoff técnico-científico e operacional
## Representação compacta, Modal-MoE e certificação pré-Qwen

**Data de corte:** 28 de julho de 2026  
**Repositório:** `rornellas/llm-thinking-rep`  
**PR remoto em andamento:** `#1 — research: audited pre-Qwen certification and FAIL-resolution experiments`  
**Decisão científica global congelada:** **`NO_GO_FOR_OLMOE_OR_QWEN`**  
**Resultado positivo mais recente:** **`TEACHER_WIDTH_65_REPLICATION_PASS`**  
**Próximo gate recomendado:** arquitetura **alignment-tolerant / free-basis / shared-basis** que supere o baseline convencional teacher-informed de 65%.

---

# 0. Como usar este handoff

Este documento é a fonte de continuidade para outra sessão, agente ou pesquisador. Ele substitui como visão consolidada o handoff de 27 de julho de 2026 e os resumos intermediários produzidos durante a certificação pré-Qwen.

Ele foi escrito para permitir que a próxima sessão:

1. compreenda a hipótese original e o que restou dela;
2. saiba exatamente o que foi demonstrado, refutado ou ainda não testado;
3. encontre os artefatos primários, commits, hashes, configs e relatórios;
4. não reutilize holdouts consumidos nem relaxe gates retrospectivamente;
5. reconcilie corretamente o Git local com a branch e o PR remoto;
6. execute o próximo experimento sem reconstruir decisões ou repetir linhas encerradas;
7. mantenha o padrão obrigatório de double-check factual e revisão adversarial.

## Regra de leitura

Antes de implementar qualquer novo teste, ler nesta ordem:

1. este handoff;
2. `docs/methodology/IMPORTANT_CLAIM_VERIFICATION_STANDARD.md`;
3. `results/VERDICT.md`;
4. `docs/results/2026-07-28-teacher-informed-width-fresh-replication.md`;
5. `docs/results/2026-07-28-fail-resolution-mathematical-analysis.md`;
6. o pré-registro do próximo experimento, que deve ser criado **antes** de qualquer métrica held-out.

---

# 1. Estado executivo

## 1.1 Situação atual em uma frase

A pesquisa demonstrou que a estrutura Modal é funcional e transplantável em modelos pequenos, mas a forma **escalar global pós-hoc** não venceu um baseline convencional com compute equivalente; em contrapartida, um student convencional teacher-informed mantendo **65% da largura dos especialistas** passou uma replicação fresca de cinco seeds e agora define o baseline mínimo que uma arquitetura compartilhada precisa superar.

## 1.2 Decisões atuais

| Pergunta | Estado |
|---|---|
| O harness metodológico detecta wiring errado, leakage e controles conhecidos? | **PASS no escopo pequeno testado** |
| Scalar Modal pode representar uma função genuinamente Modal conhecida? | **PASS** |
| Scalar Modal K2 pode substituir funcionalmente uma camada MoE convencional pequena? | **SUPPORTED / FUNCTIONAL_ONLY** |
| Scalar Modal K2 vence narrow convencional em compute dominante semelhante? | **REFUTADO nos screens atuais** |
| Aumentar K preserva compute? | **Não; fidelidade aproxima-se somente em K não econômico** |
| Resíduos low-rank específicos resolvem o equal-compute fail? | **Não no desenho testado** |
| Bases clusterizadas por routing resolvem? | **Não no desenho testado** |
| Seleção de coordenadas existentes a 35% resolve? | **Não** |
| Student convencional teacher-informed a 65% preserva qualidade? | **PASS em cinco teachers pequenos novos** |
| Há speedup real medido? | **UNTESTED** |
| Está autorizado testar OLMoE/Qwen? | **NÃO** |

## 1.3 Resultado positivo mais recente

Protocolo:

```text
pre-qwen-teacher-informed-width-fresh-replication-v1.1
```

Resultado primário:

```text
65% conventional narrow student
expert parameters/full:       65.000%
routed matrix compute/full:   65.000%
hypothesis Δ loss:            +0.01325 nat
crossed UCB95:                +0.01914 nat
perplexity ratio mean:        1.01334x
perplexity ratio at UCB95:    1.01932x
worst seed:                   +0.01608 nat
```

Comparação com 50%:

```text
65% - 50%:
mean:  -0.01648 nat
95%:   [-0.02611, -0.00570]
```

Conclusão estreita:

> Em cinco teachers MoE pequenos e novos, o student convencional teacher-informed de 65% passou todos os gates pré-registrados e foi superior ao student de 50%.

Limite:

> Esse resultado não é Modal, não mede runtime e não autoriza extrapolação para modelos reais.

---

# 2. Hierarquia de fontes e confiança

## 2.1 Fonte canônica local mais completa

Artefato:

```text
pre-qwen-teacher-width-fresh-v1.git.bundle
```

SHA-256 verificado:

```text
8f0ca3009e09f2130aad564fc52bb5597e5de12c356fa53d27b0b875545f8036
```

Branch contida:

```text
agent/teacher-informed-width-replication-v1
```

Head:

```text
c8f4841e47b7c7bd9df3ac30e06c395f57c3b1d9
results: record audited teacher-informed width replication
```

O bundle registra histórico completo e foi validado com `git bundle verify`.

## 2.2 Pacote compacto de resultados

Artefato:

```text
pre-qwen-teacher-width-fresh-v1-results.zip
```

SHA-256 verificado:

```text
f3d5d8d8104c429d89118bae59ba5ffc1fd7e93396229d9e8cb925b6eccf42a7
```

## 2.3 Estado remoto no GitHub, verificado na data de corte

PR:

```text
https://github.com/rornellas/llm-thinking-rep/pull/1
```

Metadados:

```text
state:       open
draft:       true
mergeable:   true
base:        main
base SHA:    3de79997fed2e83fef7201d8d74c7fb0d17450df
head branch: agent/pre-qwen-certification-v2
head SHA:    7e868028671c7ecf513f7fbfc1b6e2320506ec76
commits:     56
files:       34
```

O PR remoto **não contém** o relatório:

```text
docs/results/2026-07-28-teacher-informed-width-fresh-replication.md
```

A leitura desse caminho na branch remota retornou `404`. Portanto, o resultado de 65% está preservado no bundle e nos artefatos locais, mas não está integrado ao PR remoto atual.

## 2.4 Import remoto incompleto

A branch do PR contém:

```text
.github/import/native-alignment-v1.b64
.github/workflows/_import-native-alignment.yml
```

O workflow pretendia extrair um bundle de “native alignment screens and convergence protocol”, validar:

```text
tests/test_alignment_tolerant.py
tests/test_modal.py
tests/test_tiny_lm_smoke.py
```

e comitar:

```text
research: add native alignment screens and convergence protocol
```

O arquivo compactado esperado possui SHA-256:

```text
9e57be34c7d67cc5a90684be8eb3798be4de4b350f9675f9cb1b8039cd8bec8c
```

Porém, os arquivos extraídos não aparecem entre os arquivos modificados do PR e o importador continua presente. Tratar isso como **import não concluído**. Antes de reimplementar qualquer arquitetura alignment-tolerant, extrair e inspecionar esse bundle.

## 2.5 Handoff histórico

Arquivo histórico:

```text
HANDOFF_MODAL_LLM_RESEARCH_2026-07-27(1).md
```

É útil para a trajetória anterior — Testes 0.x, 2.x, 4.x, 5.5 e 6.2 — mas sua recomendação de avançar diretamente ao transplante real foi superada pela auditoria pré-Qwen. A decisão atual é `NO_GO`.

---

# 3. Verificação factual realizada para este handoff

## 3.1 Testes automatizados

No checkout reconstruído do bundle:

```bash
PYTHONPATH=. pytest -q
```

Resultado:

```text
17 passed
```

Há um warning não bloqueante em `test_conditional_pruning.py` sobre converter tensor com `requires_grad=True` para float sem `detach()`.

### Observação operacional

Executar apenas:

```bash
pytest -q
```

sem instalar o pacote ou definir `PYTHONPATH` falha com `ModuleNotFoundError: pre_qwen_certification`.

Usar uma destas opções:

```bash
python -m pip install -e .
python -m pytest -q
```

ou:

```bash
PYTHONPATH=. pytest -q
```

## 3.2 Manifesto específico do experimento de 65%

Comando:

```bash
sha256sum -c results/pre-qwen-teacher-width-fresh/v1/sha256sums.txt
```

Resultado:

```text
24/24 artefatos: OK
```

Inclui:

- cinco checkpoints congelados;
- cinco registros por seed;
- logs e status;
- métricas agregadas;
- relatório;
- auditoria independente.

## 3.3 Reexecução do auditor independente

O auditor espera que `metrics.json` e os arquivos por seed estejam no próprio `output-dir`. Para reexecutar sem alterar os artefatos canônicos:

```bash
cp -a results/pre-qwen-teacher-width-fresh/v1 /tmp/teacher-width-audit
PYTHONPATH=. python scripts/audit_teacher_width_fresh.py \
  --config configs/pre_qwen_teacher_width_fresh_v1.yaml \
  --output-dir /tmp/teacher-width-audit
cat /tmp/teacher-width-audit/adversarial-audit/VERDICT.md
```

Resultado revalidado:

```text
Audit: PASS
Mismatches: 0
```

## 3.4 Defeito conhecido: manifesto global desatualizado

O arquivo:

```text
DELIVERY_MANIFEST.sha256
```

não corresponde integralmente ao estado final do bundle. Dois arquivos divergem:

```text
results/PRE_QWEN_GATE_STATUS.json
results/VERDICT.md
```

Isso aconteceu porque eles foram atualizados por experimentos posteriores sem regenerar o manifesto global.

O manifesto específico do experimento de 65% passa. Mesmo assim, antes de publicar ou declarar integridade global, regenerar `DELIVERY_MANIFEST.sha256` e validar novamente.

## 3.5 Defeito conhecido: runner antigo não é autocontido

O script:

```text
scripts/run_conditional_ablation_seed.py
```

não pode ser executado nesta reconstrução porque importa módulos ausentes:

```text
pre_qwen_certification.dimension_data
pre_qwen_certification.fail_resolution
pre_qwen_certification.selective_data
pre_qwen_certification.attribution_data
```

Os resultados canônicos da ablação condicional e seus auditores estão presentes, mas o pipeline de rerun não está integralmente autocontido no bundle atual.

Antes de prometer rerun completo desses experimentos, recuperar os módulos da branch remota, dos bundles intermediários ou do bundle `native-alignment-v1`.

---

# 4. Objetivo científico

A ambição de longo prazo é encontrar uma arquitetura de linguagem que melhore simultaneamente:

- qualidade por parâmetro;
- qualidade por FLOP ativo;
- tráfego de memória;
- latência e throughput;
- refinamento adaptativo;
- compatibilidade com linguagem, raciocínio, código, contexto e ferramentas;
- possibilidade futura de mapeamento eficiente para hardware especializado.

A pesquisa começou com uma intuição de representação modal/temporal — modos, frequências, fase, pulsos e ciclos — mas foi reformulada em uma hipótese computacional verificável.

A formulação sobrevivente é:

> Funções especialistas podem compartilhar parte substancial da computação, desde que a especialização não seja reduzida a poucos coeficientes escalares globais; ela provavelmente exige subespaços, transformações e budgets específicos por especialista ou precisa ser imposta durante o treinamento antes da especialização endurecer.

---

# 5. Arquitetura Modal original

## 5.1 Especialista SwiGLU

Para especialista `e`:

\[
f_e(x)=D_e\left[\operatorname{SiLU}(G_ex)\odot U_ex\right]
\]

## 5.2 Modal-MoE escalar

\[
G_e=G_0+\sum_{k=1}^{K}a^G_{e,k}G_k
\]

\[
U_e=U_0+\sum_{k=1}^{K}a^U_{e,k}U_k
\]

\[
D_e=D_0+\sum_{k=1}^{K}a^D_{e,k}D_k
\]

As matrizes compartilhadas são full-rank. O baixo rank está no eixo dos especialistas.

## 5.3 Execução direta

\[
g_k=G_kx,\quad u_k=U_kx
\]

\[
g_e=g_0+\sum_ka^G_{e,k}g_k
\]

\[
u_e=u_0+\sum_ka^U_{e,k}u_k
\]

\[
z_e=\operatorname{SiLU}(g_e)\odot u_e
\]

A saída roteada pode ser reorganizada como:

\[
s_k=\sum_e\pi_ea^D_{e,k}z_e
\]

\[
y=\sum_kD_ks_k
\]

Essa agregação antes do `down` é exata para a parametrização linear do `down`; não exige reconstruir `D_e`.

## 5.4 Razões idealizadas

Com `E` especialistas, top-`T` e `K` modos residuais:

\[
\text{expert parameter ratio}\approx\frac{K+1}{E}
\]

\[
\text{dominant matrix compute ratio}\approx\frac{K+1}{T}
\]

Para top-4:

| K | Compute matricial idealizado |
|---:|---:|
| 0 | 25% |
| 1 | 50% |
| 2 | 75% |
| 3 | 100% |
| 7 | 200% |
| 8 | 225% |

O problema observado é que os teachers convencionais mais desenvolvidos só se aproximam da fidelidade em K alto, quando a economia de compute desaparece.

---

# 6. Diagnóstico matemático do principal FAIL

Empilhando os pesos dos especialistas:

\[
\mathcal W=
\begin{bmatrix}
\operatorname{vec}(W_1)^\top\\
\vdots\\
\operatorname{vec}(W_E)^\top
\end{bmatrix}
\]

O Modal escalar impõe:

\[
\operatorname{rank}(\mathcal W)\le K+1
\]

Pelo teorema de Eckart–Young, a melhor aproximação linear de rank `K+1` preserva somente os primeiros valores singulares e incorre em erro residual:

\[
\sum_{j>K+1}\sigma_j^2
\]

A distilação funcional pode encontrar uma solução melhor que PCA em Frobenius, porque pondera regiões de ativação e o suffix do modelo, mas não remove a restrição de dimensão no eixo dos especialistas.

Conclusão:

> O global scalar expert-axis family possui um gargalo representacional real para especialistas convencionais já especializados. O problema não é apenas optimizer ou número de passos.

---

# 7. Evolução experimental

## 7.1 Fase histórica anterior à certificação pré-Qwen

O handoff de 27 de julho registra:

- PCA e decomposições post-hoc em OLMoE: `FAIL`;
- Modal-MoE treinado desde o início: `PASS` em pequenos modelos;
- vantagens compute-matched em WikiText-2 pequeno;
- modos progressivos e controlador de utilidade marginal: `PASS`;
- patches causais + Modal no mesmo checkpoint: `PASS` no Teste 5.5;
- gargalo de eventos top-k/quantizado: screen positivo no Teste 6.2.

Esses resultados motivaram o transplante, mas a auditoria posterior mostrou que eram insuficientes para autorizar OLMoE/Qwen.

### Cuidado sobre Teste 5.5

O throughput de aproximadamente `5,75x` era CPU teacher-forced contra um baseline convencional com loop Python. Não equivale a speedup autoregressivo ou GPU.

### Cuidado sobre Teste 6.2

Era single-seed, selecionava entre políticas no mesmo conjunto e estimava tráfego residual sem todos os custos de escala, índices e top-k. Tratar como evidência exploratória de representabilidade, não confirmação de hardware.

---

## 7.2 Certificação sintética pré-Qwen

Decisão:

```text
PRE_QWEN_SYNTHETIC_CERTIFIED
```

Resultados principais:

| Teste | Resultado |
|---|---:|
| Modal direto vs reconstruído | NRMSE `2,50e-7` |
| Capture/replay correto | NRMSE `0` |
| Fault injection mais fraco | NRMSE `0,03516` |
| Recuperação de teacher Modal K2 | aproximadamente `2,5e-7` |
| K1 insuficiente para teacher K2 | NRMSE `0,20–0,28` |
| Teacher não-Modal com K2 | NRMSE `0,96–0,98` |
| Menor dano por ablação de código | NRMSE `0,434` |
| Controle input-blind | NRMSE aproximadamente `1,0` |
| Targets embaralhados | NRMSE aproximadamente `1,12` |

A curva entre teacher perfeitamente Modal e teacher progressivamente não-Modal foi monotônica.

Conclusão:

> O harness discrimina verdade conhecida, falsidade conhecida, leakage e wiring errado no escopo sintético.

---

## 7.3 Transplante convencional controlado v1.3

Decisão:

```text
CONTROLLED_TRANSPLANT_FUNCTIONAL_ONLY
```

Scalar Modal K2:

```text
Δ loss vs teacher:
mean  +0.00736 nat
95%   [+0.00322, +0.01261]
```

Modal menos narrow compute-matched:

```text
mean  +0.00645 nat
95%   [+0.00139, +0.01344]
```

Gates:

| Gate | Resultado |
|---|---|
| Fidelidade closed-loop | PASS |
| Compressão | PASS |
| Causalidade dos códigos | PASS |
| Vantagem sobre narrow compute-matched | FAIL |

Conclusão:

> A função convencional é transplantável, mas isso não estabelece vantagem arquitetural.

---

## 7.4 Screens de resolução do FAIL

### Mais otimização e closed-loop

Closed-loop ajudou materialmente em teachers curtos, mas o ganho não sobreviveu a teachers de orçamento maior.

### Teachers de 900 passos

Os diretórios históricos chamados `mature-*` contêm teachers de 900 passos que ainda estavam melhorando. Não chamá-los de convergidos.

Scalar K2 closed-loop:

```text
parameters/full:       25.208%
adjusted compute/full: 83.333%
Δ loss:                +0.05607 nat
UCB95:                 +0.08135
```

### Curva K0–K8

O melhor rank compute-reducing para top-4 é K2. Resultado:

```text
K2 Δ loss: +0.09665 nat
UCB95:      +0.13612
```

K7–K8 aproximaram qualidade, porém com aproximadamente 200%–225% do compute matricial original.

### Resíduos low-rank específicos

Melhor configuração:

```text
K1 + residual rank 3 + annealing
parameters/full:       36.771%
adjusted compute/full: 74.167%
Δ loss:                +0.04082 nat
UCB95:                 +0.05992
```

Contra baseline parameter-matched:

```text
-0.01861 nat; UCB95 -0.00213
```

Contra baseline matrix-matched:

```text
+0.03464 nat; LCB95 +0.01416
```

Conclusão:

> Há sinal de eficiência por parâmetro, não de qualidade por compute.

### Bases clusterizadas

Melhor screen:

```text
G2/R3 annealed
parameters/full:       36.667%
adjusted compute/full: 70.000%
Δ loss:                +0.05922 nat
UCB95:                 +0.07829
```

Não venceu o baseline compute-matched.

### Whole-expert hot/cold

Preservar alguns especialistas inteiros e comprimir os demais não estabeleceu vantagem sobre random ou narrows.

---

## 7.5 Fixed-width attribution 35%

Decisão:

```text
FIXED_WIDTH_ATTRIBUTION_FAIL
```

| Método | Δ loss | UCB95 |
|---|---:|---:|
| Fisher 35% | +0,06721 | +0,08051 |
| Output energy 35% | +0,05758 | +0,07259 |
| Magnitude 35% | +0,08439 | +0,09950 |
| Random 35% | +0,10298 | +0,12030 |
| Fisher 50% secundário | +0,02033 | +0,02867 |

Fisher venceu magnitude e random, mas não output energy e falhou o gate absoluto em 35%.

---

## 7.6 Conditional loss-ablation 35%

Decisão:

```text
CONDITIONAL_ABLATION_FAIL
```

O método recalculava utilidade marginal depois de cada etapa:

```text
40 -> 36 -> 32 -> 28 -> 24 -> 20 -> 17 -> 14
```

Resultado:

| Candidato | Δ loss | UCB95 |
|---|---:|---:|
| Conditional iterative | +0,08712 | +0,10285 |
| Exact one-shot | +0,08039 | +0,09752 |
| Fisher | +0,07233 | +0,08407 |
| Output energy | +0,06603 | +0,07962 |
| Magnitude | +0,10707 | +0,12483 |
| Random | +0,11155 | +0,12966 |
| Full continuation | -0,01540 | -0,01191 |

O iterativo foi pior que output energy e Fisher. A estabilidade entre metades também falhou:

```text
Jaccard:  0.226–0.292
Spearman: 0.245–0.488
```

Conclusão:

> O problema não é apenas escolher melhor um subset das coordenadas do teacher. A família de subsets com 35% da largura é insuficiente neste escopo.

---

## 7.7 Replicação fresca teacher-informed 65%

Decisão:

```text
TEACHER_WIDTH_65_REPLICATION_PASS
```

### Desenho

```text
seeds:         91121, 92129, 93133, 94151, 95153
layers:        2
d_model:       24
d_ff:          40
experts:       12
top-k:         4
teacher steps: 2200
train docs:    28
hyp docs:      20
OOD docs:      12
bootstrap:     20000 crossed seed/document
```

Candidatos:

```text
35% anchor negativo
50% comparator
65% primário
75% capacity control
100% full continuation
```

### Resultado

| Candidato | Params | Compute | Hyp Δ | UCB95 | OOD Δ | UCB95 | Worst seed |
|---|---:|---:|---:|---:|---:|---:|---:|
| 35% | 35% | 35% | +0,08626 | +0,10517 | -0,02408 | +0,03363 | +0,10496 |
| 50% | 50% | 50% | +0,02973 | +0,03984 | -0,01345 | +0,03085 | +0,04134 |
| **65%** | **65%** | **65%** | **+0,01325** | **+0,01914** | **-0,01157** | **+0,01021** | **+0,01608** |
| 75% | 75% | 75% | +0,00412 | +0,01144 | -0,00942 | +0,01467 | +0,01217 |
| Full | 100% | 100% | -0,00431 | +0,00207 | -0,00208 | +0,00860 | +0,00569 |

### Robustez

Leave-one-seed-out do candidato 65%:

```text
UCB95 range: +0.01877 a +0.02102
```

Todos abaixo da margem de `+0,030`.

### Limites

- teachers não demonstradamente convergidos;
- character LM sintético;
- uma geometria pequena;
- OOD CE não implica paridade de distribuição;
- KL OOD `0,13431` e top-1 OOD `81,11%`;
- nenhum runtime medido;
- resultado convencional, não Modal.

---

# 8. Crença atual na teoria

## 8.1 Hipótese enfraquecida/refutada

Baixa confiança:

> Um MoE convencional maduro pode ser comprimido pós-hoc em poucas matrizes full-rank globais combinadas apenas por coeficientes escalares por especialista, preservando vantagem de compute.

Confiança subjetiva atual: aproximadamente `10%–20%`.

## 8.2 Hipótese sobrevivente

Confiança moderada:

> Especialistas compartilham estrutura e computação, mas precisam de sistemas de coordenadas ou subespaços específicos, side factors, ranks heterogêneos ou treinamento nativo sob a restrição.

## 8.3 Caminho em que há mais confiança

> Impor uma arquitetura compacta durante o treinamento ou destilar desde cedo, antes que a especialização convencional se torne geometricamente incompatível.

Confiança subjetiva: aproximadamente `65%–75%` de mover a fronteira qualidade × compute em escala controlada, não de alcançar imediatamente paridade com Qwen grande.

---

# 9. Mentalidade e modo de operação obrigatórios

## 9.1 Buscar falsificação, não confirmação

A pergunta não é “como fazer passar?”. É:

```text
qual observação refutaria esta hipótese?
qual baseline forte explica o mesmo ganho?
qual custo está escondido?
qual dependência entre exemplos estreita artificialmente o intervalo?
```

## 9.2 Toda afirmação importante exige double-check

Procedimento obrigatório:

1. formular claim estreito;
2. preservar evidência primária;
3. recalcular por implementação independente;
4. executar auditoria adversarial/subagente;
5. testar controles positivos e negativos;
6. reportar seeds e clusters;
7. usar holdout selado para confirmação;
8. atribuir grau: `VERIFIED`, `SUPPORTED`, `PROVISIONAL`, `REFUTED` ou `UNTESTED`;
9. não mudar gate silenciosamente;
10. preservar negativos.

## 9.3 Diferenciar métricas

Nunca tratar como equivalentes:

- parâmetros armazenados;
- parâmetros ativos;
- FLOPs/MACs analíticos;
- tráfego de memória;
- memória residente;
- throughput teacher-forced;
- latência autoregressiva;
- speedup end-to-end.

## 9.4 Baselines primeiro

Toda nova arquitetura deve comparar contra:

- teacher;
- narrow teacher-informed 65%;
- narrow parameter-matched;
- narrow compute-matched;
- free-basis narrow;
- shared-base + LoRA forte;
- quantização/pruning quando pertinente.

## 9.5 Nunca reutilizar holdout consumido

Os documentos hypothesis/OOD do protocolo de 65% foram consumidos para essa arquitetura. O sucessor deve usar novos documentos e novo protocolo.

## 9.6 Convergência precisa ser demonstrada

Não chamar teacher de “maduro” por número fixo de passos.

Definir plateau, por exemplo:

```text
média de melhora da validation loss por janela de N checkpoints
< epsilon por M avaliações consecutivas
```

ou orçamento máximo + análise explícita de tendência.

## 9.7 Resultado negativo é conclusão válida

Não transformar `FAIL` em `PASS` narrativo. Um fail deve:

- permanecer no ledger;
- bloquear scale-up;
- gerar nova hipótese materialmente diferente;
- nunca justificar relaxamento do gate no mesmo holdout.

---

# 10. Próxima classe arquitetural

A próxima tentativa não deve selecionar coordenadas existentes nem aumentar apenas K.

## 10.1 Free-basis narrow

Primeiro baseline:

\[
\hat f_e(x)=\sum_{j=1}^{m}\hat d_{e,j}\operatorname{SiLU}(\hat g_{e,j}^{\top}x)(\hat u_{e,j}^{\top}x)
\]

com `m` reduzido e todas as coordenadas livres.

Isso testa se o fail de 35% é devido à restrição de subset ou a capacidade insuficiente.

## 10.2 Shared base + LoRA independente

\[
W_e=W_0+A_eB_e
\]

Baseline forte e expressivo.

## 10.3 Shared base + side factor compartilhado

\[
W_e=W_0+A_eB
\]

Compartilha o fator maior e mantém transformação específica por especialista.

## 10.4 Basis bank + side factors

\[
W_e=W_0+A_e\left(\sum_{k=1}^{K}c_{e,k}B_k\right)
\]

Permite mais de um subespaço sem exigir combinação escalar de matrizes inteiras.

## 10.5 Parametrização bilateral alignment-tolerant

\[
W_e=W_0+L_eBR_e
\]

ou, separadamente para `gate/up/down`, side factors de entrada e saída. Essa família tolera rotações e desalinhamento de coordenadas.

---

# 11. Inicialização e loss recomendadas

## 11.1 Não usar SVD de peso puro como única inicialização

O objetivo deve ser ponderado pelas ativações:

\[
\min_{W_0,A_e,B}
\sum_{e,x}\pi_e(x)
\left\|
(W_e-W_0-A_eB)x
\right\|_2^2
\]

Com covariância de entrada `Σx`, aproximar:

\[
(W_e-W_0)\Sigma_x^{1/2}
\]

Isso alinha a decomposição às regiões funcionalmente visitadas.

## 11.2 Loss em estágios

### Estágio local

```text
MSE normalizada da saída MoE
+ cosine loss
+ per-expert weighted loss
```

### Estágio closed-loop

\[
\mathcal L=
\lambda_{local}\|y_S-y_T\|^2
+\lambda_{KL}D_{KL}(p_T\Vert p_S)
+\lambda_{CE}CE(p_S,y)
\]

### Estágio de orçamento

Adicionar regularização ou Lagrangiano para parâmetros/compute e reduzir rank/largura progressivamente.

---

# 12. Próximo programa experimental recomendado

## Fase A — concluir/reconciliar o código existente

1. extrair o bundle remoto `native-alignment-v1`;
2. verificar seu SHA-256;
3. inspecionar arquivos antes de reimplementar;
4. recuperar módulos faltantes dos runners antigos;
5. atualizar manifesto global;
6. integrar commits do teacher-width no PR.

## Fase B — teachers com plateau explícito

Treinar teachers em ao menos três escalas:

```text
small:  d_model 96
medium: d_model 192 ou 256
large-controlled: d_model 384 ou 512
```

Profundidade crescente, múltiplas seeds e corpus multi-domínio.

Critério de plateau pré-definido.

## Fase C — comparação de classes

Candidatos:

```text
teacher
narrow teacher-informed 65%
free-basis narrow 50%
free-basis narrow 65%
shared base + independent LoRA
shared base + shared side factor
basis bank + side factors
scalar Modal K2
full continuation control
```

Todos pareados por seed, batches e budget.

## Fase D — gates sugeridos

Para uma candidata com menos de 65% dos parâmetros/compute:

```text
hypothesis Δ loss UCB95 <= +0.020 ou +0.030 nat
OOD Δ loss UCB95 <= +0.040 ou +0.050 nat
candidate - narrow65 UCB95 <= 0
nenhuma seed > +0.050 ou +0.060 nat
full control UCB95 <= +0.010 nat
exact parameter ratio < 0.65
exact compute ratio < 0.65
independent audit PASS
```

Os valores finais devem ser justificados por power analysis e congelados antes do run.

## Fase E — trajetória de especialização

Salvar checkpoints do teacher ao longo do treino e medir:

- effective rank entre especialistas;
- minimum K para orçamento de loss;
- narrow width mínimo;
- residual rank necessário;
- entropia e especialização do router;
- compressibilidade funcional.

Hipótese:

```text
início: experts altamente compartilháveis
meio: especialização cresce
fim: scalar Modal pós-hoc perde eficiência
```

Se confirmada, priorizar arquitetura nativa/destilação precoce.

## Fase F — runtime

Somente após qualidade:

- Grouped GEMM otimizada;
- batch 1, 8, 32;
- prefill e decode;
- routing assimétrico;
- sorting, compaction, scatter;
- memória residente e tráfego;
- CPU e GPU.

Separar:

```text
ANALYTIC_COMPUTE_PASS
MEASURED_LAYER_RUNTIME_PASS
MEASURED_END_TO_END_PASS
```

## Fase G — modelo real

Somente depois de superar narrow65 em escalas controladas:

```text
OLMoE layer única
-> três layers espaçadas
-> múltiplas layers
-> checkpoint externo Qwen
```

---

# 13. Stop rules

## Abandonar scalar Modal pós-hoc se

- K necessário continuar >= top-k;
- side factors simples não melhorarem com largura;
- vantagem existir apenas em teachers curtos;
- narrow65 permanecer dominante.

## Abandonar seleção de coordenadas existentes se

- free-basis superar subsets consistentemente;
- 35% subset continuar falhando em novos teachers;
- estabilidade de seleção permanecer baixa.

## Bloquear OLMoE/Qwen se

- nenhuma candidata entrar na fronteira de Pareto;
- teachers não estiverem em plateau;
- auditor independente falhar;
- resultados dependerem de uma seed;
- runtime analítico não se converter em runtime medido.

---

# 14. Git: estado e procedimento correto

## 14.1 Histórico local do bundle

```text
03297e7 chore: reconstruct audited pre-Qwen base after runtime reset
2988a42 experiment: preregister fresh teacher-informed width replication
e7092ac fix: cover frozen vocabulary before fresh width replication
c8f4841 results: record audited teacher-informed width replication
```

O commit raiz `03297e7` é uma reconstrução independente e não possui o histórico original do `main`. Não tentar fazer fast-forward dele para o repositório remoto.

## 14.2 Como aplicar apenas os commits úteis

Clonar o bundle:

```bash
git clone pre-qwen-teacher-width-fresh-v1.git.bundle /tmp/teacher-width
git -C /tmp/teacher-width switch -c agent/teacher-informed-width-replication-v1 \
  origin/agent/teacher-informed-width-replication-v1
```

Gerar patches apenas após o root reconstruído:

```bash
git -C /tmp/teacher-width format-patch --stdout \
  03297e770ca5ab300df95df019302185c65bad1d..c8f4841e47b7c7bd9df3ac30e06c395f57c3b1d9 \
  > /tmp/teacher-width-series.patch
```

No checkout real do GitHub:

```bash
git switch agent/pre-qwen-certification-v2
git switch -c agent/integrate-teacher-width-v1
git am -3 /tmp/teacher-width-series.patch
```

Resolver conflitos manualmente em:

```text
docs/methodology/IMPORTANT_CLAIM_LEDGER.json
results/PRE_QWEN_GATE_STATUS.json
results/VERDICT.md
```

Não sobrescrever resultados mais novos da branch remota.

## 14.3 Alternativa de integração por arquivos

Os commits pós-root adicionam/modificam somente o conjunto listado por:

```bash
git diff --name-only 03297e7..c8f4841
```

Arquivos centrais:

```text
configs/pre_qwen_teacher_width_fresh_v1.yaml
docs/prereg/PRE_QWEN_TEACHER_WIDTH_FRESH_V1.md
docs/results/2026-07-28-teacher-informed-width-fresh-replication.md
pre_qwen_certification/dimension_pruning.py
pre_qwen_certification/teacher_width_data.py
scripts/run_teacher_width_fresh_seed.py
scripts/aggregate_teacher_width_fresh.py
scripts/audit_teacher_width_fresh.py
tests/test_teacher_width_fresh.py
results/pre-qwen-teacher-width-fresh/v1/**
```

## 14.4 Commits recomendados

Manter separação:

```text
experiment: preregister ...
fix: ... before held-out evaluation
results: record audited ...
audit: strengthen independent verification
```

Não misturar pré-registro e resultado no mesmo commit.

## 14.5 Política para checkpoints

Os cinco checkpoints atuais têm aproximadamente 1,25 MB cada e cabem no Git, mas repetir esse padrão aumentará rapidamente o repositório.

Política recomendada:

- comitar configs, reports, metrics, per-seed summaries e hashes;
- armazenar checkpoints grandes em Actions artifacts ou release;
- comitar apenas checkpoints pequenos necessários à reprodução/testes;
- registrar URL/ID do artifact e SHA-256 no repositório.

---

# 15. Runbook do experimento de 65%

## 15.1 Preparação

```bash
python -m pip install -e .
python -m pytest -q
```

## 15.2 Rodar seeds

```bash
OUT=results/pre-qwen-teacher-width-fresh/vNEXT
for SEED in 91121 92129 93133 94151 95153; do
  PYTHONPATH=. python scripts/run_teacher_width_fresh_seed.py \
    --seed "$SEED" \
    --config configs/pre_qwen_teacher_width_fresh_vNEXT.yaml \
    --output-dir "$OUT"
done
```

Não reutilizar a versão `v1.1` como nova confirmação. Criar nova config, novas seeds e novos documentos.

## 15.3 Agregar

```bash
PYTHONPATH=. python scripts/aggregate_teacher_width_fresh.py \
  --config configs/pre_qwen_teacher_width_fresh_vNEXT.yaml \
  --output-dir "$OUT"
```

## 15.4 Auditar

```bash
PYTHONPATH=. python scripts/audit_teacher_width_fresh.py \
  --config configs/pre_qwen_teacher_width_fresh_vNEXT.yaml \
  --output-dir "$OUT"
```

## 15.5 Hashes

```bash
python scripts/rebuild_result_hashes.py "$OUT"
sha256sum -c "$OUT/sha256sums.txt"
```

---

# 16. Mapa de arquivos

## Núcleo

```text
pre_qwen_certification/modal.py
pre_qwen_certification/tiny_lm.py
pre_qwen_certification/controlled_transplant.py
pre_qwen_certification/metrics.py
pre_qwen_certification/harness.py
```

## Seleção e largura

```text
pre_qwen_certification/dimension_pruning.py
pre_qwen_certification/conditional_pruning.py
pre_qwen_certification/teacher_width_data.py
```

## Certificação

```text
pre_qwen_certification/certify.py
scripts/run_pre_qwen_certification.py
scripts/verify_certification_artifact.py
scripts/independent_audit_controlled_result.py
scripts/replay_controlled_result.py
```

## Teacher-width

```text
configs/pre_qwen_teacher_width_fresh_v1.yaml
docs/prereg/PRE_QWEN_TEACHER_WIDTH_FRESH_V1.md
scripts/run_teacher_width_fresh_seed.py
scripts/aggregate_teacher_width_fresh.py
scripts/audit_teacher_width_fresh.py
tests/test_teacher_width_fresh.py
```

## Evidência

```text
results/pre-qwen-certification/v1/
results/controlled-small-moe-transplant/v1.3/
results/pre-qwen-fixed-width-attribution/v1/
results/pre-qwen-conditional-ablation/v1/
results/pre-qwen-teacher-width-fresh/v1/
```

---

# 17. Defeitos e pendências conhecidas

1. `DELIVERY_MANIFEST.sha256` está desatualizado em dois arquivos.
2. O runner condicional importa quatro módulos ausentes na reconstrução.
3. O bundle local possui root independente; não é branch diretamente baseada no GitHub main.
4. PR #1 não contém o resultado teacher-width.
5. PR #1 contém import `native-alignment-v1` incompleto.
6. `pytest` exige instalação editável ou `PYTHONPATH=.`.
7. Teachers de 2200 passos não demonstraram plateau.
8. OOD CE não representa paridade comportamental.
9. Nenhum runtime real foi medido.
10. Os holdouts consumidos não podem ser reutilizados para escolher sucessores.

---

# 18. Checklist da próxima sessão

## Primeiros 30 minutos

- [ ] Ler este handoff.
- [ ] Verificar hashes do ZIP e Git bundle.
- [ ] Clonar o bundle e checkout `c8f4841`.
- [ ] Rodar `PYTHONPATH=. pytest -q`.
- [ ] Verificar `sha256sums.txt` do teacher-width.
- [ ] Reexecutar auditor em cópia temporária.
- [ ] Consultar PR #1 e confirmar head atual.
- [ ] Extrair `native-alignment-v1.b64` e inspecionar conteúdo.

## Antes de novo experimento

- [ ] Criar protocolo versionado novo.
- [ ] Criar seeds e documentos novos.
- [ ] Definir plateau de teacher.
- [ ] Definir baseline narrow65.
- [ ] Definir candidatos e fórmulas de parâmetros/compute.
- [ ] Fazer power analysis.
- [ ] Congelar gates.
- [ ] Implementar auditor independente antes de olhar resultado final.

## Antes de claim importante

- [ ] Edição estrita do texto da claim.
- [ ] Evidência primária e hashes.
- [ ] Recomputação independente.
- [ ] Auditoria adversarial.
- [ ] Controles positivos e negativos.
- [ ] Resultados por seed/domínio.
- [ ] Limitações explícitas.
- [ ] Atualização do claim ledger.

---

# 19. Prompt pronto para continuar em outra sessão

```text
Continue a pesquisa no repositório rornellas/llm-thinking-rep.

Leia integralmente o arquivo:
HANDOFF_MODAL_MOE_RESEARCH_2026-07-28_FULL.md

Estado científico congelado:
- NO_GO_FOR_OLMOE_OR_QWEN.
- Scalar Modal pós-hoc é funcional, mas não vence narrow compute-matched.
- Seleção de coordenadas existentes em 35% falhou.
- Um student convencional teacher-informed em 65% passou uma replicação fresca
  de cinco seeds e agora é o baseline Pareto mínimo.

Fontes locais:
- pre-qwen-teacher-width-fresh-v1.git.bundle
- head c8f4841e47b7c7bd9df3ac30e06c395f57c3b1d9
- SHA do bundle:
  8f0ca3009e09f2130aad564fc52bb5597e5de12c356fa53d27b0b875545f8036

Estado remoto:
- PR #1 aberto e draft.
- branch agent/pre-qwen-certification-v2.
- head verificado no corte: 7e868028671c7ecf513f7fbfc1b6e2320506ec76.
- o resultado teacher-width ainda não está integrado.
- existe um bundle native-alignment-v1 ainda não extraído; inspecione-o antes de
  reimplementar a próxima arquitetura.

Primeiro objetivo:
1. reconciliar o bundle local com a branch remota sem cherry-pickar o root
   reconstruído 03297e7;
2. corrigir manifesto global e módulos ausentes;
3. criar novo pré-registro para comparação alignment-tolerant;
4. testar free-basis narrow, shared-base+LoRA, shared-side-factor e basis-bank
   contra narrow teacher-informed 65%;
5. usar teachers com plateau explícito, novos documentos, várias seeds,
   bootstrap crossed seed/document e auditor independente;
6. não usar OLMoE/Qwen até uma candidata superar narrow65 com menos de 65% de
   parâmetros/compute e passar runtime medido.

Modo de operação obrigatório:
- double-check factual e auditoria adversarial em toda claim importante;
- nenhuma mudança de gate após observar resultado;
- nenhum reuso de holdout consumido;
- preservar e publicar resultados negativos;
- separar parâmetros, compute analítico, tráfego, memória e runtime;
- trabalhar proativamente até emitir verdict reproduzível.
```

---

# 20. Conclusão

A pesquisa não está no ponto de provar compressão de Qwen. Ela está em um ponto mais valioso: possui um harness falsificável, resultados negativos informativos, um diagnóstico matemático do gargalo escalar e um baseline convencional de 65% reproduzido em cinco seeds.

O próximo resultado louvável não será “mais um Modal que funciona”. Será uma arquitetura que:

1. iguale ou supere o narrow teacher-informed de 65%;
2. use menos de 65% dos parâmetros e do compute projetado;
3. preserve a vantagem em teachers com plateau e em escala crescente;
4. sobreviva a avaliação nova, auditoria independente e runtime real.

Até isso acontecer, a decisão correta permanece:

```text
NO_GO_FOR_OLMOE_OR_QWEN
```
