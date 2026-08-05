# Revisão factual adversarial multilentes — Reality Gate 1A

**Data:** 5 de agosto de 2026  
**Protocolo:** `reality-gate-1a-static-heterogeneous-rank-v1`  
**Veredito automático preservado:** `REALITY_GATE_1A_FAIL`  
**Disposição científica adversarial:** `STATIC_HETEROGENEITY_NOT_SUPPORTED__DYNAMIC_RANK_BLOCKED__POST_HOC_DEPRIORITIZED`  
**Decisão global:** `NO_GO_FOR_OLMOE_OR_QWEN`

## 1. Veredito executivo

O Reality Gate 1A cumpriu sua função de falsificação. A hipótese primária não foi sustentada:

> Sob os budgets, duas escalas, quatro seeds, dados WikiText-2 pinados e regra exata de orçamento testados, a alocação espectral heterogênea estática não superou rank uniforme e não aproximou a fidelidade funcional do baseline convencional `narrow65`.

O resultado é mais forte que um simples `FAIL` numérico:

- nenhum dos oito teachers científicos atingiu o critério pré-registrado de plateau;
- na escala `medium`, o MILP espectral retornou exatamente o rank uniforme em todas as quatro seeds e em todos os checkpoints de trajetória;
- na escala `small`, retornou exatamente o uniforme em três seeds finais e desviou minimamente em apenas uma;
- o controle baseado somente em frequência de routing também retornou rank uniforme em todas as células finais;
- onde houve um desvio espectral pequeno, não houve vantagem estabelecida sobre o uniforme;
- `narrow65` permaneceu significativamente melhor em loss na escala média e substancialmente melhor em KL, top-1 e NRMSE local nas duas escalas;
- o rank residual necessário para preservar 95% da energia ficou muito acima dos ranks executados: aproximadamente 25,3 de 32 na escala small e 37,8 de 48 na medium.

A ausência de plateau impede a alegação ampla de que heterogeneidade estática falha em teachers maduros. Entretanto, ela não resgata o mecanismo atual: mesmo enquanto os teachers ainda especializavam, a função-objetivo do alocador não encontrou heterogeneidade útil ao longo da trajetória. Portanto, não existe base empírica para adicionar um controlador dinâmico por token sobre esse mecanismo.

## 2. Integridade e reprodutibilidade

A evidência é utilizável e foi preservada integralmente:

- dataset oficial `Salesforce/wikitext`, subset `wikitext-2-raw-v1`;
- revisão imutável `b08601e04326c79dfdd32d625aee71d232d685c3` congelada antes do primeiro carregamento;
- BPE byte-level treinado somente no split de treino;
- artigos preservados por offsets e usados como unidade estatística;
- `test` e OOD carregados somente depois de congelar candidates, ranks e checkpoint hash;
- quatro seeds em duas escalas;
- piloto separado sem acesso ao holdout final;
- identidade exata como controle positivo;
- parâmetros e compute analítico conferidos no treino e nos heldouts;
- bootstrap cruzado por seed e documento;
- comparações pareadas;
- leave-one-seed-out;
- auditor independente sem importar o agregador ou o bootstrap do projeto;
- auditoria retornou `PASS`, zero divergências;
- checkpoints, JSONs por janela, logs, tokenizer, arrays, ambiente e SHA-256 versionados;
- workflow `31011802255` concluído com sucesso;
- resultado consolidado no commit `cd57ba42d69036f188b1bc067a895f98e72ef43e`.

O source commit científico foi `456c3beb87eace9a196a46bd24ab33090b0bc2eb`. A configuração e o manifesto de dados possuem hashes registrados no ambiente do run.

## 3. Plateau: falha clara, não marginal

### Escala small

| Seed | Passos finais | Slope por passo | Melhora/range da janela | Drift L1 de routing | Plateau |
|---:|---:|---:|---:|---:|---|
| 111731 | 3200 | `-1,5119e-4` | `0,10712` | `0,33496` | não |
| 121747 | 3200 | `-1,4713e-4` | `0,10080` | `0,33511` | não |
| 131759 | 3200 | `-1,5998e-4` | `0,11113` | `0,39856` | não |
| 141767 | 3200 | `-1,6300e-4` | `0,11448` | `0,44551` | não |

### Escala medium

| Seed | Passos finais | Slope por passo | Melhora/range da janela | Drift L1 de routing | Plateau |
|---:|---:|---:|---:|---:|---|
| 111731 | 4500 | `-1,0090e-4` | `0,09375` | `0,39746` | não |
| 121747 | 4500 | `-1,0095e-4` | `0,09156` | `0,27356` | não |
| 131759 | 4500 | `-1,0754e-4` | `0,09653` | `0,27999` | não |
| 141767 | 4500 | `-1,1461e-4` | `0,10205` | `0,31360` | não |

Os limites congelados de slope eram `1,2e-5` para small e `8e-6` para medium. Os slopes observados eram aproximadamente uma ordem de grandeza mais negativos. A amplitude de loss e o routing drift também estavam longe dos limites. Isso não é uma reprovação por ruído de fronteira; os teachers ainda estavam materialmente aprendendo e reorganizando routing.

O piloto de engenharia atingiu plateau, mas sua seed, orçamento e papel eram explicitamente não científicos. Ele prova que o detector pode disparar; não substitui plateau nas oito células confirmatórias.

## 4. Colapso da alocação heterogênea

### Escala medium

Em todas as quatro seeds finais:

```text
heterogeneous-spectral = [12, 12, ..., 12]
heterogeneous-routing  = [12, 12, ..., 12]
uniform-rank           = [12, 12, ..., 12]
```

O mesmo ocorreu nos checkpoints de 25%, 50%, 75% e final. Consequentemente, spectral, routing-only e uniforme produziram exatamente os mesmos outputs e métricas.

### Escala small

Em três seeds finais:

```text
heterogeneous-spectral = [8, 8, ..., 8]
```

Na seed `141767`, o alocador encontrou somente uma redistribuição pequena:

```text
[9, 9, 8, 8, 8, 7, 8, 7, 8, 8, 8, 8]
```

Ela manteve o mesmo rank total e praticamente o mesmo compute esperado. Não produziu vantagem estabelecida; a diferença de loss contra o uniforme nessa seed foi desfavorável. O controle routing-only permaneceu uniforme.

### Interpretação

O MILP não foi impedido de criar heterogeneidade. Ele possuía:

- rank máximo superior ao uniforme;
- prefix constraints corretas;
- igualdade de orçamento total;
- limite de compute ponderado pelo routing;
- utilidades marginais por expert e modo.

A solução uniforme indica que, sob a utilidade espectral normalizada e os budgets testados, as diferenças marginais entre experts não eram fortes o suficiente para justificar transferência de rank. A hipótese estática falhou antes mesmo de qualquer controlador online.

## 5. Resultados agregados

### Small

| Candidato | Δ loss | KL | Top-1 | Local NRMSE | Params | Compute |
|---|---:|---:|---:|---:|---:|---:|
| heterogeneous-spectral | `+0,00604` | `0,01453` | `84,51%` | `0,19035` | `45,83%` | `62,50%` |
| uniform-rank | `+0,00595` | `0,01451` | `84,49%` | `0,19026` | `45,83%` | `62,50%` |
| routing-only | `+0,00595` | `0,01451` | `84,49%` | `0,19026` | `45,83%` | `62,50%` |
| narrow65 | `-0,00142` | `0,00468` | `91,62%` | `0,08759` | `65,62%` | `65,62%` |

Comparações load-bearing:

```text
spectral - uniform loss
mean  +0,000087
95%   [-0,000145, +0,000495]

spectral - narrow65 loss
mean  +0,007460
95%   [+0,005427, +0,009378]

spectral - narrow65 KL
mean  +0,009849
95%   [+0,008531, +0,011803]

spectral - narrow65 top-1
mean  -0,071135
95%   [-0,077942, -0,064773]

spectral - narrow65 local NRMSE
mean  +0,102763
95%   [+0,099066, +0,106251]
```

A margem pré-registrada de loss contra narrow65 era `+0,010`; por isso o gate de loss small passou formalmente. Isso não representa equivalência funcional: KL, top-1 e NRMSE local mostram vantagem grande e consistente do narrow65.

### Medium

| Candidato | Δ loss | KL | Top-1 | Local NRMSE | Params | Compute |
|---|---:|---:|---:|---:|---:|---:|
| heterogeneous-spectral | `+0,00800` | `0,02074` | `83,05%` | `0,21355` | `43,75%` | `62,50%` |
| uniform-rank | `+0,00800` | `0,02074` | `83,05%` | `0,21355` | `43,75%` | `62,50%` |
| routing-only | `+0,00800` | `0,02074` | `83,05%` | `0,21355` | `43,75%` | `62,50%` |
| narrow65 | `-0,00313` | `0,00790` | `89,85%` | `0,10512` | `64,58%` | `64,58%` |

Comparações load-bearing:

```text
spectral - uniform loss
mean  0
95%   [0, 0]

spectral - narrow65 loss
mean  +0,011128
95%   [+0,008783, +0,013631]

spectral - narrow65 KL
mean  +0,012840
95%   [+0,011905, +0,013891]

spectral - narrow65 top-1
mean  -0,068028
95%   [-0,073236, -0,062648]

spectral - narrow65 local NRMSE
mean  +0,108432
95%   [+0,105000, +0,112366]
```

Na escala medium, a candidata falhou inclusive a margem de loss contra narrow65.

## 6. OOD

Os gates absolutos de loss e KL OOD passaram. Isso não resgata a hipótese:

- o OOD é pequeno e determinístico;
- spectral, routing-only e uniform eram geralmente a mesma arquitetura efetiva;
- a fidelidade comportamental OOD continuou substancialmente pior que narrow65;
- resultados negativos de CE em relação ao teacher podem refletir regularização ou afastamento útil no corpus, não preservação da função do teacher.

Não é correto inferir robustez geral ou superioridade OOD a partir desse endpoint.

## 7. Trajetória de compressibilidade

O rank residual necessário para preservar 95% da energia permaneceu alto e praticamente estável do checkpoint de 25% ao final:

| Escala | Rank máximo matricial | Rank 95% médio final | Faixa observada |
|---|---:|---:|---:|
| small | 32 | `25,32` | `24–26` |
| medium | 48 | `37,80` | `37–39` |

Os ranks executados eram 8 e 12. Em ambas as escalas, eles correspondem a aproximadamente um terço do rank necessário para 95% da energia residual.

Isso explica por que a família shared-base low-rank preserva menos comportamento que narrow65: ela está impondo uma compressão matricial muito agressiva sobre resíduos que não mostram decaimento espectral rápido. Redistribuir o mesmo rank total não cria capacidade ausente; apenas a move entre experts com espectros semelhantes.

A estabilidade do rank efetivo desde 25% do treinamento enfraquece a expectativa de que a heterogeneidade útil simplesmente surgiria mais tarde no mesmo regime. Não a elimina universalmente, pois os teachers não plateauaram, mas não fornece qualquer sinal favorável.

## 8. Lentes adversariais

### 8.1 “O plateau foi estrito demais”

Não aprovado como explicação principal. As reprovações ficaram longe dos limites em slope, range e routing drift. Relaxar os gates depois do resultado seria metodologicamente inválido.

### 8.2 “Bastaria treinar por mais passos”

Mais passos poderiam responder como a trajetória termina, mas não justificam repetir automaticamente a mesma hipótese:

- o alocador colapsou ao uniforme em quase todas as células e checkpoints;
- os espectros residuais eram altos e homogêneos;
- narrow65 dominou a fidelidade comportamental;
- não houve tendência positiva com escala.

Uma extensão de maturidade pode ser executada como diagnóstico separado e pré-registrado, não como resgate do Reality Gate 1A.

### 8.3 “A utilidade espectral é a errada”

É possível. O experimento refuta a utilidade espectral normalizada testada, não todas as funções de utilidade imagináveis. Entretanto:

- routing-only também não encontrou heterogeneidade;
- o custo de uma utilidade closed-loop por modo seria muito maior;
- a estrutura residual permanece altamente rank-rich;
- perseguir sucessivos allocators pós-hoc sem sinal estático configuraria overfitting de pesquisa.

Uma função de utilidade diferente só deve retornar como parte de uma classe arquitetural ou regime de treinamento materialmente novo.

### 8.4 “O narrow65 usa mais parâmetros”

Correto: narrow65 usa aproximadamente 64,6–65,6%, contra 43,8–45,8% da candidata. Isso preserva um sinal de eficiência de parâmetros para a família low-rank. Porém o Reality Gate exigia mover a fronteira de qualidade sob compute comparável, e o narrow65 usa apenas cerca de 2–3 pontos percentuais adicionais de compute analítico. A lacuna comportamental é grande demais para ser atribuída apenas a uma pequena diferença de compute.

### 8.5 “O resultado medium idêntico prova bug”

Não há evidência de bug:

- o MILP e a álgebra passaram testes;
- rank vectors foram persistidos;
- spectral, routing-only e uniform convergiram para o mesmo vetor por solução ótima;
- o auditor reconstruiu budgets, ranks, outputs, estatística e decisão;
- a seed small `141767` demonstrou que o mecanismo consegue produzir vetor não uniforme quando a utilidade o favorece.

A identidade das métricas medium é a consequência esperada de vetores e estados iguais, não uma falha silenciosa.

## 9. Originalidade e relação com o objetivo

Rank heterogêneo estático não seria original isoladamente, e a literatura já explora alocação heterogênea. A oportunidade do projeto dependia de mostrar que:

- bases compartilhadas;
- fatores locais;
- prefixes aninhados;
- orçamento ativo;
- execução fundida;

produzem uma nova fronteira de Pareto.

O Reality Gate 1A mostra que o primeiro elo — utilidade estática da heterogeneidade pós-hoc sob essa fatoração — não foi demonstrado. Adicionar um controlador dinâmico agora aumentaria complexidade sem causa raiz comprovada.

## 10. Disposição científica

| Questão | Decisão |
|---|---|
| Preservar `REALITY_GATE_1A_FAIL` | sim |
| A auditoria é válida | sim |
| Teachers científicos atingiram plateau | não, 0/8 |
| Spectral heterogeneity venceu rank uniforme | não |
| Spectral venceu routing-only | não |
| O allocator criou heterogeneidade material | não |
| Narrow65 preservou melhor comportamento | sim |
| Dynamic rank por token está autorizado | não |
| Repetir o mesmo protocolo com gates relaxados | não |
| Declarar refutação universal de ranks heterogêneos | não |
| Despriorizar conversão pós-hoc como linha principal | sim |
| Mudar para treinamento nativo da arquitetura compacta | sim |
| OLMoE/Qwen autorizado | não |

Disposição final:

```text
STATIC_HETEROGENEITY_NOT_SUPPORTED
DYNAMIC_RANK_BLOCKED
POST_HOC_ALIGNMENT_TOLERANT_CONVERSION_DEPRIORITIZED
NATIVE_COMPACT_TRAINING_BECOMES_PRIMARY
```

## 11. Próxima hipótese aprovada

A próxima experiência deve testar a causa raiz sugerida pelos resultados:

> A estrutura compartilhada precisa ser imposta durante o treinamento, antes que os experts consolidem resíduos high-rank incompatíveis com o budget low-rank.

O próximo protocolo deve comparar, desde inicialização e com o mesmo compute de treino:

1. MoE convencional full;
2. narrow65;
3. shared-base + fatores locais treinados nativamente;
4. shared-base + prefixes aninhados treinados em múltiplos budgets;
5. baseline MoSE-like de largura aninhada;
6. baseline RFID-like de rank heterogêneo estático;
7. combinação nativa shared-base + prefixes + heterogeneidade, sem controlador online na primeira fase.

A sequência de claims deve permanecer separada:

- primeiro, arquitetura nativa supera uniforme/narrow em qualidade-compute;
- depois, prefixes menores são funcionalmente úteis;
- somente então, controlador dinâmico por token;
- por último, kernel e runtime em GPU existente.

## 12. Regra de morte da próxima fase

A arquitetura nativa será despriorizada se, em duas escalas e teachers efetivamente plateaued:

- não superar o MoE convencional ou narrow65 no Pareto qualidade-compute;
- não igualar pelo menos um baseline publicado inspirado em MoSE/RFID;
- os ganhos desaparecerem com largura;
- o runtime em GPU comum não acompanhar a economia analítica.

Até lá:

```text
NO_GO_FOR_OLMOE_OR_QWEN
NO_RUNTIME_CLAIM
NO_DYNAMIC_RANK
```
