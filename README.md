# LLM Thinking Representation Experiments

Repositório dedicado a experimentos de representação e execução eficiente de modelos de linguagem.

## Estado atual — 6 de setembro de 2026

**Objetivo:** ganho reproduzível na relação qualidade, parâmetros, memória, operações e latência, primeiro em modelos controlados e depois em modelos reais. O objetivo completo ainda não foi demonstrado.

**Última rodada concluída: DCG-1 — comparação contra modelos densos pareados.** Foram 30 treinamentos novos, seis sementes pareadas, 4.400 atualizações e avaliação em 256 artigos novos para os modelos. Veredito: **`DCG1_NO_CAPACITY_ADVANTAGE_DEMONSTRATED`**. Nenhum dos quatro contrastes primários demonstrou a superioridade pré-especificada.

| Modelo | Parâmetros totais | Perda em artigos novos, menor é melhor |
|---|---:|---:|
| Compacto rank1 treinado diretamente | 47.168 | 4,269019 |
| Compacto rank8 comprimido para rank1 | 47.168 | 4,248656 |
| Denso de mesmo tamanho, largura104 | 47.168 | 4,227333 |
| Denso de mesmo custo matricial FFN/roteador, largura80 | 42.560 | 4,258728 |

O compacto comprimido teve média favorável contra o denso menor, mas não passou o critério de superioridade; contra o denso de mesmo tamanho, teve perda média maior. Os controles de tamanho e compute são distintos: o denso de mesmo tamanho usa mais MACs analíticos. Não se afirma dominância em todos os eixos ou impossibilidade de toda a família.

**Runtime:** na implementação CPU validada, o compacto comprimido levou medianamente **3,54×** o tempo do denso de mesmo tamanho com comprimento1 e **4,63×** com comprimento64, em razões pareadas por runner. São forwards completos, dois threads, batch1, sem cache KV ou perda auxiliar; não serving/GPU.

**Decisão:** suspender escala e controladores para esta receita. `NO_GO_FOR_OLMOE_OR_QWEN` permanece. A compressão anterior continua válida contra seus próprios checkpoints, mas não deve ser apresentada como uma representação superior aos controles convencionais. O único próximo screen elegível desta mesma parametrização é uma comparação limitada de otimização com orçamento de busca igual para compacto e densos, usando somente treino/desenvolvimento. Esse screen ainda não foi iniciado.

### Evidência da rodada atual

- [Revisão científica, auditoria e decisão](docs/audits/2026-09-06-dense-control-gate-1-review.md)
- [Resultados e contrastes pré-especificados](docs/results/2026-09-06-dense-control-gate-1.md)
- [Protocolo anterior ao treinamento](docs/prereg/DENSE_CONTROL_GATE_1.md)
- [Dados brutos, checkpoints, exports, timings e hashes](results/dense-control-gate-1/)
- [Auditoria executada em outro runtime](results/dense-control-gate-1/external-archive-audit.json)
- [Diagnósticos suplementares, explicitamente descritivos](results/dense-control-gate-1/supplementary-diagnostics.json)
- [Estado para continuidade](OBJECTIVE_STATE.md) e [regras de pesquisa](AGENTS.md)

A auditoria externa conferiu83 arquivos,36 exports e os quatro contrastes por uma terceira via aritmética. Todos os exports foram recarregados; uma implementação independente de FFN/Gram reexecutou um subconjunto pré-especificado. Isso não equivale a replicação por outro grupo científico. Todos os artigos DCG-1 agora são revelados e não poderão ser usados para otimização seguida de alegação confirmatória.

## Evidência anterior — FCC-1, FA-1 e ECK-1, 4 de setembro de 2026

**Compressão confirmada:** os oito checkpoints compactos primários foram reduzidos de **95.552 para 47.168 parâmetros totais (−50,6363%)**, truncando cada resíduo de especialista de rank8 para rank1, sem retreinamento. **FCC-1 passou em 256 artigos novos para esses modelos**, quatro janelas por artigo e incerteza sobre sementes/artigos. A perda média aumentou0,0000798 nat no grupo de800 atualizações e0,0005794 no de2.200, abaixo das margens fixadas. Essa confirmação de fidelidade não é anulada pelo resultado negativo de capacidade do DCG-1.

**Falha preservada:** ECK-1 falhou na equivalência numérica em dois jobs. Pequenas diferenças de arredondamento alteraram uma escolha top-k quase empatada no caso diagnosticado. O kernel `vectorized_compact.py` permanece experimental; a qualidade validada usa o caminho original em laço. Não há aceleração certificada dessa substituição.

- [Revisão integrada da rodada](docs/audits/2026-09-04-compression-functional-review.md)
- [FCC-1 — confirmação em artigos novos](docs/results/2026-09-04-fresh-compression-check-1.md)
- [FCC-1 — protocolo](docs/prereg/FRESH_COMPRESSION_CHECK_1.md), [dados](data/fresh-compression-check-1/) e [resultados/auditorias](results/fresh-compression-check-1/)
- [FA-1 — ablação funcional e compressão](docs/results/2026-09-04-functional-ablation-1.md), [112 exports e dados brutos](results/functional-ablation-1/)
- [ECK-1 — falha e diagnóstico](docs/results/2026-09-04-exact-compact-kernel-1.md), [artefatos completos](results/exact-compact-kernel-1/)

**Escopo:** modelos pequenos, vocabulário512, contexto64, inglês/Wikipedia. Não foram demonstradas competências de raciocínio, uso de ferramentas, transferência para bilhões de parâmetros, pico de memória, aceleração em GPU, convergência ou uma fronteira global superior. SVD, deltas de baixo rank e vetorização não são contribuições novas por si só.

**Correção de unidade amostral:** os identificadores de artigo do segmentador antigo incluem subseções. FA-1 usa estatística sobre sementes condicional às janelas; sua aritmética não muda. FCC-1 e DCG-1 separam artigos top-level verdadeiros. [Registro da correção](results/functional-ablation-1/data-unit-correction.json).

## Rodada anterior — MUI-1

Seis braços e quatro sementes, 24 treinamentos. Restaurar o uso dos modos não produziu a melhoria pré-especificada sobre o compacto anterior. Veredito: `NO_PROMISING_CANDIDATE_UNDER_FROZEN_SCREEN`. Diagnóstico exploratório, orçamento curto e calibração conhecida.

- [Revisão científica MUI-1](docs/audits/2026-09-04-mui1-scientific-review.md)
- [Resultados MUI-1 e latências](docs/results/2026-09-04-mode-utilization-intervention-1.md)
- [Pré-registro MUI-1](docs/prereg/MODE_UTILIZATION_INTERVENTION_1.md)
- [Dados e checkpoints MUI-1](results/mode-utilization-intervention-1/)
- [Gate 2A anterior — FAIL](docs/results/2026-08-05-native-compact-gate-2a-analysis.md)

## Registro histórico — primeiro teste real, julho de 2026

### Modal MoE / OLMoE layer7

Hipótese original: representar as matrizes gate, up e down dos especialistas de uma camada MoE por estruturas compartilhadas, preservando função e permitindo execução direta sem reconstrução dos pesos.

O primeiro teste real foi concluído em26 de julho de2026, no modelo `allenai/OLMoE-1B-7B-0924`, camada7, todos os64 especialistas, projeções gate/up/down,131.072 coordenadas por matriz e três sementes. Execução30221181344 concluída tecnicamente; resultado científico **FAIL para PCA global direta nos pesos não alinhados**.

As projeções exigiram rank60 para explicar95% da variação residual e rank63 para99%. Em K=8, apenas cerca de14–15% da variação entre especialistas foi explicada. Isso rejeita poucos modos globais nas coordenadas atuais, não todas as decomposições após alinhamento, agrupamento ou ajuste funcional.

- [Laudo do primeiro experimento](docs/results/2026-07-26-olmoe-layer-7-raw-pca-scout.md)
- [Veredito histórico](results/latest/aggregate/VERDICT.md)
- [Resumo por projeção](results/latest/aggregate/projection_summary.csv)
- [Curvas por rank](results/latest/aggregate/rank_summary.csv)
- [Execução histórica](runs/latest.json)

O diagnóstico de alinhamento proposto naquela data é registro histórico, não o próximo passo atual. Consulte `OBJECTIVE_STATE.md` antes de iniciar trabalho ou repetir workflows.
