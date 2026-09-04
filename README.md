# LLM Thinking Representation Experiments

Repositório dedicado a experimentos de representação e execução eficiente de modelos de linguagem.

## Estado atual — 4 de setembro de 2026

**Objetivo:** ganho reproduzível na relação qualidade, parâmetros, memória, operações e latência, primeiro em modelos controlados e depois em modelos reais. O objetivo completo ainda não foi demonstrado.

**Avanço confirmado nesta rodada:** os oito checkpoints compactos primários foram reduzidos de **95.552 para 47.168 parâmetros totais (−50,6363%)**, truncando cada resíduo de especialista de rank 8 para rank 1, sem retreinamento. O teste pré-registrado **FCC-1 passou em 256 artigos novos para esses modelos**, com quatro janelas por artigo e incerteza sobre seeds e artigos. A perda média aumentou **0,0000798 nat** no grupo de 800 atualizações e **0,0005794 nat** no de 2.200, abaixo das margens fixadas. Essa é uma confirmação de fidelidade da compressão, não de uma arquitetura superior.

**Resultado negativo preservado:** a tentativa de vetorização **ECK-1 falhou na equivalência numérica** em dois jobs. Pequenas diferenças de arredondamento alteraram uma escolha top-k quase empatada no caso diagnosticado. Não há ganho de velocidade certificado nesta rodada. O kernel `vectorized_compact.py` é experimental; a qualidade confirmada usa a implementação original em laço.

**Decisão:** `NO_GO_FOR_OLMOE_OR_QWEN` permanece. Antes de escalar, comparar com modelos densos simples sob os mesmos orçamentos. Os controles densos pareados ainda não foram treinados. Todos os artigos FCC-1 agora são revelados e não poderão ser reutilizados para otimização seguida de alegação confirmatória.

### Evidência atual

- [Revisão científica integrada, limites e próximo teste discriminante](docs/audits/2026-09-04-compression-functional-review.md)
- [FCC-1 — confirmação em artigos novos](docs/results/2026-09-04-fresh-compression-check-1.md)
- [FCC-1 — protocolo](docs/prereg/FRESH_COMPRESSION_CHECK_1.md), [dados](data/fresh-compression-check-1/) e [resultados/auditorias](results/fresh-compression-check-1/)
- [FA-1 — ablação funcional e compressão](docs/results/2026-09-04-functional-ablation-1.md), [112 exports e dados brutos](results/functional-ablation-1/)
- [ECK-1 — falha e diagnóstico de roteamento](docs/results/2026-09-04-exact-compact-kernel-1.md), [artefatos completos da tentativa](results/exact-compact-kernel-1/)
- [Regras de pesquisa e commits](AGENTS.md)

**Escopo:** modelos pequenos, vocabulário 512, contexto 64, inglês/Wikipedia. Não foram demonstradas competências de raciocínio, uso de ferramentas, modelos de bilhões de parâmetros, pico de memória, aceleração em GPU ou uma melhoria da fronteira de capacidade. SVD, deltas de baixo rank e vetorização não são, por si, contribuições novas.

**Correção de unidade amostral:** os identificadores de “artigo” do segmentador antigo incluem subseções. FA-1 usa estatística condicional às janelas sobre seeds; seus números não mudam. FCC-1 separa artigos top-level verdadeiros. [Registro da correção](results/functional-ablation-1/data-unit-correction.json).

## Rodada anterior — MUI-1, setembro de 2026

MUI-1 executou seis braços e quatro seeds, totalizando 24 treinamentos. Restaurar o uso dos modos não produziu a melhoria pré-especificada sobre o compacto anterior. Veredito: `NO_PROMISING_CANDIDATE_UNDER_FROZEN_SCREEN`. O diagnóstico foi exploratório, com orçamento curto e calibração conhecida.

- [Revisão científica MUI-1](docs/audits/2026-09-04-mui1-scientific-review.md)
- [Resultados MUI-1 e latências medidas](docs/results/2026-09-04-mode-utilization-intervention-1.md)
- [Pré-registro MUI-1](docs/prereg/MODE_UTILIZATION_INTERVENTION_1.md)
- [Dados brutos e checkpoints MUI-1](results/mode-utilization-intervention-1/)
- [Gate 2A anterior — FAIL](docs/results/2026-08-05-native-compact-gate-2a-analysis.md)

## Registro histórico — primeiro teste real, julho de 2026

### Modal MoE / OLMoE layer 7

Hipótese em investigação: representar as matrizes `gate`, `up` e `down` dos especialistas de uma camada MoE por estruturas compartilhadas, preservando função e possibilitando execução direta sem reconstrução dos pesos.

O primeiro teste real foi concluído em **26 de julho de 2026**:

- modelo: `allenai/OLMoE-1B-7B-0924`;
- camada: índice `7`;
- especialistas: todos os `64`;
- projeções: `gate`, `up` e `down`;
- amostragem: `131.072` coordenadas por matriz, com três seeds independentes;
- execução: GitHub Actions run `30221181344`;
- resultado técnico: workflow concluído com sucesso;
- resultado científico: **FAIL para PCA global direta nos pesos não alinhados**.

As três projeções exigiram rank `60` para explicar 95% da variação residual e rank `63` para 99%. Em `K=8`, apenas cerca de 14–15% da variação entre especialistas foi explicada.

Isso rejeita uma base global de poucos modos nas coordenadas atuais dos neurônios, mas não rejeita decomposição após normalização de escala, alinhamento por permutação, agrupamento de especialistas ou ajuste orientado por ativações.

### Documentação histórica

- [Laudo do primeiro experimento](docs/results/2026-07-26-olmoe-layer-7-raw-pca-scout.md)
- [Veredito agregado](results/latest/aggregate/VERDICT.md)
- [Resumo por projeção](results/latest/aggregate/projection_summary.csv)
- [Curvas por rank](results/latest/aggregate/rank_summary.csv)
- [Estado da execução histórica](runs/latest.json)

### Etapa proposta naquela data — não é o próximo passo atual

**Teste 0.5 — alignment and localization diagnostic**

1. medir normas, erros e localização dos modos por especialista;
2. canonicalizar a liberdade de escala entre `up` e `down`;
3. alinhar conjuntamente os neurônios de `gate`, `up` e `down` por permutação;
4. repetir a análise espectral em um subconjunto representativo;
5. expandir para os 64 especialistas somente se o alinhamento reduzir materialmente o rank necessário.
