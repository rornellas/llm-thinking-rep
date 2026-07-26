# LLM Thinking Representation Experiments

Repositório dedicado a experimentos de representação e execução eficiente de modelos de linguagem.

## Experimento atual

### Modal MoE / OLMoE layer 7

Hipótese em investigação: representar as matrizes `gate`, `up` e `down` dos especialistas de uma camada MoE por estruturas compartilhadas, preservando função e possibilitando execução direta sem reconstrução dos pesos.

## Estado

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

## Documentação e resultados

- [Laudo do primeiro experimento](docs/results/2026-07-26-olmoe-layer-7-raw-pca-scout.md)
- [Veredito agregado](results/latest/aggregate/VERDICT.md)
- [Resumo por projeção](results/latest/aggregate/projection_summary.csv)
- [Curvas por rank](results/latest/aggregate/rank_summary.csv)
- [Estado da execução](runs/latest.json)

## Próxima etapa

**Teste 0.5 — alignment and localization diagnostic**

1. medir normas, erros e localização dos modos por especialista;
2. canonicalizar a liberdade de escala entre `up` e `down`;
3. alinhar conjuntamente os neurônios de `gate`, `up` e `down` por permutação;
4. repetir a análise espectral em um subconjunto representativo;
5. expandir para os 64 especialistas somente se o alinhamento reduzir materialmente o rank necessário.
