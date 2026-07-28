# Pré-registro — shared base + resíduos bilaterais low-rank v1

**Protocolo:** `pre-qwen-alignment-tolerant-shared-low-rank-v1`  
**Status:** congelado antes da materialização dos novos documentos hypothesis/OOD.  
**Decisão global preservada:** `NO_GO_FOR_OLMOE_OR_QWEN`.

## Pergunta

Mantendo os cinco teachers e checkpoints congelados da replicação teacher-width,
uma parametrização alignment-tolerant com largura interna completa consegue igualar
ou superar o baseline convencional teacher-informed de 65%, usando estritamente
menos de 65% dos parâmetros dos experts e do proxy de operações matriciais roteadas?

## Família testada

Para cada matriz SwiGLU de cada especialista:

\[
W_e = W_{shared} + L_e R_e.
\]

`W_shared` é full-rank e compartilhada. `L_e` e `R_e` são fatores específicos por
especialista. Diferentemente do Modal escalar, a especialização não depende de uma
combinação escalar global: cada especialista pode representar rotações e subespaços
próprios dentro do rank permitido.

A execução é direta. A projeção compartilhada é feita uma vez por token; os fatores
específicos são aplicados somente aos experts roteados. Nenhuma matriz densa por
expert é reconstruída no caminho de inferência.

## Candidatos congelados

| Candidato | Rank | Params experts | Proxy compute | Papel |
|---|---:|---:|---:|---|
| `shared-lora-r4` | 4 | 35,00% | 51,67% | âncora de menor compute |
| `shared-lora-r5` | 5 | 41,67% | 58,33% | primário |
| `shared-lora-r6` | 6 | 48,33% | 65,00% | controle de capacidade |
| `narrow65-frozen-baseline` | — | 65,00% | 65,00% | baseline forte congelado |
| `full-continuation-control` | — | 100% | 100% | controle de recuperação |

Para dimensão `D=24`, largura `H=40`, `E=12` experts e top-`T=4`, o fatorizado de
rank `r` possui:

\[
\rho_P = \frac{1}{E} + \frac{r(H+D)}{HD}
\]

\[
\rho_C = \frac{1}{T} + \frac{r(H+D)}{HD}.
\]

Essas razões contabilizam somente parâmetros dos experts e multiplicações
matriciais dominantes. Não são claims de latência.

## Teachers e baselines

- Seeds: `91121, 92129, 93133, 94151, 95153`.
- Teachers e estados `narrow65`/`full-control` vêm dos checkpoints congelados da
  replicação teacher-width.
- O baseline não é retreinado nem selecionado nos novos documentos.
- Os teachers não possuem plateau demonstrado; este protocolo é um screen em
  teachers fixos, não uma confirmação de convergência.

## Treinamento dos candidatos novos

Inicialização:

1. base compartilhada = média das matrizes dos experts do teacher;
2. resíduo de cada expert = SVD truncada, com fatores balanceados por `sqrt(sigma)`;
3. router copiado e congelado.

Treino:

- 320 passos de distilação local inicial;
- 600 passos de distilação local adicional;
- 180 passos closed-loop com pesos `0,60 local / 0,35 KL / 0,05 CE`;
- mesmos batches e seeds derivados para os três ranks.

## Dados e prevenção de leakage

- O treino reutiliza apenas o split de treino do experimento teacher-width.
- Os novos documentos `alignment-hypothesis-v1` e `alignment-ood-v1` têm conteúdo,
  seeds e identificadores novos.
- Eles são materializados somente depois que todos os candidatos estão congelados.
- Haverá auditoria de duplicatas exatas e near-duplicates entre train/hypothesis/OOD.
- Os holdouts antigos não são usados para selecionar ou ajustar candidatos.

## Estatística

- Unidade primária: célula `seed × documento`.
- Janelas do mesmo documento são agregadas antes da inferência.
- Bootstrap crossed independente sobre seeds e documentos.
- 20.000 amostras; IC de 95%.
- Comparação load-bearing: `shared-lora-r5 - narrow65-frozen-baseline`.

## Gates

O `PASS` exige simultaneamente:

- UCB95 hypothesis do rank-5 `<= +0,030 nat`;
- UCB95 OOD do rank-5 `<= +0,050 nat`;
- toda seed rank-5 `<= +0,060 nat` em hypothesis;
- UCB95 da diferença rank-5 menos narrow65 `<= 0`;
- parâmetros e compute do rank-5 estritamente `< 0,65`;
- controles full dentro dos limites congelados;
- auditoria de dados limpa;
- auditor independente `PASS`.

`PROMISING_ONLY` é permitido quando o rank-5 não passa, mas o rank-6, com 65% de
compute e menos parâmetros, iguala o narrow65. Esse resultado não satisfaz a meta
forte de compute estritamente menor.

## Interpretação antecipada

- `PASS`: justifica replicação fresca em teachers com plateau e implementação de
  runtime; não autoriza OLMoE/Qwen.
- `PROMISING_ONLY`: melhora a hipótese arquitetural, mas não cruza o baseline de
  compute.
- `FAIL`: falsifica ranks 4–6 com esta inicialização e budget nos teachers fixos;
  não falsifica toda arquitetura alignment-tolerant.
