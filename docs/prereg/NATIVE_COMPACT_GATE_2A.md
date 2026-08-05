# Pré-registro — Native Compact Gate 2A

**Protocolo:** `native-compact-gate-2a-v1`  
**Estado:** congelado antes da preparação dos dados e do treinamento científico  
**Decisão global invariável:** `NO_GO_FOR_OLMOE_OR_QWEN`

## 1. Pergunta

O Reality Gate 1A mostrou que converter experts convencionais já treinados para uma base compartilhada com resíduos low-rank não produziu heterogeneidade útil e preservou comportamento pior que `narrow65`. O Gate 2A testa a hipótese causal mais simples que restou:

> Uma base full-rank compartilhada com resíduos low-rank por expert, treinada desde a inicialização, alcança loss não inferior a um MoE convencional `narrow65` usando menos parâmetros de expert e compute analítico de expert não superior?

Este protocolo não testa rank dinâmico, seleção de prefixos, kernels, runtime ou checkpoints reais. Esses passos continuam bloqueados.

## 2. Princípio experimental

Os candidatos são treinados do zero, não transplantados de um teacher. Para cada escala e seed:

- embeddings, attention, layer norms, output head e posições começam idênticos;
- os routers começam idênticos;
- cada candidato recebe exatamente os mesmos batches, na mesma ordem;
- todos recebem o mesmo número de updates e a mesma configuração do otimizador;
- a inicialização dos transforms de expert é determinística e específica da arquitetura;
- nenhuma arquitetura recebe distilação, teacher labels ou budget adicional.

O endpoint primário é a loss no split de hipótese após o número fixo de updates. O melhor checkpoint por calibração é secundário e não substitui o endpoint final.

## 3. Dados

Será usado `Salesforce/wikitext`, subset `wikitext-103-raw-v1`, em uma revisão imutável resolvida e registrada antes do primeiro treinamento.

- `train`: treinamento de todos os candidatos;
- `validation`: calibração e seleção secundária do melhor checkpoint;
- `test`: endpoint final de hipótese;
- `ood`: código, SQL, matemática, português e estruturas determinísticas novas.

O tokenizer BPE byte-level de 512 tokens será treinado somente nos primeiros 2.048 artigos do split de treino. Artigos são preservados por offsets; nenhuma janela estatística cruza fronteiras de artigo. `test` e OOD só serão abertos depois que os estados final e best-calibration dos três candidatos forem serializados e hasheados.

## 4. Escalas e seeds

Seeds confirmatórias: `202781`, `212789`, `222793` e `232801`.

| Escala | d_model | d_ff full | Camadas | Experts | top-k | Updates | Rank nativo | d_ff narrow |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| small | 32 | 64 | 2 | 12 | 4 | 2.200 | 8 | 42 |
| medium | 40 | 80 | 3 | 16 | 4 | 3.000 | 10 | 52 |

A seed `19081` é exclusivamente piloto de engenharia, com 80 updates e sem acesso a `test` ou OOD. Ela não altera endpoints, margens ou candidatos.

## 5. Candidatos

### 5.1 `conventional-full`

MoE SwiGLU convencional com largura integral. É o teto de capacidade, não o comparator primário.

### 5.2 `conventional-narrow65`

MoE convencional treinado nativamente com aproximadamente 65% da largura intermediária. É o comparator primário simples e forte.

### 5.3 `native-shared-rank`

Cada banco segue:

\[
W_e = W_{shared} + L_e R_e.
\]

A base full-rank é aprendida conjuntamente desde o primeiro update. Cada expert possui resíduos bilaterais de rank uniforme. As projeções compartilhadas são executadas uma vez por token; somente os fatores dos experts roteados são executados por expert.

## 6. Orçamentos analíticos

Para um bloco com `E` experts, largura `h`, dimensão `d`, top-k `k` e rank `r`:

\[
P_{native}=3hd + 3E(h+d)r
\]

\[
C_{native}=3hd + 3k(h+d)r.
\]

Para `narrow65`, com largura `h_n`:

\[
P_{narrow}=3Eh_nd
\]

\[
C_{narrow}=3kh_nd.
\]

Nos dois tamanhos, o candidato primário tem compute analítico de expert de 62,5% do full. O `narrow65` fica entre 65,0% e 65,625%. A vantagem mínima pré-registrada de parâmetros de expert do primário sobre `narrow65` é 15 pontos percentuais.

Esses valores são proxies de multiplicações matriciais. Nenhuma alegação de velocidade de runtime será feita.

## 7. Treinamento e avaliação

Cada candidato recebe AdamW, weight decay, clipping, auxiliary routing loss e learning rate congelados por escala. O mesmo batch é reutilizado pelos três candidatos antes de amostrar o batch seguinte.

A cada intervalo fixo são registrados:

- loss de treino;
- loss de calibração em janelas fixas;
- distribuição de routing da primeira camada;
- melhor checkpoint de calibração.

No holdout serão registrados por seed, artigo e janela:

- cross-entropy;
- perplexidade;
- entropia preditiva;
- confiança média;
- distribuição de routing em todas as camadas;
- número de experts sem seleção.

A inclinação terminal da calibração é diagnóstica. O Gate 2A é deliberadamente um screen de budget fixo; plateau não é requisito para reinterpretar o endpoint primário.

## 8. Estatística

- unidade hierárquica: seed e artigo;
- janelas são agregadas dentro da célula seed×artigo;
- bootstrap crossed com 10.000 amostras;
- comparações load-bearing são pareadas, porque os candidatos usam as mesmas janelas e seeds;
- intervalos são de 95%;
- resultados por seed são obrigatórios;
- auditor independente recompõe estatísticas, budgets, routing, hashes, isolamento de dados e decisão sem importar o agregador do projeto.

## 9. Gates

O `PASS` exige, nas duas escalas:

1. vantagem de pelo menos 15 pontos percentuais em parâmetros de expert contra `narrow65`;
2. compute analítico de expert não superior ao `narrow65`;
3. UCB95 de `native-shared-rank − narrow65` na hypothesis loss `<= +0,010 nat`;
4. UCB95 equivalente em OOD `<= +0,020 nat`;
5. nenhuma seed com diferença de hypothesis loss acima de `+0,025 nat`;
6. nenhum expert morto em qualquer camada do primário no holdout de hipótese;
7. regressão da diferença medium contra small não superior a `+0,005 nat`;
8. proveniência, isolamento, checkpoints e auditoria independentes aprovados.

A comparação contra `conventional-full` usa UCB95 `<= +0,035 nat` como diagnóstico de capacidade, mas não é load-bearing para o PASS contra `narrow65`.

Um `MECHANISM_SIGNAL` exige pelo menos uma escala dentro de `+0,020 nat` contra `narrow65`, com budgets, routing e integridade intactos. Ele autoriza somente uma replicação fresca Gate 2B.

## 10. Stop rules

- Se o primário falhar materialmente contra `narrow65`, nested/dynamic rank permanece bloqueado.
- Se o primário perder a vantagem de parâmetros ou compute, a formulação deixa de ser Pareto-relevante.
- Se routing colapsar, o resultado não será atribuído somente à capacidade dos experts.
- Gates não serão relaxados após os resultados.
- Um PASS não autoriza OLMoE, Qwen, speedup ou claim de estado da arte.

## 11. Relação com trabalhos externos

- MoSE (`arXiv:2602.06154`) treina experts slimmable em múltiplas larguras. Esse controle é deliberadamente adiado para Gate 2B, porque mistura a hipótese de compartilhamento nativo com treinamento multi-width.
- HMoE (`arXiv:2408.10681`) testa heterogeneidade de tamanhos de expert.
- RFID-MoE (`arXiv:2602.09316`) é uma técnica post-training de compressão orientada por routing e information density; não responde à hipótese causal de treinamento nativo deste gate.
- StructMoE (`PMLR 2024`) usa experts low-rank com roteamento adicional, uma classe arquitetural diferente.

O Gate 2A procura a menor evidência necessária antes de incorporar essas complexidades.

## 12. Consequência

```text
NO_GO_FOR_OLMOE_OR_QWEN
```

Mesmo um PASS apenas autoriza o Gate 2B: prefixes aninhados, budgets múltiplos e um comparator inspirado em MoSE. Rank dinâmico por token continua condicionado a uma vantagem estática/nativa demonstrada.
