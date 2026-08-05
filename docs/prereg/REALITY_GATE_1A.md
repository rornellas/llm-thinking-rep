# Pré-registro — Reality Gate 1A

**Protocolo:** `reality-gate-1a-static-heterogeneous-rank-v1`  
**Estado:** congelado antes do treinamento científico e antes de carregar o split final do WikiText-2 no runner  
**Decisão global invariável:** `NO_GO_FOR_OLMOE_OR_QWEN`

## 1. Questão

A primeira afirmação a testar é deliberadamente menor que um controlador dinâmico:

> Em teachers convencionais que atingem um critério explícito de plateau, ranks residuais heterogêneos estáticos, escolhidos somente com pesos, routing e calibração de treino, superam rank uniforme sob os mesmos limites de parâmetros e compute esperado?

Se essa afirmação falhar, não será implementado rank dinâmico por token. Isso evita adicionar um controlador complexo a uma base que não demonstra utilidade estática.

## 2. Dados

O eixo principal usa `Salesforce/wikitext`, subset `wikitext-2-raw-v1`, revisão imutável `b08601e04326c79dfdd32d625aee71d232d685c3`, tokenizado com BPE byte-level de 512 tokens treinado somente no split de treino.

- `train`: treinamento do teacher, captura e treinamento dos students;
- `validation`: plateau e calibração da alocação;
- `test`: hypothesis final;
- OOD: código, SQL, matemática, português e estruturas determinísticas codificados pelo mesmo tokenizer.

A revisão do dataset foi congelada antes da primeira preparação científica. A preparação rejeita qualquer revisão diferente e registra o SHA, fingerprints, versões e SHA-256 de todos os arrays. Os artigos são preservados por offsets e nenhum documento estatístico atravessa fronteiras de artigo; artigos longos podem ser subdivididos, mas nunca concatenados entre si. O runner inicialmente abre apenas `train` e `validation`. `test` e `ood` são carregados somente depois que os candidatos e o checkpoint foram congelados.

A fixação explícita da revisão foi adicionada depois da validação unitária do código, mas antes de qualquer carregamento do dataset, piloto, teacher ou holdout; ela é uma correção pré-científica de reprodutibilidade, sem alteração de hipótese, candidates, seeds, endpoints ou gates.

## 3. Escalas e seeds

Quatro seeds novas são usadas em duas escalas:

| Escala | d_model | d_ff | Camadas | Experts | top-k | Rank uniforme | Rank máximo heterogêneo |
|---|---:|---:|---:|---:|---:|---:|---:|
| small | 32 | 64 | 3 | 12 | 4 | 8 | 12 |
| medium | 48 | 96 | 4 | 16 | 4 | 12 | 16 |

As seeds científicas são `111731`, `121747`, `131759` e `141767`. A seed `9091` é exclusivamente um piloto de plateau e engenharia; ela abre apenas treino e validação, não entra em nenhuma decisão e não pode alterar endpoints ou margens.

## 4. Plateau

O número máximo de passos não é tratado como convergência. A cada intervalo fixo são medidos:

- validation loss em janelas fixas;
- inclinação da loss na janela recente;
- melhora máxima e amplitude da janela;
- mudança L1 da distribuição de routing.

O plateau exige simultaneamente:

1. mínimo de passos atingido;
2. slope não mais negativo que o limite pré-registrado;
3. slope positivo também abaixo do limite, para não chamar deterioração de plateau;
4. melhora recente abaixo do limite;
5. amplitude recente abaixo do limite;
6. o maior drift L1 de routing entre avaliações adjacentes da janela abaixo do limite;
7. duas janelas consecutivas aprovadas.

Sem plateau, a célula permanece diagnóstica e o gate confirmatório falha.

## 5. Trajetória de compressibilidade

Estados do teacher são preservados em 25%, 50%, 75% do orçamento máximo e no final. Em cada estado são medidos:

- rank residual necessário para 95% da energia por expert e projeção;
- frequência de routing;
- rank uniforme;
- alocação heterogênea espectral;
- fidelidade local e closed-loop após o mesmo orçamento curto de distilação.

A finalidade é testar se a compressibilidade pós-hoc piora à medida que a especialização amadurece.

## 6. Candidatos finais

1. `uniform-rank` — mesma família shared-base, rank igual em todos os experts;
2. `heterogeneous-spectral` — candidata primária;
3. `heterogeneous-routing` — controle que usa apenas frequência de routing;
4. `narrow65` — baseline convencional teacher-informed;
5. `full-identity-control` — identidade exata do teacher.

Todos os candidatos treináveis recebem os mesmos batches, passos, learning rates e objetivo closed-loop. O router permanece congelado.

## 7. Alocação heterogênea

Para cada expert e modo residual é calculada a energia singular marginal, normalizada pela energia da projeção. A utilidade esperada é ponderada pela frequência de seleção do expert.

A alocação resolve exatamente um pequeno MILP binário, com variáveis de retenção por expert e modo e restrições de prefixo, para o problema discreto:

\[
\max_{r_e} \sum_e q_e U_e(r_e)
\]

sujeito a:

\[
\sum_e r_e \le E r_{uniform}
\]

\[
\sum_e q_e r_e \le top_k\,r_{uniform}
\]

\[
1 \le r_e \le r_{max}.
\]

As restrições de prefixo garantem que rank `r` contém todos os modos `1..r`. A igualdade de rank total faz a candidata usar exatamente o mesmo orçamento de parâmetros do uniforme; a restrição ponderada impede exceder seu compute esperado. A solução não pode usar hypothesis ou OOD.

## 8. Endpoints

- delta de cross-entropy e razão de perplexidade;
- KL teacher→student;
- top-1 agreement;
- NRMSE local;
- parâmetros dos experts;
- compute matricial esperado em treino, hypothesis e OOD;
- estabilidade entre escalas;
- trajetória de rank residual;
- comparação pareada contra rank uniforme, routing-only e narrow65.

## 9. Estatística

- unidade hierárquica: seed e documento;
- janelas são agregadas dentro da célula seed×documento;
- bootstrap crossed com 20.000 amostras;
- comparações load-bearing são pareadas;
- valores por seed e duas escalas são obrigatórios;
- auditor independente não importa o agregador nem o bootstrap do projeto;
- análise leave-one-seed-out da comparação spectral − uniform é obrigatória.

## 10. Gates

O `PASS` exige nas duas escalas:

- plateau em todas as seeds;
- parâmetros e compute esperado da candidata não superiores ao rank uniforme;
- held-out compute sem drift superior a 2 pontos percentuais;
- hypothesis loss UCB95 `<= +0.030 nat`;
- OOD loss UCB95 `<= +0.050 nat`;
- KL hypothesis UCB95 `<= 0.20` e KL OOD UCB95 `<= 0.25`;
- nenhuma seed hypothesis acima de `+0.060 nat`;
- top-1 LCB95 `>= 0.78`;
- local NRMSE UCB95 `<= 0.24`;
- spectral − uniform loss UCB95 `<= 0`;
- pelo menos dois de KL, top-1 e local superiores/não inferiores ao uniforme;
- spectral não inferior ao routing-only em loss;
- não-inferioridade a narrow65 nas margens congeladas;
- identidade, dados, tendência de escala e auditoria aprovados.

Um `MECHANISM_SIGNAL` requer ao menos uma escala plateaued com loss dentro de `+0.010 nat` do uniforme e duas melhoras comportamentais, sem violações de orçamento ou integridade.

## 11. Stop rules

- Se heterogeneidade espectral não vencer rank uniforme nas duas escalas, o controlador dinâmico fica bloqueado.
- Se routing-only igualar a candidata, a decomposição espectral não está justificada.
- Se a vantagem diminuir com escala, não haverá extrapolação para modelos reais.
- Se os teachers não atingirem plateau, o experimento não será reinterpretado como evidência positiva.
- Se a família falhar, a linha principal passa a ser arquitetura compacta treinada nativamente desde cedo, comparada a baselines MoSE/RFID-like.

## 12. Limites

Mesmo um `PASS` continua sendo um experimento controlado em WikiText-2 pequeno, com compute analítico e sem kernel. Ele não autoriza OLMoE, Qwen, speedup ou claims de estado da arte.
