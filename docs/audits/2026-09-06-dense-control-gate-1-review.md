# DCG-1 — revisão científica e decisão

Data local: 6 de setembro de 2026, America/Sao_Paulo. Execução: GitHub Actions 34072646476. Fonte científica congelada: c056e605d8c3f11b5056918f6e6d01c3a78305b3. Pré-registro: b6435490bc62e4fee4061841159b6d9189c80824.

## Veredito

**DCG1_NO_CAPACITY_ADVANTAGE_DEMONSTRATED.** A receita compartilhada/esparsa examinada não demonstrou vantagem de capacidade contra controles densos pareados. Nenhum dos quatro contrastes primários passou. A implementação original em laço também perdeu em latência. Não escalar esta receita, não investir em controlador dinâmico para resgatá-la e não tratar a compressão anterior como prova de uma fronteira superior.

A confirmação FCC-1 continua válida em seu escopo: conseguimos comprimir nossos próprios checkpoints preservando fidelidade dentro das margens. DCG-1 responde a uma pergunta diferente e mais exigente: essa representação é melhor que uma alternativa densa simples com orçamento comparável? Nesta receita, não demonstramos isso. **NO_GO_FOR_OLMOE_OR_QWEN permanece.**

## Experimento executado

Foram 30 treinamentos novos: cinco arquiteturas por seis sementes pareadas, 4.400 atualizações e 2.252.800 tokens apresentados por arquitetura/semente. O sexto braço avaliado, rank8-to1, é transformação por SVD do pai rank8, sem treinamento adicional. Preservamos estados e otimizadores em 800, 2.200 e 4.400 passos, todas as perdas de treino, métricas de desenvolvimento e 36 exports finais. Nenhum checkpoint foi escolhido pelo resultado de teste.

Mesma inicialização fora do FFN e mesmos lotes dentro de cada semente. Modelos pequenos com d_model32, duas camadas, quatro cabeças, vocabulário512 e contexto64. AdamW com taxa constante0,0003, weight_decay0,02 e clipping1. Os esparsos mantêm a penalização auxiliar de roteamento; densos não recebem uma penalização sem significado. É uma comparação de receitas fixas, não de arquiteturas individualmente otimizadas.

O holdout contém 256 artigos top-level verdadeiros, quatro janelas por artigo, 65.536 previsões por modelo. Os índices estão em [16384,24576), fora do conjunto FCC-1 e do prefixo que treinou estes modelos/tokenizer. A preparação reproduziu exatamente o prefixo antigo; os últimos artigos verdadeiros desse prefixo são97 para treino e206 para tokenizer. Também excluiu coincidências exatas de janelas de65 tokens com treino, calibração antiga e FCC-1. Isso não prova independência semântica entre temas da Wikipedia.

Cada job recebeu os dados primários apenas depois de congelar seus checkpoints e hashes. Foram usados somente dados de desenvolvimento nos marcos intermediários. Não houve busca de hiperparâmetros, ajuste após teste, extensão oportunista do treinamento ou relaxamento dos critérios.

## Resultado de qualidade e custo

| Braço | NLL em artigos novos, menor é melhor | Parâmetros totais | MACs FFN/roteador por token | Forward1 token, ms | Forward64 tokens, ms |
|---|---:|---:|---:|---:|---:|
| native-rank1 | 4,269019 | 47.168 | 15.360 | 1,3374 | 3,0821 |
| native-rank8 | 4,244904 | 95.552 | 31.488 | 1,3352 | 3,4116 |
| dense104 | 4,227333 | 47.168 | 19.968 | 0,3775 | 0,6407 |
| dense80 | 4,258728 | 42.560 | 15.360 | 0,3761 | 0,6320 |
| dense164 | 4,196007 | 58.688 | 31.488 | 0,3776 | 0,6737 |
| rank8-to1 | 4,248656 | 47.168 | 15.360 | 1,3288 | 3,0103 |

**Mesmo tamanho:** dense104 tem exatamente os47.168 parâmetros dos dois candidatos rank1. Sua perda foi menor em seis de seis sementes contra native-rank1 e em cinco de seis contra rank8-to1. Não houve demonstração de não inferioridade desses candidatos à margem de+0,010 nat pelo critério simultâneo usado. O denso de mesmo tamanho usa mais MACs analíticos que os compactos; portanto não afirmar dominância em todos os eixos matemáticos.

**Mesmo custo matricial de inferência:** dense80 inclui uma correção importante do planejamento anterior. A largura76 igualava apenas as transformações dos especialistas, mas omitia768 MACs do roteador por token, somando as duas camadas. Usar largura80 incorpora esse custo. A correção foi registrada antes de observar resultados. Esse denso usa menos parâmetros,42.560, e sua perda ficou entre os dois candidatos compactos na média.

**Custo do pai:** rank8-to1 herda o treinamento de95.552 parâmetros e31.488 MACs FFN/roteador, não o orçamento reduzido do modelo exportado. dense164 é um controle contextual desse custo matricial do pai:58.688 parâmetros, mesma contagem31.488, perda menor. A equivalência é de multiplicações matriciais forward; não de FLOPs totais de treino, incluindo backward, otimização e penalizações.

## Inferência estatística pré-especificada

| Candidato menos controle | Diferença média NLL | Limite superior t98,75% | Limite superior bootstrap cruzado98,75% |
|---|---:|---:|---:|
| native-rank1 menos dense104 | +0,041686 | +0,063024 | +0,055031 |
| native-rank1 menos dense80 | +0,010291 | +0,032025 | +0,023331 |
| rank8-to1 menos dense104 | +0,021323 | +0,041656 | +0,033615 |
| rank8-to1 menos dense80 | −0,010071 | +0,007792 | +0,000923 |

Exigimos ambos os limites superiores no máximo−0,010 nat e nenhuma semente acima de+0,010. A alocação98,75% foi pré-fixada para quatro contrastes, por Bonferroni. O t é pareado entre seis sementes; o bootstrap cruza sementes e256 artigos com10.000 reamostragens. Os métodos têm pressupostos e cobertura populacional limitada, não são garantia universal.

O sinal favorável não deve ser omitido: rank8-to1 teve NLL média0,010071 nat menor que dense80 e cumpriu o critério descritivo de não inferioridade contra esse controle. **Não passou superioridade:** os limites superiores ainda são positivos. É uma relação entre qualidade, parâmetros e custo que merece ser registrada, não uma vitória confirmada. Tampouco houve não inferioridade demonstrada contra dense104. Resultado não significativo não é prova de igualdade.

## Latência e armazenamento reais

Usamos a implementação original, não a vetorização ECK-1 reprovada. Medimos os seis exports no mesmo runner por semente, dois threads CPU, batch1, comprimentos1/64, sem cache KV e sem computar perdas auxiliares de treino. Foram cinco aquecimentos e nove blocos de dez forwards com ordem aleatorizada.

As medianas das razões pareadas rank8-to1/dense104 foram **3,5368 no comprimento1 e4,6324 no64**. Para native-rank1/dense104 foram3,5468 e4,7727. A lentidão ocorreu nessa implementação e escala; não representa um limite físico universal da arquitetura nem medição de serving autoregressivo, GPU, energia ou pico de RAM. Mas impede qualquer anúncio de ganho de execução com o código validado atual.

Os tempos médios acumulados apenas dos passos de treino foram129,69s em native-rank1,144,65s em native-rank8,37,45s em dense104,34,56s em dense80 e36,77s em dense164. São descritivos, com ordem de execução rotativa, excluindo avaliações, setup e criação dos modelos. rank8-to1 reutiliza o custo do pai; não é um sexto treinamento independente. Pequenas inversões entre larguras densas não autorizam concluir uma lei de escala de runtime.

Há também overhead de serialização: os dois modelos com47.168 parâmetros têm a mesma quantidade188.672 de bytes de tensores FP32, mas arquivos PyTorch distintos. dense104 ocupa197.685 bytes; rank8-to1,242.443; native-rank1,242.977. Muitos tensores pequenos acrescentam metadados. Não confundir contagem de parâmetros, tamanho do arquivo e memória de pico.

## Compressibilidade não é eficiência de capacidade

Na medição secundária aos4.400 passos, truncar o pai rank8 causou aumento médio de NLL de**0,0037520 nat**, KL médio**0,0026460** e concordância top1**92,7127%**. Os limites superiores t95/bootstrap95 de NLL foram0,0048455/0,0046181; os de KL,0,0034996/0,0032822. São números pequenos nessas métricas, mas não equivalência de comportamento. Não promover esse endpoint secundário a aprovação do objetivo primário perdido.

Treinar o pai maior e comprimir depois produziu perda média menor que treinar rank1 desde o início. A comparação não identifica sua causa: capacidade intermediária, inicialização e geometria da otimização podem contribuir. Não prova que a compressão pós-treino seja a melhor forma de atingir aquele tamanho; o controle denso é precisamente a evidência contrária a essa interpretação ampla.

O diagnóstico suplementar, explicitamente descritivo, recalculou stable rank por autovalores de Gram. native-rank8 terminou com média1,0342 e norma residual/comum0,1533; native-rank1 com rank1 e razão0,0387. Esses números não provam especialização útil nem explicam causalmente a diferença de perda. Também não justificam aumentar rank indiscriminadamente.

As curvas pareadas de desenvolvimento foram preservadas. dense104:5,0579 →4,4803 →4,1986; rank8-to1:5,0784 →4,4922 →4,2227, respectivamente800/2200/4400 passos. Todos ainda melhoram entre os dois marcos finais. Uma diferença entre dois pontos não certifica convergência; o experimento testa desempenho no orçamento fixado. Estender apenas o candidato depois de observar o teste violaria a comparação.

## Auditoria e limitações aceitas

Os19 testes de preflight passaram: dimensões/orçamentos, embeddings compartilhados, inicialização pareada e determinística, forward denso manual, equivalência com caminho esparso anterior, gradientes finitos, atualização de pesos, rank/storage, artigos top-level e contratos estatísticos. Não foi executada toda a suíte histórica do repositório.

A agregação verificou todos os seeds/checkpoints/exports/hashes, isolamento dos dados, coincidências de janelas, aritmética por janela/artigo e bootstrap por duas vias. Todos os36 exports finais foram recarregados. Uma implementação distinta materializou especialistas, reconstruiu rank1 por Gram e usou NLL logsumexp em float64 nos primeiros16 artigos de dois modelos da semente0. O maior erro de perda por janela foi6,31e-7, abaixo da tolerância2e-5; não se afirma reexecução independente de todas as janelas por essa implementação.

Depois, o pacote de artefatos foi baixado e auditado em outro ambiente local, Python3.13.5 em vez de3.12.14. Foram verificados **83 arquivos**, **36 exports** por armazenamento real sem duplicar embeddings, os quatro contrastes por terceira via aritmética, percentis por interpolação explícita e razões de latência. Resultado:PASS. O script foi comitado antes da leitura dos resultados agregados. Isso é independência de implementação/aritmética e runtime, não replicação do treinamento por outro grupo científico.

Os arquivos `external-archive-audit.json` e `supplementary-diagnostics.json` preservam essas execuções. São suplementos posteriores e não fazem parte do manifesto original `SHA256SUMS`, que não foi reescrito. Os dados brutos e scripts permitem regenerá-los. A gravação do script suplementar sofreu um conflito409 durante o commit de resultados do bot; foi repetida depois, sem alterar os resultados ou medições.

## Decisão de continuidade

Esta rodada está concluída. **Suspender escala, controlador dinâmico e alegações de superioridade desta receita.** dense104 passa a ser o controle obrigatório de mesmo tamanho e dense80 o de custo matricial correspondente. O resultado FCC-1 é mantido como compressão fiel em checkpoints específicos, não como objetivo final alcançado.

Ainda não está identificado se o déficit de treinamento nativo resulta principalmente de otimização ou de restrição representacional. A única continuação de baixo custo justificada para esta mesma parametrização é um screen limitado de otimização com o mesmo orçamento de busca para compacto e densos, exclusivamente em treino/desenvolvimento, pré-registrando candidatos, critérios e regra de abandono. Só um sinal consistente nesse screen justificaria nova confirmação em outros artigos. Sem esse sinal, arquivar esta parametrização e formular outra hipótese, em vez de multiplicar variantes até obter um resultado favorável.

Esse screen não foi iniciado nesta rodada. Nenhum artigo DCG-1 pode ser usado para ajustá-lo: o holdout agora está revelado. Não iniciar novo treinamento apenas porque um contraste secundário parece favorável. Uma falha nesta receita não é prova de impossibilidade da família inteira; tampouco sua possibilidade lógica é razão suficiente para continuar gastando sem teste discriminante.

## Rastreabilidade

- Protocolo: `docs/prereg/DENSE_CONTROL_GATE_1.md`.
- Modelo/execução: `pre_qwen_certification/dense_control.py`, `scripts/run_dense_control_gate_1.py`.
- Preparação: `scripts/prepare_dense_control_gate_1.py`, `data/dense-control-gate-1/`.
- Agregação e resultados: `scripts/aggregate_dense_control_gate_1.py`, `results/dense-control-gate-1/summary.json`, `docs/results/2026-09-06-dense-control-gate-1.md`.
- Auditoria local: `scripts/audit_dense_control_archive.py`, `results/dense-control-gate-1/external-archive-audit.json`.
- Diagnóstico suplementar: `scripts/diagnose_dense_control_gate_1.py`, `results/dense-control-gate-1/supplementary-diagnostics.json`.
- Pacote original: artefato10001152512, SHA256 bbff89170a3cc339b64a555944710a33b79fccb50d68d961120ef43afc10d516.
