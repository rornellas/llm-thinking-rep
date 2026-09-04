# Revisão científica — ablação funcional, compressão e execução

Data: 4 de setembro de 2026. Estado: rodada concluída, com evidência positiva limitada e falhas preservadas.

## Resultado principal

**Reduzimos de 95.552 para 47.168 os parâmetros totais dos oito checkpoints compactos primários, uma redução de 50,6363%, e confirmamos fidelidade dentro das margens pré-registradas em artigos não utilizados por esses checkpoints.** Não houve treinamento ou ajuste para obter a compressão. A transformação foi truncar cada resíduo particular de especialista, de rank nominal 8 para rank 1, por SVD.

Isso confirma uma melhoria de armazenamento em relação ao nosso próprio compacto anterior. Não confirma uma arquitetura inédita, superioridade a modelos convencionais de tamanho semelhante, aceleração de inferência, convergência do treinamento nem aplicabilidade direta a LLMs reais. `NO_GO_FOR_OLMOE_OR_QWEN` permanece.

## O que efetivamente foi executado

**FA-1:** 36 checkpoints congelados, incluindo os 24 da intervenção MUI-1 e os 12 da escala pequena do Gate 2A. Intervenções removem resíduos, substituem especialistas por matrizes médias, permutam as escolhas dos especialistas, uniformizam pesos de roteamento e truncam resíduos para rank 1. Foram preservados 112 modelos exportados de verdade, sem manter parâmetros descartados escondidos no objeto. O treino não foi repetido.

**FCC-1:** confirmação em 256 artigos verdadeiros, quatro janelas por artigo, 65.536 tokens avaliados por modelo. O protocolo foi comitado antes da preparação e avaliação. Os oito checkpoints primários foram avaliados contra suas versões comprimidas; os convencionais completo e estreito foram controles contextuais. A auditoria combinou reexecução por implementação independente de duas amostras, aritmética por múltiplas vias e checagem de hashes/contaminação exata.

**ECK-1:** tentativa separada de tornar a execução compacta mais rápida por vetorização. Os testes sintéticos de forward e gradientes passaram, mas a equivalência numérica falhou em dois dos quatro jobs nos modelos treinados. Esse teste foi reprovado; os resultados parciais favoráveis não foram promovidos a aceleração demonstrada.

## Evidência em dados novos

| Grupo de checkpoints | Atualizações do treinamento original | Δ NLL médio, rank1 menos original | Limite superior t95 | Limite superior bootstrap cruzado95 | Concordância do token mais provável |
|---|---:|---:|---:|---:|---:|
| MUI-1 legacy | 800 | +0,00007982 nat | +0,00010827 | +0,00010740 | 99,7971% |
| Gate 2A native-shared-rank | 2.200 | +0,00057942 nat | +0,00120937 | +0,00098803 | 97,6814% |

A margem pré-fixada foi Δ NLL de no máximo +0,010 nat para ambos os limites superiores. Também exigimos KL <=0,005 nat pelos dois métodos, nenhum seed acima de +0,025 nat e pelo menos 25% de redução de parâmetros totais. Todos os requisitos passaram nos dois grupos. O maior limite superior de KL entre os grupos/métodos foi aproximadamente 0,000474 nat. Os intervalos e cada seed estão no resumo bruto; não se trata de igualdade numérica perfeita, como a concordância top-1 já evidencia.

Os grupos usam seeds diferentes e não constituem uma curva de aprendizagem pareada. Nenhum deles foi declarado convergido. O ganho é em checkpoints específicos de laboratório, com d_model=32, duas camadas, vocabulário512 e contexto64, não em um modelo comercial de bilhões de parâmetros.

### O que garante que os artigos são novos para esses modelos

Reconstruímos exatamente o prefixo tokenizado que alimentou o experimento anterior. O último artigo verdadeiro usado nesse prefixo foi o97; o tokenizer foi ajustado até o artigo206. O novo sorteio usou o intervalo de índices8192 a16383, com seleção e janelas fixadas antes de calcular qualquer perda.

O dataset chama essa parte de `train`, mas, para estes modelos que só consumiram um prefixo curto, ela funciona como holdout. Essa distinção é explicitada no manifesto. Comparamos as 1.024 janelas novas de65tokens com todas as janelas possíveis dos arrays antigos de treino e calibração: nenhuma coincidência exata. Isso não prova ausência de sobreposição semântica de assuntos na Wikipedia, nem generalização para outros domínios.

O bootstrap cruza as quatro seeds com os256 artigos verdadeiros, 10.000 reamostragens. Além dele, exigimos o limite t sobre seeds. Esses procedimentos têm população e pressupostos limitados; não transformam quatro seeds em uma garantia universal.

## A descoberta funcional que muda a direção

O diagnóstico anterior tratava a concentração em aproximadamente um modo como possível defeito. A ablação mostrou uma leitura mais útil: **a função aprendida precisa de resíduos e de escolhas de especialistas, mas, nos checkpoints examinados, não precisa da maior parte das direções residuais armazenadas.**

Retirar todos os resíduos aumentou a perda em aproximadamente +0,0420 e +0,0535 nat nos dois grupos de desenvolvimento. Substituir os especialistas por uma única transformação de matrizes médias aumentou cerca de +0,0310 e +0,0302. Permutar os especialistas escolhidos aumentou cerca de +0,0413 e +0,0478. Essas intervenções falharam nos critérios de fidelidade, ao contrário da truncagem para rank1.

Portanto, o sinal não é simplesmente um modelo denso disfarçado que tolera qualquer especialista. Tampouco prova que os especialistas representem competências semânticas ou que exista uma única base global: rank1 foi aplicado separadamente a cada resíduo/projeção/especialista. O resultado é compatível com correções específicas de baixa dimensão úteis para a função aprendida. Tentar forçar rank elevado pode gastar capacidade sem benefício; isso não justifica afirmar que rank elevado nunca será necessário.

## Armazenamento e operações não são velocidade

Os bytes dos parâmetros FP32 caíram de382.208 para188.672. Os MACs analíticos das matrizes dos especialistas caíram de30.720 para14.592 por token, redução de52,5%. Essa não é uma medição de FLOPs totais, pico de RAM, energia ou tokens/s.

Na implementação em laço usada para a confirmação, a redução de rank quase não ajudou a latência com prefixo de um token e proporcionou aproximadamente9% em comprimento64 no teste de desenvolvimento. Não satisfez o critério de pelo menos10% nos dois comprimentos e grupos.

A tentativa de vetorização foi matematicamente compatível, mas não satisfez a tolerância FP32 pré-especificada em todos os checkpoints. O diagnóstico pós-hoc reproduziu um caso: na segunda camada, o token39 trocou o especialista2 pelo3. A distância entre o quarto e quinto logits do router era apenas 2,384e-7; a diferença máxima dos logits de roteamento chegou a9,537e-7. O erro final dos logits do modelo atingiu0,0566767.

Forçar as escolhas e probabilidades originais reduziu esse erro a1,431e-6, dentro da tolerância original. Isso sustenta a hipótese de amplificação por uma decisão top-k quase empatada nesse exemplo. Não autoriza generalizar o mecanismo a todos os casos nem mudar o FAIL para PASS. O kernel continua experimental e não certificado para substituir a implementação usada nas métricas de qualidade.

## Controles que impedem uma conclusão inflada

No grupo de 2.200 atualizações, em artigos novos, a perda média foi4,5021345 no compacto comprimido,4,4819518 no convencional estreito e4,4496003 no completo. Portanto, ainda existe uma diferença de qualidade frente a esses modelos, embora tenham mais parâmetros. No grupo de800 atualizações o ordenamento difere; grupos, seeds e regimes não permitem concluir que o compacto domina em geral.

O próximo teste de capacidade deve incluir um denso SwiGLU de largura104, com exatamente47.168 parâmetros totais, e um denso de largura76, com MACs das matrizes equivalentes aos14.592 do compacto rank1, mantendo o mesmo restante da arquitetura. Esses baselines ainda não foram treinados nesta rodada. Comparar apenas contra nosso compacto antigo é insuficiente para alegar melhoria da fronteira entre qualidade e custo.

SVD, bases compartilhadas com deltas de baixo rank e vetorização não são novidades por si só. D²-MoE e DeRS já exploram famílias relacionadas. Aqui há uma confirmação específica de compressibilidade e um mecanismo de falha de execução diagnosticado, não uma alegação de originalidade de arquitetura.

## Integridade, correções e replicabilidade

Uma correção importante foi preservada: o segmentador antigo separava documentos em qualquer título, incluindo subseções. Os142 identificadores em FA-1 são segmentos, não142 artigos independentes. FA-1 já utilizava incerteza apenas sobre seeds, condicional às janelas; sua aritmética não muda. Afirmações antigas de independência por artigo precisam de auditoria própria. FCC-1 usa limites top-level verdadeiros.

Em FA-1, uma implementação separada materializou todos os especialistas e recalculou rank1 por autovetores da matriz de Gram, em vez do caminho de SVD do avaliador. Conferiu112 exports e reexecutou16 intervenções completas. Em FCC-1, o mesmo tipo de implementação independente reexecutou16 artigos por checkpoint primário selecionado de cada grupo. Diferenças máximas de perda por janela foram4,768e-7 e de KL,1,218e-9. A agregação foi recomputada por multiplicidades das reamostragens; depois, em outro runtime local, rechecamos13 arquivos do pacote e todos os limites com somas explícitas em Python e percentis interpolados. Essa é independência de implementação/aritmética, não replicação por outro grupo científico.

Falhas de infraestrutura também estão registradas: dependênciaSciPy ausente no primeiro preflight de FA-1, booleanoNumPy não serializável na primeira agregação e concorrência de push. Foram corrigidas sem mudar endpoints nem refazer medições para escolher resultados. O resultado negativo de ECK-1 está integralmente preservado. Não foi executada toda a suíte histórica do repositório; foram executados os testes específicos e herdados declarados nos workflows.

## Decisão operacional

1. Manter **rank1 pós-treino em implementação original** como candidato de compressão com fidelidade confirmada no escopo FCC-1.
2. Manter o kernel vetorizado como **FAIL_INVALID_FOR_SPEEDUP_CLAIM**, sem relaxar a tolerância ou selecionar apenas células favoráveis.
3. Priorizar comparação contra densos com orçamento pareado antes de escalar ou inventar novos mecanismos de rank. Uma futura correção de runtime deve medir comportamento end-to-end perto das fronteiras de roteamento, não apenas equivalência algébrica local.

Nenhum novo treinamento de baselines foi iniciado e nenhuma busca de hiperparâmetros foi feita sobre os artigos FCC-1. Estes dados agora são revelados; o próximo protocolo confirmatório deverá usar outros artigos. Não há trabalho prometido em segundo plano.

## Rastreabilidade

- FA-1: `docs/prereg/FUNCTIONAL_ABLATION_1.md`, `results/functional-ablation-1/`, execução científica33919779426, auditoria33920315614.
- FCC-1: `docs/prereg/FRESH_COMPRESSION_CHECK_1.md`, `data/fresh-compression-check-1/`, `results/fresh-compression-check-1/`, execução33921582740; fonte congelada5f9a18b8615def758528930b2d2cbb0b67a7154b.
- ECK-1: `docs/prereg/EXACT_COMPACT_KERNEL_1.md`, `results/exact-compact-kernel-1/attempt-1/`, execução33920705855; diagnóstico em `failure-diagnostic.json`.
- Fontes primárias relacionadas: https://arxiv.org/abs/2502.17298 ; https://arxiv.org/abs/2503.01359 .
