# MUI-1 — revisão científica e decisão de continuidade

Data: 2026-09-04. Execução concluída: GitHub Actions 33917735296.
Fonte científica congelada: 0733c844c6a1f6d31e381563b29d170ce89b7e1d.

## Conclusão

Houve avanço diagnóstico, não demonstração de uma arquitetura superior. A intervenção recuperou utilização dos modos e diversidade nas sondas sintéticas, mas NÃO satisfez o critério prospectivo de melhorar a versão compacta anterior. O veredito permanece `NO_PROMISING_CANDIDATE_UNDER_FROZEN_SCREEN`; `NO_GO_FOR_OLMOE_OR_QWEN` não muda.

Não declarar que a hipótese geral morreu, que a inicialização foi demonstrada como causa única, nem que diversidade é sempre prejudicial. Também não promover um resultado secundário isolado a confirmação.

## O que foi executado

Seis braços × quatro seeds novas = 24 treinamentos completos de 800 atualizações, cada um com 409600 tokens apresentados. Mesmos lotes dentro de cada seed, mesmos parâmetros fora do MoE na inicialização, mesmo otimizador e número de atualizações. Os quatro braços compactos têm exatamente os mesmos orçamentos de parâmetros e operações matriciais analíticas.

Usamos somente treino e calibração já disponível: 150 janelas de 64 tokens na avaliação final. Isso não significa 150 artigos independentes, pois o corpus pode dividir artigos em trechos. Os intervalos reportados são sobre seeds, condicionais a essas janelas. Nenhum arquivo de test/OOD foi carregado pelo experimento. O modelo completo tem apenas 175424 parâmetros; este é um laboratório controlado, não um benchmark de capacidade de um LLM comercial.

Pré-registro: `docs/prereg/MODE_UTILIZATION_INTERVENTION_1.md`. Código: `scripts/run_mode_utilization_intervention_1.py`. Dados brutos, checkpoints e resumos: `results/mode-utilization-intervention-1/`.

## Leitura dos resultados

| Braço | Loss final (menor é melhor) | Stable rank residual final | Parâmetros totais |
|---|---:|---:|---:|
| Completo convencional | 5.100159 | — | 175424 |
| Convencional estreito | 5.181494 | — | 124736 |
| Compacto anterior | 5.108268 | 1.007 | 95552 |
| Espectro plano, amplitude pequena | 5.105133 | 1.008 | 95552 |
| Energia redistribuída, fatores gaussianos | 5.144572 | 3.653 | 95552 |
| Energia redistribuída, espectro plano | 5.153024 | 6.670 | 95552 |

### 1. Utilizar o rank nominal não bastou

O braço espectral com mais energia sustentou aproximadamente 6.67 modos efetivos de 8, contra 1.01 no anterior. Mesmo assim, sua loss média ficou +0.044755 nat acima do anterior. IC95 entre seeds: [-0.097333, +0.186844]. O gaussiano ficou +0.036304, IC95 [-0.032539, +0.105146]. Ambos pioraram em três das quatro seeds.

Portanto, a interpretação correta é **não houve benefício consistente demonstrado**, e não 'foi demonstrada piora estatisticamente inequívoca'. Os intervalos são largos. O critério pré-fixado de melhora de pelo menos 0.010 nat e nenhuma seed pior falhou.

O braço de espectro plano com amplitude pequena voltou a rank aproximadamente 1.008 e obteve apenas -0.003136 nat vs anterior, IC95 [-0.008028, +0.001756]. Apenas alterar a forma inicial do espectro sem redistribuir energia não preservou o rank no regime testado.

### 2. Diversidade inicial não é especialização aprendida

A diversidade das saídas em sondas gaussianas do braço energy-spectral foi aproximadamente 0.798 na inicialização e 0.785 após treinamento. Esses especialistas eram diferentes antes de aprender. Preservar essa diferença não prova aprendizagem de competências complementares. Os braços compactos sem energia redistribuída terminam com diversidade aproximadamente 0.012 e perdas médias melhores neste orçamento curto.

Isso justifica trocar a próxima pergunta: não 'como manter muitos modos?', mas 'quanto da qualidade depende causalmente dos resíduos e da escolha de especialistas?'. Ainda não executamos essa ablação sobre os checkpoints.

### 3. Há um sinal secundário positivo, que não deve ser escondido

Energy-gaussian venceu narrow65 nas quatro seeds, por -0.036921 nat em média, IC95 não ajustado [-0.073568, -0.000275]. Usa 95552 contra 124736 parâmetros totais: cerca de 23.4% menos. Sua vantagem analítica de MACs dos especialistas contra narrow65 é apenas 4.76%, não os 37.5% que aparecem quando o denominador é o modelo completo.

É um sinal de eficiência em um regime curto de desenvolvimento. Não comprova uma melhoria sobre o compacto anterior, que teve loss média menor, nem generalização fora da calibração conhecida. Há múltiplos contrastes exploratórios; o intervalo quase toca zero. Não houve teste contra implementações de métodos publicados nem uma confirmação nova. O FAIL do Gate 2A anterior continua válido; datasets de avaliação, seeds e orçamento não são idênticos, portanto não se deve interpretar a diferença entre estudos como uma curva de aprendizagem controlada.

### 4. Economia analítica e velocidade real divergem

No CPU medido, dois threads, batch 1, sequência de um token, o compacto anterior levou 1.548 ms contra 0.861 ms do estreito: aproximadamente 1.80× mais lento. Energy-gaussian levou 1.559 ms. Em comprimento 64, foram 3.112, 2.729 e 3.224 ms, respectivamente.

São forwards completos, sem KV cache; não representam serving autoregressivo. Os kernels convencionais e compactos têm eficiências diferentes, de modo que não é correto generalizar a diferença para qualquer implementação, GPU ou escala. Tampouco é correto anunciar aceleração baseada apenas em MACs. O treinamento compacto foi mais rápido no runner, mas isso é outra métrica e também depende da implementação.

## Auditoria e críticas aceitas

O runner passou testes de equivalência algébrica direta vs pesos reconstruídos na inicialização, normas, espectro, contagem e determinismo. A agregação verificou janelas pareadas, arquivos, checkpoints e médias por uma segunda via aritmética. Depois da execução, `scripts/audit_mui1_checkpoints.py` foi executado em outro ambiente: verificou os 13 arquivos do manifesto e os 24 checkpoints, recontou parâmetros sem duplicar embeddings compartilhados e recalculou stable rank por autovalores de Gram em vez de SVD. Resultado: PASS. Registro: `checkpoint-audit.json`.

Essa é uma auditoria numérica separada, não uma replicação independente do treinamento nem avaliação por outro pesquisador. Não foi executada uma nova suíte integral de testes do repositório; os testes foram os específicos da intervenção. A auditoria não remove limitações de escala, maturidade ou amostragem.

A intervenção de energia também altera a fração compartilhada das matrizes. A intervenção espectral também altera o balanceamento dos fatores. Não é um experimento de 'ortogonalidade pura' nem identifica uma causa única. Teoria de gradient flow com inicialização infinitesimal não prova comportamento de AdamW em MoE não linear.

## Diferenciação frente à literatura

Uma base compartilhada com resíduos particulares já aparece em D²-MoE e DeRS. Ortogonalização/fatoração, isoladamente, não são uma contribuição nova. Um eventual diferencial nosso precisa ser uma melhoria demonstrada sob um orçamento e hardware explícitos, contra controles fortes, ou um mecanismo novo validado — não renomear esses componentes.

Fontes primárias verificadas em 2026-09-04:
- Gu et al., Delta Decompression for MoE-based LLMs Compression, 2025: https://arxiv.org/abs/2502.17298 . Base compartilhada e deltas comprimidos por SVD; não reproduzido nesta rodada.
- Huang et al., DeRS: Towards Extremely Efficient Upcycled Mixture-of-Experts Models, CVPR 2025: https://arxiv.org/abs/2503.01359 . Base compartilhada e deltas leves; não reproduzido nesta rodada.
- Li, Luo e Lyu, Greedy Low-Rank Learning, ICLR 2021: https://arxiv.org/abs/2012.09839 . Motivação teórica sobre fatoração e inicialização, com condições diferentes deste experimento.

## Próxima decisão operacional — definida, ainda não executada

Despriorizar novos ajustes de inicialização e não aumentar rank nem implementar roteamento dinâmico para 'consertar' apenas o diagnóstico espectral. Antes de outro treinamento, pré-registrar uma ablação funcional barata nos checkpoints existentes: original, retirada de resíduos, substituição por transformação média e perturbação controlada da escolha dos especialistas. Comparar loss, mudança de logits e latência, com controles negativos convencionais, sem ajuste aos dados observados. Tratar essa ablação como diagnóstico pós-hoc, nunca como novo teste confirmatório.

Se retirar/mesclar especialistas mantiver qualidade, testar uma arquitetura densa simples com orçamento pareado passa a ser obrigatório. Se a escolha dos especialistas fizer diferença material, há justificativa para investigar representações compartilhadas que preservem essa especialização. Somente depois disso um protocolo mais longo, novos dados de teste, comparação com método publicado e runtime na plataforma-alvo podem sustentar uma alegação de ganho diferenciado.

Não iniciou outro treinamento automaticamente. Esta rodada está concluída, incluindo seu resultado negativo.
