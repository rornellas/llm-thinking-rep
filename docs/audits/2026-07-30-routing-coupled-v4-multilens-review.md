# Revisão factual adversarial multilentes — routing-coupled residual v4

**Data:** 30 de julho de 2026  
**Protocolo:** `pre-qwen-routing-coupled-residual-v4`  
**Veredito automático preservado:** `ROUTING_COUPLED_V4_FAIL`  
**Disposição científica adversarial:** `ROUTING_COUPLED_V4_CAUSAL_BUT_INSUFFICIENT__MOMENT_COUPLING_CLOSED`  
**Decisão global:** `NO_GO_FOR_OLMOE_OR_QWEN`

## 1. Veredito executivo

A v4 não atingiu a meta científica. A candidata primária, com 44,28% dos parâmetros dos experts e 62,5% do proxy matricial, melhorou significativamente KL, top-1 e erro local contra o rank-5 v3, e o acoplador produziu efeito causal estatisticamente detectável em KL. Entretanto:

- o efeito causal ficou abaixo da magnitude mínima pré-registrada;
- KL, top-1, fidelidade local, counterfactual e erro agregado continuam muito distantes do narrow65;
- o segundo momento não demonstrou vantagem sobre o controle mean-only;
- aumentar o espaço de acoplamento de `q=8` para `q=12` não resgatou a arquitetura;
- o erro agregado da mistura permaneceu aproximadamente 0,117 acima do narrow65;
- o controle full-width falhou por pequena margem no gate OOD.

A conclusão correta não é “o acoplamento não funciona”. É mais estreita:

> Uma pequena correção por primeiros e segundos momentos dos latentes roteados move o comportamento causalmente na direção correta, mas não oferece capacidade suficiente para fechar a lacuna estrutural. Esta forma de acoplamento por momentos está encerrada.

## 2. Integridade factual

A evidência é utilizável:

- cinco seeds;
- vinte documentos hypothesis e doze OOD novos;
- três janelas por documento;
- candidatos congelados antes de materializar os holdouts;
- configuração e commit únicos;
- checkpoints com hashes únicos;
- source checkpoints v3 verificados;
- auditoria de dados aprovada;
- bootstrap cruzado por seed e documento;
- leave-one-seed-out;
- auditor independente com zero divergências;
- checkpoints, registros por janela, logs, ambiente e hashes versionados.

A auditoria independente retornou `PASS` e reconstruiu todos os gates load-bearing.

## 3. Resultado primário

| Métrica | q8/h8 v4 | rank-5 v3 | rank-6 v3 | narrow65 |
|---|---:|---:|---:|---:|
| Parâmetros dos experts | 44,28% | 41,67% | 48,33% | 65,00% |
| Proxy de compute | 62,50% | 58,33% | 65,00% | 65,00% |
| Hypothesis delta loss | -0,07459 | -0,07769 | -0,08454 | -0,00862 |
| KL hypothesis | 0,38644 | 0,42108 | 0,37608 | 0,11655 |
| Top-1 | 67,65% | 66,12% | 68,47% | 82,69% |
| Local NRMSE | 0,36517 | 0,37057 | 0,34384 | 0,17041 |
| Counterfactual NRMSE | 0,46091 | 0,46145 | 0,42748 | 0,35036 |
| Routing aggregate error | 0,15158 | 0,15570 | 0,13524 | 0,03434 |
| Energia da correção | 0,00276 | 0 | 0 | 0 |

A candidata obteve cross-entropy significativamente menor que o narrow65:

```text
v4 - narrow65 loss
mean  -0.065975 nat
95%   [-0.125942, -0.013520]
```

Isso não é equivalência ao teacher. A divergência funcional permanece grande. Em um corpus pequeno, o student pode melhorar CE por regularização ou por se afastar do teacher em direções que ajudam esses dados.

## 4. Melhoras estabelecidas contra a v3

Três de cinco endpoints comportamentais melhoraram de forma estatisticamente estabelecida:

```text
v4 - v3 KL
mean  -0.034643
95%   [-0.049048, -0.021484]

v4 - v3 top-1
mean  +0.015278
95%   [+0.005139, +0.024792]

v4 - v3 local NRMSE
mean  -0.005396
95%   [-0.007168, -0.003669]
```

Não houve melhora estabelecida em loss ou counterfactual NRMSE:

```text
v4 - v3 loss
mean  +0.003093
95%   [-0.028750, +0.031750]

v4 - v3 counterfactual NRMSE
mean  -0.000537
95%   [-0.001644, +0.000594]
```

Portanto, a arquitetura alterou comportamento, mas não melhorou a fronteira global o suficiente.

## 5. Causalidade do acoplador

A ablação pós-treino desligou somente a correção de conjunto, mantendo todos os demais pesos.

```text
disabled - primary KL
mean  +0.012102
95%   [+0.007258, +0.017261]
```

O efeito é positivo em todas as cinco seeds e seu intervalo está completamente acima de zero. Há evidência de causalidade estatística.

O gate pré-registrado, porém, exigia:

```text
LCB95 >= +0.010
```

O observado foi:

```text
LCB95 = +0.007258
```

Logo, a formulação correta é:

> O acoplador tem efeito causal detectável, mas não atinge a magnitude mínima exigida para o mecanismo ser considerado suficiente.

A ablação de loss não demonstrou benefício:

```text
disabled - primary loss
mean  -0.004927
95%   [-0.015821, +0.006762]
```

## 6. O segundo momento não foi validado

O controle mean-only tinha exatamente o mesmo orçamento e mesma inicialização funcional.

```text
primary - mean-only KL
mean  -0.001162
95%   [-0.005049, +0.002677]

primary - mean-only loss
mean  -0.000741
95%   [-0.005731, +0.004154]
```

Não há evidência de que o segundo momento seja melhor que apenas a média ponderada. Sua contribuição não deve ser elevada a claim.

## 7. Mais capacidade de acoplamento não resgatou a hipótese

O controle q12/h8 elevou parâmetros para 45,31% e compute para 63,75%, mas obteve:

```text
KL          0.38962
Top-1       67.67%
Local       0.36502
CF          0.46084
Aggregate   0.15145
```

Esses valores são essencialmente equivalentes ao q8/h8. A limitação não parece ser apenas dimensão insuficiente do pooling.

## 8. Lacuna para o narrow65

O erro agregado, que é diretamente identificável para a mistura final, permaneceu muito maior:

```text
v4 - narrow65 routing aggregate error
mean  +0.117240
95%   [+0.104094, +0.132075]
```

Os gates absolutos também falharam por margens grandes:

| Gate | Observado | Exigido |
|---|---:|---:|
| KL UCB95 | 0,42738 | <= 0,20 |
| Top-1 LCB95 | 65,48% | >= 78% |
| Local NRMSE UCB95 | 0,38793 | <= 0,24 |
| Counterfactual NRMSE UCB95 | 0,48373 | <= 0,28 |

O acoplador não aproximou a arquitetura da qualidade funcional do narrow65 em magnitude suficiente.

## 9. Robustez de CE

Todos os deltas de loss por seed foram favoráveis:

```text
91121  -0.16070
92129  -0.07936
93133  -0.04304
94151  -0.02556
95153  -0.06431
```

Todos os intervalos leave-one-seed-out preservaram UCB abaixo de zero. Assim, o ganho de CE contra o próprio teacher é robusto neste conjunto. Ele deve ser descrito como ganho de CE no screen, não como preservação geral ou superioridade de modelo.

## 10. Controle full-width

O full control passou hypothesis:

```text
mean  +0.00117
UCB   +0.00982
limite +0.010
```

No OOD:

```text
mean  +0.00353
UCB   +0.01726
limite +0.015
```

A falha é pequena, mas real sob o protocolo. Ela bloqueia qualquer `PASS` automático. Como o full control é um estado herdado e não a identidade exata do teacher, isso sinaliza sensibilidade do holdout/continuação; não invalida as comparações pareadas entre candidatos, mas limita claims absolutos.

## 11. Lente de otimização

Na seed 91121, as losses locais oscilaram e não mostraram convergência monotônica. O estágio joint reduziu KL em alguns checkpoints, mas voltou a piorar até o final. Há espaço para otimização melhor.

Entretanto, não aprovamos outra rodada de ajuste de passos/pesos porque:

- q12 não apresentou ganho de capacidade;
- mean-only é indistinguível;
- o efeito causal é pequeno;
- quatro gates funcionais continuam distantes;
- a correção usa somente cerca de 0,28% da energia da mistura do teacher;
- a lacuna de erro agregado para narrow65 é muito maior que o efeito do acoplador.

O problema principal continua sendo alocação de capacidade e classe representacional, não apenas otimização.

## 12. Decisão multilentes

| Questão | Decisão |
|---|---|
| Aceitar o veredito automático `FAIL` | Sim |
| Existe efeito causal do acoplador | Sim, abaixo do mínimo de magnitude |
| O segundo momento foi validado | Não |
| q12 resgatou capacidade | Não |
| A arquitetura superou narrow65 em CE | Sim, somente neste screen |
| A arquitetura preservou comportamento do teacher | Não |
| Repetir o mesmo acoplador com novos pesos/passos | Não aprovado |
| Encerrar todo acoplamento de conjunto | Não |
| Encerrar acoplamento por momentos low-rank | Sim |
| Alterar `NO_GO_FOR_OLMOE_OR_QWEN` | Não |

## 13. Próxima hipótese aprovada

A próxima experiência deve realocar capacidade, não adicionar outra pequena correção residual.

Será testado um student com **prefixos low-rank aninhados e orçamento por token**, no qual:

- cada expert pode armazenar rank máximo maior;
- o total de rank ativo dos experts roteados é limitado por token;
- os ranks são alocados conforme pesos do router e utilidade marginal dos modos;
- o compute projetado permanece igual ao rank uniforme de referência;
- os prefixos são treinados sob múltiplos orçamentos para se tornarem funcionalmente úteis.

Candidatos primários propostos:

```text
max-rank 7, budget sum 20  -> params 55%, compute 58.33%
max-rank 8, budget sum 24  -> params 61.67%, compute 65%
```

Eles serão comparados com rank-5 uniforme, rank-6 uniforme, narrow65 e full. Se a alocação dinâmica não mover a fronteira, a linha pós-hoc será despriorizada e treinamento nativo desde o início se tornará o caminho principal.
