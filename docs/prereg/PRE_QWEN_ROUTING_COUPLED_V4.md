# Pré-registro — routing-coupled residual v4

**Protocolo:** `pre-qwen-routing-coupled-residual-v4`  
**Estado:** congelado antes de materializar os holdouts v4  
**Decisão global invariável:** `NO_GO_FOR_OLMOE_OR_QWEN`

## Pergunta

Uma correção pequena, compartilhada e permutation-invariant, condicionada ao conjunto de experts roteados, consegue corrigir a lacuna de coordenação observada nas v1–v3 e superar o narrow65 com menos de 65% dos parâmetros e do proxy matricial dos experts?

## Hipótese arquitetural

O rank-5 alignment-tolerant produz outputs base `f_e(x)` e latentes low-rank `z_e` já necessários ao `down`. A v4 acrescenta:

```text
u_e = A_e z_e + b_e + log(pi_e) r
mu  = sum_e pi_e u_e
var = sum_e pi_e u_e^2 - mu^2
c   = W2 SiLU(W1 [mu; var])
y   = sum_e pi_e f_e(x) + c
```

- `A_e` alinha as coordenadas low-rank específicas em um espaço comum;
- `mu` e `var` tornam a correção invariável à ordem dos slots;
- `c` representa explicitamente interação do conjunto roteado;
- `W2` é inicializado em zero, logo a função inicial é exatamente a base rank-5;
- o mesmo mecanismo aceita pesos naturais ou contrafactuais.

## Orçamento primário

Na geometria `d=24`, `d_ff=40`, `E=12`, `top-k=4`, rank-5, `q=8`, `hidden=8`:

```text
teacher expert parameters:       34,560
rank-5 base parameters:          14,400
coupling parameters:                904
primary total:                  15,304  = 44.2824%

teacher dominant matrix MACs:    11,520
rank-5 base matrix MACs:          6,720
coupling matrix MACs:               480
primary total:                   7,200  = 62.5%
```

Somente multiplicações matriciais dominantes entram no proxy; não há claim de latência.

## Candidatos

1. `rank5-coupled-q8-h8-v4` — primário, primeiro e segundo momentos;
2. `rank5-coupled-q8-h8-mean-only-control` — orçamento idêntico, segundo momento zerado;
3. `rank5-coupled-q12-h8-v4` — controle de capacidade, 63,75% do compute;
4. rank-5 v3 congelado;
5. rank-6 v3 congelado, 65% do compute;
6. narrow65 congelado;
7. full continuation congelado.

Os três candidatos coupled partem do mesmo rank-5 v3 por seed. Os dois q8 partem do mesmo estado bit a bit e recebem os mesmos minibatches.

## Dados

- treino/captura: somente `teacher-width-train-v1`;
- hypothesis: `routing-coupled-hypothesis-v4-confirmation`, seed `611953`;
- OOD: `routing-coupled-ood-v4-confirmation`;
- vinte documentos hypothesis, doze OOD, três janelas por documento;
- deduplicação exata e near-duplicate por shingle;
- holdouts materializados somente depois do congelamento dos candidatos.

Nenhum documento v1–v3 será utilizado para seleção ou decisão v4.

## Treinamento

### Fase A — somente acoplador

- base rank-5 congelada;
- 480 passos;
- mesma loss local da v3 para isolar a mudança de arquitetura;
- mesmos batches para candidatos comparáveis.

### Fase B — joint closed-loop

- 180 passos;
- todos os parâmetros do student, exceto router, liberados;
- pesos: local `0.35`, KL `0.55`, CE `0.10`.

## Endpoints primários

- cross-entropy/perplexidade closed-loop;
- KL teacher→student;
- top-1 agreement;
- NRMSE local;
- NRMSE contrafactual;
- erro agregado da mistura roteada contra narrow65;
- ablação com acoplador desligado;
- controle mean-only;
- parâmetros e compute analítico.

O termo `cross-error` permanece diagnóstico. Ele não é gate porque uma correção definida no conjunto não possui decomposição única entre experts; repartir a mesma correção por slot altera artificialmente self/cross sem alterar a saída agregada. O `routing_aggregate_error` é identificável e invariável.

## Estatística

- unidade hierárquica: seed e documento;
- janelas são agregadas dentro da célula seed×documento;
- bootstrap crossed com 20.000 amostras;
- valores por seed e leave-one-seed-out obrigatórios;
- comparação pareada em todas as diferenças load-bearing.

## Gates de `PASS`

O `PASS` exige simultaneamente:

- hypothesis UCB95 `<= +0.030 nat`;
- OOD UCB95 `<= +0.050 nat`;
- cada seed `<= +0.060 nat`;
- não-inferioridade pareada a narrow65, rank-6 e rank-5 v3;
- KL UCB95 `<= 0.20`;
- top-1 LCB95 `>= 0.78`;
- local NRMSE UCB95 `<= 0.24`;
- counterfactual NRMSE UCB95 `<= 0.28`;
- gap de `routing_aggregate_error` para narrow65 UCB95 `<= 0.04`;
- o primário não é inferior ao controle mean-only em KL;
- desligar o acoplador piora KL em pelo menos `0.01` no LCB95 ou piora loss com LCB95 não negativo;
- parâmetros `<65%`, compute `<=65%`;
- full controls, dados, aritmética e auditor independente aprovados.

## Gate de `FUNCTIONAL_SIGNAL`

Um sinal funcional exige integridade, orçamento, full controls, auditoria e causalidade do acoplador, além de **pelo menos dois de cinco** efeitos estabelecidos contra o rank-5 v3 congelado:

1. loss: UCB95 de `v4-v3 <= 0`;
2. KL: UCB95 de `v4-v3 <= 0`;
3. top-1: LCB95 de `v4-v3 >= 0`;
4. local NRMSE: UCB95 de `v4-v3 <= 0`;
5. counterfactual NRMSE: UCB95 de `v4-v3 <= 0`.

Esse rótulo não substitui o `PASS`, não autoriza checkpoint real e não pode ser concedido se desligar o acoplador não causar dano mensurável.

## Interpretação adversarial antecipada

- CE favorável com KL/top-1 ruins não constitui preservação;
- melhora causada apenas por mais parâmetros deve aparecer também no controle mean-only;
- ausência de efeito ao desligar o acoplador invalida o mecanismo;
- q12 pode diagnosticar capacidade, mas não resgata um primário que falhe causalidade;
- teachers não demonstradamente plateaued limitam a conclusão a screen de mecanismo;
- nenhum resultado altera o `NO_GO_FOR_OLMOE_OR_QWEN`.

## Stop rule

Se o acoplador não produzir melhora causal e comportamental sob o orçamento, esta forma de correção por momentos será encerrada. A pesquisa migra para:

1. arquitetura compacta treinada nativamente desde cedo;
2. ranks heterogêneos por expert/camada;
3. baseline inspirado em métodos de compressão MoE existentes;
4. somente depois, checkpoint real e kernel GPU.
