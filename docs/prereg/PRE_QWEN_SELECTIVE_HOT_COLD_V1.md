# Pré-registro — compressão seletiva hot/cold de especialistas

**Protocolo:** `pre-qwen-selective-hot-cold-screen-v1`
**Status:** desenvolvimento exploratório pré-registrado
**Decisão congelada que não pode ser alterada por este screen:** `NO_GO_FOR_OLMOE_OR_QWEN`

## Pergunta

Preservar integralmente especialistas de maior contribuição, escolhidos apenas em ativações de treino, e comprimir os demais em um banco Modal compartilhado melhora a fronteira conjunta de parâmetros, cálculo esperado e qualidade em comparação com compressão uniforme?

## Hipótese matemática

O teacher é

\[
f(x)=\sum_{e\in R(x)} \pi_e(x) f_e(x).
\]

Escolhe-se um conjunto `H` de especialistas hot usando apenas capturas de treino. O estudante executa

\[
\hat f(x)=
\sum_{e\in H\cap R(x)} \pi_e(x) f_e(x)
+
\sum_{e\in R(x)\setminus H}\pi_e(x)\hat f_e^{cold}(x),
\]

onde os especialistas hot são cópias exatas e congeladas do teacher, e os cold compartilham `K+1` matrizes Modal em `gate`, `up` e `down`.

O score primário é

\[
I_e=\mathbb E_{x\sim train}\left[\|\pi_e(x)f_e(x)\|_2^2\right].
\]

Ele equivale ao dano quadrático local de remover isoladamente a contribuição roteada do especialista. Não incorpora termos cruzados entre especialistas nem curvatura da perda final; por isso routing frequency, gate mass e seleção aleatória são controles obrigatórios.

## Candidato primário

`selective-energy-h3-k0`:

- três especialistas hot exatos;
- nove especialistas cold representados por um modo comum (`K=0`);
- router congelado;
- 540 passos de distilação local agregada;
- 180 passos de recuperação closed-loop;
- nenhuma escolha baseada em hipótese ou OOD.

## Controles

- mesmos `H=3, K=0`, mas hot escolhidos por frequência;
- mesmos `H=3, K=0`, mas hot escolhidos por massa do gate;
- mesmos `H=3, K=0`, mas hot aleatórios com seed pré-fixada;
- `H=2, K=1` e `H=4, K=0` como ablações de alocação;
- narrow convencional com parâmetros arredondados para cima;
- narrow convencional com compute esperado arredondado para cima, calculado somente nas rotas de treino.

## Dados e independência

- seleção e treinamento: corpus de treino já usado pelos teachers;
- hipótese: novos documentos determinísticos com seed `73031`;
- OOD: corpus hand-authored v2, distinto do OOD v1 consumido;
- três teachers: `62131`, `63137`, `64151`;
- duas janelas por documento, seed `11939`;
- unidade inferencial: seed de teacher × documento; janelas são agregadas dentro da célula.

Este conjunto é de desenvolvimento, não selado. Nenhum resultado pode autorizar diretamente teste em OLMoE ou Qwen.

## Custo reportado

Parâmetros de especialistas:

\[
P=3Hd f+3(K+1)df+3(E-H)K.
\]

Cálculo matricial dominante por token, condicionado às rotas:

\[
C(x)=\frac{h(x)+(K+1)\mathbf 1[c(x)>0]}{T},
\]

onde `h(x)` é o número de slots hot e `c(x)` o número de slots cold. O proxy ajustado acrescenta o custo de aplicar códigos aos slots cold. Não é uma medição de latência.

## Gates pré-definidos

O candidato primário só fica elegível para uma nova replicação selada se todos forem verdadeiros:

1. UCB95 da penalidade contra o teacher na hipótese `<= +0,03 nat`;
2. UCB95 na OOD `<= +0,05 nat`;
3. UCB95 contra narrow parameter-matched `<= 0`;
4. UCB95 contra narrow compute-matched `<= +0,005 nat`;
5. UCB95 contra seleção aleatória `<= 0`;
6. compute ajustado nas rotas de treino `<= 75%`;
7. nenhuma seed com penalidade média na hipótese superior a `+0,06 nat`.

Os thresholds, candidatos e seeds não podem ser alterados depois da primeira execução científica.

## Procedimento adversarial obrigatório

A conclusão será recalculada por script independente que não importa o agregador principal. O auditor deve verificar:

- identidade dos hot experts a partir dos scores brutos;
- ausência de dados de avaliação na seleção;
- contagem fechada de parâmetros;
- custo condicionado às rotas;
- bootstrap cruzado seed/documento;
- médias por seed e leave-one-seed-out;
- comparação com controle aleatório e ambos os narrows;
- interpretação alternativa de qualquer aparente ganho.
