# Replicação fresca da região de largura teacher-informed

**Data:** 28 de julho de 2026  
**Protocolo:** `pre-qwen-teacher-informed-width-fresh-replication-v1.1`  
**Decisão:** **TEACHER_WIDTH_65_REPLICATION_PASS**  
**Decisão global preservada:** **NO_GO_FOR_OLMOE_OR_QWEN**

## 1. Pergunta

Em teachers MoE convencionais novos, documentos novos e cinco seeds novas, um student convencional estreito, inicializado pelas coordenadas SwiGLU de maior magnitude do teacher e depois totalmente treinável, entra de forma reproduzível na região de fidelidade pré-definida quando conserva 65% da largura dos especialistas?

O teste não avalia Modal-MoE, bases compartilhadas, kernels GPU ou checkpoints reais. Ele mede a fronteira de capacidade de um baseline convencional forte que qualquer arquitetura compacta posterior precisará superar.

## 2. Desenho congelado

- Teachers: cinco seeds novas: `91121, 92129, 93133, 94151, 95153`.
- Geometria: 2 camadas, `d_model=24`, `d_ff=40`, 12 especialistas, top-4.
- Treino do teacher: 2.200 passos.
- Dados: 28 documentos de treino, 20 documentos hypothesis e 12 documentos OOD.
- Separação por documento; auditoria de duplicatas exatas e near-duplicates.
- Unidade estatística: célula seed × documento.
- Bootstrap cruzado seed/documento: 20.000 amostras.
- Router congelado e mesmos batches, budgets e objetivos para todos os candidatos.
- Os candidatos foram congelados antes da materialização dos documentos hypothesis/OOD.
- Auditor independente sem importar o agregador ou o bootstrap do experimento.

Candidatos:

```text
35%  anchor negativo
50%  comparador de menor compute
65%  candidato primário
75%  controle de capacidade
100% controle full-width do recovery loop
```

As razões de parâmetros e de operações matriciais roteadas são exatas neste modelo: a largura intermediária é o único componente alterado.

## 3. Correção de engenharia antes da avaliação

A primeira tentativa parou antes de produzir qualquer métrica hypothesis/OOD porque os documentos OOD continham caracteres ausentes do vocabulário construído com o treino. A revisão `v1.1`:

1. adicionou uma linha de cobertura de caracteres exclusivamente no split de treino;
2. adicionou um teste que prova que o vocabulário congelado cobre hypothesis e OOD;
3. não alterou candidates, widths, gates, seeds ou documentos held-out;
4. reiniciou todos os teachers e students do zero.

Como nenhuma seed produziu JSON de avaliação antes da correção, o resultado final não incorpora seleção baseada em métricas held-out.

## 4. Resultado principal

| Candidato | Parâmetros | Compute matricial | Δ loss hypothesis | UCB95 | Δ loss OOD | UCB95 | Pior seed | KL hyp | Top-1 hyp |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 35% | 35% | 35% | +0.08626 | +0.10517 | -0.02408 | +0.03363 | +0.10496 | 0.09460 | 88.694% |
| 50% | 50% | 50% | +0.02973 | +0.03984 | -0.01345 | +0.03085 | +0.04134 | 0.05044 | 92.042% |
| **65%** | **65%** | **65%** | **+0.01325** | **+0.01914** | **-0.01157** | **+0.01021** | **+0.01608** | **0.02979** | **94.028%** |
| 75% | 75% | 75% | +0.00412 | +0.01144 | -0.00942 | +0.01467 | +0.01217 | 0.02000 | 95.083% |
| Full control | 100% | 100% | -0.00431 | +0.00207 | -0.00208 | +0.00860 | +0.00569 | 0.00866 | 96.681% |

Para o candidato primário de 65%:

- penalidade média hypothesis: `+0.01325 nat`;
- intervalo crossed 95%: `[+0.00657, +0.01914]`;
- razão média de perplexidade: `1.01334×`;
- razão de perplexidade no UCB95: `1.01932×`;
- pior seed: `+0.01608 nat`;
- parâmetros dos experts: `65%` do teacher;
- operações matriciais dominantes dos experts: `65%` do teacher.

Assim, o screen estabelece uma redução exata de 35% nos parâmetros dos experts e no proxy de operações matriciais roteadas, com aproximadamente 1,33% de aumento médio de perplexidade no conjunto hypothesis.

## 5. Comparação primária 65% versus 50%

```text
65% − 50%:
  média: -0.01648 nat
  IC95:  [-0.02611, -0.00570]
```

O intervalo é inteiramente negativo. Logo, sob este protocolo, 65% é superior a 50% de forma estabelecida pelo bootstrap crossed seed/documento.

O candidato de 50% teve média `+0.02973 nat`, numericamente próxima da margem de `+0,030`, mas seu UCB95 foi `+0.03984`. Portanto, 50% **não** passa o gate confirmatório.

O anchor de 35% reproduziu a insuficiência de capacidade: LCB95 `+0.06891`, acima do piso pré-definido de `+0,030`.

## 6. Robustez

Todos os cinco resultados individuais do candidato de 65% ficaram abaixo de `+0,060 nat`:

| Seed | Δ loss hypothesis |
|---:|---:|
| 91121 | +0.01514 |
| 92129 | +0.01576 |
| 93133 | +0.00580 |
| 94151 | +0.01347 |
| 95153 | +0.01608 |

A análise leave-one-seed-out preservou o gate em todos os casos:

| Seed omitida | Média | UCB95 |
|---:|---:|---:|
| 91121 | +0.01278 | +0.02018 |
| 92129 | +0.01262 | +0.01911 |
| 93133 | +0.01511 | +0.02102 |
| 94151 | +0.01320 | +0.02013 |
| 95153 | +0.01254 | +0.01877 |

O auditor independente encontrou `0` divergências. Também confirmou:

- hashes únicos dos cinco checkpoints;
- um único hash de configuração: `98051e0a6a73f43934627262757ffaee1a359c7e407438d89b24148ef0de64b1`;
- source commit comum: `e7092acb6d00e8763fa684ed695e751a45d5f6f9`;
- recomputação das células seed/documento;
- recomputação dos intervalos crossed;
- recomputação da comparação 65%−50%;
- razões exatas de parâmetros/compute;
- ausência de overlap proibido entre splits;
- decisão final idêntica.

## 7. Leitura adversarial

### 7.1 Teachers sem convergência demonstrada

A mudança de language loss entre os dois últimos checkpoints registrados foi:

```text
{
  "91121": -0.4030330777168274,
  "92129": -0.3381621837615967,
  "93133": -0.010348796844482422,
  "94151": -0.23523610830307007,
  "95153": -0.16233861446380615
}
```

Quatro das cinco seeds ainda apresentavam queda material. Portanto, não se pode afirmar que a região de 65% persista para teachers convergidos ou mais especializados. Um teacher mais maduro pode deslocar a fronteira para larguras maiores.

### 7.2 OOD cross-entropy não implica paridade comportamental

Embora o Δ loss OOD do candidato de 65% seja negativo em média, sua KL OOD é `0.13431` e o top-1 agreement OOD é `81.111%`. Portanto, “loss OOD não pior” não significa que o student reproduziu a distribuição do teacher. O OOD gate é apenas um controle de CE, não uma prova de equivalência comportamental.

### 7.3 Sem claim de runtime

A razão de 65% é um proxy exato para parâmetros dos experts e multiplicações matriciais dominantes nesta geometria. Ela não inclui:

- eficiência de kernel;
- routing, sorting e scatter;
- memória e cache;
- ocupação de hardware;
- latência end-to-end.

Nenhuma aceleração real foi medida.

### 7.4 Sem claim Modal

O candidato vencedor é um MoE convencional estreito, inicializado com coordenadas do teacher e depois totalmente treinável. Não há compartilhamento Modal, basis bank ou side factor. O resultado estabelece um baseline forte, não valida a teoria Modal.

### 7.5 Escopo pequeno e sintético

O resultado é restrito a:

- character LM sintético;
- uma geometria pequena;
- duas camadas;
- top-4;
- cinco teachers da mesma família;
- um único protocolo de treino.

Não deve ser extrapolado para OLMoE, Qwen ou linguagem natural ampla.

## 8. Conclusão científica

A conclusão estreita suportada é:

> Em cinco teachers MoE pequenos e novos, um student convencional teacher-informed com 65% da largura preservou a qualidade dentro dos gates pré-registrados e superou o student de 50%, sob os dados, budgets e estatística definidos no protocolo v1.1.

O resultado mostra uma transição de capacidade entre 50% e 65% nesta família. Ele também fornece um baseline Pareto mais forte: qualquer arquitetura de compartilhamento futuro precisa, no mínimo, igualar a qualidade do narrow 65% usando no máximo 65% dos parâmetros e do compute projetado.

O resultado **não** muda:

```text
NO_GO_FOR_OLMOE_OR_QWEN
```

## 9. Próximo gate recomendado

O próximo experimento deve testar se a região de 65% sobrevive simultaneamente a:

1. largura e profundidade maiores;
2. teachers treinados até um critério explícito de plateau;
3. corpus multi-domínio não sintético;
4. múltiplas camadas substituídas;
5. comparação com uma arquitetura alignment-tolerant/shared-basis no mesmo orçamento de 65%;
6. benchmark de runtime contra narrow convencional otimizado.

A hipótese forte a falsificar passa a ser:

> Uma arquitetura de compartilhamento útil deve igualar ou superar o narrow teacher-informed de 65% em qualidade, com menos de 65% dos parâmetros/compute e sem perder robustez por seed ou domínio.
