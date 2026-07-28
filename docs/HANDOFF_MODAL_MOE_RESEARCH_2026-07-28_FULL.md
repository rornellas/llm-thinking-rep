# Handoff técnico-científico e operacional
## Representação compacta, Modal-MoE e certificação pré-Qwen

**Data de corte:** 28 de julho de 2026  
**Repositório:** `rornellas/llm-thinking-rep`  
**PR remoto em andamento:** `#1 — research: audited pre-Qwen certification and FAIL-resolution experiments`  
**Decisão científica global congelada:** **`NO_GO_FOR_OLMOE_OR_QWEN`**  
**Resultado positivo mais recente:** **`TEACHER_WIDTH_65_REPLICATION_PASS`**  
**Próximo gate recomendado:** arquitetura **alignment-tolerant / free-basis / shared-basis** que supere o baseline convencional teacher-informed de 65%.

---

# 0. Como usar este handoff

Este documento é a fonte de continuidade para outra sessão, agente ou pesquisador. Ele substitui como visão consolidada o handoff de 27 de julho de 2026 e os resumos intermediários produzidos durante a certificação pré-Qwen.

Ele foi escrito para permitir que a próxima sessão:

1. compreenda a hipótese original e o que restou dela;
2. saiba exatamente o que foi demonstrado, refutado ou ainda não testado;
3. encontre os artefatos primários, commits, hashes, configs e relatórios;
4. não reutilize holdouts consumidos nem relaxe gates retrospectivamente;
5. reconcilie corretamente o Git local com a branch e o PR remoto;
6. execute o próximo experimento sem reconstruir decisões ou repetir linhas encerradas;
7. mantenha o padrão obrigatório de double-check factual e revisão adversarial.

## Regra de leitura

Antes de implementar qualquer novo teste, ler nesta ordem:

1. este handoff;
2. `docs/methodology/IMPORTANT_CLAIM_VERIFICATION_STANDARD.md`;
3. `results/VERDICT.md`;
4. `docs/results/2026-07-28-teacher-informed-width-fresh-replication.md`;
5. `docs/results/2026-07-28-fail-resolution-mathematical-analysis.md`;
6. o pré-registro do próximo experimento, que deve ser criado **antes** de qualquer métrica held-out.

---

# 1. Estado executivo

## 1.1 Situação atual em uma frase

A pesquisa demonstrou que a estrutura Modal é funcional e transplantável em modelos pequenos, mas a forma **escalar global pós-hoc** não venceu um baseline convencional com compute equivalente; em contrapartida, um student convencional teacher-informed mantendo **65% da largura dos especialistas** passou uma replicação fresca de cinco seeds e agora define o baseline mínimo que uma arquitetura compartilhada precisa superar.

## 1.2 Decisões atuais

| Pergunta | Estado |
|---|---|
| O harness metodológico detecta wiring errado, leakage e controles conhecidos? | **PASS no escopo pequeno testado** |
| Scalar Modal pode representar uma função genuinamente Modal conhecida? | **PASS** |
| Scalar Modal K2 pode substituir funcionalmente uma camada MoE convencional pequena? | **SUPPORTED / FUNCTIONAL_ONLY** |
| Scalar Modal K2 vence narrow convencional em compute dominante semelhante? | **REFUTADO nos screens atuais** |
| Aumentar K preserva compute? | **Não; fidelidade aproxima-se somente em K não econômico** |
| Resíduos low-rank específicos resolvem o equal-compute fail? | **Não no desenho testado** |
| Bases clusterizadas por routing resolvem? | **Não no desenho testado** |
| Seleção de coordenadas existentes a 35% resolve? | **Não** |
| Student convencional teacher-informed a 65% preserva qualidade? | **PASS em cinco teachers pequenos novos** |
| Há speedup real medido? | **UNTESTED** |
| Está autorizado testar OLMoE/Qwen? | **NÃO** |

## 1.3 Resultado positivo mais recente

Protocolo:

```text
pre-qwen-teacher-informed-width-fresh-replication-v1.1
```

Resultado primário:

```text
65% conventional narrow student
expert parameters/full:       65.000%
routed matrix compute/full:   65.000%
hypothesis Δ loss:            +0.01325 nat
crossed UCB95:                +0.01914 nat
perplexity ratio mean:        1.01334x
perplexity ratio at UCB95:    1.01932x
worst seed:                   +0.01608 nat
```

Comparação com 50%:

```text
65% - 50%:
mean:  -0.01648 nat
95%:   [-0.02611, -0.00570]
```

Conclusão estreita:

> Em cinco teachers MoE pequenos e novos, o student convencional teacher-informed de 65% passou todos os gates pré-registrados e foi superior ao student de 50%.

Limite:

> Esse resultado não é Modal, não mede runtime e não autoriza extrapolação para modelos reais.

---

# 2. Hierarquia de fontes e confiança

## 2.1 Fonte canônica local mais completa

Artefato:

```text
pre-qwen-teacher-width-fresh-v1.git.bundle
```

SHA-256 verificado:

```text
8f0ca3009e09f2130aad564fc52bb5597e5de12c356fa53d27b0b875545f8036
```

Branch contida:

```text
agent/teacher-informed-width-replication-v1
```

Head:

```text
c8f4841e47b7c7bd9df3ac30e06c395f57c3b1d9
results: record audited teacher-informed width replication
```

O bundle registra histórico completo e foi validado com `git bundle verify`.

## 2.2 Pacote compacto de resultados

Artefato:

```text
pre-qwen-teacher-width-fresh-v1-results.zip
```

SHA-256 verificado:

```text
f3d5d8d8104c429d89118bae59ba5ffc1fd7e93396229d9e8cb925b6eccf42a7
```

## 2.3 Estado remoto no GitHub, verificado na data de corte

PR:

```text
https://github.com/rornellas/llm-thinking-rep/pull/1
```

Metadados:

```text
state:       open
draft:       true
mergeable:   true
base:        main
base SHA:    3de79997fed2e83fef7201d8d74c7fb0d17450df
head branch: agent/pre-qwen-certification-v2
head SHA:    7e868028671c7ecf513f7fbfc1b6e2320506ec76
commits:     56
files:       34
```

O PR remoto **não contém** o relatório:

```text
docs/results/2026-07-28-teacher-informed-width-fresh-replication.md
```

A leitura desse caminho na branch remota retornou `404`. Portanto, o resultado de 65% está preservado no bundle e nos artefatos locais, mas não está integrado ao PR remoto atual.

## 2.4 Import remoto incompleto

A branch do PR contém:

```text
.github/import/native-alignment-v1.b64
.github/workflows/_import-native-alignment.yml
```

O workflow pretendia extrair um bundle de “native alignment screens and convergence protocol”, validar:

```text
tests/test_alignment_tolerant.py
tests/test_modal.py
tests/test_tiny_lm_smoke.py
```

e comitar:

```text
research: add native alignment screens and convergence protocol
```

O arquivo compactado esperado possui SHA-256:

```text
9e57be34c7d67cc5a90684be8eb3798be4de4b350f9675f9cb1b8039cd8bec8c
```

Porém, os arquivos extraídos não aparecem entre os arquivos modificados do PR e o importador continua presente. Tratar isso como **import não concluído**. Antes de reimplementar qualquer arquitetura alignment-tolerant, extrair e inspecionar esse bundle.

## 2.5 Handoff histórico

Arquivo histórico:

```text
HANDOFF_MODAL_LLM_RESEARCH_2026-07-27(1).md
```

É útil para a trajetória anterior — Testes 0.x, 2.x, 4.x, 5.5 e 6.2 — mas sua recomendação de avançar diretamente ao transplante real foi superada pela auditoria pré-Qwen. A decisão atual é `NO_GO`.

---

# 3. Verificação factual realizada para este handoff

## 3.1 Testes automatizados

No checkout reconstruído do bundle:

```bash
PYTHONPATH=. pytest -q
```

Resultado:

```text
17 passed
```

Há um warning não bloqueante em `test_conditional_pruning.py` sobre converter tensor com `requires_grad=True` para float sem `detach()`.

### Observação operacional

Executar apenas:

```bash
pytest -q
```

sem instalar o pacote ou definir `PYTHONPATH` falha com `ModuleNotFoundError: pre_qwen_certification`.

Usar uma destas opções:

```bash
python -m pip install -e .
python -m pytest -q # verificado

PYTHONPATH=. pytest -q
```

## 3.2 Manifesto global desatualizado

O arquivo:

```text
DELIVERY_MANIFEST.sha256`
```

retorna falhas:

```text
failed: results/VERDICT.md
failed: results/PRE_QWEN_GATE_STATUS.json
```

Os_4�w!j�-���jם