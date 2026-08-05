# Reality Gate 1A — análise factual e adversarial

**Protocolo:** `reality-gate-1a-static-heterogeneous-rank-v1`  
**Veredito:** `REALITY_GATE_1A_FAIL`  
**Auditoria independente:** `PASS`  
**Decisão global:** `NO_GO_FOR_OLMOE_OR_QWEN`

## Pergunta

Ranks heterogêneos estáticos, alocados apenas com informação de treino e sob os mesmos orçamentos de parâmetros e compute esperado do rank uniforme, melhoram a fidelidade em teachers com plateau explícito e em duas escalas?

## Escala `medium`

| Candidate | Δ loss | KL | Top-1 | Local NRMSE | Params | Train compute | Hyp compute |
|---|---:|---:|---:|---:|---:|---:|---:|
| heterogeneous-spectral | +0.00800 | 0.02074 | 83.05% | 0.21355 | 43.75% | 62.50% | 62.50% |
| uniform-rank | +0.00800 | 0.02074 | 83.05% | 0.21355 | 43.75% | 62.50% | 62.50% |
| heterogeneous-routing | +0.00800 | 0.02074 | 83.05% | 0.21355 | 43.75% | 62.50% | 62.50% |
| narrow65 | -0.00313 | 0.00790 | 89.85% | 0.10512 | 64.58% | 64.58% | 64.58% |
| full-identity-control | +0.00000 | 0.00000 | 100.00% | 0.00000 | 100.00% | 100.00% | 100.00% |

### Comparações load-bearing

- spectral − uniform loss: `+0.000000 [+0.000000, +0.000000]`.
- spectral − uniform KL: `+0.000000 [+0.000000, +0.000000]`.
- spectral − uniform top-1: `+0.000000 [+0.000000, +0.000000]`.
- spectral − narrow65 loss: `+0.011128 [+0.008783, +0.013631]`.

### Plateau

- seed `111731`: plateau=`False`, final_step=`4500`.
- seed `121747`: plateau=`False`, final_step=`4500`.
- seed `131759`: plateau=`False`, final_step=`4500`.
- seed `141767`: plateau=`False`, final_step=`4500`.

## Escala `small`

| Candidate | Δ loss | KL | Top-1 | Local NRMSE | Params | Train compute | Hyp compute |
|---|---:|---:|---:|---:|---:|---:|---:|
| heterogeneous-spectral | +0.00604 | 0.01453 | 84.51% | 0.19035 | 45.83% | 62.50% | 62.50% |
| uniform-rank | +0.00595 | 0.01451 | 84.49% | 0.19026 | 45.83% | 62.50% | 62.50% |
| heterogeneous-routing | +0.00595 | 0.01451 | 84.49% | 0.19026 | 45.83% | 62.50% | 62.50% |
| narrow65 | -0.00142 | 0.00468 | 91.62% | 0.08759 | 65.62% | 65.62% | 65.62% |
| full-identity-control | +0.00000 | 0.00000 | 100.00% | 0.00000 | 100.00% | 100.00% | 100.00% |

### Comparações load-bearing

- spectral − uniform loss: `+0.000087 [-0.000145, +0.000495]`.
- spectral − uniform KL: `+0.000017 [-0.000009, +0.000077]`.
- spectral − uniform top-1: `+0.000206 [-0.000480, +0.001336]`.
- spectral − narrow65 loss: `+0.007460 [+0.005427, +0.009378]`.

### Plateau

- seed `111731`: plateau=`False`, final_step=`3200`.
- seed `121747`: plateau=`False`, final_step=`3200`.
- seed `131759`: plateau=`False`, final_step=`3200`.
- seed `141767`: plateau=`False`, final_step=`3200`.

## Leitura adversarial obrigatória

- Um ganho contra rank uniforme não basta se o teacher não atingiu plateau.
- Alocação espectral deve superar o controle baseado apenas em frequência; caso contrário, a complexidade estrutural não está justificada.
- Compute é um proxy analítico esperado. Nenhuma aceleração de runtime é inferida.
- Resultado em WikiText-2 pequeno não autoriza transplante para checkpoint real.
- Se heterogeneidade falhar nas duas escalas, o controlador dinâmico não será implementado; a linha pós-hoc será despriorizada.

## Integridade

- audit passed: `True`;
- mismatches: `0`;
- provenance: `True`;
- checkpoint hashes: `True`;
- data isolation: `True`;
- bootstrap cruzado por seed e documento;
- resultados, checkpoints, dados tokenizados, tokenizer, logs, ambiente e hashes versionados.
