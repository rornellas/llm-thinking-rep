# Routing-coupled residual v4 — análise factual e adversarial

**Protocolo:** `pre-qwen-routing-coupled-residual-v4`  
**Veredito automático auditado:** `ROUTING_COUPLED_V4_FAIL`  
**Auditoria independente:** `PASS`  
**Decisão global:** `NO_GO_FOR_OLMOE_OR_QWEN`

## Hipótese

A v4 adiciona uma correção permutation-invariant condicionada ao conjunto roteado, reutilizando os latentes low-rank do `down`. O objetivo é tornar a coordenação entre experts representável pela arquitetura, em vez de esperar que a loss induza covariância favorável indiretamente.

## Resultados

| Candidate | Params | Compute | Hyp delta | Hyp UCB95 | KL | Top-1 | Local | CF | Aggregate error | Correction energy |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| rank5-coupled-q8-h8-v4 | 44.28% | 62.50% | -0.07459 | -0.02307 | 0.38644 | 67.65% | 0.36517 | 0.46091 | 0.15158 | 0.00276 |
| rank5-coupled-q8-h8-mean-only-control | 44.28% | 62.50% | -0.07385 | -0.02440 | 0.38760 | 67.97% | 0.36592 | 0.46133 | 0.15203 | 0.00216 |
| rank5-coupled-q12-h8-v4 | 45.31% | 63.75% | -0.06976 | -0.02123 | 0.38962 | 67.67% | 0.36502 | 0.46084 | 0.15145 | 0.00296 |
| rank5-v3-frozen-baseline | 41.67% | 58.33% | -0.07769 | -0.01141 | 0.42108 | 66.12% | 0.37057 | 0.46145 | 0.15570 | 0.00000 |
| rank6-v3-frozen-capacity | 48.33% | 65.00% | -0.08454 | -0.03198 | 0.37608 | 68.47% | 0.34384 | 0.42748 | 0.13524 | 0.00000 |
| narrow65-frozen-baseline | 65.00% | 65.00% | -0.00862 | +0.01714 | 0.11655 | 82.69% | 0.17041 | 0.35036 | 0.03434 | 0.00000 |
| full-continuation-control | 100.00% | 100.00% | +0.00117 | +0.00982 | 0.01411 | 94.93% | 0.02844 | 0.03055 | 0.00096 | 0.00000 |

## Comparações load-bearing

- `primary_minus_narrow_loss`: `-0.065975 [-0.125942, -0.013520]`.
- `primary_minus_rank6_loss`: `+0.009949 [-0.024493, +0.049457]`.
- `primary_minus_v3_loss`: `+0.003093 [-0.028750, +0.031750]`.
- `primary_minus_v3_kl`: `-0.034643 [-0.049048, -0.021484]`.
- `primary_minus_v3_top1`: `+0.015278 [+0.005139, +0.024792]`.
- `primary_minus_v3_local`: `-0.005396 [-0.007168, -0.003669]`.
- `primary_minus_v3_counterfactual`: `-0.000537 [-0.001644, +0.000594]`.
- `primary_minus_mean_only_kl`: `-0.001162 [-0.005049, +0.002677]`.
- `disabled_minus_primary_kl`: `+0.012102 [+0.007258, +0.017261]`.
- `disabled_minus_primary_loss`: `-0.004927 [-0.015821, +0.006762]`.
- `primary_minus_narrow_aggregate_error`: `+0.117240 [+0.104094, +0.132075]`.

## Gates

- `all_arithmetic_pass`: `True`.
- `causal_coupling_kl_pass`: `False`.
- `causal_coupling_loss_pass`: `False`.
- `causal_coupling_pass`: `False`.
- `clean_data_audit_pass`: `True`.
- `every_seed_primary_pass`: `True`.
- `full_control_hypothesis_pass`: `True`.
- `full_control_ood_pass`: `False`.
- `independent_audit_pass`: `True`.
- `primary_aggregate_error_gap_pass`: `False`.
- `primary_compute_budget_pass`: `True`.
- `primary_counterfactual_pass`: `False`.
- `primary_hypothesis_pass`: `True`.
- `primary_kl_pass`: `False`.
- `primary_local_nrmse_pass`: `False`.
- `primary_ood_pass`: `True`.
- `primary_parameter_budget_pass`: `True`.
- `primary_top1_pass`: `False`.
- `primary_vs_mean_only_kl_pass`: `False`.
- `primary_vs_narrow65_pass`: `True`.
- `primary_vs_rank6_pass`: `False`.
- `primary_vs_v3_pass`: `False`.
- `required_improvement_votes`: `2`.

Behavior-improvement votes versus v3: `3`.

## Causalidade do acoplador

A candidata `coupling-disabled` usa os mesmos pesos treinados, mas zera a correção de conjunto. Diferenças positivas de KL/loss em `disabled-primary` são necessárias para atribuir o resultado ao acoplador, e não apenas ao refinamento da base rank-5.

- KL disabled-primary: `+0.012102 [+0.007258, +0.017261]`.
- loss disabled-primary: `-0.004927 [-0.015821, +0.006762]`.

## Leitura adversarial obrigatória

- CE favorável sem KL, top-1 e fidelidade local não constitui preservação geral.
- Cross-error é diagnóstico, não gate, pois uma correção de conjunto não possui alocação única por expert.
- O controle mean-only separa o valor do segundo momento do simples aumento de parâmetros.
- O controle q12 testa capacidade, mas não pode resgatar ausência de causalidade do q8 primário.
- Os teachers são checkpoints fixos herdados e não foram demonstrados como plateaued.
- Nenhuma razão analítica é um claim de speedup real.

## Integridade

- auditoria independente: `PASS`;
- divergências: `0`;
- cobertura: `True`;
- dados: `True`;
- aritmética: `True`;
- source checkpoints: `True`;
- bootstrap cruzado por seed e documento;
- leave-one-seed-out persistido no JSON da auditoria;
- checkpoints, registros por janela, logs, ambiente e hashes versionados.
