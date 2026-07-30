# Routing-coupled residual v4

**Decision:** **ROUTING_COUPLED_V4_FAIL**

| Candidate | Params | Compute | Hyp delta | UCB95 | KL | Top-1 | Local NRMSE | CF NRMSE | Aggregate error | Cross error (diag.) | Correction energy |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| rank5-coupled-q8-h8-v4 | 44.28% | 62.50% | -0.07459 | -0.02307 | 0.38644 | 67.65% | 0.36517 | 0.46091 | 0.15158 | -0.07758 | 0.00276 |
| rank5-coupled-q8-h8-mean-only-control | 44.28% | 62.50% | -0.07385 | -0.02440 | 0.38760 | 67.97% | 0.36592 | 0.46133 | 0.15203 | -0.07770 | 0.00216 |
| rank5-coupled-q12-h8-v4 | 45.31% | 63.75% | -0.06976 | -0.02123 | 0.38962 | 67.67% | 0.36502 | 0.46084 | 0.15145 | -0.07805 | 0.00296 |
| rank5-v3-frozen-baseline | 41.67% | 58.33% | -0.07769 | -0.01141 | 0.42108 | 66.12% | 0.37057 | 0.46145 | 0.15570 | -0.07070 | 0.00000 |
| rank6-v3-frozen-capacity | 48.33% | 65.00% | -0.08454 | -0.03198 | 0.37608 | 68.47% | 0.34384 | 0.42748 | 0.13524 | -0.05913 | 0.00000 |
| narrow65-frozen-baseline | 65.00% | 65.00% | -0.00862 | +0.01714 | 0.11655 | 82.69% | 0.17041 | 0.35036 | 0.03434 | -0.14066 | 0.00000 |
| full-continuation-control | 100.00% | 100.00% | +0.00117 | +0.00982 | 0.01411 | 94.93% | 0.02844 | 0.03055 | 0.00096 | +0.00007 | 0.00000 |

## Load-bearing comparisons

- `primary_minus_narrow_loss`: mean `-0.065975`, 95% `[-0.125942, -0.013520]`.
- `primary_minus_rank6_loss`: mean `+0.009949`, 95% `[-0.024493, +0.049457]`.
- `primary_minus_v3_loss`: mean `+0.003093`, 95% `[-0.028750, +0.031750]`.
- `primary_minus_v3_kl`: mean `-0.034643`, 95% `[-0.049048, -0.021484]`.
- `primary_minus_v3_top1`: mean `+0.015278`, 95% `[+0.005139, +0.024792]`.
- `primary_minus_v3_local`: mean `-0.005396`, 95% `[-0.007168, -0.003669]`.
- `primary_minus_v3_counterfactual`: mean `-0.000537`, 95% `[-0.001644, +0.000594]`.
- `primary_minus_mean_only_kl`: mean `-0.001162`, 95% `[-0.005049, +0.002677]`.
- `primary_minus_mean_only_loss`: mean `-0.000741`, 95% `[-0.005731, +0.004154]`.
- `disabled_minus_primary_kl`: mean `+0.012102`, 95% `[+0.007258, +0.017261]`.
- `disabled_minus_primary_loss`: mean `-0.004927`, 95% `[-0.015821, +0.006762]`.
- `second_disabled_minus_primary_kl`: mean `+0.005330`, 95% `[+0.001943, +0.008829]`.
- `primary_minus_narrow_aggregate_error`: mean `+0.117240`, 95% `[+0.104094, +0.132075]`.

## Gates

- `primary_hypothesis_pass`: `True`.
- `primary_ood_pass`: `True`.
- `every_seed_primary_pass`: `True`.
- `primary_vs_narrow65_pass`: `True`.
- `primary_vs_rank6_pass`: `False`.
- `primary_vs_v3_pass`: `False`.
- `primary_vs_mean_only_kl_pass`: `False`.
- `primary_kl_pass`: `False`.
- `primary_top1_pass`: `False`.
- `primary_local_nrmse_pass`: `False`.
- `primary_counterfactual_pass`: `False`.
- `primary_aggregate_error_gap_pass`: `False`.
- `causal_coupling_kl_pass`: `False`.
- `causal_coupling_loss_pass`: `False`.
- `causal_coupling_pass`: `False`.
- `full_control_hypothesis_pass`: `True`.
- `full_control_ood_pass`: `False`.
- `primary_parameter_budget_pass`: `True`.
- `primary_compute_budget_pass`: `True`.
- `all_arithmetic_pass`: `True`.
- `clean_data_audit_pass`: `True`.
- `independent_audit_pass`: `True`.
- `required_improvement_votes`: `2`.

Behavior-improvement votes versus frozen v3: `3`.

Cross error is diagnostic only because set-level corrections have no unique per-expert allocation.
The teachers are inherited fixed checkpoints; no plateau claim is made.
No runtime claim is made. Ratios are expert-only analytical proxies.
The frozen `NO_GO_FOR_OLMOE_OR_QWEN` is unchanged.
