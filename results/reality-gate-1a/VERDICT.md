# Reality Gate 1A — compressibility trajectory and heterogeneous static rank

**Decision:** **REALITY_GATE_1A_FAIL**

## Scale `small`

| Candidate | Hyp delta | UCB95 | KL | Top-1 | Local NRMSE | Params | Train compute | Hyp compute |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| heterogeneous-spectral | +0.00604 | +0.00798 | 0.01453 | 84.51% | 0.19035 | 45.83% | 62.50% | 62.50% |
| uniform-rank | +0.00595 | +0.00792 | 0.01451 | 84.49% | 0.19026 | 45.83% | 62.50% | 62.50% |
| heterogeneous-routing | +0.00595 | +0.00796 | 0.01451 | 84.49% | 0.19026 | 45.83% | 62.50% | 62.50% |
| narrow65 | -0.00142 | -0.00052 | 0.00468 | 91.62% | 0.08759 | 65.62% | 65.62% | 65.62% |
| full-identity-control | +0.00000 | +0.00000 | 0.00000 | 100.00% | 0.00000 | 100.00% | 100.00% | 100.00% |

### Comparisons

- `primary_minus_uniform_loss`: `+0.000087` 95% `[-0.000145, +0.000495]`.
- `primary_minus_uniform_kl`: `+0.000017` 95% `[-0.000009, +0.000077]`.
- `primary_minus_uniform_top1`: `+0.000206` 95% `[-0.000480, +0.001336]`.
- `primary_minus_uniform_local`: `+0.000093` 95% `[+0.000000, +0.000289]`.
- `primary_minus_routing_loss`: `+0.000087` 95% `[-0.000147, +0.000504]`.
- `primary_minus_narrow_loss`: `+0.007460` 95% `[+0.005427, +0.009378]`.
- `primary_minus_narrow_kl`: `+0.009849` 95% `[+0.008531, +0.011803]`.
- `primary_minus_narrow_top1`: `-0.071135` 95% `[-0.077942, -0.064773]`.
- `primary_minus_narrow_local`: `+0.102763` 95% `[+0.099066, +0.106251]`.

### Plateau

- `111731`: reached=`False`, final_step=`3200`.
- `121747`: reached=`False`, final_step=`3200`.
- `131759`: reached=`False`, final_step=`3200`.
- `141767`: reached=`False`, final_step=`3200`.

## Scale `medium`

| Candidate | Hyp delta | UCB95 | KL | Top-1 | Local NRMSE | Params | Train compute | Hyp compute |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| heterogeneous-spectral | +0.00800 | +0.01020 | 0.02074 | 83.05% | 0.21355 | 43.75% | 62.50% | 62.50% |
| uniform-rank | +0.00800 | +0.01022 | 0.02074 | 83.05% | 0.21355 | 43.75% | 62.50% | 62.50% |
| heterogeneous-routing | +0.00800 | +0.01012 | 0.02074 | 83.05% | 0.21355 | 43.75% | 62.50% | 62.50% |
| narrow65 | -0.00313 | -0.00117 | 0.00790 | 89.85% | 0.10512 | 64.58% | 64.58% | 64.58% |
| full-identity-control | +0.00000 | +0.00000 | 0.00000 | 100.00% | 0.00000 | 100.00% | 100.00% | 100.00% |

### Comparisons

- `primary_minus_uniform_loss`: `+0.000000` 95% `[+0.000000, +0.000000]`.
- `primary_minus_uniform_kl`: `+0.000000` 95% `[+0.000000, +0.000000]`.
- `primary_minus_uniform_top1`: `+0.000000` 95% `[+0.000000, +0.000000]`.
- `primary_minus_uniform_local`: `+0.000000` 95% `[+0.000000, +0.000000]`.
- `primary_minus_routing_loss`: `+0.000000` 95% `[+0.000000, +0.000000]`.
- `primary_minus_narrow_loss`: `+0.011128` 95% `[+0.008783, +0.013631]`.
- `primary_minus_narrow_kl`: `+0.012840` 95% `[+0.011905, +0.013891]`.
- `primary_minus_narrow_top1`: `-0.068028` 95% `[-0.073236, -0.062648]`.
- `primary_minus_narrow_local`: `+0.108432` 95% `[+0.105000, +0.112366]`.

### Plateau

- `111731`: reached=`False`, final_step=`4500`.
- `121747`: reached=`False`, final_step=`4500`.
- `131759`: reached=`False`, final_step=`4500`.
- `141767`: reached=`False`, final_step=`4500`.

## Gates

- `small__all_teachers_plateaued`: `False`.
- `small__clean_data`: `True`.
- `small__primary_parameter_budget`: `True`.
- `small__primary_train_compute_budget`: `True`.
- `small__primary_hypothesis_compute_budget`: `True`.
- `small__primary_absolute_loss`: `True`.
- `small__primary_absolute_kl`: `True`.
- `small__primary_absolute_top1`: `True`.
- `small__primary_absolute_local`: `True`.
- `small__primary_ood_loss`: `True`.
- `small__primary_ood_kl`: `True`.
- `small__primary_every_seed_loss`: `True`.
- `small__primary_vs_uniform_loss`: `False`.
- `small__primary_vs_uniform_behavior`: `False`.
- `small__primary_vs_routing_loss`: `False`.
- `small__primary_vs_narrow_loss`: `True`.
- `small__primary_vs_narrow_kl`: `True`.
- `small__primary_vs_narrow_top1`: `False`.
- `small__primary_vs_narrow_local`: `False`.
- `small__full_identity`: `True`.
- `medium__all_teachers_plateaued`: `False`.
- `medium__clean_data`: `True`.
- `medium__primary_parameter_budget`: `True`.
- `medium__primary_train_compute_budget`: `True`.
- `medium__primary_hypothesis_compute_budget`: `True`.
- `medium__primary_absolute_loss`: `True`.
- `medium__primary_absolute_kl`: `True`.
- `medium__primary_absolute_top1`: `True`.
- `medium__primary_absolute_local`: `True`.
- `medium__primary_ood_loss`: `True`.
- `medium__primary_ood_kl`: `True`.
- `medium__primary_every_seed_loss`: `True`.
- `medium__primary_vs_uniform_loss`: `True`.
- `medium__primary_vs_uniform_behavior`: `True`.
- `medium__primary_vs_routing_loss`: `True`.
- `medium__primary_vs_narrow_loss`: `False`.
- `medium__primary_vs_narrow_kl`: `True`.
- `medium__primary_vs_narrow_top1`: `False`.
- `medium__primary_vs_narrow_local`: `False`.
- `medium__full_identity`: `True`.
- `scale_trend`: `True`.
- `independent_audit`: `True`.

No runtime claim is made. Compute ratios are analytical expected routed-matrix proxies.
The frozen `NO_GO_FOR_OLMOE_OR_QWEN` remains unchanged.
