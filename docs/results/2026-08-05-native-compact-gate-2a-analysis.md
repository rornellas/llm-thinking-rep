# Native Compact Gate 2A — native shared-rank training

**Decision:** **NATIVE_COMPACT_GATE_2A_FAIL**

## Scale `small`

| Candidate | Hyp loss | UCB95 | OOD loss | Expert params | Total params | Expert compute |
|---|---:|---:|---:|---:|---:|---:|
| conventional-full | 4.44436 | 4.47506 | 5.48665 | 100.00% | 100.00% | 100.00% |
| conventional-narrow65 | 4.47802 | 4.51255 | 5.47860 | 65.62% | 71.11% | 65.62% |
| native-shared-rank | 4.49555 | 4.52645 | 5.49919 | 45.83% | 54.47% | 62.50% |

### Paired differences

- `primary_minus_narrow_hypothesis`: `+0.017524` 95% `[-0.013513, +0.048835]`.
- `primary_minus_narrow_ood`: `+0.020589` 95% `[-0.028302, +0.084516]`.
- `primary_minus_full_hypothesis`: `+0.051191` 95% `[+0.038286, +0.063727]`.
- `best_primary_minus_narrow_hypothesis`: `+0.017524` 95% `[-0.013934, +0.048633]`.

### Optimization maturity

- `conventional-full` mean terminal calibration slope: `-3.095e-04` per step.
- `conventional-narrow65` mean terminal calibration slope: `-3.108e-04` per step.
- `native-shared-rank` mean terminal calibration slope: `-2.464e-04` per step.

## Scale `medium`

| Candidate | Hyp loss | UCB95 | OOD loss | Expert params | Total params | Expert compute |
|---|---:|---:|---:|---:|---:|---:|
| conventional-full | 4.32959 | 4.35995 | 5.47500 | 100.00% | 100.00% | 100.00% |
| conventional-narrow65 | 4.37425 | 4.40755 | 5.43627 | 65.00% | 68.13% | 65.00% |
| native-shared-rank | 4.39394 | 4.42656 | 5.51072 | 43.75% | 48.77% | 62.50% |

### Paired differences

- `primary_minus_narrow_hypothesis`: `+0.019689` 95% `[+0.005384, +0.038976]`.
- `primary_minus_narrow_ood`: `+0.074448` 95% `[+0.010221, +0.147671]`.
- `primary_minus_full_hypothesis`: `+0.064351` 95% `[+0.054355, +0.073032]`.
- `best_primary_minus_narrow_hypothesis`: `+0.019689` 95% `[+0.005273, +0.039037]`.

### Optimization maturity

- `conventional-full` mean terminal calibration slope: `-2.231e-04` per step.
- `conventional-narrow65` mean terminal calibration slope: `-2.207e-04` per step.
- `native-shared-rank` mean terminal calibration slope: `-1.740e-04` per step.

## Gates

- `small__clean_data`: `True`.
- `small__primary_parameter_advantage`: `True`.
- `small__primary_compute_budget`: `True`.
- `small__primary_hypothesis_noninferior`: `False`.
- `small__primary_ood_noninferior`: `False`.
- `small__primary_full_upper_bound`: `False`.
- `small__every_seed_noninferior`: `False`.
- `small__routing_health`: `True`.
- `medium__clean_data`: `True`.
- `medium__primary_parameter_advantage`: `True`.
- `medium__primary_compute_budget`: `True`.
- `medium__primary_hypothesis_noninferior`: `False`.
- `medium__primary_ood_noninferior`: `False`.
- `medium__primary_full_upper_bound`: `False`.
- `medium__every_seed_noninferior`: `False`.
- `medium__routing_health`: `True`.
- `scale_trend`: `True`.
- `independent_audit`: `True`.

The primary endpoint is final fixed-budget hypothesis loss; best-calibration results are secondary.
Compute is an analytical expert-matrix proxy. No runtime speedup is claimed.
`NO_GO_FOR_OLMOE_OR_QWEN` remains unchanged.
