# Alignment-tolerant shared low-rank residual screen v1

**Decision:** **ALIGNMENT_TOLERANT_SHARED_LORA_FAIL**

| Candidate | Params | Compute | Hyp delta | UCB95 | OOD delta | UCB95 | Worst seed | KL hyp | Top-1 hyp |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| shared-lora-r4 | 35.00% | 51.67% | +0.01581 | +0.05365 | -0.05692 | +0.03048 | +0.04092 | 0.43452 | 66.486% |
| shared-lora-r5 | 41.67% | 58.33% | -0.00764 | +0.02589 | -0.04516 | +0.02483 | +0.02640 | 0.37658 | 68.444% |
| shared-lora-r6 | 48.33% | 65.00% | -0.01320 | +0.01637 | -0.05967 | +0.00001 | +0.01383 | 0.33876 | 70.569% |
| narrow65-frozen-baseline | 65.00% | 65.00% | -0.00388 | +0.01539 | +0.00244 | +0.03220 | +0.02090 | 0.11450 | 83.340% |
| full-continuation-control | 100.00% | 100.00% | +0.00244 | +0.00999 | -0.00509 | +0.00418 | +0.01258 | 0.01250 | 95.007% |

- Rank-5 minus narrow65: mean `-0.00376`, 95% `[-0.03058, +0.02035]`.
- Rank-6 minus narrow65: mean `-0.00932`, 95% `[-0.03889, +0.01778]`.

## Gates

- `primary_hypothesis_pass`: `True`.
- `primary_ood_pass`: `True`.
- `every_seed_primary_pass`: `True`.
- `primary_vs_narrow65_pass`: `False`.
- `primary_parameter_budget_pass`: `True`.
- `primary_compute_budget_pass`: `True`.
- `capacity_hypothesis_pass`: `True`.
- `capacity_vs_narrow65_pass`: `False`.
- `full_control_hypothesis_pass`: `True`.
- `full_control_ood_pass`: `True`.
- `all_arithmetic_pass`: `True`.
- `clean_data_audit_pass`: `True`.
- `independent_audit_pass`: `True`.

## Scope

The teachers and frozen baselines are inherited from the teacher-width replication; no plateau claim is made.
The hypothesis and OOD documents are fresh and are materialized only after candidate freezing.
Parameter and compute values are exact expert-only analytical proxies for this factorized execution; no runtime claim is made.
The frozen NO_GO_FOR_OLMOE_OR_QWEN is unchanged.
