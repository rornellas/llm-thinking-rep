# Alignment-tolerant routing-set distillation v3

**Decision:** **ALIGNMENT_TOLERANT_ROUTING_SET_V3_MECHANISM_SIGNAL**

| Candidate | Params | Compute | Hyp delta | UCB95 | KL | Top-1 | Local NRMSE | CF NRMSE | Cross error |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| shared-lora-r5-routing-set-v3 | 41.67% | 58.33% | -0.04942 | +0.00263 | 0.41681 | 66.63% | 0.36717 | 0.45569 | -0.06858 |
| shared-lora-r5-expert-v2-control | 41.67% | 58.33% | -0.04597 | +0.00923 | 0.43101 | 65.85% | 0.37606 | 0.45381 | -0.04888 |
| shared-lora-r5-aggregate-v1-control | 41.67% | 58.33% | -0.03898 | +0.01932 | 0.40504 | 66.91% | 0.35617 | 0.54829 | -0.22630 |
| shared-lora-r6-routing-set-v3 | 48.33% | 65.00% | -0.06618 | -0.02428 | 0.37295 | 67.64% | 0.34074 | 0.42323 | -0.05817 |
| narrow65-frozen-baseline | 65.00% | 65.00% | -0.00967 | +0.01555 | 0.12559 | 81.66% | 0.17185 | 0.34965 | -0.14172 |
| full-continuation-control | 100.00% | 100.00% | -0.00806 | -0.00103 | 0.01222 | 94.65% | 0.02818 | 0.03044 | +0.00006 |

- v3 minus narrow65 loss: `-0.03976`, 95% `[-0.08813, +0.00787]`.
- v3 minus expert-v2 loss: `-0.00345`, 95% `[-0.01307, +0.00602]`.
- v3 minus narrow65 cross-error: `+0.07314`, 95% `[+0.06462, +0.08419]`.

## Gates

- `primary_hypothesis_pass`: `True`.
- `primary_ood_pass`: `True`.
- `every_seed_primary_pass`: `True`.
- `primary_vs_narrow65_pass`: `False`.
- `primary_vs_expert_v2_pass`: `False`.
- `primary_kl_pass`: `False`.
- `primary_top1_pass`: `False`.
- `primary_local_nrmse_pass`: `False`.
- `primary_counterfactual_pass`: `False`.
- `primary_cross_error_gap_pass`: `False`.
- `capacity_hypothesis_pass`: `True`.
- `full_control_hypothesis_pass`: `True`.
- `full_control_ood_pass`: `True`.
- `primary_parameter_budget_pass`: `True`.
- `primary_compute_budget_pass`: `True`.
- `all_arithmetic_pass`: `True`.
- `clean_data_audit_pass`: `True`.
- `independent_audit_pass`: `True`.

The teachers are inherited fixed checkpoints; no plateau claim is made.
No runtime claim is made. Ratios are exact expert-only analytical proxies.
The frozen NO_GO_FOR_OLMOE_OR_QWEN is unchanged.
