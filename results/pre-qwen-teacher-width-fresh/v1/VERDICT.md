# Teacher-informed narrow width — fresh replication v1

**Decision:** **TEACHER_WIDTH_65_REPLICATION_PASS**

| Candidate | Params | Compute | Hyp Δ | UCB95 | OOD Δ | UCB95 | Worst seed | KL hyp | Top-1 hyp |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| magnitude-init-35 | 35.0% | 35.0% | +0.08626 | +0.10517 | -0.02408 | +0.03363 | +0.10496 | 0.09460 | 88.694% |
| magnitude-init-50 | 50.0% | 50.0% | +0.02973 | +0.03984 | -0.01345 | +0.03085 | +0.04134 | 0.05044 | 92.042% |
| magnitude-init-65 | 65.0% | 65.0% | +0.01325 | +0.01914 | -0.01157 | +0.01021 | +0.01608 | 0.02979 | 94.028% |
| magnitude-init-75 | 75.0% | 75.0% | +0.00412 | +0.01144 | -0.00942 | +0.01467 | +0.01217 | 0.02000 | 95.083% |
| full-continuation-control | 100.0% | 100.0% | -0.00431 | +0.00207 | -0.00208 | +0.00860 | +0.00569 | 0.00866 | 96.681% |

- Primary 65% minus 50%: mean `-0.01648`, 95% `[-0.02611, -0.00570]`.

## Gates

- `primary_hypothesis_pass`: `True`.
- `primary_ood_pass`: `True`.
- `every_seed_primary_pass`: `True`.
- `primary_vs_comparator_pass`: `True`.
- `capacity_pass`: `True`.
- `anchor_failure_reproduced`: `True`.
- `full_control_hypothesis_pass`: `True`.
- `full_control_ood_pass`: `True`.
- `exact_primary_parameter_ratio_pass`: `True`.
- `exact_primary_compute_ratio_pass`: `True`.
- `clean_data_audit_pass`: `True`.
- `independent_audit_pass`: `True`.

No convergence claim is made for the teachers; tail changes are preserved in metrics.json.
No runtime claim is made. Width ratios are exact routed matrix-operation and expert-parameter proxies.
The frozen NO_GO_FOR_OLMOE_OR_QWEN is unchanged.
