# Test 2.3 — functional distillation of trained experts

**Decision:** **PASS_K1**

All compatible non-expert weights and router weights are copied from the conventional teacher and frozen. Only modal matrices and codes are trained.

| Rank | Expert params | Ideal expert compute | Loss before | Loss after | Loss/teacher | KL after | MoE error before | MoE error after |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 12.51% | 50.01% | 3.5282 | 2.3833 | 1.000× | 0.0190 | 121.53% | 32.64% |
| 2 | 18.77% | 75.02% | 3.6420 | 2.3883 | 0.995× | 0.0157 | 124.03% | 29.11% |

A positive result would demonstrate functional conversion by distillation, not analytic weight decomposition. The small model remains a mechanism test; scaling and multi-domain validation remain required.
