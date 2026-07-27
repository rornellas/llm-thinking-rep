# Test 6.1 — training for sparse refinement events

**Decision:** **TRAINED_EVENT_SPARSITY_FAIL**

| Fine-tune | K3 loss | K1 loss | 90%/4-bit loss | Δ vs K3 | Event density | Residual bits/BF16 | Advantage vs random | 95% Δ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| lambda-0.000 | 2.3531 | 2.3730 | 2.3768 | +0.0237 | 9.650% | 8.662% | +0.0096 | +0.0296 |
| lambda-0.020 | 2.3643 | 2.3814 | 2.3869 | +0.0227 | 9.751% | 8.688% | +0.0095 | +0.0289 |
| lambda-0.050 | 2.3509 | 2.3728 | 2.3716 | +0.0208 | 9.874% | 8.718% | +0.0126 | +0.0277 |

- Best variant: `lambda-0.050`.
- Best 90%/4-bit Δ versus K3: `+0.0208` nat.
- Event traffic: `8.718%` of dense BF16 residual deltas.
- Improvement over lambda=0 fine-tune: `+0.0029` nat.
- Magnitude advantage over random: `+0.0126` nat.

Hoyer regularization is scale invariant and operates on per-token MoE output deltas. This remains a state-traffic screen; dense prefix computation is still used by the reference path.
