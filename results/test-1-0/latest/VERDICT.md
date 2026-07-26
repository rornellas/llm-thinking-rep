# Test 1.0 — real activation subspaces

**Decision:** **FAIL**

A basis is fitted on 67% of the captured tokens and evaluated on disjoint held-out tokens.
The PCA is uncentered because the target implementation is a linear shared projection without an additive bias.

| Tensor | Tokens | Train rank90 | Validation @128 | @256 | @384 | @512 | Max held-out capture |
|---|---:|---:|---:|---:|---:|---:|---:|
| ffn_moe_out-7 | 768 | 306 | 32.36% | 41.66% | 48.32% | 53.61% | 53.85% |
| ffn_norm-7 | 768 | 281 | 45.56% | 54.89% | 60.71% | 64.93% | 65.06% |
| ffn_norm-7 (reshaped) | 768 | 282 | 46.58% | 56.04% | 61.75% | 65.90% | 65.97% |

At rank 512 the idealized combined gate/up/down factorized compute ratio is 29.17% of the original top-8 projections, before kernel overhead.
This is an oracle subspace screen. It does not yet prove that expert-specific low-dimensional cores preserve logits.
