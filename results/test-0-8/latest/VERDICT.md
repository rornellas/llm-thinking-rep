# Test 0.8 — permutation-invariant neuron dictionary screen

**Decision:** **FAIL**

Queries are matched to all neurons from other experts using training coordinates and then scored on disjoint holdout coordinates and all 2,048 dimensions.

| Metric | Mean | Median | p95 | Maximum |
|---|---:|---:|---:|---:|
| Train-selected top1 / training | 0.3821 | 0.3767 | 0.4297 | 0.6797 |
| Train-selected top1 / holdout | 0.0566 | 0.0435 | 0.2471 | 0.6412 |
| Train-selected top1 / exact | 0.0387 | 0.0181 | 0.2376 | 0.6000 |
| Best-holdout among train top64 / exact | 0.0671 | 0.0266 | 0.2815 | 0.6000 |
| Random cross-expert / exact | 0.0090 | 0.0087 | 0.0316 | 0.0561 |

A compact directly executed dictionary needs very high cross-expert reuse; cosine-like scores near zero indicate unrelated triplets, while scores near one indicate reusable atoms.
The top64 oracle is deliberately optimistic and is not a deployable selection rule.
