# Test 0.5 — scale and permutation alignment

**Pilot continuation gate:** **FAIL**

The matching coordinates and evaluation coordinates are disjoint.

| Subset | Start expert | Joint K=4 before | Joint K=4 after | Gain | Rank95 after | Holdout score |
|---|---:|---:|---:|---:|---:|---:|
| subset-1 | 0 | 28.55% | 28.57% | +0.03% | 15 | 0.5104 |
| subset-1 | 30 | 28.55% | 28.64% | +0.09% | 15 | 0.5106 |
| subset-2 | 9 | 28.37% | 28.42% | +0.05% | 15 | 0.5135 |
| subset-2 | 43 | 28.37% | 28.34% | -0.03% | 15 | 0.5117 |

Gate rule: each independent subset needs at least one start with a joint K=4 gain of 12 percentage points and aligned rank95 at most 11.

A PASS authorizes all-64 alignment. A FAIL redirects the project toward permutation-invariant neuron dictionaries, clustering, or activation-aware factorization.
