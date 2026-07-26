# Test 0.6 — shared feature subspaces

**Decision:** **FAIL**

This tests a shared input basis for `gate`+`up` and a shared output basis for `down`; it does not require one-to-one neuron alignment.

| Rank | Gate+up validation capture | Down validation capture | Worst expert p05 | Parameter ratio | Ideal compute ratio |
|---:|---:|---:|---:|---:|---:|
| 128 | 10.20% | 10.12% | 7.34% | 6.38% | 7.29% |
| 256 | 17.47% | 17.46% | 13.80% | 12.76% | 14.58% |
| 384 | 24.15% | 24.27% | 20.31% | 19.14% | 21.88% |
| 512 | 30.54% | 30.74% | 26.92% | 25.52% | 29.17% |
| 768 | 42.85% | 43.16% | 39.48% | 38.28% | 43.75% |
| 1024 | 54.73% | 55.01% | 51.79% | 51.04% | 58.33% |

PASS requires at least 90% held-out weight energy in both branches by rank ≤512, with p05 expert capture ≥80%.
BORDERLINE permits rank ≤1024 and authorizes an activation-aware test, but not an acceleration claim.
