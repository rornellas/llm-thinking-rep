# Test 0.9 — neuron-wise directly executable expert modes

**Decision:** **FAIL**

Unlike whole-matrix PCA, every intermediate neuron has its own expert coefficients while shared full-rank mode matrices are still projected only once.

| Projection | Mean K=5 explained | Worst-seed p10 K=5 | Mean K=7 explained | Median rank95 |
|---|---:|---:|---:|---:|
| down | 16.45% | 16.01% | 22.18% | 54.0 |
| gate | 20.01% | 17.71% | 25.99% | 53.0 |
| up | 16.42% | 16.01% | 22.15% | 54.0 |

| K | Parameter ratio | Compression | Ideal compute ratio |
|---:|---:|---:|---:|
| 1 | 3.17% | 31.51× | 25.05% |
| 3 | 6.40% | 15.63× | 50.15% |
| 5 | 9.62% | 10.40× | 75.24% |
| 7 | 12.84% | 7.79× | 100.34% |
| 12 | 20.90% | 4.79× | 163.09% |
| 16 | 27.34% | 3.66× | 213.28% |

The scalar-vector code overhead is included. Dense residuals are not included and would invalidate the acceleration path.
For `down`, the exact linear aggregation remains available: each shared matrix consumes the router-weighted sum of code-scaled intermediate vectors.
