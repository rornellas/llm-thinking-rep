# Test 2.10b — extended marginal-utility price search

**Decision:** **MARGINAL_UTILITY_PASS**

Chosen utility price: `0.0300` nat per residual mode (train-calibration only).

| Policy | Validation loss | Δ vs K1 | Mean K | p95 K | Exact compute | Bucket16 | Bucket32 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Static K=0 | 2.4611 | +0.0135 | 0 | 0 | 12.500% | 12.500% | 12.500% |
| Static K=1 | 2.4476 | +0.0000 | 1 | 1 | 25.049% | 25.049% | 25.049% |
| Static K=2 | 2.4381 | -0.0094 | 2 | 2 | 37.598% | 37.598% | 37.598% |
| Static K=3 | 2.4324 | -0.0152 | 3 | 3 | 50.146% | 50.146% | 50.146% |
| **Marginal dynamic** | **2.4498** | **+0.0022** | **0.247** | **2** | **16.018%** | **16.306%** | **16.636%** |

- Test paired-bootstrap UCB95 vs static K1: `+0.0038` nat.
- Bucket16 compute advantage vs K1: `+8.743%`.
- Rank counts: `{'0': 42213, '1': 3471, '2': 1713, '3': 1755}`.

Each controller predicts the cumulative final-language-loss improvement of K=1,2,3 over K=0 for its layer. Inference chooses the prefix maximizing predicted benefit minus a calibrated price per residual mode. Reported bucket cost assumes one rank compaction before grouped GEMMs; scan/scatter latency remains for a kernel benchmark.
