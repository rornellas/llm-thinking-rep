# Test 2.10b — extended marginal-utility price search

**Decision:** **MARGINAL_UTILITY_PASS**

Chosen utility price: `0.0260` nat per residual mode (train-calibration only).

| Policy | Validation loss | Δ vs K1 | Mean K | p95 K | Exact compute | Bucket16 | Bucket32 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Static K=0 | 2.4595 | +0.0144 | 0 | 0 | 12.500% | 12.500% | 12.500% |
| Static K=1 | 2.4451 | +0.0000 | 1 | 1 | 25.049% | 25.049% | 25.049% |
| Static K=2 | 2.4340 | -0.0111 | 2 | 2 | 37.598% | 37.598% | 37.598% |
| Static K=3 | 2.4291 | -0.0160 | 3 | 3 | 50.146% | 50.146% | 50.146% |
| **Marginal dynamic** | **2.4470** | **+0.0019** | **0.270** | **2** | **16.300%** | **16.576%** | **16.914%** |

- Test paired-bootstrap UCB95 vs static K1: `+0.0031` nat.
- Bucket16 compute advantage vs K1: `+8.473%`.
- Rank counts: `{'0': 41349, '1': 3946, '2': 2251, '3': 1606}`.

Each controller predicts the cumulative final-language-loss improvement of K=1,2,3 over K=0 for its layer. Inference chooses the prefix maximizing predicted benefit minus a calibrated price per residual mode. Reported bucket cost assumes one rank compaction before grouped GEMMs; scan/scatter latency remains for a kernel benchmark.
