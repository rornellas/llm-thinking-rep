# Test 2.10b — extended marginal-utility price search

**Decision:** **MARGINAL_UTILITY_PASS**

Chosen utility price: `0.0200` nat per residual mode (train-calibration only).

| Policy | Validation loss | Δ vs K1 | Mean K | p95 K | Exact compute | Bucket16 | Bucket32 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Static K=0 | 2.4591 | +0.0167 | 0 | 0 | 12.500% | 12.500% | 12.500% |
| Static K=1 | 2.4424 | +0.0000 | 1 | 1 | 25.049% | 25.049% | 25.049% |
| Static K=2 | 2.4332 | -0.0093 | 2 | 2 | 37.598% | 37.598% | 37.598% |
| Static K=3 | 2.4282 | -0.0143 | 3 | 3 | 50.146% | 50.146% | 50.146% |
| **Marginal dynamic** | **2.4439** | **+0.0014** | **0.498** | **2** | **19.158%** | **19.476%** | **19.716%** |

- Test paired-bootstrap UCB95 vs static K1: `+0.0026` nat.
- Bucket16 compute advantage vs K1: `+5.573%`.
- Rank counts: `{'0': 33945, '1': 7744, '2': 5673, '3': 1790}`.

Each controller predicts the cumulative final-language-loss improvement of K=1,2,3 over K=0 for its layer. Inference chooses the prefix maximizing predicted benefit minus a calibrated price per residual mode. Reported bucket cost assumes one rank compaction before grouped GEMMs; scan/scatter latency remains for a kernel benchmark.
