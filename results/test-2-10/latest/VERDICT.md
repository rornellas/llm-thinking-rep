# Test 2.10 — marginal-utility rank controller

**Decision:** **MARGINAL_UTILITY_PASS**

Chosen utility price: `0.0200` nat per residual mode (train-calibration only).

| Policy | Validation loss | Δ vs K1 | Mean K | p95 K | Exact compute | Bucket16 | Bucket32 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Static K=0 | 2.4862 | +0.0171 | 0 | 0 | 12.500% | 12.500% | 12.500% |
| Static K=1 | 2.4691 | +0.0000 | 1 | 1 | 25.049% | 25.049% | 25.049% |
| Static K=2 | 2.4601 | -0.0090 | 2 | 2 | 37.598% | 37.598% | 37.598% |
| Static K=3 | 2.4563 | -0.0128 | 3 | 3 | 50.146% | 50.146% | 50.146% |
| **Marginal dynamic** | **2.4712** | **+0.0021** | **0.499** | **3** | **19.173%** | **19.452%** | **19.773%** |

- Test paired-bootstrap UCB95 vs static K1: `+0.0037` nat.
- Bucket16 compute advantage vs K1: `+5.597%`.
- Rank counts: `{'0': 35802, '1': 6083, '2': 3363, '3': 3904}`.

Each controller predicts the cumulative final-language-loss improvement of K=1,2,3 over K=0 for its layer. Inference chooses the prefix maximizing predicted benefit minus a calibrated price per residual mode. Reported bucket cost assumes one rank compaction before grouped GEMMs; scan/scatter latency remains for a kernel benchmark.
