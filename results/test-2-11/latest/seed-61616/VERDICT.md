# Test 2.10b — extended marginal-utility price search

**Decision:** **MARGINAL_UTILITY_PASS**

Chosen utility price: `0.0160` nat per residual mode (train-calibration only).

| Policy | Validation loss | Δ vs K1 | Mean K | p95 K | Exact compute | Bucket16 | Bucket32 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Static K=0 | 2.4393 | +0.0198 | 0 | 0 | 12.500% | 12.500% | 12.500% |
| Static K=1 | 2.4195 | +0.0000 | 1 | 1 | 25.049% | 25.049% | 25.049% |
| Static K=2 | 2.4125 | -0.0070 | 2 | 2 | 37.598% | 37.598% | 37.598% |
| Static K=3 | 2.4063 | -0.0132 | 3 | 3 | 50.146% | 50.146% | 50.146% |
| **Marginal dynamic** | **2.4226** | **+0.0031** | **0.600** | **3** | **20.445%** | **20.685%** | **20.966%** |

- Test paired-bootstrap UCB95 vs static K1: `+0.0042` nat.
- Bucket16 compute advantage vs K1: `+4.363%`.
- Rank counts: `{'0': 33590, '1': 5843, '2': 5496, '3': 4223}`.

Each controller predicts the cumulative final-language-loss improvement of K=1,2,3 over K=0 for its layer. Inference chooses the prefix maximizing predicted benefit minus a calibrated price per residual mode. Reported bucket cost assumes one rank compaction before grouped GEMMs; scan/scatter latency remains for a kernel benchmark.
