# Test 2.10b — extended marginal-utility price search

**Decision:** **MARGINAL_UTILITY_PASS**

Chosen utility price: `0.0140` nat per residual mode (train-calibration only).

| Policy | Validation loss | Δ vs K1 | Mean K | p95 K | Exact compute | Bucket16 | Bucket32 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Static K=0 | 2.4393 | +0.0198 | 0 | 0 | 12.500% | 12.500% | 12.500% |
| Static K=1 | 2.4195 | +0.0000 | 1 | 1 | 25.049% | 25.049% | 25.049% |
| Static K=2 | 2.4114 | -0.0080 | 2 | 2 | 37.598% | 37.598% | 37.598% |
| Static K=3 | 2.4060 | -0.0135 | 3 | 3 | 50.146% | 50.146% | 50.146% |
| **Marginal dynamic** | **2.4220** | **+0.0025** | **0.658** | **3** | **21.175%** | **21.429%** | **21.734%** |

- Test paired-bootstrap UCB95 vs static K1: `+0.0038` nat.
- Bucket16 compute advantage vs K1: `+3.620%`.
- Rank counts: `{'0': 32172, '1': 6839, '2': 4901, '3': 5240}`.

Each controller predicts the cumulative final-language-loss improvement of K=1,2,3 over K=0 for its layer. Inference chooses the prefix maximizing predicted benefit minus a calibrated price per residual mode. Reported bucket cost assumes one rank compaction before grouped GEMMs; scan/scatter latency remains for a kernel benchmark.
