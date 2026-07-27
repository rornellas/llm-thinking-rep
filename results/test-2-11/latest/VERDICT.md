# Test 2.11 — multi-seed marginal-utility controller

**Decision:** **ROBUST_MARGINAL_UTILITY_PASS**

| Seed | Lambda | Δ loss vs K1 | UCB95 | Mean K | p95 K | Exact compute | Bucket16 | Advantage vs K1 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 61616 | 0.0140 | +0.0025 | +0.0038 | 0.658 | 3 | 21.175% | 21.429% | +3.620% |
| 71717 | 0.0260 | +0.0019 | +0.0031 | 0.270 | 2 | 16.300% | 16.576% | +8.473% |
| 81818 | 0.0200 | +0.0014 | +0.0026 | 0.498 | 2 | 19.158% | 19.476% | +5.573% |

- Mean bucket16 compute: `19.160%`.
- Mean advantage over static K1: `+5.889%`.
- Worst held-out Δ loss: `+0.0025` nat.
- Worst paired-bootstrap UCB95: `+0.0038` nat.
- Worst bucket16 compute: `21.429%`.

Every seed trains a new progressive Modal-MoE and new utility controllers. Utility price selection is confined to train-split calibration data.
