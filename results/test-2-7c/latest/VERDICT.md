# Test 2.7 — learned causal rank controller

**Decision:** **LEARNED_CONTROLLER_PASS**

Chosen threshold: `0.60` (selected only on training-calibration batches).

| Policy | Validation loss | Mean K | p95 K | Projected OLMoE compute |
|---|---:|---:|---:|---:|
| Static K=0 | 2.4192 | 0 | 0 | 12.500% |
| Static K=1 | 2.3991 | 1 | 1 | 25.049% |
| Static K=2 | 2.3902 | 2 | 2 | 37.598% |
| Static K=3 | 2.3855 | 3 | 3 | 50.146% |
| **Learned dynamic** | **2.4048** | **0.737** | **3** | **22.014%** |

- Dynamic minus static-K1 loss: `+0.0057` nat.
- Compute advantage over static K1: `+3.035%` of original OLMoE expert projections.
- Rank counts across all tokens and layers: `{'0': 34439, '1': 2448, '2': 3007, '3': 9258}`.
- Controller-only MLP overhead estimate: `0.263%` of original OLMoE expert projection MACs.

The controller sees the local normalized token state, router summaries, and statistics of mode 0 only. Additional modes are selected before their projections. The validation pass actually applies different ranks per token and layer; it is not an oracle recombination of independent uniform-rank passes.
