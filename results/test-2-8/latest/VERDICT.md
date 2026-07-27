# Test 2.7 — learned causal rank controller

**Decision:** **LEARNED_CONTROLLER_PASS**

Chosen threshold: `0.60` (selected only on training-calibration batches).

| Policy | Validation loss | Mean K | p95 K | Projected OLMoE compute |
|---|---:|---:|---:|---:|
| Static K=0 | 2.4483 | 0 | 0 | 12.500% |
| Static K=1 | 2.4274 | 1 | 1 | 25.049% |
| Static K=2 | 2.4145 | 2 | 2 | 37.598% |
| Static K=3 | 2.4082 | 3 | 3 | 50.146% |
| **Learned dynamic** | **2.4275** | **0.709** | **3** | **21.656%** |

- Dynamic minus static-K1 loss: `+0.0001` nat.
- Compute advantage over static K1: `+3.393%` of original OLMoE expert projections.
- Rank counts across all tokens and layers: `{'0': 34622, '1': 2271, '2': 4215, '3': 8044}`.
- Controller-only MLP overhead estimate: `0.263%` of original OLMoE expert projection MACs.

The controller sees the local normalized token state, router summaries, and statistics of mode 0 only. Additional modes are selected before their projections. The validation pass actually applies different ranks per token and layer; it is not an oracle recombination of independent uniform-rank passes.
