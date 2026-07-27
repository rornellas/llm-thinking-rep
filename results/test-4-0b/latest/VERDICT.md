# Test 2.7 — learned causal rank controller

**Decision:** **LEARNED_CONTROLLER_PASS**

Chosen threshold: `0.60` (selected only on training-calibration batches).

| Policy | Validation loss | Mean K | p95 K | Projected OLMoE compute |
|---|---:|---:|---:|---:|
| Static K=0 | 5.8979 | 0 | 0 | 12.500% |
| Static K=1 | 5.8868 | 1 | 1 | 25.049% |
| Static K=2 | 5.8832 | 2 | 2 | 37.598% |
| Static K=3 | 5.8808 | 3 | 3 | 50.146% |
| **Learned dynamic** | **5.8899** | **0.622** | **3** | **20.566%** |

- Dynamic minus static-K1 loss: `+0.0031` nat.
- Compute advantage over static K1: `+4.483%` of original OLMoE expert projections.
- Rank counts across all tokens and layers: `{'0': 36282, '1': 2414, '2': 3218, '3': 7238}`.
- Controller-only MLP overhead estimate: `0.263%` of original OLMoE expert projection MACs.

The controller sees the local normalized token state, router summaries, and statistics of mode 0 only. Additional modes are selected before their projections. The validation pass actually applies different ranks per token and layer; it is not an oracle recombination of independent uniform-rank passes.
