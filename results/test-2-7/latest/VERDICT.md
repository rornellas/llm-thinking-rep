# Test 2.7 — learned causal rank controller

**Decision:** **BORDERLINE**

Chosen threshold: `0.65` (selected only on training-calibration batches).

| Policy | Validation loss | Mean K | p95 K | Projected OLMoE compute |
|---|---:|---:|---:|---:|
| Static K=0 | 2.4210 | 0 | 0 | 12.500% |
| Static K=1 | 2.4016 | 1 | 1 | 25.049% |
| Static K=2 | 2.3943 | 2 | 2 | 37.598% |
| Static K=3 | 2.3889 | 3 | 3 | 50.146% |
| **Learned dynamic** | **2.4126** | **0.420** | **3** | **18.035%** |

- Dynamic minus static-K1 loss: `+0.0110` nat.
- Compute advantage over static K1: `+7.014%` of original OLMoE expert projections.
- Rank counts across all tokens and layers: `{'0': 33602, '1': 1710, '2': 1447, '3': 4201}`.
- Controller-only MLP overhead estimate: `0.263%` of original OLMoE expert projection MACs.

The controller sees the local normalized token state, router summaries, and statistics of mode 0 only. Additional modes are selected before their projections. The validation pass actually applies different ranks per token and layer; it is not an oracle recombination of independent uniform-rank passes.
