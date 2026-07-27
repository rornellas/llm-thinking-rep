# Test 2.6 — nested progressive modes and token oracle

**Decision:** **PROGRESSIVE_PASS**

| Active K | Modes | Validation loss | Loss/full baseline | Loss/full prefix | Projected OLMoE params | Projected OLMoE compute |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 1 | 2.4373 | 1.037× | 1.011× | 1.562% | 12.500% |
| 1 | 2 | 2.4205 | 1.030× | 1.004× | 3.174% | 25.049% |
| 2 | 3 | 2.4125 | 1.027× | 1.000× | 4.785% | 37.598% |
| 3 | 4 | 2.4116 | 1.026× | 1.000× | 6.396% | 50.146% |

## Token-level oracle
- tolerance `0.02` nat: mean K `1.114`, p95 `3`, projected mean OLMoE compute `26.484%`, counts `{'0': 10265, '1': 2686, '2': 2451, '3': 5078}`.
- tolerance `0.05` nat: mean K `0.902`, p95 `3`, projected mean OLMoE compute `23.818%`, counts `{'0': 11617, '1': 2762, '2': 2593, '3': 3508}`.
- tolerance `0.10` nat: mean K `0.625`, p95 `3`, projected mean OLMoE compute `20.340%`, counts `{'0': 13554, '1': 2788, '2': 2407, '3': 1731}`.

The oracle compares per-token cross-entropy of nested prefixes from the same checkpoint. It is an upper bound on a learned controller and does not include controller or scheduling overhead.
