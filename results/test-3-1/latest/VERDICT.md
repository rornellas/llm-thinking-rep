# Test 3.1 — assignment-supervised real OLMoE down distillation

**Decision:** **FAIL**

| Variant | Params | Compute | Best step | Train assignment error | Val assignment error | Val aggregate error | Aggregate cosine | Mean-code ablation | Shuffle ablation | Zero ablation |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| scalar-k0 | 1.562% | 12.500% | 500 | 182.34% | 187.86% | 174.92% | 0.14680 | n/a | n/a | n/a |
| scalar-k1 | 3.125% | 25.049% | 500 | 182.95% | 188.31% | 175.02% | 0.14740 | 1.000× | 1.001× | 1.000× |
| neuronwise-k1 | 3.174% | 25.049% | 500 | 184.46% | 189.14% | 176.40% | 0.14753 | 0.992× | 1.004× | 0.992× |
| neuronwise-k2 | 4.785% | 37.598% | 500 | 186.35% | 190.66% | 178.26% | 0.14526 | 0.986× | 1.004× | 0.985× |

Training uses the real per-assignment `D_e z_e` tensors as well as the weighted aggregate output. Output coordinates are randomly sampled during optimization; all 2,048 dimensions are used for held-out evaluation. The split is by token, so no assignment from a validation token appears in training.
