# Test 3.0 — real OLMoE layer-7 down-branch modal distillation

**Decision:** **FAIL**

| Variant | Params | Compression | Ideal compute | Initial error | Final val error | Mean cosine | p05 cosine | Mean-code ablation | Shuffle ablation | Zero-code ablation |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| scalar-k0 | 1.562% | 64.00× | 12.500% | 114.98% | 115.32% | 0.32505 | 0.12237 | n/a | n/a | n/a |
| scalar-k1 | 3.125% | 32.00× | 25.049% | 114.99% | 115.32% | 0.32506 | 0.12465 | 1.000× | 1.000× | 1.000× |
| neuronwise-k1 | 3.174% | 31.51× | 25.049% | 114.98% | 115.32% | 0.32487 | 0.12401 | 1.000× | 1.000× | 1.000× |
| neuronwise-k2 | 4.785% | 20.90× | 37.598% | 114.99% | 115.26% | 0.32530 | 0.12562 | 1.000× | 1.000× | 1.000× |

The target consists of real layer-7 Q4_K_M expert activations from the official OLMoE graph. Only tokens in the training split update the modal matrices and codes; validation tokens are disjoint. This test isolates `down`: teacher SwiGLU states and original router decisions are supplied to the student.
