# Test 2.5 — 64-expert/top-8 scaling and code ablations

**Decision:** **SCALE64_AND_CODES_USED**

| Variant | Expert params | Ideal expert compute | Validation loss | Loss/full | Router entropy | Mean-code ablation | Shuffle ablation | Zero-code ablation |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | 100.000% | 100.000% | 2.4310 | 1.000× | 0.954 | n/a | n/a | n/a |
| scalar-k1 | 3.133% | 25.008% | 2.4781 | 1.019× | 0.964 | 1.030× | 1.037× | 1.028× |
| neuronwise-k1 | 4.167% | 26.042% | 2.4775 | 1.019× | 0.967 | 1.018× | 1.022× | 1.020× |
| scalar-k2 | 4.704% | 37.516% | 2.4663 | 1.015× | 0.962 | 1.050× | 1.042× | 1.054× |
| neuronwise-k2 | 6.771% | 39.583% | 2.4682 | 1.015× | 0.963 | 1.033× | 1.031× | 1.036× |

An ablation penalty above 1.0 means expert-specific codes carry information beyond the common mode. This test matches OLMoE's 64-expert/top-8 geometry but remains a small character language model.
