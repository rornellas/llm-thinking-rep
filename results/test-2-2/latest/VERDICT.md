# Test 2.2 — neuron-wise modal codes trained from initialization

**Decision:** **PASS_K1**

| Variant | Expert params | Ideal expert compute | Validation loss | Loss/full | Parameters |
|---|---:|---:|---:|---:|---:|
| baseline | 100.00% | 100.00% | 2.3878 | 1.000× | 1,270,560 |
| scalar-k1 | 12.51% | 50.01% | 2.4336 | 1.019× | 238,464 |
| scalar-k2 | 18.77% | 75.02% | 2.4097 | 1.009× | 312,288 |
| neuronwise-k1 | 13.54% | 51.04% | 2.4189 | 1.013× | 250,656 |
| neuronwise-k2 | 20.83% | 77.08% | 2.4182 | 1.013× | 336,672 |

Projected OLMoE K=1 budget: **3.174% parameters**, **25.049% ideal expert compute**.
Projected OLMoE K=2 budget: **4.785% parameters**, **37.598% ideal expert compute**.

The extra expressivity comes from elementwise expert codes; the number of shared matrix multiplications is unchanged. The down branch remains aggregated before its shared matrices.
