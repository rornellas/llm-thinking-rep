# Test 6.2 — hard top-k event bottleneck

**Decision:** **HARD_EVENT_BOTTLENECK_PASS**

| Training | Evaluation | Loss | Δ own K3 | Δ control K3 | Density | Residual bits/BF16 | Advantage vs random |
|---|---|---:|---:|---:|---:|---:|---:|
| full-control | dense-k3 | 2.3514 | +0.0000 | +0.0000 | 100.000% | 100.000% | n/a |
| full-control | dense-k1 | 2.3877 | +0.0363 | +0.0363 | 100.000% | 100.000% | n/a |
| full-control | topk-90%-q4 | 2.3762 | +0.0248 | +0.0248 | 10.417% | 8.854% | +0.0242 |
| topk90-fp | dense-k3 | 2.3504 | +0.0000 | -0.0011 | 100.000% | 100.000% | n/a |
| topk90-fp | dense-k1 | 2.3700 | +0.0196 | +0.0186 | 100.000% | 100.000% | n/a |
| topk90-fp | topk-90%-qNone | 2.3563 | +0.0059 | +0.0049 | 10.417% | 100.000% | +0.0264 |
| topk90-q4 | dense-k3 | 2.3539 | +0.0000 | +0.0024 | 100.000% | 100.000% | n/a |
| topk90-q4 | dense-k1 | 2.3752 | +0.0214 | +0.0238 | 100.000% | 100.000% | n/a |
| topk90-q4 | topk-90%-q4 | 2.3597 | +0.0058 | +0.0082 | 10.417% | 8.854% | +0.0250 |
| topk95-q4 | dense-k3 | 2.3545 | +0.0000 | +0.0031 | 100.000% | 100.000% | n/a |
| topk95-q4 | dense-k1 | 2.3711 | +0.0166 | +0.0196 | 100.000% | 100.000% | n/a |
| topk95-q4 | topk-95%-q4 | 2.3643 | +0.0098 | +0.0128 | 5.208% | 7.552% | +0.0161 |

- Best: `topk95-q4` / `topk-95%-q4`.
- Δ versus own dense K3: `+0.0098` nat.
- Δ versus control dense K3: `+0.0128` nat.
- Residual traffic: `7.552%` of dense BF16 deltas.
- Advantage over random: `+0.0161` nat.
- Improvement over post-hoc control: `+0.0150` nat.

The top-k mask and fake quantizer are in the fine-tuning forward pass. Sorting/index selection cost is not included in the residual traffic ratio.
