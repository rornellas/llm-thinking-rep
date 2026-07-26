# Test 0.7 — blockwise expert modes

**Decision:** **FAIL**

| Projection | Block | Mean K=4 explained | p10 K=4 explained | Mean rank90 | Mean rank95 | K90 compute ratio | K95 compute ratio |
|---|---:|---:|---:|---:|---:|---:|---:|
| gate | 16 | 13.49% | 13.09% | 48.01 | 54.37 | 612.65% | 692.09% |
| gate | 32 | 10.00% | 9.78% | 53.15 | 58.00 | 676.86% | 737.50% |
| gate | 64 | 8.54% | 8.39% | 55.00 | 59.00 | 700.00% | 750.00% |
| gate | 128 | 7.97% | 7.81% | 56.00 | 59.50 | 712.50% | 756.25% |
| up | 16 | 12.98% | 12.70% | 48.29 | 54.78 | 616.11% | 697.22% |
| up | 32 | 9.48% | 9.36% | 53.99 | 58.01 | 687.40% | 737.60% |
| up | 64 | 7.94% | 7.88% | 55.95 | 59.02 | 711.91% | 750.20% |
| up | 128 | 7.28% | 7.23% | 56.00 | 60.00 | 712.50% | 762.50% |
| down | 16 | 13.00% | 12.70% | 48.25 | 54.76 | 615.62% | 696.97% |
| down | 32 | 9.47% | 9.32% | 53.99 | 58.02 | 687.40% | 737.70% |
| down | 64 | 7.92% | 7.85% | 55.98 | 59.12 | 712.30% | 751.56% |
| down | 128 | 7.26% | 7.18% | 56.00 | 60.00 | 712.50% | 762.50% |

A compute ratio below 100% is necessary but not sufficient for a real kernel speedup.
PASS requires strong K=4 locality and a variable-rank 90% reconstruction budget below 90% of top-8 compute.
