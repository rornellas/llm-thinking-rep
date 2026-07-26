# Modal MoE — aggregated geometric screen

Runs aggregated: **3**

> This verdict concerns only weight-space geometry. It does not establish functional preservation or speedup.

| Projection | Rank 90% | Rank 95% | Rank 99% | Common energy | Stable rank | Verdict |
|---|---:|---:|---:|---:|---:|---|
| down | 57.00 ± 0.00 | 60.00 ± 0.00 | 63.00 ± 0.00 | 1.56% ± 0.01% | 56.95 ± 0.15 | **FAIL** |
| gate | 56.00 ± 0.00 | 60.00 ± 0.00 | 63.00 ± 0.00 | 2.00% ± 0.01% | 50.11 ± 0.02 | **FAIL** |
| up | 56.00 ± 0.00 | 60.00 ± 0.00 | 63.00 ± 0.00 | 1.56% ± 0.00% | 56.25 ± 0.35 | **FAIL** |

## Interpretation rule

- `PASS`: mean rank-95 ≤ 8 and mean rank-99 ≤ 16.
- `BORDERLINE`: mean rank-95 ≤ 16 and mean rank-99 ≤ 32.
- `FAIL`: above those limits.

These thresholds are project triage gates. A `PASS` only authorizes the activation-aware experiment.
