# Test 5.3 — causal context dependency of compact units

**Decision:** **COMPACT_CONTEXT_INCONCLUSIVE**

Unconditional train-frequency baseline: `4.6086` BpB.

| Representation | Intervention | BpB | Δ vs correct | Worst paired LCB95 | Context bytes/s |
|---|---|---:|---:|---:|---:|
| fixed-mean32 | correct | 4.6245 ± 0.0102 | +0.0000 | +0.0000 | 930200 |
| fixed-mean32 | rolled | 4.6369 ± 0.0118 | +0.0124 | +0.0053 | 933625 |
| fixed-mean32 | shuffle | 4.6306 ± 0.0124 | +0.0061 | -0.0005 | 875978 |
| fixed-mean32 | zero | 4.6315 ± 0.0127 | +0.0070 | -0.0006 | 917318 |
| fixed-mean32 | last64 | 4.6246 ± 0.0102 | +0.0001 | -0.0005 | 896569 |
| fixed-mean32 | last32 | 4.6244 ± 0.0104 | -0.0001 | -0.0016 | 922427 |
| fixed-mean32 | reverse | 4.6360 ± 0.0142 | +0.0115 | +0.0041 | 924090 |
| resampler16 | correct | 4.6303 ± 0.0139 | +0.0000 | +0.0000 | 738017 |
| resampler16 | rolled | 4.6306 ± 0.0142 | +0.0004 | +0.0000 | 737247 |
| resampler16 | shuffle | 4.6303 ± 0.0139 | -0.0000 | -0.0000 | 710962 |
| resampler16 | zero | 4.6308 ± 0.0145 | +0.0006 | -0.0000 | 734903 |
| resampler16 | last64 | 4.6304 ± 0.0141 | +0.0002 | -0.0002 | 739996 |
| resampler16 | last32 | 4.6305 ± 0.0142 | +0.0003 | -0.0000 | 738668 |
| resampler16 | reverse | 4.6303 ± 0.0139 | -0.0000 | -0.0000 | 737803 |

## Per-representation decision
- **fixed-mean32: WEAK_CONTEXT_SIGNAL** — rolled `+0.0124` BpB, shuffle `+0.0061`, last32 `-0.0001`, rolled worst LCB95 `+0.0053`.
- **resampler16: CONTEXT_NOT_ESTABLISHED** — rolled `+0.0004` BpB, shuffle `-0.0000`, last32 `+0.0003`, rolled worst LCB95 `+0.0000`.

All interventions preserve the held-out targets. Rolled context is the primary causal control because it remains in-distribution while breaking the context-target relationship.
