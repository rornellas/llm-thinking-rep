# Alignment-tolerant expert-distillation v2

**Decision:** **ALIGNMENT_TOLERANT_EXPERT_DISTILL_V2_FAIL**

**Evidence status:** `PARTIAL_EVIDENCE_CHECKPOINT`

| Candidate | Params | Compute | Hyp Δ | Hyp UCB95 | OOD Δ | OOD UCB95 | KL hyp | Top-1 hyp | Local NRMSE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| shared-lora-r5 expert-distill v2 | 41.67% | 58.33% | -0.01735 | +0.01366 | -0.06017 | -0.00518 | 0.36538 | 68.34% | 0.33787 |
| narrow65 frozen baseline | 65.00% | 65.00% | — | — | — | — | 0.11489 | 82.50% | 0.16827 |

## Load-bearing comparisons

```text
rank5-v2 - narrow65, hypothesis Δ loss
mean: -0.01274 nat
95%:  [-0.05144, +0.02407]
```

```text
rank5-v2 - rank5-v1, expert NRMSE
mean: -0.30858
95%:  [-0.32293, -0.29416]
```

```text
rank5-v2 - narrow65, cross-error term
mean: +0.08946
95%:  [+0.08232, +0.09677]
```

## Gate interpretation

- Absolute hypothesis loss: pass.
- Absolute OOD loss: pass.
- Parameter budget below 65%: pass.
- Compute proxy below 65%: pass.
- Paired non-inferiority to narrow65: fail/inconclusive.
- KL preservation: fail.
- Top-1 preservation: fail.
- Local and per-expert functional fidelity: fail.
- Global decision remains `NO_GO_FOR_OLMOE_OR_QWEN`.

The expert-wise objective substantially reduces individual expert error but does not reproduce the favorable cross-expert error covariance of the conventional narrow65 baseline.

Raw per-seed checkpoints and full JSON records are not yet included in this remote checkpoint; see the analysis report for scope limitations.
