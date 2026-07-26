# OLMoE layer 7 — raw weight-space PCA scout

**Date:** 2026-07-26  
**Workflow run:** `30221181344`  
**Status:** successful  
**Geometric screen:** **FAIL**

## Question tested

Can the 64 trained experts in layer 7 of `allenai/OLMoE-1B-7B-0924` be represented, in their current neuron coordinates, by a shared mean and a small number of global modes along the expert axis?

This is a weight-space geometric screen only. It does not test activation-aware reconstruction, neuron alignment, perplexity, task quality, or kernel speed.

## Reproducible setup

- Model revision: `3a970199d0f87db4e3e57275abb93812bf10fd83`
- Layer: zero-based index `7`
- Experts: all `64`
- Projections: `gate`, `up`, and `down`
- Sampled coordinates per matrix: `131,072 / 2,097,152` (`6.25%`)
- Independent seeds: `17`, `23`, `31`
- Weight payload read: `805,306,368` bytes (`768 MiB`)
- Gram matrices accumulated in FP64
- Validation suite: `8` tests passed before the real run

## Aggregate result

| Projection | Rank 90% | Rank 95% | Rank 99% | Common energy | Stable rank | Participation ratio | Screen |
|---|---:|---:|---:|---:|---:|---:|---|
| `down` | 57 | 60 | 63 | 1.5636% | 56.95 | 62.85 | **FAIL** |
| `gate` | 56 | 60 | 63 | 2.0030% | 50.11 | 62.20 | **FAIL** |
| `up` | 56 | 60 | 63 | 1.5638% | 56.25 | 62.79 | **FAIL** |

The maximum centered rank is 63. Requiring 60 modes for 95% of residual variance and 63 modes for 99% means the expert-axis spectrum is close to full dimensional.

## Low-rank region relevant to acceleration

At `K = 8` modes:

| Projection | Residual variance explained | Error relative to original matrix | Mean expert error | p95 expert error | Idealized parameter ratio |
|---|---:|---:|---:|---:|---:|
| `gate` | 15.04% | 91.25% | 89.36% | 99.18% | 14.06% |
| `up` | 13.75% | 92.14% | 91.00% | 99.19% | 14.06% |
| `down` | 13.60% | 92.22% | 91.28% | 99.12% | 14.06% |

This is not a near miss. In raw coordinates, eight global modes preserve only about 14–15% of the variation between experts.

At `K = 48`, the representation already consumes about 76.56% of the original parameter count, yet the matrix-level relative error remains roughly 45–47%. That point is neither a useful compression regime nor a plausible acceleration regime.

## Stability

All three independent coordinate samples returned exactly the same rank thresholds:

- `rank95 = 60` for all projections and all seeds;
- `rank99 = 63` for all projections and all seeds;
- standard deviations of the scalar spectrum metrics were very small.

The result is therefore not plausibly explained by an unlucky 6.25% coordinate sample. A full-coordinate PCA may refine decimals, but is unlikely to change the scientific decision.

## Interpretation

### What this result falsifies

It strongly rejects the direct hypothesis:

> In the experts' current neuron coordinates, a single global shared mean plus a few linear modes along the expert axis is sufficient.

The common mean contains only about 1.56–2.00% of total weight energy. For `up` and `down`, that is close to the `1/64` scale expected when averaging largely unaligned, weakly correlated expert matrices.

The participation ratios are close to the theoretical maximum of 63, which indicates a broad, nearly flat spectrum rather than a small set of dominant expert modes.

### What this result does not falsify

It does **not** reject:

1. scale-canonicalized experts;
2. joint neuron permutation alignment across `gate`, `up`, and `down`;
3. activation-aware or Fisher-weighted decompositions;
4. clustered bases or a global core plus exceptional experts;
5. nonlinear tensor decompositions;
6. functional compression even when raw weight distance is large.

A SwiGLU expert has exact hidden-neuron permutation symmetry, and `up`/`down` also have a scale freedom. Direct PCA before resolving those symmetries is deliberately stringent and can produce a false negative for functional similarity.

## Decision

Do **not** spend another run on exact full-coordinate PCA of the unaligned raw weights. The three-seed screen is sufficiently decisive.

Proceed to **Test 0.5: alignment and localization diagnostic**:

1. export per-expert norms, errors, leverage scores, and mode localization;
2. canonicalize the exact `up`/`down` scale freedom;
3. construct joint neuron signatures from `gate` rows, `up` rows, and corresponding `down` columns;
4. align a representative subset of experts using bipartite matching;
5. rerun the spectrum test after alignment;
6. continue to all 64 experts only if alignment materially changes the curve.

Operational continuation gates:

- `K=8` residual variance explained should rise materially above the current 14–15%; a useful early signal would be at least 40%;
- `rank95` should fall substantially below 60; a practical continuation threshold is at most 32;
- improvement must be stable across seeds and not depend on one reference expert.

If scale plus permutation alignment leaves the spectrum near its current shape, the global modal hypothesis should be stopped for this architecture and effort redirected to activation-aware clustering or another representation family.

## Repository outputs

Compact results are persisted under:

- `results/latest/aggregate/VERDICT.md`
- `results/latest/aggregate/projection_summary.csv`
- `results/latest/aggregate/rank_summary.csv`
- `results/latest/seed-17/`
- `results/latest/seed-23/`
- `results/latest/seed-31/`
- `results/latest/cloud-run-tail.log`
- `runs/latest.json`

The full GitHub Actions artifact for run `30221181344` is retained separately by GitHub Actions for seven days.
