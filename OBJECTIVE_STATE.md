# Objective state — 2026-09-04

## Objective

Demonstrate a reproducible quality/parameters/memory/compute/latency advantage against strong relevant alternatives, eventually on real language models and affordable existing hardware. Do not confuse compression of an overparameterized in-house checkpoint with an improved capacity frontier or architectural novelty.

## Completed and verified

- MUI-1: restoring residual stable rank did not consistently improve language modeling. Original FAIL remains.
- FA-1: functional ablations on36 frozen checkpoints;112 actual exports. Removing/averaging experts failed fidelity screens. Per-expert residual rank8->rank1 passed fidelity/storage screens for both primary cohorts. Known calibration only.
- FCC-1: fresh article confirmation PASS for the eight primary compact checkpoints.95552->47168 total parameters (-50.6363%). Mean delta NLL +0.0000798154 (800-update cohort), +0.0005794240 (2200-update cohort). Both seed and crossed seed/article upper confidence limits passed frozen NLL/KL margins.256 true articles,1024 windows,65536 prediction tokens/model. Independent subset model reexecution and three arithmetic paths passed.
- ECK-1: FAIL_INVALID_FOR_SPEEDUP_CLAIM. Synthetic algebra/gradient tests passed but two jobs failed trained-model FP32 parity. All cells preserved. A posthoc trace reproduced a near-tied top-k expert swap; forcing original routing reduced logit error0.0566767->0.000001431. No tolerance change, no selective speedup claim.
- Data-unit correction: older 'article' IDs include subsections. FA-1 seed-conditional uncertainty is unchanged. FCC-1 uses actual top-level article boundaries.

## Current validated claim

Post-training rank-one residual compression of THESE small compact checkpoints preserves next-token distribution fidelity within the predeclared margins on the FCC-1 fresh English-Wikipedia sample while reducing parameter storage. The validated inference implementation is the ORIGINAL loop path, not the failed vectorized replacement.

## Unmet requirements / do not claim

- No demonstrated superiority to parameter-matched dense models.
- No certified inference speedup; no GPU, serving, peak-memory or energy claim.
- No demonstrated reasoning/tool-use/coding capability retention.
- No convergence, large-model transfer or novel-architecture claim.
- NO_GO_FOR_OLMOE_OR_QWEN remains unchanged.

## Next discriminating work, not yet executed

Compare a simple dense SwiGLU width104 (47168 total parameters) and width76 (matched14592 expert-matrix MACs/token) against native rank1 and the existing rank8->rank1 procedure, keeping non-MoE architecture/data budgets controlled. Register the complete protocol, new seeds, endpoints and stop rules before training. Report training and inference budgets separately. Dense controls are mandatory before scaling; faster execution of the old loop alone would not establish the scientific objective.

A future vectorized-kernel repair requires a new protocol and end-to-end checks around routing boundaries. Do not relax ECK-1 after observing failure. No dynamic-rank controller is justified by the current evidence.

## Data exposure

FCC-1 articles are now revealed. Do not optimize with them and call a later evaluation confirmatory. New confirmation requires disjoint fresh articles. The immutable FCC-1 source snapshot is5f9a18b8615def758528930b2d2cbb0b67a7154b; workflow33921582740.

## Entry points

- `docs/audits/2026-09-04-compression-functional-review.md`
- `docs/results/2026-09-04-fresh-compression-check-1.md`
- `results/fresh-compression-check-1/summary.json`
- `results/fresh-compression-check-1/external-archive-audit.json`
- `results/functional-ablation-1/finalization.json`
- `results/exact-compact-kernel-1/failure-summary.json`
- `results/exact-compact-kernel-1/failure-diagnostic.json`

The experiments above are complete. No dense-baseline training or hyperparameter search was started in this round.
