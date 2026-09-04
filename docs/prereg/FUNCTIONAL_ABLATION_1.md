# Functional Ablation 1 (FA-1)

Date: 2026-09-04. Prospective analysis plan for a POSTHOC DEVELOPMENT DIAGNOSTIC.
Registered before observing any ablation output. No new training, no fresh-holdout claim.
Earlier FAIL gates and NO_GO_FOR_OLMOE_OR_QWEN remain unchanged.

## Question

Do learned expert-specific residuals and input-dependent routing materially support next-token predictions? Can a simpler representation preserve function and improve actual storage/runtime? Geometric diversity alone is not a success endpoint.

## Frozen inputs

All final small-scale MUI-1 checkpoints: six arms x seeds [904031,904043,904051,904073], 800 updates. Also all final small-scale Native Compact Gate 2A checkpoints: conventional-full, conventional-narrow65, native-shared-rank x seeds [202781,212789,222793,232801], 2200 updates. The two cohorts differ in seeds and training length; they are NOT a paired learning curve and neither is declared converged. Never choose best-calibration checkpoints.

Calibration only: committed data/native-compact-gate-2a validation tokens, same bounded construction and exactly the MUI-1 seed 904091 / one-window-per-chunk selection (expected 150 windows x 64 tokens). No test or OOD arrays loaded. Calibration has been exposed; none of the following is confirmatory generalization evidence. Metrics weight windows equally; windows/chunks are not assumed independent articles. Uncertainty is across four independent training seeds, conditional on these fixed windows.

## Frozen interventions, all layers simultaneously

For every checkpoint:
- original: unchanged function, inference-only wrapper without training auxiliary losses.
- mean-matrices: replace the expert bank by ONE SwiGLU with uniform mean gate/up/down effective matrices. Remove router and expert parameters from the exported model. Mean matrices are NOT mean nonlinear outputs; this is an explicit intervention, not an algebraic equivalence claim.
- permute-1, permute-5, permute-7: compute natural top-k IDs and probabilities, replace each expert ID e by (e+offset) mod E, retaining slot weights. These bijections preserve cardinality and the multiset of marginal route loads up to expert relabeling, not each expert's original load. Upstream intervention can change downstream natural routes. Report offsets individually and their fixed average, never choose a convenient permutation.
- uniform-selected: natural top-k IDs but weights exactly 1/k.

For compact checkpoints additionally:
- common-only: one SwiGLU using common gate/up/down, no residuals or router.
- rank1: best Frobenius rank-1 approximation (SVD) of EACH learned residual, balanced factors, same common matrices and router, direct compact execution. No refitting or calibration-dependent rank allocation.

Both dense and low-rank exported models retain unchanged non-MoE weights. Parameter and byte counts must exclude discarded weights and count tied embeddings only once.

## Endpoints and interpretation fixed before results

For each arm/cohort: paired mean NLL difference to original, KL(original || intervention), token argmax agreement, per-seed values, two-sided t95 and one-sided upper t95 (df=3), plus every raw window. Report original NLL itself and verify reproduction against MUI-1 stored losses within 2e-5 nat; Gate 2A calibration windows differ so only wrapper parity is required there. Permutation-average is the mean of all three specified offsets, per seed.

The two PRIMARY compact targets are MUI-1 legacy and Gate 2A native-shared-rank. Other arms are controls, not candidates selected by their results. A simplification (rank1, common-only, mean-matrices) passes a DEVELOPMENT fidelity screen separately per cohort if NLL upper one-sided t95 <= +0.010 nat, no seed NLL delta > +0.025, and KL upper t95 <= 0.005 nat. It is a cross-budget development candidate only if it passes BOTH primary cohorts with >=25% fewer total parameters than each original. This does not unlock checkpoint-scale gates or prove broad quality.

For routing dependence, report permutation-average and uniform-selected loss changes; a positive loss delta is evidence conditional on that intervention, not evidence of an optimal learned assignment. Destructive mean-matrix changes in independently initialized conventional experts are controls and not proof they are incompressible after learning an alignment. Residual removal changing quality does not imply every residual direction is necessary.

## Timing and engineering controls

Inference wrapper omits training auxiliary computations for ALL methods and returns only logits. Check equality to original model before any scientific evaluation. Fixed CPU threads=2, batch=1, sequence lengths 1 and 64, no KV cache; 5 warmups, 11 randomized timing blocks, 10 forwards/block, fixed timing seed 904117. Benchmarks cover original and actual dense/rank1 exports, not permutation diagnostics. Report paired median ratio per checkpoint plus aggregate, hardware, environment, serialized bytes, unique parameter count and analytical expert matrix MACs. These are whole-prefix forwards, not autoregressive tokens/second or GPU/production claims. A latency benefit requires at least 10% reduction in both lengths in both primary cohorts; do not infer it from MACs.

Implementation changes without functional quality gain are reported as engineering gains only. Ordinary SVD truncation, matrix averaging and pruning are prior art, not a novel architecture.

## Integrity / stop rules

Before inspecting scientific outputs, test synthetic original-wrapper parity, permutation-plus-inverse-router reindex invariance, rank-1 SVD exactness for constructed rank-1 residuals, zero-residual common equivalence, and actual parameter removal. Deterministic evaluation; weights_only checkpoint loads. Hash input checkpoints, loaded arrays, source and protocol; serialize exported candidates and raw window results. Independent numeric recomputation and rerun of at least one full primary checkpoint outside the first runner are required before promoting any compression result. An independent code path is not an independent research group.

On failures of parity, checkpoint reproduction, finite metrics or artifact integrity, mark INVALID and fix engineering before interpreting results; preserve failed logs. Stop after fixed ablations. No automatic parameter search. Follow-up training requires a new committed plan informed by this diagnostic, with these calibration observations labeled exposed.

## Primary literature framing

Tian et al., Beyond Geometric Complementarity: Coherent Overlap in Sparse Mixture-of-Experts Routing, 2026-07-30, https://arxiv.org/abs/2607.28308 : reported geometric overlap does not alone establish pruning value. Not reproduced here.
Gu et al., Delta Decompression for MoE-based LLMs Compression, 2025, https://arxiv.org/abs/2502.17298 : shared components and low-rank residual compression precede this project.
This diagnostic aims at a falsifiable project decision, not a claim of publication-level novelty.
