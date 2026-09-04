# Mode Utilization Intervention 1 (MUI-1)

Date: 2026-09-04. Status: prospective exploratory intervention; NOT a confirmatory gate.
Parent source: f7be4cf3dd7890f4b9643fc4cf5d4f7a60f836eb.
All earlier FAIL decisions and NO_GO_FOR_OLMOE_OR_QWEN remain unchanged.

## Question and rationale

Native Compact Gate 2A lost to narrow65. Posthoc weight diagnostics found residual stable rank near 1 and effective expert cosine near 0.99. These observations do not identify causality or prove functional equivalence. The implemented residual initializer draws BOTH factors with standard deviation 0.02/sqrt(r). For independent entries, Var((LR)_ij) = 0.02^4/r. Common matrices use Xavier initialization. This creates a very small initial expert-specific signal. Small-initialization rank bias is known in matrix factorization; extending that explanation to this nonlinear MoE with AdamW is a hypothesis, not a theorem.

Test interventions on initialization, not a new architecture, regularizer, controller, or posthoc rank allocation. No novelty claim for orthogonal initialization or low-rank factorization.

## Frozen design

Small geometry and optimizer from configs/native_compact_gate_2a.yaml: d_model=32, d_ff=64, 12 experts, top-4, 2 layers, sequence 64, batch 8, AdamW lr=0.0003, weight_decay=0.02, aux=0.01, clip=1.0. Rank=8, narrow width=42.
Seeds: [904031, 904043, 904051, 904073]. Exactly 800 updates per arm and identical training batches within each seed. No early stopping or selection of best checkpoint. Diagnostics at 0, 200, 400, 800. Training cost is equal updates/tokens, NOT equal FLOPs/time. The short budget does not reproduce the previous 2200-update gate and cannot certify mature training.

Six arms:
1. conventional-full (unchanged).
2. conventional-narrow65 (unchanged).
3. legacy: unchanged native-shared-rank initialization.
4. spectral-tiny: flatten residual nonzero singular values, preserving each original residual's Frobenius norm and left/right subspaces; use balanced SVD factors. Common unchanged.
5. energy-gaussian: same Gaussian factors/directions as legacy, rescaled equally on both sides so each residual has norm ||original common||/sqrt(2). Common divided by sqrt(2).
6. energy-spectral: same as arm 5 but flatten the residual spectrum while preserving its norm and subspaces, with balanced factors.

The energy intervention reallocates expected matrix energy from shared to expert-specific weights; it is not a pure factor-scale intervention. The spectral intervention changes both spectrum and factor gauge/conditioning; these are not separately identified. Identical non-MoE/router/common draws are paired. All compact arms have identical parameter and analytical inference MAC budgets. Finite-sample cross terms mean effective matrix norms need not be identical; measure rather than assume them.

## Data and access restriction

Use committed data/native-compact-gate-2a, loading ONLY train and validation. Train uses the same bounded 700000 tokens and article-preserving construction as Gate 2A. Evaluation uses the already revealed calibration split, one fixed window/article, evaluation seed 904091. It is NOT fresh heldout evidence. Do not load test or OOD arrays. This deliberate development screen is not allowed to spend another holdout on unvalidated mechanisms. Record selected document IDs, starts, losses, source/data hashes and identical batch-stream hashes.

## Endpoints and prespecified interpretation

Primary mechanistic contrasts: energy-spectral minus legacy; energy-gaussian minus legacy; energy-spectral minus energy-gaussian; spectral-tiny minus legacy. Also record factorial interaction and both energy arms vs narrow65. Primary functional endpoint: final calibration negative log-likelihood, lower is better. Primary rank endpoint: mean residual stable rank, ||R||_F^2 / ||R||_2^2, across projections/experts/layers. Stable rank is NOT algebraic rank.

Report each seed, paired means, 95% Student-t intervals across independent seeds (df=3), and descriptive document-level records. Seed-only uncertainty is conditional on this fixed calibration corpus and does not establish document/domain generalization. All contrasts are exploratory; do not cherry-pick an arm or claim confirmatory significance from multiple contrasts.

A candidate is promising for a NEW longer protocol only if: mean final loss vs legacy <= -0.010 nat; no seed worsens vs legacy; mean stable rank at least twice legacy; and upper one-sided t95 bound vs narrow65 <= +0.010 nat. The last condition is a screen, not a noninferiority certification. If rank improves without loss improvement, reject the claim that rank restoration alone is sufficient. If energy-gaussian matches/beats energy-spectral, do not attribute benefits uniquely to orthogonality. If all fail, report failure and stop; do not tune using these results in this protocol.

Secondary diagnostics: effective matrix cosine and norm; all-expert output diversity on fixed synthetic Gaussian probes (not a language-quality metric); terminal training loss; initial gradient norms; routing health on calibration in every layer; analytical parameter/MAC accounting; measured CPU end-to-end inference at batch 1, lengths 1 and 64, with warmup and randomized repeated timing blocks. Timings are implementation/hardware-specific, no GPU speedup claim. Initialization SVD is setup-only; no runtime reconstruction.

## Integrity and execution

Preflight tests: algebraic direct-vs-materialized equivalence for all initialization arms, invariant compact parameter counts, matched non-MoE state, spectral energy/rank properties, deterministic seeds. Failed preflight invalidates the run. Source and protocol must be committed before training. Preserve raw per-seed outputs, final checkpoints, environment, hashes, logs, timings, and an independently implemented arithmetic audit. An arithmetic audit is not an independent scientific replication or another researcher's review. Use direct commits to main, no PR. Four standard CPU jobs; 25-minute job cap, no paid GPU. Do not start another scientific run automatically after seeing results.

## Relevant prior art

Li, Luo & Lyu, ICLR 2021, Towards Resolving the Implicit Bias of Gradient Descent for Matrix Factorization: Greedy Low-Rank Learning, https://arxiv.org/abs/2012.09839 . Scope: factorization/gradient-flow theory, not this MoE.
Jin et al., ICML 2023, Understanding Incremental Learning of Gradient Descent: A Fine-grained Analysis of Matrix Sensing, https://arxiv.org/abs/2301.11500 . Same extrapolation restriction.
The experiment is a project-specific causal/development test, not evidence of a new publishable architecture.
