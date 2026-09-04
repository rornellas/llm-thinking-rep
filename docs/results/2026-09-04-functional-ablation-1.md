# FA-1 — Functional ablation of frozen small-model checkpoints

**Scope:** posthoc diagnostic, known calibration, four seeds per cohort; 800 and 2200 updates are different seed cohorts, not a paired learning curve.

**Screen:** `DEVELOPMENT_COMPRESSION_SIGNAL_PENDING_REEXECUTION`. All historical gates remain unchanged.

36 checkpoints; 112 real exports; 150 windows from 142 legacy segment IDs, not 142 independent top-level articles.

## Primary targets

| Cohort | Intervention | NLL delta | One-sided t95 upper | KL | Parameters | Latency ratio L1 / L64 |
|---|---|---:|---:|---:|---:|---|
| mui1 | original | +0.000000 | +0.000000 | 0.000000 | 95552 | 1.000 / 1.000 |
| mui1 | common-only | +0.042006 | +0.062855 | 0.021243 | 39488 | 0.239 / 0.205 |
| mui1 | mean-matrices | +0.030953 | +0.050305 | 0.016566 | 39488 | 0.241 / 0.203 |
| mui1 | rank1 | +0.000049 | +0.000099 | 0.000013 | 47168 | 0.989 / 0.914 |
| mui1 | permutation-average | +0.041334 | +0.069072 | 0.021990 | — | — |
| mui1 | uniform-selected | +0.000682 | +0.001073 | 0.000165 | 95552 | — |
| gate2a | original | +0.000000 | +0.000000 | 0.000000 | 95552 | 1.000 / 1.000 |
| gate2a | common-only | +0.053459 | +0.059371 | 0.040863 | 39488 | 0.241 / 0.186 |
| gate2a | mean-matrices | +0.030231 | +0.034643 | 0.023678 | 39488 | 0.242 / 0.187 |
| gate2a | rank1 | +0.000630 | +0.001684 | 0.000262 | 47168 | 0.989 / 0.910 |
| gate2a | permutation-average | +0.047767 | +0.057879 | 0.037757 | — | — |
| gate2a | uniform-selected | +0.001679 | +0.001986 | 0.001103 | 95552 | — |

## All controls: mean matrices and permuted routing

| Cohort / arm | Original NLL | Mean-matrix delta | Permutation-average delta |
|---|---:|---:|---:|
| gate2a/conventional-full | 4.447847 | +0.928684 | +0.377382 |
| gate2a/conventional-narrow65 | 4.484289 | +0.862095 | +0.370120 |
| gate2a/native-shared-rank | 4.502323 | +0.030231 | +0.047767 |
| mui1/conventional-full | 5.100159 | +0.216628 | +0.159834 |
| mui1/conventional-narrow65 | 5.181494 | +0.148813 | +0.100819 |
| mui1/energy-gaussian | 5.144572 | +0.084743 | +0.076236 |
| mui1/energy-spectral | 5.153024 | +0.079525 | +0.074577 |
| mui1/legacy | 5.108268 | +0.030953 | +0.041334 |
| mui1/spectral-tiny | 5.105133 | +0.030210 | +0.042192 |

## Interpretation limits

Intervals describe training-seed variation conditional on exposed calibration windows, not domain or article uncertainty. All analyses are exploratory and not multiplicity-adjusted.
Mean matrices are not mean nonlinear outputs. Cyclic permutations alter assignments without separately isolating per-expert load effects; upstream interventions may alter later natural routes.
Runtime is CPU, two threads, batch one, whole-prefix forwards, no KV cache. Every timed model omits auxiliary training losses. Parameter count excludes discarded weights and counts the tied embedding once.
SVD truncation/averaging are conventional methods. Passing this screen is not evidence of a novel architecture or a broad LLM Pareto improvement.
The numeric audit verifies paired raw metrics and export hashes. Independent checkpoint reexecution remains pending and is required before elevating the signal.

## Finalization audit

Independent materialized all-expert forwards and Gram-eigenvector rank-one reconstruction: **PASS** on two primary checkpoints, all 16 interventions. Export storage and raw arithmetic were also checked. See `results/functional-ablation-1/independent-audit.json` and `finalization.json`. The pending label above records the pre-audit aggregation state; the audit is now complete. No heldout generalization or novel-architecture claim is added.

## Data-unit correction (2026-09-04)

The legacy splitter treated subsection headings as document boundaries. The raw `unique_articles` field counts segment IDs. Seed-only confidence intervals above are unchanged because they were conditional on fixed windows. This correction must not be used to assume article independence in older analyses. FCC-1 uses top-level article boundaries. See `data-unit-correction.json`; raw metrics are retained unchanged.
