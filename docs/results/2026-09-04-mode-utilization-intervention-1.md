# MUI-1 — Initialization intervention, 2026-09-04

**Scope:** prospective exploratory screen; reused calibration, four new seeds, 800 updates. Not a fresh holdout test or a repetition of Gate 2A.

**Verdict:** `NO_PROMISING_CANDIDATE_UNDER_FROZEN_SCREEN`. `NO_GO_FOR_OLMOE_OR_QWEN` unchanged.

| Arm | Calibration loss | Residual stable rank | Expert params/full | Expert MACs/full |
|---|---:|---:|---:|---:|
| conventional-full | 5.100159 | — | 100.00% | 100.00% |
| conventional-narrow65 | 5.181494 | — | 65.62% | 65.62% |
| legacy | 5.108268 | 1.007 | 45.83% | 62.50% |
| spectral-tiny | 5.105133 | 1.008 | 45.83% | 62.50% |
| energy-gaussian | 5.144572 | 3.653 | 45.83% | 62.50% |
| energy-spectral | 5.153024 | 6.670 | 45.83% | 62.50% |

## Frozen contrasts

| Contrast | Mean loss difference | Two-sided seed t95 interval |
|---|---:|---:|
| energy-spectral__minus__legacy | +0.044755 | [-0.097333, +0.186844] |
| energy-gaussian__minus__legacy | +0.036304 | [-0.032539, +0.105146] |
| energy-spectral__minus__energy-gaussian | +0.008452 | [-0.070763, +0.087667] |
| spectral-tiny__minus__legacy | -0.003136 | [-0.008028, +0.001756] |
| energy-spectral__minus__conventional-narrow65 | -0.028470 | [-0.074355, +0.017416] |
| energy-gaussian__minus__conventional-narrow65 | -0.036921 | [-0.073568, -0.000275] |
| legacy__minus__conventional-narrow65 | -0.073225 | [-0.175438, +0.028988] |
| factorial_interaction | +0.011587 | [-0.067780, +0.090955] |

Intervals quantify seed variation conditional on these windows. All contrasts are exploratory and not multiplicity-adjusted.

## Measured runtime (descriptive)

| Arm | Length 1, ms | Length 64, ms |
|---|---:|---:|
| conventional-full | 0.873 | 3.432 |
| conventional-narrow65 | 0.861 | 2.729 |
| legacy | 1.548 | 3.112 |
| spectral-tiny | 1.547 | 2.984 |
| energy-gaussian | 1.559 | 3.224 |
| energy-spectral | 1.551 | 2.894 |

Two CPU threads; batch 1; full forwards without KV cache. Medians of randomized repeated blocks, then median across runners. No GPU, cached-decode, energy, or production speed claim.

## Interpretation constraints

- Better residual stable rank alone is not better language modeling.
- A win over legacy alone is not a win over narrow65 or a new Pareto frontier.
- Energy intervention also changes common/expert energy allocation; spectral intervention also changes factor gauge.
- Synthetic output diversity is a mechanism probe, not proof of useful language specialization.
- Fixed short training may favor one learning trajectory. No mature-model or broad generalization claim.
- Lower matrix-MAC counts are not measured latency. The conventional kernel and compact kernel have different implementation efficiency.
- Arithmetic checks passed through separate implementations; not an independent scientific replication.

Full raw windows, histories, initial gradients, routing, timings, environment, data/source hashes and checkpoints are retained beside summary.json. No automatic follow-up run.
