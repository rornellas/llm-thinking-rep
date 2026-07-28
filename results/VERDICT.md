# Pre-Qwen gate verdict — audited status after FAIL-resolution screens

**Frozen decision:** **NO_GO_FOR_OLMOE_OR_QWEN**

## Confirmed foundation

- Synthetic methodological harness: **PASS**.
- Controlled conventional-MoE transplant: **FUNCTIONAL_ONLY**.
- Parameter compression and expert-code causality: **PASS** in the frozen v1.3 scope.
- Exact replay and independent frozen-result audit: **PASS**.
- Independent aggregate audit of follow-up screens: **PASS**.
- Important-claim double-check and adversarial review: **MANDATORY**.

## Corrected interpretation

The historical directories named `mature-*` contain 900-step teachers that were still improving materially at the final checkpoint. They are longer-budget, non-converged teachers. This weakens claims about asymptotic geometry, but does not rescue any tested candidate: every development gate remains failed.

## Tested FAIL hypotheses

| Hypothesis | Result |
|---|---|
| More local optimization | insufficient by itself |
| Closed-loop refinement | helps substantially, but does not survive longer-budget teachers |
| Higher global scalar K | near-fidelity only at non-compute-reducing K7–K8 |
| Neuron-wise scalar codes | no established benefit in the initial screen |
| Expert-specific low-rank residuals | parameter-efficiency signal; equal-compute gate fails |
| Router-semantic clustered bases | no parameter advantage; equal-compute gate fails |

## Load-bearing longer-budget results

```text
scalar K2 closed-loop:
  parameters/full:       25.208%
  adjusted compute/full: 83.333%
  Δ loss:                +0.05607 nat; UCB95 +0.08135

best compute-reducing scalar rank (K2, top-4):
  Δ loss:                +0.09665 nat; UCB95 +0.13612

residual K1/R3 annealed:
  parameters/full:       36.771%
  adjusted compute/full: 74.167%
  Δ loss:                +0.04082 nat; UCB95 +0.05992
  vs parameter baseline: -0.01861 nat; UCB95 -0.00213
  vs matrix baseline:    +0.03464 nat; LCB95 +0.01416

clustered G2/R3 annealed:
  parameters/full:       36.667%
  adjusted compute/full: 70.000%
  Δ loss:                +0.05922 nat; UCB95 +0.07829
  vs parameter baseline: -0.00021 nat; UCB95 +0.01481
  vs matrix baseline:    +0.05304 nat; LCB95 +0.03074
```

## Decision

No tested scalar, residual, or clustered student is eligible for a fresh sealed replication, OLMoE, or Qwen. The evidence now supports a **representational bottleneck** in the global scalar expert-axis family, not merely an optimization failure.

The next experiment must use a materially different, alignment-tolerant function class—preferably expert-specific side factors with shared basis banks and activation/Fisher-weighted training—or train the compact constraint before specialization hardens. Gates must not be relaxed, the consumed v1.3 holdout must not be reused, and runtime remains untested.
