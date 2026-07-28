# Pre-Qwen gate verdict — audited status through alignment-tolerant v1

**Frozen decision:** **NO_GO_FOR_OLMOE_OR_QWEN**

## Current scientific state

The research has established three distinct results:

1. the methodological harness and controlled transplant machinery work in the tested small-model scope;
2. global scalar expert-axis Modal compression is functionally possible but does not beat strong conventional narrowing at comparable dominant matrix arithmetic;
3. an alignment-tolerant shared-base plus bilateral low-rank family is materially more promising, but its first preregistered screen did not establish non-inferiority to the frozen narrow65 baseline and showed substantially worse distributional fidelity.

## Confirmed foundation

- Synthetic methodological harness: **PASS**.
- Controlled conventional-MoE transplant: **FUNCTIONAL_ONLY**.
- Exact replay and independent frozen-result audits: **PASS**.
- Important-claim double-check and adversarial review: **MANDATORY**.
- Conventional teacher-informed narrow65 baseline: **PASS** in five fresh small teachers.
- Alignment-tolerant v1 independent audit: **PASS**, zero mismatches.
- Runtime speedup: **UNTESTED**.

## Scalar and residual FAIL-resolution results

| Hypothesis | Result |
|---|---|
| More local optimization | insufficient by itself |
| Closed-loop refinement | helps, but does not survive longer-budget teachers |
| Higher global scalar K | near-fidelity only at non-compute-reducing K7–K8 |
| Existing-coordinate selection at 35% | fails absolute fidelity |
| Conditional iterative pruning at 35% | fails and is unstable |
| Expert-specific low-rank residuals around scalar Modal | parameter-efficiency signal; equal-compute gate fails |
| Router-semantic clustered bases | equal-compute gate fails |

The historical `mature-*` teachers were still improving at their final 900-step checkpoint. They are longer-budget, non-converged teachers; no asymptotic claim is supported.

## Strong conventional baseline

The five-seed teacher-width replication established:

```text
narrow teacher-informed 65%
expert parameters/full:       65.000%
routed matrix compute/full:   65.000%
hypothesis Δ loss:            +0.01325 nat
crossed UCB95:                +0.01914 nat
65% minus 50% mean:           -0.01648 nat
65% minus 50% IC95:           [-0.02611, -0.00570]
```

This is the current minimum Pareto baseline for shared architectures. The result is conventional, small-scale, synthetic, and does not include measured runtime.

## Alignment-tolerant shared low-rank v1

The new family represents each expert matrix as:

\[
W_e = W_{shared} + L_e R_e.
\]

The preregistered rank-5 primary used:

```text
expert parameters/full:       41.6667%
routed matrix compute/full:   58.3333%
```

It passed every absolute fidelity, per-seed, budget, arithmetic, clean-data, control, and independent-audit gate. The load-bearing paired comparison did not pass:

```text
rank5 hypothesis Δ loss:      -0.00764 nat
rank5 hypothesis UCB95:       +0.02589 nat
rank5 minus narrow65 mean:    -0.00376 nat
rank5 minus narrow65 IC95:    [-0.03058, +0.02035]
```

The point estimate favored rank-5, but the interval includes inferior outcomes. Therefore non-inferiority to narrow65 is **NOT_ESTABLISHED**.

Behavioral fidelity was materially worse:

```text
                         rank5       narrow65
KL hypothesis:           0.37658     0.11450
Top-1 agreement:         68.44%      83.34%
local NRMSE hypothesis:  0.34904     0.17163
```

The correct v1 verdict is:

```text
ALIGNMENT_TOLERANT_SHARED_LORA_FAIL
```

This is a strict preregistered FAIL, not an architectural collapse. The family shows a real parameter/compute signal, but aggregate-output distillation may permit expert errors to cancel for observed routing mixtures.

## Next falsifiable experiment

The next screen must keep the rank-5 inference budget fixed and test a materially different training objective:

- explicit weighted loss for each routed expert output;
- aggregate layer-output loss;
- stronger closed-loop KL;
- fresh hypothesis and OOD documents;
- frozen v1 rank-5 and frozen narrow65 comparators;
- behavioral gates for KL, top-1 agreement, and local NRMSE;
- independent recalculation before accepting the decision.

A positive mechanism screen would justify fresh replication on teachers with explicit plateau. It would not authorize OLMoE/Qwen until quality survives larger plateaued teachers and analytical savings become measured runtime savings.

## Decision

No tested candidate is eligible for OLMoE or Qwen scale-up. The current evidence supports:

- a representational bottleneck in global scalar expert-axis Modal compression;
- a promising but unconfirmed alignment-tolerant factorization signal;
- the need to reduce expert-level functional mismatch before increasing statistical power or scale.

```text
NO_GO_FOR_OLMOE_OR_QWEN
```
