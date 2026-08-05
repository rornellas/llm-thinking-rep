# Pre-Qwen gate verdict — audited status through Reality Gate 1A

**Frozen decision:** **`NO_GO_FOR_OLMOE_OR_QWEN`**

## Executive state

The research program has now established six load-bearing conclusions:

1. the methodological harness, positive/negative controls, sealed evaluation, replay, and independent audit work in the tested controlled scope;
2. global scalar expert-axis Modal compression is functionally possible but does not beat strong conventional narrowing at comparable dominant matrix arithmetic;
3. shared-base plus bilateral expert low-rank residuals provide a real parameter-efficiency signal, but post-hoc uniform low rank preserves teacher behavior substantially worse than `narrow65`;
4. expert-wise, routing-set, and explicit route-set coupling objectives move specific error components, but none closes the behavioral gap;
5. a first/second-moment route-set correction has a small causal KL effect, but is insufficient and that coupling line is closed;
6. static spectral heterogeneous rank did not improve over uniform rank in a two-scale WikiText-2 Reality Gate, usually collapsed exactly to uniform, and does not justify dynamic rank allocation.

## Confirmed foundation

- Synthetic methodological harness: **PASS**.
- Controlled conventional-MoE transplant: **FUNCTIONAL_ONLY**.
- Exact replay and independent frozen-result audits: **PASS**.
- Important-claim double-check and adversarial review: **MANDATORY**.
- Conventional teacher-informed `narrow65`: **STRONG BASELINE**.
- Measured runtime speedup: **UNTESTED**.
- Real-checkpoint transfer: **BLOCKED**.

## Historical negative-result sequence

| Line | Audited conclusion |
|---|---|
| Scalar global modes | fidelity approaches target only at non-compute-reducing rank |
| Existing-coordinate selection/pruning | fails absolute fidelity at 35% |
| Expert-specific residuals around scalar Modal | parameter signal, equal-compute gate fails |
| Router-semantic clustered bases | equal-compute gate fails |
| Shared-base bilateral low rank v1 | absolute CE signal, behavioral fidelity and narrow65 gate fail |
| Expert-wise objective v2 | individual expert error improves, final mixture does not |
| Routing-set objective v3.1 | KL/covariance mechanism signal, absolute behavior fails |
| Explicit moment coupling v4 | small causal KL effect, insufficient; moment coupling closed |
| Static spectral heterogeneous rank | no advantage over uniform; usually identical allocation |

## Reality Gate 1A

Protocol:

```text
reality-gate-1a-static-heterogeneous-rank-v1
```

Data and provenance:

```text
Salesforce/wikitext
subset: wikitext-2-raw-v1
revision: b08601e04326c79dfdd32d625aee71d232d685c3
four seeds × two scales
article-level statistical units
source commit: 456c3beb87eace9a196a46bd24ab33090b0bc2eb
workflow run: 31011802255
results commit: cd57ba42d69036f188b1bc067a895f98e72ef43e
independent audit: PASS, zero mismatches
```

Automatic verdict:

```text
REALITY_GATE_1A_FAIL
```

Adversarial disposition:

```text
STATIC_HETEROGENEITY_NOT_SUPPORTED
DYNAMIC_RANK_BLOCKED
POST_HOC_ALIGNMENT_TOLERANT_CONVERSION_DEPRIORITIZED
NATIVE_COMPACT_TRAINING_BECOMES_PRIMARY
```

### Plateau

None of the eight scientific teachers passed the frozen plateau criterion:

```text
small:  0 / 4
medium: 0 / 4
total:  0 / 8
```

Loss slopes remained approximately one order of magnitude more negative than the allowed thresholds, and routing distributions continued to move materially. Therefore the experiment does not support a mature-teacher universal refutation.

### Allocation behavior

At the final checkpoint:

```text
medium spectral allocation:
  exactly uniform in 4/4 seeds

small spectral allocation:
  exactly uniform in 3/4 seeds
  one minimal redistribution in 1/4 seeds

routing-only allocation:
  exactly uniform in all final cells
```

In medium, spectral, routing-only, and uniform candidates were therefore functionally identical. The rare small-scale deviation produced no established advantage.

### Compressibility trajectory

The residual rank required for 95% spectral energy remained high from 25% of training through the final checkpoint:

```text
small:  mean final rank 25.32 of 32, observed 24–26
medium: mean final rank 37.80 of 48, observed 37–39
```

The executed ranks were 8 and 12. Redistributing the same rank total cannot recover capacity when residual spectra are high-rank and similar across experts.

### Small scale

```text
spectral:
  parameters             45.83%
  compute proxy          62.50%
  hypothesis delta loss  +0.00604
  KL                     0.01453
  top-1                  84.51%
  local NRMSE            0.19035

spectral - uniform loss:
  mean +0.000087
  95%  [-0.000145, +0.000495]

spectral - narrow65 loss:
  mean +0.007460
  95%  [+0.005427, +0.009378]

spectral - narrow65 top-1:
  mean -0.071135
  95%  [-0.077942, -0.064773]
```

### Medium scale

```text
spectral:
  parameters             43.75%
  compute proxy          62.50%
  hypothesis delta loss  +0.00800
  KL                     0.02074
  top-1                  83.05%
  local NRMSE            0.21355

spectral - uniform:
  exactly 0 on all load-bearing endpoints

spectral - narrow65 loss:
  mean +0.011128
  95%  [+0.008783, +0.013631]

spectral - narrow65 top-1:
  mean -0.068028
  95%  [-0.073236, -0.062648]
```

`narrow65` used only approximately 2–3 percentage points more analytical compute than the low-rank family, but preserved teacher behavior substantially better in both scales.

## What is and is not refuted

Supported:

- the tested spectral and routing-frequency allocators did not create useful static heterogeneity;
- the current static mechanism does not justify a dynamic controller;
- post-hoc shared-base low-rank compression remains behind conventional narrowing in behavioral fidelity;
- the next primary line should impose compact structure during training.

Not supported:

- a universal claim that heterogeneous rank can never work;
- a mature-teacher asymptotic claim, because plateau was not reached;
- any runtime or hardware claim;
- any OLMoE/Qwen transfer claim.

## Closed or blocked lines

The following are no longer approved as the next primary experiment:

- increasing global scalar K;
- existing-coordinate selection or conditional backward pruning;
- router-semantic static basis clusters;
- another objective-only rank-5 distillation round;
- another first/second-moment route-set correction;
- static spectral rank redistribution under the Reality Gate 1A formulation;
- dynamic rank allocation built on that static utility;
- scale-up to OLMoE or Qwen;
- hardware specialization before a GPU-visible software advantage.

## Next primary experiment

The approved next line is **native compact training**. The architecture must be constrained from initialization, before conventional experts develop high-rank residuals that are expensive to compress post hoc.

Required comparison from scratch, with equal training data and controlled training compute:

1. conventional full MoE;
2. conventional `narrow65`;
3. shared-base plus expert-local factors trained natively;
4. shared-base plus nested prefixes trained across multiple budgets;
5. MoSE-like nested-width baseline;
6. RFID-like static heterogeneous-rank baseline;
7. the combined native shared-base + nested-prefix candidate.

Claims must be staged:

```text
A. native compact architecture moves quality-compute Pareto
B. smaller prefixes are independently functional
C. heterogeneous allocation adds value over uniform
D. dynamic per-token control adds value over static
E. analytical savings become measured GPU runtime savings
F. only then test a real MoE checkpoint
```

## Decision

No tested candidate combines behavioral fidelity, parameter efficiency, compute advantage, scale robustness, and measured runtime. The research remains scientifically healthy because negative evidence is changing the theory rather than being explained away.

```text
NO_GO_FOR_OLMOE_OR_QWEN
NO_DYNAMIC_RANK
NO_RUNTIME_CLAIM
NATIVE_COMPACT_TRAINING_PRIMARY
```
