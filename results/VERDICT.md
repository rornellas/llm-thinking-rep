# Pre-Qwen gate verdict — audited status through routing-coupled v4

**Frozen decision:** **`NO_GO_FOR_OLMOE_OR_QWEN`**

## Current scientific state

The program has established five load-bearing conclusions:

1. the methodological harness, controls, sealed evaluation, replay, and independent audit work in the tested small-model scope;
2. global scalar expert-axis Modal compression is functionally possible but does not beat strong conventional narrowing at comparable dominant matrix arithmetic;
3. shared-base plus bilateral low-rank residuals offer a real parameter-efficiency signal, but uniform low rank does not preserve teacher behavior as well as narrow65;
4. expert-wise and routing-set objectives can move specific error components, yet changing the objective alone does not close the behavioral gap;
5. an explicit route-set moment correction has a statistically detectable causal effect, but the effect is too small and the moment-coupling form is insufficient.

## Confirmed foundation

- Synthetic methodological harness: **PASS**.
- Controlled conventional-MoE transplant: **FUNCTIONAL_ONLY**.
- Exact replay and independent frozen-result audits: **PASS**.
- Important-claim double-check and adversarial review: **MANDATORY**.
- Conventional teacher-informed narrow65 baseline: **PASS** in five fresh small teachers.
- Runtime speedup: **UNTESTED**.
- Teacher plateau: **NOT_DEMONSTRATED**.

## Strong conventional baseline

```text
narrow teacher-informed 65%
expert parameters/full:       65.000%
routed matrix compute/full:   65.000%
hypothesis delta loss:        +0.01325 nat
crossed UCB95:                +0.01914 nat
65% minus 50% mean:           -0.01648 nat
65% minus 50% IC95:           [-0.02611, -0.00570]
```

This remains the minimum strong comparator for shared architectures. It is conventional, small-scale, synthetic, and has no measured runtime claim.

## Alignment-tolerant sequence

### v1 — aggregate-output objective

```text
architecture: shared base + bilateral expert low-rank residual
primary: rank 5
parameters: 41.67%
compute proxy: 58.33%
verdict: ALIGNMENT_TOLERANT_SHARED_LORA_FAIL
```

Absolute CE gates passed, but non-inferiority to narrow65 was not established and behavioral fidelity was much worse.

### v2 — expert-wise objective

```text
verdict: ALIGNMENT_TOLERANT_EXPERT_DISTILL_V2_FAIL
```

Expert NRMSE improved strongly, but favorable cross-expert error cancellation was lost. The final mixture and behavioral fidelity did not improve sufficiently.

### v3.1 — routing-set counterfactual objective

```text
verdict: ALIGNMENT_TOLERANT_ROUTING_SET_V3_MECHANISM_SIGNAL
adversarial disposition:
OBJECTIVE_ONLY_LINE_CLOSED__ARCHITECTURE_CHANGE_REQUIRED
```

KL and cross-error improved relative to v2, but KL, top-1, local NRMSE, and counterfactual NRMSE still failed by large margins. The automatic mechanism signal was preserved, while an identical replication was rejected.

### v4 — explicit route-set moment correction

Primary architecture:

```text
rank-5 shared low-rank base
+ expert-aligned routed latent pooling
+ weighted first/second moments
+ zero-initialized shared correction MLP
```

Budget:

```text
expert parameters/full:       44.2824%
routed matrix compute/full:   62.5000%
```

Automatic verdict:

```text
ROUTING_COUPLED_V4_FAIL
```

Adversarial disposition:

```text
ROUTING_COUPLED_V4_CAUSAL_BUT_INSUFFICIENT__MOMENT_COUPLING_CLOSED
```

The v4 primary improved three behavioral endpoints relative to frozen v3:

```text
v4 - v3 KL:       -0.034643  95% [-0.049048, -0.021484]
v4 - v3 top-1:    +0.015278  95% [+0.005139, +0.024792]
v4 - v3 local:    -0.005396  95% [-0.007168, -0.003669]
```

The post-training ablation established a small causal KL contribution:

```text
coupling-disabled - primary KL:
mean  +0.012102
95%   [+0.007258, +0.017261]
```

The effect failed the preregistered minimum-magnitude gate `LCB95 >= +0.010`. The second moment did not beat the equal-budget mean-only control, and q12 did not rescue the architecture.

The gap to narrow65 remained large:

```text
v4 - narrow65 routing aggregate error:
mean  +0.117240
95%   [+0.104094, +0.132075]
```

Absolute v4 behavioral endpoints:

```text
KL:                  0.38644
Top-1:              67.65%
local NRMSE:         0.36517
counterfactual:      0.46091
routing aggregate:   0.15158
```

Cross-entropy was significantly better than narrow65 on the fresh small corpus, but this is not teacher-behavior preservation and cannot support a general superiority claim.

## Closed lines

The following are not approved as the next primary experiment:

- increasing global scalar K;
- selecting only existing teacher coordinates at 35%;
- conditional backward pruning of existing coordinates;
- static router-semantic basis clusters;
- another round of objective-only tuning for the same rank-5 family;
- another first/second-moment route-set correction with only larger pooling dimension.

## Next falsifiable experiment

The next architecture reallocates low-rank capacity per token under a strict total-rank budget:

```text
max rank 7 per expert, active rank sum 20
stored expert parameters: 55%
projected routed matrix compute: 58.33%

max rank 8 per expert, active rank sum 24
stored expert parameters: 61.67%
projected routed matrix compute: 65%
```

The active prefix allocated to each routed expert will depend on router weight and train-only marginal mode utility. Prefixes will be trained under multiple budgets so smaller prefixes remain useful.

Required comparators:

- uniform rank-5;
- uniform rank-6;
- narrow65;
- full control;
- routing-weight-only allocation control;
- static heterogeneous allocation control.

If budgeted heterogeneous rank does not move the quality-compute frontier, post-hoc alignment-tolerant conversion will be deprioritized and native constrained training from early checkpoints becomes the primary research branch.

## Decision

No tested candidate is eligible for OLMoE or Qwen scale-up. There is evidence for reusable structure and for small causal route-set corrections, but no architecture yet combines behavioral fidelity with a decisive compute advantage.

```text
NO_GO_FOR_OLMOE_OR_QWEN
```
