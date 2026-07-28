# Pre-Qwen FAIL resolution — status after longer-budget-teacher tests

**Date:** 2026-07-28
**Scope:** exploratory hypothesis development after the frozen `controlled-small-moe-transplant-v1.3` decision.
**Frozen decision:** `NO_GO_FOR_OLMOE_OR_QWEN` remains unchanged.

## Executive result

The original scalar K2 failure contained two effects:

1. an optimization/objective mismatch that can be substantially reduced by longer local distillation and closed-loop refinement; and
2. a representation-capacity limit that reappears when the conventional teacher is trained more fully.

On the original 300-step controlled teachers, scalar K2 with 540 local steps plus 120 closed-loop steps reached exploratory parity. On new 900-step teachers, the same family remained materially behind the teacher and behind a matrix-compute-matched narrow conventional expert. Increasing scalar expert-axis rank reduced error only by spending more than the original expert compute. Adding expert-specific low-rank residuals improved the parameter/quality Pareto frontier but did not recover the equal-compute frontier.

## Claim-verification policy

From this point onward, important scientific and engineering claims follow `docs/methodology/IMPORTANT_CLAIM_VERIFICATION_STANDARD.md`. The minimum procedure is:

- narrow claim statement;
- raw evidence and hashes;
- independent numerical recomputation;
- adversarial falsification review;
- positive and negative controls;
- cluster-aware uncertainty and sensitivity checks;
- fresh sealed evaluation for confirmatory decisions;
- explicit claim grade and limitations.

## 1. Audit of frozen v1.3

The independent adversarial audit confirmed the recorded arithmetic and replay but corrected the interpretation:

- only three teacher seeds were available;
- the 300-step teachers were still improving;
- the matrix-compute comparator used three times as many expert parameters as scalar K2;
- the data were generated from a common template family;
- the scalar student had only 180 local-distillation steps.

Corrected frozen conclusion: scalar K2 was functionally viable and lost to a stronger, three-times-larger narrow model at equal dominant matrix arithmetic. This blocked an equal-compute superiority claim, but did not isolate optimization from representation capacity or refute parameter efficiency.

## 2. Exploratory FAIL-resolution screen

Twelve candidates were evaluated on fresh development documents across the original three teachers.

Best Modal candidate:

```text
scalar-k2-s540-j120
parameters/full:       25.208%
matrix compute/full:   75.000%
adjusted compute/full: 83.333%
local NRMSE:            0.1819
closed-loop delta:     -0.00248 nat
95% interval:          [-0.00777, +0.00242]
KL:                     0.00298
```

Supported exploratory hypotheses:

- closed-loop refinement after local distillation;
- asymmetric allocation as a secondary signal;
- parity within the matrix-matched margin on the development split;
- superiority over a parameter-matched narrow comparator on that split.

The adversarial audit graded this `PROMISING_BUT_POST_SELECTION`: the winner was selected among multiple candidates on the same development split, token CE can adapt to the small distribution, the teachers were immature, and there was no natural-language OOD confirmation.

## 3. Longer-budget teacher screen

Three new 900-step conventional teachers were trained. Their losses were still descending at the final checkpoint, so they are longer-budget teachers rather than demonstrated mature/plateaued teachers. The exploratory parity disappeared:

| Candidate | Params | Adjusted compute | Hypothesis delta | UCB95 | OOD delta | OOD UCB95 |
|---|---:|---:|---:|---:|---:|---:|
| scalar K2 local | 25.208% | 83.333% | +0.10449 | +0.15869 | -0.00248 | +0.02486 |
| scalar K2 closed-loop | 25.208% | 83.333% | +0.05607 | +0.08135 | -0.00789 | +0.02128 |
| narrow parameter-matched | 25.000% | 25.000% | +0.11988 | +0.15506 | +0.01761 | +0.06144 |
| narrow matrix-matched | 75.000% | 75.000% | +0.00618 | +0.01291 | +0.00114 | +0.01791 |

Interpretation:

- closed-loop refinement helps materially;
- scalar K2 is more parameter-efficient than a similarly sized narrow expert on the in-distribution screen;
- scalar K2 does not match the stronger conventional expert at similar dominant matrix arithmetic;
- no candidate qualified for a new sealed protocol.

## 4. Scalar rank curve on longer-budget teachers

Scalar ranks `K=0..8` were tested. No rank achieved the preregistered `UCB95 <= +0.02 nat` gate. The closest points were:

```text
K=7: mean +0.01148 nat; UCB95 +0.02525; matrix compute 200%; adjusted 229.167%
K=8: mean +0.00919 nat; UCB95 +0.02253; matrix compute 225%; adjusted 258.333%
```

For top-4 routing, scalar Modal only reduces dominant matrix arithmetic when `K <= 2`. Thus the rank required for near-fidelity is incompatible with compute reduction in this controlled mature-teacher setting.

## 5. Expert-residual hypothesis

A new family was tested:

```text
W_e = W_modal,e + A_e B_e
```

where the shared Modal term captures cross-expert structure and `A_e B_e` is a small expert-specific low-rank residual.

Primary preregistered candidate:

```text
scalar K=1 + expert residual rank=3 + annealed closed-loop recovery
parameters/full:       36.771%
matrix compute/full:   50.000%
adjusted compute/full: 74.167%
hypothesis delta:      +0.04082 nat
hypothesis UCB95:      +0.05992 nat
OOD delta:             +0.01635 nat
OOD UCB95:             +0.03838 nat
```

Paired comparisons:

```text
vs parameter-matched narrow:
  mean -0.01861 nat; UCB95 -0.00213  -> residual Modal better on development

vs matrix-compute-matched narrow:
  mean +0.03464 nat; LCB95 +0.01416  -> residual Modal clearly worse

vs scalar K2 closed-loop:
  mean -0.01525 nat; UCB95 +0.00857  -> promising, not established
```

Decision: `RESIDUAL_MODAL_NOT_YET_READY`.

The independent adversarial audit reproduced all load-bearing arithmetic and statistics. Its corrected conclusion is that the residual architecture creates a genuine parameter-efficiency signal but is not a validated replacement architecture and is not ready for OLMoE/Qwen.

## 6. Router-semantic clustered bases

A second structural repair tested two or three router-semantic group bases plus expert-specific low-rank residuals. The preregistered G2/R3 annealed candidate used 36.667% of full expert parameters and 70% adjusted arithmetic, but produced `+0.05922 nat` absolute hypothesis loss delta with UCB95 `+0.07829`.

Paired comparisons:

```text
vs parameter-similar narrow dff15:
  mean -0.00021 nat; UCB95 +0.01481  -> no established advantage

vs matrix-compute-matched narrow:
  mean +0.05304 nat; LCB95 +0.03074 -> clearly worse

vs unclustered residual K1/R3:
  independent audit mean +0.01840 nat; LCB95 +0.00260 -> clustering was worse
```

Decision: `RESIDUAL_MODAL_NOT_YET_READY`. Static router-semantic clusters did not solve the bottleneck.

## 7. Current scientific position

### Supported

- the capture/replay and synthetic falsification harness works;
- scalar Modal can be trained and transplanted functionally in small controlled tasks;
- closed-loop refinement is materially better than local-only distillation;
- scalar expert-axis rank alone is insufficient for mature conventional teachers at compute-reducing K;
- expert-specific low-rank residuals improve the parameter/quality frontier;
- the current residual design still loses to a conventional narrow expert at comparable arithmetic.

### Not supported

- GO for OLMoE or Qwen;
- equal-compute superiority of scalar or residual Modal;
- runtime improvement;
- broad natural-language OOD generalization;
- a claim that the architecture is generally unable to work at scale.

## 8. Next hypothesis class

The next candidate must change the function class rather than merely increase scalar K. The highest-value directions are:

1. **Activation/Fisher-weighted factorization:** optimize the basis under the teacher activation covariance instead of raw Frobenius weight error.
2. **Conditional shared mode banks:** learn a small number of mode banks selected by route/context so that unrelated experts do not share one global basis.
3. **Shared residual dictionary:** replace independent `A_e B_e` factors with a small shared dictionary plus expert coefficients, retaining the useful residual directions without the full selected-expert gather path.
4. **Whole-model constrained distillation:** train all Modal layers jointly from initialization rather than forcing a mature conventional layer into a restricted local function class.
5. **Token-conditional residual execution:** preserve the residual capacity found here but execute it only when predicted marginal utility exceeds cost, with an actual kernel-aware objective.

Each direction requires a development screen first. A fresh sealed protocol is created only after one candidate passes teacher fidelity, parameter-matched and compute-matched comparisons, OOD checks, and the independent adversarial audit.
