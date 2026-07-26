# Research program — efficient MoE representations

## Objective

Find a representation and execution graph that materially reduces model bytes and inference cost while preserving the function of a trained or trainable language model. Compression and acceleration are evaluated separately; every branch has an explicit stopping rule.

## Empirical facts established so far

Experiments on layer 7 of `allenai/OLMoE-1B-7B-0924` have rejected the following post-training, raw-weight hypotheses:

1. **Global modes along the expert axis.** Rank 60 is required for 95% of residual variance; rank 63 for 99%.
2. **Hidden equivalence after exact scale and permutation symmetries.** Joint alignment changes the low-rank curve by less than 0.1 percentage point.
3. **Shared input/output feature subspaces of raw weights.** Rank 512 retains only about 31% of held-out weight energy.
4. **Low-rank modes localized by matrix tile.** Even `16×16` tiles require approximately rank 48 for 90% reconstruction.
5. **A global dictionary of reusable SwiGLU neurons.** Cross-expert triplet similarity is near zero out of sample.

These failures are mutually reinforcing. They indicate that the trained experts are genuinely diverse in raw parameter geometry; the result is not explained by one coordinate symmetry or one global basis choice.

## Current pivot

The highest-priority hypothesis is now **functional, activation-conditioned compressibility**:

\[
G_e x \approx A^G_e(B^G_e x),\qquad
U_e x \approx A^U_e(B^U_e x),\qquad
D_e z \approx A^D_e(B^D_e z)
\]

The matrices are not required to be close globally. They need to agree only on the states actually routed to each expert. This permits a post-training factorization even when Frobenius-distance probes fail.

For equal reduced rank `r`, a per-expert joint gate/up factorization plus a down factorization has the idealized dominant parameter ratio

\[
\frac{r(2d+3m)}{3dm}
\]

and the same first-order arithmetic ratio for routed projections. With `d=2048`, `m=1024`, and `r=512`, this is approximately 29.17% of the original top-8 expert projections, before kernel overhead.

## Active branches

### A. Real activation subspaces — highest priority

Capture held-out layer inputs and MoE outputs from real text. Determine whether a basis learned on calibration tokens retains high energy on unseen tokens.

Continuation gate:

- strong: at least 90% held-out energy by rank 512;
- exploratory: at least 70%;
- close branch: below 70% under a representative corpus.

### B. Route-conditioned reduced-rank experts

Use the original router to partition tokens by selected expert. For every expert, solve a reduced-rank regression against only the tokens routed to it. Evaluate the complete SwiGLU path, not isolated matrix reconstruction.

Required measurements:

- gate/up output error;
- nonlinear intermediate error after `SiLU(g) ⊙ u`;
- expert output error;
- router-weighted layer output error;
- rank distribution by expert and domain;
- exact parameter and arithmetic budgets.

### C. Functional distillation into a constrained modal layer

If analytic factorization is insufficient, train a student layer against the original MoE outputs while enforcing the desired execution graph. This asks whether a shared modal representation can be learned even though unconstrained experts did not spontaneously adopt it.

### D. Modal MoE trained from initialization

Compare a conventional sparse SwiGLU MoE with a model whose expert matrices are parameterized from shared full-rank modes from the start. The modal `down` branch must be executed after aggregation, without reconstructing expert weights.

A positive result here would show architectural trainability, not post-training compressibility. It would justify scaling studies and a custom kernel even if branches A–C fail on an existing checkpoint.

## Closed-branch rule

A negative result closes only the exact hypothesis tested. It does not justify changing thresholds after observing the result. A new test must alter a material assumption: data distribution, objective, parameterization, conditioning variable, or execution graph.

## Priority order

```text
real activation subspaces
→ route-conditioned expert functions
→ complete layer replacement and logits
→ constrained functional distillation
→ train-from-initialization modal MoE
→ hardware kernel only after a quality-preserving representation exists
```

Temporal/Fourier representations of tokens remain a separate research line. They should re-enter only after the MoE representation branch has a verified functional mechanism or has been conclusively closed.
