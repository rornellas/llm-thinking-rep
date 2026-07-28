# Adversarial audit — post-v1.3 FAIL-resolution screens

**Grade:** `SUPPORTED_REPRESENTATIONAL_BOTTLENECK__NO_GO_UNCHANGED`

**Independent arithmetic/statistics audit:** `True`

## Corrected conclusion

The follow-up evidence supports a representational, not merely optimization, bottleneck for global scalar expert-axis modes. Closed-loop optimization helps, but scalar rank, expert low-rank residuals, and router-semantic clustered bases do not reach the conventional equal-compute frontier in this controlled longer-budget screen. The original v1.3 NO-GO remains correct. The conclusion is provisional with respect to converged teachers, natural data, larger scales, and materially different function classes.

## Load-bearing checks

- `teacher_recomputation`: **True**.
- `rank_curve_recomputation`: **True**.
- `residual_independent_audit`: **True**.
- `clustered_recomputation`: **True**.
- `longer_budget_teacher_not_converged`: **True**.
- `no_compute_reducing_scalar_rank_passes`: **True**.

## Findings

- **HIGH** — The files historically named 'mature-teacher' contain 900-step teachers whose training losses were still falling by 0.157-0.437 nat from step 720 to 900. They are longer-budget teachers, not demonstrated optimization plateaus.
- **HIGH** — The scalar-rank curve isolates a representational bottleneck: all compute-reducing ranks K<=2 miss the teacher gate, while near-fidelity appears only around K7-K8, where analytic expert arithmetic is greater than the original top-4 MoE.
- **HIGH** — Expert-specific low-rank residuals improve the parameter/quality frontier but remain inferior to conventional narrowing at comparable dominant matrix compute.
- **HIGH** — Router-semantic clustered bases do not solve the failure. The preregistered clustered G2/R3 candidate is statistically indistinguishable from the parameter-similar narrow baseline and clearly worse than the matrix-matched one.
- **MEDIUM** — All follow-up screens are exploratory, reuse development distributions, and have three teacher seeds. They can reject candidate mechanisms but cannot authorize a real checkpoint without a fresh preregistered sealed replication.
- **MEDIUM** — Negative token-loss deltas on the hand-authored OOD set do not imply teacher fidelity when KL remains large; task loss and teacher-function preservation must remain separate endpoints.
- **MEDIUM** — Compute ratios are analytic operation-count proxies. No GPU kernel, memory-traffic, routing, compaction, or latency claim is validated by these screens.

## Clustered primary comparisons

- `clustered-g2-r3-annealed__minus__narrow-dff15-local__hypothesis`: mean `-0.00021`, LCB95 `-0.01749`, UCB95 `+0.01480`.
- `clustered-g2-r3-annealed__minus__narrow-matrix-matched__hypothesis`: mean `+0.05304`, LCB95 `+0.03018`, UCB95 `+0.07226`.
- `clustered-g2-r3-annealed__minus__residual-k1-r3-annealed__hypothesis`: mean `+0.01840`, LCB95 `+0.00260`, UCB95 `+0.03323`.

This audit is independent at the implementation level: it does not import the experimental package. It follows `docs/methodology/IMPORTANT_CLAIM_VERIFICATION_STANDARD.md`.
