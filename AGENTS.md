# Research operating rules

This repository is an experimental research workspace.

## Git workflow

- Commit research work directly to `main`.
- Do not create feature branches or pull requests unless the user explicitly requests them.
- Make small, meaningful commits after each coherent implementation, experiment, audit, result, or correction.
- Preserve negative results and superseded hypotheses; do not rewrite history to make the research appear cleaner.
- Commit every artifact required to reproduce or audit a claim: source, configuration, preregistration, seeds, raw metrics, checkpoints when practical, logs, environment metadata, hashes, reports, and independent audits.

## Scientific standard

- Seek the simplest explanation that survives adversarial testing.
- Separate mechanism evidence, functional quality, parameter savings, analytical compute, measured runtime, and generalization claims.
- Pre-register load-bearing hypotheses, candidates, endpoints, thresholds, statistics, and stop rules before observing confirmatory results.
- Use fresh held-out data for each confirmatory protocol; never reuse revealed holdouts for optimization.
- Compare against the strongest relevant conventional and published-method-inspired baselines at matched parameter and compute budgets.
- Recompute important results independently and perform a factual, adversarial, multi-lens review before elevating a claim.
- Reviews are evidence, not authority: accept or reject each criticism based on logic, code, and data.
- Do not relax gates after seeing results. Record failures accurately and change the hypothesis or architecture when the evidence requires it.
- Keep `NO_GO_FOR_OLMOE_OR_QWEN` in force until a candidate passes the explicitly defined pre-real-checkpoint gates.

## Objective

Find a reproducible architecture or compatible combination of techniques that moves the quality–parameters–compute–memory–latency Pareto frontier, first on controlled models and then on real MoE checkpoints and existing affordable hardware.
