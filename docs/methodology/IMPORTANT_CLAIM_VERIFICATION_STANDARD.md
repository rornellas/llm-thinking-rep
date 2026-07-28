# Standard for important scientific and engineering claims

**Status:** mandatory for this repository from 2026-07-28 onward.

An important claim is any statement that can change a research direction, authorize a larger experiment, support a performance/compression claim, or be repeated externally. Such a claim is not accepted from a single narrative or a single code path.

## Required procedure

1. **State the claim narrowly.** Name the model, layer, data, metric, comparator, precision, hardware and scope. Separate parameter count, active arithmetic, memory traffic and measured latency.
2. **Attach primary evidence.** Preserve raw records, configuration, code commit, environment and SHA-256 manifests. Derived tables are not primary evidence.
3. **Recompute independently.** A second implementation must reconstruct the load-bearing numbers without importing the implementation that produced the original decision whenever practical.
4. **Run an adversarial review.** An independent reviewer/subagent receives the claim, protocol and raw evidence and is instructed to falsify it. It must check leakage, comparator fairness, unit of replication, selection on holdout, hidden costs, convergence, sensitivity and alternative interpretations.
   - When a separate agent runtime is available, use it. When it is not available, use a separately implemented auditor that does not import the experiment code, run an explicit falsification checklist, and disclose that this is implementation independence rather than model-agent independence.
5. **Test negative and positive controls.** A harness must reject deliberately false cases and recover cases with known truth before its outputs authorize a major decision.
6. **Report robustness, not only the favored estimate.** Include per-seed values, cluster-aware uncertainty, leave-one-cluster-out checks, gate sensitivity and relevant Pareto comparators.
7. **Use a sealed final evaluation for confirmatory claims.** Selection and debugging must not observe the final holdout. A consumed holdout cannot be reused to choose a successor architecture.
8. **Record limitations and claim grade.** Every important claim is labeled as one of:
   - `VERIFIED`: raw evidence, independent recomputation and adversarial review all pass;
   - `SUPPORTED`: multiple checks agree, but a material external or scale validation remains;
   - `PROVISIONAL`: exploratory evidence only;
   - `REFUTED`: the stated claim failed its frozen criterion.
9. **No silent gate changes.** Any changed threshold, metric or comparator creates a new protocol version and a new holdout.
10. **Preserve negative results.** A failed gate is a result, not an invitation to reinterpret the same holdout.

## Minimum claim ledger fields

```yaml
claim_id:
claim_text:
grade:
protocol_version:
commit_sha:
raw_evidence:
independent_recomputation:
adversarial_review:
positive_controls:
negative_controls:
comparators:
statistical_unit:
limitations:
decision_authorized:
```

## Operational note

A cryptographic seed commitment verifies consistency between a revealed secret and a preregistered digest. By itself it does not prove that a local operator never saw the secret. Strong operational secrecy requires process isolation or an externally logged CI job in which training jobs cannot access the finalization secret.
