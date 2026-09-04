# ECK-1 — failed exact-kernel certification

**Verdict: FAIL_INVALID_FOR_SPEEDUP_CLAIM.** Synthetic algebra/gradient tests passed, but cells0 and1 failed full trained-model FP32 logit parity. Cells2 and3 are preserved, not selected as a successful benchmark.

Observed first cell0 maximum logit difference:0.05667674541473389. Original allclose atol2e-5/rtol2e-5 is unchanged. No valid aggregate speedup is claimed.

All original raw outputs are in `results/exact-compact-kernel-1/attempt-1/`. No benchmark was repeated for this report.

## Posthoc mechanism trace

Diagnostic status: `REPRODUCED`. See `failure-diagnostic.json` for layer inputs, selected experts, routing margins and the fixed-routing counterfactual.

Forcing original routes, if it reduces the mismatch, supports that mechanism for this example only; it does not reverse the failed certification or establish safe serving behavior.

Compression-fidelity claims from FA-1/FCC-1 use the original loop implementation, not this failed vectorized kernel.
