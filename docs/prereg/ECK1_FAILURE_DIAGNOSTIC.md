# ECK-1 failure diagnostic

2026-09-04. POSTHOC diagnostic; no confirmatory or runtime endpoint.
ECK-1 run33920705855 failed exact FP32 parity in cells0/1 after synthetic tests passed. Cell0 reported maximum logit error0.05667674541473389 at MUI-1 seed904031, rank8, segment native-calibration-article-0085. Successful cells2/3 cannot be selected to claim a successful benchmark.

Hypothesis to check, not yet established: floating-point operation reordering changes downstream top-k selection near a routing boundary, amplifying small arithmetic differences.

Before inspecting layer traces, freeze this check: load original MUI legacy seeds904031 then904043, same150calibration windows and exact ECK implementation. Find the first allclose failure (original tolerance atol2e-5 rtol2e-5), record every layer's input/output maximum errors, routed-ID differences, and top4/top5 logit margins. Reexecute the vectorized implementation with original top-k IDs AND slot probabilities forced in all layers, then compare final logits. Report raw failure and forced-route comparison. If no failure reproduces on this runner, report NOT_REPRODUCED and preserve the original failed artifacts. Do not tune a kernel, alter tolerances or repeat timings.

A reduction of mismatch after forcing routing supports this specific boundary-sensitivity mechanism, not a universal explanation, an architecture gain or a successful ECK-1. Original ECK-1 remains FAIL_INVALID_FOR_SPEEDUP_CLAIM regardless of this diagnosis.
