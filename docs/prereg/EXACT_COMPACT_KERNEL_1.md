# Exact Compact Kernel 1 (ECK-1)

2026-09-04. Prospective ENGINEERING benchmark, not a new architecture or quality experiment.
Motivation: FA-1 local primary examples suggest rank-one truncation can preserve function while short-sequence runtime remains dominated by expert-by-expert dispatch. Aggregate FA-1/audit is still pending at registration; the candidate below is fixed regardless of the result.

## Hypothesis and scope

A single vectorized selected-factor implementation of W_e=C+L_e R_e can reduce CPU overhead while preserving the SAME stored function. There is no fitting, new training, rank search, calibration-dependent selection, or custom hardware. Do not claim vectorization, SVD truncation, or a shared base is novel. NO_GO_FOR_OLMOE_OR_QWEN remains.

## Frozen candidates and sources

Every one of the eight primary checkpoints: MUI-1 legacy seeds [904031,904043,904051,904073] and Gate2A native-shared-rank seeds [202781,212789,222793,232801]. Five timed candidates per checkpoint: original loop rank8, vectorized rank8, loop rank1 SVD export, vectorized rank1 SVD export, conventional-narrow65 from the corresponding training seed/cohort. Also conventional-full as a sixth reference. No winner chosen among kernel variants: only one vectorized implementation will be benchmarked. Unsuccessful results are retained without tuning in this protocol.

All models use the same inference-only wrapper (no auxiliary training losses), FP32 CPU two threads, batch1, prefix lengths1 and64, no KV cache. Same source checkpoint inputs, no test/OOD loaded. The rank1 transformation is exactly FA-1's frozen transformation. It must not be confused with function-exact rank8 vectorization: rank1 has FA-1's measured approximation error.

## Algorithm fixed before benchmark

Keep common gate/up/down matrices; stack each uniformly ranked factor bank into one contiguous tensor. Compute shared gate/up once for all tokens; gather only top-k left/right factors; use batched tensor contractions for residual gate/up, then the original SwiGLU nonlinearity. Aggregate weighted hidden states before the shared down projection and add the selected low-rank down residuals. Never reconstruct full per-expert matrices in the timed function. Router/top-k/softmax unchanged. No torch.compile, no adaptive thresholds, no custom extension.

## Preflight and functional endpoint

Synthetic forward and gradient parity for ranks1 and8 versus original loop implementation, two geometries including E12/H64/D32/k4 and E5/H24/D16/k2. Tolerances FP32 atol2e-5 rtol2e-4 (gradients atol3e-5 rtol3e-4). Assert exact parameter-value equality after stacking and same unique scalar count. All 150 known calibration windows on all eight primary checkpoints: vectorized vs corresponding loop logits allclose atol2e-5 rtol2e-5, mean NLL difference <=2e-5 nat. Preserve max logit error and per-window losses. Failed parity invalidates a runtime claim; no weakening tolerance post hoc. This verifies implementation, not generalization.

## Timing endpoint

Use actual calibration window inputs, not different tokens for different candidates. For each prefix length, the first four fixed windows are round-robin workloads within each timing block. Five warmup forwards per model and then 15 timing blocks, 12 forwards per model per block; candidate order randomized with seed904211. Report each raw timing block, median latency and paired ratio to loop rank8 and conventional-narrow65, each checkpoint separately and cohort medians. Same CPU thread settings and no concurrent training in that runner. Count parameters, model-weight bytes, and analytical expert MACs. Selected-factor temporaries are not model-weight storage and no peak-memory reduction claim is made.

A useful engineering result requires vectorized rank1 >=10% lower median latency than BOTH loop rank8 and conventional-narrow65 at BOTH prefix lengths in BOTH cohort medians, with all parity checks passing. Report unfavorable individual seeds and all baselines. Only combine this with FA-1's compression-fidelity claim if FA-1's cross-cohort screen and independent audit pass. Never call the result better language quality than narrow65 merely because it is faster.

## Integrity

Source/protocol committed before measured run. Four bounded standard CPU jobs, index pairing two cohorts; each cap15minutes. Hash inputs/code/protocol, retain raw metrics, environment/lscpu and preflight logs. Independently recompute all ratios and parameter counts. Stop after the frozen benchmark; a subsequent kernel change requires a new protocol. Distinguish this software result from inference speed on GPUs, long-context serving or full-scale LLMs.
