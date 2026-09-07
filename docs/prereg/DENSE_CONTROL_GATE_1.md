# Dense Control Gate 1 (DCG-1)

Date: 2026-09-06. Prospective fixed-recipe capacity comparison, registered before new training and evaluation. Parent research state: 4525e7b1edf3df83ff761a110d9a3ecc5d7dad38. All historical outcomes remain unchanged; NO_GO_FOR_OLMOE_OR_QWEN remains in force for every outcome of this experiment.

## Question

Does the small shared-base MoE provide useful predictive capacity beyond a simple dense SwiGLU at matched deployed parameter or forward matrix-compute budgets? Distinguish native rank-one training from training a larger rank-eight parent and compressing it. FCC-1 only confirmed fidelity to our own parent, not an improved capacity frontier.

## Frozen design

Six new seeds: 906101, 906103, 906107, 906109, 906119, 906121. Five models trained per seed, same batches, exactly 4400 updates each; 512 tokens/update (batch8, context64). Checkpoints and development measurements at 800, 2200, 4400; primary endpoint is final4400, never best calibration. No stopping or extending based on loss. No learning-rate search, distillation, finetuning, warm starts, or candidate selection. The prior compact loop implementation is used, not the failed ECK-1 kernel.

Common architecture: vocabulary512, d_model32, two layers, four attention heads, tied token/output embeddings, context64. Same initialization for every non-FFN parameter within a seed. Sparse models have 12 experts, top4, common width64. AdamW lr0.0003, weight_decay0.02, gradient_clip1.0, no scheduler. Sparse auxiliary coefficient0.01, unchanged from Gate2A; dense models have no meaningless load-balancing penalty. Report this objective difference rather than pretending all gradients are identical.

Trainable arms:
- native-rank1: unchanged legacy factor initializer with residual rank1; 47168 total parameters.
- native-rank8: unchanged legacy factor initializer with residual rank8; 95552 parameters.
- dense104: ordinary dense SwiGLU width104; 47168 parameters, exact deployed-size match to native-rank1 and rank8-to1.
- dense80: ordinary dense width80; 42560 parameters, matches rank1 FFN matrix MACs INCLUDING its router.
- dense164: ordinary dense width164; 58688 parameters, matches rank8 FFN matrix MACs INCLUDING its router. Contextual training-compute control for the larger-parent route.

A sixth evaluated arm, rank8-to1, is obtained by fixed SVD truncation of native-rank8 after each saved milestone, without updates. The native rank1 and rank8 arms share initial common matrices and router as well as backbone; factor shapes differ. Dense matrices use Xavier uniform initialization. All model states and all seeds are retained, including poor outcomes.

Accounting per token over two layers: native-rank1 uses14592 expert-transform MACs plus768 router MACs=15360; dense80 uses15360. native-rank8 uses30720+768=31488; dense164 uses31488. dense104 uses19968. Shared attention/output matrix costs are identical for a fixed sequence. These are matrix-operation counts, NOT measured total FLOPs or total training cost: routing, nonlinearities, auxiliary gradients, backward/optimizer work and kernel efficiency differ. The earlier suggestion of dense76 matched only expert transforms, omitting the router; dense80 is the corrected stronger control, selected before outcomes. rank8-to1 inherits rank8's training cost, not rank1's deployed cost.

## Data

Training and development retain the committed bounded train/validation data and tokenizer from native-compact-gate-2a: 700000 training-token cap, 120000 calibration cap, max segment2048, original segment-uniform sampler. Legacy segment IDs are NOT independent articles. Development uses one fixed window/segment, seed906127. No test/OOD arrays or FCC-1 evaluation metrics are used for training or selection.

Primary holdout:256 distinct true top-level articles from the pinned WikiText103 training split, original article indices[16384,24576), disjoint from FCC-1 pool[8192,16384) and the bounded training/tokenizer prefix. Select in order of NumPy RNG906131 permutation, reject duplicate title/text, require512 tokens after tokenization, truncate at4096. Four stratified65-token windows/article (64 predictions), window RNG906137. Exclude exact65-token matches to all old train/validation starts and to FCC-1 windows; reject an article only for prespecified data validity reasons, never loss. Reconstruct the old prefix byte-for-byte to verify the training/tokenizer boundary; preserve article indices/titles, hashes, starts and rejection reasons. Dataset revision is taken from the existing immutable parent manifest. Primary NLL weights true articles equally. The nominal dataset split name does not imply these models trained on the reserved articles.

Fresh data preparation has no access to model losses. Scientific training jobs do not download/read primary holdout data until AFTER checkpoints and their hashes are frozen. Primary holdout evaluated ONLY at4400; earlier milestones use development data only. This tests a fixed training recipe, not the best-tuned performance of every architecture. Generalization scope is tiny English-Wikipedia next-token models, not reasoning, tools, OOD, mature LLMs or global optimality.

## Endpoints and uncertainty

Four load-bearing differences, all final fresh article NLL, lower better:
1. native-rank1 minus dense104;
2. native-rank1 minus dense80;
3. rank8-to1 minus dense104;
4. rank8-to1 minus dense80.

For each, report seed means, paired Student-t interval and upper one-sided98.75% bound (df5), and crossed seed/article bootstrap with10000 draws, RNG906139, upper98.75% bound. 98.75% is a Bonferroni familywise95% allocation for these four differences. Require BOTH upper bounds<=-0.010 nat and no seed difference>+0.010 for a meaningful-superiority comparison. A candidate earns a capacity signal only if BOTH its size-matched and compute-matched comparisons meet those criteria. Multiple tests, modest seeds and bootstrap assumptions still limit inference. Do not rename a non-significant result a tie.

Separately report noninferiority descriptively using the SAME simultaneous upper bounds<=+0.010. It is not the primary superiority gate. Report parent-minus-compressed fidelity (paired NLL, KL, top1), rank8-to1 minus dense164, and development trajectories as secondary, without promoting them to primary confirmation. Do not pool checkpoints from earlier experiments with these new seeds.

A dominant dense outcome is a useful negative result: if both compressed alternatives lack a capacity signal, stop scaling and controller work for this recipe. Do not tune against this holdout. A positive capacity signal only authorizes a later independent scaled/tuned baseline protocol, not a real checkpoint transplant.

## Runtime and integrity

Report actual train wall seconds/arm (rotating order) and tokens; separately benchmark all six final inference models on the same runner, CPU two threads, batch1, lengths1 and64. Inference omits ALL training auxiliary computation in every arm, uses full-prefix forward without KV cache. Use fixed synthetic token inputs, five warmups, nine randomized paired blocks, ten forwards/block, RNG906149. Preserve every timing block, median ratios per seed and hardware metadata. No production/GPU speed claim. Measured inference gains require median paired ratio<=0.9 against BOTH dense controls at BOTH lengths, checked independently of the capacity gate. Do not claim a full quality/storage/latency advance unless all applicable conditions hold.

Preflight without scientific data: exact parameter/MAC counts; tied embedding count; no hidden discarded parent; identical non-FFN initialization; deterministic initialization; dense training/inference parity; sparse original/inference parity; finite backward gradients and updates in every arm; rank8-to1 spectral property; article boundary tests; statistical zero/constant/shift tests. Source/config/protocol must be committed before scientific training. Export checkpoints at every milestone, final inference exports, per-window/per-article NLL, hashes, batch-stream hash, environment, data manifests and logs.

Aggregation must fail closed on missing seeds, mismatched source/data, nonfinite results or wrong counts. Arithmetic audit uses independently computed window/article means and an equivalent draw-count representation of bootstrap, and recalculates t bounds. All final exported states are reloaded; reexecution compares losses for a fixed subset to recorded results. A separate FFN materialization path verifies a prespecified subset of checkpoints, including native and truncated candidates, without changing thresholds to rescue numerical failures. Distinguish implementation reexecution from independent research replication. Errors before outcome observation may be corrected and recorded; no scientific threshold is relaxed after a failure.

## Prior work and claim limits

D2-MoE (Gu et al.,2025, arXiv2502.17298) already uses a shared base and low-rank deltas. SVD, dense SwiGLU and low-rank residuals are not claimed as new. Model quality depends on training budget as well as size (Kaplan et al.,2020, arXiv2001.08361). PyTorch numerical-accuracy documentation warns that batched and sliced computations need not be bitwise identical; ECK-1 also demonstrated routing-boundary amplification here. This experiment tests an explicit empirical capacity claim, not architectural novelty.
