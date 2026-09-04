# Fresh Compression Check 1 (FCC-1)

2026-09-04. Prospective confirmation of one narrow claim: compressing the eight specified compact checkpoints from residual rank8 to rank1 preserves next-token distributions on previously unused English Wikipedia articles within fixed margins. NOT confirmation of a novel architecture, superiority to conventional models, mature training, OOD capability, or real-LLM applicability. All earlier gates and NO_GO_FOR_OLMOE_OR_QWEN remain.

## Prior evidence and correction

FA-1 passed development fidelity/storage screens in MUI-1 legacy and Gate2A native-shared-rank, but used exposed calibration. Its code's 'unique_articles' count actually names segments: legacy group_articles splits at ANY heading including subsections. Thus 142 segment IDs must NOT be described as 142 independent original articles. FA-1 intervals were seed-only conditional on windows, so their arithmetic is unchanged. This study preserves true top-level articles for sampling and statistics.

## Frozen models and hypothesis

MUI legacy final seeds [904031,904043,904051,904073], 800 updates; Gate2A native-shared-rank final seeds [202781,212789,222793,232801], 2200 updates. Primary candidate: FA-1's rank1 SVD truncation of each residual, common/router unchanged, no fitting. Baseline: original rank8 checkpoint. Contextual controls: corresponding conventional-narrow65 and conventional-full, reported but not a new superiority gate. No selecting seed, checkpoint, projection or rank using fresh outcomes.

## Fresh data before evaluation

Dataset Salesforce/wikitext, wikitext-103-raw-v1, training split, revision read from immutable data/native-compact-gate-2a/manifest.json. Use the ORIGINAL tokenizer, never refit it. Reconstruct top-level articles only: a heading begins and ends with '=', and stripping the outer '=' plus surrounding spaces leaves a body that does NOT begin or end with '='. Subsection headings stay inside their article. Synthetic parser tests required.

Candidate pool: zero-based true article IDs [8192,16384). Permute using numpy default_rng seed904301; accept the first256 eligible unique articles. Encode at most the first4096 tokens/article; at least512 tokens required. Sampling cannot inspect models or losses. Four fixed length64 next-token windows/article, one start sampled from each of four disjoint start-position strata with RNG seed904307. Exclude exact token-window duplicates in the committed model train and validation arrays, and duplicates among selected windows; choose another start within that same stratum in RNG order, then reject the article if a stratum has no valid start. Store selected original article IDs/titles, raw-text SHA256, tokenized windows, starts, rejection reasons and source hashes.

Provenance prerequisite: reconstruct the stored train-token prefix using the LEGACY segmenter and tokenizer, compare exactly against its committed array, and find the latest true article contributing to either this training prefix or the first2048 legacy segments used for tokenizer fitting. Require that entire consumed region ends BEFORE article8192. Otherwise INVALID, no evaluation. This proves selected article IDs are outside the parent models' training/tokenizer region; it does not establish zero semantic overlap across all Wikipedia. Sampling these articles from the dataset's nominal train split is acceptable because these particular models only consumed a bounded prefix; label it clearly.

No old test/OOD arrays evaluated. No model run before data manifest is written. Fresh outcomes are exposed only once for this fixed candidate. Failure ends the protocol without changing margins.

## Endpoints and frozen decision

For each cohort, each seed and each of256 true articles, average NLL, KL(original||rank1) and token argmax agreement over its four windows. Weight articles equally. Primary paired delta: rank1 minus original. Per cohort require: one-sided95% upper bound of delta NLL <=0.010 nat using BOTH seed Student-t (df3, conditional on articles) AND crossed bootstrap over seeds and true articles; KL upper bound <=0.005 nat by BOTH methods; and each seed mean NLL delta<=0.025. Both cohorts must pass. Model parameters after compression <=75% original and exported counts already audited in FA-1. Primary pair only; narrow/full differences are descriptive, not new tests selected after results.

Crossed bootstrap10000 replicates, RNG904313; resample four training seeds and256 articles independently with replacement, sharing article selection across seeds. Report two-sided95% interval, upper one-sided95%, all seed values and raw article/window metrics. These are limited to this sampled English-Wikipedia corpus and these training regimes; four seeds do not estimate arbitrary training variation perfectly. No general reasoning/tool-use/code capability claim.

## Audit and storage

Store all four cells, checkpoints input hashes, tokenizer/source/data hashes, exact windows, article labels, environment, logs and thresholds. Independently recompute per-article metrics and both confidence procedures using different aggregation code or a separate numeric path, verify window uniqueness and parent-prefix disjointness. Reexecute one primary seed per cohort on a deterministic16-article subset using the earlier independent materialized-all-expert implementation and Gram eigenvector rank1 (not the evaluator's low-rank code). Tolerances max per-window NLL difference2e-5, KL difference2e-6. Stop on non-finite values or disagreement. One data job and four bounded CPU cells, no paid hardware and no training.

A PASS confirms small-checkpoint compression fidelity on new articles within the stated margins. It does not prove an improved model-capacity frontier: a dense model with47168 parameters (SwiGLU width104) and a compute-matched dense width76 remain required competitors before that claim.
