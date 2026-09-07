# Objective state — 2026-09-06

## Objective

Demonstrate a reproducible quality/parameters/memory/compute/latency advantage against strong relevant alternatives, eventually on real language models and affordable hardware. Compression of our own overparameterized checkpoint is not an improved capacity frontier or architectural novelty.

## Active experiment — DCG-1, results pending

Workflow34072646476, immutable scientific source c056e605d8c3f11b5056918f6e6d01c3a78305b3. Protocol `docs/prereg/DENSE_CONTROL_GATE_1.md` committed before training/holdout preparation. Preflight completed:19 tests passed. Data preparation completed:256 true articles from reserved indices[16384,24576), disjoint from FCC-1 and the training/tokenizer prefix, no exact excluded-window matches. Six training cells are in progress; no outcome is inferred from successful infrastructure.

Five architectures per seed, six new seeds,4400 fixed updates each: native rank1, native rank8, dense104 (exact rank1 parameter match), dense80 (rank1 matrix MACs including router), dense164 (rank8 matrix MACs including router). rank8-to1 is a sixth evaluated arm, not a sixth training. Same backbone initialization and batches within seed. Primary holdout is downloaded in each training job only after frozen checkpoint/export hashes exist. Development snapshots at800/2200/4400 are not candidate-selection opportunities.

Earlier width76/160 compute suggestions omitted the sparse router. DCG-1 uses80/164 to include its768 matrix MACs/token across two layers; this correction preceded outcomes. Primary four contrasts have simultaneous98.75% upper bounds; both seed-t and seed/article bootstrap must meet the frozen meaningful-superiority margin. CPU timing is separate. A separate-runtime archive audit is prepared, not yet executed. Do not rerun or extend training based on partial outcomes.

## Previously completed and verified

- MUI-1: restoring residual stable rank did not consistently improve language modeling. Original FAIL remains.
- FA-1: functional ablations on36 checkpoints;112 exports. Removing/averaging experts failed fidelity screens. Per-expert rank8->rank1 passed storage/fidelity screens in two primary cohorts. Known calibration only.
- FCC-1: fresh article confirmation PASS for eight small compact checkpoints.95552->47168 parameters (-50.6363%). Mean delta NLL+0.0000798154 at800 updates and+0.0005794240 at2200 updates.256 true articles,1024 windows,65536 scored tokens/model. Independent subset reexecution and separate arithmetic checks passed.
- ECK-1: FAIL_INVALID_FOR_SPEEDUP_CLAIM. Two cells failed trained-model FP32 parity. A posthoc trace reproduced a near-tied expert swap; forcing original routes reduced logit error0.0566767->0.000001431. No tolerance relaxation or selective speed claim.
- Data-unit correction: older article IDs include subsections. FA-1 seed-conditional uncertainty is unchanged. FCC-1 and DCG-1 use true top-level article boundaries.

## Current validated claim, unchanged while DCG-1 runs

Post-training rank-one residual compression of the eight FCC-1 checkpoints preserves next-token distribution fidelity within the declared margins on that fresh English-Wikipedia sample while reducing parameter storage. The validated inference implementation is the original loop path, not ECK-1.

## Unmet requirements

No demonstrated superiority to parameter-matched dense models; no certified inference speedup; no GPU/serving/peak-memory/energy claim; no reasoning/tool-use/coding retention result; no convergence, large-model transfer or novel-architecture claim. NO_GO_FOR_OLMOE_OR_QWEN remains unchanged.

## Completion requirements for the active run

Read all six complete cells, aggregate without changing criteria, preserve checkpoints/raw articles/timings/source hashes, execute the separate-runtime archive audit, inspect paired development trajectories, and update README plus the scientific review with the actual verdict. No additional training or tuning is authorized by a provisional partial result.

## Data exposure and continuity

FCC-1 articles are revealed and cannot be recycled as confirmation. DCG-1's separate article pool is reserved for the current frozen protocol; after this run it is also exposed. Before continuing, inspect workflow34072646476 and repository results; do not blindly launch a duplicate.

Entry points: `docs/prereg/DENSE_CONTROL_GATE_1.md`, `pre_qwen_certification/dense_control.py`, `scripts/run_dense_control_gate_1.py`, `scripts/aggregate_dense_control_gate_1.py`, `scripts/audit_dense_control_archive.py`, and `.github/workflows/dense-control-gate-1.yml`.

Historical review: `docs/audits/2026-09-04-compression-functional-review.md`. Prior validated results: `results/fresh-compression-check-1/summary.json` and `results/fresh-compression-check-1/external-archive-audit.json`.
