# Objective state — 2026-09-06

## Objective

Demonstrate a reproducible quality/parameters/memory/compute/latency advantage against strong relevant alternatives, eventually on real language models and affordable hardware. Compressing our own checkpoint does not establish a better capacity frontier or architectural novelty. The full objective remains unmet.

## Latest completed experiment — DCG-1

**Verdict: DCG1_NO_CAPACITY_ADVANTAGE_DEMONSTRATED.** Workflow 34072646476 completed; immutable scientific source c056e605d8c3f11b5056918f6e6d01c3a78305b3. The protocol was committed before training and new holdout preparation. All 30 trainings, six paired seeds, 4400 fixed updates and 36 final exports completed. All 19 preflight tests passed. No scientific jobs remain pending.

256 true top-level articles, four windows each, 65536 scored tokens/model, from reserved indices [16384,24576). Parent training/tokenizer prefix reproduced exactly. No exact excluded-window intersections with prior train/validation/FCC-1 data. Training jobs received the primary holdout only after freezing their checkpoint hashes. Development checkpoints at 800/2200/4400 were not used to choose a winner.

| Arm | Final fresh NLL | Total parameters | FFN matrix MACs including router |
|---|---:|---:|---:|
| native-rank1 | 4.269019 | 47168 | 15360 |
| native-rank8 | 4.244904 | 95552 | 31488 |
| dense104 | 4.227333 | 47168 | 19968 |
| dense80 | 4.258728 | 42560 | 15360 |
| dense164 | 4.196007 | 58688 | 31488 |
| rank8-to1 | 4.248656 | 47168 | 15360 |

None of four simultaneous primary superiority comparisons passed. dense104 matched deployed parameter count and had lower mean NLL than both compact rank1 candidates. It beat native-rank1 in all six seeds and rank8-to1 in five. rank8-to1 had a favorable mean difference of -0.010071 nat vs dense80 and met descriptive noninferiority, but did NOT demonstrate prespecified superiority. Preserve this secondary tradeoff without promoting it.

Measured paired CPU forward latency of rank8-to1 relative to dense104: median ratio 3.5368 at length1, 4.6324 at length64. Two threads, batch1, no auxiliary loss or KV cache. These are implementation-specific full-prefix forwards, not serving/GPU measurements. rank8-to1 inherits larger-parent training cost. Earlier width76/160 compute suggestions omitted the router; corrected controls80/164 were specified before outcomes.

## Completed audits and supplemental interpretation

Original workflow checked raw metrics, hashes, all 36 reloaded exports, and two bootstrap paths. Separate all-expert/Gram-based reexecution verified two prespecified seed0 candidates on 16 articles each; maximum per-window NLL error 6.31e-7. This subset is not a full independent replication.

A separate Python3.13.5 runtime audited 83 manifest files, actual storage of 36 exports, all four primary contrasts with a third bootstrap arithmetic path, manually interpolated percentiles and paired latency ratios. PASS, recorded in external-archive-audit.json. The initial manifest remains frozen; later supplements are recorded separately.

Supplementary checkpoint diagnostics are descriptive: native-rank8 residual stable rank 1.0342; all models still improved between development steps2200 and4400. No convergence claim. Training larger then compressing had lower mean NLL than training rank1 directly, but did not overcome dense104. This does not identify optimization vs representational capacity causally.

Secondary compressed-parent fidelity at4400: mean NLL increase 0.0037520 nat, KL0.0026460, top1 agreement92.7127%. These are not the winning capacity endpoint and do not establish reasoning/tool-use preservation.

## Prior validated work, unchanged

- MUI-1: restoring residual stable rank did not consistently improve language modeling. Original FAIL remains.
- FA-1: functional ablations on 36 checkpoints, 112 exports. Removing/averaging experts failed fidelity screens; rank8-to1 passed storage/fidelity screens in two primary cohorts. Known calibration only.
- FCC-1: fidelity confirmation PASS for eight small compact checkpoints. 95552 -> 47168 parameters (-50.6363%). Mean delta NLL +0.0000798154 at800 updates and +0.0005794240 at2200. 256 fresh true articles; independent subset reexecution and separate arithmetic audits passed. Compression of these parents is valid; superiority to dense alternatives is not implied.
- ECK-1: FAIL_INVALID_FOR_SPEEDUP_CLAIM. Two cells failed trained-model FP32 parity. Near-tied routing amplified floating-point changes in a reproduced case. No relaxed tolerance or selective speedup claim. Use the original loop for validated quality results.
- Data-unit correction: old article IDs included subsections. FA-1 seed-conditional uncertainty unchanged. FCC-1 and DCG-1 use true top-level article boundaries.

## Operational decision

Suspend scaling, dynamic-rank controllers and performance claims for this fixed recipe. Keep dense104 as mandatory equal-size baseline and dense80 as router-inclusive equal-FFN-MAC baseline. NO_GO_FOR_OLMOE_OR_QWEN remains unchanged. No evidence of a novel or globally optimal architecture, broad LLM transfer, reasoning/tool/coding retention, GPU/serving/peak-memory/energy gains.

The unresolved narrow question is whether the native deficit is mainly optimization or representation. The only justified continuation of this same parameterization is a bounded, equally budgeted optimization screen for both compact and dense models, on training/development ONLY. Register candidate recipes, total search cost, decision thresholds and abandonment rule before running it. Only a consistent development signal would justify another independent confirmation; otherwise archive this parameterization rather than proliferate variants. This screen has NOT been started. Do not tune or extend DCG-1 after its negative result.

## Data exposure and continuation safety

FCC-1 and DCG-1 holdouts are now revealed. Neither can be used for tuning and then called fresh confirmation. DCG-1 source/results are immutable evidence, not an active training request. Before future work, read the review and actual verdict; do not blindly launch the completed workflow again.

## Entry points

- `docs/audits/2026-09-06-dense-control-gate-1-review.md`
- `docs/results/2026-09-06-dense-control-gate-1.md`
- `docs/prereg/DENSE_CONTROL_GATE_1.md`
- `results/dense-control-gate-1/summary.json`
- `results/dense-control-gate-1/external-archive-audit.json`
- `results/dense-control-gate-1/supplementary-diagnostics.json`
- `scripts/audit_dense_control_archive.py`
- `scripts/diagnose_dense_control_gate_1.py`
- `docs/audits/2026-09-04-compression-functional-review.md`

All work described above is completed, preserved directly on main, and no training or new confirmatory protocol is running.
