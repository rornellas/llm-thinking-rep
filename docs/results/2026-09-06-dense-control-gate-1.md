# DCG-1 — Matched dense controls

**Verdict:** `DCG1_NO_CAPACITY_ADVANTAGE_DEMONSTRATED`.

Thirty new trainings, six paired seeds, 4400 fixed updates, 256 fresh true articles and 65536 scored tokens/model. All prior gates remain unchanged.

| Arm | Fresh NLL | Total parameters | FFN MACs including router | Forward L1 ms | Forward L64 ms |
|---|---:|---:|---:|---:|---:|
| native-rank1 | 4.269019 | 47168 | 15360 | 1.3374 | 3.0821 |
| native-rank8 | 4.244904 | 95552 | 31488 | 1.3352 | 3.4116 |
| dense104 | 4.227333 | 47168 | 19968 | 0.3775 | 0.6407 |
| dense80 | 4.258728 | 42560 | 15360 | 0.3761 | 0.6320 |
| dense164 | 4.196007 | 58688 | 31488 | 0.3776 | 0.6737 |
| rank8-to1 | 4.248656 | 47168 | 15360 | 1.3288 | 3.0103 |

## Prespecified primary contrasts

| Candidate minus control | Mean delta NLL | Seed upper98.75% | Crossed upper98.75% | Superiority |
|---|---:|---:|---:|---|
| native-rank1__minus__dense104 | +0.041686 | +0.063024 | +0.055031 | False |
| native-rank1__minus__dense80 | +0.010291 | +0.032025 | +0.023331 | False |
| rank8-to1__minus__dense104 | +0.021323 | +0.041656 | +0.033615 | False |
| rank8-to1__minus__dense80 | -0.010071 | +0.007792 | +0.000923 | False |

Both simultaneous upper bounds must be <= -0.010 nat; no seed may exceed +0.010. Four contrasts use Bonferroni allocation. Bootstrap does not substitute for independent replication.

## Secondary compression fidelity

Mean compressed-parent NLL delta: +0.0037520; mean KL: 0.0026460; top1 agreement: 92.7127%.

## Limits

Fixed optimizer/initialization recipe, tiny model, context64 and English Wikipedia only. No per-architecture tuning, reasoning/tool tests, OOD generalization or global optimum claim. rank8-to1 inherits larger-parent training cost. Router matrix costs are included; nonlinearities, auxiliary work, optimizer and full backward FLOPs are not.

Timings are paired within six CPU runners, two threads, batch1, no auxiliary loss and no KV cache. They are full-prefix forwards, not production autoregressive throughput. All timing blocks and each seed are retained.

Audits verify source/data/checkpoint hashes, train/eval separation, actual exported storage, reloads, raw-window arithmetic and two bootstrap implementations. Independent all-expert/Gram/logsumexp reexecution covers the first16 fresh articles of two prespecified models from seed0, not all examples or an independent research group.

Fresh DCG-1 articles are now exposed. Do not use them for optimization and claim a subsequent confirmation.
