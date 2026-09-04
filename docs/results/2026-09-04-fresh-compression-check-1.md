# FCC-1 — Fresh article compression fidelity

**Verdict:** `FRESH_COMPRESSION_FIDELITY_PASS`. `NO_GO_FOR_OLMOE_OR_QWEN` unchanged.

256 true top-level Wikipedia articles, four windows each, 65536 scored tokens/model. Selected outside the bounded training/tokenizer prefix using frozen seeds and original tokenizer. No refitting or candidate selection.

| Cohort | Mean delta NLL | Seed t95 upper | Crossed95 upper | Mean KL | Top1 agreement |
|---|---:|---:|---:|---:|---:|
| mui1 | +0.0000798 | +0.0001083 | +0.0001074 | 0.0000143 | 99.7971% |
| gate2a | +0.0005794 | +0.0012094 | +0.0009880 | 0.0002665 | 97.6814% |

Parameters: 95552 original -> 47168 compressed (50.6363% reduction); expert matrix MACs30720 ->14592 (52.5% analytical reduction). These are not runtime or total-model compute measurements.

## Contextual conventional controls

| Cohort | Model | Fresh article mean NLL |
|---|---|---:|
| mui1 | original | 5.0927163 |
| mui1 | rank1 | 5.0927961 |
| mui1 | narrow65 | 5.1670287 |
| mui1 | full | 5.0837684 |
| gate2a | original | 4.5015550 |
| gate2a | rank1 | 4.5021345 |
| gate2a | narrow65 | 4.4819518 |
| gate2a | full | 4.4496003 |

## Limits and integrity

The primary confirmation concerns rank1 vs its own compact parent. It does not imply parity with conventional-full or narrow65, neither of which was the primary hypothesis.
English Wikipedia, sequence64, vocabulary512, tiny checkpoints. Neither reasoning, tool use, coding tasks, OOD capability nor a mature large-model scaling law was measured.
The two training budgets use different seeds; they are not a paired learning curve. Four training seeds and article resampling have limited population coverage.
The legacy segmenter split sections at every heading. Older labels naming these segments articles must not be treated as independent true articles. FCC-1 uses verified top-level boundaries and article-weighted statistics.
Independent materialized/Gram-based reexecution passed on16 articles per primary seed0 cohort; all raw article arithmetic, draw-count bootstrap and exact-window exclusion checks passed. This is not an independent research group replication.
Source/raw windows/manifests/individual seed outcomes and bootstrap draws are committed alongside the report. No loss-dependent selection or threshold change occurred.
