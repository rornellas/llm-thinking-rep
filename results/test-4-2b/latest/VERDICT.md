# Test 4.2 — WikiText-2 multi-seed compute-matched replication

**Decision:** **MULTISEED_MODAL_ADVANTAGE**

| Variant | Runs | Expert params | Ideal expert compute | Validation loss | Range |
|---|---:|---:|---:|---:|---:|
| baseline-dff32 | 3 | 25.000% | 25.000% | 6.0955 ± 0.0753 | 5.9954–6.1770 |
| baseline-dff48 | 3 | 37.500% | 37.500% | 6.0450 ± 0.0468 | 6.0069–6.1110 |
| baseline-full | 3 | 100.000% | 100.000% | 6.0023 ± 0.0375 | 5.9615–6.0521 |
| modal-k1 | 3 | 3.125% | 25.000% | 5.9764 ± 0.0117 | 5.9605–5.9884 |
| modal-k2 | 3 | 4.688% | 37.500% | 5.9672 ± 0.0093 | 5.9591–5.9803 |

- K1 paired advantages over d_ff=32: `[0.1537, 0.015, 0.1886]`; mean `+0.1191` nat; worst `+0.0150`.
- K2 paired advantages over d_ff=48: `[0.0478, 0.037, 0.1488]`; mean `+0.0779` nat; worst `+0.0370`.
- Modal K1/full loss ratios: `[0.9998, 0.9978, 0.9895]`.

Each paired comparison shares seed, tokenizer, batches, model width, 64-expert/top-8 geometry, and optimization budget. Only the expert parametrization and matched intermediate width differ.
