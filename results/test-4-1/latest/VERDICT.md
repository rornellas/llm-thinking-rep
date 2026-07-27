# Test 4.1 — WikiText-2 compute-matched conventional experts

**Decision:** **MODAL_BEATS_MATCHED_NARROW**

| Variant | Expert params | Ideal expert compute | Validation loss | Loss/full |
|---|---:|---:|---:|---:|
| baseline-full | 100.000% | 100.000% | 6.0034 | 1.000× |
| baseline-dff32 | 25.000% | 25.000% | 6.0600 | 1.009× |
| baseline-dff48 | 37.500% | 37.500% | 6.0541 | 1.008× |
| modal-k1 | 3.125% | 25.000% | 5.9796 | 0.996× |
| modal-k2 | 4.688% | 37.500% | 5.9573 | 0.992× |

- Modal K1 advantage over conventional d_ff=32 at matched 25% expert arithmetic: `+0.0804` nat.
- Modal K2 advantage over conventional d_ff=48 at matched 37.5% expert arithmetic: `+0.0968` nat.

All variants use the same tokenizer, corpus, transformer width, 64 experts, top-8 router geometry, training steps, and seed.
