# Test 5.4 — autoregressive byte-patch language model

**Decision:** **AUTOREGRESSIVE_PATCH_FAIL**

| Model | Runs | Patch | BpB | Rolled Δ | Worst rolled LCB95 | Params | Latent positions | Attention work/byte | Target bytes/s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| byte | 3 | 1 | 3.5021 ± 0.0051 | +0.0264 | +0.0215 | 125,760 | 143 | 100.000% | 128918 |
| patch16-gru | 3 | 16 | 3.2459 ± 0.0048 | +0.0317 | +0.0276 | 171,264 | 8 | 0.313% | 217493 |
| patch4-gru | 3 | 4 | 3.3098 ± 0.0096 | +0.0305 | +0.0257 | 172,992 | 35 | 5.991% | 219301 |
| patch8-gru | 3 | 8 | 3.2805 ± 0.0053 | +0.0305 | +0.0264 | 171,840 | 17 | 1.413% | 239519 |
| patch8-mean | 3 | 8 | 3.4632 ± 0.0029 | +0.0080 | +0.0047 | 146,880 | 17 | 1.413% | 340423 |

- Byte baseline: `3.5021` BpB.
- Patch4: ratio `0.9451`, context `False`.
- Patch8: ratio `0.9367`, context `False`.
- Patch16: ratio `0.9269`, context `False`.
- Patch8 GRU advantage over mean encoder: `+0.1827` BpB.

Every model receives the same 144 raw bytes and predicts the same 128 raw target bytes. Patch models are autoregressive within each target patch; rolled-context evaluation keeps targets fixed and replaces preceding context with another held-out example.
