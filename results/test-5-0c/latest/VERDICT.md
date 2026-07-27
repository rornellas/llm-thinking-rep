# Test 5.0 — adaptive byte-unit representation screen

**Decision:** **COMPRESSION_SIGNAL**

| Representation | Bits/byte | Units/context | Bytes/unit | Units/KiB | Attention work/byte | Model bytes/s |
|---|---:|---:|---:|---:|---:|---:|
| byte | 4.8006 | 128.00 | 1.00 | 1024.0 | 100.000% | 22605 |
| bpe512 | 4.8001 | 61.86 | 2.07 | 494.9 | 23.522% | 39815 |
| fixed | 4.7994 | 32.00 | 4.00 | 256.0 | 6.250% | 53990 |
| adaptive | 4.7989 | 32.10 | 3.99 | 256.8 | 6.304% | 45374 |
| random-matched | 4.7992 | 32.32 | 3.96 | 258.6 | 6.388% | 43602 |

- Adaptive/BPE BpB ratio: `0.9997`.
- Adaptive unit reduction versus BPE: `+48.12%`.
- Adaptive BpB advantage versus fixed: `+0.0005`.
- Adaptive BpB advantage versus random matched boundaries: `+0.0003`.
- Calibrated adaptive information budget: `12.200` bits; fixed patch size `4`.

All variants predict the same next raw byte from the same byte windows with identical model parameter count and initialization. This is a representation screen, not yet a full autoregressive patch decoder.
