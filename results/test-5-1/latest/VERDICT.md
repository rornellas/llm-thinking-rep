# Test 5.1 — neural adaptive byte-unit boundaries

**Decision:** **PATCH_COMPRESSION_ONLY**

| Representation | Runs | Bits/byte | Units/context | Bytes/unit | Attention work/byte | Context bytes/s |
|---|---:|---:|---:|---:|---:|---:|
| bigram-adaptive | 3 | 4.6192 ± 0.0171 | 32.05 | 3.99 | 6.282% | 63617 |
| bpe512 | 3 | 4.6193 ± 0.0169 | 61.86 | 2.07 | 23.510% | 53020 |
| byte | 3 | 4.6191 ± 0.0176 | 128.00 | 1.00 | 100.000% | 30645 |
| fixed | 3 | 4.6196 ± 0.0168 | 32.00 | 4.00 | 6.250% | 72793 |
| neural-adaptive | 3 | 4.6193 ± 0.0172 | 32.04 | 4.00 | 6.285% | 58777 |
| random-matched | 3 | 4.6194 ± 0.0172 | 32.48 | 3.94 | 6.458% | 59283 |

- Neural/BPE BpB ratio: `1.0000`.
- Neural unit reduction versus BPE: `+48.21%`.
- Mean neural advantage over fixed: `+0.0002` BpB.
- Mean neural advantage over random matched: `+0.0001` BpB.
- Beats fixed in every seed: `False`; random in every seed: `False`.
- Boundary teacher: `113664` parameters, validation `3.4534` BpB, training `33.3` s.
- Neural threshold: `11.769` bits; bigram threshold: `12.180` bits; fixed patch `4` bytes.

The task predicts the next 16 raw bytes from the same 128-byte context. The boundary teacher and all thresholds are train-only; final comparisons use the held-out WikiText-2 validation split.
