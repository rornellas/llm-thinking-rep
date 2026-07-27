# Test 5.2 — learned continuous latent resampler

**Decision:** **HALF_SLOT_COMPRESSION_SIGNAL**

| Compressor | Runs | Slots | BpB | Params | Compressor params | Attention work/byte | Context bytes/s | Effective rank | Mean slot cosine |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| conv32 | 3 | 32 | 4.6237 ± 0.0098 | 393,024 | 16,448 | 6.250% | 146953 | 25.3 | 0.871 |
| fixed-gru32 | 3 | 32 | 4.6235 ± 0.0100 | 401,536 | 24,960 | 6.250% | 106037 | 16.2 | 0.986 |
| fixed-mean32 | 3 | 32 | 4.6236 ± 0.0095 | 376,576 | 0 | 6.250% | 167321 | 25.0 | 0.282 |
| resampler16 | 3 | 16 | 4.6236 ± 0.0099 | 411,072 | 34,496 | 1.562% | 131640 | 7.4 | 0.641 |
| resampler32 | 3 | 32 | 4.6234 ± 0.0099 | 412,096 | 35,520 | 6.250% | 111091 | 10.0 | 0.674 |
| resampler32-frozen | 3 | 32 | 4.6238 ± 0.0097 | 376,576 | 0 | 6.250% | 129315 | 14.8 | 0.274 |

- Best fixed compressor: `fixed-gru32`.
- Resampler32 advantages over paired best-fixed controls: `[0.0008, -0.0006, -0.0005]`; mean `-0.0001` BpB.
- Resampler32 advantages over frozen control: `[0.0007, 0.0, 0.0005]`.
- Resampler32 parameter ratio to best fixed: `1.026`.
- Resampler16/Resampler32 BpB ratio: `1.0000`; paired deltas `[0.0006, 0.0, -0.0002]`.

Every model predicts the next 16 bytes from identical 128-byte contexts. Global resampling cost is included in measured model throughput; attention-work ratios describe only the downstream latent Transformer.
