# Test 2.8 — robust learned rank controller

**Decision:** **ROBUST_CONTROLLER_BORDERLINE**

Thresholds were selected on train-split calibration data using a paired-bootstrap 95% upper bound of +0.003 nat versus static K=1.

| Seed | Threshold | Dynamic loss | Δ vs K1 | Test UCB95 | Mean K | Exact compute | Bucket16 | Bucket32 | Bucket64 | Pass |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 10101 | 0.550 | 2.4024 | -0.0006 | +0.0014 | 1.057 | 26.028% | 26.336% | 26.617% | 27.268% | no |
| 20202 | 0.575 | 2.3933 | -0.0008 | +0.0010 | 1.100 | 26.572% | 26.818% | 27.091% | 27.693% | no |
| 30303 | 0.575 | 2.4081 | -0.0017 | -0.0004 | 1.037 | 25.782% | 26.042% | 26.298% | 26.876% | no |

## Aggregate
- mean Δ loss vs static K1: `-0.0011` nat; worst `-0.0006`.
- mean exact projected compute: `26.127%`.
- mean bucket16 projected compute: `26.399%`; worst `26.818%`.

Bucket cost assumes one upfront compaction by predicted final rank. Mode k is executed only for tokens requiring rank >= k, with each grouped GEMM padded to the indicated tile size. Compaction/scatter latency itself is not yet benchmarked.
