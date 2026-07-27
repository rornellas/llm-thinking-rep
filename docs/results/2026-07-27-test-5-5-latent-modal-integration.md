# Test 5.5 — Causal byte patches + Modal-MoE integration

## Decision

**INTEGRATED_PASS_WITH_THROUGHPUT**

Workflow run: `30289211653`

## Question

Can a causal compact byte representation and Modal-MoE K1 operate in the same autoregressive language model without a material quality loss, while each mechanism remains causally active?

## Design

The paired factorial experiment crossed:

- representation: individual bytes vs causal 8-byte GRU patches;
- experts: conventional full-width, conventional 25%-width, and Modal K1;
- three independent seeds;
- 64 experts, top-8 routing, two global layers;
- identical WikiText-2 raw-byte windows and identical raw-byte targets;
- 700 optimization steps per variant.

The integrated candidate was `patch8-modal`.

## Main results

| Variant | BpB | Parameters | Expert parameters/full | Global positions | Attention work/byte | Joint expert compute/byte-full | Target bytes/s |
|---|---:|---:|---:|---:|---:|---:|---:|
| byte-full | 3.5444 ± 0.0069 | 2,443,584 | 100.000% | 143 | 100.000% | 100.000% | 45,906 |
| byte-narrow | 3.5723 ± 0.0040 | 674,112 | 25.000% | 143 | 100.000% | 25.000% | 53,377 |
| byte-modal | 3.5806 ± 0.0041 | 158,400 | 3.141% | 143 | 100.000% | 26.562% | 140,884 |
| patch8-full | 3.4401 ± 0.0083 | 2,489,664 | 100.000% | 17 | 1.413% | 11.888% | 84,994 |
| patch8-narrow | 3.4400 ± 0.0085 | 720,192 | 25.000% | 17 | 1.413% | 2.972% | 97,170 |
| **patch8-modal** | **3.4439 ± 0.0072** | **204,480** | **3.141%** | **17** | **1.413%** | **3.158%** | **263,534** |

The integrated candidate used approximately 8.37% of the total parameters of `byte-full`, a 91.63% reduction in this test model.

## Quality and interaction gates

- Integrated minus byte-full: `-0.1006` BpB; pooled UCB95 `-0.0969`.
- Integrated minus patch8-narrow: `+0.0038` BpB; pooled UCB95 `+0.0048`.
- Predeclared non-inferiority margin: `+0.0100` BpB.
- Factorial interaction: `-0.0045` BpB; pooled UCB95 `-0.0026`.

The integrated model was materially better than the byte/full reference and remained within the non-inferiority margin of the compute-matched patch/narrow control. The negative interaction estimate provides no evidence of antagonism between compact patches and Modal-MoE in this setting.

## Causal context gate

The first target patch was evaluated separately to avoid dilution by later teacher-forced target bytes.

Across the three seeds, replacing the correct prefix with another held-out prefix increased first-patch BpB by:

- `+0.3694`, LCB95 `+0.3311`;
- `+0.3898`, LCB95 `+0.3495`;
- `+0.3757`, LCB95 `+0.3362`.

Context-signal retention relative to `patch8-full` was `105.3%`, `97.7%`, and `94.7%`. The integrated model therefore used the compact context causally rather than relying only on local target statistics.

## Expert-code gate

Pooled interventions on the trained integrated model:

| Intervention | Δ BpB | LCB95 |
|---|---:|---:|
| Replace expert codes by their mean | +0.0036 | +0.0032 |
| Shuffle codes among experts | +0.0058 | +0.0052 |
| Remove residual codes | +0.0037 | +0.0031 |

All three interventions degraded held-out quality with positive lower confidence bounds. Expert identity was not ignored.

## Throughput

Measured CPU reference throughput for `patch8-modal` was 5.51x–6.08x the `byte-full` reference across seeds, mean 5.75x. This includes the local patch encoder and decoder.

This is not yet a production GPU claim. The implementations have different optimization levels, and the conventional expert path uses a Python expert loop. The result demonstrates a strong end-to-end reference implementation advantage and justifies a fused-kernel benchmark.

## Conclusion

At this scale, the two mechanisms function together:

1. quality is preserved under the predeclared margin;
2. there is no measured antagonistic factorial interaction;
3. compact context remains causally relevant;
4. expert-specific modal codes remain causally relevant;
5. the combined model establishes a substantially better parameter/sequence/CPU-throughput point.

## Remaining validation

Before a broad scientific claim, repeat at greater depth, context length, data volume, and model width; include code, mathematics, Portuguese, and synthetic long-range retrieval; and benchmark fused GPU kernels against an optimized Grouped-GEMM MoE baseline.
