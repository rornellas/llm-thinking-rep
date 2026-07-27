# Test 5.5 — causal byte patches + Modal-MoE integration

**Decision:** **INTEGRATED_PASS_WITH_THROUGHPUT**

| Variant | Runs | BpB | First-patch BpB | First rolled Δ | Params | Expert params/full | Global positions | Attention work/byte | Joint expert compute/byte-full | Target bytes/s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| byte-full | 3 | 3.5444 ± 0.0069 | 3.5256 | +0.3593 | 2,443,584 | 100.000% | 143 | 100.000% | 100.000% | 45906 |
| byte-modal | 3 | 3.5806 ± 0.0041 | 3.5738 | +0.3290 | 158,400 | 3.141% | 143 | 100.000% | 26.562% | 140884 |
| byte-narrow | 3 | 3.5723 ± 0.0040 | 3.5739 | +0.3136 | 674,112 | 25.000% | 143 | 100.000% | 25.000% | 53377 |
| patch8-full | 3 | 3.4401 ± 0.0083 | 3.4592 | +0.3821 | 2,489,664 | 100.000% | 17 | 1.413% | 11.888% | 84994 |
| patch8-modal | 3 | 3.4439 ± 0.0072 | 3.4641 | +0.3783 | 204,480 | 3.141% | 17 | 1.413% | 3.158% | 263534 |
| patch8-narrow | 3 | 3.4400 ± 0.0085 | 3.4572 | +0.3761 | 720,192 | 25.000% | 17 | 1.413% | 2.972% | 97170 |

## Paired gates

- Integrated minus byte-full: `-0.1006` BpB; UCB95 `-0.0969`.
- Integrated minus patch8-narrow: `+0.0038` BpB; UCB95 `+0.0048`.
- Factorial interaction: `-0.0045` BpB; UCB95 `-0.0026`.
- Integrated/byte-full throughput ratios: `[6.076, 5.658, 5.511]`; mean `5.748x`.

## Modal code interventions

| Policy | Δ BpB | LCB95 | Pass |
|---|---:|---:|---|
| mean-code | +0.0036 | +0.0032 | yes |
| shuffle-code | +0.0058 | +0.0052 | yes |
| zero-residual | +0.0037 | +0.0031 | yes |

- Quality gate: `True`.
- Factorial interaction gate: `True`.
- Causal context gate: `True`.
- Expert-code gate: `True`.
- Measured throughput gate: `True`.

All models predict identical raw-byte targets from identical windows. The first-target-patch intervention avoids dilution by later teacher-forced target bytes. Idealized work ratios exclude local patch encoding/decoding; measured target bytes/s includes the entire forward path.
