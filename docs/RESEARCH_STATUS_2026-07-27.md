# Research status — 2026-07-27

## Objective

Develop a language-model representation and execution stack that improves the
quality-per-byte, quality-per-FLOP, memory traffic, and latency frontier through:

1. compact language units derived from raw bytes;
2. shared full-rank Modal-MoE transformations with small expert codes;
3. nested refinement modes and adaptive compute;
4. sparse event/delta representations suitable for future hardware co-design.

This document is a decision record. It separates results that have survived
controls from open hypotheses and branches that should not be revisited without
new evidence.

---

## Current architectural thesis

The strongest surviving formulation is:

```text
raw bytes
  -> compact local units or patches
  -> latent context states
  -> shared full-rank Modal-MoE matrices
  -> expert-specific scalar or neuron-wise codes
  -> nested refinement modes K=0..N
  -> optional utility-based rank selection
  -> optional sparse event encoding of refinement deltas
```

A conventional trained MoE should not be assumed to contain a low-dimensional
linear basis that can be recovered post hoc. Compact structure must be imposed
during training or learned through functional distillation.

---

## Confirmed results

### A. Directly trained Modal-MoE is more efficient than narrowing experts

**Test 4.2b — WikiText-2, three paired seeds, 64 experts/top-8**

Result: `MULTISEED_MODAL_ADVANTAGE`.

- Modal K1 used 25% ideal expert arithmetic and approximately 3.125% of
  conventional expert parameters.
- Modal K2 used 37.5% ideal expert arithmetic and approximately 4.688% of
  conventional expert parameters.
- K1 beat the 25%-width conventional expert in every seed; paired advantages:
  `0.1537`, `0.0150`, `0.1886` nat, mean `0.1191` nat.
- K2 beat the 37.5%-width conventional expert in every seed; paired advantages:
  `0.0478`, `0.0370`, `0.1488` nat, mean `0.0779` nat.
- Modal K1/full loss ratios were `0.9998`, `0.9978`, `0.9895`.

Source: `results/test-4-2b/latest/`.

**Interpretation:** the result is not explained by the task merely accepting
smaller expert MLPs. Sharing full-rank matrix modes across experts is a more
efficient parametrization at this scale.

### B. Expert-specific codes carry functional information

**Test 2.5 — 64 experts/top-8 code ablations**

Replacing codes by their mean, shuffling them among experts, or zeroing residual
codes degraded validation loss. The router did not simply ignore expert
identity.

Source: `results/test-2-5/latest/`.

### C. Nested prefixes are trainable

**Test 2.6 — K=0..3 from one checkpoint**

Result: `PROGRESSIVE_PASS`.

At 0.05 nat oracle tolerance, mean rank was approximately `0.902`, with
projected expert compute approximately `23.818%` of the conventional top-8
projection work.

Source: `results/test-2-6/latest/`.

### D. Utility-based dynamic rank selection is robust in the small model

**Test 2.11 — three fresh seeds**

Result: `ROBUST_MARGINAL_UTILITY_PASS`.

- mean bucket-16 projected compute: `19.160%`;
- static K1 projected compute: `25.049%`;
- mean advantage: `5.889` percentage points;
- worst held-out loss increase: `0.0025` nat;
- worst paired-bootstrap UCB95: `0.0038` nat.

Source: `results/test-2-11/latest/`.

**Interpretation:** adaptive refinement is viable, but the static K1 path
remains the deployment baseline until real kernel latency is measured.

### E. Byte contexts can be compressed into far fewer latent units

**Test 5.0c — WikiText-2 raw bytes**

Result: `COMPRESSION_SIGNAL`.

A 128-byte context was represented using approximately:

- byte: 128 units, 100% relative attention work;
- reversible byte-BPE512: 61.86 units, 23.52% attention work;
- fixed 4-byte patches: 32 units, 6.25% attention work;
- adaptive patches: 32.10 units, 6.30% attention work;
- random matched patches: 32.32 units, 6.39% attention work.

All variants had nearly identical next-byte BpB in the undertrained screen.
Adaptive boundaries did **not** materially beat fixed or random controls.

Source: `results/test-5-0c/latest/`.

**Interpretation:** local latent patch encoding is promising; boundary
intelligence is not yet demonstrated.

### F. Refinement deltas contain a strong event-compression signal

**Test 6.0 — post-training sparse/quantized refinement deltas**

Strict result: `FAIL`, but with a relevant near miss.

- 75% target sparsity and 4-bit events: `+0.0047` nat versus K3,
  residual traffic `12.398%` of dense BF16 deltas.
- 90% target sparsity and 4-bit events: `+0.0138` nat versus K3,
  residual traffic `8.635%`.
- magnitude-selected 90%/4-bit events beat a random mask at the same event rate
  by approximately `0.0120` nat.

Source: `results/test-6-0/latest/`.

**Interpretation:** event identity matters, and substantial delta traffic can be
removed. The model was not trained for sparse deltas; the next hardware-proxy
experiment should impose event sparsity during training.

---

## Refuted or closed branches

### Post-hoc low-rank expert-axis decomposition of OLMoE weights

Raw PCA, scale/permutation alignment, clustered/neuron-wise variants, feature
subspaces, and cross-expert neuron dictionaries all failed to expose a compact
basis at useful K. Rank-95 remained close to the maximum expert-axis rank.

Do not spend further compute on another linear PCA variant over the same layer
without a genuinely different mathematical assumption.

### Post-hoc distillation of the real OLMoE layer-7 down branch at low modal rank

Both aggregate-output and per-assignment supervision failed badly on held-out
real activations. The small model's successful functional distillation does not
transfer trivially to one frozen branch of an unconstrained trained OLMoE.

### Global linear activation subspace

A single shared low-dimensional activation basis did not preserve sufficient
held-out energy. Future activation compression must be conditional, nonlinear,
or trained end to end.

### Literal mapping of semantic tokens to CPU clock frequencies

A clock cycle is synchronization, not a semantic carrier. The hardware-relevant
version of the idea is sparse event timing, delta coding, phase/amplitude only
on hardware that processes those quantities natively, and model/hardware
co-design.

---

## Experiments currently active

### Test 5.1 — neural adaptive byte-unit boundaries

Purpose:

- replace bigram boundaries with a train-only causal byte-GRU surprisal model;
- predict the next 16 bytes per context instead of one;
- run three paired seeds;
- compare against fixed, random matched, bigram, byte, and reversible BPE;
- require paired-bootstrap evidence that boundary placement itself matters.

Source: `experiments/test_5_1_neural_adaptive_byte_units.py`.

### Test 4.3 — width-scaling curve

Purpose:

- repeat full/narrow/Modal K1/K2 comparisons at d_model 64, 96, 128, and 160;
- hold 64 experts/top-8, depth, corpus, and token budget fixed;
- determine whether Modal efficiency persists as matrix dimensions grow.

Source: `experiments/test_4_3_wikitext_width_scaling.py`.

---

## Next gates

### Gate 1 — boundary intelligence

Continue to end-to-end adaptive units only if Test 5.1 shows that neural
boundaries beat fixed and random matched controls across seeds. Otherwise retain
fixed local patches as the simpler representation and test learned soft
compression rather than discrete boundaries.

### Gate 2 — scale trend

Continue to a larger token-level Modal model if Test 4.3 retains positive paired
advantages at all widths. If the advantage collapses with width, inspect code
capacity, number of modes, and router utilization before increasing scale.

### Gate 3 — event-producing training

Implement Test 6.1 with an explicit sparse event bottleneck or train-time delta
regularizer. A useful gate is:

```text
>=90% zero residual events
<=4 bits per nonzero event
<=0.010 nat held-out loss increase versus K3
magnitude/event controller beats random event placement
```

### Gate 4 — real systems performance

After the above gates, implement a grouped rank-compaction kernel and compare
against an optimized conventional MoE baseline. Required measurements:

- prefill and decode separately;
- actual HBM/DRAM bytes;
- compaction and scatter latency;
- grouped GEMM padding;
- end-to-end tokens/s and time-to-first-token;
- static K1 versus dynamic utility controller.

No speed claim should be made from ideal FLOPs alone.

---

## Current default candidates

| Concern | Default candidate |
|---|---|
| Expert parametrization | neuron-wise Modal K1 or scalar Modal K1, selected by scale test |
| Static deployment | K1 |
| Adaptive deployment | marginal-utility controller with rank compaction |
| Input representation | fixed local byte patches until Test 5.1 proves intelligent boundaries |
| Hardware/event representation | dense K0 plus sparse quantized refinement deltas |
| Conversion of legacy MoE | full functional distillation, not analytic weight decomposition |

---

## Scientific caution

All positive architecture results remain small-model results. They establish
mechanism and reject several trivial explanations; they do not yet establish
billion-parameter scaling, production latency, or superiority on broad language
benchmarks. The next experiments must prioritize paired controls and scaling
over adding more architectural components at once.
