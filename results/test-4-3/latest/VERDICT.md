# Test 4.3 — WikiText-2 Modal-MoE width scaling

**Decision:** **WIDTH_SCALING_MODAL_ADVANTAGE**

| Scale | d_model | d_ff | Full loss | Narrow25 | Modal K1 | K1 advantage | Narrow37.5 | Modal K2 | K2 advantage |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| w64 | 64 | 96 | 6.3434 | 6.4022 | 6.3697 | +0.0325 | 6.3427 | 6.3242 | +0.0185 |
| w96 | 96 | 128 | 6.0416 | 6.2189 | 6.0432 | +0.1757 | 6.2124 | 6.0873 | +0.1251 |
| w128 | 128 | 192 | 5.9374 | 6.0020 | 5.9422 | +0.0597 | 6.0028 | 5.9100 | +0.0929 |
| w160 | 160 | 256 | 5.8839 | 5.9719 | 5.8637 | +0.1082 | 5.9253 | 5.8494 | +0.0759 |

- K1 advantage mean/worst: `+0.0940` / `+0.0325` nat.
- K2 advantage mean/worst: `+0.0781` / `+0.0185` nat.
- Worst Modal K1/full ratio: `1.0041`; K2/full: `1.0076`.

This is a controlled small-model scaling screen. It increases dense matrix dimensions while holding 64-expert/top-8 routing, depth, corpus, token budget, and paired initialization policy fixed.
