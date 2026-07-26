# Test 2.0 — modal MoE trainability under architectural constraint

**Decision:** **PASS**

| Variant | Parameters | Expert parameter ratio | Ideal expert compute ratio | Validation loss | Loss vs baseline | Utilization entropy |
|---|---:|---:|---:|---:|---:|---:|
| baseline | 1,270,560 | 100.00% | 100.00% | 2.3946 | 1.000× | 0.978 |
| modal-k1 | 238,464 | 12.50% | 50.00% | 2.4415 | 1.020× | 0.962 |
| modal-k2 | 312,288 | 18.75% | 75.00% | 2.4139 | 1.008× | 0.978 |
| modal-k3 | 386,112 | 25.00% | 100.00% | 2.4274 | 1.014× | 0.980 |

The experiment asks whether optimization can coordinate experts around shared full-rank matrix modes when that structure is present from initialization. It does not claim transfer to billion-parameter training from a character-level task.
The modal `down` projection is executed after aggregation, so the tested graph matches the intended future kernel algebra rather than reconstructing expert matrices.
