# Preregistration — teacher-informed narrow width fresh replication v1

**Protocol:** `pre-qwen-teacher-informed-width-fresh-replication-v1.1`  
**Status:** preregistered fresh development replication  
**Frozen consequence:** no outcome changes `NO_GO_FOR_OLMOE_OR_QWEN`.

## Question

Does a conventional narrow MoE initialized from the teacher's highest-magnitude
SwiGLU coordinates reproducibly enter the predefined fidelity region at 65% of
expert width on fresh teachers and fresh documents?

## Design

Five new teacher seeds and a new deterministic corpus family are used. Every
candidate receives identical captured layer inputs, frozen teacher routing,
optimizer budgets, batch sequences, and closed-loop loss weights. The only
candidate difference is retained expert width.

```text
35%  negative anchor
50%  lower-compute comparator
65%  primary
75%  capacity control
100% full continuation control
```

Teacher coordinates are ranked independently per expert by the combined squared
norm of the matching gate row, up row, and down column. The selected tensors are
then fully trainable; this is an initialization, not permanent pruning.

## Isolation

- Train documents are materialized before teacher training.
- Candidate checkpoints are frozen before hypothesis and OOD documents are
  materialized.
- Splits are by document, with exact and five-word-shingle overlap audits.
- The evaluation unit is document; windows inside a document are averaged.
- Statistics use a crossed seed/document bootstrap with 20,000 draws.
- A second implementation must reconstruct pairings, intervals, gates, and the
  decision from raw seed files.

## Primary gates

The 65% candidate must satisfy:

1. hypothesis UCB95 `<= +0.030 nat`;
2. OOD UCB95 `<= +0.050 nat`;
3. every seed hypothesis mean `<= +0.060 nat`;
4. paired 65%-minus-50% UCB95 `<= 0`;
5. exact parameter ratio `65%`;
6. exact routed matrix-compute ratio `65%`;
7. full control hypothesis UCB95 `<= +0.010 nat`;
8. full control OOD UCB95 `<= +0.015 nat`;
9. clean split audit;
10. independent audit PASS.

The 75% capacity control must independently have hypothesis UCB95 `<= +0.030`.
The 35% anchor is expected to fail; its LCB95 should be at least `+0.030` to
confirm that the fresh task still exposes the bottleneck.

## Interpretation boundary

A PASS establishes only a reproducible capacity region for a conventional
teacher-informed narrow student in this controlled scale. It is not evidence for
Modal-MoE, shared-basis compression, latency improvement, or transfer to a real
checkpoint.

## Engineering correction before evaluation

The first execution attempt stopped before any hypothesis or OOD metric because
the frozen train vocabulary did not contain all characters used by the OOD
documents. No seed produced an evaluation JSON. Revision v1.1 adds a train-only
character-coverage line, adds a frozen-vocabulary compatibility test, changes no
candidate, gate, width, seed, or held-out document, and restarts all teachers and
students from scratch.
