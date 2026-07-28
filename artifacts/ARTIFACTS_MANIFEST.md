# Research artifact manifest — 2026-07-28

This branch versions the source code, preregistration, aggregated metrics, verdicts, independent auditor, reproducibility scripts, and the full continuity handoff for the teacher-informed-width replication.

## Canonical Git state

- Branch: `agent/teacher-width-65-replication`
- Scientific parent commit: `8f8c472caf54ccdb208a7323114fc4cd519385e7`
- Result: `TEACHER_WIDTH_65_REPLICATION_PASS`
- Global decision retained: `NO_GO_FOR_OLMOE_OR_QWEN`

## Versioned evidence

- `configs/pre_qwen_teacher_width_fresh_v1.yaml`
- `docs/prereg/PRE_QWEN_TEACHER_WIDTH_FRESH_V1.md`
- `docs/results/2026-07-28-teacher-informed-width-fresh-replication.md`
- `results/pre-qwen-teacher-width-fresh/v1/metrics.json`
- `results/pre-qwen-teacher-width-fresh/v1/VERDICT.md`
- `results/pre-qwen-teacher-width-fresh/v1/adversarial-audit/VERDICT.md`
- `results/pre-qwen-teacher-width-fresh/v1/sha256sums.txt`
- `scripts/run_teacher_width_fresh_seed.py`
- `scripts/aggregate_teacher_width_fresh.py`
- `scripts/audit_teacher_width_fresh.py`
- `tests/test_teacher_width_fresh.py`
- `docs/HANDOFF_MODAL_MOE_RESEARCH_2026-07-28_FULL.md`

## Preserved binary packages and hashes

The following transport archives were produced from the same research state. They are duplicate packaging artifacts rather than canonical scientific sources:

| Artifact | SHA-256 |
|---|---|
| `pre-qwen-teacher-width-fresh-v1-results.zip` | `f3d5d8d8104c429d89118bae59ba5ffc1fd7e93396229d9e8cb925b6eccf42a7` |
| `pre-qwen-teacher-width-fresh-v1.git.bundle` | `8f0ca3009e09f2130aad564fc52bb5597e5de12c356fa53d27b0b875545f8036` |
| `HANDOFF_MODAL_MOE_RESEARCH_2026-07-28_FULL.md` | `470b1b04801bbe2d163016ef593b553fe43dc3dd2274c26b7a732bcdcb2357fd` |
| `HANDOFF_MODAL_MOE_2026-07-28_PACKAGE.zip` | `1f85560312ec118f28d4ef910db3fee6e9d92df3f66a189e207c15c6ac954aa8` |

Large checkpoints and full per-seed transport archives are deliberately not duplicated as ordinary Git blobs. Their expected hashes remain recorded in `results/pre-qwen-teacher-width-fresh/v1/sha256sums.txt`; the run scripts reproduce them. For long-term binary retention, use Git LFS or an immutable release/Actions artifact while keeping this Git tree as the canonical, reviewable source of truth.
