# Research artifact manifest — 2026-07-28

This branch versions the certification source, preregistrations, experiment runners, aggregate metrics, verdicts, independent auditors, reproducibility scripts, and continuity handoffs.

## Canonical remote state

- Branch: `agent/pre-qwen-certification-v2`
- Integrated teacher-width commit: `93ade7827598cd46c31e3cb01d06162a78feb424`
- Teacher-width result: `TEACHER_WIDTH_65_REPLICATION_PASS`
- Global decision retained: `NO_GO_FOR_OLMOE_OR_QWEN`

## Directly versioned teacher-width evidence

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

## Transport packages and known hashes

| Artifact | SHA-256 |
|---|---|
| `pre-qwen-teacher-width-fresh-v1-results.zip` | `f3d5d8d8104c429d89118bae59ba5ffc1fd7e93396229d9e8cb925b6eccf42a7` |
| `pre-qwen-teacher-width-fresh-v1.git.bundle` | `8f0ca3009e09f2130aad564fc52bb5597e5de12c356fa53d27b0b875545f8036` |
| `HANDOFF_MODAL_MOE_RESEARCH_2026-07-28_FULL.md` | `470b1b04801bbe2d163016ef593b553fe43dc3dd2274c26b7a732bcdcb2357fd` |
| `HANDOFF_MODAL_MOE_2026-07-28_PACKAGE.zip` | `1f85560312ec118f28d4ef910db3fee6e9d92df3f66a189e207c15c6ac954aa8` |

## Full-artifact regeneration

The repository includes a token-gated GitHub Actions workflow that regenerates the five frozen teacher-width seeds, performs aggregation and independent audit, packages source/results/checkpoints/logs/history, writes `artifacts/2026-07-28/ARTIFACT_INDEX.json`, and commits every generated artifact back to the research branch.

Trigger token:

```text
/regenerate-all-artifacts-v2-65pass-af08
```

A fresh regeneration was requested after commit `93ade7827598cd46c31e3cb01d06162a78feb424`. The workflow result must be inspected before claiming that all regenerated binary artifacts are present and valid.
