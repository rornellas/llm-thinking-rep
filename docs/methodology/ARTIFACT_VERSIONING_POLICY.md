# Artifact versioning policy

**Status:** mandatory for this research repository.

## Rule

Every artifact that supports, changes, reproduces, audits, or communicates a scientific or engineering conclusion must be committed to Git before the work is considered delivered.

This includes:

- source code and tests;
- frozen configurations and preregistrations;
- raw and aggregated metrics;
- per-seed and per-document records needed for recalculation;
- checkpoints needed to reproduce a verdict;
- logs and environment records;
- independent and adversarial audits;
- manifests and cryptographic hashes;
- reports, handoffs, decision ledgers, and verdicts;
- delivery archives, patches, and Git bundles that were generated or shared.

## Operational consequences

1. A sandbox path, chat attachment, Actions artifact, or local workspace is not a durable delivery by itself.
2. Every experiment must write into a versioned result directory and produce a `sha256sums.txt` or equivalent manifest.
3. Negative, failed, aborted, and superseded results remain versioned; they must not be silently overwritten.
4. Large files below GitHub's hard file limit are committed directly. Files above that limit require Git LFS or a deliberately versioned external object store plus an immutable manifest; omission is not allowed.
5. Derived archives do not replace their unpacked canonical files. When an archive is generated or supplied, both its exact bytes and its SHA-256 are versioned.
6. A handoff is incomplete unless the referenced commits and artifacts are present in the repository.
7. Important claims continue to follow `IMPORTANT_CLAIM_VERIFICATION_STANDARD.md`.

## Required experiment layout

```text
results/<experiment>/<version>/
  config.resolved.yaml
  metrics.json
  raw-or-per-seed records
  checkpoints or immutable checkpoint references
  logs/
  environment.txt
  VERDICT.md
  sha256sums.txt
  adversarial-audit/
```

## Completion gate

Work is considered complete only when all of the following are true:

- the branch and commit SHA are recorded;
- the working tree is clean;
- manifests verify;
- relevant tests pass;
- every artifact referenced in the report or handoff resolves to a repository path;
- the PR description or commit message states what was preserved.

## Enforcement

Any future run that cannot persist all required artifacts must terminate as `INCOMPLETE_ARTIFACT_PERSISTENCE`; it may not emit a scientific `PASS` or `FAIL` verdict as if the run were durably reproducible.
