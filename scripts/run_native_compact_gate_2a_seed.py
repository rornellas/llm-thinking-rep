#!/usr/bin/env python3
"""Entry point with a tracked-source cleanliness guard for Native Compact Gate 2A."""
from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import run_native_compact_gate_2a_seed_impl as runner


def require_clean() -> None:
    expected = os.environ.get("SOURCE_COMMIT")
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    if expected and head != expected:
        raise RuntimeError(f"source commit drift: expected {expected}, observed {head}")

    protected = [
        ".github/workflows/run-native-compact-gate-2a.yml",
        "configs/native_compact_gate_2a.yaml",
        "docs/prereg/NATIVE_COMPACT_GATE_2A.md",
        "pre_qwen_certification",
        "scripts",
        "tests",
    ]
    commands = (
        ["git", "diff", "--quiet", "HEAD", "--", *protected],
        ["git", "diff", "--cached", "--quiet", "--", *protected],
    )
    for command in commands:
        result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=False)
        if result.returncode != 0:
            changed = subprocess.run(
                ["git", "status", "--short", "--untracked-files=no", "--", *protected],
                cwd=ROOT,
                capture_output=True,
                text=True,
                check=False,
            ).stdout.strip()
            raise RuntimeError(
                "scientific source/config/test files must be committed and clean"
                + (f": {changed}" if changed else f" (git rc={result.returncode})")
            )


runner.require_clean = require_clean

if __name__ == "__main__":
    raise SystemExit(runner.main())
