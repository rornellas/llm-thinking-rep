#!/usr/bin/env python3
"""Run the balanced learned controller in OLMoE's 64-expert/top-8 geometry."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def load_balanced():
    path = Path(__file__).with_name("test_2_7c_balanced_calibrated_controller.py")
    spec = importlib.util.spec_from_file_location("scale64_balanced_controller", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


balanced = load_balanced()
base = balanced.base
original_load_module = base.load_module
original_write_outputs = base.write_outputs


def scale64_load_module(name: str, path: Path):
    module = original_load_module(name, path)
    if name == "controller_base_source":
        original_config = module.Config

        class Scale64Config(original_config):
            def __init__(self, *args, **kwargs):
                kwargs.setdefault("n_experts", 64)
                kwargs.setdefault("top_k", 8)
                super().__init__(*args, **kwargs)

        module.Config = Scale64Config
    return module


def scale64_write_outputs(output_dir: Path, payload: dict):
    payload.setdefault("metadata", {})["n_experts"] = 64
    payload["metadata"]["top_k"] = 8
    payload["metadata"]["geometry_note"] = "This run uses the same expert count and routing top-k as OLMoE-1B-7B-0924. Model width and corpus remain small."
    original_write_outputs(output_dir, payload)


base.load_module = scale64_load_module
base.write_outputs = scale64_write_outputs


if __name__ == "__main__":
    raise SystemExit(base.main())
