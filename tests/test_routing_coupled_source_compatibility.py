from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch
import yaml

from pre_qwen_certification.alignment_tolerant import SharedLowRankResidualMoE
from pre_qwen_certification.controlled_transplant import _tiny_config
from pre_qwen_certification.modal import ConventionalSwiGLUMoE
from pre_qwen_certification.routing_coupled import RoutingCoupledResidualMoE
from pre_qwen_certification.tiny_lm import TinyMoELanguageModel, make_narrow_student

ROOT = Path(__file__).resolve().parents[1]


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_paths(seed: int) -> tuple[Path, Path]:
    root = ROOT / "results/pre-qwen-alignment-routing-set/v3"
    nested = root / f"seed-{seed}"
    checkpoint = nested / f"frozen-candidates-seed-{seed}.pt"
    payload = nested / f"seed-{seed}.json"
    if checkpoint.exists() and payload.exists():
        return checkpoint, payload
    return root / f"frozen-candidates-seed-{seed}.pt", root / f"seed-{seed}.json"


def test_v4_loads_the_frozen_v3_checkpoint_and_all_baselines() -> None:
    config = yaml.safe_load(
        (ROOT / "configs/pre_qwen_routing_coupled_v4.yaml").read_text(encoding="utf-8")
    )
    source_config = ROOT / config["source"]["routing_set_config"]
    checkpoint_path, payload_path = _source_paths(91121)
    assert checkpoint_path.exists() and payload_path.exists()
    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    assert _sha(checkpoint_path) == payload["metadata"]["checkpoint_sha256"]
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    assert checkpoint["configuration_sha256"] == _sha(source_config)

    tiny = _tiny_config(config["model"])
    vocabulary = list(checkpoint["vocabulary"])
    teacher = TinyMoELanguageModel(len(vocabulary), tiny)
    teacher.load_state_dict(checkpoint["teacher_state"])
    teacher_moe = teacher.blocks[0].moe
    assert isinstance(teacher_moe, ConventionalSwiGLUMoE)

    base5 = SharedLowRankResidualMoE(tiny.geometry, rank=5)
    base5.load_state_dict(
        checkpoint["candidate_states"][config["source"]["primary_base_state"]]
    )
    base6 = SharedLowRankResidualMoE(tiny.geometry, rank=6)
    base6.load_state_dict(
        checkpoint["candidate_states"][config["source"]["rank6_state"]]
    )
    narrow = make_narrow_student(teacher_moe, d_ff=26)
    narrow.load_state_dict(
        checkpoint["candidate_states"][config["source"]["narrow_state"]]
    )
    full = ConventionalSwiGLUMoE(tiny.geometry)
    full.load_state_dict(
        checkpoint["candidate_states"][config["source"]["full_state"]]
    )
    coupled = RoutingCoupledResidualMoE.from_base(base5, set_dim=8, hidden_dim=8)

    x = torch.randn(17, tiny.d_model)
    with torch.no_grad():
        teacher_output, routing = teacher_moe(x)
        for module in (base5, base6, narrow, full, coupled):
            output, current = module(
                x,
                forced_top_ids=routing.top_ids,
                forced_weights=routing.weights,
            )
            assert output.shape == teacher_output.shape
            assert current.top_ids.shape == routing.top_ids.shape
            assert torch.isfinite(output).all()

    accounting = coupled.accounting()
    assert accounting.parameter_ratio == config["candidates"][0]["expected_parameter_ratio"]
    assert accounting.compute_ratio == config["candidates"][0]["expected_compute_ratio"]
