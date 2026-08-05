from __future__ import annotations

from pathlib import Path

import yaml


def test_native_compact_gate_2a_is_frozen_and_pareto_relevant() -> None:
    config = yaml.safe_load(Path("configs/native_compact_gate_2a.yaml").read_text(encoding="utf-8"))
    assert config["protocol_version"] == "native-compact-gate-2a-v1"
    assert config["status"] == "preregistered_before_data_preparation_or_scientific_training"
    assert config["seeds"] == [202781, 212789, 222793, 232801]
    assert config["gates"]["minimum_expert_parameter_advantage"] == 0.15
    assert config["gates"]["primary_vs_narrow_hypothesis_loss_ucb_max"] == 0.010
    assert config["gates"]["signal_min_scales"] == 1
    assert "NO_GO_FOR_OLMOE_OR_QWEN" in config["decision_policy"]["consequence"]

    for scale in config["scales"].values():
        model = scale["model"]
        d = int(model["d_model"])
        h = int(model["d_ff"])
        experts = int(model["n_experts"])
        top_k = int(model["top_k"])
        rank = int(scale["native_rank"])
        narrow = int(scale["narrow_d_ff"])
        native_parameter_ratio = 1 / experts + (h + d) * rank / (h * d)
        native_compute_ratio = 1 / top_k + (h + d) * rank / (h * d)
        narrow_ratio = narrow / h
        assert native_compute_ratio <= narrow_ratio
        assert narrow_ratio - native_parameter_ratio >= 0.15
        assert abs(native_compute_ratio - 0.625) < 1e-12
        assert int(scale["evaluation_interval"]) < int(model["training_steps"])
