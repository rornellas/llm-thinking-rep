from __future__ import annotations

from pathlib import Path

import yaml


def test_reality_gate_protocol_is_frozen_and_budgeted() -> None:
    config = yaml.safe_load(Path("configs/reality_gate_1a.yaml").read_text())
    assert config["protocol_version"] == "reality-gate-1a-static-heterogeneous-rank-v1"
    assert config["status"] == "preregistered_before_data_preparation_or_scientific_training"
    assert config["seeds"] == [111731, 121747, 131759, 141767]
    assert config["data"]["dataset_source"] == "Salesforce/wikitext"
    assert config["data"]["dataset_subset"] == "wikitext-2-raw-v1"
    assert config["data"]["dataset_revision"] == "b08601e04326c79dfdd32d625aee71d232d685c3"
    assert abs(sum(config["training"]["joint_weights"].values()) - 1.0) < 1e-12
    for scale in config["scales"].values():
        model = scale["model"]
        rank = scale["uniform_rank"]
        shared = 3 * model["d_ff"] * model["d_model"]
        factors = 3 * model["top_k"] * rank * (model["d_ff"] + model["d_model"])
        teacher = 3 * model["top_k"] * model["d_ff"] * model["d_model"]
        assert abs((shared + factors) / teacher - 0.625) < 1e-12
        assert scale["max_rank"] > rank
        assert scale["plateau"]["minimum_steps"] < scale["plateau"]["maximum_steps"]
        assert "maximum_positive_slope_per_step" in scale["plateau"]
    assert config["gates"]["signal_min_scales"] == 1
    assert config["gates"]["routing_loss_difference_ucb_max"] == 0.0
