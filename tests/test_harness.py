from __future__ import annotations

import torch

from pre_qwen_certification.harness import capture_layer, run_fault_matrix
from pre_qwen_certification.modal import MoEGeometry, ScalarModalMoE, set_seed


def test_capture_replay_is_exact_and_faults_are_detected() -> None:
    set_seed(37)
    geometry = MoEGeometry(d_model=12, d_ff=18, n_experts=8, top_k=2)
    model = ScalarModalMoE(geometry, rank=2)
    inputs = torch.randn(64, geometry.d_model)
    capture = capture_layer(
        model,
        inputs,
        document_ids=[f"doc-{index // 8}" for index in range(len(inputs))],
        sequence_ids=[f"seq-{index // 4}" for index in range(len(inputs))],
        token_positions=torch.arange(len(inputs)),
        layer_id=7,
        residual=inputs * 0.1,
    )
    matrix = run_fault_matrix(model, capture)
    assert matrix["none"]["nrmse"] <= 1e-10
    for name, metrics in matrix.items():
        if name != "none":
            assert metrics["nrmse"] >= 0.01, name
