from __future__ import annotations

from pre_qwen_certification.reality_gate import PlateauRule, plateau_window_status


def rule() -> PlateauRule:
    return PlateauRule(
        evaluation_interval=100,
        minimum_steps=300,
        maximum_steps=1000,
        window=4,
        patience=2,
        maximum_negative_slope_per_step=0.00002,
        maximum_positive_slope_per_step=0.00002,
        maximum_window_improvement=0.02,
        maximum_window_range=0.03,
        maximum_route_l1_change=0.05,
        validation_windows_per_document=1,
        validation_seed=1,
    )


def test_stable_history_passes_plateau_window() -> None:
    history = [
        {"step": 100, "validation_loss": 2.0000, "route_distribution": [1.0, 1.0]},
        {"step": 200, "validation_loss": 1.9985, "route_distribution": [1.01, 0.99]},
        {"step": 300, "validation_loss": 1.9975, "route_distribution": [1.00, 1.00]},
        {"step": 400, "validation_loss": 1.9970, "route_distribution": [1.01, 0.99]},
    ]
    status = plateau_window_status(history, rule())
    assert status["eligible"]
    assert status["plateau"]


def test_descending_history_does_not_pass() -> None:
    history = [
        {"step": 100, "validation_loss": 2.0, "route_distribution": [1.0, 1.0]},
        {"step": 200, "validation_loss": 1.9, "route_distribution": [1.0, 1.0]},
        {"step": 300, "validation_loss": 1.8, "route_distribution": [1.0, 1.0]},
        {"step": 400, "validation_loss": 1.7, "route_distribution": [1.0, 1.0]},
    ]
    assert not plateau_window_status(history, rule())["plateau"]


def test_materially_rising_history_does_not_pass() -> None:
    history = [
        {"step": 100, "validation_loss": 1.90, "route_distribution": [1.0, 1.0]},
        {"step": 200, "validation_loss": 1.92, "route_distribution": [1.0, 1.0]},
        {"step": 300, "validation_loss": 1.94, "route_distribution": [1.0, 1.0]},
        {"step": 400, "validation_loss": 1.96, "route_distribution": [1.0, 1.0]},
    ]
    assert not plateau_window_status(history, rule())["plateau"]
