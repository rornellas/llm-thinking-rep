"""Calibrate layer-output error against closed-loop language-model damage."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import platform
import sys
from typing import Any, Sequence

import numpy as np
import torch
import yaml

from .controlled_transplant import _load, _tiny_config
from .controlled_transplant_staged import _development_corpus, _load_frozen
from .tiny_lm import evaluate_closed_loop, install_output_perturbation


def run_local_global_calibration(
    controlled_config_path: Path,
    frozen_dir: Path,
    output_dir: Path,
    *,
    levels: Sequence[float] = (0.0, 0.05, 0.10, 0.20, 0.30, 0.40),
) -> dict[str, Any]:
    config = _load(controlled_config_path)
    tiny = _tiny_config(config["model"])
    torch.set_num_threads(int(config.get("threads", 2)))
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    corpus = _development_corpus(config)
    data = config["data"]
    layer_id = int(config["transplant"]["layer_id"])
    rows: list[dict[str, float | int]] = []
    for seed in [int(value) for value in config["seeds"]]:
        teacher, _, _, _, _ = _load_frozen(config, frozen_dir, seed)
        for index, level in enumerate(levels):
            candidate = install_output_perturbation(
                teacher,
                layer_id=layer_id,
                target_nrmse=float(level),
                seed=seed + 50_000 + index * 100,
            )
            metrics, _ = evaluate_closed_loop(
                teacher,
                candidate,
                corpus,
                split="development",
                windows_per_document=int(data["development_windows_per_document"]),
                seed=seed + 3000,
            )
            rows.append(
                {
                    "seed": seed,
                    "target_local_nrmse": float(level),
                    "loss_delta": float(metrics["loss_delta"]),
                    "kl_teacher_to_candidate": float(metrics["kl_teacher_to_candidate"]),
                    "perplexity_ratio": float(metrics["perplexity_ratio"]),
                    "top1_agreement": float(metrics["top1_agreement"]),
                }
            )
    curve: list[dict[str, float]] = []
    for level in levels:
        selected = [row for row in rows if row["target_local_nrmse"] == float(level)]
        curve.append(
            {
                "target_local_nrmse": float(level),
                "loss_delta_mean": float(np.mean([row["loss_delta"] for row in selected])),
                "loss_delta_worst": float(max(row["loss_delta"] for row in selected)),
                "kl_mean": float(np.mean([row["kl_teacher_to_candidate"] for row in selected])),
                "kl_worst": float(max(row["kl_teacher_to_candidate"] for row in selected)),
                "perplexity_ratio_worst": float(max(row["perplexity_ratio"] for row in selected)),
                "top1_agreement_worst": float(min(row["top1_agreement"] for row in selected)),
            }
        )
    x = np.asarray([row["target_local_nrmse"] for row in curve])
    kl = np.asarray([row["kl_mean"] for row in curve])
    loss = np.asarray([row["loss_delta_mean"] for row in curve])
    payload = {
        "metadata": {
            "source_protocol": config["protocol_version"],
            "scope": "development-only calibration; sealed data not accessed",
            "levels": [float(value) for value in levels],
            "config_sha256": hashlib.sha256(controlled_config_path.read_bytes()).hexdigest(),
            "frozen_checkpoint_sha256": {
                str(seed): hashlib.sha256(
                    (frozen_dir / f"frozen-candidates-seed-{seed}.pt").read_bytes()
                ).hexdigest()
                for seed in [int(value) for value in config["seeds"]]
            },
        },
        "curve": curve,
        "runs": rows,
        "diagnostics": {
            "local_nrmse_to_kl_correlation": float(np.corrcoef(x, kl)[0, 1]),
            "local_nrmse_to_loss_delta_correlation": float(np.corrcoef(x, loss)[0, 1]),
            "kl_nondecreasing": bool(np.all(np.diff(kl) >= -1e-8)),
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    lines = [
        "# Local-to-global error calibration",
        "",
        "Development-only perturbation study; no sealed data was accessed.",
        "",
        "| Target local NRMSE | Mean Δloss | Worst Δloss | Mean KL | Worst KL | Worst PPL ratio | Worst top-1 |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in curve:
        lines.append(
            f"| {row['target_local_nrmse']:.3f} | {row['loss_delta_mean']:+.5f} | "
            f"{row['loss_delta_worst']:+.5f} | {row['kl_mean']:.5f} | {row['kl_worst']:.5f} | "
            f"{row['perplexity_ratio_worst']:.5f} | {row['top1_agreement_worst']:.3%} |"
        )
    lines += [
        "",
        f"- Correlation local NRMSE -> KL: `{payload['diagnostics']['local_nrmse_to_kl_correlation']:.5f}`.",
        f"- Correlation local NRMSE -> loss delta: `{payload['diagnostics']['local_nrmse_to_loss_delta_correlation']:.5f}`.",
        "",
        "These curves calibrate local thresholds for this controlled model only. They must be remeasured per layer and checkpoint on OLMoE/Qwen.",
    ]
    (output_dir / "VERDICT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (output_dir / "environment.txt").write_text(
        f"python={sys.version.replace(chr(10), ' ')}\n"
        f"platform={platform.platform()}\n"
        f"torch={torch.__version__}\n"
        f"numpy={np.__version__}\n"
        f"threads={torch.get_num_threads()}\n",
        encoding="utf-8",
    )
    files = sorted(
        path for path in output_dir.iterdir()
        if path.is_file() and path.name != "sha256sums.txt"
    )
    (output_dir / "sha256sums.txt").write_text(
        "\n".join(
            f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.name}"
            for path in files
        ) + "\n",
        encoding="utf-8",
    )
    return payload
