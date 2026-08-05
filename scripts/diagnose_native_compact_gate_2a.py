#!/usr/bin/env python3
"""Post-hoc matrix diagnostics for the frozen Native Compact Gate 2A checkpoints.

This analysis is explicitly non-preregistered. It cannot change the frozen Gate 2A
verdict; it only diagnoses whether nominal low-rank residual capacity was actually
used during native training.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pre_qwen_certification.native_compact import (
    CANDIDATES,
    NativeArchitectureSpec,
    build_paired_candidate_models,
)
from pre_qwen_certification.tiny_lm import TinyLMConfig


PROJECTIONS = ("gate", "up", "down")
NATIVE = "native-shared-rank"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tiny_config(values: Mapping[str, Any]) -> TinyLMConfig:
    return TinyLMConfig(
        seq_len=int(values["seq_len"]),
        batch_size=int(values["batch_size"]),
        d_model=int(values["d_model"]),
        n_heads=int(values["n_heads"]),
        n_layers=int(values["n_layers"]),
        d_ff=int(values["d_ff"]),
        n_experts=int(values["n_experts"]),
        top_k=int(values["top_k"]),
        teacher_steps=int(values["training_steps"]),
        student_steps=0,
        learning_rate=float(values["learning_rate"]),
        student_learning_rate=float(values["learning_rate"]),
        weight_decay=float(values["weight_decay"]),
        aux_weight=float(values["aux_weight"]),
        grad_clip=float(values["grad_clip"]),
    )


def verify_core_source_unchanged(scientific_commit: str) -> None:
    protected = [
        "pre_qwen_certification/native_compact.py",
        "pre_qwen_certification/heterogeneous_rank.py",
        "pre_qwen_certification/modal.py",
        "pre_qwen_certification/tiny_lm.py",
    ]
    result = subprocess.run(
        ["git", "diff", "--quiet", scientific_commit, "HEAD", "--", *protected],
        cwd=ROOT,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            "core architecture code changed after the scientific source commit; "
            "initial-state reconstruction would not be exact"
        )


def pairwise_cosine(matrices: Sequence[torch.Tensor]) -> dict[str, float]:
    vectors = torch.stack([matrix.reshape(-1).double() for matrix in matrices])
    vectors = vectors / vectors.norm(dim=1, keepdim=True).clamp_min(1e-12)
    gram = vectors @ vectors.T
    mask = torch.triu(torch.ones_like(gram, dtype=torch.bool), diagonal=1)
    values = gram[mask]
    return {
        "mean": float(values.mean()),
        "minimum": float(values.min()),
        "maximum": float(values.max()),
    }


def centered_variance_ratio(matrices: Sequence[torch.Tensor]) -> float:
    bank = torch.stack([matrix.double() for matrix in matrices])
    mean = bank.mean(dim=0)
    centered = (bank - mean).square().sum()
    total = bank.square().sum().clamp_min(1e-12)
    return float(centered / total)


def spectrum_metrics(matrix: torch.Tensor) -> dict[str, float | int]:
    singular = torch.linalg.svdvals(matrix.double())
    energy = singular.square()
    total = energy.sum().clamp_min(1e-24)
    normalized = energy / total
    cumulative = torch.cumsum(normalized, dim=0)
    rank95 = int(
        torch.searchsorted(
            cumulative,
            torch.tensor(0.95, dtype=cumulative.dtype, device=cumulative.device),
        ).item()
    ) + 1
    stable_rank = float(total / energy[0].clamp_min(1e-24))
    return {
        "top1_energy": float(normalized[0]),
        "top2_energy": float(normalized[: min(2, normalized.numel())].sum()),
        "top3_energy": float(normalized[: min(3, normalized.numel())].sum()),
        "stable_rank": stable_rank,
        "rank95": rank95,
    }


def native_projection(
    state: Mapping[str, torch.Tensor],
    layer_id: int,
    projection: str,
    n_experts: int,
) -> tuple[torch.Tensor, list[torch.Tensor], list[torch.Tensor]]:
    common = state[f"blocks.{layer_id}.moe.common_{projection}"].double()
    residuals: list[torch.Tensor] = []
    matrices: list[torch.Tensor] = []
    for expert in range(n_experts):
        left = state[f"blocks.{layer_id}.moe.{projection}_left.{expert}"].double()
        right = state[f"blocks.{layer_id}.moe.{projection}_right.{expert}"].double()
        residual = left @ right
        residuals.append(residual)
        matrices.append(common + residual)
    return common, residuals, matrices


def conventional_projection(
    state: Mapping[str, torch.Tensor],
    layer_id: int,
    projection: str,
) -> list[torch.Tensor]:
    bank = state[f"blocks.{layer_id}.moe.{projection}"].double()
    return [bank[expert] for expert in range(bank.shape[0])]


def analyze_state(
    state: Mapping[str, torch.Tensor],
    *,
    candidate: str,
    config: TinyLMConfig,
    nominal_rank: int,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for layer_id in range(config.n_layers):
        for projection in PROJECTIONS:
            if candidate == NATIVE:
                common, residuals, matrices = native_projection(
                    state,
                    layer_id,
                    projection,
                    config.n_experts,
                )
                residual_spectra = [spectrum_metrics(residual) for residual in residuals]
                records.append(
                    {
                        "candidate": candidate,
                        "layer": layer_id,
                        "projection": projection,
                        "pairwise_cosine": pairwise_cosine(matrices),
                        "centered_expert_variance_ratio": centered_variance_ratio(matrices),
                        "effective_matrix_rank95_mean": float(
                            np.mean([spectrum_metrics(matrix)["rank95"] for matrix in matrices])
                        ),
                        "residual_common_norm_ratio_mean": float(
                            np.mean(
                                [
                                    float(residual.norm() / common.norm().clamp_min(1e-12))
                                    for residual in residuals
                                ]
                            )
                        ),
                        "residual_effective_norm_ratio_mean": float(
                            np.mean(
                                [
                                    float(residual.norm() / matrix.norm().clamp_min(1e-12))
                                    for residual, matrix in zip(residuals, matrices, strict=True)
                                ]
                            )
                        ),
                        "residual_top1_energy_mean": float(
                            np.mean([float(value["top1_energy"]) for value in residual_spectra])
                        ),
                        "residual_top2_energy_mean": float(
                            np.mean([float(value["top2_energy"]) for value in residual_spectra])
                        ),
                        "residual_stable_rank_mean": float(
                            np.mean([float(value["stable_rank"]) for value in residual_spectra])
                        ),
                        "residual_rank95_mean": float(
                            np.mean([int(value["rank95"]) for value in residual_spectra])
                        ),
                        "nominal_rank": int(nominal_rank),
                    }
                )
            else:
                matrices = conventional_projection(state, layer_id, projection)
                records.append(
                    {
                        "candidate": candidate,
                        "layer": layer_id,
                        "projection": projection,
                        "pairwise_cosine": pairwise_cosine(matrices),
                        "centered_expert_variance_ratio": centered_variance_ratio(matrices),
                        "effective_matrix_rank95_mean": float(
                            np.mean([spectrum_metrics(matrix)["rank95"] for matrix in matrices])
                        ),
                    }
                )
    return records


def summarize(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    def mean(path: Sequence[str]) -> float:
        values: list[float] = []
        for record in records:
            current: Any = record
            for key in path:
                current = current[key]
            values.append(float(current))
        return float(np.mean(values))

    result: dict[str, Any] = {
        "banks": len(records),
        "pairwise_cosine_mean": mean(("pairwise_cosine", "mean")),
        "centered_expert_variance_ratio_mean": mean(("centered_expert_variance_ratio",)),
        "effective_matrix_rank95_mean": mean(("effective_matrix_rank95_mean",)),
    }
    if records and records[0]["candidate"] == NATIVE:
        nominal_rank = int(records[0]["nominal_rank"])
        stable_rank = mean(("residual_stable_rank_mean",))
        result.update(
            {
                "nominal_rank": nominal_rank,
                "residual_common_norm_ratio_mean": mean(("residual_common_norm_ratio_mean",)),
                "residual_effective_norm_ratio_mean": mean(("residual_effective_norm_ratio_mean",)),
                "residual_top1_energy_mean": mean(("residual_top1_energy_mean",)),
                "residual_top2_energy_mean": mean(("residual_top2_energy_mean",)),
                "residual_stable_rank_mean": stable_rank,
                "residual_rank95_mean": mean(("residual_rank95_mean",)),
                "stable_rank_utilization_ratio": stable_rank / nominal_rank,
            }
        )
    return result


def write_report(path: Path, payload: Mapping[str, Any]) -> None:
    lines = [
        "# Diagnóstico pós-hoc — Native Compact Gate 2A",
        "",
        "**Status:** análise pós-hoc; não altera o veredito pré-registrado.",
        "",
        f"**Veredito congelado:** `{payload['frozen_verdict']}`",
        "",
    ]
    for scale, values in payload["scales"].items():
        lines.extend(
            [
                f"## Escala `{scale}`",
                "",
                "| Estado/candidato | Cosine entre experts | Variância centrada | Residual/common | Top-1 energia residual | Stable rank residual | Uso do rank nominal |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for phase in ("initial", "final"):
            for candidate in CANDIDATES:
                summary = values[phase][candidate]
                lines.append(
                    "| {phase}/{candidate} | {cos:.5f} | {variance:.5f} | {ratio} | {top1} | {stable} | {util} |".format(
                        phase=phase,
                        candidate=candidate,
                        cos=float(summary["pairwise_cosine_mean"]),
                        variance=float(summary["centered_expert_variance_ratio_mean"]),
                        ratio=(
                            f"{float(summary['residual_common_norm_ratio_mean']):.5f}"
                            if candidate == NATIVE
                            else "—"
                        ),
                        top1=(
                            f"{float(summary['residual_top1_energy_mean']):.5f}"
                            if candidate == NATIVE
                            else "—"
                        ),
                        stable=(
                            f"{float(summary['residual_stable_rank_mean']):.3f}"
                            if candidate == NATIVE
                            else "—"
                        ),
                        util=(
                            f"{float(summary['stable_rank_utilization_ratio']):.2%}"
                            if candidate == NATIVE
                            else "—"
                        ),
                    )
                )
        diagnosis = values["diagnosis"]
        lines.extend(
            [
                "",
                f"- rank-collapse signal: `{diagnosis['optimization_rank_collapse_signal']}`;",
                f"- effective-expert-similarity signal: `{diagnosis['effective_expert_similarity_signal']}`;",
                "",
            ]
        )
    lines.extend(
        [
            "## Interpretação limitada",
            "",
            "O diagnóstico mede pesos congelados, não causalidade. Ele mostra se o rank nominal foi utilizado,",
            "mas não prova sozinho se a causa é inicialização, otimização, regularização ou a função do corpus.",
            "A próxima intervenção elegível é um controle simples de utilização dos modos; nested/dynamic rank",
            "e checkpoint real continuam bloqueados.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-manifest", type=Path, required=True)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    config: dict[str, Any] = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    data_manifest = json.loads(args.data_manifest.read_text(encoding="utf-8"))
    metrics = json.loads((args.results_dir / "metrics.json").read_text(encoding="utf-8"))
    scientific_commit = str(metrics["source_commit"])
    verify_core_source_unchanged(scientific_commit)
    if str(metrics["decision"]["verdict"]) != "NATIVE_COMPACT_GATE_2A_FAIL":
        raise RuntimeError("post-hoc diagnostic is frozen to the Gate 2A failure result")

    scales: dict[str, Any] = {}
    detailed: list[dict[str, Any]] = []
    for scale, scale_cfg in config["scales"].items():
        model_config = tiny_config(scale_cfg["model"])
        architecture = NativeArchitectureSpec(
            native_rank=int(scale_cfg["native_rank"]),
            narrow_d_ff=int(scale_cfg["narrow_d_ff"]),
        )
        phase_records: dict[str, dict[str, list[dict[str, Any]]]] = {
            "initial": {candidate: [] for candidate in CANDIDATES},
            "final": {candidate: [] for candidate in CANDIDATES},
        }
        for seed in [int(value) for value in config["seeds"]]:
            initial_models = build_paired_candidate_models(
                int(data_manifest["vocab_size"]),
                model_config,
                architecture,
                seed=seed,
            )
            checkpoint_path = args.results_dir / scale / f"frozen-candidates-seed-{seed}.pt"
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            if str(checkpoint["source_commit"]) != scientific_commit:
                raise RuntimeError(f"source commit mismatch in {checkpoint_path}")
            for candidate in CANDIDATES:
                initial_records = analyze_state(
                    initial_models[candidate].state_dict(),
                    candidate=candidate,
                    config=model_config,
                    nominal_rank=int(scale_cfg["native_rank"]),
                )
                final_records = analyze_state(
                    checkpoint["final_states"][candidate],
                    candidate=candidate,
                    config=model_config,
                    nominal_rank=int(scale_cfg["native_rank"]),
                )
                for record in initial_records:
                    detailed.append(
                        {
                            **record,
                            "scale": scale,
                            "seed": seed,
                            "phase": "initial",
                        }
                    )
                for record in final_records:
                    detailed.append(
                        {
                            **record,
                            "scale": scale,
                            "seed": seed,
                            "phase": "final",
                        }
                    )
                phase_records["initial"][candidate].extend(initial_records)
                phase_records["final"][candidate].extend(final_records)

        summaries = {
            phase: {
                candidate: summarize(records)
                for candidate, records in candidate_records.items()
            }
            for phase, candidate_records in phase_records.items()
        }
        final_native = summaries["final"][NATIVE]
        diagnosis = {
            "optimization_rank_collapse_signal": bool(
                float(final_native["residual_top1_energy_mean"]) >= 0.90
                and float(final_native["stable_rank_utilization_ratio"]) <= 0.25
            ),
            "effective_expert_similarity_signal": bool(
                float(final_native["pairwise_cosine_mean"]) >= 0.95
                and float(final_native["centered_expert_variance_ratio_mean"]) <= 0.05
            ),
        }
        scales[scale] = {
            **summaries,
            "diagnosis": diagnosis,
        }

    payload = {
        "schema_version": "native-compact-gate-2a-posthoc-diagnostics-v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "posthoc_not_preregistered": True,
        "cannot_change_frozen_verdict": True,
        "frozen_verdict": str(metrics["decision"]["verdict"]),
        "scientific_source_commit": scientific_commit,
        "results_commit_input_sha256": sha256_file(args.results_dir / "metrics.json"),
        "config_sha256": sha256_file(args.config),
        "data_manifest_sha256": sha256_file(args.data_manifest),
        "scales": scales,
        "detailed_records": detailed,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_report(args.report, payload)
    print(args.output_json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
