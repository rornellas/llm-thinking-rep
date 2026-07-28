"""Executable pre-Qwen certification protocol.

This suite is intentionally falsification-oriented.  A PASS means that the
measurement and transplantation harness discriminates exact low-rank expert-axis
structure from near-boundary and negative controls.  It is not a claim about a
real pretrained checkpoint.
"""
from __future__ import annotations

import copy
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import yaml

from .harness import capture_layer, run_fault_matrix, save_capture
from .metrics import (
    crossed_hierarchical_bootstrap,
    summarize_groups,
    tensor_metrics,
)
from .modal import ConventionalSwiGLUMoE, MoEGeometry, ScalarModalMoE
from .synthetic import (
    add_independent_expert_noise,
    evaluate_student,
    input_blind_baseline,
    make_modal_teacher,
    make_negative_teacher,
    monotonicity,
    rank_sweep,
    sample_document_activations,
    shuffled_target_control,
)


class CertificationError(RuntimeError):
    pass


def load_config(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise CertificationError("configuration root must be a mapping")
    required = {
        "protocol_version",
        "geometry",
        "seeds",
        "data",
        "known_truth",
        "optimization",
        "statistics",
        "gates",
    }
    missing = sorted(required - payload.keys())
    if missing:
        raise CertificationError(f"missing configuration keys: {missing}")
    return payload


def _geometry(config: Mapping[str, Any]) -> MoEGeometry:
    geometry = MoEGeometry(
        d_model=int(config["d_model"]),
        d_ff=int(config["d_ff"]),
        n_experts=int(config["n_experts"]),
        top_k=int(config["top_k"]),
    )
    geometry.validate()
    return geometry


def _hash_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_hashes(root: Path) -> dict[str, str]:
    paths = sorted((root / "pre_qwen_certification").glob("*.py"))
    return {str(path.relative_to(root)): _hash_file(path) for path in paths}


def _intervention_metrics(
    teacher: ConventionalSwiGLUMoE,
    student: ScalarModalMoE,
    inputs: torch.Tensor,
    *,
    seed: int,
) -> dict[str, dict[str, float]]:
    teacher.eval()
    with torch.no_grad():
        target, routing = teacher(inputs)
    result: dict[str, dict[str, float]] = {}
    for intervention in ("mean-code", "shuffle-code", "zero-residual"):
        modified = copy.deepcopy(student)
        with torch.no_grad():
            tensors = (
                modified.gate_codes,
                modified.up_codes,
                modified.down_codes,
            )
            if intervention == "mean-code":
                for codes in tensors:
                    codes.copy_(codes.mean(dim=0, keepdim=True).expand_as(codes))
            elif intervention == "shuffle-code":
                permutation = torch.randperm(
                    modified.geometry.n_experts,
                    generator=torch.Generator().manual_seed(seed),
                )
                for codes in tensors:
                    codes.copy_(codes.index_select(0, permutation))
            else:
                for codes in tensors:
                    codes.zero_()
            prediction, _ = modified(
                inputs,
                forced_top_ids=routing.top_ids,
                forced_weights=routing.weights,
            )
        result[intervention] = tensor_metrics(prediction, target).as_dict()
    return result


def _gate(
    name: str,
    passed: bool,
    observed: object,
    rule: str,
) -> dict[str, object]:
    return {
        "name": name,
        "passed": bool(passed),
        "observed": observed,
        "rule": rule,
    }


def _write_report(output_dir: Path, payload: dict[str, Any]) -> None:
    decision = payload["decision"]
    lines = [
        "# Pre-Qwen Certification V1",
        "",
        f"**Decision:** **{decision['verdict']}**",
        "",
        "This result certifies the synthetic methodological harness only. It does not certify transfer to OLMoE or Qwen.",
        "",
        "## Gates",
        "",
        "| Gate | Pass | Observed | Rule |",
        "|---|---|---|---|",
    ]
    for gate in decision["gates"]:
        observed = json.dumps(gate["observed"], ensure_ascii=False, sort_keys=True)
        lines.append(
            f"| `{gate['name']}` | {'yes' if gate['passed'] else 'no'} | "
            f"`{observed}` | {gate['rule']} |"
        )
    lines += [
        "",
        "## Known-truth rank recovery",
        "",
        "| Seed | Rank | Before NRMSE | After NRMSE | Weight error |",
        "|---:|---:|---:|---:|---:|",
    ]
    for seed_result in payload["known_truth"]["seeds"]:
        for row in seed_result["rank_sweep"]:
            lines.append(
                f"| {seed_result['seed']} | {row['rank']} | "
                f"{row['before']['nrmse']:.6g} | {row['after']['nrmse']:.6g} | "
                f"{row['decomposition']['max_weight_relative_error']:.6g} |"
            )
    lines += [
        "",
        "## Negative and leakage controls",
        "",
        f"- Mean low-rank negative-control NRMSE: `{payload['negative_control']['mean_nrmse']:.6f}`.",
        f"- Mean input-blind NRMSE: `{payload['leakage_controls']['input_blind_mean_nrmse']:.6f}`.",
        f"- Mean shuffled-target NRMSE: `{payload['leakage_controls']['shuffled_target_mean_nrmse']:.6f}`.",
        "",
        "## Boundary curve",
        "",
        "| Epsilon | Mean K2 NRMSE |",
        "|---:|---:|",
    ]
    for row in payload["boundary_control"]["curve"]:
        lines.append(f"| {row['epsilon']:.4f} | {row['mean_nrmse']:.6f} |")
    lines += [
        "",
        "## Interpretation",
        "",
        "A PASS means the harness reproduces exact Modal algebra, detects injected wiring faults, recovers a hidden rank-2 teacher, rejects rank-1 and independent-expert controls, exhibits a monotonic boundary curve, and rejects input-blind/shuffled-target leakage controls.",
        "",
        "The next gate is a controlled transplant into a separately trained conventional small MoE, followed by sealed evaluation. A real pretrained model must not be attempted solely on the basis of this synthetic PASS.",
    ]
    (output_dir / "VERDICT.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def _write_hashes(output_dir: Path) -> None:
    candidates = sorted(
        path
        for path in output_dir.iterdir()
        if path.is_file() and path.name != "sha256sums.txt"
    )
    (output_dir / "sha256sums.txt").write_text(
        "\n".join(f"{_hash_file(path)}  {path.name}" for path in candidates) + "\n",
        encoding="utf-8",
    )


def run_certification(
    config_path: Path,
    output_dir: Path,
    *,
    root: Path | None = None,
) -> dict[str, Any]:
    config = load_config(config_path)
    root = root or config_path.resolve().parents[1]
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(int(config.get("threads", 2)))
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        # Safe when a parent test process already initialized interop threads.
        pass

    geometry = _geometry(config["geometry"])
    seeds = [int(value) for value in config["seeds"]]
    if len(seeds) < 3:
        raise CertificationError("at least three independent training seeds are required")
    data_config = config["data"]
    train = sample_document_activations(
        d_model=geometry.d_model,
        documents=int(data_config["train_documents"]),
        tokens_per_document=int(data_config["tokens_per_document"]),
        seed=int(data_config["train_seed"]),
        prefix="train",
    )
    validation = sample_document_activations(
        d_model=geometry.d_model,
        documents=int(data_config["validation_documents"]),
        tokens_per_document=int(data_config["tokens_per_document"]),
        seed=int(data_config["validation_seed"]),
        prefix="sealed-synthetic",
        domain_shift=float(data_config.get("validation_domain_shift", 0.15)),
    )

    known_config = config["known_truth"]
    optimization = config["optimization"]
    target_rank = int(known_config["teacher_rank"])
    ranks = [int(value) for value in known_config["rank_sweep"]]
    if target_rank not in ranks or target_rank - 1 not in ranks:
        raise CertificationError("rank_sweep must include teacher_rank and teacher_rank-1")

    known_seed_results: list[dict[str, Any]] = []
    positive_records: list[dict[str, object]] = []
    negative_rows: list[dict[str, object]] = []
    input_blind_rows: list[dict[str, float]] = []
    shuffled_rows: list[dict[str, float]] = []
    code_interventions: list[dict[str, object]] = []
    fault_matrix: dict[str, dict[str, float]] | None = None
    capture_metadata: dict[str, object] | None = None
    algebra_metrics: dict[str, float] | None = None

    for seed_index, seed in enumerate(seeds):
        modal_teacher = make_modal_teacher(
            geometry,
            rank=target_rank,
            seed=seed,
            residual_scale=float(known_config["residual_scale"]),
            code_scale=float(known_config["code_scale"]),
        )
        conventional_teacher = ConventionalSwiGLUMoE.from_modal(modal_teacher)

        with torch.no_grad():
            direct, routing = modal_teacher(validation.inputs)
            reconstructed, _ = modal_teacher.reference_reconstructed(
                validation.inputs,
                forced_top_ids=routing.top_ids,
                forced_weights=routing.weights,
            )
        current_algebra = tensor_metrics(direct, reconstructed).as_dict()
        if algebra_metrics is None or current_algebra["nrmse"] > algebra_metrics["nrmse"]:
            algebra_metrics = current_algebra

        sweep, sweep_records = rank_sweep(
            conventional_teacher,
            train,
            validation,
            ranks=ranks,
            seed=seed,
            fine_tune_steps=int(optimization["fine_tune_steps"]),
            batch_size=int(optimization["batch_size"]),
            lr=float(optimization["learning_rate"]),
        )
        positive_records.extend(
            row
            for row in sweep_records
            if row["variant"] == f"k{target_rank}-before"
        )
        exact_student, _ = ScalarModalMoE.from_conventional_svd(
            conventional_teacher, target_rank, freeze_router=True
        )
        interventions = _intervention_metrics(
            conventional_teacher,
            exact_student,
            validation.inputs,
            seed=seed + 700,
        )
        code_interventions.append({"seed": seed, "metrics": interventions})

        if seed_index == 0:
            residual = validation.inputs * 0.10
            capture = capture_layer(
                conventional_teacher,
                validation.inputs,
                document_ids=validation.document_ids,
                sequence_ids=validation.sequence_ids,
                token_positions=validation.token_positions,
                layer_id=7,
                residual=residual,
            )
            capture_metadata = save_capture(output_dir / "synthetic-capture.pt", capture)
            fault_matrix = run_fault_matrix(conventional_teacher, capture)

        negative_teacher = make_negative_teacher(geometry, seed=seed + 10_000)
        negative_student, decomposition = ScalarModalMoE.from_conventional_svd(
            negative_teacher, target_rank, freeze_router=True
        )
        negative_metrics, negative_records = evaluate_student(
            negative_teacher,
            negative_student,
            validation,
            seed=seed,
            label="negative-k2",
        )
        negative_rows.append(
            {
                "seed": seed,
                "metrics": negative_metrics,
                "decomposition": decomposition,
            }
        )

        input_blind_rows.append(
            input_blind_baseline(conventional_teacher, train, validation)
        )
        shuffled_metrics, shuffled_history = shuffled_target_control(
            conventional_teacher,
            train,
            validation,
            rank=target_rank,
            seed=seed + 20_000,
            steps=int(optimization["shuffled_target_steps"]),
            batch_size=int(optimization["batch_size"]),
            lr=float(optimization["learning_rate"]),
        )
        shuffled_rows.append(shuffled_metrics)
        known_seed_results.append(
            {
                "seed": seed,
                "rank_sweep": sweep,
                "code_interventions": interventions,
                "shuffled_target_history": shuffled_history,
            }
        )

    if algebra_metrics is None or fault_matrix is None or capture_metadata is None:
        raise AssertionError("certification did not produce core diagnostics")

    boundary_epsilons = [float(value) for value in known_config["boundary_epsilons"]]
    boundary_curve: list[dict[str, float]] = []
    for epsilon in boundary_epsilons:
        values: list[float] = []
        for seed in seeds:
            modal_teacher = make_modal_teacher(
                geometry,
                rank=target_rank,
                seed=seed,
                residual_scale=float(known_config["residual_scale"]),
                code_scale=float(known_config["code_scale"]),
            )
            conventional_teacher = ConventionalSwiGLUMoE.from_modal(modal_teacher)
            boundary_teacher = add_independent_expert_noise(
                conventional_teacher,
                epsilon=epsilon,
                seed=seed + 30_000,
            )
            boundary_student, _ = ScalarModalMoE.from_conventional_svd(
                boundary_teacher, target_rank, freeze_router=True
            )
            metrics, _ = evaluate_student(
                boundary_teacher,
                boundary_student,
                validation,
                seed=seed,
                label=f"boundary-{epsilon}",
            )
            values.append(float(metrics["nrmse"]))
        boundary_curve.append(
            {
                "epsilon": epsilon,
                "mean_nrmse": float(np.mean(values)),
                "worst_nrmse": float(max(values)),
            }
        )
    boundary_monotonicity = monotonicity(
        [row["epsilon"] for row in boundary_curve],
        [row["mean_nrmse"] for row in boundary_curve],
    )

    statistics_config = config["statistics"]
    positive_bootstrap = crossed_hierarchical_bootstrap(
        positive_records,
        value_key="nrmse",
        samples=int(statistics_config["bootstrap_samples"]),
        random_seed=int(statistics_config["bootstrap_seed"]),
        confidence=float(statistics_config["confidence"]),
    )
    positive_by_domain = summarize_groups(
        positive_records, value_key="nrmse", group_keys=("domain",)
    )

    positive_target_nrmse = [
        next(row for row in item["rank_sweep"] if row["rank"] == target_rank)[
            "before"
        ]["nrmse"]
        for item in known_seed_results
    ]
    lower_rank_nrmse = [
        next(row for row in item["rank_sweep"] if row["rank"] == target_rank - 1)[
            "before"
        ]["nrmse"]
        for item in known_seed_results
    ]
    negative_nrmse = [float(row["metrics"]["nrmse"]) for row in negative_rows]
    input_blind_nrmse = [float(row["nrmse"]) for row in input_blind_rows]
    shuffled_nrmse = [float(row["nrmse"]) for row in shuffled_rows]
    intervention_min = min(
        float(metric["nrmse"])
        for row in code_interventions
        for metric in row["metrics"].values()
    )
    fault_values = [
        float(metrics["nrmse"])
        for name, metrics in fault_matrix.items()
        if name != "none"
    ]

    gates_config = config["gates"]
    gates = [
        _gate(
            "modal_algebra",
            algebra_metrics["nrmse"] <= float(gates_config["algebra_nrmse_max"]),
            algebra_metrics,
            f"NRMSE <= {gates_config['algebra_nrmse_max']}",
        ),
        _gate(
            "capture_replay_identity",
            fault_matrix["none"]["nrmse"]
            <= float(gates_config["replay_nrmse_max"]),
            fault_matrix["none"],
            f"correct replay NRMSE <= {gates_config['replay_nrmse_max']}",
        ),
        _gate(
            "fault_sensitivity",
            min(fault_values) >= float(gates_config["fault_nrmse_min"]),
            {"minimum_fault_nrmse": min(fault_values), "faults": fault_matrix},
            f"every injected fault NRMSE >= {gates_config['fault_nrmse_min']}",
        ),
        _gate(
            "known_truth_recovery",
            max(positive_target_nrmse)
            <= float(gates_config["positive_rank_nrmse_max"])
            and float(positive_bootstrap["ucb"])
            <= float(gates_config["positive_rank_nrmse_max"]),
            {
                "per_seed": positive_target_nrmse,
                "hierarchical_bootstrap": positive_bootstrap,
            },
            f"every seed and hierarchical UCB <= {gates_config['positive_rank_nrmse_max']}",
        ),
        _gate(
            "lower_rank_rejection",
            min(lower_rank_nrmse)
            >= float(gates_config["lower_rank_nrmse_min"]),
            {"per_seed": lower_rank_nrmse},
            f"teacher_rank-1 NRMSE >= {gates_config['lower_rank_nrmse_min']} in every seed",
        ),
        _gate(
            "independent_expert_rejection",
            min(negative_nrmse)
            >= float(gates_config["negative_rank_nrmse_min"]),
            {"per_seed": negative_nrmse},
            f"negative-control K{target_rank} NRMSE >= {gates_config['negative_rank_nrmse_min']} in every seed",
        ),
        _gate(
            "boundary_monotonicity",
            bool(boundary_monotonicity["nondecreasing"])
            and float(boundary_monotonicity["pearson"])
            >= float(gates_config["boundary_correlation_min"])
            and boundary_curve[-1]["mean_nrmse"]
            >= float(gates_config["boundary_final_nrmse_min"]),
            {
                "curve": boundary_curve,
                "monotonicity": boundary_monotonicity,
            },
            "boundary error must be nondecreasing, strongly correlated with epsilon, and materially nonzero at the final epsilon",
        ),
        _gate(
            "expert_code_causality",
            intervention_min >= float(gates_config["code_ablation_nrmse_min"]),
            {
                "minimum_intervention_nrmse": intervention_min,
                "per_seed": code_interventions,
            },
            f"every code intervention NRMSE >= {gates_config['code_ablation_nrmse_min']}",
        ),
        _gate(
            "input_blind_rejection",
            min(input_blind_nrmse)
            >= float(gates_config["input_blind_nrmse_min"]),
            {"per_seed": input_blind_nrmse},
            f"input-blind NRMSE >= {gates_config['input_blind_nrmse_min']} in every seed",
        ),
        _gate(
            "shuffled_target_rejection",
            min(shuffled_nrmse)
            >= float(gates_config["shuffled_target_nrmse_min"]),
            {"per_seed": shuffled_nrmse},
            f"shuffled-target NRMSE >= {gates_config['shuffled_target_nrmse_min']} in every seed",
        ),
    ]
    all_passed = all(bool(gate["passed"]) for gate in gates)

    payload: dict[str, Any] = {
        "metadata": {
            "protocol_version": config["protocol_version"],
            "executed_at": datetime.now(timezone.utc).isoformat(),
            "config_path": str(config_path),
            "config_sha256": _hash_file(config_path),
            "source_hashes": _source_hashes(root),
            "git_sha": os.environ.get("GITHUB_SHA"),
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "threads": torch.get_num_threads(),
            "scope_warning": (
                "Synthetic methodological certification only; no real pretrained "
                "checkpoint is evaluated."
            ),
        },
        "configuration": config,
        "algebra": algebra_metrics,
        "capture": capture_metadata,
        "fault_matrix": fault_matrix,
        "known_truth": {
            "teacher_rank": target_rank,
            "seeds": known_seed_results,
            "hierarchical_bootstrap": positive_bootstrap,
            "by_domain": positive_by_domain,
        },
        "negative_control": {
            "runs": negative_rows,
            "mean_nrmse": float(np.mean(negative_nrmse)),
        },
        "boundary_control": {
            "curve": boundary_curve,
            "monotonicity": boundary_monotonicity,
        },
        "leakage_controls": {
            "input_blind": input_blind_rows,
            "input_blind_mean_nrmse": float(np.mean(input_blind_nrmse)),
            "shuffled_target": shuffled_rows,
            "shuffled_target_mean_nrmse": float(np.mean(shuffled_nrmse)),
        },
        "decision": {
            "verdict": "PRE_QWEN_SYNTHETIC_CERTIFIED" if all_passed else "PRE_QWEN_CERTIFICATION_FAILED",
            "passed": all_passed,
            "gates": gates,
            "next_gate": (
                "controlled conventional small-MoE transplant with sealed test set"
                if all_passed
                else "repair failed methodological gates before any real-checkpoint transplant"
            ),
        },
    }

    (output_dir / "metrics.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with (output_dir / "per_example.jsonl").open("w", encoding="utf-8") as handle:
        for row in positive_records:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    (output_dir / "config.resolved.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )
    _write_report(output_dir, payload)
    (output_dir / "environment.txt").write_text(
        "\n".join(
            [
                f"python={sys.version.replace(chr(10), ' ')}",
                f"platform={platform.platform()}",
                f"torch={torch.__version__}",
                f"numpy={np.__version__}",
                f"threads={torch.get_num_threads()}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    _write_hashes(output_dir)
    return payload
