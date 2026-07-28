"""Controlled closed-loop transplant into a separately trained conventional MoE.

This is the second pre-real-checkpoint gate.  Unlike the known-truth synthetic
suite, the teacher experts are initialized and trained conventionally.  The
primary Modal rank and all gates are fixed in configuration before execution.
"""
from __future__ import annotations

import copy
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import sys
from typing import Any, Mapping

import numpy as np
import torch
import yaml

from .certify import CertificationError
from .metrics import crossed_hierarchical_bootstrap, summarize_groups
from .modal import ScalarModalMoE
from .tiny_lm import (
    CharacterCorpus,
    TinyLMConfig,
    capture_training_layer,
    distill_layer_student,
    evaluate_closed_loop,
    evaluate_local_student,
    expert_parameter_count,
    generate_multidomain_documents,
    install_student,
    make_narrow_student,
    train_teacher,
)


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise CertificationError("controlled transplant config must be a mapping")
    return payload


def _tiny_config(payload: Mapping[str, Any]) -> TinyLMConfig:
    return TinyLMConfig(**{key: value for key, value in payload.items()})


def _code_intervention(
    student: ScalarModalMoE,
    policy: str,
    *,
    seed: int,
) -> ScalarModalMoE:
    modified = copy.deepcopy(student)
    with torch.no_grad():
        tensors = (modified.gate_codes, modified.up_codes, modified.down_codes)
        if policy == "mean-code":
            for codes in tensors:
                codes.copy_(codes.mean(dim=0, keepdim=True).expand_as(codes))
        elif policy == "shuffle-code":
            permutation = torch.randperm(
                modified.geometry.n_experts,
                generator=torch.Generator().manual_seed(seed),
            )
            for codes in tensors:
                codes.copy_(codes.index_select(0, permutation))
        elif policy == "zero-residual":
            for codes in tensors:
                codes.zero_()
        else:
            raise ValueError(policy)
    return modified


def _gate(name: str, passed: bool, observed: object, rule: str) -> dict[str, object]:
    return {"name": name, "passed": bool(passed), "observed": observed, "rule": rule}


def _write_report(output_dir: Path, payload: dict[str, Any]) -> None:
    d = payload["decision"]
    lines = [
        "# Controlled Small-MoE Transplant V1",
        "",
        f"**Decision:** **{d['verdict']}**",
        "",
        f"**GO for a real pretrained checkpoint:** **{'YES' if d['go_for_real_checkpoint'] else 'NO'}**",
        "",
        "The teacher is a separately trained conventional character-level Transformer MoE. The primary Modal rank was frozen before execution and the sealed documents were generated only after all candidates were trained.",
        "",
        "## Gates",
        "",
        "| Gate | Pass | Observed | Rule |",
        "|---|---|---|---|",
    ]
    for gate in d["gates"]:
        observed = json.dumps(gate["observed"], sort_keys=True)
        lines.append(
            f"| `{gate['name']}` | {'yes' if gate['passed'] else 'no'} | `{observed}` | {gate['rule']} |"
        )
    lines += [
        "",
        "## Per-seed sealed results",
        "",
        "| Seed | Modal Δloss | Modal KL | Modal PPL ratio | Modal top-1 | Narrow Δloss | Narrow KL |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["runs"]:
        modal = row["sealed_modal"]
        narrow = row["sealed_narrow"]
        lines.append(
            f"| {row['seed']} | {modal['loss_delta']:+.5f} | {modal['kl_teacher_to_candidate']:.5f} | "
            f"{modal['perplexity_ratio']:.5f} | {modal['top1_agreement']:.3%} | "
            f"{narrow['loss_delta']:+.5f} | {narrow['kl_teacher_to_candidate']:.5f} |"
        )
    lines += [
        "",
        "## Development rank curve",
        "",
        "| Seed | K | Local NRMSE | Δloss | KL | PPL ratio | Expert params/full | Ideal compute/full |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for run in payload["runs"]:
        for row in run["development_rank_curve"]:
            lines.append(
                f"| {run['seed']} | {row['rank']} | {row['local']['nrmse']:.4f} | "
                f"{row['closed_loop']['loss_delta']:+.4f} | {row['closed_loop']['kl_teacher_to_candidate']:.4f} | "
                f"{row['closed_loop']['perplexity_ratio']:.4f} | {row['expert_parameter_ratio']:.3%} | "
                f"{row['idealized_expert_compute_ratio']:.3%} |"
            )
    lines += [
        "",
        "## Interpretation",
        "",
        d["interpretation"],
        "",
        "A functional-only result does not authorize a Qwen experiment. It means the layer can be transplanted within the quality budget, but the current scalar Modal student has not yet shown superiority over a strong compute-matched conventional narrowing baseline. The next action is to improve the student parameterization or optimization under the same sealed protocol, not to relax the gates.",
    ]
    (output_dir / "VERDICT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_controlled_transplant(
    config_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    config = _load(config_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    tiny = _tiny_config(config["model"])
    torch.set_num_threads(int(config.get("threads", 2)))
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    seeds = [int(value) for value in config["seeds"]]
    if len(seeds) < 3:
        raise CertificationError("controlled transplant requires at least three seeds")
    primary_rank = int(config["transplant"]["primary_rank"])
    rank_curve = [int(value) for value in config["transplant"]["development_ranks"]]
    if primary_rank not in rank_curve:
        raise CertificationError("development_ranks must include primary_rank")
    layer_id = int(config["transplant"]["layer_id"])
    if not 0 <= layer_id < tiny.n_layers:
        raise CertificationError("layer_id is outside the tiny model")

    data = config["data"]
    train_documents = generate_multidomain_documents(
        split="train",
        documents=int(data["train_documents"]),
        seed=int(data["train_seed"]),
    )
    calibration_documents = generate_multidomain_documents(
        split="calibration",
        documents=int(data["calibration_documents"]),
        seed=int(data["calibration_seed"]),
    )
    development_documents = generate_multidomain_documents(
        split="development",
        documents=int(data["development_documents"]),
        seed=int(data["development_seed"]),
    )
    development_corpus = CharacterCorpus(
        {
            "train": train_documents,
            "calibration": calibration_documents,
            "development": development_documents,
        },
        tiny.seq_len,
    )

    trained: list[dict[str, Any]] = []
    for seed in seeds:
        teacher, teacher_history = train_teacher(
            development_corpus, tiny, seed=seed
        )
        teacher_moe = teacher.blocks[layer_id].moe
        full_expert_parameters = expert_parameter_count(teacher_moe)
        captured = capture_training_layer(
            teacher,
            development_corpus,
            split="train",
            layer_id=layer_id,
            batches=int(config["transplant"]["capture_batches"]),
            batch_size=tiny.batch_size,
            seed=seed + 1000,
        )

        curve: list[dict[str, Any]] = []
        primary_student: ScalarModalMoE | None = None
        for rank in rank_curve:
            student, decomposition = ScalarModalMoE.from_conventional_svd(
                teacher_moe, rank, freeze_router=True
            )
            history = distill_layer_student(
                student,
                captured,
                steps=tiny.student_steps,
                batch_size=int(config["transplant"]["student_batch_size"]),
                learning_rate=tiny.student_learning_rate,
                seed=seed + 2000 + rank,
            )
            local, _ = evaluate_local_student(
                teacher,
                student,
                development_corpus,
                split="development",
                layer_id=layer_id,
                windows_per_document=int(data["development_windows_per_document"]),
                seed=seed + 3000,
            )
            candidate = install_student(teacher, student, layer_id=layer_id)
            closed, _ = evaluate_closed_loop(
                teacher,
                candidate,
                development_corpus,
                split="development",
                windows_per_document=int(data["development_windows_per_document"]),
                seed=seed + 3000,
            )
            curve.append(
                {
                    "rank": rank,
                    "decomposition": decomposition,
                    "local": local,
                    "closed_loop": closed,
                    "history": history,
                    "expert_parameter_ratio": student.expert_transform_parameter_count()
                    / full_expert_parameters,
                    "idealized_expert_compute_ratio": student.idealized_expert_compute_ratio(),
                }
            )
            if rank == primary_rank:
                primary_student = copy.deepcopy(student)
        if primary_student is None:
            raise AssertionError("primary student was not trained")

        matrix_compute_ratio = (primary_rank + 1) / tiny.top_k
        narrow_d_ff = max(4, int(round(tiny.d_ff * matrix_compute_ratio)))
        narrow = make_narrow_student(teacher_moe, d_ff=narrow_d_ff)
        narrow_history = distill_layer_student(
            narrow,
            captured,
            steps=tiny.student_steps,
            batch_size=int(config["transplant"]["student_batch_size"]),
            learning_rate=tiny.student_learning_rate,
            seed=seed + 4000,
        )
        narrow_local, _ = evaluate_local_student(
            teacher,
            narrow,
            development_corpus,
            split="development",
            layer_id=layer_id,
            windows_per_document=int(data["development_windows_per_document"]),
            seed=seed + 3000,
        )
        narrow_model = install_student(teacher, narrow, layer_id=layer_id)
        narrow_development, _ = evaluate_closed_loop(
            teacher,
            narrow_model,
            development_corpus,
            split="development",
            windows_per_document=int(data["development_windows_per_document"]),
            seed=seed + 3000,
        )

        # Freeze all candidate states before the sealed documents are materialized.
        torch.save(
            {
                "teacher": teacher.state_dict(),
                "modal": primary_student.state_dict(),
                "narrow": narrow.state_dict(),
                "vocabulary": development_corpus.itos,
                "config": config,
                "seed": seed,
            },
            output_dir / f"frozen-candidates-seed-{seed}.pt",
        )
        trained.append(
            {
                "seed": seed,
                "teacher": teacher,
                "modal": primary_student,
                "narrow": narrow,
                "teacher_history": teacher_history,
                "development_rank_curve": curve,
                "narrow_history": narrow_history,
                "narrow_local_development": narrow_local,
                "narrow_closed_loop_development": narrow_development,
                "full_expert_parameters": full_expert_parameters,
                "narrow_d_ff": narrow_d_ff,
                "vocabulary": development_corpus.itos,
            }
        )

    # Materialize the sealed split only after every teacher/student/baseline state
    # has been frozen and written to disk.
    sealed_documents = generate_multidomain_documents(
        split="sealed",
        documents=int(data["sealed_documents"]),
        seed=int(data["sealed_seed"]),
    )

    runs: list[dict[str, Any]] = []
    modal_records: list[dict[str, object]] = []
    narrow_records: list[dict[str, object]] = []
    intervention_rows: list[dict[str, Any]] = []
    for item in trained:
        seed = int(item["seed"])
        corpus = CharacterCorpus(
            {
                "train": train_documents,
                "calibration": calibration_documents,
                "development": development_documents,
                "sealed": sealed_documents,
            },
            tiny.seq_len,
            vocabulary=item["vocabulary"],
        )
        teacher = item["teacher"]
        modal_model = install_student(teacher, item["modal"], layer_id=layer_id)
        narrow_model = install_student(teacher, item["narrow"], layer_id=layer_id)
        sealed_modal, modal_seed_records = evaluate_closed_loop(
            teacher,
            modal_model,
            corpus,
            split="sealed",
            windows_per_document=int(data["sealed_windows_per_document"]),
            seed=int(data["sealed_window_seed"]),
        )
        sealed_narrow, narrow_seed_records = evaluate_closed_loop(
            teacher,
            narrow_model,
            corpus,
            split="sealed",
            windows_per_document=int(data["sealed_windows_per_document"]),
            seed=int(data["sealed_window_seed"]),
        )
        for row in modal_seed_records:
            row["seed"] = seed
            row["variant"] = "modal-primary"
            modal_records.append(row)
        for row in narrow_seed_records:
            row["seed"] = seed
            row["variant"] = "narrow-compute-matched"
            narrow_records.append(row)

        interventions: dict[str, dict[str, float]] = {}
        for index, policy in enumerate(("mean-code", "shuffle-code", "zero-residual")):
            modified = _code_intervention(item["modal"], policy, seed=seed + 5000 + index)
            modified_model = install_student(teacher, modified, layer_id=layer_id)
            metrics, _ = evaluate_closed_loop(
                teacher,
                modified_model,
                corpus,
                split="sealed",
                windows_per_document=int(data["sealed_windows_per_document"]),
                seed=int(data["sealed_window_seed"]),
            )
            interventions[policy] = metrics
        intervention_rows.append({"seed": seed, "metrics": interventions})

        runs.append(
            {
                "seed": seed,
                "teacher_history": item["teacher_history"],
                "development_rank_curve": item["development_rank_curve"],
                "primary_rank": primary_rank,
                "sealed_modal": sealed_modal,
                "sealed_narrow": sealed_narrow,
                "narrow_d_ff": item["narrow_d_ff"],
                "narrow_local_development": item["narrow_local_development"],
                "narrow_closed_loop_development": item["narrow_closed_loop_development"],
                "expert_parameter_ratio_modal": item["modal"].expert_transform_parameter_count()
                / item["full_expert_parameters"],
                "idealized_expert_compute_ratio_modal": item["modal"].idealized_expert_compute_ratio(),
                "expert_parameter_ratio_narrow": expert_parameter_count(item["narrow"])
                / item["full_expert_parameters"],
                "matrix_compute_ratio_narrow": item["narrow_d_ff"] / tiny.d_ff,
                "code_interventions": interventions,
            }
        )

    stats = config["statistics"]
    modal_delta_bootstrap = crossed_hierarchical_bootstrap(
        modal_records,
        value_key="loss_delta",
        samples=int(stats["bootstrap_samples"]),
        random_seed=int(stats["bootstrap_seed"]),
        confidence=float(stats["confidence"]),
    )
    narrow_delta_bootstrap = crossed_hierarchical_bootstrap(
        narrow_records,
        value_key="loss_delta",
        samples=int(stats["bootstrap_samples"]),
        random_seed=int(stats["bootstrap_seed"]) + 1,
        confidence=float(stats["confidence"]),
    )
    # Paired advantage records use exactly the same seed/document/window cells.
    narrow_index = {
        (row["seed"], row["document_id"], row["start"]): row
        for row in narrow_records
    }
    advantage_records: list[dict[str, object]] = []
    for row in modal_records:
        key = (row["seed"], row["document_id"], row["start"])
        narrow_row = narrow_index[key]
        advantage_records.append(
            {
                "seed": row["seed"],
                "document_id": row["document_id"],
                "modal_minus_narrow_loss_delta": float(row["loss_delta"])
                - float(narrow_row["loss_delta"]),
            }
        )
    advantage_bootstrap = crossed_hierarchical_bootstrap(
        advantage_records,
        value_key="modal_minus_narrow_loss_delta",
        samples=int(stats["bootstrap_samples"]),
        random_seed=int(stats["bootstrap_seed"]) + 2,
        confidence=float(stats["confidence"]),
    )

    gates_cfg = config["gates"]
    modal_metrics = [run["sealed_modal"] for run in runs]
    modal_parameter_ratios = [float(run["expert_parameter_ratio_modal"]) for run in runs]
    modal_compute_ratios = [float(run["idealized_expert_compute_ratio_modal"]) for run in runs]
    primary_dev_local = [
        next(
            row for row in run["development_rank_curve"] if row["rank"] == primary_rank
        )["local"]["nrmse"]
        for run in runs
    ]
    intervention_kl_deltas = [
        float(metrics["kl_teacher_to_candidate"])
        - float(run["sealed_modal"]["kl_teacher_to_candidate"])
        for run in runs
        for metrics in run["code_interventions"].values()
    ]

    fidelity_pass = (
        max(primary_dev_local) <= float(gates_cfg["development_local_nrmse_max"])
        and max(float(row["loss_delta"]) for row in modal_metrics)
        <= float(gates_cfg["sealed_loss_delta_max_per_seed"])
        and float(modal_delta_bootstrap["ucb"])
        <= float(gates_cfg["sealed_loss_delta_ucb_max"])
        and max(float(row["kl_teacher_to_candidate"]) for row in modal_metrics)
        <= float(gates_cfg["sealed_kl_max"])
        and max(float(row["perplexity_ratio"]) for row in modal_metrics)
        <= float(gates_cfg["sealed_perplexity_ratio_max"])
        and min(float(row["top1_agreement"]) for row in modal_metrics)
        >= float(gates_cfg["sealed_top1_agreement_min"])
    )
    compression_pass = (
        max(modal_parameter_ratios) <= float(gates_cfg["expert_parameter_ratio_max"])
        and max(modal_compute_ratios) <= float(gates_cfg["idealized_compute_ratio_max"])
    )
    advantage_pass = float(advantage_bootstrap["ucb"]) <= float(
        gates_cfg["modal_minus_narrow_loss_ucb_max"]
    )
    codes_pass = min(intervention_kl_deltas) >= float(
        gates_cfg["code_intervention_kl_increase_min"]
    )

    gates = [
        _gate(
            "closed_loop_fidelity",
            fidelity_pass,
            {
                "development_local_nrmse": primary_dev_local,
                "sealed_per_seed": modal_metrics,
                "loss_delta_bootstrap": modal_delta_bootstrap,
            },
            "primary rank must satisfy local, sealed loss, KL, perplexity, and top-1 gates in every seed",
        ),
        _gate(
            "parameter_and_compute_compression",
            compression_pass,
            {
                "expert_parameter_ratios": modal_parameter_ratios,
                "idealized_compute_ratios": modal_compute_ratios,
            },
            "expert parameter and idealized compute ratios must stay below predeclared limits",
        ),
        _gate(
            "compute_matched_advantage",
            advantage_pass,
            {
                "modal_minus_narrow_loss_bootstrap": advantage_bootstrap,
                "narrow_loss_delta_bootstrap": narrow_delta_bootstrap,
            },
            f"UCB of Modal-minus-narrow loss penalty <= {gates_cfg['modal_minus_narrow_loss_ucb_max']}",
        ),
        _gate(
            "expert_code_causality",
            codes_pass,
            {
                "minimum_kl_increase": min(intervention_kl_deltas),
                "all_kl_increases": intervention_kl_deltas,
            },
            f"every code intervention must increase teacher KL by at least {gates_cfg['code_intervention_kl_increase_min']}",
        ),
    ]

    if fidelity_pass and compression_pass and advantage_pass and codes_pass:
        verdict = "CONTROLLED_TRANSPLANT_PASS"
        go = True
        interpretation = (
            "The primary scalar Modal student preserved the separately trained conventional teacher, met compression gates, and was non-inferior to the compute-matched narrow baseline under the paired hierarchical test. A single-layer OLMoE smoke is justified; Qwen remains premature until OLMoE multi-layer confirmation."
        )
    elif fidelity_pass and compression_pass and codes_pass:
        verdict = "CONTROLLED_TRANSPLANT_FUNCTIONAL_ONLY"
        go = False
        interpretation = (
            "The scalar Modal student preserved closed-loop function and compressed parameters/idealized compute, but did not satisfy the predeclared comparison against the strong compute-matched conventional narrowing baseline. This is evidence of transplantability, not evidence of architectural advantage."
        )
    else:
        verdict = "CONTROLLED_TRANSPLANT_FAIL"
        go = False
        interpretation = (
            "At least one essential fidelity, compression, or causal-use gate failed. Repair the student or harness and rerun under a new preregistered protocol version."
        )

    payload: dict[str, Any] = {
        "metadata": {
            "protocol_version": config["protocol_version"],
            "executed_at": datetime.now(timezone.utc).isoformat(),
            "config_sha256": _hash(config_path),
            "git_sha": os.environ.get("GITHUB_SHA"),
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "sealed_materialization_note": (
                "sealed documents were generated only after all candidate state_dicts "
                "were frozen and persisted"
            ),
        },
        "configuration": config,
        "runs": runs,
        "sealed_statistics": {
            "modal_loss_delta": modal_delta_bootstrap,
            "narrow_loss_delta": narrow_delta_bootstrap,
            "modal_minus_narrow_loss_delta": advantage_bootstrap,
            "modal_by_domain": summarize_groups(
                modal_records, value_key="loss_delta", group_keys=("domain",)
            ),
        },
        "decision": {
            "verdict": verdict,
            "go_for_real_checkpoint": go,
            "gates": gates,
            "interpretation": interpretation,
            "next_action": (
                "prepare preregistered OLMoE single-layer smoke"
                if go
                else "test improved Modal parameterization/optimization against the same controls"
            ),
        },
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with (output_dir / "sealed_per_window.jsonl").open("w", encoding="utf-8") as handle:
        for row in modal_records + narrow_records:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    (output_dir / "config.resolved.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )
    _write_report(output_dir, payload)
    (output_dir / "environment.txt").write_text(
        f"python={sys.version.replace(chr(10), ' ')}\nplatform={platform.platform()}\n"
        f"torch={torch.__version__}\nnumpy={np.__version__}\nthreads={torch.get_num_threads()}\n",
        encoding="utf-8",
    )
    files = sorted(
        path for path in output_dir.iterdir() if path.is_file() and path.name != "sha256sums.txt"
    )
    (output_dir / "sha256sums.txt").write_text(
        "\n".join(f"{_hash(path)}  {path.name}" for path in files) + "\n",
        encoding="utf-8",
    )
    return payload
