"""Resumable staged runner for controlled-small-MoE transplantation.

Each teacher/student seed is trained in an independent process and persisted
before the sealed split is materialized.  A separate finalize phase loads only
frozen checkpoints, creates the sealed documents, and computes the verdict.
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
from typing import Any

import numpy as np
import torch
import yaml

from .certify import CertificationError
from .controlled_transplant import (
    _code_intervention,
    _gate,
    _load,
    _tiny_config,
    _write_report,
)
from .metrics import crossed_hierarchical_bootstrap, summarize_groups
from .modal import ConventionalSwiGLUMoE, MoEGeometry, NeuronwiseModalMoE, ScalarModalMoE
from .tiny_lm import (
    CharacterCorpus,
    TinyMoELanguageModel,
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




def _source_hashes(root: Path) -> dict[str, str]:
    return {
        str(path.relative_to(root)): _hash(path)
        for path in sorted((root / "pre_qwen_certification").glob("*.py"))
    }


def _derive_sealed_seed(secret: str, label: str) -> int:
    if not secret:
        raise CertificationError("sealed secret must not be empty")
    digest = hashlib.sha256(
        label.encode("utf-8") + b"\0" + secret.encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], "big", signed=False) % (2**63 - 1)


def _resolve_sealed_seeds(data: dict[str, Any]) -> tuple[int, int, dict[str, Any]]:
    commitment = data.get("sealed_seed_commitment_sha256")
    if commitment is None:
        # Backward-compatible path for the v1.1 pilot. It is procedural rather
        # than cryptographically sealed and must be labelled accordingly.
        return (
            int(data["sealed_seed"]),
            int(data["sealed_window_seed"]),
            {"mode": "visible-seed-pilot", "commitment_verified": False},
        )
    environment_name = str(data.get("sealed_secret_env", "PRE_QWEN_SEALED_SECRET"))
    secret = os.environ.get(environment_name, "")
    if not secret:
        raise CertificationError(
            f"sealed evaluation requires environment variable {environment_name!r}"
        )
    observed = hashlib.sha256(secret.encode("utf-8")).hexdigest()
    if observed != str(commitment):
        raise CertificationError(
            f"sealed secret commitment mismatch: expected {commitment}, observed {observed}"
        )
    return (
        _derive_sealed_seed(secret, "documents"),
        _derive_sealed_seed(secret, "windows"),
        {
            "mode": "sha256-committed-secret",
            "secret_environment": environment_name,
            "seed_commitment_sha256": str(commitment),
            "commitment_verified": True,
        },
    )


def _modal_class(config: dict[str, Any]):
    parameterization = str(config["transplant"].get("parameterization", "scalar"))
    if parameterization == "scalar":
        return ScalarModalMoE
    if parameterization == "neuronwise":
        return NeuronwiseModalMoE
    raise CertificationError(f"unsupported Modal parameterization: {parameterization}")


def _development_corpus(config: dict[str, Any]) -> CharacterCorpus:
    tiny = _tiny_config(config["model"])
    data = config["data"]
    return CharacterCorpus(
        {
            "train": generate_multidomain_documents(
                split="train",
                documents=int(data["train_documents"]),
                seed=int(data["train_seed"]),
            ),
            "calibration": generate_multidomain_documents(
                split="calibration",
                documents=int(data["calibration_documents"]),
                seed=int(data["calibration_seed"]),
            ),
            "development": generate_multidomain_documents(
                split="development",
                documents=int(data["development_documents"]),
                seed=int(data["development_seed"]),
            ),
        },
        tiny.seq_len,
    )


def train_controlled_seed(
    config_path: Path,
    output_dir: Path,
    *,
    seed: int,
) -> dict[str, Any]:
    config = _load(config_path)
    configured_seeds = [int(value) for value in config["seeds"]]
    if seed not in configured_seeds:
        raise CertificationError(f"seed {seed} is not preregistered")
    tiny = _tiny_config(config["model"])
    torch.set_num_threads(int(config.get("threads", 2)))
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    output_dir.mkdir(parents=True, exist_ok=True)
    corpus = _development_corpus(config)
    transplant = config["transplant"]
    data = config["data"]
    layer_id = int(transplant["layer_id"])
    primary_rank = int(transplant["primary_rank"])
    ranks = [int(value) for value in transplant["development_ranks"]]

    teacher, teacher_history = train_teacher(corpus, tiny, seed=seed)
    teacher_moe = teacher.blocks[layer_id].moe
    if not isinstance(teacher_moe, ConventionalSwiGLUMoE):
        raise TypeError("controlled teacher layer is not conventional MoE")
    full_expert_parameters = expert_parameter_count(teacher_moe)
    captured = capture_training_layer(
        teacher,
        corpus,
        split="train",
        layer_id=layer_id,
        batches=int(transplant["capture_batches"]),
        batch_size=tiny.batch_size,
        seed=seed + 1000,
    )

    rank_curve: list[dict[str, Any]] = []
    modal_class = _modal_class(config)
    primary_student: ScalarModalMoE | NeuronwiseModalMoE | None = None
    for rank in ranks:
        student, decomposition = modal_class.from_conventional_svd(
            teacher_moe, rank, freeze_router=True
        )
        history = distill_layer_student(
            student,
            captured,
            steps=tiny.student_steps,
            batch_size=int(transplant["student_batch_size"]),
            learning_rate=tiny.student_learning_rate,
            seed=seed + 2000 + rank,
        )
        local, _ = evaluate_local_student(
            teacher,
            student,
            corpus,
            split="development",
            layer_id=layer_id,
            windows_per_document=int(data["development_windows_per_document"]),
            seed=seed + 3000,
        )
        candidate = install_student(teacher, student, layer_id=layer_id)
        closed, _ = evaluate_closed_loop(
            teacher,
            candidate,
            corpus,
            split="development",
            windows_per_document=int(data["development_windows_per_document"]),
            seed=seed + 3000,
        )
        rank_curve.append(
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
        batch_size=int(transplant["student_batch_size"]),
        learning_rate=tiny.student_learning_rate,
        seed=seed + 4000,
    )
    narrow_local, _ = evaluate_local_student(
        teacher,
        narrow,
        corpus,
        split="development",
        layer_id=layer_id,
        windows_per_document=int(data["development_windows_per_document"]),
        seed=seed + 3000,
    )
    narrow_model = install_student(teacher, narrow, layer_id=layer_id)
    narrow_closed, _ = evaluate_closed_loop(
        teacher,
        narrow_model,
        corpus,
        split="development",
        windows_per_document=int(data["development_windows_per_document"]),
        seed=seed + 3000,
    )

    metadata = {
        "seed": seed,
        "teacher_history": teacher_history,
        "development_rank_curve": rank_curve,
        "primary_rank": primary_rank,
        "narrow_d_ff": narrow_d_ff,
        "narrow_history": narrow_history,
        "narrow_local_development": narrow_local,
        "narrow_closed_loop_development": narrow_closed,
        "full_expert_parameters": full_expert_parameters,
        "expert_parameter_ratio_modal": primary_student.expert_transform_parameter_count()
        / full_expert_parameters,
        "idealized_expert_compute_ratio_modal": primary_student.idealized_expert_compute_ratio(),
        "expert_parameter_ratio_narrow": expert_parameter_count(narrow)
        / full_expert_parameters,
        "matrix_compute_ratio_narrow": narrow_d_ff / tiny.d_ff,
        "config_sha256": _hash(config_path),
        "parameterization": str(config["transplant"].get("parameterization", "scalar")),
    }
    checkpoint_path = output_dir / f"frozen-candidates-seed-{seed}.pt"
    torch.save(
        {
            "teacher_state": teacher.state_dict(),
            "modal_state": primary_student.state_dict(),
            "narrow_state": narrow.state_dict(),
            "vocabulary": corpus.itos,
            "metadata": metadata,
        },
        checkpoint_path,
    )
    metadata["checkpoint_sha256"] = _hash(checkpoint_path)
    (output_dir / f"development-seed-{seed}.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return metadata


def _load_frozen(
    config: dict[str, Any],
    output_dir: Path,
    seed: int,
) -> tuple[TinyMoELanguageModel, ScalarModalMoE | NeuronwiseModalMoE, ConventionalSwiGLUMoE, list[str], dict[str, Any]]:
    tiny = _tiny_config(config["model"])
    path = output_dir / f"frozen-candidates-seed-{seed}.pt"
    payload = torch.load(path, map_location="cpu", weights_only=False)
    metadata = payload["metadata"]
    vocabulary = list(payload["vocabulary"])
    teacher = TinyMoELanguageModel(len(vocabulary), tiny)
    teacher.load_state_dict(payload["teacher_state"])
    modal = _modal_class(config)(tiny.geometry, int(config["transplant"]["primary_rank"]))
    modal.load_state_dict(payload["modal_state"])
    narrow_geometry = MoEGeometry(
        tiny.d_model,
        int(metadata["narrow_d_ff"]),
        tiny.n_experts,
        tiny.top_k,
    )
    narrow = ConventionalSwiGLUMoE(narrow_geometry)
    narrow.load_state_dict(payload["narrow_state"])
    return teacher, modal, narrow, vocabulary, metadata


def finalize_controlled_transplant(
    config_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    config = _load(config_path)
    tiny = _tiny_config(config["model"])
    torch.set_num_threads(int(config.get("threads", 2)))
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    seeds = [int(value) for value in config["seeds"]]
    if len(seeds) < 3:
        raise CertificationError("at least three seeds are required")
    checkpoint_hashes: dict[str, str] = {}
    for seed in seeds:
        path = output_dir / f"frozen-candidates-seed-{seed}.pt"
        if not path.exists():
            raise CertificationError(f"missing frozen seed checkpoint: {path}")
        development_path = output_dir / f"development-seed-{seed}.json"
        if not development_path.exists():
            raise CertificationError(f"missing frozen seed metadata: {development_path}")
        development = json.loads(development_path.read_text(encoding="utf-8"))
        observed_hash = _hash(path)
        expected_hash = str(development.get("checkpoint_sha256", ""))
        if observed_hash != expected_hash:
            raise CertificationError(
                f"checkpoint digest mismatch for seed {seed}: "
                f"expected {expected_hash}, observed {observed_hash}"
            )
        if str(development.get("config_sha256")) != _hash(config_path):
            raise CertificationError(f"configuration digest mismatch for seed {seed}")
        checkpoint_hashes[str(seed)] = observed_hash
    data = config["data"]
    train_documents = generate_multidomain_documents(
        split="train", documents=int(data["train_documents"]), seed=int(data["train_seed"])
    )
    calibration_documents = generate_multidomain_documents(
        split="calibration", documents=int(data["calibration_documents"]), seed=int(data["calibration_seed"])
    )
    development_documents = generate_multidomain_documents(
        split="development", documents=int(data["development_documents"]), seed=int(data["development_seed"])
    )
    # This is the first point at which sealed content is materialized. The
    # confirmatory protocol resolves its seeds from an externally held secret.
    sealed_seed, sealed_window_seed, sealed_provenance = _resolve_sealed_seeds(data)
    sealed_documents = generate_multidomain_documents(
        split="sealed", documents=int(data["sealed_documents"]), seed=sealed_seed
    )
    layer_id = int(config["transplant"]["layer_id"])

    runs: list[dict[str, Any]] = []
    modal_records: list[dict[str, object]] = []
    narrow_records: list[dict[str, object]] = []
    for seed in seeds:
        teacher, modal, narrow, vocabulary, metadata = _load_frozen(config, output_dir, seed)
        corpus = CharacterCorpus(
            {
                "train": train_documents,
                "calibration": calibration_documents,
                "development": development_documents,
                "sealed": sealed_documents,
            },
            tiny.seq_len,
            vocabulary=vocabulary,
        )
        modal_model = install_student(teacher, modal, layer_id=layer_id)
        narrow_model = install_student(teacher, narrow, layer_id=layer_id)
        sealed_modal, modal_seed_records = evaluate_closed_loop(
            teacher,
            modal_model,
            corpus,
            split="sealed",
            windows_per_document=int(data["sealed_windows_per_document"]),
            seed=sealed_window_seed,
        )
        sealed_narrow, narrow_seed_records = evaluate_closed_loop(
            teacher,
            narrow_model,
            corpus,
            split="sealed",
            windows_per_document=int(data["sealed_windows_per_document"]),
            seed=sealed_window_seed,
        )
        for row in modal_seed_records:
            row.update({"seed": seed, "variant": "modal-primary"})
            modal_records.append(row)
        for row in narrow_seed_records:
            row.update({"seed": seed, "variant": "narrow-compute-matched"})
            narrow_records.append(row)

        interventions: dict[str, dict[str, float]] = {}
        for index, policy in enumerate(("mean-code", "shuffle-code", "zero-residual")):
            modified = _code_intervention(modal, policy, seed=seed + 5000 + index)
            modified_model = install_student(teacher, modified, layer_id=layer_id)
            metrics, _ = evaluate_closed_loop(
                teacher,
                modified_model,
                corpus,
                split="sealed",
                windows_per_document=int(data["sealed_windows_per_document"]),
                seed=sealed_window_seed,
            )
            interventions[policy] = metrics
        runs.append(
            {
                **metadata,
                "sealed_modal": sealed_modal,
                "sealed_narrow": sealed_narrow,
                "code_interventions": interventions,
            }
        )

    stats = config["statistics"]
    bootstrap_kwargs = {
        "samples": int(stats["bootstrap_samples"]),
        "confidence": float(stats["confidence"]),
    }
    modal_delta_bootstrap = crossed_hierarchical_bootstrap(
        modal_records,
        value_key="loss_delta",
        random_seed=int(stats["bootstrap_seed"]),
        **bootstrap_kwargs,
    )
    narrow_delta_bootstrap = crossed_hierarchical_bootstrap(
        narrow_records,
        value_key="loss_delta",
        random_seed=int(stats["bootstrap_seed"]) + 1,
        **bootstrap_kwargs,
    )
    narrow_index = {
        (row["seed"], row["document_id"], row["start"]): row for row in narrow_records
    }
    advantage_records: list[dict[str, object]] = []
    for row in modal_records:
        narrow_row = narrow_index[(row["seed"], row["document_id"], row["start"])]
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
        random_seed=int(stats["bootstrap_seed"]) + 2,
        **bootstrap_kwargs,
    )

    primary_rank = int(config["transplant"]["primary_rank"])
    primary_dev_local = [
        next(row for row in run["development_rank_curve"] if row["rank"] == primary_rank)["local"]["nrmse"]
        for run in runs
    ]
    modal_metrics = [run["sealed_modal"] for run in runs]
    parameter_ratios = [float(run["expert_parameter_ratio_modal"]) for run in runs]
    compute_ratios = [float(run["idealized_expert_compute_ratio_modal"]) for run in runs]
    intervention_kl_deltas = [
        float(metrics["kl_teacher_to_candidate"])
        - float(run["sealed_modal"]["kl_teacher_to_candidate"])
        for run in runs
        for metrics in run["code_interventions"].values()
    ]
    gates_cfg = config["gates"]
    fidelity_pass = (
        max(primary_dev_local) <= float(gates_cfg["development_local_nrmse_max"])
        and max(float(row["loss_delta"]) for row in modal_metrics) <= float(gates_cfg["sealed_loss_delta_max_per_seed"])
        and float(modal_delta_bootstrap["ucb"]) <= float(gates_cfg["sealed_loss_delta_ucb_max"])
        and max(float(row["kl_teacher_to_candidate"]) for row in modal_metrics) <= float(gates_cfg["sealed_kl_max"])
        and max(float(row["perplexity_ratio"]) for row in modal_metrics) <= float(gates_cfg["sealed_perplexity_ratio_max"])
        and min(float(row["top1_agreement"]) for row in modal_metrics) >= float(gates_cfg["sealed_top1_agreement_min"])
    )
    compression_pass = (
        max(parameter_ratios) <= float(gates_cfg["expert_parameter_ratio_max"])
        and max(compute_ratios) <= float(gates_cfg["idealized_compute_ratio_max"])
    )
    advantage_pass = float(advantage_bootstrap["ucb"]) <= float(gates_cfg["modal_minus_narrow_loss_ucb_max"])
    codes_pass = min(intervention_kl_deltas) >= float(gates_cfg["code_intervention_kl_increase_min"])
    gates = [
        _gate(
            "closed_loop_fidelity",
            fidelity_pass,
            {"development_local_nrmse": primary_dev_local, "sealed_per_seed": modal_metrics, "loss_delta_bootstrap": modal_delta_bootstrap},
            "primary rank must satisfy local, sealed loss, KL, perplexity, and top-1 gates in every seed",
        ),
        _gate(
            "parameter_and_compute_compression",
            compression_pass,
            {"expert_parameter_ratios": parameter_ratios, "idealized_compute_ratios": compute_ratios},
            "expert parameter and idealized compute ratios must remain below the preregistered limits",
        ),
        _gate(
            "compute_matched_advantage",
            advantage_pass,
            {"modal_minus_narrow_loss_bootstrap": advantage_bootstrap, "narrow_loss_delta_bootstrap": narrow_delta_bootstrap},
            f"UCB of Modal-minus-narrow loss penalty <= {gates_cfg['modal_minus_narrow_loss_ucb_max']}",
        ),
        _gate(
            "expert_code_causality",
            codes_pass,
            {"minimum_kl_increase": min(intervention_kl_deltas), "all_kl_increases": intervention_kl_deltas},
            f"every code intervention must increase teacher KL by at least {gates_cfg['code_intervention_kl_increase_min']}",
        ),
    ]
    if fidelity_pass and compression_pass and advantage_pass and codes_pass:
        verdict = "CONTROLLED_TRANSPLANT_PASS"
        go = True
        interpretation = (
            f"The primary {config['transplant'].get('parameterization', 'scalar')} Modal student preserved the separately trained conventional teacher, met compression gates, and was non-inferior to the compute-matched narrow baseline under the paired hierarchical test. A preregistered OLMoE single-layer smoke is justified; Qwen remains premature until OLMoE multi-layer confirmation."
        )
    elif fidelity_pass and compression_pass and codes_pass:
        verdict = "CONTROLLED_TRANSPLANT_FUNCTIONAL_ONLY"
        go = False
        interpretation = (
            f"The {config['transplant'].get('parameterization', 'scalar')} Modal student preserved closed-loop function and compressed parameters/idealized compute, but failed the preregistered comparison against the strong compute-matched conventional narrowing baseline. This establishes transplantability, not architectural advantage."
        )
    else:
        verdict = "CONTROLLED_TRANSPLANT_FAIL"
        go = False
        interpretation = (
            "At least one essential fidelity, compression, or causal-use gate failed. Repair the student or harness under a new preregistered protocol version before any real-checkpoint transplant."
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
            "source_hashes": _source_hashes(config_path.resolve().parents[1]),
            "frozen_checkpoint_sha256": checkpoint_hashes,
            "sealed_evaluation": sealed_provenance,
            "sealed_materialization_note": "sealed documents and windows were generated only in finalize, after all seed checkpoints existed",
        },
        "configuration": config,
        "runs": runs,
        "sealed_statistics": {
            "modal_loss_delta": modal_delta_bootstrap,
            "narrow_loss_delta": narrow_delta_bootstrap,
            "modal_minus_narrow_loss_delta": advantage_bootstrap,
            "modal_by_domain": summarize_groups(modal_records, value_key="loss_delta", group_keys=("domain",)),
        },
        "decision": {
            "verdict": verdict,
            "go_for_real_checkpoint": go,
            "gates": gates,
            "interpretation": interpretation,
            "next_action": "prepare preregistered OLMoE single-layer smoke" if go else "improve Modal student under the same controls; do not run Qwen",
        },
    }
    (output_dir / "metrics.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    # Reveal the controlled synthetic holdout only after the decision has been
    # computed. This preserves one-shot validity while allowing exact replay.
    with (output_dir / "sealed_documents.jsonl").open("w", encoding="utf-8") as handle:
        for document in sealed_documents:
            handle.write(json.dumps(asdict(document), ensure_ascii=False, sort_keys=True) + "\n")
    if sealed_provenance.get("mode") == "sha256-committed-secret":
        environment_name = str(sealed_provenance["secret_environment"])
        reveal_secret = os.environ[environment_name]
        reveal = {
            "protocol_version": config["protocol_version"],
            "revealed_after_decision": True,
            "revealed_at": datetime.now(timezone.utc).isoformat(),
            "secret": reveal_secret,
            "seed_commitment_sha256": hashlib.sha256(
                reveal_secret.encode("utf-8")
            ).hexdigest(),
            "document_seed": sealed_seed,
            "window_seed": sealed_window_seed,
            "warning": "This protocol version is consumed. Do not reuse this holdout for model selection.",
        }
        (output_dir / "sealed_reveal.json").write_text(
            json.dumps(reveal, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    with (output_dir / "sealed_per_window.jsonl").open("w", encoding="utf-8") as handle:
        for row in modal_records + narrow_records:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    (output_dir / "config.resolved.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    _write_report(output_dir, payload)
    (output_dir / "environment.txt").write_text(
        f"python={sys.version.replace(chr(10), ' ')}\nplatform={platform.platform()}\ntorch={torch.__version__}\nnumpy={np.__version__}\nthreads={torch.get_num_threads()}\n",
        encoding="utf-8",
    )
    files = sorted(path for path in output_dir.iterdir() if path.is_file() and path.name != "sha256sums.txt")
    (output_dir / "sha256sums.txt").write_text(
        "\n".join(f"{_hash(path)}  {path.name}" for path in files) + "\n",
        encoding="utf-8",
    )
    return payload
