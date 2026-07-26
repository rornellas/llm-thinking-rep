#!/usr/bin/env python3
"""Permutation-invariant cross-expert neuron reuse diagnostic.

Each SwiGLU intermediate neuron is treated as a functional triplet
``(gate row, up row, down column)``. The exact up/down sign and reciprocal
scale symmetries are removed by component normalization and joint sign
matching. Query neurons are matched to neurons from other experts using
training coordinates, then evaluated on disjoint held-out coordinates and on
all 2,048 coordinates.

This is a conservative screen for a global neuron dictionary. If even the best
match among all 64k existing neurons has poor exact similarity, a compact
dictionary that reuses atoms across experts is unlikely to preserve raw
weights without activation-aware retraining.
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from modal_moe_weight_probe_v2 import (
    HttpRangeSafeTensorSource,
    LocalSafeTensorSource,
    build_default_remote_url,
    discover_layer_keys,
)

EPS = 1e-12
WEIGHTS = (1.0, 0.5, 0.5)
MAX_SCORE = sum(WEIGHTS)


@dataclass
class MatchResult:
    query_expert: int
    query_neuron: int
    train_top1_expert: int
    train_top1_neuron: int
    train_top1_score: float
    holdout_top1_score: float
    holdout_top1_gate_cos: float
    holdout_top1_up_cos: float
    holdout_top1_down_cos: float
    oracle_top64_expert: int
    oracle_top64_neuron: int
    oracle_top64_holdout_score: float
    exact_top1_score: float
    exact_top1_gate_cos: float
    exact_top1_up_cos: float
    exact_top1_down_cos: float
    exact_oracle64_score: float
    exact_oracle64_gate_cos: float
    exact_oracle64_up_cos: float
    exact_oracle64_down_cos: float
    exact_random_score: float


def row_normalize(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(norms, EPS)


def disjoint_indices(total: int, train_n: int, holdout_n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    if train_n + holdout_n > total:
        raise ValueError("signature dimensions exceed feature dimension")
    rng = np.random.default_rng(seed)
    ids = rng.choice(total, size=train_n + holdout_n, replace=False)
    return np.sort(ids[:train_n]), np.sort(ids[train_n:])


def query_neurons(intermediate: int, count: int, seed: int, expert_id: int) -> np.ndarray:
    rng = np.random.default_rng(seed + 1009 * expert_id)
    return np.sort(rng.choice(intermediate, size=count, replace=False)).astype(np.int32)


def build_signatures(
    source: Any,
    discovered: dict[str, list[tuple[int, str]]],
    train_input: np.ndarray,
    holdout_input: np.ndarray,
    train_output: np.ndarray,
    holdout_output: np.ndarray,
    queries_per_expert: int,
    query_seed: int,
) -> dict[str, Any]:
    key_maps = {p: dict(entries) for p, entries in discovered.items()}
    expert_ids = sorted(key_maps["gate"])
    first_desc = source.descriptor(key_maps["gate"][expert_ids[0]])
    intermediate, hidden = first_desc.shape
    total = len(expert_ids) * intermediate

    arrays = {
        "g_train": np.empty((total, len(train_input)), dtype=np.float32),
        "u_train": np.empty((total, len(train_input)), dtype=np.float32),
        "d_train": np.empty((total, len(train_output)), dtype=np.float32),
        "g_hold": np.empty((total, len(holdout_input)), dtype=np.float32),
        "u_hold": np.empty((total, len(holdout_input)), dtype=np.float32),
        "d_hold": np.empty((total, len(holdout_output)), dtype=np.float32),
        "owner": np.empty(total, dtype=np.int16),
        "neuron": np.empty(total, dtype=np.int16),
    }
    query_records: list[dict[str, Any]] = []

    for expert_position, expert_id in enumerate(expert_ids):
        gate = source.get_tensor(key_maps["gate"][expert_id]).to(torch.float32).cpu().numpy()
        up = source.get_tensor(key_maps["up"][expert_id]).to(torch.float32).cpu().numpy()
        down = source.get_tensor(key_maps["down"][expert_id]).to(torch.float32).cpu().numpy()
        start = expert_position * intermediate
        stop = start + intermediate
        arrays["g_train"][start:stop] = row_normalize(gate[:, train_input])
        arrays["u_train"][start:stop] = row_normalize(up[:, train_input])
        arrays["d_train"][start:stop] = row_normalize(down[train_output, :].T)
        arrays["g_hold"][start:stop] = row_normalize(gate[:, holdout_input])
        arrays["u_hold"][start:stop] = row_normalize(up[:, holdout_input])
        arrays["d_hold"][start:stop] = row_normalize(down[holdout_output, :].T)
        arrays["owner"][start:stop] = expert_id
        arrays["neuron"][start:stop] = np.arange(intermediate, dtype=np.int16)

        for neuron_id in query_neurons(intermediate, queries_per_expert, query_seed, expert_id):
            query_records.append({
                "expert": int(expert_id),
                "neuron": int(neuron_id),
                "global_index": int(start + neuron_id),
                "gate": np.asarray(gate[neuron_id], dtype=np.float32).copy(),
                "up": np.asarray(up[neuron_id], dtype=np.float32).copy(),
                "down": np.asarray(down[:, neuron_id], dtype=np.float32).copy(),
            })
        print(f"signatures: expert {expert_id} ({expert_position + 1}/{len(expert_ids)})", flush=True)
        del gate, up, down

    arrays["queries"] = query_records
    arrays["expert_ids"] = expert_ids
    arrays["intermediate"] = intermediate
    arrays["hidden"] = hidden
    arrays["key_maps"] = key_maps
    return arrays


def score_components(
    qg: np.ndarray,
    qu: np.ndarray,
    qd: np.ndarray,
    cg: np.ndarray,
    cu: np.ndarray,
    cd: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    sg = qg @ cg.T
    su = qu @ cu.T
    sd = qd @ cd.T
    ud = WEIGHTS[1] * su + WEIGHTS[2] * sd
    signs = np.where(ud >= 0.0, 1.0, -1.0).astype(np.float32)
    total = (WEIGHTS[0] * sg + np.abs(ud)) / MAX_SCORE
    return total, sg, signs * su, signs * sd, signs


def exact_component_score(query: dict[str, Any], candidate: dict[str, np.ndarray]) -> tuple[float, float, float, float]:
    qg = query["gate"] / max(float(np.linalg.norm(query["gate"])), EPS)
    qu = query["up"] / max(float(np.linalg.norm(query["up"])), EPS)
    qd = query["down"] / max(float(np.linalg.norm(query["down"])), EPS)
    cg = candidate["gate"] / max(float(np.linalg.norm(candidate["gate"])), EPS)
    cu = candidate["up"] / max(float(np.linalg.norm(candidate["up"])), EPS)
    cd = candidate["down"] / max(float(np.linalg.norm(candidate["down"])), EPS)
    sg = float(np.dot(qg, cg))
    su0 = float(np.dot(qu, cu))
    sd0 = float(np.dot(qd, cd))
    sign = 1.0 if WEIGHTS[1] * su0 + WEIGHTS[2] * sd0 >= 0 else -1.0
    su, sd = sign * su0, sign * sd0
    total = (WEIGHTS[0] * sg + WEIGHTS[1] * su + WEIGHTS[2] * sd) / MAX_SCORE
    return total, sg, su, sd


def run_matching(data: dict[str, Any], top_n: int, batch_size: int, seed: int) -> tuple[list[dict[str, Any]], dict[int, set[int]]]:
    queries: list[dict[str, Any]] = data["queries"]
    owner = data["owner"]
    rng = np.random.default_rng(seed)
    preliminary: list[dict[str, Any]] = []
    needed: dict[int, set[int]] = {}

    for batch_start in range(0, len(queries), batch_size):
        batch = queries[batch_start:batch_start + batch_size]
        qidx = np.array([q["global_index"] for q in batch], dtype=np.int64)
        train_total, _, _, _, _ = score_components(
            data["g_train"][qidx], data["u_train"][qidx], data["d_train"][qidx],
            data["g_train"], data["u_train"], data["d_train"],
        )
        for row, query in enumerate(batch):
            train_total[row, owner == query["expert"]] = -np.inf
        kth = min(top_n, train_total.shape[1])
        top_indices = np.argpartition(train_total, -kth, axis=1)[:, -kth:]

        for row, query in enumerate(batch):
            candidate_ids = top_indices[row]
            order = np.argsort(train_total[row, candidate_ids])[::-1]
            candidate_ids = candidate_ids[order]
            top1 = int(candidate_ids[0])
            hold_total, hold_g, hold_u, hold_d, _ = score_components(
                data["g_hold"][query["global_index"]:query["global_index"] + 1],
                data["u_hold"][query["global_index"]:query["global_index"] + 1],
                data["d_hold"][query["global_index"]:query["global_index"] + 1],
                data["g_hold"][candidate_ids], data["u_hold"][candidate_ids], data["d_hold"][candidate_ids],
            )
            oracle_pos = int(np.argmax(hold_total[0]))
            oracle = int(candidate_ids[oracle_pos])
            different = np.flatnonzero(owner != query["expert"])
            random_id = int(rng.choice(different))
            for gid in (top1, oracle, random_id):
                needed.setdefault(int(data["owner"][gid]), set()).add(int(data["neuron"][gid]))
            preliminary.append({
                "query": query,
                "top1": top1,
                "oracle": oracle,
                "random": random_id,
                "train_top1_score": float(train_total[row, top1]),
                "holdout_top1_score": float(hold_total[0, 0]),
                "holdout_top1_gate_cos": float(hold_g[0, 0]),
                "holdout_top1_up_cos": float(hold_u[0, 0]),
                "holdout_top1_down_cos": float(hold_d[0, 0]),
                "oracle_top64_holdout_score": float(hold_total[0, oracle_pos]),
            })
        print(f"matched {min(batch_start + batch_size, len(queries))}/{len(queries)} queries", flush=True)
    return preliminary, needed


def load_exact_candidates(data: dict[str, Any], source: Any, needed: dict[int, set[int]]) -> dict[tuple[int, int], dict[str, np.ndarray]]:
    result: dict[tuple[int, int], dict[str, np.ndarray]] = {}
    key_maps = data["key_maps"]
    for position, expert_id in enumerate(sorted(needed), start=1):
        ids = sorted(needed[expert_id])
        gate = source.get_tensor(key_maps["gate"][expert_id]).to(torch.float32).cpu().numpy()
        up = source.get_tensor(key_maps["up"][expert_id]).to(torch.float32).cpu().numpy()
        down = source.get_tensor(key_maps["down"][expert_id]).to(torch.float32).cpu().numpy()
        for neuron_id in ids:
            result[(expert_id, neuron_id)] = {
                "gate": np.asarray(gate[neuron_id], dtype=np.float32).copy(),
                "up": np.asarray(up[neuron_id], dtype=np.float32).copy(),
                "down": np.asarray(down[:, neuron_id], dtype=np.float32).copy(),
            }
        print(f"exact candidates: expert {expert_id} ({position}/{len(needed)})", flush=True)
        del gate, up, down
    return result


def finalize(preliminary: Sequence[dict[str, Any]], data: dict[str, Any], exact: dict[tuple[int, int], dict[str, np.ndarray]]) -> list[MatchResult]:
    results: list[MatchResult] = []
    for item in preliminary:
        query = item["query"]
        def identity(gid: int) -> tuple[int, int]:
            return int(data["owner"][gid]), int(data["neuron"][gid])
        top1_key = identity(item["top1"])
        oracle_key = identity(item["oracle"])
        random_key = identity(item["random"])
        top1_exact = exact_component_score(query, exact[top1_key])
        oracle_exact = exact_component_score(query, exact[oracle_key])
        random_exact = exact_component_score(query, exact[random_key])
        results.append(MatchResult(
            query_expert=query["expert"],
            query_neuron=query["neuron"],
            train_top1_expert=top1_key[0],
            train_top1_neuron=top1_key[1],
            train_top1_score=item["train_top1_score"],
            holdout_top1_score=item["holdout_top1_score"],
            holdout_top1_gate_cos=item["holdout_top1_gate_cos"],
            holdout_top1_up_cos=item["holdout_top1_up_cos"],
            holdout_top1_down_cos=item["holdout_top1_down_cos"],
            oracle_top64_expert=oracle_key[0],
            oracle_top64_neuron=oracle_key[1],
            oracle_top64_holdout_score=item["oracle_top64_holdout_score"],
            exact_top1_score=top1_exact[0],
            exact_top1_gate_cos=top1_exact[1],
            exact_top1_up_cos=top1_exact[2],
            exact_top1_down_cos=top1_exact[3],
            exact_oracle64_score=oracle_exact[0],
            exact_oracle64_gate_cos=oracle_exact[1],
            exact_oracle64_up_cos=oracle_exact[2],
            exact_oracle64_down_cos=oracle_exact[3],
            exact_random_score=random_exact[0],
        ))
    return results


def describe(values: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "p05": float(np.percentile(values, 5)),
        "p25": float(np.percentile(values, 25)),
        "p50": float(np.percentile(values, 50)),
        "p75": float(np.percentile(values, 75)),
        "p95": float(np.percentile(values, 95)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


def summarize(results: Sequence[MatchResult], hidden: int, intermediate: int, top_k: int) -> dict[str, Any]:
    arrays = {
        name: np.array([getattr(r, name) for r in results], dtype=np.float64)
        for name in (
            "train_top1_score", "holdout_top1_score", "oracle_top64_holdout_score",
            "exact_top1_score", "exact_oracle64_score", "exact_random_score",
            "exact_top1_gate_cos", "exact_top1_up_cos", "exact_top1_down_cos",
            "exact_oracle64_gate_cos", "exact_oracle64_up_cos", "exact_oracle64_down_cos",
        )
    }
    thresholds = {}
    for label in ("exact_top1_score", "exact_oracle64_score"):
        thresholds[label] = {
            str(t): float(np.mean(arrays[label] >= t))
            for t in (0.25, 0.40, 0.60, 0.80, 0.90)
        }
    dictionary_budgets = []
    for q in (1024, 2048, 4096, 8192, 16384, 32768, 65536):
        dictionary_budgets.append({
            "atoms": q,
            "idealized_compute_ratio": q / (top_k * intermediate),
            "idealized_compute_reduction": 1.0 - q / (top_k * intermediate),
        })
    verdict = "PASS" if (
        float(np.median(arrays["exact_top1_score"])) >= 0.80
        and float(np.mean(arrays["exact_top1_score"] >= 0.70)) >= 0.90
    ) else ("BORDERLINE" if float(np.median(arrays["exact_oracle64_score"])) >= 0.60 else "FAIL")
    return {
        "verdict": verdict,
        "queries": len(results),
        "statistics": {name: describe(values) for name, values in arrays.items()},
        "threshold_fractions": thresholds,
        "dictionary_budgets": dictionary_budgets,
        "interpretation": {
            "score_range": [-1.0, 1.0],
            "score_definition": "(gate cosine + 0.5*signed up cosine + 0.5*signed down cosine) / 2",
            "top1_is_selected_only_on_training_coordinates": True,
            "oracle64_uses_holdout_only_as_an_upper_bound": True,
        },
    }


def write_outputs(output_dir: Path, payload: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "neuron_dictionary.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    rows = payload["matches"]
    with (output_dir / "matches.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader(); writer.writerows(rows)
    s = payload["summary"]
    stats = s["statistics"]
    lines = [
        "# Test 0.8 — permutation-invariant neuron dictionary screen",
        "",
        f"**Decision:** **{s['verdict']}**",
        "",
        "Queries are matched to all neurons from other experts using training coordinates and then scored on disjoint holdout coordinates and all 2,048 dimensions.",
        "",
        "| Metric | Mean | Median | p95 | Maximum |",
        "|---|---:|---:|---:|---:|",
    ]
    for key, title in (
        ("train_top1_score", "Train-selected top1 / training"),
        ("holdout_top1_score", "Train-selected top1 / holdout"),
        ("exact_top1_score", "Train-selected top1 / exact"),
        ("exact_oracle64_score", "Best-holdout among train top64 / exact"),
        ("exact_random_score", "Random cross-expert / exact"),
    ):
        v = stats[key]
        lines.append(f"| {title} | {v['mean']:.4f} | {v['p50']:.4f} | {v['p95']:.4f} | {v['max']:.4f} |")
    lines += [
        "",
        "A compact directly executed dictionary needs very high cross-expert reuse; cosine-like scores near zero indicate unrelated triplets, while scores near one indicate reusable atoms.",
        "The top64 oracle is deliberately optimistic and is not a deployable selection rule.",
    ]
    (output_dir / "VERDICT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def synthetic_self_test() -> None:
    rng = np.random.default_rng(73)
    experts, intermediate, hidden, atoms = 8, 32, 96, 24
    ag = row_normalize(rng.normal(size=(atoms, hidden)).astype(np.float32))
    au = row_normalize(rng.normal(size=(atoms, hidden)).astype(np.float32))
    ad = row_normalize(rng.normal(size=(atoms, hidden)).astype(np.float32))
    queries = []
    candidates = []
    owner = []
    for e in range(experts):
        assignment = rng.integers(0, atoms, size=intermediate)
        for j, a in enumerate(assignment):
            sign = rng.choice([-1.0, 1.0])
            candidates.append((ag[a] + rng.normal(scale=0.01, size=hidden), sign * au[a] + rng.normal(scale=0.01, size=hidden), sign * ad[a] + rng.normal(scale=0.01, size=hidden)))
            owner.append(e)
            if j < 4:
                queries.append((e, len(candidates) - 1))
    cg = row_normalize(np.stack([x[0] for x in candidates]).astype(np.float32))
    cu = row_normalize(np.stack([x[1] for x in candidates]).astype(np.float32))
    cd = row_normalize(np.stack([x[2] for x in candidates]).astype(np.float32))
    owner_arr = np.asarray(owner)
    scores = []
    for e, idx in queries:
        total, *_ = score_components(cg[idx:idx+1], cu[idx:idx+1], cd[idx:idx+1], cg, cu, cd)
        total[0, owner_arr == e] = -np.inf
        scores.append(float(np.max(total)))
    if np.median(scores) < 0.95:
        raise AssertionError(scores)
    print(f"self-test passed: median cross-expert atom score={np.median(scores):.4f}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--model-id", default="allenai/OLMoE-1B-7B-0924")
    parser.add_argument("--revision", default="3a970199d0f87db4e3e57275abb93812bf10fd83")
    parser.add_argument("--shard-name", default="model-00002-of-00003.safetensors")
    parser.add_argument("--shard-path", type=Path)
    parser.add_argument("--remote-shard-url")
    parser.add_argument("--cache-dir", type=Path, default=Path(".cache/olmoe-neuron-ranges"))
    parser.add_argument("--layer", type=int, default=7)
    parser.add_argument("--train-dims", type=int, default=48)
    parser.add_argument("--holdout-dims", type=int, default=96)
    parser.add_argument("--queries-per-expert", type=int, default=8)
    parser.add_argument("--top-n", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260729)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--output-dir", type=Path, default=Path("results/neuron_dictionary"))
    args = parser.parse_args(argv)

    if args.self_test:
        synthetic_self_test(); return 0

    started = time.perf_counter()
    if args.shard_path:
        source = LocalSafeTensorSource(args.shard_path)
    else:
        url = args.remote_shard_url or build_default_remote_url(args.model_id, args.revision, args.shard_name)
        source = HttpRangeSafeTensorSource(url, cache_dir=args.cache_dir)
    discovered = discover_layer_keys(source.keys(), args.layer)
    first_key = discovered["gate"][0][1]
    intermediate, hidden = source.descriptor(first_key).shape
    train_in, hold_in = disjoint_indices(hidden, args.train_dims, args.holdout_dims, args.seed + 1)
    train_out, hold_out = disjoint_indices(hidden, args.train_dims, args.holdout_dims, args.seed + 2)
    data = build_signatures(
        source, discovered, train_in, hold_in, train_out, hold_out,
        args.queries_per_expert, args.seed + 3,
    )
    preliminary, needed = run_matching(data, args.top_n, args.batch_size, args.seed + 4)
    exact_candidates = load_exact_candidates(data, source, needed)
    results = finalize(preliminary, data, exact_candidates)
    summary = summarize(results, hidden, intermediate, args.top_k)
    payload = {
        "metadata": {
            "model_id": args.model_id,
            "revision": args.revision,
            "layer": args.layer,
            "experts": len(data["expert_ids"]),
            "intermediate": intermediate,
            "hidden": hidden,
            "train_dims_per_component": args.train_dims,
            "holdout_dims_per_component": args.holdout_dims,
            "queries_per_expert": args.queries_per_expert,
            "top_n": args.top_n,
            "seed": args.seed,
            "elapsed_seconds": time.perf_counter() - started,
            "source": source.metadata(),
            "same_expert_candidates_are_excluded": True,
        },
        "summary": summary,
        "matches": [asdict(x) for x in results],
    }
    write_outputs(args.output_dir, payload)
    print((args.output_dir / "VERDICT.md").read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
