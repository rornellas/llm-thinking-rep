#!/usr/bin/env python3
"""MUI-1: prospective development screen; never loads test/OOD arrays."""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import random
import subprocess
import sys
import time
import numpy as np
import torch
import torch.nn.functional as F
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pre_qwen_certification.native_compact import (
    NativeArchitectureSpec, build_paired_candidate_models, candidate_accounting,
    evaluate_native_model, FULL, NARROW, PRIMARY, route_health,
)
from pre_qwen_certification.reality_gate import routing_distribution
from pre_qwen_certification.reality_gate_data import load_prepared_arrays, sha256_file
from scripts.run_native_compact_gate_2a_seed_impl import _tiny_config, _training_corpus

ARMS = (FULL, NARROW, 'legacy', 'spectral-tiny', 'energy-gaussian', 'energy-spectral')
COMPACT = ARMS[2:]
SEEDS = (904031, 904043, 904051, 904073)
STEPS = 800
EVAL_SEED = 904091
OUT = ROOT / 'results/mode-utilization-intervention-1'


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')


@torch.no_grad()
def intervene(model, arm):
    """Preserve original factor subspaces; intervene only on scale/spectrum/gauge."""
    assert arm in COMPACT
    if arm == 'legacy':
        return
    for block in model.blocks:
        module = block.moe
        for bank in ('gate', 'up', 'down'):
            common = getattr(module, 'common_' + bank)
            original_norm = float(common.norm())
            lefts, rights = getattr(module, bank + '_left'), getattr(module, bank + '_right')
            for left, right in zip(lefts, rights, strict=True):
                residual = left.double() @ right.double()
                target_norm = original_norm / math.sqrt(2) if arm.startswith('energy') else float(residual.norm())
                if arm in ('spectral-tiny', 'energy-spectral'):
                    u, _, vh = torch.linalg.svd(residual, full_matrices=False)
                    rank = left.shape[1]
                    root = math.sqrt(target_norm / math.sqrt(rank))
                    left.copy_(u[:, :rank] * root)
                    right.copy_(vh[:rank, :] * root)
                else:
                    scale = math.sqrt(target_norm / float(residual.norm()))
                    left.mul_(scale)
                    right.mul_(scale)
            if arm.startswith('energy'):
                common.div_(math.sqrt(2))


def build(vocab, cfg, seed):
    original = build_paired_candidate_models(vocab, cfg, NativeArchitectureSpec(8, 42), seed=seed)
    accounting = {key: value.as_dict() for key, value in candidate_accounting(original, cfg).items()}
    models = {FULL: original[FULL], NARROW: original[NARROW]}
    for arm in COMPACT:
        models[arm] = copy.deepcopy(original[PRIMARY])
        intervene(models[arm], arm)
    return models, {arm: accounting[arm if arm in (FULL, NARROW) else PRIMARY] for arm in ARMS}


@torch.no_grad()
def diagnostic(model):
    values = {'stable_rank': [], 'residual_common_ratio': [], 'effective_cosine': [],
              'effective_frobenius': [], 'output_diversity': []}
    gen = torch.Generator().manual_seed(904099)
    x = torch.randn(32, model.config.d_model, generator=gen)
    for block in model.blocks:
        m = block.moe
        for bank in ('gate', 'up', 'down'):
            if hasattr(m, 'common_' + bank):
                common = getattr(m, 'common_' + bank).double()
                matrices = []
                for left, right in zip(getattr(m, bank + '_left'), getattr(m, bank + '_right'), strict=True):
                    residual = left.double() @ right.double()
                    singular = torch.linalg.svdvals(residual)
                    values['stable_rank'].append(float(singular.square().sum() / singular[0].square().clamp_min(1e-30)))
                    values['residual_common_ratio'].append(float(residual.norm() / common.norm().clamp_min(1e-30)))
                    matrices.append(common + residual)
                weights = torch.stack(matrices)
            else:
                weights = getattr(m, bank).double()
            flat = weights.flatten(1)
            normed = F.normalize(flat, dim=-1)
            cosine = normed @ normed.T
            offdiag = ~torch.eye(len(flat), dtype=torch.bool)
            values['effective_cosine'].append(float(cosine[offdiag].mean()))
            values['effective_frobenius'].append(float(flat.norm(dim=-1).mean()))
        outputs = []
        for expert in range(m.geometry.n_experts):
            if hasattr(m, '_expert_output'):
                y = m._expert_output(x, expert)
            else:
                y = F.linear(F.silu(F.linear(x, m.gate[expert])) * F.linear(x, m.up[expert]), m.down[expert])
            outputs.append(y.double())
        outputs = torch.stack(outputs)
        values['output_diversity'].append(float((outputs - outputs.mean(0, keepdim=True)).square().sum() / outputs.square().sum().clamp_min(1e-30)))
    return {key: float(np.mean(val)) if val else None for key, val in values.items()}


@torch.no_grad()
def self_test():
    spec = yaml.safe_load((ROOT / 'configs/native_compact_gate_2a.yaml').read_text())
    cfg = _tiny_config(spec['scales']['small']['model'], steps=STEPS)
    models, accounting = build(64, cfg, 19081)
    reference = models['legacy'].state_dict()
    counts = set()
    algebra_error = {}
    for arm, model in models.items():
        for name, tensor in model.state_dict().items():
            if '.moe.' not in name or '.moe.router.' in name:
                assert torch.equal(tensor, reference[name]), (arm, name)
        if arm not in COMPACT:
            continue
        counts.add(sum(p.numel() for p in model.parameters()))
        m = model.blocks[0].moe
        gen = torch.Generator().manual_seed(55)
        x = torch.randn(17, cfg.d_model, generator=gen)
        direct, routing = m(x)
        g, u, d = m.reconstruct_weights()
        expanded = torch.zeros_like(direct)
        for expert in range(cfg.n_experts):
            ids, slots = (routing.top_ids == expert).nonzero(as_tuple=True)
            chosen = x[ids]
            y = F.linear(F.silu(F.linear(chosen, g[expert])) * F.linear(chosen, u[expert]), d[expert])
            expanded.index_add_(0, ids, y * routing.weights[ids, slots, None])
        err = float((expanded - direct).abs().max())
        assert torch.allclose(expanded, direct, rtol=3e-5, atol=3e-6), (arm, err)
        algebra_error[arm] = err
    assert len(counts) == 1
    for arm in ('spectral-tiny', 'energy-spectral'):
        assert abs(diagnostic(models[arm])['stable_rank'] - 8) < 1e-5
    for layer in range(cfg.n_layers):
        for bank in ('gate', 'up', 'down'):
            legacy = models['legacy'].blocks[layer].moe
            c = getattr(legacy, 'common_' + bank)
            for arm in COMPACT[1:]:
                changed = models[arm].blocks[layer].moe
                for i in range(cfg.n_experts):
                    l, r = getattr(changed, bank + '_left')[i], getattr(changed, bank + '_right')[i]
                    expected = c.norm() / math.sqrt(2) if arm.startswith('energy') else (getattr(legacy, bank + '_left')[i] @ getattr(legacy, bank + '_right')[i]).norm()
                    assert torch.isclose((l @ r).norm(), expected, rtol=2e-5, atol=1e-8)
    again, _ = build(64, cfg, 19081)
    for arm in ARMS:
        assert all(torch.equal(v, again[arm].state_dict()[k]) for k, v in models[arm].state_dict().items())
    return {'passed': True, 'algebra_max_abs_error': algebra_error, 'compact_parameter_counts': list(counts)}


@torch.inference_mode()
def benchmark(models, vocab):
    # No KV cache: short-sequence full forwards, not autoregressive serving throughput.
    result = {}
    rng = random.Random(904101)
    for length in (1, 64):
        tokens = torch.arange(length).remainder(vocab)[None, :]
        for model in models.values():
            model.eval()
            for _ in range(5):
                model(tokens)
        timing = {arm: [] for arm in ARMS}
        for _ in range(9):
            order = list(ARMS)
            rng.shuffle(order)
            for arm in order:
                start = time.perf_counter_ns()
                for _ in range(10):
                    models[arm](tokens)
                timing[arm].append((time.perf_counter_ns() - start) / 1e6 / 10)
        result[str(length)] = {arm: {'blocks_ms': vals, 'median_ms': float(np.median(vals))} for arm, vals in timing.items()}
    return result


def run(seed):
    if seed not in SEEDS:
        raise ValueError('seed is not preregistered')
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    tests = self_test()
    cfg_path = ROOT / 'configs/native_compact_gate_2a.yaml'
    config = yaml.safe_load(cfg_path.read_text())
    small = config['scales']['small']
    cfg = _tiny_config(small['model'], steps=STEPS)
    data_root = ROOT / 'data/native-compact-gate-2a'
    arrays, manifest = load_prepared_arrays(data_root, splits=('train', 'validation'))
    assert set(arrays) == {'train', 'validation'}
    corpus, train_docs, calibration_docs = _training_corpus(arrays, manifest, config, small)
    models, accounting = build(corpus.vocab_size, cfg, seed)
    optimizers = {arm: torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay) for arm, model in models.items()}
    histories = {arm: [] for arm in ARMS}
    recent = {arm: [] for arm in ARMS}
    initial_gradients = {}
    raw_final = {}
    stream = hashlib.sha256()
    generator = torch.Generator().manual_seed(seed + 101)
    training_seconds = {arm: 0.0 for arm in ARMS}
    for step in range(STEPS + 1):
        if step > 0:
            tokens, targets = corpus.sample_batch('train', cfg.batch_size, generator)
            stream.update(tokens.numpy().tobytes())
            stream.update(targets.numpy().tobytes())
            order = list(ARMS)
            # Rotate execution order to avoid a fixed timing/thermal advantage.
            order = order[step % len(order):] + order[:step % len(order)]
            for arm in order:
                model = models[arm]
                model.train()
                torch.manual_seed(seed * 1000 + step)
                start = time.perf_counter()
                logits, auxiliary, _ = model(tokens)
                loss = F.cross_entropy(logits.flatten(0, 1), targets.flatten())
                objective = loss + cfg.aux_weight * auxiliary
                optimizers[arm].zero_grad(set_to_none=True)
                objective.backward()
                if step == 1:
                    groups = {'common': 0.0, 'factors': 0.0, 'router': 0.0, 'other': 0.0}
                    for name, p in model.named_parameters():
                        if p.grad is None:
                            continue
                        group = 'common' if '.common_' in name else 'factors' if '_left.' in name or '_right.' in name else 'router' if '.router.' in name else 'other'
                        groups[group] += float(p.grad.double().square().sum())
                    initial_gradients[arm] = {key: math.sqrt(val) for key, val in groups.items()}
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                optimizers[arm].step()
                training_seconds[arm] += time.perf_counter() - start
                recent[arm].append(float(loss.detach()))
        if step not in (0, 200, 400, STEPS):
            continue
        for arm, model in models.items():
            metrics, records = evaluate_native_model(model, corpus, split='calibration', windows_per_document=1, evaluation_seed=EVAL_SEED)
            row = {'step': step, 'loss': metrics['loss'], 'train_loss': float(np.mean(recent[arm][-200:])) if recent[arm] else None, **diagnostic(model)}
            histories[arm].append(row)
            print(json.dumps({'seed': seed, 'arm': arm, **row}), flush=True)
            if step == STEPS:
                raw_final[arm] = records
    route = {}
    for arm, model in models.items():
        route[arm] = {}
        for layer in range(cfg.n_layers):
            dist = routing_distribution(model, corpus, split='calibration', layer_id=layer, windows_per_document=1, seed=EVAL_SEED)
            route[arm][str(layer)] = {'frequencies': list(map(float, dist)), **route_health(dist, cfg.top_k)}
    latency = benchmark(models, corpus.vocab_size)
    OUT.mkdir(parents=True, exist_ok=True)
    checkpoint = OUT / f'seed-{seed}.pt'
    torch.save({arm: model.state_dict() for arm, model in models.items()}, checkpoint)
    source = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    data_hashes = {}
    for split, prepared in arrays.items():
        data_hashes[split] = {key: hashlib.sha256(getattr(prepared, key).tobytes()).hexdigest() for key in ('tokens', 'offsets')}
    record = {'protocol': 'MUI-1', 'scope': 'exploratory_reused_calibration_short_budget', 'seed': seed,
              'source_commit': source, 'workflow_run_id': os.environ.get('GITHUB_RUN_ID'),
              'steps': STEPS, 'tokens_per_arm': STEPS * cfg.batch_size * cfg.seq_len,
              'batch_stream_sha256': stream.hexdigest(), 'loaded_splits': sorted(arrays),
              'data_array_sha256': data_hashes, 'data_manifest_sha256': sha256_file(data_root / 'manifest.json'),
              'training_documents': len(train_docs), 'calibration_documents': len(calibration_docs),
              'self_tests': tests, 'accounting': accounting, 'histories': histories,
              'final_document_records': raw_final, 'initial_gradients': initial_gradients,
              'routing': route, 'training_seconds': training_seconds, 'inference': latency,
              'checkpoint_sha256': sha256_file(checkpoint),
              'source_file_sha256': sha256_file(Path(__file__)),
              'protocol_sha256': sha256_file(ROOT / 'docs/prereg/MODE_UTILIZATION_INTERVENTION_1.md'),
              'environment': {'python': sys.version, 'torch': torch.__version__, 'numpy': np.__version__,
                  'platform': platform.platform(), 'threads': torch.get_num_threads(), 'cpu': platform.processor()}}
    dump(OUT / f'seed-{seed}.json', record)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int)
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    if args.self_test:
        torch.set_num_threads(2)
        print(json.dumps(self_test(), indent=2))
    else:
        run(args.seed)
