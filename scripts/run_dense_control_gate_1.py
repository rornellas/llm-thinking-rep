#!/usr/bin/env python3
"""DCG-1: separate training/freeze and fresh evaluation commands."""
from __future__ import annotations
import argparse
import copy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import platform
import random
import statistics
import subprocess
import sys
import time
import numpy as np
import torch
import torch.nn.functional as F
import yaml
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pre_qwen_certification.dense_control import (
    SEEDS, TRAIN_ARMS, EVAL_ARMS, MILESTONES, CONFIG, build_models, compress, counts, state_hash, self_test,
)
from pre_qwen_certification.reality_gate_data import load_prepared_arrays
from scripts.run_native_compact_gate_2a_seed_impl import _training_corpus
DATA = ROOT / 'data/dense-control-gate-1'
OUT = ROOT / 'results/dense-control-gate-1'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False)+'\n')


def provenance():
    files = ['pre_qwen_certification/dense_control.py', 'scripts/run_dense_control_gate_1.py',
             'pre_qwen_certification/heterogeneous_rank.py', 'pre_qwen_certification/functional_ablation.py',
             'pre_qwen_certification/tiny_lm.py', 'pre_qwen_certification/reality_gate_data.py',
             'pre_qwen_certification/modal.py', 'pre_qwen_certification/native_compact.py',
             'scripts/run_native_compact_gate_2a_seed_impl.py',
             'docs/prereg/DENSE_CONTROL_GATE_1.md', 'configs/native_compact_gate_2a.yaml']
    return {'source_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
            'source_sha256': {name: sha(ROOT/name) for name in files}}


def environment():
    return {'python': sys.version, 'torch': str(torch.__version__), 'numpy': np.__version__,
            'platform': platform.platform(), 'threads': torch.get_num_threads(),
            'cpu': subprocess.check_output(['lscpu'], text=True)}


@torch.inference_mode()
def losses(model, windows):
    model.eval()
    return [float(F.cross_entropy(model(w.inputs[None, :])[0].flatten(0, 1), w.targets)) for w in windows]


def train(index):
    if (DATA/'windows.npy').exists():
        raise RuntimeError('Primary holdout must not exist in the training workspace')
    folder = OUT / f'cell-{index}'
    folder.mkdir(parents=True, exist_ok=True)
    assert not (folder/'frozen.json').exists(), 'Never overwrite an existing training result'
    seed = SEEDS[index]
    parent = ROOT/'data/native-compact-gate-2a'
    arrays, manifest = load_prepared_arrays(parent, splits=('train', 'validation'))
    cfg = yaml.safe_load((ROOT/'configs/native_compact_gate_2a.yaml').read_text())
    corpus, train_docs, dev_docs = _training_corpus(arrays, manifest, cfg, cfg['scales']['small'])
    assert corpus.vocab_size == 512
    windows = corpus.fixed_windows('calibration', windows_per_document=1, seed=906127)
    models = build_models(seed)
    optimizers = {name: torch.optim.AdamW(m.parameters(), lr=CONFIG.learning_rate,
                                         weight_decay=CONFIG.weight_decay) for name, m in models.items()}
    generator = torch.Generator().manual_seed(seed + 101)
    digests = {name: hashlib.sha256() for name in TRAIN_ARMS}
    timings = {name: 0.0 for name in TRAIN_ARMS}
    training_losses = {name: [] for name in TRAIN_ARMS}
    history, snapshots = [], {}
    initial_hashes = {name: state_hash(m) for name, m in models.items()}
    for step in range(4401):
        if step:
            tokens, targets = corpus.sample_batch('train', 8, generator)
            batch_bytes = tokens.numpy().tobytes() + targets.numpy().tobytes()
            offset = step % len(TRAIN_ARMS)
            order = TRAIN_ARMS[offset:] + TRAIN_ARMS[:offset]
            for name in order:
                model = models[name]
                model.train()
                digests[name].update(batch_bytes)
                torch.manual_seed(seed*10000 + step)
                started = time.perf_counter()
                logits, auxiliary = model(tokens, auxiliary=True)
                loss = F.cross_entropy(logits.flatten(0, 1), targets.flatten())
                objective = loss + CONFIG.aux_weight * auxiliary
                optimizers[name].zero_grad(set_to_none=True)
                objective.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG.grad_clip, error_if_nonfinite=True)
                optimizers[name].step()
                timings[name] += time.perf_counter()-started
                training_losses[name].append(float(loss.detach()))
        if step not in (0,) + MILESTONES:
            continue
        evaluated = dict(models)
        evaluated['rank8-to1'] = compress(models['native-rank8'])
        for name in EVAL_ARMS:
            values = losses(evaluated[name], windows)
            row = {'step': step, 'arm': name, 'development_nll': statistics.mean(values),
                   'window_losses': values,
                   'recent_train_nll': statistics.mean(training_losses[name][-200:]) if step and name in TRAIN_ARMS else None}
            history.append(row)
            print(json.dumps({k: v for k, v in row.items() if k != 'window_losses'}), flush=True)
        if step:
            path = folder/f'training-step-{step}.pt'
            torch.save({'models': {name: m.state_dict() for name, m in models.items()},
                        'optimizers': {name: opt.state_dict() for name, opt in optimizers.items()},
                        'sampler_state': generator.get_state(), 'step': step, 'seed': seed}, path)
            snapshots[path.name] = sha(path)
    exports = {}
    for name, model in evaluated.items():
        path = folder/f'{name}.pt'
        torch.save(model.state_dict(), path)
        exports[name] = {'file': path.name, 'sha256': sha(path), 'bytes': path.stat().st_size,
                         'state_sha256': state_hash(model), 'accounting': counts(model)}
    hashes = {name: value.hexdigest() for name, value in digests.items()}
    assert len(set(hashes.values())) == 1
    payload = {'protocol': 'DCG-1', 'index': index, 'seed': seed, **provenance(),
        'configuration': asdict(CONFIG), 'loaded_training_splits': list(arrays),
        'primary_holdout_present_during_training': False, 'parent_manifest_sha256': sha(parent/'manifest.json'),
        'training_array_sha256': {p.name: sha(p) for p in parent.glob('train-*.npy')},
        'steps': 4400, 'tokens_per_trainable_arm': 2252800, 'batch_stream_sha256': hashes,
        'initial_state_sha256': initial_hashes, 'training_seconds': timings, 'training_losses': training_losses,
        'development_windows': [{'document_id': w.document_id, 'start': w.start} for w in windows],
        'training_segments': len(train_docs), 'development_segments': len(dev_docs),
        'history': history, 'training_checkpoints': snapshots, 'exports': exports, 'environment': environment()}
    dump(folder/'frozen.json', payload)
    print('FROZEN', sha(folder/'frozen.json'), flush=True)


@torch.inference_mode()
def benchmark(models):
    rng = random.Random(906149)
    result = {}
    for length in (1, 64):
        tokens = torch.arange(length).remainder(512)[None, :]
        for model in models.values():
            model.eval()
            for _ in range(5):
                model(tokens)
        samples = {name: [] for name in EVAL_ARMS}
        orders = []
        for _ in range(9):
            order = list(EVAL_ARMS)
            rng.shuffle(order); orders.append(order)
            for name in order:
                start = time.perf_counter_ns()
                for _ in range(10):
                    models[name](tokens)
                samples[name].append((time.perf_counter_ns()-start)/1e6/10)
        result[str(length)] = {'orders': orders, 'models': {
            name: {'blocks_ms': values, 'median_ms': statistics.median(values)} for name, values in samples.items()}}
    return result


@torch.inference_mode()
def evaluate(index):
    folder = OUT/f'cell-{index}'
    frozen = json.loads((folder/'frozen.json').read_text())
    frozen_hash = sha(folder/'frozen.json')
    assert frozen['source_commit'] == provenance()['source_commit']
    for name, h in frozen['source_sha256'].items():
        assert sha(ROOT/name) == h
    for name, h in frozen['training_checkpoints'].items():
        assert sha(folder/name) == h
    manifest = json.loads((DATA/'manifest.json').read_text())
    assert manifest['protocol'] == 'DCG-1' and manifest['parent_prefix_exact_reproduction']
    assert manifest['pool_bounds'] == [16384, 24576] and manifest['exact_duplicate_window_intersections'] == 0
    for name, h in manifest['sha256'].items():
        assert sha(DATA/name) == h
    array = np.load(DATA/'windows.npy', allow_pickle=False)
    assert array.shape == (256, 4, 65)
    articles = json.loads((DATA/'articles.json').read_text())
    models = build_models(SEEDS[index])
    models['rank8-to1'] = compress(models['native-rank8'])
    for name, model in models.items():
        info = frozen['exports'][name]
        assert sha(folder/info['file']) == info['sha256']
        state = torch.load(folder/info['file'], map_location='cpu', weights_only=True)
        stores = {v.untyped_storage().data_ptr(): v.untyped_storage().nbytes() for v in state.values()}
        assert sum(stores.values()) == info['accounting']['parameter_bytes']
        model.load_state_dict(state)
        assert state_hash(model) == info['state_sha256'] and counts(model) == info['accounting']
        model.eval()
    records = []
    for group, article in zip(array, articles, strict=True):
        assert [hashlib.sha256(w.tobytes()).hexdigest() for w in group] == article['window_sha256']
        rows = []
        for ids, start in zip(group, article['starts'], strict=True):
            values = torch.tensor(ids, dtype=torch.long)
            logits = {name: model(values[:-1][None, :])[0] for name, model in models.items()}
            nll = {name: float(F.cross_entropy(z.flatten(0, 1), values[1:])) for name, z in logits.items()}
            lp = F.log_softmax(logits['native-rank8'].double(), -1)
            lq = F.log_softmax(logits['rank8-to1'].double(), -1)
            kl = float((lp.exp()*(lp-lq)).sum(-1).mean())
            agreement = float((logits['native-rank8'].argmax(-1) == logits['rank8-to1'].argmax(-1)).double().mean())
            rows.append({'start': start, 'loss': nll, 'compression_kl': kl, 'compression_top1_agreement': agreement})
        records.append({'article_id': article['article_id'], 'windows': rows,
                        'loss': {name: statistics.mean(w['loss'][name] for w in rows) for name in EVAL_ARMS}})
    timings = benchmark(models)
    reload_error = {}
    for name, template in models.items():
        replica = copy.deepcopy(template)
        replica.load_state_dict(torch.load(folder/f'{name}.pt', map_location='cpu', weights_only=True))
        errors = []
        for ids, expected in zip(array[0], records[0]['windows'], strict=True):
            values = torch.tensor(ids, dtype=torch.long)
            nll = float(F.cross_entropy(replica(values[:-1][None, :])[0].flatten(0, 1), values[1:]))
            errors.append(abs(nll-expected['loss'][name]))
        reload_error[name] = max(errors)
    assert max(reload_error.values()) <= 2e-5
    independent = {'required': index == 0, 'passed': None}
    if index == 0:
        from scripts.audit_functional_ablation_1 import independent_model
        alternatives = {
            'native-rank1': independent_model(models['native-rank1'].state_dict(), 'original', 512),
            'rank8-to1': independent_model(models['native-rank8'].state_dict(), 'rank1', 512),
        }
        errors = {name: 0.0 for name in alternatives}
        for group, previous in zip(array[:16], records[:16], strict=True):
            for ids, expected in zip(group, previous['windows'], strict=True):
                values = torch.tensor(ids, dtype=torch.long)
                for name, model in alternatives.items():
                    z = model(values[:-1][None, :])[0]
                    z = z.flatten(0, 1).double()
                    nll = float((torch.logsumexp(z, -1) - z.gather(1, values[1:, None]).flatten()).mean())
                    errors[name] = max(errors[name], abs(nll-expected['loss'][name]))
        independent = {'required': True, 'passed': max(errors.values()) <= 2e-5,
                       'articles': 16, 'windows_per_model': 64, 'max_window_loss_error': errors,
                       'method': 'all-expert materialization; Gram rank1; float64 logsumexp NLL'}
    result = {'protocol': 'DCG-1', 'index': index, 'seed': SEEDS[index], **provenance(),
              'frozen_file_sha256': frozen_hash, 'data_manifest_sha256': sha(DATA/'manifest.json'),
              'articles': records, 'inference': timings, 'reload_max_window_errors': reload_error,
              'independent_subset_audit': independent, 'environment': environment()}
    dump(folder/'evaluation.json', result)
    assert sha(folder/'frozen.json') == frozen_hash
    print(json.dumps({'seed': SEEDS[index], 'mean_nll': {
        name: statistics.mean(r['loss'][name] for r in records) for name in EVAL_ARMS},
        'independent_subset_audit': independent}), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=('preflight', 'train', 'evaluate'))
    parser.add_argument('--index', type=int, choices=range(6))
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    if args.mode == 'preflight':
        print(json.dumps(self_test(), indent=2)); return
    if args.index is None:
        parser.error('--index is required for train/evaluate')
    (train if args.mode == 'train' else evaluate)(args.index)


if __name__ == '__main__':
    main()
