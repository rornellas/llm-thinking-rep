#!/usr/bin/env python3
"""DCG-1 frozen four-contrast gate and separate arithmetic paths."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import numpy as np
from scipy.stats import t
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'results/dense-control-gate-1'
DATA = ROOT/'data/dense-control-gate-1'
SEEDS = (906101, 906103, 906107, 906109, 906119, 906121)
ARMS = ('native-rank1', 'native-rank8', 'dense104', 'dense80', 'dense164', 'rank8-to1')


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def draws():
    rng = np.random.default_rng(906139)
    return rng.integers(0, 6, (10000, 6)), rng.integers(0, 256, (10000, 256))


def paired_stats(matrix, seed_draws, article_draws, upper=.9875):
    x = np.asarray(matrix, dtype=np.float64)
    assert x.shape == (6, 256) and np.isfinite(x).all()
    means = [math.fsum(row.tolist())/256 for row in x]
    mean = statistics.mean(means)
    se = statistics.stdev(means)/math.sqrt(6)
    alternative_se = math.sqrt(math.fsum((m-mean)**2 for m in means)/30)
    assert abs(se-alternative_se) < 1e-12
    seed_counts = np.stack([np.bincount(d, minlength=6) for d in seed_draws])
    article_counts = np.stack([np.bincount(d, minlength=256) for d in article_draws])
    boot = np.einsum('bi,ij,bj->b', seed_counts.astype(float), x, article_counts.astype(float), optimize=True)/1536
    max_error = 0.0
    for start in range(0, len(seed_draws), 200):
        direct = x[seed_draws[start:start+200, :, None], article_draws[start:start+200, None, :]].mean((1, 2))
        max_error = max(max_error, float(np.max(np.abs(direct-boot[start:start+200]))))
    assert max_error < 1e-10
    return {'mean': mean, 'per_seed': means, 'seed_ci95': [mean-float(t.ppf(.975, 5))*se, mean+float(t.ppf(.975, 5))*se],
            'seed_upper': mean+float(t.ppf(upper, 5))*se, 'crossed_upper': float(np.quantile(boot, upper)),
            'crossed_ci95': np.quantile(boot, [.025, .975]).tolist(), 'upper_probability': upper,
            'bootstrap_audit_max_difference': max_error}


def self_test():
    s, a = draws()
    zero = paired_stats(np.zeros((6, 256)), s, a)
    assert zero['seed_upper'] == zero['crossed_upper'] == 0
    fixed = paired_stats(np.full((6, 256), 2.0), s, a)
    assert fixed['seed_upper'] == fixed['crossed_upper'] == 2
    matrix = np.random.default_rng(17).normal(size=(6, 256))
    base = paired_stats(matrix, s, a)
    shifted = paired_stats(matrix+3, s, a)
    for key in ('mean', 'seed_upper', 'crossed_upper'):
        assert abs(shifted[key]-base[key]-3) < 1e-10
    json.dumps(base, allow_nan=False)
    return {'passed': True, 'tests': ['zero', 'constant', 'shift_equivariance', 'two_bootstrap_paths', 'JSON_scalars']}


def main():
    frozen = [json.loads((OUT/f'cell-{i}/frozen.json').read_text()) for i in range(6)]
    cells = [json.loads((OUT/f'cell-{i}/evaluation.json').read_text()) for i in range(6)]
    manifest = json.loads((DATA/'manifest.json').read_text())
    assert manifest['parent_prefix_exact_reproduction'] and manifest['pool_bounds'] == [16384, 24576]
    assert max(manifest['last_training_true_article'], manifest['last_tokenizer_true_article']) < 8192
    for name, h in manifest['sha256'].items():
        assert sha(DATA/name) == h
    articles = json.loads((DATA/'articles.json').read_text())
    article_ids = [a['article_id'] for a in articles]
    assert len(article_ids) == len(set(article_ids)) == 256 and min(article_ids) >= 16384
    array = np.load(DATA/'windows.npy', allow_pickle=False)
    assert array.shape == (256, 4, 65)
    selected = {hashlib.sha256(w.tobytes()).digest() for group in array for w in group}
    assert len(selected) == 1024
    collisions = 0
    for split in ('train', 'validation'):
        path = ROOT/f'data/native-compact-gate-2a/{split}-tokens.npy'
        assert sha(path) == manifest['parent_array_sha256'][path.name]
        old = np.load(path, allow_pickle=False).astype(np.int32, copy=False)
        for start in range(len(old)-64):
            collisions += hashlib.sha256(old[start:start+65].tobytes()).digest() in selected
    previous = ROOT/'data/fresh-compression-check-1/windows.npy'
    assert sha(previous) == manifest['FCC1_windows_sha256']
    for w in np.load(previous, allow_pickle=False).reshape(-1, 65):
        collisions += hashlib.sha256(w.tobytes()).digest() in selected
    assert collisions == 0
    assert len({f['source_commit'] for f in frozen} | {c['source_commit'] for c in cells}) == 1
    assert len({json.dumps(f['source_sha256'], sort_keys=True) for f in frozen}) == 1
    counts = frozen[0]['exports']
    for i, (f, c) in enumerate(zip(frozen, cells, strict=True)):
        folder = OUT/f'cell-{i}'
        assert f['seed'] == c['seed'] == SEEDS[i] and f['index'] == c['index'] == i
        assert f['steps'] == 4400 and f['tokens_per_trainable_arm'] == 2252800
        assert f['loaded_training_splits'] == ['train', 'validation'] and not f['primary_holdout_present_during_training']
        assert set(f['batch_stream_sha256']) == set(ARMS[:-1]) and len(set(f['batch_stream_sha256'].values())) == 1
        assert sha(folder/'frozen.json') == c['frozen_file_sha256']
        assert sha(DATA/'manifest.json') == c['data_manifest_sha256']
        assert manifest['protocol_sha256'] == f['source_sha256']['docs/prereg/DENSE_CONTROL_GATE_1.md']
        assert c['source_sha256'] == f['source_sha256']
        for name, h in f['source_sha256'].items():
            assert sha(ROOT/name) == h
        for name, h in f['training_checkpoints'].items():
            assert sha(folder/name) == h
        for arm, info in f['exports'].items():
            assert sha(folder/info['file']) == info['sha256'] and (folder/info['file']).stat().st_size == info['bytes']
            assert info['accounting'] == counts[arm]['accounting']
        for arm in ARMS[:-1]:
            assert len(f['training_losses'][arm]) == 4400 and all(math.isfinite(v) for v in f['training_losses'][arm])
        assert [a['article_id'] for a in c['articles']] == article_ids
        for row, meta, group in zip(c['articles'], articles, array, strict=True):
            assert [w['start'] for w in row['windows']] == meta['starts']
            assert [hashlib.sha256(w.tobytes()).hexdigest() for w in group] == meta['window_sha256']
            for arm in ARMS:
                values = [w['loss'][arm] for w in row['windows']]
                assert len(values) == 4 and all(math.isfinite(v) for v in values)
                assert abs(math.fsum(values)/4-row['loss'][arm]) < 1e-12
            for w in row['windows']:
                assert w['compression_kl'] >= -1e-12 and 0 <= w['compression_top1_agreement'] <= 1
        assert max(c['reload_max_window_errors'].values()) <= 2e-5
        for length in ('1', '64'):
            for arm in ARMS:
                r = c['inference'][length]['models'][arm]
                assert len(r['blocks_ms']) == 9 and all(v > 0 and math.isfinite(v) for v in r['blocks_ms'])
                assert statistics.median(r['blocks_ms']) == r['median_ms']
    assert counts['native-rank1']['accounting']['total_parameters'] == counts['dense104']['accounting']['total_parameters'] == 47168
    assert counts['native-rank1']['accounting']['ffn_matrix_macs_including_router'] == counts['dense80']['accounting']['ffn_matrix_macs_including_router'] == 15360
    s, a = draws()
    np.savez_compressed(OUT/'bootstrap-draws.npz', seed_draws=s, article_draws=a)
    matrices = {arm: np.asarray([[row['loss'][arm] for row in c['articles']] for c in cells]) for arm in ARMS}
    result = {'protocol': 'DCG-1', 'source_commit': cells[0]['source_commit'], 'seeds': list(SEEDS),
              'articles': 256, 'scored_tokens_per_model': 65536, 'steps': 4400, 'trainings': 30,
              'global_decision': 'NO_GO_FOR_OLMOE_OR_QWEN', 'arms': {}, 'primary_contrasts': {}, 'candidate_gates': {},
              'audit': {'arithmetic_and_integrity': True, 'exact_window_collisions': collisions,
                        'independent_subset': cells[0]['independent_subset_audit'], 'all_exports_reloaded': 36}}
    for arm in ARMS:
        per_seed = [float(row.mean()) for row in matrices[arm]]
        result['arms'][arm] = {'mean_nll': statistics.mean(per_seed), 'per_seed_nll': per_seed,
            'accounting': counts[arm]['accounting'], 'serialized_bytes': [f['exports'][arm]['bytes'] for f in frozen],
            'training_parent': 'native-rank8' if arm == 'rank8-to1' else arm,
            'mean_training_seconds': statistics.mean(f['training_seconds']['native-rank8' if arm == 'rank8-to1' else arm] for f in frozen),
            'forward_ms': {length: statistics.median(c['inference'][length]['models'][arm]['median_ms'] for c in cells) for length in ('1', '64')}}
    for candidate in ('native-rank1', 'rank8-to1'):
        conditions, runtime = {}, {}
        for control in ('dense104', 'dense80'):
            stat = paired_stats(matrices[candidate]-matrices[control], s, a)
            stat['meaningful_superiority'] = bool(stat['seed_upper'] <= -.010 and stat['crossed_upper'] <= -.010 and max(stat['per_seed']) <= .010)
            stat['descriptive_noninferiority'] = bool(stat['seed_upper'] <= .010 and stat['crossed_upper'] <= .010)
            result['primary_contrasts'][candidate+'__minus__'+control] = stat
            conditions[control] = stat['meaningful_superiority']
            for length in ('1', '64'):
                ratios = [c['inference'][length]['models'][candidate]['median_ms']/c['inference'][length]['models'][control]['median_ms'] for c in cells]
                runtime[control+'_L'+length] = {'per_seed_ratios': ratios, 'median_ratio': statistics.median(ratios)}
        capacity = all(conditions.values())
        speed = all(r['median_ratio'] <= .9 for r in runtime.values())
        result['candidate_gates'][candidate] = {'comparisons': conditions, 'capacity_signal': capacity,
            'runtime_ratios': runtime, 'runtime_gain_both_controls_and_lengths': speed, 'joint_signal': capacity and speed}
    result['secondary'] = {
        'rank8_to1_minus_dense164': paired_stats(matrices['rank8-to1']-matrices['dense164'], s, a, upper=.95),
        'compression_delta_nll': paired_stats(matrices['rank8-to1']-matrices['native-rank8'], s, a, upper=.95)}
    for metric in ('compression_kl', 'compression_top1_agreement'):
        matrix = [[statistics.mean(w[metric] for w in row['windows']) for row in c['articles']] for c in cells]
        result['secondary'][metric] = paired_stats(matrix, s, a, upper=.95)
    audit_ok = cells[0]['independent_subset_audit']['passed'] is True
    result['verdict'] = ('DCG1_AUDIT_FAIL' if not audit_ok else 'DCG1_CAPACITY_SIGNAL' if any(g['capacity_signal'] for g in result['candidate_gates'].values()) else 'DCG1_NO_CAPACITY_ADVANTAGE_DEMONSTRATED')
    (OUT/'summary.json').write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False)+'\n')
    lines = ['# DCG-1 — Matched dense controls', '', '**Verdict:** `'+result['verdict']+'`.', '',
        'Thirty new trainings, six paired seeds, 4400 fixed updates, 256 fresh true articles and 65536 scored tokens/model. All prior gates remain unchanged.', '',
        '| Arm | Fresh NLL | Total parameters | FFN MACs including router | Forward L1 ms | Forward L64 ms |',
        '|---|---:|---:|---:|---:|---:|']
    for arm, row in result['arms'].items():
        acc = row['accounting']; timing = row['forward_ms']
        lines.append(f"| {arm} | {row['mean_nll']:.6f} | {acc['total_parameters']} | {acc['ffn_matrix_macs_including_router']} | {timing['1']:.4f} | {timing['64']:.4f} |")
    lines += ['', '## Prespecified primary contrasts', '', '| Candidate minus control | Mean delta NLL | Seed upper98.75% | Crossed upper98.75% | Superiority |', '|---|---:|---:|---:|---|']
    for key, row in result['primary_contrasts'].items():
        lines.append(f"| {key} | {row['mean']:+.6f} | {row['seed_upper']:+.6f} | {row['crossed_upper']:+.6f} | {row['meaningful_superiority']} |")
    lines += ['', 'Both simultaneous upper bounds must be <= -0.010 nat; no seed may exceed +0.010. Four contrasts use Bonferroni allocation. Bootstrap does not substitute for independent replication.', '',
        '## Secondary compression fidelity', '',
        f"Mean compressed-parent NLL delta: {result['secondary']['compression_delta_nll']['mean']:+.7f}; mean KL: {result['secondary']['compression_kl']['mean']:.7f}; top1 agreement: {result['secondary']['compression_top1_agreement']['mean']:.4%}.", '',
        '## Limits', '',
        'Fixed optimizer/initialization recipe, tiny model, context64 and English Wikipedia only. No per-architecture tuning, reasoning/tool tests, OOD generalization or global optimum claim. rank8-to1 inherits larger-parent training cost. Router matrix costs are included; nonlinearities, auxiliary work, optimizer and full backward FLOPs are not.', '',
        'Timings are paired within six CPU runners, two threads, batch1, no auxiliary loss and no KV cache. They are full-prefix forwards, not production autoregressive throughput. All timing blocks and each seed are retained.', '',
        'Audits verify source/data/checkpoint hashes, train/eval separation, actual exported storage, reloads, raw-window arithmetic and two bootstrap implementations. Independent all-expert/Gram/logsumexp reexecution covers the first16 fresh articles of two prespecified models from seed0, not all examples or an independent research group.', '',
        'Fresh DCG-1 articles are now exposed. Do not use them for optimization and claim a subsequent confirmation.']
    (ROOT/'docs/results/2026-09-06-dense-control-gate-1.md').write_text('\n'.join(lines)+'\n')
    files = sorted(p for d in (OUT, DATA) for p in d.rglob('*') if p.is_file() and p.name != 'SHA256SUMS')
    (OUT/'SHA256SUMS').write_text(''.join(sha(p)+'  '+str(p.relative_to(ROOT))+'\n' for p in files))
    print(json.dumps({'verdict': result['verdict'], 'primary_contrasts': result['primary_contrasts'], 'candidate_gates': result['candidate_gates']}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--self-test', action='store_true')
    if parser.parse_args().self_test:
        print(json.dumps(self_test(), indent=2))
    else:
        main()
