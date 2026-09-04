#!/usr/bin/env python3
"""Aggregate MUI-1 development results without changing any historical gate."""
from __future__ import annotations
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
import numpy as np
from scipy.stats import t

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/mode-utilization-intervention-1'
SEEDS = (904031, 904043, 904051, 904073)
ARMS = ('conventional-full', 'conventional-narrow65', 'legacy', 'spectral-tiny', 'energy-gaussian', 'energy-spectral')
NARROW = ARMS[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def interval(values):
    x = np.asarray(values, dtype=float)
    mean = float(x.mean())
    se = float(x.std(ddof=1) / math.sqrt(len(x)))
    q = float(t.ppf(0.975, len(x)-1))
    return {'per_seed': x.tolist(), 'mean': mean, 'ci95': [mean-q*se, mean+q*se],
            'ucb95_one_sided': mean+float(t.ppf(0.95, len(x)-1))*se}


def main():
    cells = [json.loads((OUT / f'seed-{s}.json').read_text()) for s in SEEDS]
    failures = []
    source = {cell['source_commit'] for cell in cells}
    assert len(source) == 1, 'source commits differ'
    assert len({cell['protocol_sha256'] for cell in cells}) == 1
    assert len({cell['source_file_sha256'] for cell in cells}) == 1
    assert len({json.dumps(cell['data_array_sha256'], sort_keys=True) for cell in cells}) == 1
    expected_keys = None
    for seed, cell in zip(SEEDS, cells, strict=True):
        assert cell['seed'] == seed and cell['steps'] == 800 and cell['tokens_per_arm'] == 409600
        assert cell['loaded_splits'] == ['train', 'validation']
        assert cell['self_tests']['passed']
        assert sha(OUT / f'seed-{seed}.pt') == cell['checkpoint_sha256']
        assert set(cell['histories']) == set(ARMS)
        compact_counts = {cell['accounting'][arm]['total_parameters'] for arm in ARMS[2:]}
        compact_macs = {cell['accounting'][arm]['expert_macs_per_token'] for arm in ARMS[2:]}
        assert len(compact_counts) == len(compact_macs) == 1
        for arm in ARMS:
            history = cell['histories'][arm]
            assert [row['step'] for row in history] == [0, 200, 400, 800]
            records = cell['final_document_records'][arm]
            keys = [(r['document_id'], r['start']) for r in records]
            if expected_keys is None:
                expected_keys = keys
            assert keys == expected_keys, 'paired evaluation windows differ'
            # Separate arithmetic path: Python fsum instead of evaluator numpy.mean.
            direct = math.fsum(r['loss'] for r in records) / len(records)
            assert abs(direct-history[-1]['loss']) < 1e-10
            assert all(math.isfinite(row['loss']) for row in history)
    final = {arm: [cell['histories'][arm][-1] for cell in cells] for arm in ARMS}
    contrasts = {}
    for a, b in [('energy-spectral','legacy'), ('energy-gaussian','legacy'),
                 ('energy-spectral','energy-gaussian'), ('spectral-tiny','legacy'),
                 ('energy-spectral',NARROW), ('energy-gaussian',NARROW), ('legacy',NARROW)]:
        contrasts[a+'__minus__'+b] = interval([x['loss']-y['loss'] for x,y in zip(final[a], final[b], strict=True)])
    contrasts['factorial_interaction'] = interval([
        cell['histories']['energy-spectral'][-1]['loss'] - cell['histories']['energy-gaussian'][-1]['loss']
        - cell['histories']['spectral-tiny'][-1]['loss'] + cell['histories']['legacy'][-1]['loss']
        for cell in cells])
    # df=3 constants supply a second statistical implementation independent of scipy.
    for key, stat in contrasts.items():
        vals = stat['per_seed']
        mean = statistics.mean(vals)
        se = statistics.stdev(vals)/2
        assert abs(mean-stat['mean']) < 1e-12
        assert abs(mean+3.182446305284263*se-stat['ci95'][1]) < 1e-10, key
        assert abs(mean+2.353363434801827*se-stat['ucb95_one_sided']) < 1e-10, key
    gates = {}
    for arm in ('energy-gaussian','energy-spectral'):
        vs_legacy = contrasts[arm+'__minus__legacy']
        vs_narrow = contrasts[arm+'__minus__'+NARROW]
        rank_ratio = float(np.mean([r['stable_rank'] for r in final[arm]]) / np.mean([r['stable_rank'] for r in final['legacy']]))
        conditions = {'improves_legacy_by_001': vs_legacy['mean'] <= -0.010,
                      'no_seed_worsens_legacy': max(vs_legacy['per_seed']) <= 0,
                      'rank_at_least_doubles': rank_ratio >= 2,
                      'narrow_screen_ucb_le_001': vs_narrow['ucb95_one_sided'] <= 0.010}
        gates[arm] = {'conditions': conditions, 'rank_ratio': rank_ratio, 'promising': all(conditions.values())}
    summary = {'protocol':'MUI-1', 'status':'completed', 'scope':'exploratory_reused_calibration_short_budget',
        'seeds':list(SEEDS), 'source_commit':next(iter(source)), 'steps':800,
        'global_decision':'NO_GO_FOR_OLMOE_OR_QWEN',
        'screen_verdict':'PROMISING_DEVELOPMENT_SIGNAL' if any(v['promising'] for v in gates.values()) else 'NO_PROMISING_CANDIDATE_UNDER_FROZEN_SCREEN',
        'contrasts':contrasts, 'screen_gates':gates, 'arithmetic_audit':{'passed':True, 'paired_windows':True, 'checkpoint_hashes':True,
        'note':'Separate numerical paths, not an independent research replication.'}, 'arms':{}}
    for arm in ARMS:
        summary['arms'][arm] = {'final_loss':interval([r['loss'] for r in final[arm]]),
            'initial_stable_rank_mean':float(np.mean([c['histories'][arm][0]['stable_rank'] for c in cells])) if arm in ARMS[2:] else None,
            'final_stable_rank_mean':float(np.mean([r['stable_rank'] for r in final[arm]])) if arm in ARMS[2:] else None,
            'final_effective_cosine_mean':float(np.mean([r['effective_cosine'] for r in final[arm]])),
            'final_output_diversity_mean':float(np.mean([r['output_diversity'] for r in final[arm]])),
            'initial_output_diversity_mean':float(np.mean([c['histories'][arm][0]['output_diversity'] for c in cells])),
            'accounting':cells[0]['accounting'][arm],
            'mean_training_seconds':float(np.mean([c['training_seconds'][arm] for c in cells])),
            'max_dead_experts':max(info['dead_experts'] for c in cells for info in c['routing'][arm].values()),
            'median_forward_ms_by_length':{length:float(np.median([c['inference'][length][arm]['median_ms'] for c in cells])) for length in ('1','64')}}
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2,sort_keys=True,allow_nan=False)+'\n')
    lines = ['# MUI-1 — Initialization intervention, 2026-09-04', '',
        '**Scope:** prospective exploratory screen; reused calibration, four new seeds, 800 updates. Not a fresh holdout test or a repetition of Gate 2A.', '',
        '**Verdict:** `'+summary['screen_verdict']+'`. `NO_GO_FOR_OLMOE_OR_QWEN` unchanged.', '',
        '| Arm | Calibration loss | Residual stable rank | Expert params/full | Expert MACs/full |',
        '|---|---:|---:|---:|---:|']
    for arm, row in summary['arms'].items():
        rank = '—' if row['final_stable_rank_mean'] is None else f"{row['final_stable_rank_mean']:.3f}"
        acc = row['accounting']
        lines.append(f"| {arm} | {row['final_loss']['mean']:.6f} | {rank} | {acc['expert_parameter_ratio']:.2%} | {acc['expert_compute_ratio']:.2%} |")
    lines += ['', '## Frozen contrasts', '', '| Contrast | Mean loss difference | Two-sided seed t95 interval |', '|---|---:|---:|']
    for key, row in contrasts.items():
        lines.append(f"| {key} | {row['mean']:+.6f} | [{row['ci95'][0]:+.6f}, {row['ci95'][1]:+.6f}] |")
    lines += ['', 'Intervals quantify seed variation conditional on these windows. All contrasts are exploratory and not multiplicity-adjusted.', '',
        '## Measured runtime (descriptive)', '', '| Arm | Length 1, ms | Length 64, ms |', '|---|---:|---:|']
    for arm, row in summary['arms'].items():
        latency = row['median_forward_ms_by_length']
        lines.append(f"| {arm} | {latency['1']:.3f} | {latency['64']:.3f} |")
    lines += ['', 'Two CPU threads; batch 1; full forwards without KV cache. Medians of randomized repeated blocks, then median across runners. No GPU, cached-decode, energy, or production speed claim.', '',
        '## Interpretation constraints', '',
        '- Better residual stable rank alone is not better language modeling.',
        '- A win over legacy alone is not a win over narrow65 or a new Pareto frontier.',
        '- Energy intervention also changes common/expert energy allocation; spectral intervention also changes factor gauge.',
        '- Synthetic output diversity is a mechanism probe, not proof of useful language specialization.',
        '- Fixed short training may favor one learning trajectory. No mature-model or broad generalization claim.',
        '- Lower matrix-MAC counts are not measured latency. The conventional kernel and compact kernel have different implementation efficiency.',
        '- Arithmetic checks passed through separate implementations; not an independent scientific replication.', '',
        'Full raw windows, histories, initial gradients, routing, timings, environment, data/source hashes and checkpoints are retained beside summary.json. No automatic follow-up run.']
    report = ROOT/'docs/results/2026-09-04-mode-utilization-intervention-1.md'
    report.write_text('\n'.join(lines)+'\n')
    paths = sorted(p for p in OUT.rglob('*') if p.is_file() and p.name != 'SHA256SUMS')
    (OUT/'SHA256SUMS').write_text(''.join(f'{sha(p)}  {p.relative_to(ROOT)}\n' for p in paths))
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
