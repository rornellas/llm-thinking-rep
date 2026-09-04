#!/usr/bin/env python3
"""Aggregate FA-1. Every comparison remains exploratory, calibration-conditional."""
from __future__ import annotations
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
from scipy.stats import t
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/functional-ablation-1'
SEEDS={'mui1':[904031,904043,904051,904073],'gate2a':[202781,212789,222793,232801]}
PRIMARY={'mui1':'legacy','gate2a':'native-shared-rank'}

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def stats(values):
    assert len(values)==4 and all(math.isfinite(x) for x in values)
    m=statistics.mean(values); se=statistics.stdev(values)/2
    return {'per_seed':values,'mean':m,'ci95':[m-t.ppf(.975,3)*se,m+t.ppf(.975,3)*se],
            'ucb95':m+t.ppf(.95,3)*se}

def main():
    cells=[json.loads((OUT/f'cell-{i}/results.json').read_text()) for i in range(4)]
    assert all(c['loaded_splits']==['validation'] and c['windows']==150 and len(c['results'])==9 for c in cells)
    assert len({c['source_commit'] for c in cells})==1
    assert len({json.dumps(c['source_sha256'],sort_keys=True) for c in cells})==1
    assert len({json.dumps(c['array_sha256'],sort_keys=True) for c in cells})==1
    groups={}; window_keys=None; exports=0
    for cell in cells:
        folder=OUT/f"cell-{cell['index']}"
        for model in cell['results']:
            cohort,arm,seed=model['cohort'],model['arm'],model['seed']
            assert seed==SEEDS[cohort][cell['index']]
            assert model['wrapper_max_logit_error']<=2e-5
            groups.setdefault((cohort,arm),[]).append(model)
            interventions=model['interventions']
            assert set(interventions)==({'original','mean-matrices','permute-1','permute-5','permute-7','uniform-selected'} | ({'common-only','rank1'} if 'rank1' in interventions else set()))
            baseline=interventions['original']['records']
            for kind,row in interventions.items():
                keys=[(r['document_id'],r['start']) for r in row['records']]
                if window_keys is None: window_keys=keys
                assert keys==window_keys
                for metric in ('loss','delta_loss','kl','top1_agreement'):
                    direct=math.fsum(r[metric] for r in row['records'])/len(keys)
                    assert abs(direct-row['mean'][metric])<1e-10
                for a,b in zip(row['records'],baseline,strict=True):
                    assert abs(a['loss']-b['loss']-a['delta_loss'])<1e-10
                    assert a['kl']>=-1e-12 and 0<=a['top1_agreement']<=1
                if 'export' in row:
                    p=folder/row['export']['file']
                    assert p.stat().st_size==row['export']['bytes'] and sha(p)==row['export']['sha256']
                    exports+=1
    assert len(groups)==9 and sum(len(v) for v in groups.values())==36
    summary={'protocol':'FA-1','scope':'posthoc_development_calibration_only','source_commit':cells[0]['source_commit'],
        'windows':150,'unique_articles':cells[0]['unique_articles'],'checkpoints':36,'exports':exports,
        'global_decision':'NO_GO_FOR_OLMOE_OR_QWEN','numeric_audit_passed':True,'groups':{},'primary_screens':{},
        'independent_reexecution_status':'PENDING'}
    for (cohort,arm),models in sorted(groups.items()):
        assert [m['seed'] for m in models]==SEEDS[cohort]
        key=cohort+'/'+arm; result={}
        for kind in models[0]['interventions']:
            rows=[m['interventions'][kind] for m in models]
            result[kind]={metric:stats([r['mean'][metric] for r in rows]) for metric in ('loss','delta_loss','kl','top1_agreement')}
            assert len({json.dumps(r['accounting'],sort_keys=True) for r in rows})==1
            result[kind]['accounting']=rows[0]['accounting']
            base=models[0]['interventions']['original']['accounting']
            result[kind]['parameter_ratio']=rows[0]['accounting']['total_parameters']/base['total_parameters']
            if 'timing' in rows[0]:
                result[kind]['timing']={length:{'ratios':[r['timing'][length]['ratio_to_original'] for r in rows],
                    'median_ratio':statistics.median(r['timing'][length]['ratio_to_original'] for r in rows),
                    'median_ms':statistics.median(r['timing'][length]['median_ms'] for r in rows)} for length in ('1','64')}
                result[kind]['serialized_bytes']=[r['export']['bytes'] for r in rows]
        result['permutation-average']={metric:stats([statistics.mean(m['interventions']['permute-'+str(o)]['mean'][metric] for o in (1,5,7)) for m in models]) for metric in ('loss','delta_loss','kl','top1_agreement')}
        summary['groups'][key]=result
    for kind in ('rank1','common-only','mean-matrices'):
        screens={}
        for cohort,arm in PRIMARY.items():
            r=summary['groups'][cohort+'/'+arm][kind]
            conditions={'nll_ucb_le_001':r['delta_loss']['ucb95']<=.010,'no_seed_above_0025':max(r['delta_loss']['per_seed'])<=.025,
                        'kl_ucb_le_0005':r['kl']['ucb95']<=.005,'parameters_at_most_075':r['parameter_ratio']<=.75}
            latency=all(r['timing'][length]['median_ratio']<=.9 for length in ('1','64'))
            screens[cohort]={'conditions':conditions,'fidelity_and_storage':all(conditions.values()),'latency_gain_both_lengths':latency}
        summary['primary_screens'][kind]={'cohorts':screens,
            'cross_budget_fidelity_storage':all(r['fidelity_and_storage'] for r in screens.values()),
            'cross_budget_latency':all(r['latency_gain_both_lengths'] for r in screens.values())}
    summary['screen_verdict']='DEVELOPMENT_COMPRESSION_SIGNAL_PENDING_REEXECUTION' if any(r['cross_budget_fidelity_storage'] for r in summary['primary_screens'].values()) else 'NO_CROSS_BUDGET_FIDELITY_STORAGE_PASS'
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2,sort_keys=True,allow_nan=False)+'\n')
    lines=['# FA-1 — Functional ablation of frozen small-model checkpoints','',
        '**Scope:** posthoc diagnostic, known calibration, four seeds per cohort; 800 and 2200 updates are different seed cohorts, not a paired learning curve.',
        '',f"**Screen:** `{summary['screen_verdict']}`. All historical gates remain unchanged.",
        '',f"36 checkpoints; {exports} real exports; 150 windows from {summary['unique_articles']} original article IDs.",
        '', '## Primary targets', '', '| Cohort | Intervention | NLL delta | One-sided t95 upper | KL | Parameters | Latency ratio L1 / L64 |',
        '|---|---|---:|---:|---:|---:|---|']
    for cohort,arm in PRIMARY.items():
        for kind in ('original','common-only','mean-matrices','rank1','permutation-average','uniform-selected'):
            r=summary['groups'][cohort+'/'+arm][kind]
            params=str(r['accounting']['total_parameters']) if 'accounting' in r else '—'
            timing=' / '.join(f"{r['timing'][l]['median_ratio']:.3f}" for l in ('1','64')) if 'timing' in r else '—'
            lines.append(f"| {cohort} | {kind} | {r['delta_loss']['mean']:+.6f} | {r['delta_loss']['ucb95']:+.6f} | {r['kl']['mean']:.6f} | {params} | {timing} |")
    lines+=['','## All controls: mean matrices and permuted routing','','| Cohort / arm | Original NLL | Mean-matrix delta | Permutation-average delta |','|---|---:|---:|---:|']
    for key,r in summary['groups'].items():
        lines.append(f"| {key} | {r['original']['loss']['mean']:.6f} | {r['mean-matrices']['delta_loss']['mean']:+.6f} | {r['permutation-average']['delta_loss']['mean']:+.6f} |")
    lines+=['','## Interpretation limits','',
        'Intervals describe training-seed variation conditional on exposed calibration windows, not domain or article uncertainty. All analyses are exploratory and not multiplicity-adjusted.',
        'Mean matrices are not mean nonlinear outputs. Cyclic permutations alter assignments without separately isolating per-expert load effects; upstream interventions may alter later natural routes.',
        'Runtime is CPU, two threads, batch one, whole-prefix forwards, no KV cache. Every timed model omits auxiliary training losses. Parameter count excludes discarded weights and counts the tied embedding once.',
        'SVD truncation/averaging are conventional methods. Passing this screen is not evidence of a novel architecture or a broad LLM Pareto improvement.',
        'The numeric audit verifies paired raw metrics and export hashes. Independent checkpoint reexecution remains pending and is required before elevating the signal.']
    (ROOT/'docs/results/2026-09-04-functional-ablation-1.md').write_text('\n'.join(lines)+'\n')
    paths=sorted(p for p in OUT.rglob('*') if p.is_file() and p.name!='SHA256SUMS')
    (OUT/'SHA256SUMS').write_text(''.join(f'{sha(p)}  {p.relative_to(ROOT)}\n' for p in paths))
    print(json.dumps({'screen_verdict':summary['screen_verdict'],'primary_screens':summary['primary_screens']},indent=2))

if __name__=='__main__':main()
