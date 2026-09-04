#!/usr/bin/env python3
"""Fixed ECK-1 aggregation; no tuning and no broad generalization claims."""
from pathlib import Path
import hashlib,json,math,statistics,sys
import torch
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/exact-compact-kernel-1'
SEEDS={'mui1':[904031,904043,904051,904073],'gate2a':[202781,212789,222793,232801]}
NAMES=('loop8','vector8','loop1','vector1','narrow65','full')

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    cells=[json.loads((OUT/f'cell-{i}/results.json').read_text()) for i in range(4)]
    assert len({c['source_commit'] for c in cells})==1
    assert len({json.dumps(c['source_sha256'],sort_keys=True) for c in cells})==1
    groups={};export_count=0;max_logit_error=0.0
    for i,c in enumerate(cells):
        assert c['index']==i and c['loaded_splits']==['validation'] and len(c['results'])==2
        for row in c['results']:
            cohort=row['cohort'];assert row['seed']==SEEDS[cohort][i]
            groups.setdefault(cohort,[]).append(row)
            assert set(row['total_parameters'])==set(NAMES)
            for p in row['parity'].values():
                assert len(p['records'])==150
                delta=math.fsum(r['delta_loss'] for r in p['records'])/150
                assert abs(delta-p['mean_nll_delta'])<1e-12 and abs(delta)<=2e-5
                max_logit_error=max(max_logit_error,p['max_logit_error'])
                assert p['max_logit_error']==max(r['max_logit_error'] for r in p['records'])
                for r in p['records']:assert abs((r['vector_loss']-r['loop_loss'])-r['delta_loss'])<1e-12
            for length,vals in row['timing'].items():
                assert length in ('1','64') and set(vals)==set(NAMES)
                for name,v in vals.items():
                    assert len(v['blocks_ms'])==15 and all(math.isfinite(x) and x>0 for x in v['blocks_ms'])
                    # Independent order statistic, not statistics.median used by runner.
                    med=sorted(v['blocks_ms'])[7]
                    assert abs(med-v['median_ms'])<1e-12
                    for base,label in [('loop8','ratio_to_loop8'),('narrow65','ratio_to_narrow65')]:
                        assert abs(med/sorted(vals[base]['blocks_ms'])[7]-v[label])<1e-12
            for name,e in row['exports'].items():
                p=OUT/f'cell-{i}'/e['file'];assert sha(p)==e['sha256'] and p.stat().st_size==e['bytes']
                state=torch.load(p,map_location='cpu',weights_only=True)['state_dict']
                unique={v.untyped_storage().data_ptr():v.untyped_storage().nbytes() for v in state.values()}
                assert sum(unique.values())==row['parameter_bytes'][name]==4*row['total_parameters'][name]
                export_count+=1
    result={'protocol':'ECK-1','source_commit':cells[0]['source_commit'],'checkpoints':8,'exported_models':export_count,
        'numeric_audit_passed':True,'max_logit_difference':max_logit_error,'cohorts':{},'global_decision':'NO_GO_FOR_OLMOE_OR_QWEN'}
    for cohort,rows in groups.items():
        assert len(rows)==4
        table={}
        for name in NAMES:
            table[name]={'parameters':rows[0]['total_parameters'][name],'parameter_bytes':rows[0]['parameter_bytes'][name],
                'expert_macs':rows[0]['expert_macs'][name],'timing':{}}
            for length in ('1','64'):
                rs=[r['timing'][length][name] for r in rows]
                table[name]['timing'][length]={'median_ms':statistics.median(v['median_ms'] for v in rs),
                    'ratio_to_loop8':statistics.median(v['ratio_to_loop8'] for v in rs),
                    'ratio_to_narrow65':statistics.median(v['ratio_to_narrow65'] for v in rs),
                    'individual_ratio_to_loop8':[v['ratio_to_loop8'] for v in rs],
                    'individual_ratio_to_narrow65':[v['ratio_to_narrow65'] for v in rs]}
        passed=all(table['vector1']['timing'][l][r]<=.9 for l in ('1','64') for r in ('ratio_to_loop8','ratio_to_narrow65'))
        result['cohorts'][cohort]={'candidates':table,'engineering_screen_passed':passed}
    result['engineering_screen_passed']=all(v['engineering_screen_passed'] for v in result['cohorts'].values())
    fa_path=ROOT/'results/functional-ablation-1/finalization.json'
    if fa_path.exists():
        fa=json.loads(fa_path.read_text())
        result['fa1_audited_fidelity_storage']=bool(fa['audited_development_compression_signal'] and fa['independent_reexecution_status']=='PASS')
    else:
        result['fa1_audited_fidelity_storage']=False
    result['combined_development_signal']=result['engineering_screen_passed'] and result['fa1_audited_fidelity_storage']
    result['claim_scope']='Same frozen small checkpoints, exposed calibration and CPU prefix forwards only; no training capacity or novelty claim.'
    (OUT/'summary.json').write_text(json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+'\n')
    lines=['# ECK-1 — Exact compact inference implementation','',f"**Engineering screen passed:** `{result['engineering_screen_passed']}`.",
        f"**Combined with independently audited FA-1 fidelity/storage:** `{result['combined_development_signal']}`.",
        '',f"Eight original primary checkpoints; {export_count} vectorized exports; maximum checked logit difference {max_logit_error:.9g}.",
        '', '| Cohort | Candidate | Parameters | Prefix 1 ms | Prefix 64 ms | Ratio vs narrow 1 / 64 |',
        '|---|---|---:|---:|---:|---|']
    for cohort,group in result['cohorts'].items():
        for name,r in group['candidates'].items():
            a=r['timing']['1'];b=r['timing']['64']
            lines.append(f"| {cohort} | {name} | {r['parameters']} | {a['median_ms']:.4f} | {b['median_ms']:.4f} | {a['ratio_to_narrow65']:.3f} / {b['ratio_to_narrow65']:.3f} |")
    lines+=['','Ratios are medians of paired per-checkpoint ratios, not ratios of aggregate medians. Raw blocks and every unfavorable seed are retained in cell results.',
        '', '## Claim limits','',
        'Vector8 is a function-preserving software change to rank8. Vector1 is function-preserving relative to the approximate rank1 model, not exactly equivalent to rank8. Its quality loss must be taken from FA-1.',
        'All timing candidates omit auxiliary training losses. CPU, FP32, two threads, batch1, actual four calibration prefixes, lengths1/64, no KV cache. No production decode, GPU, peak-memory, energy or long-context extrapolation.',
        'Neither SVD truncation nor vectorization is novel. Faster than narrow65 does not imply better language quality than narrow65. Old native-training FAIL results and NO_GO_FOR_OLMOE_OR_QWEN remain.',
        'The next capacity comparison must include simple dense models matched in parameters and/or active compute, not only the old loop implementation.']
    (ROOT/'docs/results/2026-09-04-exact-compact-kernel-1.md').write_text('\n'.join(lines)+'\n')
    paths=sorted(p for p in OUT.rglob('*') if p.is_file() and p.name!='SHA256SUMS')
    (OUT/'SHA256SUMS').write_text(''.join(sha(p)+'  '+str(p.relative_to(ROOT))+'\n' for p in paths))
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
