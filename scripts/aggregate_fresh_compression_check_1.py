#!/usr/bin/env python3
"""FCC-1 frozen intersection gate with article-aware confidence bounds."""
from pathlib import Path
import hashlib,json,math,statistics
import numpy as np
from scipy.stats import t
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/fresh-compression-check-1'
DATA=ROOT/'data/fresh-compression-check-1'
SEEDS={'mui1':[904031,904043,904051,904073],'gate2a':[202781,212789,222793,232801]}

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def summary_stats(matrix,seed_draws,article_draws):
    x=np.asarray(matrix,dtype=np.float64);assert x.shape==(4,256) and np.isfinite(x).all()
    means=[float(row.mean()) for row in x];mean=statistics.mean(means);se=statistics.stdev(means)/2
    seed_t={'mean':mean,'per_seed':means,'ci95':[mean-float(t.ppf(.975,3))*se,mean+float(t.ppf(.975,3))*se],
            'ucb95':mean+float(t.ppf(.95,3))*se}
    boot=x[seed_draws[:,:,None],article_draws[:,None,:]].mean(axis=(1,2))
    # Independent representation of the same resamples by integer multiplicities.
    sc=np.stack([np.bincount(v,minlength=4) for v in seed_draws])
    ac=np.stack([np.bincount(v,minlength=256) for v in article_draws])
    alternative=np.einsum('bi,ij,bj->b',sc.astype(float),x,ac.astype(float),optimize=True)/1024
    error=float(np.max(np.abs(boot-alternative)));assert error<1e-10
    quantiles=np.quantile(boot,[.025,.95,.975])
    # Python fsum article means provide an additional non-numpy arithmetic path.
    independent=math.fsum(math.fsum(row.tolist())/256 for row in x)/4
    assert abs(independent-mean)<1e-12
    return {'seed_t':seed_t,'crossed_bootstrap':{'ci95':[float(quantiles[0]),float(quantiles[2])],
             'ucb95':float(quantiles[1]),'replicates':10000,'audit_max_difference':error}}

def main():
    cells=[json.loads((OUT/f'cell-{i}.json').read_text()) for i in range(4)]
    assert len({c['source_commit'] for c in cells})==1 and len({c['source_script_sha256'] for c in cells})==1
    assert len({c['data_manifest_sha256'] for c in cells})==1 and len({c['prereg_sha256'] for c in cells})==1
    manifest=json.loads((DATA/'manifest.json').read_text())
    assert sha(DATA/'manifest.json')==cells[0]['data_manifest_sha256']
    for name,h in manifest['sha256'].items():assert sha(DATA/name)==h
    assert manifest['parent_prefix_exact_reproduction'] and manifest['duplicate_window_intersection']==0
    articles=json.loads((DATA/'articles.json').read_text());article_ids=[a['article_id'] for a in articles]
    assert len(article_ids)==len(set(article_ids))==256
    assert min(article_ids)>=8192>max(manifest['last_training_true_article'],manifest['last_tokenizer_true_article'])
    arrays=np.load(DATA/'windows.npy',allow_pickle=False)
    candidate_hashes={hashlib.sha256(w.tobytes()).digest() for article in arrays for w in article}
    assert len(candidate_hashes)==1024
    collisions=0
    parent=ROOT/'data/native-compact-gate-2a'
    for split in ('train','validation'):
        p=parent/f'{split}-tokens.npy';assert sha(p)==manifest['parent_array_sha256'][p.name]
        old=np.load(p,allow_pickle=False).astype(np.int32,copy=False)
        for start in range(len(old)-64):collisions+=hashlib.sha256(old[start:start+65].tobytes()).digest() in candidate_hashes
    assert collisions==0
    groups={}
    for i,c in enumerate(cells):
        assert c['index']==i and len(c['results'])==2
        for row in c['results']:
            assert row['seed']==SEEDS[row['cohort']][i]
            assert sha(ROOT/row['source_checkpoint'])==row['checkpoint_sha256']
            assert [a['article_id'] for a in row['articles']]==article_ids
            for a,meta in zip(row['articles'],articles,strict=True):
                assert [w['start'] for w in a['windows']]==meta['starts']
                for w in a['windows']:
                    assert abs(w['loss']['rank1']-w['loss']['original']-w['delta_nll'])<1e-12
                    assert w['kl']>=-1e-12 and 0<=w['top1_agreement']<=1
                for metric in ('delta_nll','kl','top1_agreement'):
                    assert abs(math.fsum(w[metric] for w in a['windows'])/4-a['mean'][metric])<1e-12
                for name in ('original','rank1','narrow65','full'):
                    assert abs(math.fsum(w['loss'][name] for w in a['windows'])/4-a['loss'][name])<1e-12
            if i==0:assert row['independent_subset_audit']['passed'] and row['independent_subset_audit']['articles']==16
            groups.setdefault(row['cohort'],[]).append(row)
    rng=np.random.default_rng(904313)
    seed_draws=rng.integers(0,4,size=(10000,4),dtype=np.int16)
    article_draws=rng.integers(0,256,size=(10000,256),dtype=np.int16)
    np.savez_compressed(OUT/'bootstrap-draws.npz',seeds=seed_draws,articles=article_draws)
    result={'protocol':'FCC-1','source_commit':cells[0]['source_commit'],'data_manifest_sha256':sha(DATA/'manifest.json'),
      'articles':256,'windows':1024,'tokens_scored_per_model':65536,'cohorts':{},'global_decision':'NO_GO_FOR_OLMOE_OR_QWEN',
      'integrity_audit':{'passed':True,'exact_duplicate_window_intersections':collisions,'article_level_statistics':True,
        'independent_materialized_subset_reexecutions':2,'bootstrap_second_numeric_path':True},
      'scope':'Fresh article confirmation of compression fidelity for eight small checkpoints; not a model-capacity frontier or novel-architecture claim.'}
    for cohort,rows in groups.items():
        metrics={}
        for key in ('delta_nll','kl','top1_agreement'):
            matrix=[[a['mean'][key] for a in r['articles']] for r in rows]
            metrics[key]=summary_stats(matrix,seed_draws,article_draws)
        conditions={}
        for key,margin in [('delta_nll',.010),('kl',.005)]:
            for method in ('seed_t','crossed_bootstrap'):
                conditions[key+'_'+method]=metrics[key][method]['ucb95']<=margin
        conditions['every_seed_delta_le_0025']=max(metrics['delta_nll']['seed_t']['per_seed'])<=.025
        params=rows[0]['accounting'];conditions['parameters_le_075']=params['rank1']['total_parameters']/params['original']['total_parameters']<=.75
        controls={name:{'mean_loss':statistics.mean(statistics.mean(a['loss'][name] for a in r['articles']) for r in rows),
                        'per_seed_loss':[statistics.mean(a['loss'][name] for a in r['articles']) for r in rows]} for name in ('original','rank1','narrow65','full')}
        result['cohorts'][cohort]={'metrics':metrics,'conditions':conditions,'passed':all(conditions.values()),'accounting':params,'contextual_controls':controls}
    result['passed']=all(c['passed'] for c in result['cohorts'].values())
    result['verdict']='FRESH_COMPRESSION_FIDELITY_PASS' if result['passed'] else 'FRESH_COMPRESSION_FIDELITY_FAIL'
    (OUT/'summary.json').write_text(json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+'\n')
    lines=['# FCC-1 — Fresh article compression fidelity','',f"**Verdict:** `{result['verdict']}`. `NO_GO_FOR_OLMOE_OR_QWEN` unchanged.",
      '', '256 true top-level Wikipedia articles, four windows each, 65536 scored tokens/model. Selected outside the bounded training/tokenizer prefix using frozen seeds and original tokenizer. No refitting or candidate selection.',
      '', '| Cohort | Mean delta NLL | Seed t95 upper | Crossed95 upper | Mean KL | Top1 agreement |', '|---|---:|---:|---:|---:|---:|']
    for cohort,c in result['cohorts'].items():
        m=c['metrics'];n=m['delta_nll']
        lines.append(f"| {cohort} | {n['seed_t']['mean']:+.7f} | {n['seed_t']['ucb95']:+.7f} | {n['crossed_bootstrap']['ucb95']:+.7f} | {m['kl']['seed_t']['mean']:.7f} | {m['top1_agreement']['seed_t']['mean']:.4%} |")
    lines+=['','Parameters: 95552 original -> 47168 compressed (50.6363% reduction); expert matrix MACs30720 ->14592 (52.5% analytical reduction). These are not runtime or total-model compute measurements.',
      '', '## Contextual conventional controls', '', '| Cohort | Model | Fresh article mean NLL |', '|---|---|---:|']
    for cohort,c in result['cohorts'].items():
        for name,r in c['contextual_controls'].items():lines.append(f"| {cohort} | {name} | {r['mean_loss']:.7f} |")
    lines+=['','## Limits and integrity','',
      'The primary confirmation concerns rank1 vs its own compact parent. It does not imply parity with conventional-full or narrow65, neither of which was the primary hypothesis.',
      'English Wikipedia, sequence64, vocabulary512, tiny checkpoints. Neither reasoning, tool use, coding tasks, OOD capability nor a mature large-model scaling law was measured.',
      'The two training budgets use different seeds; they are not a paired learning curve. Four training seeds and article resampling have limited population coverage.',
      'The legacy segmenter split sections at every heading. Older labels naming these segments articles must not be treated as independent true articles. FCC-1 uses verified top-level boundaries and article-weighted statistics.',
      'Independent materialized/Gram-based reexecution passed on16 articles per primary seed0 cohort; all raw article arithmetic, draw-count bootstrap and exact-window exclusion checks passed. This is not an independent research group replication.',
      'Source/raw windows/manifests/individual seed outcomes and bootstrap draws are committed alongside the report. No loss-dependent selection or threshold change occurred.']
    (ROOT/'docs/results/2026-09-04-fresh-compression-check-1.md').write_text('\n'.join(lines)+'\n')
    paths=sorted(p for d in (OUT,DATA) for p in d.rglob('*') if p.is_file() and p.name!='SHA256SUMS')
    (OUT/'SHA256SUMS').write_text(''.join(sha(p)+'  '+str(p.relative_to(ROOT))+'\n' for p in paths))
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
