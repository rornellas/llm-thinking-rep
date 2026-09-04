"""Separate post-run audit of MUI-1 files; no training or model imports.
Reconstruct rank metrics using a Gram eigenvalue instead of the runner's SVD.
Count unique checkpoint tensor storage to handle tied token/output embeddings.
"""
from pathlib import Path
import hashlib, json, math, statistics, sys
import torch
from scipy.stats import t

ROOT=Path(sys.argv[1] if len(sys.argv)>1 else Path(__file__).resolve().parents[1])
OUT=ROOT/'results/mode-utilization-intervention-1'
torch.set_num_threads(2)
checks=[]
for line in (OUT/'SHA256SUMS').read_text().splitlines():
    expected,rel=line.split('  ',1)
    assert hashlib.sha256((ROOT/rel).read_bytes()).hexdigest()==expected,rel
    checks.append(rel)
seeds=[904031,904043,904051,904073]
s=json.loads((OUT/'summary.json').read_text())
rows=[]
for seed in seeds:
    raw=json.loads((OUT/f'seed-{seed}.json').read_text())
    state=torch.load(OUT/f'seed-{seed}.pt',map_location='cpu',weights_only=True)
    for arm, weights in state.items():
        storage={v.untyped_storage().data_ptr():v.untyped_storage().nbytes()//v.element_size() for v in weights.values()}
        count=sum(storage.values())
        assert count==raw['accounting'][arm]['total_parameters'],(arm,count)
        records=raw['final_document_records'][arm]
        loss=statistics.mean(r['loss'] for r in records)
        assert abs(loss-raw['histories'][arm][-1]['loss'])<1e-10
        norms=[]
        if arm not in ['conventional-full','conventional-narrow65']:
            for layer in range(2):
                for bank in ['gate','up','down']:
                    for e in range(12):
                        pre=f'blocks.{layer}.moe.{bank}'
                        residual=weights[pre+f'_left.{e}'].double()@weights[pre+f'_right.{e}'].double()
                        gram=residual.T@residual if residual.shape[0]>=residual.shape[1] else residual@residual.T
                        spectral=float(torch.linalg.eigvalsh(gram)[-1])
                        norms.append(float(residual.square().sum())/spectral)
            rank=statistics.mean(norms)
            assert abs(rank-raw['histories'][arm][-1]['stable_rank'])<1e-10
        else:
            rank=None
        rows.append({'seed':seed,'arm':arm,'parameters':count,'loss':loss,'stable_rank':rank})
for arm,expected in s['arms'].items():
    vals=[r['loss'] for r in rows if r['arm']==arm]
    assert abs(statistics.mean(vals)-expected['final_loss']['mean'])<1e-10
    se=statistics.stdev(vals)/math.sqrt(len(vals))
    assert abs(statistics.mean(vals)+t.ppf(.975,3)*se-expected['final_loss']['ci95'][1])<1e-10
report={'passed':True,'verified_files':len(checks),'models':len(rows),'seeds':seeds,
        'method':'Independent post-run numeric path, not independent training replication',
        'checks':['manifest hashes','unique-storage parameter counts','raw-window loss means','Gram eigenvalue stable rank','seed t intervals'],
        'audit_script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'source_commit':s['source_commit'],'rows':rows}
p=OUT/'checkpoint-audit.json'
p.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items() if k!='rows'},indent=2))
