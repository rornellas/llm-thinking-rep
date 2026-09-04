#!/usr/bin/env python3
"""Recompute an FCC-1 artifact ZIP without importing the experiment code.

Usage: python scripts/audit_fcc1_archive.py /path/to/fcc1-complete.zip
The explicit Python resample summation is slower but independent of numpy mean
and multiplicity-einsum implementations used by the primary aggregator.
"""
from pathlib import Path
import argparse,hashlib,io,json,math,statistics,zipfile
import numpy as np

def main(path):
    z=zipfile.ZipFile(path);base='results/fresh-compression-check-1/'
    summary=json.loads(z.read(base+'summary.json'))
    verified=0
    for line in z.read(base+'SHA256SUMS').decode().splitlines():
        expected,name=line.split('  ',1)
        assert hashlib.sha256(z.read(name)).hexdigest()==expected,name
        verified+=1
    cells=[json.loads(z.read(base+f'cell-{i}.json')) for i in range(4)]
    draws=np.load(io.BytesIO(z.read(base+'bootstrap-draws.npz')))
    checks=[]
    for cohort in ('mui1','gate2a'):
        rows=[next(r for r in cell['results'] if r['cohort']==cohort) for cell in cells]
        for metric in ('delta_nll','kl','top1_agreement'):
            x=np.asarray([[math.fsum(w[metric] for w in article['windows'])/4 for article in row['articles']] for row in rows])
            means=[math.fsum(row)/256 for row in x]
            mean=statistics.mean(means);se=statistics.stdev(means)/2
            ucb=mean+2.353363434801827*se
            expected=summary['cohorts'][cohort]['metrics'][metric]
            assert abs(mean-expected['seed_t']['mean'])<1e-12
            assert abs(ucb-expected['seed_t']['ucb95'])<1e-12
            bootstrap=[]
            for seeds,articles in zip(draws['seeds'],draws['articles'],strict=True):
                bootstrap.append(math.fsum(float(x[i,j]) for i in seeds for j in articles)/1024)
            bootstrap.sort()
            position=.95*(len(bootstrap)-1);lower=int(position);fraction=position-lower
            upper=bootstrap[lower]*(1-fraction)+bootstrap[lower+1]*fraction
            error=abs(upper-expected['crossed_bootstrap']['ucb95'])
            assert error<1e-12
            checks.append({'cohort':cohort,'metric':metric,'mean':mean,'seed_t_ucb95':ucb,
                           'crossed_ucb95':upper,'bootstrap_error':error})
    report={'passed':True,'archive_sha256':hashlib.sha256(Path(path).read_bytes()).hexdigest(),
        'verified_manifest_files':verified,'third_arithmetic_path_checks':checks,
        'verdict':summary['verdict'],
        'scope':'Separate archive arithmetic verification, not new training or an independent research group.'}
    print(json.dumps(report,indent=2,sort_keys=True))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('archive',type=Path);main(parser.parse_args().archive)
