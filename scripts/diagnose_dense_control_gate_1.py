#!/usr/bin/env python3
"""Supplementary checkpoint/trajectory diagnostics; never changes DCG-1 gates."""
from pathlib import Path
import argparse, hashlib, json, statistics, sys
import torch


def main(root):
    out = root/'results/dense-control-gate-1'
    rows = []
    for index in range(6):
        folder = out/f'cell-{index}'
        raw = json.loads((folder/'frozen.json').read_text())
        for arm in ('native-rank1','native-rank8','dense104','dense80','dense164','rank8-to1'):
            history = [r for r in raw['history'] if r['arm']==arm]
            nll = {r['step']:r['development_nll'] for r in history}
            record = {'seed':raw['seed'],'arm':arm,'development_nll':nll,
                      'development_slope_2200_4400':(nll[4400]-nll[2200])/2200,
                      'terminal_training_nll': next(r['recent_train_nll'] for r in history if r['step']==4400)}
            if arm.startswith('native') or arm=='rank8-to1':
                state = torch.load(folder/f'{arm}.pt',map_location='cpu',weights_only=True)
                ranks, ratios = [], []
                for layer in range(2):
                    for bank in ('gate','up','down'):
                        pre=f'blocks.{layer}.moe.'
                        common=state[pre+'common_'+bank].double()
                        for expert in range(12):
                            residual=state[pre+bank+f'_left.{expert}'].double()@state[pre+bank+f'_right.{expert}'].double()
                            gram=residual.T@residual
                            spectral=float(torch.linalg.eigvalsh(gram)[-1])
                            ranks.append(float(residual.square().sum())/max(spectral,1e-30))
                            ratios.append(float(residual.norm()/common.norm()))
                record['residual_stable_rank_mean']=statistics.mean(ranks)
                record['residual_to_common_norm_mean']=statistics.mean(ratios)
            rows.append(record)
    arms={}
    for arm in sorted({r['arm'] for r in rows}):
        selected=[r for r in rows if r['arm']==arm]
        arms[arm]={'mean_development_nll':{str(step):statistics.mean(r['development_nll'][step] for r in selected) for step in (0,800,2200,4400)},
                   'mean_development_slope_2200_4400':statistics.mean(r['development_slope_2200_4400'] for r in selected)}
        if 'residual_stable_rank_mean' in selected[0]:
            arms[arm]['mean_residual_stable_rank']=statistics.mean(r['residual_stable_rank_mean'] for r in selected)
            arms[arm]['mean_residual_to_common_norm']=statistics.mean(r['residual_to_common_norm_mean'] for r in selected)
    payload={'scope':'supplementary descriptive frozen-checkpoint analysis; not confirmatory or causal',
             'no_new_training_or_holdout_access':True,
             'limits':'Two checkpoints define a coarse slope, not convergence. Stable rank is not useful capacity by itself.',
             'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'rows':rows,'arms':arms}
    (out/'supplementary-diagnostics.json').write_text(json.dumps(payload,indent=2,sort_keys=True)+'\n')
    print(json.dumps(arms,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
    torch.set_num_threads(2);main(parser.parse_args().root)
