#!/usr/bin/env python3
"""FA-1 audit: serialized storage, arithmetic, and independent materialized forwards.

Does not use the functional_ablation transformations or inference wrapper.
Rank-one residuals are obtained from Gram eigenvectors, not the runner's SVD.
"""
from __future__ import annotations
import hashlib
import json
import math
from pathlib import Path
import platform
import statistics
import sys
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import yaml
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from pre_qwen_certification.modal import Routing
from pre_qwen_certification.tiny_lm import TinyMoELanguageModel, TinyLMConfig
from pre_qwen_certification.reality_gate_data import load_prepared_arrays, documents_from_prepared_split, ArrayTokenCorpus
OUT=ROOT/'results/functional-ablation-1'

class MaterializedExperts(nn.Module):
    def __init__(self,state,layer,kind):
        super().__init__()
        pre=f'blocks.{layer}.moe.'
        router=state[pre+'router.weight']
        self.register_buffer('router',router.clone())
        self.kind=kind
        for bank in ('gate','up','down'):
            common=state[pre+'common_'+bank]
            weights=[]
            for i in range(12):
                l=state[pre+bank+f'_left.{i}'];r=state[pre+bank+f'_right.{i}']
                if kind=='rank1':
                    residual=l.double()@r.double()
                    _,vectors=torch.linalg.eigh(residual.T@residual)
                    v=vectors[:,-1:]
                    correction=(residual@v)@v.T
                    weight=common+correction.float()
                else:
                    weight=common+l@r
                weights.append(weight)
            weights=torch.stack(weights)
            if kind=='common-only':
                weights=common[None,:,:].expand_as(weights).clone()
            if kind=='mean-matrices':
                weights=weights.double().mean(0).float()[None,:,:].expand_as(weights).clone()
            self.register_buffer(bank,weights)

    def forward(self,x):
        logits=F.linear(x,self.router)
        values,ids=torch.topk(logits,4,dim=-1)
        weights=values.softmax(-1)
        if self.kind.startswith('permute-'):
            ids=(ids+int(self.kind.split('-')[1]))%12
        if self.kind=='uniform-selected':
            weights=torch.full_like(weights,.25)
        # Evaluate ALL expert functions instead of gathering selected matrices or factors.
        all_outputs=[]
        for i in range(12):
            hidden=F.silu(F.linear(x,self.gate[i]))*F.linear(x,self.up[i])
            all_outputs.append(F.linear(hidden,self.down[i]))
        output=x.new_zeros(x.shape)
        for i,y in enumerate(all_outputs):
            mass=((ids==i).to(x.dtype)*weights).sum(-1,keepdim=True)
            output+=mass*y
        return output,Routing(logits,ids,weights)


def independent_model(state,kind,vocab):
    cfg=TinyLMConfig(n_experts=12)
    m=TinyMoELanguageModel(vocab,cfg)
    missing,extra=m.load_state_dict({k:v for k,v in state.items() if '.moe.' not in k},strict=False)
    assert not extra and all('.moe.' in k for k in missing)
    for layer,block in enumerate(m.blocks):
        block.moe=MaterializedExperts(state,layer,kind)
    return m.eval()


@torch.inference_mode()
def main():
    torch.set_num_threads(2)
    verified=[]
    for line in (OUT/'SHA256SUMS').read_text().splitlines():
        expected,path=line.split('  ',1)
        assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==expected,path
        verified.append(path)
    all_results=[];export_checks=0;max_arithmetic=0.0
    for file in sorted(OUT.glob('cell-*/results.json')):
        raw=json.loads(file.read_text())
        for model in raw['results']:
            for kind,r in model['interventions'].items():
                for metric in ('loss','delta_loss','kl','top1_agreement'):
                    error=abs(statistics.mean(w[metric] for w in r['records'])-r['mean'][metric])
                    max_arithmetic=max(error,max_arithmetic)
                    assert error<1e-10
                if 'export' in r:
                    state=torch.load(file.parent/r['export']['file'],map_location='cpu',weights_only=True)['state_dict']
                    stores={v.untyped_storage().data_ptr():v.untyped_storage().nbytes() for v in state.values()}
                    assert sum(stores.values())==r['accounting']['parameter_bytes']
                    assert all(v.dtype==torch.float32 for v in state.values())
                    assert sum(stores.values())//4==r['accounting']['total_parameters']
                    export_checks+=1
            all_results.append(model)
    data,manifest=load_prepared_arrays(ROOT/'data/native-compact-gate-2a',splits=('validation',))
    docs=documents_from_prepared_split(data['validation'],prefix='native-calibration',domain='wikitext103-validation',
        maximum_document_tokens=2048,minimum_document_tokens=66,maximum_total_tokens=120000)
    corpus=ArrayTokenCorpus({'calibration':docs},seq_len=64,vocab_size=manifest['vocab_size'])
    windows=corpus.fixed_windows('calibration',windows_per_document=1,seed=904091)
    evaluations=[]
    for cohort,seed,arm in [('mui1',904031,'legacy'),('gate2a',202781,'native-shared-rank')]:
        expected=next(x for x in all_results if (x['cohort'],x['seed'],x['arm'])==(cohort,seed,arm))
        if cohort=='mui1':
            path=ROOT/f'results/mode-utilization-intervention-1/seed-{seed}.pt'
            state=torch.load(path,map_location='cpu',weights_only=True)[arm]
        else:
            path=ROOT/f'results/native-compact-gate-2a/small/frozen-candidates-seed-{seed}.pt'
            state=torch.load(path,map_location='cpu',weights_only=True)['final_states'][arm]
        baseline=independent_model(state,'original',manifest['vocab_size'])
        ref=[baseline(w.inputs[None,:])[0] for w in windows]
        for kind,previous in expected['interventions'].items():
            model=independent_model(state,kind,manifest['vocab_size'])
            losses=[];kls=[];max_loss=0.0;max_kl=0.0
            for w,z0,r in zip(windows,ref,previous['records'],strict=True):
                assert (w.document_id,w.start)==(r['document_id'],r['start'])
                z=model(w.inputs[None,:])[0]
                loss=float(F.cross_entropy(z.flatten(0,1),w.targets))
                lp=F.log_softmax(z0.double(),-1);lq=F.log_softmax(z.double(),-1)
                kl=float((lp.exp()*(lp-lq)).sum(-1).mean())
                max_loss=max(max_loss,abs(loss-r['loss']))
                max_kl=max(max_kl,abs(kl-r['kl']))
                losses.append(loss);kls.append(kl)
            assert max_loss<=2e-5,(cohort,kind,max_loss)
            assert max_kl<=2e-6,(cohort,kind,max_kl)
            row={'cohort':cohort,'seed':seed,'arm':arm,'intervention':kind,'loss':statistics.mean(losses),
                 'kl':statistics.mean(kls),'max_window_loss_difference':max_loss,'max_window_kl_difference':max_kl}
            evaluations.append(row)
            print(json.dumps(row),flush=True)
    report={'passed':True,'method':'independent materialized all-expert forwards plus Gram eigenvector rank-one reconstruction',
        'not_an_independent_training_replication':True,'verified_files':len(verified),'verified_exports':export_checks,
        'max_arithmetic_difference':max_arithmetic,'primary_checkpoints_reexecuted':2,'interventions_reexecuted':len(evaluations),
        'environment':{'python':sys.version,'torch':torch.__version__,'numpy':np.__version__,'platform':platform.platform()},
        'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'rows':evaluations}
    (OUT/'independent-audit.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
    print('AUDIT PASS',export_checks,len(evaluations))

if __name__=='__main__':main()
