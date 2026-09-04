#!/usr/bin/env python3
"""FCC-1 fixed candidate, fresh true-article holdout, no training or tuning."""
from pathlib import Path
import argparse,hashlib,json,platform,statistics,subprocess,sys
import numpy as np
import torch
import torch.nn.functional as F
import yaml
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.run_functional_ablation_1 import load_models,SEEDS
from scripts.run_native_compact_gate_2a_seed_impl import _tiny_config
from pre_qwen_certification.functional_ablation import InferenceLM,accounting
from scripts.audit_functional_ablation_1 import independent_model
DATA=ROOT/'data/fresh-compression-check-1'
OUT=ROOT/'results/fresh-compression-check-1'

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

@torch.inference_mode()
def run(index):
    torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    manifest=json.loads((DATA/'manifest.json').read_text())
    assert manifest['articles']==256 and manifest['parent_prefix_exact_reproduction']
    assert max(manifest['last_training_true_article'],manifest['last_tokenizer_true_article'])<8192
    for name,h in manifest['sha256'].items():assert sha(DATA/name)==h
    arrays=np.load(DATA/'windows.npy',allow_pickle=False);assert arrays.shape==(256,4,65)
    articles=json.loads((DATA/'articles.json').read_text())
    for a,row in zip(arrays,articles,strict=True):
        assert [hashlib.sha256(w.tobytes()).hexdigest() for w in a]==row['window_sha256']
    cfg=_tiny_config(yaml.safe_load((ROOT/'configs/native_compact_gate_2a.yaml').read_text())['scales']['small']['model'])
    results=[]
    for cohort in ('mui1','gate2a'):
        seed=SEEDS[cohort][index];loaded,path=load_models(cohort,seed,cfg,512)
        label='legacy' if cohort=='mui1' else 'native-shared-rank';source=loaded[label]
        models={'original':InferenceLM(source,'original'),'rank1':InferenceLM(source,'rank1'),
                'narrow65':InferenceLM(loaded['conventional-narrow65'],'original'),
                'full':InferenceLM(loaded['conventional-full'],'original')}
        metrics=[]
        for group,article in zip(arrays,articles,strict=True):
            windows=[]
            for ids,start in zip(group,article['starts'],strict=True):
                ids=torch.tensor(ids,dtype=torch.long);tokens=ids[:-1][None,:];targets=ids[1:]
                logits={name:m(tokens) for name,m in models.items()}
                loss={name:float(F.cross_entropy(z.flatten(0,1),targets)) for name,z in logits.items()}
                lp=F.log_softmax(logits['original'].double(),-1);lq=F.log_softmax(logits['rank1'].double(),-1)
                kl=float((lp.exp()*(lp-lq)).sum(-1).mean())
                agreement=float((logits['original'].argmax(-1)==logits['rank1'].argmax(-1)).double().mean())
                windows.append({'start':start,'loss':loss,'delta_nll':loss['rank1']-loss['original'],'kl':kl,'top1_agreement':agreement})
            metrics.append({'article_id':article['article_id'],'windows':windows,
                'mean':{key:statistics.mean(w[key] for w in windows) for key in ('delta_nll','kl','top1_agreement')},
                'loss':{name:statistics.mean(w['loss'][name] for w in windows) for name in models}})
        audit={'required':index==0,'passed':None}
        if index==0:
            # Independent source-state materialization and Gram-eigenvector truncation.
            state=source.state_dict();a=independent_model(state,'original',512);b=independent_model(state,'rank1',512)
            max_loss=max_kl=0.0
            for group,previous in zip(arrays[:16],metrics[:16],strict=True):
                for ids,expected in zip(group,previous['windows'],strict=True):
                    ids=torch.tensor(ids,dtype=torch.long);z0=a(ids[:-1][None,:])[0];z1=b(ids[:-1][None,:])[0]
                    l0=float(F.cross_entropy(z0.flatten(0,1),ids[1:]));l1=float(F.cross_entropy(z1.flatten(0,1),ids[1:]))
                    lp=F.log_softmax(z0.double(),-1);lq=F.log_softmax(z1.double(),-1)
                    kl=float((lp.exp()*(lp-lq)).sum(-1).mean())
                    max_loss=max(max_loss,abs(l0-expected['loss']['original']),abs(l1-expected['loss']['rank1']))
                    max_kl=max(max_kl,abs(kl-expected['kl']))
            assert max_loss<=2e-5 and max_kl<=2e-6,(cohort,max_loss,max_kl)
            audit={'required':True,'passed':True,'articles':16,'windows':64,'max_window_loss_difference':max_loss,'max_window_kl_difference':max_kl}
        counts={name:accounting(m) for name,m in models.items()}
        assert counts['rank1']['total_parameters']==47168 and counts['original']['total_parameters']==95552
        row={'cohort':cohort,'seed':seed,'source_checkpoint':str(path.relative_to(ROOT)),'checkpoint_sha256':sha(path),
             'accounting':counts,'articles':metrics,'independent_subset_audit':audit}
        results.append(row)
        print(json.dumps({'cohort':cohort,'seed':seed,'mean':{k:statistics.mean(r['mean'][k] for r in metrics) for k in ('delta_nll','kl','top1_agreement')},'audit':audit}),flush=True)
    OUT.mkdir(parents=True,exist_ok=True)
    payload={'protocol':'FCC-1','index':index,'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
      'data_manifest_sha256':sha(DATA/'manifest.json'),'source_script_sha256':sha(Path(__file__)),
      'prereg_sha256':sha(ROOT/'docs/prereg/FRESH_COMPRESSION_CHECK_1.md'),
      'environment':{'python':sys.version,'torch':torch.__version__,'numpy':np.__version__,'platform':platform.platform(),'threads':torch.get_num_threads()},
      'results':results}
    (OUT/f'cell-{index}.json').write_text(json.dumps(payload,indent=2,sort_keys=True,allow_nan=False)+'\n')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--index',type=int,choices=range(4),required=True);run(p.parse_args().index)
