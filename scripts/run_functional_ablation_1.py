#!/usr/bin/env python3
"""FA-1 frozen-checkpoint diagnostic. Explicitly reads only validation arrays."""
from __future__ import annotations
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import subprocess
import sys
import time
import numpy as np
import torch
import torch.nn.functional as F
import yaml
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from pre_qwen_certification.functional_ablation import InferenceLM, accounting, BASE_INTERVENTIONS, COMPACT_INTERVENTIONS
from pre_qwen_certification.heterogeneous_rank import HeterogeneousSharedLowRankResidualMoE
from pre_qwen_certification.native_compact import build_paired_candidate_models, NativeArchitectureSpec, PRIMARY
from pre_qwen_certification.reality_gate_data import load_prepared_arrays, documents_from_prepared_split, ArrayTokenCorpus, sha256_file
from scripts.run_native_compact_gate_2a_seed_impl import _tiny_config

SEEDS={'mui1':(904031,904043,904051,904073),'gate2a':(202781,212789,222793,232801)}
OUT=ROOT/'results/functional-ablation-1'


def corpus_and_config():
    c=yaml.safe_load((ROOT/'configs/native_compact_gate_2a.yaml').read_text())
    cfg=_tiny_config(c['scales']['small']['model'])
    arrays,manifest=load_prepared_arrays(ROOT/'data/native-compact-gate-2a',splits=('validation',))
    docs=documents_from_prepared_split(arrays['validation'],prefix='native-calibration',domain='wikitext103-validation',
        maximum_document_tokens=2048,minimum_document_tokens=cfg.seq_len+2,maximum_total_tokens=120000)
    corpus=ArrayTokenCorpus({'calibration':docs},seq_len=cfg.seq_len,vocab_size=manifest['vocab_size'])
    windows=corpus.fixed_windows('calibration',windows_per_document=1,seed=904091)
    assert len(windows)==150
    return cfg,corpus,windows


def load_models(cohort,seed,cfg,vocab):
    templates=build_paired_candidate_models(vocab,cfg,NativeArchitectureSpec(8,42),seed=seed)
    if cohort=='mui1':
        path=ROOT/f'results/mode-utilization-intervention-1/seed-{seed}.pt'
        states=torch.load(path,map_location='cpu',weights_only=True)
    else:
        path=ROOT/f'results/native-compact-gate-2a/small/frozen-candidates-seed-{seed}.pt'
        states=torch.load(path,map_location='cpu',weights_only=True)['final_states']
    import copy
    models={}
    for arm,state in states.items():
        m=copy.deepcopy(templates[arm if arm in templates else PRIMARY])
        m.load_state_dict(state,strict=True)
        m.eval()
        models[arm]=m
    return models,path


@torch.inference_mode()
def time_models(models,vocab):
    rng=random.Random(904117)
    result={}
    for length in (1,64):
        tokens=torch.arange(length).remainder(vocab)[None,:]
        for m in models.values():
            for _ in range(5):
                m(tokens)
        blocks={name:[] for name in models}
        for _ in range(11):
            names=list(models); rng.shuffle(names)
            for name in names:
                start=time.perf_counter_ns()
                for _ in range(10):
                    models[name](tokens)
                blocks[name].append((time.perf_counter_ns()-start)/1e7)
        result[str(length)]={name:{'blocks_ms':vals,'median_ms':float(np.median(vals))} for name,vals in blocks.items()}
    return result


@torch.inference_mode()
def evaluate_arm(model,windows,cohort,seed,arm,folder):
    compact=isinstance(model.blocks[0].moe,HeterogeneousSharedLowRankResidualMoE)
    kinds=COMPACT_INTERVENTIONS if compact else BASE_INTERVENTIONS
    candidates={kind:InferenceLM(model,kind) for kind in kinds}
    original=candidates['original']
    max_parity=0.0
    ref=[]
    for w in windows:
        tokens=w.inputs[None,:]
        old=model(tokens)[0]
        new=original(tokens)
        max_parity=max(max_parity,float((old-new).abs().max()))
        assert torch.allclose(old,new,rtol=2e-5,atol=2e-5), ('wrapper mismatch',cohort,seed,arm)
        ref.append(new.clone())
    rows={}
    for kind,m in candidates.items():
        current=[]
        for w,r in zip(windows,ref,strict=True):
            z=r if kind=='original' else m(w.inputs[None,:])
            loss=float(F.cross_entropy(z.flatten(0,1),w.targets.flatten()))
            base_loss=float(F.cross_entropy(r.flatten(0,1),w.targets.flatten()))
            lp=F.log_softmax(r.double(),dim=-1)
            lq=F.log_softmax(z.double(),dim=-1)
            kl=float((lp.exp()*(lp-lq)).sum(-1).mean())
            agree=float((r.argmax(-1)==z.argmax(-1)).double().mean())
            current.append({'document_id':w.document_id,'start':w.start,'loss':loss,
                'delta_loss':loss-base_loss,'kl':kl,'top1_agreement':agree})
        rows[kind]={'records':current,'mean':{key:float(np.mean([v[key] for v in current])) for key in ('loss','delta_loss','kl','top1_agreement')},
                    'accounting':accounting(m)}
    if cohort=='mui1':
        previous=json.loads((ROOT/f'results/mode-utilization-intervention-1/seed-{seed}.json').read_text())
        delta=rows['original']['mean']['loss']-previous['histories'][arm][-1]['loss']
        assert abs(delta)<2e-5,('MUI reproduction',cohort,seed,arm,delta)
    eligible={k:v for k,v in candidates.items() if k in ('original','mean-matrices','common-only','rank1')}
    latencies=time_models(eligible,model.token_embedding.num_embeddings)
    for kind,m in eligible.items():
        destination=folder/f'{cohort}-{seed}-{arm}-{kind}.pt'
        torch.save({'cohort':cohort,'seed':seed,'arm':arm,'intervention':kind,'state_dict':m.state_dict()},destination)
        rows[kind]['export']={'file':destination.name,'bytes':destination.stat().st_size,'sha256':sha256_file(destination)}
        for length in ('1','64'):
            info=latencies[length][kind]
            rows[kind].setdefault('timing',{})[length]={**info,'ratio_to_original':info['median_ms']/latencies[length]['original']['median_ms']}
    return {'cohort':cohort,'seed':seed,'arm':arm,'wrapper_max_logit_error':max_parity,
            'interventions':rows}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--index',type=int,required=True,choices=range(4))
    p.add_argument('--cohort',choices=('both','mui1','gate2a'),default='both')
    p.add_argument('--only-arm')
    p.add_argument('--output-dir',type=Path,default=OUT)
    p.add_argument('--source-commit')
    args=p.parse_args()
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    folder=args.output_dir/f'cell-{args.index}'
    folder.mkdir(parents=True,exist_ok=True)
    cfg,corpus,windows=corpus_and_config()
    source=args.source_commit or subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    cohorts=('mui1','gate2a') if args.cohort=='both' else (args.cohort,)
    results=[]; hashes={}
    for cohort in cohorts:
        seed=SEEDS[cohort][args.index]
        models,path=load_models(cohort,seed,cfg,corpus.vocab_size)
        hashes[str(path.relative_to(ROOT))]=sha256_file(path)
        for arm,model in models.items():
            if args.only_arm and arm!=args.only_arm:
                continue
            result=evaluate_arm(model,windows,cohort,seed,arm,folder)
            results.append(result)
            print(json.dumps({'cohort':cohort,'seed':seed,'arm':arm,'summary':{k:v['mean'] for k,v in result['interventions'].items()}}),flush=True)
        del models
        gc.collect()
    data=ROOT/'data/native-compact-gate-2a'
    record={'protocol':'FA-1','status':'completed','source_commit':source,'index':args.index,
        'scope':'posthoc_development_calibration_only','loaded_splits':['validation'],
        'checkpoint_sha256':hashes,'array_sha256':{n:sha256_file(data/n) for n in ('validation-tokens.npy','validation-offsets.npy','tokenizer.json','manifest.json')},
        'source_sha256':{n:sha256_file(ROOT/n) for n in ('pre_qwen_certification/functional_ablation.py','scripts/run_functional_ablation_1.py')},
        'windows':len(windows),'unique_articles':len(set(w.document_id for w in windows)),
        'environment':{'python':sys.version,'torch':torch.__version__,'numpy':np.__version__,'platform':platform.platform(),
                       'cpu':platform.processor(),'threads':torch.get_num_threads(),'workflow_run_id':os.environ.get('GITHUB_RUN_ID')},
        'results':results}
    (folder/'results.json').write_text(json.dumps(record,indent=2,sort_keys=True,allow_nan=False)+'\n')

if __name__=='__main__':
    main()
