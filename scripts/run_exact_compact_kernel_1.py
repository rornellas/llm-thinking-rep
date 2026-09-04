#!/usr/bin/env python3
"""ECK-1: one fixed exact kernel, all eight primary checkpoints."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import platform
import random
import statistics
import subprocess
import sys
import time
import torch
import torch.nn.functional as F
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.run_functional_ablation_1 import corpus_and_config, load_models, SEEDS
from pre_qwen_certification.functional_ablation import InferenceLM,accounting
from pre_qwen_certification.vectorized_compact import vectorize_inference_model
OUT=ROOT/'results/exact-compact-kernel-1'


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


@torch.inference_mode()
def benchmark(models,windows):
    result={};rng=random.Random(904211)
    for length in (1,64):
        inputs=[w.inputs[:length][None,:] for w in windows[:4]]
        for model in models.values():
            for i in range(5):model(inputs[i%4])
        records={name:[] for name in models}
        for block in range(15):
            order=list(models);rng.shuffle(order)
            for name in order:
                t0=time.perf_counter_ns()
                for i in range(12):models[name](inputs[i%4])
                records[name].append((time.perf_counter_ns()-t0)/12e6)
        result[str(length)]={name:{'blocks_ms':v,'median_ms':statistics.median(v)} for name,v in records.items()}
        for name,row in result[str(length)].items():
            row['ratio_to_loop8']=row['median_ms']/result[str(length)]['loop8']['median_ms']
            row['ratio_to_narrow65']=row['median_ms']/result[str(length)]['narrow65']['median_ms']
    return result


@torch.inference_mode()
def run(index):
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    folder=OUT/f'cell-{index}';folder.mkdir(parents=True,exist_ok=True)
    cfg,corpus,windows=corpus_and_config()
    results=[];hashes={}
    for cohort in ('mui1','gate2a'):
        seed=SEEDS[cohort][index]
        loaded,path=load_models(cohort,seed,cfg,corpus.vocab_size)
        hashes[str(path.relative_to(ROOT))]=sha(path)
        primary=loaded['legacy' if cohort=='mui1' else 'native-shared-rank']
        loop8=InferenceLM(primary,'original');loop1=InferenceLM(primary,'rank1')
        models={'loop8':loop8,'vector8':vectorize_inference_model(loop8),'loop1':loop1,
                'vector1':vectorize_inference_model(loop1),'narrow65':InferenceLM(loaded['conventional-narrow65'],'original'),
                'full':InferenceLM(loaded['conventional-full'],'original')}
        parameter_counts={name:sum(p.numel() for p in m.parameters()) for name,m in models.items()}
        assert parameter_counts['loop8']==parameter_counts['vector8']==95552
        assert parameter_counts['loop1']==parameter_counts['vector1']==47168
        parity={}
        for rank in (1,8):
            rows=[];max_error=0.0
            for w in windows:
                a=models[f'loop{rank}'](w.inputs[None,:]);b=models[f'vector{rank}'](w.inputs[None,:])
                error=float((a-b).abs().max());max_error=max(max_error,error)
                assert torch.allclose(a,b,atol=2e-5,rtol=2e-5),(cohort,seed,rank,w.document_id,error)
                la=float(F.cross_entropy(a.flatten(0,1),w.targets));lb=float(F.cross_entropy(b.flatten(0,1),w.targets))
                rows.append({'document_id':w.document_id,'start':w.start,'loop_loss':la,'vector_loss':lb,'delta_loss':lb-la,'max_logit_error':error})
            mean_delta=statistics.mean(r['delta_loss'] for r in rows)
            assert abs(mean_delta)<=2e-5
            parity[str(rank)]={'max_logit_error':max_error,'mean_nll_delta':mean_delta,'records':rows}
        timings=benchmark(models,windows)
        artifacts={}
        for name in ('vector8','vector1'):
            dest=folder/f'{cohort}-{seed}-{name}.pt'
            torch.save({'cohort':cohort,'seed':seed,'kernel':name,'state_dict':models[name].state_dict()},dest)
            artifacts[name]={'file':dest.name,'sha256':sha(dest),'bytes':dest.stat().st_size}
        row={'cohort':cohort,'seed':seed,'parity':parity,'timing':timings,'total_parameters':parameter_counts,
             'parameter_bytes':{k:4*v for k,v in parameter_counts.items()},
             'expert_macs':{name:accounting(models['loop1' if name=='vector1' else 'loop8' if name=='vector8' else name])['expert_matrix_macs_per_token'] for name in models},
             'exports':artifacts}
        results.append(row)
        print(json.dumps({'cohort':cohort,'seed':seed,'timing':{l:{k:{a:b for a,b in v.items() if a!='blocks_ms'} for k,v in rows.items()} for l,rows in timings.items()},'parity':{r:{k:v for k,v in p.items() if k!='records'} for r,p in parity.items()}}),flush=True)
    record={'protocol':'ECK-1','index':index,'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        'checkpoint_sha256':hashes,'loaded_splits':['validation'],'calibration_windows':150,
        'source_sha256':{p:sha(ROOT/p) for p in ('docs/prereg/EXACT_COMPACT_KERNEL_1.md','pre_qwen_certification/vectorized_compact.py','scripts/run_exact_compact_kernel_1.py','tests/test_vectorized_compact.py')},
        'environment':{'python':sys.version,'torch':torch.__version__,'platform':platform.platform(),'threads':torch.get_num_threads(),
                       'lscpu':subprocess.check_output(['lscpu'],text=True)},'results':results}
    (folder/'results.json').write_text(json.dumps(record,indent=2,sort_keys=True,allow_nan=False)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--index',type=int,choices=range(4),required=True);run(p.parse_args().index)
