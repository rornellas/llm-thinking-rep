#!/usr/bin/env python3
"""Posthoc ECK-1 routing trace, never a tolerance change or speed certificate."""
from pathlib import Path
import copy,json,platform,sys
import torch
from torch import nn
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.run_functional_ablation_1 import corpus_and_config,load_models
from pre_qwen_certification.functional_ablation import InferenceLM
from pre_qwen_certification.vectorized_compact import vectorize_inference_model

class ForcedRoute(nn.Module):
    def __init__(self,source,ids,weights):
        super().__init__();self.source=source;self.ids=ids;self.weights=weights
    def forward(self,x):
        return self.source(x,forced_top_ids=self.ids,forced_weights=self.weights)

@torch.inference_mode()
def main():
    torch.set_num_threads(2)
    cfg,corpus,windows=corpus_and_config()
    found=None
    for seed in (904031,904043):
        loaded,_=load_models('mui1',seed,cfg,corpus.vocab_size)
        original=InferenceLM(loaded['legacy'],'original')
        vector=vectorize_inference_model(original)
        traces={'original':{},'vector':{}}
        handles=[]
        def hook(label,index):
            def record(module,inputs,output):
                y,r=output
                traces[label][index]={'input':inputs[0].clone(),'output':y.clone(),
                    'ids':r.top_ids.clone(),'weights':r.weights.clone(),'logits':r.logits.clone()}
            return record
        for label,model in (('original',original),('vector',vector)):
            for i,b in enumerate(model.blocks):handles.append(b.moe.register_forward_hook(hook(label,i)))
        for w in windows:
            a=original(w.inputs[None,:]);b=vector(w.inputs[None,:])
            if torch.allclose(a,b,atol=2e-5,rtol=2e-5):continue
            layer_rows=[]
            for i in range(len(original.blocks)):
                x=traces['original'][i];y=traces['vector'][i]
                changed=(x['ids'].sort(-1).values!=y['ids'].sort(-1).values).any(-1)
                positions=changed.nonzero().flatten().tolist()
                sorted_logits=x['logits'].sort(-1,descending=True).values
                margins=sorted_logits[:,3]-sorted_logits[:,4]
                layer_rows.append({'layer':i,'max_input_error':float((x['input']-y['input']).abs().max()),
                    'max_output_error':float((x['output']-y['output']).abs().max()),
                    'max_router_logit_error':float((x['logits']-y['logits']).abs().max()),
                    'changed_token_positions':positions,
                    'reference_boundary_margins':[float(margins[p]) for p in positions],
                    'reference_top_ids':[x['ids'][p].tolist() for p in positions],
                    'vector_top_ids':[y['ids'][p].tolist() for p in positions]})
            for h in handles:h.remove()
            forced=copy.deepcopy(vector)
            for i,block in enumerate(forced.blocks):
                r=traces['original'][i]
                block.moe=ForcedRoute(block.moe,r['ids'],r['weights'])
            c=forced(w.inputs[None,:])
            found={'seed':seed,'document_id':w.document_id,'start':w.start,
                'unforced_max_logit_error':float((a-b).abs().max()),'layers':layer_rows,
                'forced_original_routes_max_logit_error':float((a-c).abs().max()),
                'forced_original_routes_meet_original_tolerance':bool(torch.allclose(a,c,atol=2e-5,rtol=2e-5))}
            break
        if found:break
        for h in handles:h.remove()
    result={'scope':'posthoc_specific_failure_counterfactual','original_run':33920705855,
       'original_verdict':'FAIL_INVALID_FOR_SPEEDUP_CLAIM',
       'status':'REPRODUCED' if found else 'NOT_REPRODUCED_ON_THIS_RUNNER','trace':found,
       'environment':{'python':sys.version,'torch':torch.__version__,'platform':platform.platform()},
       'note':'Forcing original IDs and probabilities isolates the downstream routing path jointly, not every floating-point effect.'}
    p=ROOT/'results/exact-compact-kernel-1/failure-diagnostic.json';p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    print(json.dumps(result,indent=2),flush=True)

if __name__=='__main__':main()
