#!/usr/bin/env python3
"""FCC-1 data preparation, independent of models and losses."""
from __future__ import annotations
from pathlib import Path
import hashlib,json,platform,sys
from itertools import islice
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
OUT=ROOT/'data/fresh-compression-check-1'


def top_heading(text):
    s=text.strip()
    if len(s)<3 or not(s.startswith('=') and s.endswith('=')):return False
    body=s[1:-1].strip()
    return bool(body) and not body.startswith('=') and not body.endswith('=')


def true_articles(lines):
    current=[]
    for raw in lines:
        s=str(raw).strip()
        if not s:continue
        if top_heading(s) and current:
            yield '\n'.join(current)+'\n';current=[]
        current.append(s)
    if current:yield '\n'.join(current)+'\n'


def digest(data):return hashlib.sha256(data).hexdigest()
def sha(path):return digest(path.read_bytes())


def self_test():
    assert top_heading(' = Article = ')
    assert not top_heading(' = = Section = = ')
    assert not top_heading('=  =  Subsection  =  =')
    assert not top_heading('== Section ==')
    assert not top_heading('ordinary text')
    rows=list(true_articles(['= A =','a','= = Section = =','b','= B =','c']))
    assert len(rows)==2 and '= = Section = =' in rows[0]


def main():
    import datasets,tokenizers
    from datasets import load_dataset
    from tokenizers import Tokenizer
    from scripts.prepare_native_compact_wikitext103 import group_articles
    self_test()
    parent=ROOT/'data/native-compact-gate-2a'
    old=json.loads((parent/'manifest.json').read_text())
    assert old['tokenizer_training_documents']==2048 and old['vocab_size']==512
    tokenizer=Tokenizer.from_file(str(parent/'tokenizer.json'))
    assert sha(parent/'tokenizer.json')==old['sha256']['tokenizer.json']
    ds=load_dataset(old['source'],old['subset'],revision=old['revision'],split='train')
    texts=list(islice(true_articles(row['text'] for row in ds),16384))
    if len(texts)!=16384:raise RuntimeError('Frozen article pool unavailable')
    stored=np.load(parent/'train-tokens.npy',allow_pickle=False)
    rebuilt=[];segments=0;last_train=-1;last_tokenizer=-1
    for article_id,text in enumerate(texts):
        for segment in group_articles(text.splitlines()):
            if segments<2048:last_tokenizer=article_id
            segments+=1
            if len(rebuilt)<len(stored):
                ids=tokenizer.encode(segment).ids
                take=min(len(ids),len(stored)-len(rebuilt))
                if take:rebuilt.extend(ids[:take]);last_train=article_id
            if len(rebuilt)==len(stored) and segments>=2048:break
        if len(rebuilt)==len(stored) and segments>=2048:break
    assert np.array_equal(np.asarray(rebuilt,dtype=stored.dtype),stored),'Parent train-prefix reconstruction mismatch'
    last_used=max(last_train,last_tokenizer)
    assert last_used<8192
    # Exact 65-token sequence hashes include the prediction target. Include all starts,
    # even starts crossing old segment boundaries: conservative duplicate exclusion.
    old_windows=set()
    old_array_hashes={}
    for split in ('train','validation'):
        p=parent/f'{split}-tokens.npy';array=np.load(p,allow_pickle=False).astype(np.int32,copy=False)
        old_array_hashes[p.name]=sha(p)
        for start in range(len(array)-64):old_windows.add(hashlib.sha256(array[start:start+65].tobytes()).digest())
    pool=np.random.default_rng(904301).permutation(np.arange(8192,16384))
    wrng=np.random.default_rng(904307)
    used_windows=set();used_text=set();used_titles=set()
    records=[];window_tokens=[];rejections=[]
    for article_id_np in pool:
        article_id=int(article_id_np);text=texts[article_id];title=text.splitlines()[0]
        text_sha=digest(text.encode())
        if text_sha in used_text or title in used_titles:
            rejections.append({'article_id':article_id,'reason':'duplicate_article'});continue
        ids=np.asarray(tokenizer.encode(text).ids[:4096],dtype=np.int32)
        if len(ids)<512:
            rejections.append({'article_id':article_id,'reason':'too_short'});continue
        edges=np.linspace(0,len(ids)-64,5,dtype=int)
        starts=[];windows=[];provisional=set();valid=True
        for lo,hi in zip(edges[:-1],edges[1:],strict=True):
            found=False
            for start_np in wrng.permutation(np.arange(lo,hi)):
                start=int(start_np);window=ids[start:start+65]
                assert len(window)==65
                h=hashlib.sha256(window.tobytes()).digest()
                if h not in old_windows and h not in used_windows and h not in provisional:
                    starts.append(start);windows.append(window.copy());provisional.add(h);found=True;break
            if not found:valid=False;break
        if not valid:
            rejections.append({'article_id':article_id,'reason':'no_unique_window_in_stratum'});continue
        used_windows.update(provisional);used_text.add(text_sha);used_titles.add(title)
        records.append({'article_id':article_id,'title':title,'raw_text_sha256':text_sha,
                        'encoded_prefix_tokens':len(ids),'starts':starts,'window_sha256':[digest(w.tobytes()) for w in windows]})
        window_tokens.append(np.stack(windows))
        if len(records)==256:break
    assert len(records)==256 and len(used_windows)==1024
    OUT.mkdir(parents=True,exist_ok=True)
    path=OUT/'windows.npy';np.save(path,np.stack(window_tokens))
    (OUT/'articles.json').write_text(json.dumps(records,indent=2,ensure_ascii=False)+'\n')
    manifest={'protocol':'FCC-1','source':old['source'],'subset':old['subset'],'revision':old['revision'],'nominal_split':'train',
      'role_for_these_checkpoints':'fresh_holdout_outside_bounded_training_and_tokenizer_prefix',
      'parent_prefix_exact_reproduction':True,'last_training_true_article':last_train,'last_tokenizer_true_article':last_tokenizer,
      'pool_bounds':[8192,16384],'selection_seed':904301,'window_seed':904307,'articles':256,'windows_per_article':4,
      'tokens_per_window':65,'prediction_tokens_per_window':64,'duplicate_window_intersection':0,
      'old_window_hash_count':len(old_windows),'rejections':rejections,'parent_array_sha256':old_array_hashes,
      'tokenizer_sha256':sha(parent/'tokenizer.json'),'parent_manifest_sha256':sha(parent/'manifest.json'),
      'sha256':{'windows.npy':sha(path),'articles.json':sha(OUT/'articles.json')},'source_script_sha256':sha(Path(__file__)),
      'prereg_sha256':sha(ROOT/'docs/prereg/FRESH_COMPRESSION_CHECK_1.md'),
      'license':'Source WikiText: CC-BY-SA-3.0 / GFDL; retain source/revision attribution.',
      'environment':{'python':sys.version,'platform':platform.platform(),'numpy':np.__version__,'datasets':datasets.__version__,'tokenizers':tokenizers.__version__}}
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n')
    print(json.dumps({k:v for k,v in manifest.items() if k not in ('rejections','environment')},indent=2),flush=True)

if __name__=='__main__':main()
