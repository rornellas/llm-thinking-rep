#!/usr/bin/env python3
"""DCG-1 holdout preparation: no model imports and no outcome-dependent selection."""
from __future__ import annotations
import hashlib
import json
from itertools import islice
from pathlib import Path
import platform
import sys
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.prepare_fresh_compression_check_1 import true_articles, self_test
OUT = ROOT / 'data/dense-control-gate-1'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    import datasets
    import tokenizers
    from datasets import load_dataset
    from tokenizers import Tokenizer
    from scripts.prepare_native_compact_wikitext103 import group_articles
    self_test()
    parent = ROOT / 'data/native-compact-gate-2a'
    old = json.loads((parent / 'manifest.json').read_text())
    assert old['tokenizer_training_documents'] == 2048 and old['vocab_size'] == 512
    assert sha(parent / 'tokenizer.json') == old['sha256']['tokenizer.json']
    tokenizer = Tokenizer.from_file(str(parent / 'tokenizer.json'))
    ds = load_dataset(old['source'], old['subset'], revision=old['revision'], split='train')
    texts = list(islice(true_articles(row['text'] for row in ds), 24576))
    assert len(texts) == 24576, 'Frozen article pool unavailable'
    stored = np.load(parent / 'train-tokens.npy', allow_pickle=False)
    rebuilt, segments, last_train, last_tokenizer = [], 0, -1, -1
    for article_id, text in enumerate(texts):
        for segment in group_articles(text.splitlines()):
            if segments < 2048:
                last_tokenizer = article_id
            segments += 1
            if len(rebuilt) < len(stored):
                ids = tokenizer.encode(segment).ids
                take = min(len(ids), len(stored) - len(rebuilt))
                if take:
                    rebuilt.extend(ids[:take]); last_train = article_id
            if len(rebuilt) == len(stored) and segments >= 2048:
                break
        if len(rebuilt) == len(stored) and segments >= 2048:
            break
    assert np.array_equal(np.asarray(rebuilt, dtype=stored.dtype), stored)
    assert max(last_train, last_tokenizer) < 8192
    excluded = set()
    old_hashes = {}
    for split in ('train', 'validation'):
        path = parent / f'{split}-tokens.npy'
        assert sha(path) == old['sha256'][path.name]
        old_hashes[path.name] = sha(path)
        array = np.load(path, allow_pickle=False).astype(np.int32, copy=False)
        for start in range(len(array) - 64):
            excluded.add(hashlib.sha256(array[start:start+65].tobytes()).digest())
    previous = ROOT / 'data/fresh-compression-check-1'
    prev_manifest = json.loads((previous / 'manifest.json').read_text())
    assert sha(previous / 'windows.npy') == prev_manifest['sha256']['windows.npy']
    old_articles = json.loads((previous / 'articles.json').read_text())
    assert max(a['article_id'] for a in old_articles) < 16384
    for window in np.load(previous / 'windows.npy', allow_pickle=False).reshape(-1, 65):
        excluded.add(hashlib.sha256(window.tobytes()).digest())
    used_hashes = {a['raw_text_sha256'] for a in old_articles}
    used_titles = {a['title'] for a in old_articles}
    used_windows = set()
    records, selected, rejections = [], [], []
    pool = np.random.default_rng(906131).permutation(np.arange(16384, 24576))
    rng = np.random.default_rng(906137)
    for value in pool:
        article_id = int(value)
        text = texts[article_id]
        title = text.splitlines()[0]
        text_hash = hashlib.sha256(text.encode()).hexdigest()
        if text_hash in used_hashes or title in used_titles:
            rejections.append({'article_id': article_id, 'reason': 'duplicate_title_or_text'}); continue
        ids = np.asarray(tokenizer.encode(text).ids[:4096], dtype=np.int32)
        if len(ids) < 512:
            rejections.append({'article_id': article_id, 'reason': 'too_short'}); continue
        edges = np.linspace(0, len(ids)-64, 5, dtype=int)
        windows, starts, hashes = [], [], set()
        for lo, hi in zip(edges[:-1], edges[1:], strict=True):
            for start_np in rng.permutation(np.arange(lo, hi)):
                start = int(start_np)
                window = ids[start:start+65]
                h = hashlib.sha256(window.tobytes()).digest()
                if h not in excluded and h not in used_windows and h not in hashes:
                    assert len(window) == 65
                    windows.append(window.copy()); starts.append(start); hashes.add(h); break
        if len(windows) != 4:
            rejections.append({'article_id': article_id, 'reason': 'no_unique_window_in_stratum'}); continue
        used_hashes.add(text_hash); used_titles.add(title); used_windows.update(hashes)
        records.append({'article_id': article_id, 'title': title, 'raw_text_sha256': text_hash,
                        'encoded_prefix_tokens': len(ids), 'starts': starts,
                        'window_sha256': [hashlib.sha256(w.tobytes()).hexdigest() for w in windows]})
        selected.append(np.stack(windows))
        if len(records) == 256:
            break
    assert len(records) == 256 and len(used_windows) == 1024 and not (used_windows & excluded)
    OUT.mkdir(parents=True, exist_ok=True)
    np.save(OUT / 'windows.npy', np.stack(selected))
    (OUT / 'articles.json').write_text(json.dumps(records, indent=2, ensure_ascii=False)+'\n')
    manifest = {'protocol': 'DCG-1', 'source': old['source'], 'subset': old['subset'], 'revision': old['revision'],
        'nominal_split': 'train', 'role': 'fresh_holdout_outside_bounded_training_tokenizer_and_FCC1_pools',
        'parent_prefix_exact_reproduction': True, 'last_training_true_article': last_train,
        'last_tokenizer_true_article': last_tokenizer, 'pool_bounds': [16384, 24576],
        'selection_seed': 906131, 'window_seed': 906137, 'articles': 256, 'windows_per_article': 4,
        'scored_tokens_per_model': 65536, 'exact_duplicate_window_intersections': 0,
        'parent_array_sha256': old_hashes, 'tokenizer_sha256': sha(parent/'tokenizer.json'),
        'FCC1_windows_sha256': sha(previous/'windows.npy'), 'FCC1_articles_sha256': sha(previous/'articles.json'),
        'source_script_sha256': sha(Path(__file__)),
        'protocol_sha256': sha(ROOT/'docs/prereg/DENSE_CONTROL_GATE_1.md'),
        'sha256': {name: sha(OUT/name) for name in ('windows.npy', 'articles.json')},
        'rejections': rejections,
        'license': 'Source WikiText: CC-BY-SA-3.0 / GFDL; retain source and revision attribution.',
        'environment': {'python': sys.version, 'platform': platform.platform(), 'numpy': np.__version__,
                        'datasets': datasets.__version__, 'tokenizers': tokenizers.__version__}}
    (OUT/'manifest.json').write_text(json.dumps(manifest, indent=2, sort_keys=True)+'\n')
    print(json.dumps({k: v for k, v in manifest.items() if k != 'rejections'}, indent=2))


if __name__ == '__main__':
    main()
