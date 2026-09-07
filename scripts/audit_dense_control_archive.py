#!/usr/bin/env python3
"""DCG-1 archive audit in a separate runtime. No training or changed thresholds."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
import platform
import statistics
import sys
import numpy as np
from scipy.stats import t
import torch


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit(root):
    out = root/'results/dense-control-gate-1'
    verified = []
    for line in (out/'SHA256SUMS').read_text().splitlines():
        expected, path = line.split('  ', 1)
        assert sha(root/path) == expected, path
        verified.append(path)
    summary = json.loads((out/'summary.json').read_text())
    raw = [json.loads((out/f'cell-{i}/evaluation.json').read_text()) for i in range(6)]
    frozen = [json.loads((out/f'cell-{i}/frozen.json').read_text()) for i in range(6)]
    matrices = {}
    model_counts = []
    for name in summary['arms']:
        matrix = [[math.fsum(w['loss'][name] for w in article['windows'])/4 for article in cell['articles']] for cell in raw]
        mean = math.fsum(math.fsum(row)/256 for row in matrix)/6
        assert abs(mean-summary['arms'][name]['mean_nll']) < 1e-12
        matrices[name] = np.asarray(matrix)
        for i, f in enumerate(frozen):
            info = f['exports'][name]
            state = torch.load(out/f'cell-{i}'/info['file'], map_location='cpu', weights_only=True)
            unique = {v.untyped_storage().data_ptr(): v.untyped_storage().nbytes() for v in state.values()}
            assert sum(unique.values())//4 == info['accounting']['total_parameters']
            assert all(v.dtype == torch.float32 for v in state.values())
            model_counts.append({'seed': f['seed'], 'arm': name, 'parameters': sum(unique.values())//4})
    samples = np.load(out/'bootstrap-draws.npz', allow_pickle=False)
    seed_draws, article_draws = samples['seed_draws'], samples['article_draws']
    assert seed_draws.shape == (10000, 6) and article_draws.shape == (10000, 256)
    records = {}
    for key, expected in summary['primary_contrasts'].items():
        candidate, control = key.split('__minus__')
        x = matrices[candidate]-matrices[control]
        per_seed = [math.fsum(row.tolist())/256 for row in x]
        mean = math.fsum(per_seed)/6
        stderr = math.sqrt(math.fsum((v-mean)**2 for v in per_seed)/30)
        bound = mean + float(t.ppf(.9875, 5))*stderr
        # Average sampled seeds first, then articles; a third arithmetic path.
        boot = np.empty(10000)
        for i, (sd, ad) in enumerate(zip(seed_draws, article_draws, strict=True)):
            seed_mean = np.add.reduce(x[sd], axis=0)/6
            boot[i] = math.fsum(seed_mean[ad].tolist())/256
        ordered = sorted(boot.tolist())
        position = .9875*(len(ordered)-1)
        low = math.floor(position)
        percentile = ordered[low] + (position-low)*(ordered[low+1]-ordered[low])
        assert abs(mean-expected['mean']) < 1e-12
        assert abs(bound-expected['seed_upper']) < 1e-12
        assert abs(percentile-expected['crossed_upper']) < 1e-10
        passed = bound <= -.010 and percentile <= -.010 and max(per_seed) <= .010
        assert passed == expected['meaningful_superiority']
        records[key] = {'mean': mean, 'seed_upper': bound, 'crossed_upper': percentile, 'passed': passed}
    timing_error = 0.0
    for candidate, gate in summary['candidate_gates'].items():
        for key, row in gate['runtime_ratios'].items():
            control, length = key.split('_L')
            ratios = []
            for cell in raw:
                measurements = cell['inference'][length]['models']
                ratios.append(statistics.median(measurements[candidate]['blocks_ms'])/
                              statistics.median(measurements[control]['blocks_ms']))
            timing_error = max(timing_error, abs(statistics.median(ratios)-row['median_ratio']))
    assert timing_error < 1e-12
    report = {'passed': True, 'scope': 'archive arithmetic, real serialized storage and timing ratios; no new training',
              'source_commit': summary['source_commit'], 'verified_files': len(verified),
              'verified_model_exports': len(model_counts), 'models': model_counts, 'primary_contrasts': records,
              'max_timing_ratio_error': timing_error, 'audit_script_sha256': sha(Path(__file__)),
              'environment': {'python': sys.version, 'torch': str(torch.__version__), 'numpy': np.__version__,
                              'platform': platform.platform()},
              'not_an_independent_training_replication': True}
    (out/'external-archive-audit.json').write_text(json.dumps(report, indent=2, sort_keys=True)+'\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'models'}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    audit(parser.parse_args().root)
