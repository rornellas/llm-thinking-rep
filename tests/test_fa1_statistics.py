import json
import statistics
from scripts.aggregate_functional_ablation_1 import stats


def test_native_json_safe_statistics_and_conditions():
    values=[0.001,0.002,0.003,0.004]
    s=stats(values)
    se=statistics.stdev(values)/2
    assert abs(s['ucb95']-(statistics.mean(values)+2.353363434801827*se))<1e-12
    assert abs(s['ci95'][1]-(statistics.mean(values)+3.182446305284263*se))<1e-12
    payload={'statistics':s,'conditions':{'noninferior':s['ucb95']<=0.010}}
    assert type(payload['conditions']['noninferior']) is bool
    assert json.loads(json.dumps(payload,allow_nan=False))==payload


def test_zero_variance_has_zero_width_interval():
    s=stats([0.0]*4)
    assert s['ucb95']==0 and s['ci95']==[0,0]
    json.dumps({'condition':s['ucb95']<=.010},allow_nan=False)
