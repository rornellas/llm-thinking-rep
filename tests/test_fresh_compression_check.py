import json
import numpy as np
from scripts.prepare_fresh_compression_check_1 import self_test, true_articles
from scripts.aggregate_fresh_compression_check_1 import summary_stats


def test_top_level_article_boundaries():
    self_test()
    rows=list(true_articles(['= A =','body','== Section ==','other','= = Section 2 = =','more','= B =','end']))
    assert len(rows)==2 and 'Section 2' in rows[0] and rows[1].startswith('= B =')


def test_crossed_bootstrap_constant_and_json_types():
    rng=np.random.default_rng(100)
    sd=rng.integers(0,4,size=(10000,4),dtype=np.int16)
    ad=rng.integers(0,256,size=(10000,256),dtype=np.int16)
    out=summary_stats(np.full((4,256),0.002),sd,ad)
    assert abs(out['seed_t']['mean']-.002)<1e-12
    assert abs(out['seed_t']['ucb95']-.002)<1e-12
    assert abs(out['crossed_bootstrap']['ucb95']-.002)<1e-12
    assert out['crossed_bootstrap']['audit_max_difference']<1e-10
    condition=out['seed_t']['ucb95']<=.010
    assert type(condition) is bool
    json.dumps({'out':out,'condition':condition},allow_nan=False)
