"""Preflight tests never read scientific corpora or checkpoints."""
import numpy as np
import pytest
import torch
from pre_qwen_certification.dense_control import self_test as model_test
from scripts.aggregate_dense_control_gate_1 import self_test as stats_test, paired_stats, draws
from scripts.prepare_fresh_compression_check_1 import self_test as article_test


def test_model_budget_pairing_and_backward():
    torch.set_num_threads(2)
    assert model_test()['passed']


def test_statistics_constant_shift_and_independent_paths():
    assert stats_test()['passed']


def test_top_level_articles_not_sections():
    article_test()


@pytest.mark.parametrize('shape', [(4, 256), (6, 255)])
def test_missing_seeds_or_articles_rejected(shape):
    s, a = draws()
    with pytest.raises(AssertionError):
        paired_stats(np.zeros(shape), s, a)


def test_nonfinite_results_rejected():
    s, a = draws()
    values = np.zeros((6, 256)); values[0, 0] = np.nan
    with pytest.raises(AssertionError):
        paired_stats(values, s, a)
