import numpy as np
import pytest

from ml_playground.evaluation.statistics import (
    friedman_test,
    nemenyi_test,
    wilcoxon_test,
)


def test_wilcoxon_test():
    a = np.array([0.9, 0.8, 0.7, 0.6])
    b = np.array([0.85, 0.75, 0.65, 0.55])
    result = wilcoxon_test(a, b)
    assert "statistic" in result
    assert "p_value" in result
    assert isinstance(result["statistic"], float)
    assert isinstance(result["p_value"], float)


def test_wilcoxon_identical_returns_p_one():
    a = np.array([0.8, 0.8, 0.8])
    b = np.array([0.8, 0.8, 0.8])
    result = wilcoxon_test(a, b)
    assert result["p_value"] == 1.0


def test_friedman_test():
    a = np.array([0.9, 0.8, 0.7])
    b = np.array([0.85, 0.75, 0.65])
    c = np.array([0.8, 0.7, 0.6])
    result = friedman_test(a, b, c)
    assert "statistic" in result
    assert "p_value" in result


def test_nemenyi_test():
    data = np.array(
        [
            [0.9, 0.8, 0.7],
            [0.85, 0.8, 0.75],
            [0.95, 0.85, 0.8],
        ]
    )
    result = nemenyi_test(data)
    assert "avg_ranks" in result
    assert "cd" in result
    assert "n_models" in result
    assert "n_datasets" in result
    assert len(result["avg_ranks"]) == 3
    assert result["cd"] > 0


def test_nemenyi_single_dataset():
    data = np.array([[0.9, 0.8, 0.7]])
    result = nemenyi_test(data)
    assert len(result["avg_ranks"]) == 3


def test_nemenyi_returns_float_cd():
    data = np.array(
        [
            [0.9, 0.8, 0.7],
            [0.85, 0.8, 0.75],
        ]
    )
    result = nemenyi_test(data, alpha=0.05)
    assert isinstance(result["cd"], float)
