import polars as pl
import pytest

from ml_playground.experiments.executor import run_experiment


@pytest.fixture
def sample_csv(tmp_path):
    df = pl.DataFrame(
        {
            "feat1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            "feat2": [0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5],
            "target": [0, 0, 0, 0, 1, 1, 1, 1],
        }
    )
    path = tmp_path / "test.csv"
    df.write_csv(path)
    return str(path)


def test_run_experiment_holdout(sample_csv):
    config = {
        "data": {
            "path": sample_csv,
            "target": "target",
            "test_size": 0.25,
            "random_state": 42,
        },
        "preprocessing": {"scaling": {"method": "standard"}},
        "model": {"name": "logistic", "params": {"random_state": 42}},
        "metrics": ["accuracy", "f1"],
    }
    result = run_experiment(config)
    assert result["model"] == "logistic"
    assert "accuracy" in result["metrics"]
    assert "f1" in result["metrics"]
    assert 0 <= result["metrics"]["accuracy"] <= 1


def test_run_experiment_cv(sample_csv):
    config = {
        "data": {"path": sample_csv, "target": "target", "stratified": True},
        "cross_validation": {
            "method": "kfold",
            "n_splits": 3,
            "shuffle": True,
            "random_state": 42,
        },
        "preprocessing": {"scaling": {"method": "standard"}},
        "model": {"name": "logistic", "params": {"random_state": 42}},
        "metrics": ["accuracy"],
    }
    result = run_experiment(config)
    assert result["model"] == "logistic"
    assert "accuracy" in result["metrics"]
    assert "cv_scores" in result
    assert len(result["cv_scores"]["accuracy"]) == 3


def test_run_experiment_roc_auc(sample_csv):
    config = {
        "data": {
            "path": sample_csv,
            "target": "target",
            "test_size": 0.25,
            "random_state": 42,
        },
        "model": {"name": "logistic", "params": {"random_state": 42}},
        "metrics": ["roc_auc"],
    }
    result = run_experiment(config)
    assert "roc_auc" in result["metrics"]
