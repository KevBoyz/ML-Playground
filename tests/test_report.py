import pandas as pd

from ml_playground.experiments.report import results_to_csv


def test_results_to_csv_creates_file(tmp_path):
    results = [
        {"name": "rf", "params": {"n": 100}, "metrics": {"acc": 0.9}},
        {"name": "svm", "params": {"c": 1.0}, "metrics": {"acc": 0.85}},
    ]
    path = results_to_csv(results, str(tmp_path / "report.csv"))
    df = pd.read_csv(path)
    assert len(df) == 2
    assert "model" in df.columns
    assert "param_n" in df.columns
    assert "param_c" in df.columns
    assert "metric_acc" in df.columns


def test_results_to_csv_with_errors(tmp_path):
    results = [
        {"name": "rf", "error": "timeout"},
    ]
    path = results_to_csv(results, str(tmp_path / "report.csv"))
    df = pd.read_csv(path)
    assert df.iloc[0]["error"] == "timeout"


def test_results_to_csv_empty(tmp_path):
    path = results_to_csv([], str(tmp_path / "empty.csv"))
    with open(path, encoding="utf-8") as f:
        content = f.read().strip()
    assert content == ""
