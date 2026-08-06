from pathlib import Path

import pandas as pd

from ml_playground.experiments.report import results_to_csv, write_experiment_reports


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


def test_experiment_reports_omit_empty_errors_file(tmp_path):
    config = {
        "experiment_name": "report_contract",
        "task": "classification",
        "selection": {"primary_metric": "accuracy"},
        "outputs": {
            "root": str(tmp_path / "reports"),
            "save_predictions": False,
            "figures": False,
        },
        "views": {},
    }
    bundle = {
        "results": [{"name": "model", "status": "success", "metrics": {"accuracy": 1.0}}]
    }

    artifacts = write_experiment_reports(config, bundle, "run_without_errors")

    errors_path = tmp_path / "reports" / "report_contract" / "tables" / "run_without_errors_errors.csv"
    assert "errors" not in artifacts
    assert not errors_path.exists()


def test_experiment_reports_write_errors_file_when_needed(tmp_path):
    config = {
        "experiment_name": "report_contract",
        "task": "classification",
        "selection": {"primary_metric": "accuracy"},
        "outputs": {
            "root": str(tmp_path / "reports"),
            "save_predictions": False,
            "figures": False,
        },
        "views": {},
    }
    bundle = {
        "results": [{"name": "broken_model", "status": "error", "error": "timeout"}]
    }

    artifacts = write_experiment_reports(config, bundle, "run_with_errors")

    errors_path = Path(artifacts["errors"])
    assert errors_path.is_file()
    assert errors_path.name == "run_with_errors_errors.csv"
