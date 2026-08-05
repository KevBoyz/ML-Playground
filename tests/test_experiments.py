from pathlib import Path

from ml_playground.experiments.config import discover_experiments, load_experiment
from ml_playground.experiments.runner import run_grid
from ml_playground.models import get_model


def test_discover_and_load_named_experiment():
    paths = discover_experiments("experiments")
    by_name = {path.name: path for path in paths}
    assert set(by_name) >= {
        "customer_segments",
        "iris_baseline",
        "iris_no_scaling",
        "salary_regression",
    }

    config = load_experiment(by_name["iris_baseline"])
    assert config["experiment_name"] == "iris_baseline"
    assert config["task"] == "classification"
    assert Path(config["data"]["path"]).name == "iris.csv"
    assert {entry["name"] for entry in config["models"]} == {
        "knn",
        "logistic_regression",
        "svm",
        "decision_tree",
    }


def test_decision_tree_is_registered():
    model = get_model("decision_tree", {"max_depth": 3, "random_state": 42})
    assert model.max_depth == 3


def test_experiment_report_tree_is_isolated(tmp_path):
    config = load_experiment("experiments/iris_baseline")
    config["outputs"]["root"] = str(tmp_path / "reports")
    config["project_root"] = str(tmp_path)
    config["models"] = [
        {"name": "decision_tree", "params": {"max_depth": [3], "random_state": [42]}}
    ]

    result = run_grid(config, write_reports=True)

    report_root = tmp_path / "reports" / "iris_baseline"
    assert result["best"]["name"] == "decision_tree"
    assert (report_root / "metrics").is_dir()
    assert (report_root / "tables").is_dir()
    assert (report_root / "figures").is_dir()
    assert list((report_root / "tables").glob("*_manifest.json"))
    assert list((tmp_path / "models" / "iris_baseline").rglob("model.joblib"))
