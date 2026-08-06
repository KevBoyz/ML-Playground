from pathlib import Path

from ml_playground.experiments.config import discover_experiments, load_experiment
from ml_playground.experiments.runner import run_grid
from ml_playground.models import get_model


def test_discover_and_load_named_temporary_experiment(tmp_path):
    experiments_root = tmp_path / "experiments"
    experiment = experiments_root / "iris_contract"
    experiment.mkdir(parents=True)
    (tmp_path / "iris.csv").write_text(
        "feature_a,feature_b,target\n1,1,0\n2,2,1\n3,3,1\n",
        encoding="utf-8",
    )
    (experiment / "experiment.yaml").write_text(
        "name: iris_contract\n"
        "task: classification\n"
        "data:\n"
        "  path: iris.csv\n"
        "  target: target\n"
        "  features: [feature_a, feature_b]\n"
        "selection:\n"
        "  primary_metric: f1_macro\n"
        "  direction: maximize\n",
        encoding="utf-8",
    )
    (experiment / "models.yaml").write_text(
        "models:\n"
        "  - name: decision_tree\n"
        "    params:\n"
        "      max_depth: [3]\n",
        encoding="utf-8",
    )
    (experiment / "preprocessing.yaml").write_text("{}\n", encoding="utf-8")
    (experiment / "metrics.yaml").write_text(
        "classification:\n"
        "  names: [accuracy, f1_macro]\n"
        "  primary: f1_macro\n",
        encoding="utf-8",
    )

    paths = discover_experiments(experiments_root)
    by_name = {path.name: path for path in paths}
    assert set(by_name) == {"iris_contract"}

    config = load_experiment(by_name["iris_contract"], project_root=tmp_path)
    assert config["experiment_name"] == "iris_contract"
    assert config["task"] == "classification"
    assert Path(config["data"]["path"]).name == "iris.csv"
    assert {entry["name"] for entry in config["models"]} == {"decision_tree"}


def test_decision_tree_is_registered():
    model = get_model("decision_tree", {"max_depth": 3, "random_state": 42})
    assert model.max_depth == 3


def test_experiment_report_tree_is_isolated(tmp_path):
    dataset = tmp_path / "classification.csv"
    dataset.write_text(
        "feature_a,feature_b,target\n"
        "1,1,0\n1,2,0\n2,1,0\n"
        "5,5,1\n5,6,1\n6,5,1\n"
        "9,9,2\n9,10,2\n10,9,2\n",
        encoding="utf-8",
    )
    config = {
        "experiment_name": "classification_report",
        "project_root": str(tmp_path),
        "task": "classification",
        "data": {"path": str(dataset), "target": "target", "random_state": 42},
        "cross_validation": {"method": "holdout", "test_size": 1 / 3, "random_state": 42},
        "models": [
            {"name": "decision_tree", "params": {"max_depth": [3], "random_state": [42]}}
        ],
        "preprocessing": {},
        "metrics": ["accuracy", "f1_macro"],
        "selection": {"primary_metric": "f1_macro", "direction": "maximize"},
        "outputs": {
            "root": str(tmp_path / "reports"),
            "save_model": True,
            "save_predictions": True,
            "figures": True,
        },
    }

    result = run_grid(config, write_reports=True)

    report_root = tmp_path / "reports" / "classification_report"
    assert result["best"]["name"] == "decision_tree"
    assert (report_root / "metrics").is_dir()
    assert (report_root / "tables").is_dir()
    assert (report_root / "figures").is_dir()
    assert list((report_root / "tables").glob("*_manifest.json"))
    assert list((tmp_path / "models" / "classification_report").rglob("model.joblib"))
