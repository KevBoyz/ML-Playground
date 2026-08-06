from pathlib import Path

import polars as pl

from ml_playground.experiments.config import load_experiment
from ml_playground.experiments.executor import run_experiment
from ml_playground.experiments.runner import run_grid


def test_regression_runner_supports_holdout_and_diagnostics(tmp_path):
    path = tmp_path / "regression.csv"
    pl.DataFrame(
        {
            "years": list(range(1, 21)),
            "projects": [value % 4 for value in range(1, 21)],
            "salary": [2500 + 450 * value + 120 * (value % 4) for value in range(1, 21)],
        }
    ).write_csv(path)
    config = {
        "task": "regression",
        "data": {"path": str(path), "target": "salary", "random_state": 42},
        "cross_validation": {"method": "holdout", "test_size": 0.25, "random_state": 42},
        "model": {"name": "linear_regression", "params": {}},
        "metrics": ["mae", "rmse", "r2"],
    }

    result = run_experiment(config)

    assert result["task"] == "regression"
    assert {"mae", "rmse", "r2"}.issubset(result["metrics"])
    assert len(result["predictions"]["y_true"]) == result["n_test"]


def test_clustering_runner_returns_internal_metrics_and_labels(tmp_path):
    path = tmp_path / "clusters.csv"
    pl.DataFrame(
        {
            "income": [10.0, 11.0, 9.5, 70.0, 71.0, 69.5],
            "spending": [12.0, 11.5, 10.5, 75.0, 74.0, 76.0],
        }
    ).write_csv(path)
    config = {
        "task": "clustering",
        "data": {"path": str(path), "features": ["income", "spending"], "random_state": 42},
        "model": {"name": "kmeans", "params": {"n_clusters": 2, "random_state": 42, "n_init": 10}},
        "metrics": ["silhouette", "inertia", "cluster_count", "noise_ratio"],
    }

    result = run_experiment(config)

    assert result["task"] == "clustering"
    assert result["metrics"]["cluster_count"] == 2
    assert result["metrics"]["noise_ratio"] == 0.0
    assert len(result["predictions"]["labels"]) == 6


def test_loader_accepts_clustering_without_cross_validation_and_normalizes_views(tmp_path):
    dataset = tmp_path / "customers.csv"
    pl.DataFrame({"income": [1.0, 2.0, 8.0, 9.0], "score": [1.0, 2.0, 8.0, 9.0]}).write_csv(dataset)
    experiment = tmp_path / "cluster_contract"
    experiment.mkdir()
    (experiment / "experiment.yaml").write_text(
        """name: cluster_contract
task: clustering
data:
  path: customers.csv
  features: [income, score]
selection:
  primary_metric: silhouette
""",
        encoding="utf-8",
    )
    (experiment / "models.yaml").write_text(
        """models:
  - name: kmeans
    params:
      n_clusters: [2]
      random_state: [42]
""",
        encoding="utf-8",
    )
    (experiment / "preprocessing.yaml").write_text("{}\n", encoding="utf-8")
    (experiment / "metrics.yaml").write_text(
        """clustering:
  names: [silhouette, inertia, cluster_count]
""",
        encoding="utf-8",
    )
    (experiment / "views.yaml").write_text(
        """views:
  clustering:
    - name: cluster_size
      enabled: true
      scope: best
""",
        encoding="utf-8",
    )

    config = load_experiment(experiment, project_root=tmp_path)

    assert config["cross_validation"] == {"method": "none"}
    assert config["views"]["clustering"][0]["name"] == "cluster_size"
    assert config["selection"]["direction"] == "maximize"


def test_regression_and_clustering_reports_include_configured_views(tmp_path):
    regression_path = tmp_path / "regression.csv"
    pl.DataFrame(
        {"x": list(range(1, 13)), "target": [3 * value + 2 for value in range(1, 13)]}
    ).write_csv(regression_path)
    regression_config = {
        "experiment_name": "regression_views",
        "project_root": str(tmp_path),
        "task": "regression",
        "data": {"path": str(regression_path), "target": "target", "random_state": 42},
        "cross_validation": {"method": "holdout", "test_size": 0.25, "random_state": 42},
        "models": [{"name": "linear_regression", "params": {}}],
        "metrics": ["rmse", "r2"],
        "selection": {"primary_metric": "rmse", "direction": "minimize"},
        "views": {
            "common": [{"name": "model_comparison", "enabled": True, "scope": "candidates", "params": {}}],
            "regression": [
                {"name": "predicted_vs_actual", "enabled": True, "scope": "best", "params": {}},
                {"name": "residuals_vs_fitted", "enabled": True, "scope": "best", "params": {}},
            ],
        },
        "outputs": {"root": str(tmp_path / "reports"), "save_model": True, "save_predictions": True, "figures": True},
    }

    regression_result = run_grid(regression_config, write_reports=True)

    assert Path(regression_result["reports"]["residuals"]).is_file()
    assert Path(regression_result["reports"]["view_predicted_vs_actual"]).is_file()
    assert Path(regression_result["reports"]["view_residuals_vs_fitted"]).is_file()

    cluster_path = tmp_path / "clusters.csv"
    pl.DataFrame(
        {"x": [1.0, 1.2, 1.1, 8.0, 8.1, 8.2], "y": [1.0, 1.1, 1.2, 8.0, 8.2, 8.1]}
    ).write_csv(cluster_path)
    cluster_config = {
        "experiment_name": "cluster_views",
        "project_root": str(tmp_path),
        "task": "clustering",
        "data": {"path": str(cluster_path), "features": ["x", "y"], "random_state": 42},
        "cross_validation": {"method": "none"},
        "models": [{"name": "kmeans", "params": {"n_clusters": [2, 3], "random_state": [42], "n_init": [10]}}],
        "metrics": ["silhouette", "inertia", "cluster_count", "noise_ratio"],
        "selection": {"primary_metric": "silhouette", "direction": "maximize"},
        "views": {
            "common": [{"name": "model_comparison", "enabled": True, "scope": "candidates", "params": {}}],
            "clustering": [
                {"name": "elbow_curve", "enabled": True, "scope": "candidates", "params": {}},
                {"name": "silhouette_curve", "enabled": True, "scope": "candidates", "params": {}},
                {"name": "cluster_scatter", "enabled": True, "scope": "best", "params": {}},
                {"name": "cluster_size", "enabled": True, "scope": "best", "params": {}},
            ],
        },
        "outputs": {"root": str(tmp_path / "reports"), "save_model": True, "save_predictions": True, "figures": True},
    }

    cluster_result = run_grid(cluster_config, write_reports=True)

    assert Path(cluster_result["reports"]["cluster_sizes"]).is_file()
    assert Path(cluster_result["reports"]["view_elbow_curve"]).is_file()
    assert Path(cluster_result["reports"]["view_silhouette_curve"]).is_file()
    assert Path(cluster_result["reports"]["view_cluster_scatter"]).is_file()
    assert Path(cluster_result["reports"]["view_cluster_size"]).is_file()
