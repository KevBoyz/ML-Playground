"""Scaffold self-contained experiments from task-aware starter contracts."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml


_SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")


def initialize_experiment(
    experiment_dir: str | Path,
    *,
    task: str,
    name: str | None = None,
    overwrite: bool = False,
) -> list[Path]:
    """Create the coordinated YAML files required for a new experiment."""

    directory = Path(experiment_dir)
    experiment_name = name or directory.name
    if not _SAFE_NAME.fullmatch(experiment_name):
        raise ValueError("O nome do experimento deve conter apenas letras, números, '_' ou '-'")
    if task not in _TEMPLATES:
        raise ValueError(f"Tarefa sem template: {task!r}. Disponíveis: {sorted(_TEMPLATES)}")

    documents = _documents_for(task, experiment_name)
    existing = [directory / filename for filename in documents if (directory / filename).exists()]
    if existing and not overwrite:
        formatted = ", ".join(str(path) for path in existing)
        raise FileExistsError(f"O template recusou sobrescrever arquivos existentes: {formatted}")

    directory.mkdir(parents=True, exist_ok=True)
    created = []
    for filename, document in documents.items():
        path = directory / filename
        path.write_text(
            yaml.safe_dump(document, allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )
        created.append(path)
    return created


def _documents_for(task: str, name: str) -> dict[str, dict[str, Any]]:
    documents = _TEMPLATES[task](name)
    return {filename: document for filename, document in documents.items() if document is not None}


def _supervised_common(name: str, task: str, metrics: list[str], primary: str) -> dict[str, dict[str, Any]]:
    return {
        "experiment.yaml": {
            "name": name,
            "task": task,
            "data": {
                "path": "data/processed/dataset.csv",
                "read_options": {},
                "target": "target",
                "id_column": "record_id",
                "metadata_columns": [],
                "features": {"numeric": ["feature_numeric"], "categorical": ["feature_category"]},
                "schema": {"mode": "strict"},
                "random_state": 42,
            },
            "evaluation": {
                "protocol": "nested_cv",
                "splitter": {"name": "stratified_kfold" if task == "classification" else "kfold", "n_splits": 5},
            },
            "outputs": {"root": "reports", "save_model": True, "save_predictions": True, "figures": True},
            "selection": {"primary_metric": primary, "direction": "maximize" if primary != "rmse" else "minimize", "tie_breakers": ["metric_std", "candidate_id"]},
            "provenance": {"recipe_ref": "recipes/prepare_dataset.py", "recipe_revision": "untracked", "source_description": "Tabela preparada pelo usuário"},
            "search": {"strategy": "grid", "max_candidates": 24},
            "execution": {"n_jobs": 1, "on_candidate_error": "continue"},
        },
        "models.yaml": _classification_models() if task == "classification" else _regression_models(),
        "preprocessing.yaml": _preprocessing(),
        "metrics.yaml": {task: {"names": metrics, "primary": primary}},
        "cross_validation.yaml": {"method": "stratified_kfold" if task == "classification" else "kfold", "n_splits": 5, "shuffle": True, "random_state": 42},
        "views.yaml": {"views": {"common": [{"name": "model_comparison", "enabled": True, "scope": "candidates"}], task: _supervised_views(task)}},
    }


def _classification_template(name: str) -> dict[str, dict[str, Any]]:
    return _supervised_common(
        name,
        "classification",
        ["accuracy", "f1_macro", "precision_macro", "recall_macro", "roc_auc"],
        "f1_macro",
    )


def _regression_template(name: str) -> dict[str, dict[str, Any]]:
    return _supervised_common(name, "regression", ["mae", "rmse", "r2"], "rmse")


def _clustering_template(name: str) -> dict[str, dict[str, Any]]:
    return {
        "experiment.yaml": {
            "name": name,
            "task": "clustering",
            "data": {
                "path": "data/processed/dataset.csv",
                "read_options": {},
                "id_column": "record_id",
                "metadata_columns": [],
                "features": {"numeric": ["feature_numeric_a", "feature_numeric_b"]},
                "schema": {"mode": "strict"},
                "random_state": 42,
            },
            "outputs": {"root": "reports", "save_model": True, "save_predictions": True, "figures": True},
            "selection": {"primary_metric": "silhouette", "direction": "maximize", "tie_breakers": ["candidate_id"]},
            "provenance": {"recipe_ref": "recipes/prepare_dataset.py", "recipe_revision": "untracked", "source_description": "Tabela preparada pelo usuário"},
            "search": {"strategy": "grid", "max_candidates": 24},
            "execution": {"n_jobs": 1, "on_candidate_error": "continue"},
        },
        "models.yaml": {
            "models": [
                {"name": "kmeans", "params": {"n_clusters": [2, 3, 4], "random_state": [42]}},
                {"name": "dbscan", "params": {"eps": [0.3, 0.5], "min_samples": [5]}},
            ]
        },
        "preprocessing.yaml": {"numeric": {"steps": [{"name": "scaling", "category": "scaling", "method": "standard"}]}},
        "metrics.yaml": {"clustering": {"names": ["silhouette", "davies_bouldin", "cluster_count", "noise_ratio"], "primary": "silhouette"}},
        "views.yaml": {"views": {"common": [{"name": "model_comparison", "enabled": True, "scope": "candidates"}], "clustering": [{"name": "cluster_scatter", "enabled": True, "scope": "best"}, {"name": "cluster_size", "enabled": True, "scope": "best"}]}},
    }


def _classification_models() -> dict[str, Any]:
    return {
        "models": [
            {"name": "dummy_classifier", "params": {"strategy": ["prior"]}},
            {"name": "logistic_regression", "params": {"C": [0.1, 1.0], "max_iter": [1000]}},
            {"name": "decision_tree", "params": {"max_depth": [3, 6], "random_state": [42]}},
        ]
    }


def _regression_models() -> dict[str, Any]:
    return {
        "models": [
            {"name": "dummy_regressor", "params": {"strategy": ["mean"]}},
            {"name": "ridge", "params": {"alpha": [0.1, 1.0]}},
            {"name": "random_forest_regressor", "params": {"n_estimators": [200], "random_state": [42], "n_jobs": [1]}},
        ]
    }


def _preprocessing() -> dict[str, Any]:
    return {
        "numeric": {"steps": [{"name": "imputation", "category": "imputation", "method": "median"}, {"name": "scaling", "category": "scaling", "method": "standard"}]},
        "categorical": {"steps": [{"name": "imputation", "category": "imputation", "method": "mode"}, {"name": "encoding", "category": "encoding", "method": "onehot"}]},
    }


def _supervised_views(task: str) -> list[dict[str, Any]]:
    if task == "classification":
        return [{"name": "confusion_matrix", "enabled": True, "scope": "best"}, {"name": "roc_curve", "enabled": True, "scope": "best"}]
    return [{"name": "predicted_vs_actual", "enabled": True, "scope": "best"}, {"name": "residuals_vs_fitted", "enabled": True, "scope": "best"}]


_TEMPLATES = {
    "classification": _classification_template,
    "regression": _regression_template,
    "clustering": _clustering_template,
}
