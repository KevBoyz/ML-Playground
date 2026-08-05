"""Task-aware execution of one model configuration."""

from __future__ import annotations

import time

import numpy as np
from sklearn.model_selection import (
    KFold,
    RepeatedKFold,
    RepeatedStratifiedKFold,
    StratifiedKFold,
    train_test_split,
)

from ml_playground.data.loader import auto_read
from ml_playground.evaluation.metrics import (
    CLASSIFICATION,
    REGRESSION,
    compute_clustering_metrics,
    compute_metrics,
)
from ml_playground.models import get_model
from ml_playground.preprocessing.pipelines import build_model_pipeline


def run_experiment(config: dict) -> dict:
    """Run one model configuration for a supported task."""

    started = time.perf_counter()
    task = config.get("task", "classification")
    if task == "clustering":
        result = _run_clustering(config)
    elif task in {"classification", "regression"}:
        result = _run_supervised(config, task)
    else:
        raise ValueError(f"Tarefa não suportada: {task}")

    model_config = config["model"]
    result["status"] = "success"
    result["duration_seconds"] = time.perf_counter() - started
    result["model"] = model_config["name"]
    result["params"] = model_config.get("params", {})
    result["task"] = task
    return result


def fit_full_pipeline(config: dict):
    """Fit the selected pipeline on all configured rows for persistence."""

    task = config.get("task", "classification")
    X, y, _ = _load_dataset(config, task)
    return _fit_pipeline(config, X, y)


def _run_supervised(config: dict, task: str) -> dict:
    X, y, row_ids = _load_dataset(config, task)
    cv_config = dict(config.get("cross_validation") or config.get("validation") or {})
    cv_method = cv_config.get("method", "holdout")
    random_state = cv_config.get("random_state", config["data"].get("random_state", 42))
    metric_names = config.get("metrics", [])

    if cv_method == "holdout":
        return _run_holdout(X, y, row_ids, config, metric_names, random_state, task, cv_config)
    return _run_cv(X, y, row_ids, config, metric_names, cv_config, random_state, task)


def _run_holdout(X, y, row_ids, config, metric_names, random_state, task, cv_config):
    data_config = config["data"]
    test_size = cv_config.get("test_size", data_config.get("test_size", 0.2))
    stratified = cv_config.get("stratified", data_config.get("stratified", task == "classification"))
    stratify = y if task == "classification" and stratified else None

    X_train, X_test, y_train, y_test, _, test_row_ids = train_test_split(
        X,
        y,
        row_ids,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )
    pipeline = _fit_pipeline(config, X_train, y_train)
    y_pred, y_score, y_proba = _predict_outputs(pipeline, X_test)
    metrics = compute_metrics(
        y_test,
        y_pred,
        metric_names,
        task=task,
        y_score=y_score,
        y_proba=y_proba,
    )
    return {
        "metrics": _scalar_metrics(metrics),
        "predictions": {
            "row_ids": _json_values(test_row_ids),
            "y_true": _json_values(y_test),
            "y_pred": _json_values(y_pred),
            "y_score": _json_values(y_score),
        },
        "pipeline": pipeline,
        "n_train": len(y_train),
        "n_test": len(y_test),
    }


def _run_cv(X, y, row_ids, config, metric_names, cv_config, random_state, task):
    splitter = _build_splitter(cv_config, config["data"], random_state, task)
    registry = CLASSIFICATION if task == "classification" else REGRESSION
    cv_scores = {metric: [] for metric in metric_names if metric in registry}
    folds = []

    for fold_number, (train_idx, test_idx) in enumerate(splitter.split(X, y), start=1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        pipeline = _fit_pipeline(config, X_train, y_train)
        y_pred, y_score, y_proba = _predict_outputs(pipeline, X_test)
        fold_metrics = compute_metrics(
            y_test,
            y_pred,
            metric_names,
            task=task,
            y_score=y_score,
            y_proba=y_proba,
        )
        for metric, value in fold_metrics.items():
            if metric in cv_scores and np.isscalar(value):
                cv_scores[metric].append(float(value))
        folds.append(
            {
                "fold": fold_number,
                "n_train": len(train_idx),
                "n_test": len(test_idx),
                "metrics": _scalar_metrics(fold_metrics),
                "predictions": {
                    "row_ids": _json_values(row_ids[test_idx]),
                    "y_true": _json_values(y_test),
                    "y_pred": _json_values(y_pred),
                    "y_score": _json_values(y_score),
                },
            }
        )

    averages = {metric: float(np.mean(values)) for metric, values in cv_scores.items() if values}
    stds = {metric: float(np.std(values, ddof=0)) for metric, values in cv_scores.items() if values}
    return {
        "metrics": averages,
        "metric_std": stds,
        "cv_scores": {metric: [float(value) for value in values] for metric, values in cv_scores.items()},
        "folds": folds,
    }


def _run_clustering(config: dict) -> dict:
    X, _, row_ids = _load_dataset(config, "clustering")
    pipeline = _fit_pipeline(config, X, None)
    estimator = pipeline.named_steps["model"]
    labels = getattr(estimator, "labels_", None)
    if labels is None:
        raise ValueError(f"O modelo {config['model']['name']} não expôs labels_")

    transformed = pipeline.named_steps["preprocessing"].transform(X)
    metrics, notes = compute_clustering_metrics(
        transformed,
        labels,
        config.get("metrics", []),
        inertia=getattr(estimator, "inertia_", None),
    )
    return {
        "metrics": metrics,
        "metric_notes": notes,
        "predictions": {
            "row_ids": _json_values(row_ids),
            "labels": _json_values(labels),
        },
        "pipeline": pipeline,
        "n_samples": len(X),
    }


def _load_dataset(config: dict, task: str):
    frame = auto_read(config["data"]["path"])
    columns = set(frame.columns)
    data_config = config["data"]
    target_col = data_config.get("target")
    if task in {"classification", "regression"} and target_col not in columns:
        raise ValueError(f"Target não encontrada no dataset: {target_col}")

    features = data_config.get("features")
    if features is None:
        features = [column for column in frame.columns if column != target_col]
    missing = sorted(set(features) - columns)
    if missing:
        raise ValueError(f"Features não encontradas no dataset: {missing}")
    if not features:
        raise ValueError("O experimento deve declarar ao menos uma feature")

    X = frame.select(features).to_pandas()
    y = frame[target_col].to_numpy() if task in {"classification", "regression"} else None
    return X, y, np.arange(len(X))


def _fit_pipeline(config, X_train, y_train=None):
    model_config = config["model"]
    model = get_model(model_config["name"], model_config.get("params", {}))
    pipeline = build_model_pipeline(config.get("preprocessing", {}), X_train, model)
    if y_train is None:
        pipeline.fit(X_train)
    else:
        pipeline.fit(X_train, y_train)
    return pipeline


def _predict_outputs(pipeline, X):
    y_pred = pipeline.predict(X)
    y_proba = None
    y_score = None
    if hasattr(pipeline, "predict_proba"):
        y_proba = pipeline.predict_proba(X)
        y_score = y_proba
    elif hasattr(pipeline, "decision_function"):
        y_score = pipeline.decision_function(X)
    return y_pred, y_score, y_proba


def _build_splitter(cv_config, data_config, random_state, task):
    method = cv_config.get("method", "kfold")
    n_splits = int(cv_config.get("n_splits", 5))
    shuffle = bool(cv_config.get("shuffle", True))
    seed = random_state if shuffle else None
    stratified = data_config.get("stratified", task == "classification")

    if method == "kfold" and task == "classification" and stratified:
        method = "stratified_kfold"
    if method == "kfold":
        return KFold(n_splits=n_splits, shuffle=shuffle, random_state=seed)
    if method == "stratified_kfold":
        return StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=seed)
    if method == "repeated_kfold":
        return RepeatedKFold(
            n_splits=n_splits,
            n_repeats=int(cv_config.get("n_repeats", 3)),
            random_state=random_state,
        )
    if method == "repeated_stratified_kfold":
        return RepeatedStratifiedKFold(
            n_splits=n_splits,
            n_repeats=int(cv_config.get("n_repeats", 3)),
            random_state=random_state,
        )
    raise ValueError(f"Método de validação não suportado: {method}")


def _scalar_metrics(metrics):
    return {
        name: float(value) if np.isscalar(value) else value.tolist()
        for name, value in metrics.items()
    }


def _json_values(values):
    if values is None:
        return None
    return np.asarray(values).tolist()


def _get_y_score(model, X):
    """Backward-compatible score helper used by existing callers."""

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X)
        return probabilities[:, 1] if probabilities.ndim == 2 and probabilities.shape[1] == 2 else probabilities
    if hasattr(model, "decision_function"):
        return model.decision_function(X)
    return None
