"""Metric definitions and task-aware metric evaluation.

The public registries intentionally retain the short names used by the first
experiment contract.  ``compute_metrics`` additionally accepts metric specs so
that a run can state choices such as averaging, positive class and labels
instead of relying on an implicit sklearn default.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    calinski_harabasz_score,
    cohen_kappa_score,
    confusion_matrix,
    davies_bouldin_score,
    f1_score,
    log_loss,
    matthews_corrcoef,
    max_error,
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    root_mean_squared_error,
    silhouette_score,
)
from sklearn.preprocessing import label_binarize


class MetricUnavailableError(ValueError):
    """Raised when a requested metric needs an output the model did not expose."""


# These dictionaries are also a small public capability registry used by the
# experiment layer.  Keep aliases rather than removing the original names.
CLASSIFICATION = {
    "accuracy": accuracy_score,
    "balanced_accuracy": balanced_accuracy_score,
    "precision": precision_score,
    "recall": recall_score,
    "f1": f1_score,
    "precision_macro": lambda y, p: precision_score(
        y, p, average="macro", zero_division=0
    ),
    "precision_weighted": lambda y, p: precision_score(
        y, p, average="weighted", zero_division=0
    ),
    "recall_macro": lambda y, p: recall_score(
        y, p, average="macro", zero_division=0
    ),
    "recall_weighted": lambda y, p: recall_score(
        y, p, average="weighted", zero_division=0
    ),
    "f1_macro": lambda y, p: f1_score(y, p, average="macro", zero_division=0),
    "f1_weighted": lambda y, p: f1_score(y, p, average="weighted", zero_division=0),
    "roc_auc": roc_auc_score,
    "average_precision": average_precision_score,
    "pr_auc": average_precision_score,
    "log_loss": log_loss,
    "brier_score": brier_score_loss,
    "kappa": cohen_kappa_score,
    "mcc": matthews_corrcoef,
    "confusion_matrix": confusion_matrix,
}

REGRESSION = {
    "mae": mean_absolute_error,
    "mse": mean_squared_error,
    "rmse": root_mean_squared_error,
    "r2": r2_score,
    "mape": mean_absolute_percentage_error,
    "max_error": max_error,
}

CLUSTERING = {
    "silhouette",
    "calinski_harabasz",
    "davies_bouldin",
    "inertia",
    "cluster_count",
    "noise_ratio",
    "cluster_size_min",
    "cluster_size_max",
}


def normalize_metric_specs(metrics: Iterable[str | Mapping[str, Any]] | Mapping[str, Any]) -> list[dict[str, Any]]:
    """Normalize legacy metric names and explicit metric definitions.

    A spec can be written as ``{"name": "f1", "average": "macro"}`` or
    ``{"name": "f1", "id": "f1_churn", "params": {"pos_label": 1}}``.
    The returned id is the result key.  A duplicate id is rejected because it
    would make metric selection/reporting ambiguous.
    """

    if isinstance(metrics, Mapping):
        # Support the YAML shape ``metrics: {names: [...]}`` while keeping a
        # single metric spec unambiguous.
        if "names" in metrics or "metrics" in metrics:
            metrics = metrics.get("names", metrics.get("metrics", []))
        else:
            metrics = [metrics]

    specs: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for entry in metrics:
        if isinstance(entry, str):
            spec = {"name": entry, "id": entry, "params": {}}
        elif isinstance(entry, Mapping):
            name = entry.get("name", entry.get("metric"))
            if not isinstance(name, str) or not name:
                raise ValueError("Uma métrica declarada deve conter 'name'")
            params = dict(entry.get("params") or {})
            # Flat fields are useful in compact YAML.  Reserved fields retain
            # their structural meaning instead of being passed to sklearn.
            params.update(
                {
                    key: value
                    for key, value in entry.items()
                    if key not in {"name", "metric", "id", "key", "params"}
                }
            )
            metric_id = entry.get("id", entry.get("key", name))
            if not isinstance(metric_id, str) or not metric_id:
                raise ValueError("O id de uma métrica deve ser uma string não vazia")
            spec = {"name": name, "id": metric_id, "params": params}
        else:
            raise TypeError("Métricas devem ser strings ou mapas")
        if spec["id"] in seen_ids:
            raise ValueError(f"Id de métrica duplicado: {spec['id']}")
        seen_ids.add(spec["id"])
        specs.append(spec)
    return specs


def compute_metrics(
    y_true,
    y_pred,
    metrics,
    *,
    task: str = "classification",
    y_score=None,
    y_proba=None,
    class_labels=None,
):
    """Compute configured metrics using only outputs valid for each metric.

    Probability/ranking metrics deliberately fail when a classifier does not
    expose the required output.  Computing ROC-AUC from hard predictions makes
    a candidate look evaluable when it is not, so it is never used as a
    fallback.
    """

    if task == "classification":
        registry = CLASSIFICATION
    elif task == "regression":
        registry = REGRESSION
    else:
        raise ValueError(f"Tarefa de métricas não suportada: {task}")

    results = {}
    for spec in normalize_metric_specs(metrics):
        name = spec["name"]
        if name not in registry:
            # Preserve the legacy behaviour: unsupported names from an
            # unvalidated direct API call are ignored.
            continue
        if task == "classification":
            results[spec["id"]] = _compute_classification_metric(
                name,
                y_true,
                y_pred,
                spec["params"],
                y_score=y_score,
                y_proba=y_proba,
                class_labels=class_labels,
            )
        else:
            results[spec["id"]] = _compute_regression_metric(
                name,
                y_true,
                y_pred,
                spec["params"],
            )
    return results


def _compute_classification_metric(
    name: str,
    y_true,
    y_pred,
    params: Mapping[str, Any],
    *,
    y_score,
    y_proba,
    class_labels,
):
    params = dict(params)
    if name == "roc_auc":
        if y_score is None:
            raise MetricUnavailableError("roc_auc exige score contínuo ou predict_proba")
        return _roc_auc(y_true, y_score, class_labels=class_labels, **params)
    if name in {"average_precision", "pr_auc"}:
        if y_score is None:
            raise MetricUnavailableError(f"{name} exige score contínuo ou predict_proba")
        return _average_precision(y_true, y_score, class_labels=class_labels, **params)
    if name == "log_loss":
        if y_proba is None:
            raise MetricUnavailableError("log_loss exige predict_proba")
        return log_loss(y_true, y_proba, labels=class_labels, **params)
    if name == "brier_score":
        if y_proba is None and y_score is None:
            raise MetricUnavailableError("brier_score exige predict_proba ou score contínuo")
        scores = y_proba if y_proba is not None else y_score
        positive_scores = _binary_scores(scores, class_labels, params.pop("pos_label", None))
        return brier_score_loss(y_true, positive_scores, **params)
    if name in {"precision", "recall", "f1"} or name.startswith(
        ("precision_", "recall_", "f1_")
    ):
        return _averaged_classification_metric(name, y_true, y_pred, params)
    return CLASSIFICATION[name](y_true, y_pred, **params)


def _averaged_classification_metric(name: str, y_true, y_pred, params: dict[str, Any]):
    base_name, _, declared_average = name.partition("_")
    scorer = {
        "precision": precision_score,
        "recall": recall_score,
        "f1": f1_score,
    }[base_name]
    if declared_average:
        params.setdefault("average", declared_average)
    # Existing short names were binary when possible and weighted for a
    # multiclass target.  Keep that compatibility while allowing config to
    # declare a stable averaging rule.
    explicit_average = "average" in params
    params.setdefault("zero_division", 0)
    try:
        return scorer(y_true, y_pred, **params)
    except ValueError:
        if declared_average or explicit_average:
            raise
        params["average"] = "weighted"
        params.pop("pos_label", None)
        return scorer(y_true, y_pred, **params)


def _compute_regression_metric(name: str, y_true, y_pred, params: Mapping[str, Any]):
    return REGRESSION[name](y_true, y_pred, **dict(params))


def _roc_auc(y_true, y_score, *, class_labels=None, **params):
    scores = np.asarray(y_score)
    labels = _labels_for_scores(y_true, class_labels)
    if scores.ndim == 1:
        return roc_auc_score(y_true, scores, **params)
    if scores.ndim != 2:
        raise ValueError("roc_auc recebeu scores com dimensão inválida")
    if scores.shape[1] == 2:
        positive_scores = _binary_scores(scores, labels, params.pop("pos_label", None))
        return roc_auc_score(y_true, positive_scores, **params)
    params.setdefault("multi_class", "ovr")
    params.setdefault("average", "macro")
    if labels is not None:
        params.setdefault("labels", labels)
    return roc_auc_score(y_true, scores, **params)


def _average_precision(y_true, y_score, *, class_labels=None, **params):
    scores = np.asarray(y_score)
    labels = _labels_for_scores(y_true, class_labels)
    if scores.ndim == 1:
        return average_precision_score(y_true, scores, **params)
    if scores.ndim != 2:
        raise ValueError("average_precision recebeu scores com dimensão inválida")
    if scores.shape[1] == 2:
        positive_scores = _binary_scores(scores, labels, params.pop("pos_label", None))
        return average_precision_score(y_true, positive_scores, **params)
    if labels is None:
        labels = np.unique(np.asarray(y_true))
    y_binary = label_binarize(y_true, classes=labels)
    params.setdefault("average", "macro")
    return average_precision_score(y_binary, scores, **params)


def _labels_for_scores(y_true, class_labels):
    if class_labels is None:
        return np.unique(np.asarray(y_true))
    labels = np.asarray(class_labels)
    if labels.ndim != 1:
        raise ValueError("class_labels deve ter uma dimensão")
    return labels


def _binary_scores(scores, labels, pos_label):
    values = np.asarray(scores)
    if values.ndim == 1:
        return values
    if values.ndim != 2 or values.shape[1] != 2:
        raise ValueError("A métrica binária exige um vetor ou duas colunas de score")
    if labels is None or len(labels) != 2:
        index = 1
    elif pos_label is None:
        index = 1
    else:
        matches = np.flatnonzero(labels == pos_label)
        if len(matches) != 1:
            raise ValueError(f"pos_label não foi encontrado nas classes: {pos_label!r}")
        index = int(matches[0])
    return values[:, index]


def compute_clustering_metrics(X, labels, metrics, *, inertia=None):
    """Compute internal clustering metrics and explain unavailable values."""

    metric_names = [spec["name"] for spec in normalize_metric_specs(metrics)]
    unknown = set(metric_names) - CLUSTERING
    if unknown:
        raise ValueError(f"Métricas de clusterização não suportadas: {sorted(unknown)}")

    values = {}
    notes = {}
    label_array = np.asarray(labels)
    if label_array.ndim != 1:
        raise ValueError("Os labels de clusterização devem ter uma dimensão")
    if len(label_array) == 0:
        raise ValueError("Clusterização não retornou labels")

    non_noise = label_array != -1
    effective_labels = label_array[non_noise]
    clusters, counts = np.unique(effective_labels, return_counts=True)
    cluster_count = len(clusters)
    if "cluster_count" in metric_names:
        values["cluster_count"] = cluster_count
    if "noise_ratio" in metric_names:
        values["noise_ratio"] = float(1 - non_noise.mean())
    if "cluster_size_min" in metric_names:
        if cluster_count:
            values["cluster_size_min"] = int(counts.min())
        else:
            notes["cluster_size_min"] = "Nenhum cluster não-ruído foi encontrado"
    if "cluster_size_max" in metric_names:
        if cluster_count:
            values["cluster_size_max"] = int(counts.max())
        else:
            notes["cluster_size_max"] = "Nenhum cluster não-ruído foi encontrado"
    if "inertia" in metric_names:
        if inertia is None:
            notes["inertia"] = "O estimador não expõe inertia_"
        else:
            values["inertia"] = float(inertia)

    internal_metrics = {"silhouette", "calinski_harabasz", "davies_bouldin"}
    requested_internal = internal_metrics.intersection(metric_names)
    if not requested_internal:
        return values, notes
    if cluster_count < 2 or cluster_count >= len(effective_labels):
        reason = "A métrica exige entre 2 e n-1 clusters não-ruído"
        notes.update({metric: reason for metric in requested_internal})
        return values, notes

    dense_X = _dense_array(X)[non_noise]
    if "silhouette" in requested_internal:
        values["silhouette"] = float(silhouette_score(dense_X, effective_labels))
    if "calinski_harabasz" in requested_internal:
        values["calinski_harabasz"] = float(calinski_harabasz_score(dense_X, effective_labels))
    if "davies_bouldin" in requested_internal:
        values["davies_bouldin"] = float(davies_bouldin_score(dense_X, effective_labels))
    return values, notes


def _dense_array(values):
    return values.toarray() if hasattr(values, "toarray") else np.asarray(values)
