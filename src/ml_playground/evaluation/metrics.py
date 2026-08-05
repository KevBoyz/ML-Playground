import numpy as np
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
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
)
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score

CLASSIFICATION = {
    "accuracy": accuracy_score,
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
    "f1_macro": lambda y, p: f1_score(y, p, average="macro"),
    "f1_weighted": lambda y, p: f1_score(y, p, average="weighted"),
    "roc_auc": roc_auc_score,
    "log_loss": log_loss,
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


def compute_metrics(
    y_true,
    y_pred,
    metrics,
    *,
    task="classification",
    y_score=None,
    y_proba=None,
):
    if task == "classification":
        registry = CLASSIFICATION
    elif task == "regression":
        registry = REGRESSION
    else:
        raise ValueError(f"Tarefa de métricas não suportada: {task}")

    results = {}
    for name in metrics:
        fn = registry.get(name)
        if fn is None:
            continue
        if name == "roc_auc" and y_score is not None:
            results[name] = _roc_auc(y_true, y_score)
        elif name == "log_loss" and y_proba is not None:
            results[name] = log_loss(y_true, y_proba)
        else:
            try:
                results[name] = fn(y_true, y_pred)
            except ValueError:
                if name in {"precision", "recall", "f1"}:
                    results[name] = {
                        "precision": precision_score,
                        "recall": recall_score,
                        "f1": f1_score,
                    }[name](y_true, y_pred, average="weighted", zero_division=0)
                else:
                    raise
    return results


def _roc_auc(y_true, y_score):
    scores = np.asarray(y_score)
    if scores.ndim == 1 or scores.shape[1] == 2:
        return roc_auc_score(y_true, scores if scores.ndim == 1 else scores[:, 1])
    return roc_auc_score(y_true, scores, multi_class="ovr", average="macro")


def compute_clustering_metrics(X, labels, metrics, *, inertia=None):
    """Compute internal clustering metrics and explain unavailable values."""

    unknown = set(metrics) - CLUSTERING
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
    if "cluster_count" in metrics:
        values["cluster_count"] = cluster_count
    if "noise_ratio" in metrics:
        values["noise_ratio"] = float(1 - non_noise.mean())
    if "cluster_size_min" in metrics:
        if cluster_count:
            values["cluster_size_min"] = int(counts.min())
        else:
            notes["cluster_size_min"] = "Nenhum cluster não-ruído foi encontrado"
    if "cluster_size_max" in metrics:
        if cluster_count:
            values["cluster_size_max"] = int(counts.max())
        else:
            notes["cluster_size_max"] = "Nenhum cluster não-ruído foi encontrado"
    if "inertia" in metrics:
        if inertia is None:
            notes["inertia"] = "O estimador não expõe inertia_"
        else:
            values["inertia"] = float(inertia)

    internal_metrics = {"silhouette", "calinski_harabasz", "davies_bouldin"}
    requested_internal = internal_metrics.intersection(metrics)
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
