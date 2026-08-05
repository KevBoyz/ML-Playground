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
    else:
        registry = REGRESSION

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
