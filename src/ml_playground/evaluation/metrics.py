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
        if name in ("roc_auc", "log_loss") and y_score is not None:
            results[name] = fn(y_true, y_score)
        else:
            results[name] = fn(y_true, y_pred)
    return results
