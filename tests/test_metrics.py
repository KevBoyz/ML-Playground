import numpy as np
import pytest
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, r2_score

from ml_playground.evaluation.metrics import (
    CLASSIFICATION,
    REGRESSION,
    compute_metrics,
)


def test_classification_registry_has_all_config_metrics():
    expected = {
        "accuracy",
        "precision",
        "recall",
        "f1",
        "f1_macro",
        "f1_weighted",
        "roc_auc",
        "log_loss",
        "kappa",
        "mcc",
        "confusion_matrix",
    }
    assert expected.issubset(CLASSIFICATION.keys())


def test_regression_registry_has_all_config_metrics():
    expected = {"mae", "mse", "rmse", "r2", "mape", "max_error"}
    assert expected.issubset(REGRESSION.keys())


def test_compute_metrics_single():
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 1, 1, 1])
    result = compute_metrics(y_true, y_pred, ["accuracy"])
    assert result["accuracy"] == accuracy_score(y_true, y_pred)


def test_compute_metrics_multiple():
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 1, 1, 1])
    result = compute_metrics(y_true, y_pred, ["accuracy", "f1"])
    assert "accuracy" in result
    assert "f1" in result


def test_compute_metrics_f1_macro():
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 1, 1, 1])
    result = compute_metrics(y_true, y_pred, ["f1_macro"])
    expected = f1_score(y_true, y_pred, average="macro")
    assert result["f1_macro"] == expected


def test_compute_metrics_with_y_score():
    y_true = np.array([0, 0, 1, 1])
    y_score = np.array([0.1, 0.4, 0.6, 0.9])
    result = compute_metrics(y_true, y_pred=None, metrics=["roc_auc"], y_score=y_score)
    assert result["roc_auc"] == 1.0


def test_compute_metrics_unknown_metric_skipped():
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 1, 1, 1])
    result = compute_metrics(y_true, y_pred, ["nonexistent"])
    assert result == {}


def test_compute_metrics_regression():
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.1, 1.9, 3.2])
    result = compute_metrics(y_true, y_pred, ["r2", "rmse"], task="regression")
    assert "r2" in result
    assert "rmse" in result


def test_compute_metrics_regression_rmse():
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.0, 2.0, 3.0])
    result = compute_metrics(y_true, y_pred, ["rmse"], task="regression")
    assert result["rmse"] == 0.0


def test_compute_metrics_empty_metrics():
    y_true = np.array([0, 1])
    y_pred = np.array([0, 1])
    result = compute_metrics(y_true, y_pred, [])
    assert result == {}


def test_compute_metrics_confusion_matrix():
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 1, 1, 1])
    result = compute_metrics(y_true, y_pred, ["confusion_matrix"])
    cm = result["confusion_matrix"]
    assert cm.shape == (2, 2)
    assert cm[0, 0] == 1
    assert cm[1, 1] == 2
