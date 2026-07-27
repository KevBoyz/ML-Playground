import matplotlib

matplotlib.use("Agg")
import numpy as np

from ml_playground.visualization.barplots import plot_metric_comparison
from ml_playground.visualization.comparison import plot_cd_diagram
from ml_playground.visualization.heatmaps import (
    plot_confusion_matrix,
    plot_correlation_heatmap,
)
from ml_playground.visualization.importance import plot_feature_importance
from ml_playground.visualization.roc import plot_roc_curves


def test_plot_metric_comparison_returns_figure():
    results = {"rf": {"acc": 0.9}, "svm": {"acc": 0.85}}
    fig = plot_metric_comparison(results, metric="acc")
    assert fig is not None
    assert fig.axes[0].get_ylabel() == "acc"


def test_plot_metric_comparison_custom_names():
    results = {"rf": {"f1": 0.9}, "svm": {"f1": 0.85}}
    fig = plot_metric_comparison(results, metric="f1", model_names=["rf", "svm"])
    assert len(fig.axes[0].get_xticklabels()) == 2


def test_plot_confusion_matrix_returns_figure():
    cm = np.array([[10, 2], [3, 15]])
    fig = plot_confusion_matrix(cm, class_names=["A", "B"])
    assert fig is not None
    assert "Confusion matrix" in fig.axes[0].get_title()


def test_plot_confusion_matrix_default_names():
    cm = np.array([[5]])
    fig = plot_confusion_matrix(cm)
    assert fig is not None


def test_plot_correlation_heatmap_returns_figure():
    corr = np.array([[1.0, 0.5], [0.5, 1.0]])
    fig = plot_correlation_heatmap(corr, feature_names=["x", "y"])
    assert fig is not None


def test_plot_roc_curves_returns_figure():
    y_true = np.array([0, 0, 1, 1])
    y_scores = [np.array([0.1, 0.4, 0.6, 0.9])]
    fig = plot_roc_curves(y_true, y_scores, model_names=["Model A"])
    assert fig is not None
    assert "ROC" in fig.axes[0].get_title()


def test_plot_roc_curves_multiple_models():
    y_true = np.array([0, 0, 0, 1, 1, 1])
    scores = [
        np.array([0.3, 0.4, 0.5, 0.6, 0.7, 0.8]),
        np.array([0.2, 0.3, 0.4, 0.5, 0.6, 0.7]),
    ]
    fig = plot_roc_curves(y_true, scores, model_names=["A", "B"])
    assert len(fig.axes[0].get_lines()) == 3  # 2 curves + diagonal


def test_plot_feature_importance_returns_figure():
    imp = [0.5, 0.3, 0.2]
    fig = plot_feature_importance(imp, feature_names=["f1", "f2", "f3"])
    assert fig is not None
    assert "Importance" in fig.axes[0].get_xlabel()


def test_plot_feature_importance_sorted():
    imp = [0.2, 0.5, 0.3]
    fig = plot_feature_importance(imp, feature_names=["a", "b", "c"])
    yticklabels = [t.get_text() for t in fig.axes[0].get_yticklabels()]
    assert yticklabels[0] == "b"


def test_plot_cd_diagram_returns_figure():
    ranks = [1.5, 2.0, 2.5]
    fig = plot_cd_diagram(ranks, cd=0.8, model_names=["A", "B", "C"])
    assert fig is not None
    assert "Average Rank" in fig.axes[0].get_xlabel()


def test_plot_cd_diagram_aligns_models():
    ranks = [3.0, 1.0, 2.0]
    fig = plot_cd_diagram(ranks, cd=0.5)
    texts = [t.get_text() for t in fig.axes[0].texts]
    assert "M1" in texts
