import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve


def plot_roc_curves(
    y_true, y_scores_list, model_names=None, figsize=(8, 6), title="ROC curves"
):
    if model_names is None:
        model_names = [f"Model {i}" for i in range(len(y_scores_list))]
    fig, ax = plt.subplots(figsize=figsize)
    for y_score, name in zip(y_scores_list, model_names):
        fpr, tpr, _ = roc_curve(y_true, y_score)
        auc = np.trapezoid(tpr, fpr)
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend(loc="lower right")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.tight_layout()
    return fig
