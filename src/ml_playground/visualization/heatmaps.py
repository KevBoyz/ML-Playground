import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_confusion_matrix(
    cm, class_names=None, figsize=(6, 5), title="Confusion matrix"
):
    if class_names is None:
        class_names = [str(i) for i in range(cm.shape[0])]
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax)
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center",
                color="white" if cm[i, j] > cm.max() / 2 else "black",
            )
    fig.tight_layout()
    return fig


def plot_correlation_heatmap(
    corr_matrix, feature_names=None, figsize=(10, 8), title="Correlation matrix"
):
    if feature_names is None:
        feature_names = [str(i) for i in range(corr_matrix.shape[0])]
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(corr_matrix, interpolation="nearest", cmap="RdBu_r", vmin=-1, vmax=1)
    fig.colorbar(im, ax=ax)
    ax.set_xticks(range(len(feature_names)))
    ax.set_yticks(range(len(feature_names)))
    ax.set_xticklabels(feature_names, rotation=45, ha="right")
    ax.set_yticklabels(feature_names)
    ax.set_title(title)
    fig.tight_layout()
    return fig
