import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_feature_importance(
    importances, feature_names=None, figsize=(8, 5), title="Feature importance"
):
    if feature_names is None:
        feature_names = [f"Feature {i}" for i in range(len(importances))]
    idx = np.argsort(importances)[::-1]
    sorted_imp = np.array(importances)[idx]
    sorted_names = np.array(feature_names)[idx]
    fig, ax = plt.subplots(figsize=figsize)
    ax.barh(range(len(sorted_imp)), sorted_imp, color="steelblue")
    ax.set_yticks(range(len(sorted_imp)))
    ax.set_yticklabels(sorted_names)
    ax.set_xlabel("Importance")
    ax.set_title(title)
    ax.invert_yaxis()
    fig.tight_layout()
    return fig
