import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_metric_comparison(
    results, metric="accuracy", model_names=None, figsize=(8, 5), title=None
):
    if model_names is None:
        model_names = list(results.keys())
    values = [results[m].get(metric, 0) for m in model_names]
    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(range(len(model_names)), values, color="steelblue")
    ax.set_xticks(range(len(model_names)))
    ax.set_xticklabels(model_names, rotation=45, ha="right")
    ax.set_ylabel(metric)
    ax.set_title(title or f"Model comparison — {metric}")
    ax.set_ylim(0, 1)
    for i, v in enumerate(values):
        ax.text(i, v + 0.01, f"{v:.3f}", ha="center", fontsize=9)
    fig.tight_layout()
    return fig
