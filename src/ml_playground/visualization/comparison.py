import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_cd_diagram(
    avg_ranks, cd, model_names=None, figsize=(8, 4), title="Critical Difference"
):
    if model_names is None:
        model_names = [f"M{i}" for i in range(len(avg_ranks))]
    n = len(avg_ranks)
    idx = np.argsort(avg_ranks)
    sorted_ranks = np.array(avg_ranks)[idx]
    sorted_names = np.array(model_names)[idx]
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(sorted_ranks[0] - 0.5, sorted_ranks[-1] + 0.5)
    ax.set_ylim(-1, n + 1)
    for i, (rank, name) in enumerate(zip(sorted_ranks, sorted_names)):
        ax.plot(rank, i, "o", color="steelblue", markersize=10)
        ax.text(rank - 0.15, i + 0.3, name, ha="center", va="bottom", fontsize=10)
    ax.plot([sorted_ranks[0] - cd, sorted_ranks[0]], [n, n], "k-", lw=2)
    ax.text(
        sorted_ranks[0] - cd / 2,
        n + 0.3,
        f"CD = {cd:.3f}",
        ha="center",
        fontsize=9,
    )
    ax.set_xlabel("Average Rank")
    ax.set_title(title)
    ax.yaxis.set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    fig.tight_layout()
    return fig
