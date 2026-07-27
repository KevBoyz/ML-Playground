import numpy as np
from scipy.stats import friedmanchisquare, rankdata, wilcoxon


def friedman_test(*samples):
    stat, p_value = friedmanchisquare(*samples)
    return {"statistic": float(stat), "p_value": float(p_value)}


def wilcoxon_test(a, b):
    stat, p_value = wilcoxon(a, b)
    return {"statistic": float(stat), "p_value": float(p_value)}


def nemenyi_test(data, alpha=0.05):
    arr = np.asarray(data, dtype=float)
    n_datasets, n_models = arr.shape
    avg_ranks = np.apply_along_axis(rankdata, 1, arr).mean(axis=0)
    import scipy.stats as stats

    q = stats.studentized_range.ppf(1 - alpha, n_models, np.inf)
    cd = q * np.sqrt(n_models * (n_models + 1) / (6.0 * n_datasets))
    return {
        "avg_ranks": avg_ranks.tolist(),
        "cd": float(cd),
        "n_models": int(n_models),
        "n_datasets": int(n_datasets),
    }
