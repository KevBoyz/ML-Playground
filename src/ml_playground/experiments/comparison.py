import numpy as np

from ml_playground.evaluation.statistics import friedman_test, nemenyi_test


def _extract_table(results, metric):
    datasets = list(results.keys())
    models = list(results[datasets[0]].keys())
    table = np.array(
        [
            [results[d][m].get("metrics", {}).get(metric, 0) for m in models]
            for d in datasets
        ]
    )
    return table, datasets, models


def compare_results(results: dict, metric: str = "accuracy") -> dict:
    table, datasets, models = _extract_table(results, metric)
    friedman = friedman_test(*table.T)
    nemenyi = nemenyi_test(table)
    return {
        "metric": metric,
        "models": models,
        "datasets": datasets,
        "friedman": friedman,
        "nemenyi": nemenyi,
    }
