import numpy as np

from ml_playground.experiments.comparison import compare_results


def test_compare_results():
    results = {
        "d1": {
            "rf": {"metrics": {"accuracy": 0.9}},
            "svm": {"metrics": {"accuracy": 0.85}},
            "knn": {"metrics": {"accuracy": 0.8}},
        },
        "d2": {
            "rf": {"metrics": {"accuracy": 0.85}},
            "svm": {"metrics": {"accuracy": 0.8}},
            "knn": {"metrics": {"accuracy": 0.75}},
        },
        "d3": {
            "rf": {"metrics": {"accuracy": 0.95}},
            "svm": {"metrics": {"accuracy": 0.9}},
            "knn": {"metrics": {"accuracy": 0.85}},
        },
    }
    comp = compare_results(results, metric="accuracy")
    assert comp["metric"] == "accuracy"
    assert comp["models"] == ["rf", "svm", "knn"]
    assert comp["datasets"] == ["d1", "d2", "d3"]
    assert "friedman" in comp
    assert "nemenyi" in comp
    assert comp["friedman"]["p_value"] > 0


def test_compare_results_different_metric():
    results = {
        "d1": {
            "rf": {"metrics": {"f1": 0.9}},
            "svm": {"metrics": {"f1": 0.8}},
            "knn": {"metrics": {"f1": 0.7}},
        },
        "d2": {
            "rf": {"metrics": {"f1": 0.85}},
            "svm": {"metrics": {"f1": 0.75}},
            "knn": {"metrics": {"f1": 0.65}},
        },
        "d3": {
            "rf": {"metrics": {"f1": 0.95}},
            "svm": {"metrics": {"f1": 0.85}},
            "knn": {"metrics": {"f1": 0.75}},
        },
    }
    comp = compare_results(results, metric="f1")
    assert comp["metric"] == "f1"
    assert comp["datasets"] == ["d1", "d2", "d3"]
