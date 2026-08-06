"""Model registry and task compatibility metadata."""

from __future__ import annotations

from typing import Any


_registry: dict[str, type] | None = None


MODEL_METADATA: dict[str, dict[str, Any]] = {
    "dummy_classifier": {
        "tasks": {"classification"},
        "predict_proba": True,
        "baseline": True,
    },
    "random_forest": {"tasks": {"classification"}, "importance": True},
    "gaussian_nb": {"tasks": {"classification"}, "predict_proba": True},
    "svm": {
        "tasks": {"classification"},
        "decision_function": True,
        "predict_proba": "parameter",
    },
    "knn": {"tasks": {"classification"}},
    "decision_tree": {"tasks": {"classification"}, "importance": True, "tree": True},
    "logistic_regression": {
        "tasks": {"classification"},
        "predict_proba": True,
        "decision_function": True,
        "importance": True,
    },
    "logistic": {
        "tasks": {"classification"},
        "alias_for": "logistic_regression",
        "predict_proba": True,
        "decision_function": True,
        "importance": True,
    },
    "xgboost": {"tasks": {"classification"}, "importance": True, "optional": "xgboost"},
    "lightgbm": {"tasks": {"classification"}, "importance": True, "optional": "lightgbm"},
    "dummy_regressor": {"tasks": {"regression"}},
    "linear_regression": {"tasks": {"regression"}, "importance": True},
    "ridge": {"tasks": {"regression"}, "importance": True},
    "lasso": {"tasks": {"regression"}, "importance": True},
    "elastic_net": {"tasks": {"regression"}, "importance": True},
    "random_forest_regressor": {"tasks": {"regression"}, "importance": True},
    "kmeans": {"tasks": {"clustering"}, "cluster_centers": True, "predict": True},
    "dbscan": {"tasks": {"clustering"}, "noise": True},
    "agglomerative": {"tasks": {"clustering"}},
}


def _load() -> dict[str, type]:
    global _registry
    if _registry is not None:
        return _registry

    from sklearn.cluster import AgglomerativeClustering, DBSCAN, KMeans
    from sklearn.dummy import DummyClassifier, DummyRegressor
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, LogisticRegression, Ridge
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.naive_bayes import GaussianNB
    from sklearn.svm import SVC
    from sklearn.tree import DecisionTreeClassifier

    registry = {
        "dummy_classifier": DummyClassifier,
        "random_forest": RandomForestClassifier,
        "gaussian_nb": GaussianNB,
        "svm": SVC,
        "knn": KNeighborsClassifier,
        "decision_tree": DecisionTreeClassifier,
        "logistic_regression": LogisticRegression,
        "logistic": LogisticRegression,
        "dummy_regressor": DummyRegressor,
        "linear_regression": LinearRegression,
        "ridge": Ridge,
        "lasso": Lasso,
        "elastic_net": ElasticNet,
        "random_forest_regressor": RandomForestRegressor,
        "kmeans": KMeans,
        "dbscan": DBSCAN,
        "agglomerative": AgglomerativeClustering,
    }

    try:
        from xgboost import XGBClassifier

        registry["xgboost"] = XGBClassifier
    except ImportError:
        pass
    try:
        from lightgbm import LGBMClassifier

        registry["lightgbm"] = LGBMClassifier
    except ImportError:
        pass

    _registry = registry
    return _registry


def get_model(name: str, params: dict | None = None):
    """Create a configured model instance from a registered name."""

    registry = _load()
    entry = registry.get(name)
    if entry is None:
        raise ValueError(f"Modelo '{name}' não encontrado. Disponíveis: {list(registry)}")
    return entry(**(params or {}))


def get_model_metadata(name: str) -> dict[str, Any]:
    """Return declarative compatibility information without instantiating a model."""

    metadata = MODEL_METADATA.get(name)
    if metadata is None:
        raise ValueError(f"Modelo '{name}' não encontrado. Disponíveis: {sorted(MODEL_METADATA)}")
    return metadata


def model_supports_task(name: str, task: str) -> bool:
    return task in get_model_metadata(name)["tasks"]


def __getattr__(name):
    if name == "MODELS":
        return _load()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
