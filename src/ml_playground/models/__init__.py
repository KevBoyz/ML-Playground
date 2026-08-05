_registry = None


def _load():
    global _registry
    if _registry is not None:
        return _registry
    from lightgbm import LGBMClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.svm import SVC
    from sklearn.tree import DecisionTreeClassifier
    from xgboost import XGBClassifier

    _registry = {
        "random_forest": RandomForestClassifier,
        "svm": SVC,
        "knn": KNeighborsClassifier,
        "decision_tree": DecisionTreeClassifier,
        "logistic_regression": LogisticRegression,
        "logistic": LogisticRegression,
        "xgboost": XGBClassifier,
        "lightgbm": LGBMClassifier,
    }
    return _registry


def get_model(name: str, params: dict | None = None):
    registry = _load()
    entry = registry.get(name)
    if entry is None:
        raise ValueError(
            f"Modelo '{name}' não encontrado. Disponíveis: {list(registry)}"
        )
    return entry(**(params or {}))


def __getattr__(name):
    if name == "MODELS":
        return _load()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
