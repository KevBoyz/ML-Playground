import pytest
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

from ml_playground.models import MODELS, get_model


def test_registry_contains_core_models():
    assert "random_forest" in MODELS
    assert "svm" in MODELS
    assert "knn" in MODELS
    assert "logistic" in MODELS
    assert "xgboost" in MODELS
    assert "lightgbm" in MODELS


def test_registry_maps_to_sklearn_classes():
    assert MODELS["random_forest"] is RandomForestClassifier
    assert MODELS["svm"] is SVC
    assert MODELS["knn"] is KNeighborsClassifier
    assert MODELS["logistic"] is LogisticRegression
    assert MODELS["xgboost"] is XGBClassifier
    assert MODELS["lightgbm"] is LGBMClassifier


def test_get_model_default_params():
    model = get_model("random_forest")
    assert isinstance(model, RandomForestClassifier)
    assert model.n_estimators == 100


def test_get_model_with_params():
    model = get_model("random_forest", {"n_estimators": 50, "max_depth": 5})
    assert isinstance(model, RandomForestClassifier)
    assert model.n_estimators == 50
    assert model.max_depth == 5


def test_get_model_svm():
    model = get_model("svm", {"kernel": "rbf", "C": 1.0})
    assert isinstance(model, SVC)
    assert model.kernel == "rbf"


def test_get_model_knn():
    model = get_model("knn", {"n_neighbors": 7})
    assert isinstance(model, KNeighborsClassifier)
    assert model.n_neighbors == 7


def test_get_model_logistic():
    model = get_model("logistic", {"C": 0.1})
    assert isinstance(model, LogisticRegression)
    assert model.C == 0.1


def test_get_model_unknown():
    with pytest.raises(ValueError, match="Modelo 'magic' não encontrado"):
        get_model("magic")


def test_get_model_unknown_message_lists_available():
    with pytest.raises(ValueError, match="random_forest"):
        get_model("nonexistent")


def test_model_instances_are_fittable():
    import numpy as np
    from sklearn.utils.validation import check_is_fitted

    X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    y = np.array([0, 0, 1, 1])
    model = get_model("logistic")
    model.fit(X, y)
    check_is_fitted(model)
    preds = model.predict(X)
    assert len(preds) == 4
