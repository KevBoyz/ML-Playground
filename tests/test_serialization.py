import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.validation import check_is_fitted

from ml_playground.models.serialization import load_model, save_model


def test_save_load_model(tmp_path):
    model = RandomForestClassifier(n_estimators=10)
    model.fit(np.array([[1, 2], [3, 4]]), np.array([0, 1]))
    path = save_model(model, str(tmp_path / "model.joblib"))
    loaded, metadata = load_model(path)
    check_is_fitted(loaded)
    assert isinstance(loaded, RandomForestClassifier)


def test_save_model_returns_string_path(tmp_path):
    model = RandomForestClassifier()
    path = save_model(model, str(tmp_path / "m.joblib"))
    assert isinstance(path, str)
    assert path.endswith(".joblib")


def test_load_model_returns_metadata(tmp_path):
    model = RandomForestClassifier()
    path = save_model(model, str(tmp_path / "m.joblib"), {"acc": 0.9})
    _, meta = load_model(path)
    assert meta["acc"] == 0.9
    assert "saved_at" in meta
