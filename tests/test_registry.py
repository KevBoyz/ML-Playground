import numpy as np
import pytest
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.decomposition import PCA

from ml_playground.preprocessing.registry import (
    get_transformer,
    IMPUTATION,
    SCALING,
    ENCODING,
    TRANSFORMATION,
    FEATURE_SELECTION,
    FEATURE_ENGINEERING,
    DIMENSIONALITY,
    OUTLIERS,
    IQRRemover,
    ZScoreRemover,
)


def test_get_transformer_imputation():
    t = get_transformer("imputation", "mean")
    assert isinstance(t, SimpleImputer)
    assert t.strategy == "mean"


def test_get_transformer_scaling():
    t = get_transformer("scaling", "standard")
    assert isinstance(t, StandardScaler)

    t = get_transformer("scaling", "robust")
    assert isinstance(t, RobustScaler)

    t = get_transformer("scaling", "minmax")
    assert isinstance(t, MinMaxScaler)


def test_get_transformer_with_params():
    t = get_transformer("imputation", "mean", {"add_indicator": True})
    assert t.add_indicator is True


def test_get_transformer_passthrough():
    assert get_transformer("transformation", "none") == "passthrough"
    assert get_transformer("feature_selection", "none") == "passthrough"
    assert get_transformer("dimensionality", "none") == "passthrough"


def test_get_transformer_pca():
    t = get_transformer("dimensionality", "pca", {"n_components": 0.95})
    assert isinstance(t, PCA)
    assert t.n_components == 0.95


def test_get_transformer_categoria_invalida():
    with pytest.raises(ValueError, match="Categoria desconhecida"):
        get_transformer("magic", "foo")


def test_get_transformer_metodo_invalido():
    with pytest.raises(ValueError, match="Método 'xyz' não encontrado"):
        get_transformer("scaling", "xyz")


def test_iqr_remover():
    X = np.array([[1], [2], [3], [100]])
    remover = IQRRemover(multiplier=1.5)
    remover.fit(X)
    result = remover.transform(X)
    assert result[-1][0] <= 100
    assert result[-1][0] < 100


def test_zscore_remover():
    X = np.array([[1], [1], [1], [1], [1e6]])
    remover = ZScoreRemover(threshold=1.0)
    remover.fit(X)
    result = remover.transform(X)
    assert result.shape == X.shape
    assert np.issubdtype(result.dtype, np.floating)


def test_outliers_registry():
    assert "iqr" in OUTLIERS
    assert "zscore" in OUTLIERS


def test_registry_supports_regression_selection_and_iterative_imputation():
    assert "iterative" in IMPUTATION
    assert "f_regression" in FEATURE_SELECTION
    assert "polynomial" in FEATURE_ENGINEERING
