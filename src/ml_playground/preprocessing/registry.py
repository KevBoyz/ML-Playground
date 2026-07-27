import numpy as np
from sklearn.decomposition import PCA, KernelPCA
from sklearn.feature_selection import SelectKBest, VarianceThreshold
from sklearn.feature_selection import mutual_info_classif, f_classif
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.preprocessing import (
    FunctionTransformer,
    MinMaxScaler,
    OneHotEncoder,
    OrdinalEncoder,
    PowerTransformer,
    RobustScaler,
    StandardScaler,
)

IMPUTATION = {
    "mean": (SimpleImputer, {"strategy": "mean"}),
    "median": (SimpleImputer, {"strategy": "median"}),
    "mode": (SimpleImputer, {"strategy": "most_frequent"}),
    "constant": (SimpleImputer, {"strategy": "constant"}),
    "knn": KNNImputer,
}

SCALING = {
    "standard": StandardScaler,
    "robust": RobustScaler,
    "minmax": MinMaxScaler,
}

ENCODING = {
    "onehot": OneHotEncoder,
    "ordinal": OrdinalEncoder,
}

TRANSFORMATION = {
    "none": "passthrough",
    "log": None,
    "boxcox": PowerTransformer,
    "yeojohnson": PowerTransformer,
}

FEATURE_SELECTION = {
    "none": "passthrough",
    "variance_threshold": VarianceThreshold,
    "mutual_info": lambda k=10: SelectKBest(mutual_info_classif, k=k),
    "f_classif": lambda k=10: SelectKBest(f_classif, k=k),
}

DIMENSIONALITY = {
    "none": "passthrough",
    "pca": PCA,
    "kernel_pca": KernelPCA,
}


class IQRRemover:
    def __init__(self, multiplier: float = 1.5):
        self.multiplier = multiplier
        self.lower = None
        self.upper = None

    def fit(self, X, y=None):
        q1 = np.percentile(X, 25, axis=0)
        q3 = np.percentile(X, 75, axis=0)
        iqr = q3 - q1
        self.lower = q1 - self.multiplier * iqr
        self.upper = q3 + self.multiplier * iqr
        return self

    def transform(self, X, y=None):
        return X.clip(self.lower, self.upper)


class ZScoreRemover:
    def __init__(self, threshold: float = 3.0):
        self.threshold = threshold
        self.mean_ = None
        self.std_ = None

    def fit(self, X, y=None):
        self.mean_ = np.mean(X, axis=0)
        self.std_ = np.std(X, axis=0)
        return self

    def transform(self, X, y=None):
        z = np.abs((X - self.mean_) / self.std_)
        mask = z < self.threshold
        return np.where(
            mask, X, self.mean_ + self.threshold * self.std_ * np.sign(X - self.mean_)
        )


OUTLIERS = {
    "iqr": IQRRemover,
    "zscore": ZScoreRemover,
}


def get_registry():
    return {
        "imputation": IMPUTATION,
        "scaling": SCALING,
        "encoding": ENCODING,
        "transformation": TRANSFORMATION,
        "feature_selection": FEATURE_SELECTION,
        "dimensionality": DIMENSIONALITY,
        "outliers": OUTLIERS,
    }


def get_transformer(category: str, method: str, params: dict | None = None):
    cat = get_registry().get(category)
    if cat is None:
        raise ValueError(f"Categoria desconhecida: {category}")

    entry = cat.get(method)
    if entry is None:
        raise ValueError(f"Método '{method}' não encontrado em {category}")

    if entry == "passthrough":
        return "passthrough"

    if isinstance(entry, tuple):
        cls, defaults = entry
        merged = {**defaults, **(params or {})}
        return cls(**merged)

    if isinstance(entry, type):
        return entry(**(params or {}))

    return entry
