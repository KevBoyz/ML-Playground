import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split

from ml_playground.data.loader import auto_read
from ml_playground.evaluation.metrics import CLASSIFICATION, compute_metrics
from ml_playground.models import get_model
from ml_playground.preprocessing.pipelines import build_pipeline


def _to_pandas(df):
    return df.to_pandas()


def run_experiment(config: dict) -> dict:
    df = auto_read(config["data"]["path"])
    target_col = config["data"]["target"]
    y = df[target_col].to_numpy()
    X = _to_pandas(df.drop(target_col))

    cv_config = config.get("cross_validation", {})
    cv_method = cv_config.get("method", "holdout")
    rs = config["data"].get("random_state")
    metric_names = config.get("metrics", [])

    if cv_method == "holdout":
        return _run_holdout(
            X,
            y,
            config,
            metric_names,
            rs,
        )
    return _run_cv(
        X,
        y,
        config,
        metric_names,
        cv_config,
        rs,
    )


def _run_holdout(X, y, config, metric_names, rs):
    test_size = config["data"].get("test_size", 0.2)
    stratified = config["data"].get("stratified", False)
    stratify = y if stratified else None

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=rs,
        stratify=stratify,
    )

    pipe = build_pipeline(config.get("preprocessing", {}))
    X_train_t = pipe.fit_transform(X_train, y_train)
    X_test_t = pipe.transform(X_test)

    model = get_model(config["model"]["name"], config["model"].get("params", {}))
    model.fit(X_train_t, y_train)
    y_pred = model.predict(X_test_t)
    y_score = _get_y_score(model, X_test_t)

    metrics = compute_metrics(y_test, y_pred, metric_names, y_score=y_score)
    return {
        "model": config["model"]["name"],
        "params": config["model"].get("params", {}),
        "metrics": metrics,
    }


def _run_cv(X, y, config, metric_names, cv_config, rs):
    n_splits = cv_config.get("n_splits", 5)
    shuffle = cv_config.get("shuffle", True)
    stratified = config["data"].get("stratified", False)

    if stratified:
        cv = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=rs)
    else:
        cv = KFold(n_splits=n_splits, shuffle=shuffle, random_state=rs)

    cv_scores = {m: [] for m in metric_names if m in CLASSIFICATION}
    for train_idx, test_idx in cv.split(X, y):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        pipe = build_pipeline(config.get("preprocessing", {}))
        X_train_t = pipe.fit_transform(X_train, y_train)
        X_test_t = pipe.transform(X_test)

        model = get_model(config["model"]["name"], config["model"].get("params", {}))
        model.fit(X_train_t, y_train)
        y_pred = model.predict(X_test_t)
        y_score = _get_y_score(model, X_test_t)

        fold = compute_metrics(y_test, y_pred, metric_names, y_score=y_score)
        for m, v in fold.items():
            cv_scores[m].append(v)

    avg = {m: float(np.mean(v)) for m, v in cv_scores.items()}
    return {
        "model": config["model"]["name"],
        "params": config["model"].get("params", {}),
        "metrics": avg,
        "cv_scores": {m: [float(x) for x in v] for m, v in cv_scores.items()},
    }


def _get_y_score(model, X):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return None
