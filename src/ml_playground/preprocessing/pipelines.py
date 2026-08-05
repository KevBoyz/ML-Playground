from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from ml_playground.preprocessing.registry import get_transformer


def _is_flat_config(config):
    return isinstance(config, dict) and not any(
        k in config for k in ("name", "category")
    )


def build_pipeline(config: list[dict] | dict) -> Pipeline:
    if _is_flat_config(config):
        config = _flat_to_steps(config)

    if not config:
        return Pipeline([("passthrough", "passthrough")])

    pipeline_steps = []
    for step in config:
        name = step.get("name", step.get("category", "step"))
        method = step.get("method", "none")
        params = step.get("params", {})
        transformer = get_transformer(step["category"], method, params)
        if transformer != "passthrough":
            pipeline_steps.append((name, transformer))

    if not pipeline_steps:
        return Pipeline([("passthrough", "passthrough")])

    return Pipeline(pipeline_steps)


def build_preprocessor(config: dict | list[dict], X) -> ColumnTransformer | Pipeline:
    """Build a typed preprocessor that fits only inside each training split."""

    if isinstance(config, list):
        return build_pipeline(config)

    if not isinstance(config, dict):
        raise TypeError("A configuração de preprocessing deve ser um mapa ou lista")

    if not any(key in config for key in ("numeric", "categorical")):
        return build_pipeline(config)

    numeric_columns, categorical_columns = _column_groups(X)
    transformers = []
    for branch, columns in (
        ("numeric", numeric_columns),
        ("categorical", categorical_columns),
    ):
        if not columns:
            continue
        branch_config = config.get(branch, {}) or {}
        steps = branch_config.get("steps", [])
        transformers.append((branch, build_pipeline(steps), columns))

    if not transformers:
        return Pipeline([("passthrough", "passthrough")])
    return ColumnTransformer(transformers, remainder="drop")


def build_model_pipeline(config: dict | list[dict], X, model) -> Pipeline:
    """Compose preprocessing and estimator into one fitted artifact."""

    preprocessor = build_preprocessor(config, X)
    return Pipeline([("preprocessing", preprocessor), ("model", model)])


def _column_groups(X) -> tuple[list, list]:
    if hasattr(X, "select_dtypes"):
        numeric = X.select_dtypes(include=["number"]).columns.tolist()
        categorical = [column for column in X.columns if column not in numeric]
        return numeric, categorical

    n_columns = getattr(X, "shape", (0, 0))[1]
    return list(range(n_columns)), []


def _flat_to_steps(flat: dict) -> list[dict]:
    steps = []
    for category, cfg in flat.items():
        method = cfg.get("method", "none")
        params = cfg.get("params", {})
        steps.append(
            {
                "name": category,
                "category": category,
                "method": method,
                "params": params,
            }
        )
    return steps


def build_column_transformer(transformations: list[dict]) -> ColumnTransformer:
    transformers = []
    for t in transformations:
        name = t.get("name", t.get("category", "step"))
        method = t.get("method", "none")
        params = t.get("params", {})
        columns = t.get("columns", None)
        transformer = get_transformer(t["category"], method, params)
        if transformer != "passthrough":
            transformers.append((name, transformer, columns))
    return ColumnTransformer(transformers)
