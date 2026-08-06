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

    if "groups" in config:
        return _build_explicit_groups(config["groups"], X, config.get("remainder", "drop"))

    if not any(key in config for key in ("numeric", "categorical")):
        return build_pipeline(config)

    numeric_columns, categorical_columns = _column_groups(X)
    transformers = []
    for branch, columns in (
        ("numeric", numeric_columns),
        ("categorical", categorical_columns),
    ):
        branch_config = config.get(branch, {}) or {}
        explicit_columns = branch_config.get("columns")
        if explicit_columns is not None:
            columns = _validate_columns(explicit_columns, X, branch)
        if not columns:
            continue
        steps = branch_config.get("steps", [])
        transformers.append((branch, build_pipeline(steps), columns))

    if not transformers:
        return Pipeline([("passthrough", "passthrough")])
    return ColumnTransformer(transformers, remainder=config.get("remainder", "drop"))


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


def _build_explicit_groups(groups, X, remainder):
    if not isinstance(groups, dict) or not groups:
        raise ValueError("preprocessing.groups deve ser um mapa não vazio")

    transformers = []
    claimed = set()
    for name, group in groups.items():
        if not isinstance(group, dict):
            raise ValueError(f"preprocessing.groups.{name} deve ser um mapa")
        columns = _validate_columns(group.get("columns"), X, name)
        overlap = claimed.intersection(columns)
        if overlap:
            raise ValueError(f"Colunas repetidas entre grupos de preprocessing: {sorted(overlap)}")
        claimed.update(columns)
        transformers.append((name, build_pipeline(group.get("steps", [])), columns))
    return ColumnTransformer(transformers, remainder=remainder)


def _validate_columns(columns, X, group):
    if not isinstance(columns, list) or not columns:
        raise ValueError(f"O grupo {group!r} deve declarar uma lista não vazia de columns")
    available = list(getattr(X, "columns", []))
    missing = sorted(set(columns) - set(available))
    if missing:
        raise ValueError(f"Colunas ausentes no grupo {group!r}: {missing}")
    if len(set(columns)) != len(columns):
        raise ValueError(f"O grupo {group!r} contém colunas duplicadas")
    return columns


def transformed_feature_names(pipeline, input_features) -> list[str]:
    """Resolve nomes finais quando o preprocessor expõe a API do sklearn."""

    preprocessor = getattr(pipeline, "named_steps", {}).get("preprocessing", pipeline)
    if hasattr(preprocessor, "get_feature_names_out"):
        return [str(name) for name in preprocessor.get_feature_names_out(input_features)]
    return [str(name) for name in input_features]


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
