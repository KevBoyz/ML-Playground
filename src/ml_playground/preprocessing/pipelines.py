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
