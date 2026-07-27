from itertools import product


def expand_params(params: dict) -> list[dict]:
    keys, values = [], []
    for k, v in params.items():
        keys.append(k)
        values.append(v if isinstance(v, list) else [v])
    return [dict(zip(keys, combo)) for combo in product(*values)]


def build_model_grid(models_config: list[dict]) -> list[dict]:
    grid = []
    for entry in models_config:
        name = entry["name"]
        for combo in expand_params(entry.get("params", {})):
            grid.append({"name": name, "params": combo})
    return grid
