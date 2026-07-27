from copy import deepcopy

from ml_playground.experiments.executor import run_experiment
from ml_playground.experiments.grid import build_model_grid
from ml_playground.experiments.tracker import create_run


def run_grid(base_config: dict, models_config: list[dict]) -> list[dict]:
    grid = build_model_grid(models_config)
    tracker = create_run(base_config) if base_config.get("track") else None
    results = []
    for entry in grid:
        exp = deepcopy(base_config)
        exp["model"] = {"name": entry["name"], "params": entry["params"]}
        try:
            result = run_experiment(exp)
            result["name"] = entry["name"]
            result["params"] = entry["params"]
            results.append(result)
        except Exception as e:
            results.append(
                {
                    "name": entry["name"],
                    "params": entry["params"],
                    "error": str(e),
                }
            )
    return {"results": results, "tracker": tracker}
