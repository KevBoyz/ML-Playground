"""Grid orchestration, model selection and artifact persistence."""

from copy import deepcopy
from pathlib import Path

from ml_playground.experiments.executor import fit_full_pipeline, run_experiment
from ml_playground.experiments.grid import build_model_grid
from ml_playground.experiments.report import write_experiment_reports
from ml_playground.experiments.tracker import create_run
from ml_playground.models.serialization import save_model


def run_grid(
    base_config: dict,
    models_config: list[dict] | None = None,
    *,
    write_reports: bool | None = None,
) -> dict:
    """Execute all model combinations and optionally publish one experiment run."""

    model_entries = models_config if models_config is not None else base_config.get("models", [])
    grid = build_model_grid(model_entries)
    is_experiment = bool(base_config.get("experiment_name"))
    tracker = None
    if is_experiment:
        tracker = create_run(base_config, persist=False)
    elif base_config.get("track"):
        tracker = create_run(base_config)

    results = []
    for entry in grid:
        experiment_config = deepcopy(base_config)
        experiment_config["model"] = {
            "name": entry["name"],
            "params": entry["params"],
        }
        try:
            result = run_experiment(experiment_config)
            result.update({"name": entry["name"], "params": entry["params"]})
            results.append(result)
        except Exception as exc:
            results.append(
                {
                    "name": entry["name"],
                    "params": entry["params"],
                    "status": "error",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )

    best_result = select_best_result(results, base_config.get("selection", {}))
    output = {
        "results": results,
        "tracker": tracker,
        "best": best_result,
    }

    if write_reports is None:
        write_reports = is_experiment
    if write_reports and is_experiment and tracker is not None:
        model_path = _persist_best_model(base_config, best_result, tracker["run_id"])
        output["reports"] = write_experiment_reports(
            base_config,
            output,
            tracker["run_id"],
            best_result=best_result,
            model_path=model_path,
        )
    return output


def select_best_result(results: list[dict], selection: dict | None = None) -> dict | None:
    """Select the best successful result using metric direction and variability."""

    selection = selection or {}
    metric = selection.get("primary_metric", "accuracy")
    direction = selection.get("direction", "maximize")
    candidates = [
        result
        for result in results
        if result.get("status", "success") == "success"
        and metric in result.get("metrics", {})
    ]
    if not candidates:
        return None

    if direction == "maximize":
        primary = max(result["metrics"][metric] for result in candidates)
    else:
        primary = min(result["metrics"][metric] for result in candidates)
    tied = [result for result in candidates if result["metrics"][metric] == primary]
    return min(
        tied,
        key=lambda result: result.get("metric_std", {}).get(metric, 0.0),
    )


def _persist_best_model(config, best_result, run_id):
    if best_result is None or not config.get("outputs", {}).get("save_model", True):
        return None

    final_config = deepcopy(config)
    final_config["model"] = {
        "name": best_result["name"],
        "params": best_result.get("params", {}),
    }
    pipeline = fit_full_pipeline(final_config)
    project_root = Path(config.get("project_root", Path.cwd()))
    path = (
        project_root
        / "models"
        / config["experiment_name"]
        / run_id
        / "model.joblib"
    )
    return save_model(
        pipeline,
        path,
        metadata={
            "experiment_name": config["experiment_name"],
            "run_id": run_id,
            "task": config.get("task", "classification"),
            "model": best_result["name"],
            "params": best_result.get("params", {}),
            "metrics": best_result.get("metrics", {}),
        },
    )
