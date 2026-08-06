"""Experiment lifecycle orchestration: preflight, search, selection and release."""

from __future__ import annotations

import hashlib
import json
import random
import time
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

from ml_playground.experiments.config import compute_config_fingerprint
from ml_playground.experiments.executor import fit_full_pipeline, run_experiment
from ml_playground.experiments.grid import build_model_grid
from ml_playground.experiments.preflight import prepare_experiment_data
from ml_playground.experiments.report import write_experiment_reports
from ml_playground.experiments.tracker import create_run
from ml_playground.models.serialization import save_model


def preflight_experiment(config: Mapping[str, Any]) -> dict[str, Any]:
    """Build the single in-memory data contract used by an experiment run.

    The returned configuration is intentionally a runtime copy: serializable
    evidence remains in ``preflight_metadata`` while frames and arrays stay in
    the private ``_prepared_data`` key.  Candidates reuse that exact payload,
    preventing a grid from rereading a changing source file.
    """

    runtime = _copy_public_config(config)
    runtime.setdefault("task", "classification")
    runtime.setdefault("data", {})
    runtime["config_fingerprint"] = runtime.get("config_fingerprint") or compute_config_fingerprint(runtime)
    prepared = prepare_experiment_data(runtime)
    runtime["_prepared_data"] = prepared
    runtime["preflight_metadata"] = prepared["preflight_metadata"]
    runtime["run_fingerprint"] = prepared["run_fingerprint"]
    return runtime


def build_run_plan(
    config: Mapping[str, Any],
    models_config: list[dict] | None = None,
) -> dict[str, Any]:
    """Preflight an experiment and describe its bounded execution before fit."""

    runtime = preflight_experiment(config)
    grid = _candidate_grid(runtime, models_config)
    selected_grid, strategy, limit = _apply_search_budget(grid, runtime)
    fits_per_candidate = _development_fit_count(runtime)
    final_test_fit = int(_has_final_test(runtime))
    deployment_refit = int(bool(runtime.get("outputs", {}).get("save_model", True)))
    return {
        "contract_version": runtime.get("contract_version", 1),
        "experiment_name": runtime.get("experiment_name"),
        "task": runtime["task"],
        "config_fingerprint": runtime["config_fingerprint"],
        "run_fingerprint": runtime["run_fingerprint"],
        "preflight": runtime["preflight_metadata"],
        "search": {
            "strategy": strategy,
            "max_candidates": limit,
            "candidate_count": len(selected_grid),
            "candidates": [
                {
                    "candidate_id": _candidate_id(runtime, entry),
                    "name": entry["name"],
                    "params": entry["params"],
                }
                for entry in selected_grid
            ],
        },
        "estimated_fits": {
            "development_per_candidate": fits_per_candidate,
            "development_total": fits_per_candidate * len(selected_grid),
            "selected_final_test": final_test_fit,
            "deployment_refit": deployment_refit,
            "total": fits_per_candidate * len(selected_grid) + final_test_fit + deployment_refit,
        },
    }


def run_grid(
    base_config: dict,
    models_config: list[dict] | None = None,
    *,
    write_reports: bool | None = None,
) -> dict:
    """Run a bounded model search and publish only the selected candidate.

    Development metrics select the candidate.  A reserved final test is
    evaluated only once, after selection, and is stored separately under
    ``best['final_test']``.  This preserves the original ``run_grid`` API while
    preventing a convenient external test file from becoming part of tuning.
    """

    model_entries = models_config if models_config is not None else base_config.get("models", [])
    raw_grid = build_model_grid(model_entries)
    try:
        config = preflight_experiment(base_config)
        grid, strategy, limit = _apply_search_budget(raw_grid, config)
    except Exception as exc:
        results = [_error_result(entry, exc) for entry in raw_grid]
        return {
            "results": results,
            "tracker": None,
            "best": None,
            "preflight_error": f"{type(exc).__name__}: {exc}",
        }

    is_experiment = bool(config.get("experiment_name"))
    tracker = None
    if is_experiment:
        tracker = create_run(config, persist=False)
    elif config.get("track"):
        tracker = create_run(config)

    if _is_nested_protocol(config):
        results = _run_nested_protocol(config, tracker)
    else:
        results = _run_candidates(config, grid)

    best_result = select_best_result(results, config.get("selection", {}))
    if best_result is not None and _has_final_test(config) and not _is_nested_protocol(config):
        _attach_final_test(config, best_result)

    output = {
        "results": results,
        "tracker": tracker,
        "best": best_result,
        "baseline": _resolve_baseline(results, config.get("selection", {})),
        "execution": {
            "search_strategy": strategy,
            "max_candidates": limit,
            "completed_candidates": len(results),
        },
    }

    if write_reports is None:
        write_reports = is_experiment
    if write_reports and is_experiment and tracker is not None:
        model_path = _persist_best_model(config, best_result, tracker)
        output["reports"] = write_experiment_reports(
            config,
            output,
            tracker["run_id"],
            best_result=best_result,
            model_path=model_path,
        )
    return output


def select_best_result(results: list[dict], selection: dict | None = None) -> dict | None:
    """Select a successful candidate with deterministic, declared tie breakers."""

    selection = selection or {}
    metric = selection.get("primary_metric", "accuracy")
    direction = selection.get("direction", "maximize")
    if direction not in {"maximize", "minimize"}:
        raise ValueError("selection.direction deve ser 'maximize' ou 'minimize'")
    candidates = [
        result
        for result in results
        if result.get("status", "success") == "success"
        and _finite_metric(result.get("metrics", {}).get(metric)) is not None
    ]
    if not candidates:
        return None

    tie_breakers = selection.get("tie_breakers", ["metric_std", "candidate_id"])
    if not isinstance(tie_breakers, list) or not all(isinstance(item, str) for item in tie_breakers):
        raise ValueError("selection.tie_breakers deve ser uma lista de strings")
    return min(
        candidates,
        key=lambda result: _selection_key(result, metric, direction, tie_breakers),
    )


def _run_candidates(config: dict, grid: list[dict]) -> list[dict]:
    execution = dict(config.get("execution") or {})
    error_policy = execution.get("on_candidate_error", "continue")
    if error_policy not in {"continue", "fail_fast"}:
        raise ValueError("execution.on_candidate_error deve ser 'continue' ou 'fail_fast'")
    max_wall_time = execution.get("max_wall_time_seconds")
    if max_wall_time is not None and float(max_wall_time) <= 0:
        raise ValueError("execution.max_wall_time_seconds deve ser positivo")

    results: list[dict] = []
    started = time.perf_counter()
    for entry in grid:
        if max_wall_time is not None and time.perf_counter() - started >= float(max_wall_time):
            results.append(
                {
                    **_candidate_identity(config, entry),
                    "status": "skipped",
                    "error": "execution_budget_exhausted",
                }
            )
            continue
        candidate_config = _candidate_config(config, entry, evaluate_final_test=False)
        try:
            result = run_experiment(candidate_config)
            result.update(_candidate_identity(config, entry))
            results.append(result)
        except Exception as exc:
            if error_policy == "fail_fast":
                raise
            results.append(_error_result(entry, exc, config=config))
    return results


def _run_nested_protocol(config: dict, tracker: dict | None) -> list[dict]:
    """Execute nested CV once because it selects candidates inside each outer fold."""

    nested_config = _copy_public_config(config)
    nested_config["_prepared_data"] = config["_prepared_data"]
    nested_config["preflight_metadata"] = config["preflight_metadata"]
    nested_config.pop("model", None)
    entry = {"name": "nested_selection", "params": {}}
    try:
        result = run_experiment(nested_config)
        result.update(_candidate_identity(config, entry))
        return [result]
    except Exception as exc:
        return [_error_result(entry, exc, config=config)]


def _attach_final_test(config: dict, best_result: dict) -> None:
    """Evaluate the frozen winner once, without replacing its development score."""

    entry = {"name": best_result["name"], "params": best_result.get("params", {})}
    final_config = _candidate_config(config, entry, evaluate_final_test=True)
    final_result = run_experiment(final_config).get("final_test")
    if final_result is None:
        raise RuntimeError("O protocolo declarou teste final, mas o executor não o produziu")
    best_result["final_test"] = final_result
    evaluation = dict(best_result.get("evaluation") or {})
    evaluation["final_test"] = {
        "configured": True,
        "evaluated": True,
        "metric_origin": "final_test",
        "source": (best_result.get("split_metadata") or {}).get("final_test", {}).get("source"),
    }
    best_result["evaluation"] = evaluation


def _persist_best_model(config: dict, best_result: dict | None, tracker: dict) -> str | None:
    if best_result is None or not config.get("outputs", {}).get("save_model", True):
        return None
    if _is_nested_protocol(config):
        deployment_model = (config.get("evaluation") or {}).get("deployment_model")
        if not isinstance(deployment_model, dict):
            return None
        entry = {"name": deployment_model["name"], "params": deployment_model.get("params", {})}
    else:
        entry = {"name": best_result["name"], "params": best_result.get("params", {})}

    final_config = _candidate_config(config, entry, evaluate_final_test=False)
    pipeline = fit_full_pipeline(final_config)
    project_root = Path(config.get("project_root", Path.cwd()))
    path = project_root / "models" / config["experiment_name"] / tracker["run_id"] / "model.joblib"
    prepared = config["_prepared_data"]
    frame = prepared["frame"]
    features = list(prepared["feature_columns"])
    feature_dtypes = {feature: str(frame.schema[feature]) for feature in features}
    id_column = config.get("data", {}).get("id_column")
    signature = {
        "features": features,
        "dtypes": feature_dtypes,
        "target": config.get("data", {}).get("target"),
        "id_column": id_column,
        "id_unique": bool(id_column),
        "allow_extra_columns": config.get("data", {}).get("schema", {}).get("mode") != "strict",
    }
    return save_model(
        pipeline,
        path,
        metadata={
            "experiment_name": config["experiment_name"],
            "run_id": tracker["run_id"],
            "task": config.get("task", "classification"),
            "model": entry["name"],
            "params": entry["params"],
            "development_metrics": best_result.get("metrics", {}),
            "final_test_metrics": (best_result.get("final_test") or {}).get("metrics", {}),
            "signature": signature,
            "provenance": config.get("provenance", {}),
            "preflight": config.get("preflight_metadata", {}),
            "run_context": tracker.get("run_context", {}),
        },
    )


def _candidate_grid(config: Mapping[str, Any], models_config: list[dict] | None) -> list[dict]:
    entries = models_config if models_config is not None else config.get("models", [])
    return build_model_grid(entries)


def _apply_search_budget(grid: list[dict], config: Mapping[str, Any]) -> tuple[list[dict], str, int | None]:
    search = dict(config.get("search") or {})
    execution = dict(config.get("execution") or {})
    strategy = str(search.get("strategy", "grid")).lower()
    if strategy not in {"grid", "random"}:
        raise ValueError("search.strategy deve ser 'grid' ou 'random'")
    configured_limit = execution.get("max_candidates", search.get("max_candidates"))
    limit = None if configured_limit is None else int(configured_limit)
    if limit is not None and limit < 1:
        raise ValueError("search.max_candidates deve ser >= 1")
    selected = list(grid)
    if limit is not None:
        if strategy == "grid":
            selected = selected[:limit]
        elif len(selected) > limit:
            seed = int(search.get("random_state", config.get("data", {}).get("random_state", 42)))
            selected = [selected[index] for index in sorted(random.Random(seed).sample(range(len(selected)), limit))]
    return selected, strategy, limit


def _candidate_config(config: dict, entry: dict, *, evaluate_final_test: bool) -> dict:
    candidate = _copy_public_config(config)
    candidate["_prepared_data"] = config["_prepared_data"]
    candidate["preflight_metadata"] = config["preflight_metadata"]
    candidate["model"] = {"name": entry["name"], "params": dict(entry.get("params", {}))}
    evaluation = dict(candidate.get("evaluation") or {})
    evaluation["evaluate_final_test"] = evaluate_final_test
    candidate["evaluation"] = evaluation
    return candidate


def _copy_public_config(config: Mapping[str, Any]) -> dict[str, Any]:
    return deepcopy({key: value for key, value in config.items() if not str(key).startswith("_")})


def _candidate_identity(config: Mapping[str, Any], entry: Mapping[str, Any]) -> dict[str, Any]:
    candidate_id = _candidate_id(config, entry)
    return {
        "name": entry["name"],
        "params": dict(entry.get("params", {})),
        "candidate_id": candidate_id,
        "trial_id": candidate_id,
        "split_id": "development",
    }


def _candidate_id(config: Mapping[str, Any], entry: Mapping[str, Any]) -> str:
    payload = {
        "run_fingerprint": config.get("run_fingerprint") or config.get("config_fingerprint"),
        "model": entry.get("name"),
        "params": entry.get("params", {}),
        "preprocessing": config.get("preprocessing", {}),
        "evaluation": _evaluation_identity(config.get("evaluation") or {}),
    }
    serialized = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str, separators=(",", ":"))
    return f"candidate-{hashlib.sha256(serialized.encode('utf-8')).hexdigest()[:16]}"


def _evaluation_identity(evaluation: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in evaluation.items()
        if key not in {"evaluate_final_test", "deployment_include_final_test"}
    }


def _error_result(entry: Mapping[str, Any], exc: Exception, *, config: Mapping[str, Any] | None = None) -> dict:
    identity = _candidate_identity(config, entry) if config is not None else {
        "name": entry.get("name", "unknown"),
        "params": dict(entry.get("params", {})),
    }
    return {**identity, "status": "error", "error": f"{type(exc).__name__}: {exc}"}


def _selection_key(result: dict, metric: str, direction: str, tie_breakers: list[str]) -> tuple:
    primary = _finite_metric(result["metrics"][metric])
    assert primary is not None
    key: list[Any] = [-primary if direction == "maximize" else primary]
    for breaker in tie_breakers:
        if breaker == "metric_std":
            key.append(_finite_metric(result.get("metric_std", {}).get(metric)) or 0.0)
        elif breaker == "candidate_id":
            key.append(str(result.get("candidate_id") or _result_identity(result)))
        else:
            value = _finite_metric(result.get("metrics", {}).get(breaker))
            key.append(float("inf") if value is None else (-value if direction == "maximize" else value))
    return tuple(key)


def _finite_metric(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number == number and number not in {float("inf"), float("-inf")} else None


def _result_identity(result: Mapping[str, Any]) -> str:
    return json.dumps(
        {"name": result.get("name", result.get("model")), "params": result.get("params", {})},
        sort_keys=True,
        default=str,
    )


def _resolve_baseline(results: list[dict], selection: Mapping[str, Any]) -> dict | None:
    declared = selection.get("baseline_candidate")
    if declared is None:
        return None
    matches = [
        result
        for result in results
        if result.get("candidate_id") == declared or result.get("name", result.get("model")) == declared
    ]
    if not matches:
        raise ValueError("selection.baseline_candidate não corresponde a um candidato válido")
    return matches[0]


def _development_fit_count(config: Mapping[str, Any]) -> int:
    evaluation = dict(config.get("evaluation") or {})
    protocol = str(evaluation.get("protocol", "development")).lower()
    splitter = dict(evaluation.get("splitter") or config.get("cross_validation") or {})
    method = str(splitter.get("method", splitter.get("name", "holdout"))).lower()
    if protocol == "nested_cv":
        outer = int((evaluation.get("outer_splitter") or splitter).get("n_splits", 5))
        inner = int((evaluation.get("inner_splitter") or {}).get("n_splits", 3))
        candidates = len(build_model_grid(config.get("models", []))) or 1
        return outer * (inner * candidates + 1)
    if method in {"holdout", "group_holdout", "temporal_holdout"}:
        return 1
    repeats = int(splitter.get("n_repeats", 3)) if method.startswith("repeated_") else 1
    return int(splitter.get("n_splits", 5)) * repeats


def _has_final_test(config: Mapping[str, Any]) -> bool:
    if config.get("task") == "clustering":
        return False
    evaluation = dict(config.get("evaluation") or {})
    final = evaluation.get("final_test")
    if isinstance(final, Mapping):
        return str(final.get("source", "none")).lower() not in {"none", "disabled", "false"}
    if str(evaluation.get("protocol", "")).lower() == "train_validation_test":
        return True
    return bool(config.get("data", {}).get("test"))


def _is_nested_protocol(config: Mapping[str, Any]) -> bool:
    return str((config.get("evaluation") or {}).get("protocol", "")).lower() in {"nested_cv", "nested"}
