"""Centralized persistence for task-aware experiment outputs."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

from ml_playground.experiments.views import merged_predictions, write_configured_views


def results_to_csv(results, path):
    """Flatten scalar result fields into a tabular CSV."""

    rows = []
    for result in results:
        row = {
            "model": result.get("name", result.get("model", "?")),
            "status": result.get("status", "error" if "error" in result else "success"),
        }
        if "duration_seconds" in result:
            row["duration_seconds"] = result["duration_seconds"]
        for key, value in result.get("params", {}).items():
            row[f"param_{key}"] = value
        for key, value in result.get("metrics", {}).items():
            if np.isscalar(value):
                row[f"metric_{key}"] = value
        for key, value in result.get("metric_std", {}).items():
            if np.isscalar(value):
                row[f"metric_std_{key}"] = value
        if "error" in result:
            row["error"] = result["error"]
        rows.append(row)
    return _write_dataframe(rows, Path(path))


def write_experiment_reports(
    config: dict,
    bundle: dict,
    run_id: str,
    *,
    best_result: dict | None = None,
    model_path: str | None = None,
) -> dict[str, str]:
    """Write task-aware outputs under ``reports/<experiment_name>/``."""

    experiment_name = config["experiment_name"]
    report_root = Path(config.get("outputs", {}).get("root", "reports")) / experiment_name
    directories = {
        "metrics": report_root / "metrics",
        "tables": report_root / "tables",
        "predictions": report_root / "predictions",
        "figures": report_root / "figures",
    }
    for directory in directories.values():
        directory.mkdir(parents=True, exist_ok=True)

    results = bundle.get("results", [])
    artifacts = {
        "summary": results_to_csv(results, directories["metrics"] / f"{run_id}_summary.csv"),
        "fold_metrics": _write_fold_metrics(results, directories["metrics"] / f"{run_id}_fold_metrics.csv"),
        "comparison": _write_comparison_table(
            results,
            config["selection"]["primary_metric"],
            directories["tables"] / f"{run_id}_model_comparison.csv",
        ),
        "errors": _write_errors(results, directories["tables"] / f"{run_id}_errors.csv"),
    }
    if config.get("outputs", {}).get("save_predictions", True):
        artifacts.update(_write_predictions(results, directories["predictions"] / f"{run_id}_predictions.csv"))
    if best_result is not None:
        artifacts.update(_write_task_tables(config["task"], best_result, run_id, directories["tables"]))

    view_artifacts, view_status = write_configured_views(
        config,
        results,
        best_result,
        run_id,
        directories["figures"],
    )
    artifacts.update(view_artifacts)

    manifest = {
        "run_id": run_id,
        "experiment_name": experiment_name,
        "task": config["task"],
        "config": _public_config(config),
        "best_result": _public_result(best_result) if best_result else None,
        "model_path": model_path,
        "artifacts": artifacts,
        "views": view_status,
    }
    manifest_path = directories["tables"] / f"{run_id}_manifest.json"
    _write_json(manifest, manifest_path)
    artifacts["manifest"] = str(manifest_path)
    return artifacts


def _write_fold_metrics(results, path: Path) -> str:
    rows = []
    for result in results:
        for fold in result.get("folds", []):
            row = {
                "model": result.get("name", result.get("model", "?")),
                "fold": fold.get("fold"),
                "n_train": fold.get("n_train"),
                "n_test": fold.get("n_test"),
            }
            row.update({f"metric_{key}": value for key, value in fold.get("metrics", {}).items()})
            rows.append(row)
    return _write_dataframe(rows, path)


def _write_comparison_table(results, metric: str, path: Path) -> str:
    rows = []
    for result in results:
        if "metrics" not in result:
            continue
        rows.append(
            {
                "model": result.get("name", result.get("model", "?")),
                "params": json.dumps(result.get("params", {}), sort_keys=True, default=str),
                "metric": metric,
                "value": result["metrics"].get(metric),
                "status": result.get("status", "success"),
            }
        )
    return _write_dataframe(rows, path)


def _write_errors(results, path: Path) -> str:
    rows = [
        {
            "model": result.get("name", result.get("model", "?")),
            "params": json.dumps(result.get("params", {}), sort_keys=True, default=str),
            "error": result.get("error"),
        }
        for result in results
        if "error" in result
    ]
    return _write_dataframe(rows, path)


def _write_predictions(results, path: Path) -> dict[str, str]:
    rows = []
    for result in results:
        model_name = result.get("name", result.get("model", "?"))
        for fold in result.get("folds", []):
            _append_prediction_rows(rows, model_name, fold.get("fold"), fold.get("predictions"))
        if "predictions" in result:
            _append_prediction_rows(rows, model_name, None, result["predictions"])
    return {"predictions": _write_dataframe(rows, path)}


def _append_prediction_rows(rows, model_name, fold, predictions):
    if not predictions:
        return
    row_ids = predictions.get("row_ids", [])
    if predictions.get("labels") is not None:
        for row_id, label in zip(row_ids, predictions["labels"]):
            rows.append({"model": model_name, "fold": fold, "row": row_id, "cluster": label})
        return
    y_true = predictions.get("y_true", [])
    y_pred = predictions.get("y_pred", [])
    y_score = predictions.get("y_score")
    for index, (true_value, pred_value) in enumerate(zip(y_true, y_pred)):
        score = None if y_score is None else y_score[index]
        row = row_ids[index] if index < len(row_ids) else index
        rows.append(
            {
                "model": model_name,
                "fold": fold,
                "row": row,
                "y_true": true_value,
                "y_pred": pred_value,
                "y_score": json.dumps(score, default=str),
            }
        )


def _write_task_tables(task: str, best_result: dict, run_id: str, directory: Path) -> dict[str, str]:
    predictions = merged_predictions(best_result)
    if not predictions:
        return {}
    if task == "classification":
        return _write_confusion_table(best_result, predictions, run_id, directory)
    if task == "regression":
        return _write_residual_table(best_result, predictions, run_id, directory)
    if task == "clustering":
        return _write_cluster_size_table(best_result, predictions, run_id, directory)
    return {}


def _write_confusion_table(result, predictions, run_id, directory):
    y_true = np.asarray(predictions.get("y_true", []))
    y_pred = np.asarray(predictions.get("y_pred", []))
    if not len(y_true):
        return {}
    labels = np.unique(np.concatenate([y_true, y_pred]))
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    path = directory / f"{run_id}_{result['name']}_confusion_matrix.csv"
    pd.DataFrame(matrix, index=labels, columns=labels).to_csv(path)
    return {"confusion_matrix": str(path)}


def _write_residual_table(result, predictions, run_id, directory):
    y_true = predictions.get("y_true", [])
    y_pred = predictions.get("y_pred", [])
    if not y_true:
        return {}
    row_ids = predictions.get("row_ids", range(len(y_true)))
    frame = pd.DataFrame({"row": row_ids, "observed": y_true, "predicted": y_pred})
    frame["residual"] = frame["observed"] - frame["predicted"]
    path = directory / f"{run_id}_{result['name']}_residuals.csv"
    frame.to_csv(path, index=False)
    return {"residuals": str(path)}


def _write_cluster_size_table(result, predictions, run_id, directory):
    labels = predictions.get("labels", [])
    if not labels:
        return {}
    counts = Counter(labels)
    rows = [
        {"cluster": label, "size": size, "is_noise": label == -1}
        for label, size in sorted(counts.items())
    ]
    path = directory / f"{run_id}_{result['name']}_cluster_sizes.csv"
    _write_dataframe(rows, path)
    return {"cluster_sizes": str(path)}


def _write_dataframe(rows, path: Path) -> str:
    pd.DataFrame(rows).to_csv(path, index=False)
    return str(path)


def _write_json(value, path: Path) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")


def _json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)


def _public_config(config):
    return {key: value for key, value in config.items() if key not in {"model"}}


def _public_result(result):
    if result is None:
        return None
    return {key: value for key, value in result.items() if key != "pipeline"}
