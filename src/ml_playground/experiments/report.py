"""Centralized persistence for experiment metrics, tables and figures."""

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

from ml_playground.visualization.barplots import plot_metric_comparison
from ml_playground.visualization.heatmaps import plot_confusion_matrix
from ml_playground.visualization.roc import plot_roc_curves


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
        if "params" in result and result["params"]:
            for key, value in result["params"].items():
                row[f"param_{key}"] = value
        if "metrics" in result:
            for key, value in result["metrics"].items():
                if np.isscalar(value):
                    row[f"metric_{key}"] = value
        if "metric_std" in result:
            for key, value in result["metric_std"].items():
                if np.isscalar(value):
                    row[f"metric_std_{key}"] = value
        if "error" in result:
            row["error"] = result["error"]
        rows.append(row)
    df = pd.DataFrame(rows)
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    return str(out)


def write_experiment_reports(
    config: dict,
    bundle: dict,
    run_id: str,
    *,
    best_result: dict | None = None,
    model_path: str | None = None,
) -> dict[str, str]:
    """Write all outputs under ``reports/<experiment_name>/``."""

    experiment_name = config["experiment_name"]
    report_root = Path(config.get("outputs", {}).get("root", "reports")) / experiment_name
    directories = {
        "metrics": report_root / "metrics",
        "tables": report_root / "tables",
        "figures": report_root / "figures",
    }
    for directory in directories.values():
        directory.mkdir(parents=True, exist_ok=True)

    results = bundle.get("results", [])
    artifacts = {}
    artifacts["summary"] = results_to_csv(
        results,
        directories["metrics"] / f"{run_id}_summary.csv",
    )
    artifacts["fold_metrics"] = _write_fold_metrics(
        results,
        directories["metrics"] / f"{run_id}_fold_metrics.csv",
    )
    artifacts["comparison"] = _write_comparison_table(
        results,
        config["selection"]["primary_metric"],
        directories["tables"] / f"{run_id}_model_comparison.csv",
    )
    if config.get("outputs", {}).get("figures", True):
        metric_figure = _write_metric_figure(
            results,
            config["selection"]["primary_metric"],
            directories["figures"] / f"{run_id}_metric_comparison.png",
        )
        if metric_figure:
            artifacts["metric_figure"] = metric_figure
    artifacts["errors"] = _write_errors(
        results,
        directories["tables"] / f"{run_id}_errors.csv",
    )
    if config.get("outputs", {}).get("save_predictions", True):
        artifacts.update(
            _write_predictions(results, directories["tables"] / f"{run_id}_predictions.csv")
        )

    if best_result is not None:
        artifacts.update(
            _write_best_outputs(
                best_result,
                run_id,
                directories,
                figures_enabled=config.get("outputs", {}).get("figures", True),
            )
        )
        if config.get("outputs", {}).get("figures", True):
            roc_figure = _write_roc_figure(
                best_result,
                directories["figures"] / f"{run_id}_roc_curves.png",
            )
            if roc_figure:
                artifacts["roc_figure"] = roc_figure

    manifest = {
        "run_id": run_id,
        "experiment_name": experiment_name,
        "config": _public_config(config),
        "best_result": _public_result(best_result) if best_result else None,
        "model_path": model_path,
        "artifacts": artifacts,
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
            row.update({f"metric_{k}": v for k, v in fold.get("metrics", {}).items()})
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


def _write_metric_figure(results, metric: str, path: Path) -> str | None:
    comparison = {}
    for index, result in enumerate(results, start=1):
        if "metrics" not in result or metric not in result["metrics"]:
            continue
        label = result.get("name", result.get("model", "model"))
        if label in comparison:
            label = f"{label}_{index}"
        comparison[label] = {metric: result["metrics"][metric]}
    if not comparison:
        return None
    figure = plot_metric_comparison(comparison, metric=metric)
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return str(path)


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
    y_true = predictions.get("y_true", [])
    y_pred = predictions.get("y_pred", [])
    y_score = predictions.get("y_score")
    for index, (true_value, pred_value) in enumerate(zip(y_true, y_pred)):
        score = None if y_score is None else y_score[index]
        rows.append(
            {
                "model": model_name,
                "fold": fold,
                "row": index,
                "y_true": true_value,
                "y_pred": pred_value,
                "y_score": json.dumps(score, default=str),
            }
        )


def _write_best_outputs(best_result, run_id, directories, *, figures_enabled=True) -> dict[str, str]:
    predictions = best_result.get("predictions")
    if not predictions and best_result.get("folds"):
        predictions = _merge_fold_predictions(best_result["folds"])
    if not predictions:
        return {}

    y_true = np.asarray(predictions["y_true"])
    y_pred = np.asarray(predictions["y_pred"])
    labels = np.unique(np.concatenate([y_true, y_pred]))
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    matrix_path = directories["tables"] / f"{run_id}_{best_result['name']}_confusion_matrix.csv"
    pd.DataFrame(matrix, index=labels, columns=labels).to_csv(matrix_path)

    artifacts = {"confusion_matrix": str(matrix_path)}
    if figures_enabled:
        figure = plot_confusion_matrix(matrix, class_names=[str(label) for label in labels])
        figure_path = directories["figures"] / f"{run_id}_{best_result['name']}_confusion_matrix.png"
        figure.savefig(figure_path, dpi=150)
        plt.close(figure)
        artifacts["confusion_figure"] = str(figure_path)
    return artifacts


def _write_roc_figure(best_result, path: Path) -> str | None:
    predictions = best_result.get("predictions")
    if not predictions and best_result.get("folds"):
        predictions = _merge_fold_predictions(best_result["folds"])
    if not predictions or predictions.get("y_score") is None:
        return None
    figure = plot_roc_curves(
        np.asarray(predictions["y_true"]),
        [np.asarray(predictions["y_score"])],
        model_names=[best_result["name"]],
    )
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return str(path)


def _merge_fold_predictions(folds):
    merged = {"y_true": [], "y_pred": [], "y_score": []}
    has_score = False
    for fold in folds:
        predictions = fold.get("predictions", {})
        merged["y_true"].extend(predictions.get("y_true", []))
        merged["y_pred"].extend(predictions.get("y_pred", []))
        score = predictions.get("y_score")
        if score is not None:
            has_score = True
            merged["y_score"].extend(score)
    if not has_score:
        merged["y_score"] = None
    return merged


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
