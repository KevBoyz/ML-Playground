"""Configured diagnostic views backed by normalized experiment results."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import NearestNeighbors

from ml_playground.data.loader import auto_read
from ml_playground.visualization.barplots import plot_metric_comparison
from ml_playground.visualization.heatmaps import plot_confusion_matrix
from ml_playground.visualization.roc import plot_roc_curves


IMPLEMENTED_VIEWS = {
    "model_comparison",
    "confusion_matrix",
    "roc_curve",
    "predicted_vs_actual",
    "residuals_vs_fitted",
    "elbow_curve",
    "silhouette_curve",
    "k_distance",
    "cluster_scatter",
    "cluster_size",
}


def write_configured_views(config: dict, results: list[dict], best_result: dict | None, run_id: str, directory: Path):
    """Render enabled diagnostic figures and return paths plus per-view status."""

    task = config["task"]
    entries = _configured_entries(config, task)
    if not config.get("outputs", {}).get("figures", True):
        return {}, [{"name": entry["name"], "status": "skipped", "reason": "figures_disabled"} for entry in entries]

    artifacts = {}
    statuses = []
    for entry in entries:
        name = entry["name"]
        if not entry["enabled"]:
            statuses.append({"name": name, "status": "skipped", "reason": "disabled"})
            continue
        if name not in IMPLEMENTED_VIEWS:
            statuses.append({"name": name, "status": "skipped", "reason": "not_implemented"})
            continue
        try:
            path = _render_view(name, entry, config, results, best_result, run_id, directory)
        except (ValueError, KeyError, TypeError) as exc:
            statuses.append({"name": name, "status": "skipped", "reason": str(exc)})
            continue
        except Exception as exc:  # The run remains useful if one optional figure fails.
            statuses.append({"name": name, "status": "failed", "reason": f"{type(exc).__name__}: {exc}"})
            continue
        if path is None:
            statuses.append({"name": name, "status": "skipped", "reason": "insufficient_artifacts"})
            continue
        artifact_name = f"view_{name}"
        artifacts[artifact_name] = str(path)
        statuses.append({"name": name, "status": "generated", "path": str(path)})
    return artifacts, statuses


def merged_predictions(result: dict | None) -> dict | None:
    """Return final-test, canonical OOF or non-duplicated fold predictions.

    Repeated CV produces more than one estimate for the same row.  Visual
    diagnostics must not concatenate those estimates as independent examples,
    otherwise ROC curves, confusion matrices and residual plots are distorted.
    """

    if not result:
        return None
    final_test = result.get("final_test")
    if isinstance(final_test, dict) and final_test.get("predictions"):
        return final_test["predictions"]
    if result.get("oof_predictions"):
        return result["oof_predictions"]
    if result.get("predictions"):
        return result["predictions"]
    folds = result.get("folds", [])
    if not folds:
        return None
    merged = {"row_ids": [], "y_true": [], "y_pred": [], "y_score": []}
    has_score = False
    seen_rows = set()
    for fold in folds:
        predictions = fold.get("predictions", {})
        row_ids = predictions.get("row_ids", [])
        y_true = predictions.get("y_true", [])
        y_pred = predictions.get("y_pred", [])
        y_score = predictions.get("y_score")
        for index, (row_id, true_value, predicted_value) in enumerate(
            zip(row_ids, y_true, y_pred)
        ):
            marker = _row_marker(row_id)
            if marker in seen_rows:
                continue
            seen_rows.add(marker)
            merged["row_ids"].append(row_id)
            merged["y_true"].append(true_value)
            merged["y_pred"].append(predicted_value)
            if y_score is not None:
                has_score = True
                merged["y_score"].append(y_score[index])
    merged["y_score"] = merged["y_score"] if has_score else None
    return merged


def _row_marker(value):
    try:
        hash(value)
    except TypeError:
        return repr(value)
    return value


def _configured_entries(config: dict, task: str) -> list[dict]:
    views = config.get("views") or {}
    entries = list(views.get("common", [])) + list(views.get(task, []))
    if entries:
        return entries
    legacy = [{"name": "model_comparison", "enabled": True, "scope": "candidates", "params": {}}]
    if task == "classification":
        legacy.extend(
            [
                {"name": "confusion_matrix", "enabled": True, "scope": "best", "params": {}},
                {"name": "roc_curve", "enabled": True, "scope": "best", "params": {}},
            ]
        )
    return legacy


def _render_view(name, entry, config, results, best_result, run_id, directory):
    scope = entry["scope"]
    params = entry["params"]
    if name == "model_comparison":
        return _model_comparison(results, config["selection"]["primary_metric"], run_id, directory)
    if name == "elbow_curve":
        return _cluster_metric_curve(results, "inertia", "Elbow curve", run_id, directory)
    if name == "silhouette_curve":
        return _cluster_metric_curve(results, "silhouette", "Silhouette curve", run_id, directory)
    if scope != "best":
        raise ValueError(f"scope {scope!r} ainda não é suportado por {name}")
    if best_result is None:
        raise ValueError("Nenhum candidato vencedor disponível")
    if name == "confusion_matrix":
        return _confusion_matrix(best_result, params, run_id, directory)
    if name == "roc_curve":
        return _roc_curve(best_result, run_id, directory)
    if name == "predicted_vs_actual":
        return _predicted_vs_actual(best_result, run_id, directory)
    if name == "residuals_vs_fitted":
        return _residuals_vs_fitted(best_result, run_id, directory)
    if name == "k_distance":
        return _k_distance(config, best_result, params, run_id, directory)
    if name == "cluster_scatter":
        return _cluster_scatter(config, best_result, params, run_id, directory)
    if name == "cluster_size":
        return _cluster_size(best_result, run_id, directory)
    raise ValueError(f"View não implementada: {name}")


def _model_comparison(results, metric, run_id, directory):
    comparison = {}
    for index, result in enumerate(results, start=1):
        value = result.get("metrics", {}).get(metric)
        if value is None or not np.isscalar(value):
            continue
        name = result.get("name", result.get("model", "model"))
        label = name if name not in comparison else f"{name}_{index}"
        comparison[label] = {metric: value}
    if not comparison:
        return None
    figure = plot_metric_comparison(comparison, metric=metric)
    path = directory / f"{run_id}_model_comparison.png"
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path


def _confusion_matrix(result, params, run_id, directory):
    predictions = merged_predictions(result)
    if not predictions or not predictions.get("y_true"):
        return None
    y_true = np.asarray(predictions["y_true"])
    y_pred = np.asarray(predictions["y_pred"])
    labels = np.unique(np.concatenate([y_true, y_pred]))
    matrix = confusion_matrix(y_true, y_pred, labels=labels, normalize="true" if params.get("normalize") else None)
    figure = plot_confusion_matrix(matrix, class_names=[str(label) for label in labels])
    path = directory / f"{run_id}_{result['name']}_confusion_matrix.png"
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path


def _roc_curve(result, run_id, directory):
    predictions = merged_predictions(result)
    if not predictions or predictions.get("y_score") is None:
        return None
    figure = plot_roc_curves(
        np.asarray(predictions["y_true"]),
        [np.asarray(predictions["y_score"])],
        model_names=[result["name"]],
    )
    path = directory / f"{run_id}_{result['name']}_roc_curve.png"
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path


def _predicted_vs_actual(result, run_id, directory):
    predictions = merged_predictions(result)
    if not predictions or not predictions.get("y_true"):
        return None
    y_true = np.asarray(predictions["y_true"], dtype=float)
    y_pred = np.asarray(predictions["y_pred"], dtype=float)
    figure, axis = plt.subplots(figsize=(7, 6))
    axis.scatter(y_true, y_pred, alpha=0.65)
    lower, upper = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
    axis.plot([lower, upper], [lower, upper], "r--", label="Previsão perfeita")
    axis.set(xlabel="Observado", ylabel="Predito", title="Observado versus predito")
    axis.legend()
    figure.tight_layout()
    path = directory / f"{run_id}_{result['name']}_predicted_vs_actual.png"
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path


def _residuals_vs_fitted(result, run_id, directory):
    predictions = merged_predictions(result)
    if not predictions or not predictions.get("y_true"):
        return None
    y_true = np.asarray(predictions["y_true"], dtype=float)
    y_pred = np.asarray(predictions["y_pred"], dtype=float)
    figure, axis = plt.subplots(figsize=(7, 5))
    axis.scatter(y_pred, y_true - y_pred, alpha=0.65)
    axis.axhline(0, color="black", linestyle="--", linewidth=1)
    axis.set(xlabel="Valores ajustados", ylabel="Resíduos", title="Resíduos versus ajustados")
    figure.tight_layout()
    path = directory / f"{run_id}_{result['name']}_residuals_vs_fitted.png"
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path


def _cluster_metric_curve(results, metric, title, run_id, directory):
    points = []
    for result in results:
        value = result.get("metrics", {}).get(metric)
        if result.get("name") != "kmeans" or value is None:
            continue
        clusters = result.get("params", {}).get("n_clusters")
        if clusters is not None:
            points.append((clusters, value))
    if not points:
        return None
    points.sort(key=lambda item: item[0])
    figure, axis = plt.subplots(figsize=(7, 5))
    axis.plot([point[0] for point in points], [point[1] for point in points], marker="o")
    axis.set(xlabel="Número de clusters (k)", ylabel=metric, title=title)
    figure.tight_layout()
    path = directory / f"{run_id}_{metric}_curve.png"
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path


def _k_distance(config, result, params, run_id, directory):
    X = _cluster_features(config)
    transformed = result["pipeline"].named_steps["preprocessing"].transform(X)
    n_neighbors = int(params.get("n_neighbors", result.get("params", {}).get("min_samples", 5)))
    if n_neighbors < 2 or n_neighbors > len(X):
        raise ValueError("n_neighbors deve estar entre 2 e o número de linhas")
    distances, _ = NearestNeighbors(n_neighbors=n_neighbors).fit(transformed).kneighbors(transformed)
    values = np.sort(distances[:, -1])
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.plot(values)
    if params.get("eps") is not None:
        axis.axhline(float(params["eps"]), color="red", linestyle="--", label="eps configurado")
        axis.legend()
    axis.set(
        xlabel="Pontos ordenados",
        ylabel=f"Distância ao {n_neighbors}º vizinho",
        title="Curva de k-distância",
    )
    figure.tight_layout()
    path = directory / f"{run_id}_{result['name']}_k_distance.png"
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path


def _cluster_scatter(config, result, params, run_id, directory):
    X = _cluster_features(config)
    columns = params.get("features") or list(X.columns[:2])
    if not isinstance(columns, list) or len(columns) != 2 or any(column not in X.columns for column in columns):
        raise ValueError("cluster_scatter exige duas features existentes")
    labels = np.asarray(merged_predictions(result).get("labels", []))
    if len(labels) != len(X):
        raise ValueError("Labels e dataset possuem tamanhos diferentes")
    figure, axis = plt.subplots(figsize=(7, 6))
    scatter = axis.scatter(X[columns[0]], X[columns[1]], c=labels, cmap="tab10", alpha=0.7)
    figure.colorbar(scatter, ax=axis, label="Cluster")
    axis.set(xlabel=columns[0], ylabel=columns[1], title="Dispersão por cluster")
    figure.tight_layout()
    path = directory / f"{run_id}_{result['name']}_cluster_scatter.png"
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path


def _cluster_size(result, run_id, directory):
    predictions = merged_predictions(result)
    labels = predictions.get("labels", []) if predictions else []
    if not labels:
        return None
    counts = Counter(labels)
    figure, axis = plt.subplots(figsize=(7, 5))
    names = ["Ruído (-1)" if label == -1 else f"Cluster {label}" for label in sorted(counts)]
    axis.bar(names, [counts[label] for label in sorted(counts)])
    axis.set(ylabel="Linhas", title="Tamanho dos clusters")
    axis.tick_params(axis="x", labelrotation=25)
    figure.tight_layout()
    path = directory / f"{run_id}_{result['name']}_cluster_size.png"
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path


def _cluster_features(config):
    prepared = config.get("_prepared_data")
    if isinstance(prepared, dict) and prepared.get("X") is not None:
        return prepared["X"]
    frame = auto_read(config["data"]["path"])
    return frame.select(config["data"]["features"]).to_pandas()
