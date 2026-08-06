"""Centralized persistence for task-aware, traceable experiment outputs."""

from __future__ import annotations

import hashlib
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

from ml_playground.experiments.tracker import RUN_CONTRACT_VERSION, build_run_context
from ml_playground.experiments.views import merged_predictions, write_configured_views


REPORT_CONTRACT_VERSION = RUN_CONTRACT_VERSION


def results_to_csv(results, path):
    """Flatten scalar result fields into a tabular CSV with trial identities."""

    traced = _traced_results(results)
    return _write_dataframe(_summary_rows(traced), Path(path))


def write_experiment_reports(
    config: dict,
    bundle: dict,
    run_id: str,
    *,
    best_result: dict | None = None,
    model_path: str | None = None,
) -> dict[str, str]:
    """Write task-aware outputs under ``reports/<experiment_name>/``.

    Returned paths retain the legacy absolute-path API.  The manifest additionally
    publishes a relative, checksummed inventory for portable consumption.
    """

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

    original_results = bundle.get("results", [])
    results = _traced_results(original_results)
    traced_best = _matching_traced_result(best_result, original_results, results)
    selection = config.get("selection", {})
    primary_metric = selection.get("primary_metric", "accuracy")

    artifacts = {
        "summary": results_to_csv(results, directories["metrics"] / f"{run_id}_summary.csv"),
        "fold_metrics": _write_fold_metrics(
            results,
            directories["metrics"] / f"{run_id}_fold_metrics.csv",
        ),
        "comparison": _write_comparison_table(
            results,
            primary_metric,
            directories["tables"] / f"{run_id}_model_comparison.csv",
        ),
    }
    errors_path = _write_errors(results, directories["tables"] / f"{run_id}_errors.csv")
    if errors_path is not None:
        artifacts["errors"] = errors_path
    if config.get("outputs", {}).get("save_predictions", True):
        artifacts.update(
            _write_predictions(
                results,
                directories["predictions"] / f"{run_id}_predictions.csv",
                formats=_prediction_formats(config.get("outputs", {})),
                selected_candidate_id=traced_best.get("candidate_id") if traced_best else None,
                save_all_candidates=bool(
                    config.get("outputs", {}).get("save_all_candidate_predictions", False)
                ),
            )
        )
    if traced_best is not None:
        artifacts.update(
            _write_task_tables(config["task"], traced_best, run_id, directories["tables"])
        )

    view_artifacts, view_status = write_configured_views(
        config,
        results,
        traced_best,
        run_id,
        directories["figures"],
    )
    artifacts.update(view_artifacts)

    run_context = _resolve_run_context(config, bundle)
    model_card = _write_model_card(
        config,
        traced_best,
        run_id,
        model_path,
        report_root,
        directories["tables"],
        run_context,
    )
    artifacts["model_card"] = str(model_card)

    relative_artifacts = {
        name: _relative_path(path, report_root)
        for name, path in artifacts.items()
        if isinstance(path, (str, Path))
    }
    inventory = _artifact_inventory(artifacts, report_root)
    model_descriptor = _file_descriptor(model_path, report_root) if model_path else None
    manifest = {
        "contract_version": REPORT_CONTRACT_VERSION,
        "run_id": run_id,
        "experiment_name": experiment_name,
        "task": config["task"],
        "run_context": run_context,
        "config": _public_config(config),
        "best_result": _public_result(traced_best) if traced_best else None,
        "selected_model": _selected_model_descriptor(traced_best),
        "final_test": _public_result((traced_best or {}).get("final_test")),
        # Backward-compatible absolute paths for existing API consumers.
        "model_path": model_path,
        "artifacts": artifacts,
        # Portable and integrity-aware representation for external consumers.
        "paths": {
            "report_root": ".",
            "model_path": _relative_path(model_path, report_root) if model_path else None,
            "artifacts": relative_artifacts,
        },
        "artifact_inventory": inventory,
        "checksums": {name: entry["sha256"] for name, entry in inventory.items()},
        "model_artifact": model_descriptor,
        "views": view_status,
    }
    manifest_path = directories["tables"] / f"{run_id}_manifest.json"
    _write_json(manifest, manifest_path)
    artifacts["manifest"] = str(manifest_path)
    return artifacts


def _traced_results(results: list[dict]) -> list[dict]:
    occurrences: Counter[str] = Counter()
    traced = []
    for result in results:
        item = dict(result)
        candidate_id = item.get("candidate_id")
        if not candidate_id:
            fingerprint = _stable_hash(
                {
                    "model": item.get("name", item.get("model", "?")),
                    "params": item.get("params", {}),
                }
            )[:12]
            base = f"candidate-{fingerprint}"
            occurrences[base] += 1
            candidate_id = f"{base}-{occurrences[base]:02d}"
        item["candidate_id"] = str(candidate_id)
        item["trial_id"] = str(item.get("trial_id") or item["candidate_id"])
        item["split_id"] = str(item.get("split_id") or _aggregate_split_id(item))
        traced.append(item)
    return traced


def _stable_hash(value: Any) -> str:
    serialized = json.dumps(value, sort_keys=True, default=str, ensure_ascii=False)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _aggregate_split_id(result: dict) -> str:
    if result.get("folds"):
        return "cross_validation"
    if result.get("predictions"):
        return "holdout"
    return "aggregate"


def _matching_traced_result(best_result, original_results, traced_results):
    if best_result is None:
        return None
    for original, traced in zip(original_results, traced_results):
        if original is best_result:
            return traced
    provided_id = best_result.get("candidate_id") if isinstance(best_result, dict) else None
    if provided_id:
        for traced in traced_results:
            if traced["candidate_id"] == str(provided_id):
                return traced
    if isinstance(best_result, dict):
        for traced in traced_results:
            if (
                traced.get("name", traced.get("model")) == best_result.get("name", best_result.get("model"))
                and traced.get("params", {}) == best_result.get("params", {})
            ):
                return traced
        return _traced_results([best_result])[0]
    return best_result


def _summary_rows(results: list[dict]) -> list[dict]:
    rows = []
    for result in results:
        row = {
            "candidate_id": result["candidate_id"],
            "trial_id": result["trial_id"],
            "split_id": result["split_id"],
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
        for key, value in (result.get("final_test") or {}).get("metrics", {}).items():
            if np.isscalar(value):
                row[f"final_test_metric_{key}"] = value
        if "error" in result:
            row["error"] = result["error"]
        rows.append(row)
    return rows


def _write_fold_metrics(results, path: Path) -> str:
    rows = []
    for result in results:
        for fold_index, fold in enumerate(result.get("folds", []), start=1):
            row = {
                "candidate_id": result["candidate_id"],
                "trial_id": str(fold.get("trial_id") or result["trial_id"]),
                "split_id": _fold_split_id(result, fold, fold_index),
                "model": result.get("name", result.get("model", "?")),
                "fold": fold.get("fold", fold_index),
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
                "candidate_id": result["candidate_id"],
                "trial_id": result["trial_id"],
                "split_id": result["split_id"],
                "model": result.get("name", result.get("model", "?")),
                "params": json.dumps(result.get("params", {}), sort_keys=True, default=str),
                "metric": metric,
                "value": result["metrics"].get(metric),
                "status": result.get("status", "success"),
            }
        )
    return _write_dataframe(rows, path)


def _write_errors(results, path: Path) -> str | None:
    rows = [
        {
            "candidate_id": result["candidate_id"],
            "trial_id": result["trial_id"],
            "split_id": result["split_id"],
            "model": result.get("name", result.get("model", "?")),
            "params": json.dumps(result.get("params", {}), sort_keys=True, default=str),
            "error": result.get("error"),
        }
        for result in results
        if "error" in result
    ]
    if not rows:
        return None
    return _write_dataframe(rows, path)


def _prediction_formats(outputs: dict) -> set[str]:
    configured = outputs.get("predictions_format", outputs.get("prediction_format", "csv"))
    if isinstance(configured, str):
        configured = [configured]
    if not isinstance(configured, (list, tuple, set)):
        raise ValueError("outputs.predictions_format deve ser csv, parquet ou uma lista")

    formats = {str(value).casefold() for value in configured}
    if "both" in formats:
        formats.remove("both")
        formats.update({"csv", "parquet"})
    if outputs.get("save_predictions_parquet"):
        formats.add("parquet")
    if not formats or not formats.issubset({"csv", "parquet"}):
        raise ValueError("outputs.predictions_format aceita somente csv, parquet ou both")
    return formats


def _write_predictions(
    results,
    path: Path,
    *,
    formats: set[str],
    selected_candidate_id: str | None,
    save_all_candidates: bool,
) -> dict[str, str]:
    rows = []
    for result in results:
        if not save_all_candidates and result.get("candidate_id") != selected_candidate_id:
            continue
        final_test = result.get("final_test")
        if isinstance(final_test, dict) and final_test.get("predictions"):
            _append_prediction_rows(rows, result, final_test, "final_test")
        elif result.get("oof_predictions"):
            _append_oof_prediction_rows(rows, result, result["oof_predictions"])
        elif "predictions" in result:
            _append_prediction_rows(rows, result, None, result["split_id"])
        elif save_all_candidates:
            for fold_index, fold in enumerate(result.get("folds", []), start=1):
                _append_prediction_rows(
                    rows,
                    result,
                    fold,
                    _fold_split_id(result, fold, fold_index),
                )

    frame = pd.DataFrame(rows)
    if frame.empty:
        frame = pd.DataFrame(
            columns=["candidate_id", "trial_id", "split_id", "model", "fold", "row"]
        )
    artifacts: dict[str, str] = {}
    if "csv" in formats:
        csv_path = _write_dataframe(frame.to_dict("records"), path)
        artifacts["predictions_csv"] = csv_path
        artifacts["predictions"] = csv_path
    if "parquet" in formats:
        parquet_path = path.with_suffix(".parquet")
        frame.to_parquet(parquet_path, index=False)
        artifacts["predictions_parquet"] = str(parquet_path)
        artifacts.setdefault("predictions", str(parquet_path))
    return artifacts


def _append_prediction_rows(rows, result: dict, fold: dict | None, split_id: str) -> None:
    predictions = fold.get("predictions") if fold is not None else result.get("predictions")
    if not predictions:
        return
    base = {
        "candidate_id": result["candidate_id"],
        "trial_id": str((fold or {}).get("trial_id") or result["trial_id"]),
        "split_id": split_id,
        "split_role": (fold or {}).get("split_role", "development"),
        "model": result.get("name", result.get("model", "?")),
        "fold": (fold or {}).get("fold"),
    }
    row_ids = predictions.get("row_ids", [])
    if predictions.get("labels") is not None:
        metadata = predictions.get("metadata") or {}
        for index, label in enumerate(predictions["labels"]):
            row = row_ids[index] if index < len(row_ids) else index
            rows.append({**base, "row": row, "cluster": label, **_metadata_row(metadata, index)})
        return

    y_true = predictions.get("y_true", [])
    y_pred = predictions.get("y_pred", [])
    y_score = predictions.get("y_score")
    y_proba = predictions.get("y_proba")
    metadata = predictions.get("metadata") or {}
    for index, (true_value, pred_value) in enumerate(zip(y_true, y_pred)):
        score = None if y_score is None else y_score[index]
        proba = None if y_proba is None else y_proba[index]
        row = row_ids[index] if index < len(row_ids) else index
        rows.append(
            {
                **base,
                "row": row,
                "y_true": true_value,
                "y_pred": pred_value,
                "y_score": json.dumps(score, default=str),
                "y_proba": json.dumps(proba, default=str),
                **_metadata_row(metadata, index),
            }
        )


def _append_oof_prediction_rows(rows, result: dict, predictions: dict) -> None:
    row_ids = predictions.get("row_ids", [])
    y_true = predictions.get("y_true", [])
    y_pred = predictions.get("y_pred", [])
    scores = predictions.get("y_score")
    probabilities = predictions.get("y_proba")
    folds = predictions.get("fold")
    split_ids = predictions.get("split_id")
    roles = predictions.get("split_role")
    scores = [None] * len(row_ids) if scores is None else scores
    probabilities = [None] * len(row_ids) if probabilities is None else probabilities
    folds = [None] * len(row_ids) if folds is None else folds
    split_ids = ["development_oof"] * len(row_ids) if split_ids is None else split_ids
    roles = ["development_oof"] * len(row_ids) if roles is None else roles
    metadata = predictions.get("metadata") or {}
    for index, (row_id, true_value, pred_value) in enumerate(zip(row_ids, y_true, y_pred)):
        rows.append(
            {
                "candidate_id": result["candidate_id"],
                "trial_id": result["trial_id"],
                "split_id": split_ids[index],
                "split_role": roles[index],
                "model": result.get("name", result.get("model", "?")),
                "fold": folds[index],
                "row": row_id,
                "y_true": true_value,
                "y_pred": pred_value,
                "y_score": json.dumps(scores[index], default=str),
                "y_proba": json.dumps(probabilities[index], default=str),
                **_metadata_row(metadata, index),
            }
        )


def _metadata_row(metadata: dict, index: int) -> dict:
    return {
        f"metadata_{name}": values[index] if index < len(values) else None
        for name, values in metadata.items()
        if isinstance(values, (list, tuple, np.ndarray, pd.Series))
    }


def _fold_split_id(result: dict, fold: dict, fold_index: int) -> str:
    return str(fold.get("split_id") or f"{result['split_id']}-fold-{fold.get('fold', fold_index)}")


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
    frame.insert(0, "split_id", result["split_id"])
    frame.insert(0, "trial_id", result["trial_id"])
    frame.insert(0, "candidate_id", result["candidate_id"])
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
        {
            "candidate_id": result["candidate_id"],
            "trial_id": result["trial_id"],
            "split_id": result["split_id"],
            "cluster": label,
            "size": size,
            "is_noise": label == -1,
        }
        for label, size in sorted(counts.items())
    ]
    path = directory / f"{run_id}_{result['name']}_cluster_sizes.csv"
    _write_dataframe(rows, path)
    return {"cluster_sizes": str(path)}


def _resolve_run_context(config: dict, bundle: dict) -> dict:
    tracker = bundle.get("tracker")
    if isinstance(tracker, dict) and isinstance(tracker.get("run_context"), dict):
        return dict(tracker["run_context"])
    return build_run_context(config)


def _write_model_card(
    config: dict,
    best_result: dict | None,
    run_id: str,
    model_path: str | None,
    report_root: Path,
    directory: Path,
    run_context: dict,
) -> Path:
    card_config = config.get("model_card") or {}
    if not isinstance(card_config, dict):
        raise TypeError("model_card deve ser um mapa quando informado")
    data_config = config.get("data", {})
    validation = config.get("cross_validation", {})
    path = directory / f"{run_id}_model_card.md"
    lines = [
        f"# Model card — {config.get('experiment_name', 'experiment')}",
        "",
        "## Identificação",
        "",
        f"- Run: `{run_id}`",
        f"- Tarefa: `{config.get('task', 'unknown')}`",
        f"- Contrato de artefatos: `{REPORT_CONTRACT_VERSION}`",
        f"- Modelo persistido: `{_relative_path(model_path, report_root) if model_path else 'não salvo'}`",
        "",
        "## Uso pretendido",
        "",
        str(card_config.get("intended_use", "Comparação reproduzível de modelos tabulares no escopo deste experimento.")),
        "",
        "## Dados e validação",
        "",
        f"- Dataset: `{_relative_path(data_config.get('path'), report_root) if data_config.get('path') else 'não informado'}`",
        f"- Target: `{data_config.get('target', 'não aplicável')}`",
        f"- Features declaradas: `{', '.join(_feature_list(data_config.get('features'))) or 'inferidas no treino'}`",
        f"- Estratégia de validação: `{json.dumps(validation, ensure_ascii=False, default=str)}`",
        f"- Protocolo de avaliação: `{json.dumps(config.get('evaluation', {}), ensure_ascii=False, default=str)}`",
        "",
        "## Candidato selecionado",
        "",
    ]
    if best_result is None:
        lines.append("Nenhum candidato válido foi selecionado.")
    else:
        lines.extend(
            [
                f"- Candidate ID: `{best_result.get('candidate_id', 'não informado')}`",
                f"- Trial ID: `{best_result.get('trial_id', 'não informado')}`",
                f"- Modelo: `{best_result.get('name', best_result.get('model', 'não informado'))}`",
                "- Parâmetros:",
                "",
                "```json",
                json.dumps(best_result.get("params", {}), indent=2, ensure_ascii=False, default=str),
                "```",
                "",
                "### Métricas",
                "",
            ]
        )
        metric_rows = [
            (name, value)
            for name, value in best_result.get("metrics", {}).items()
            if np.isscalar(value)
        ]
        if metric_rows:
            lines.extend(["| Métrica | Valor |", "| --- | ---: |"])
            lines.extend(f"| {name} | {float(value):.8g} |" for name, value in metric_rows)
        else:
            lines.append("Não há métricas escalares disponíveis.")
        final_metric_rows = [
            (name, value)
            for name, value in (best_result.get("final_test") or {}).get("metrics", {}).items()
            if np.isscalar(value)
        ]
        lines.extend(["", "### Teste final bloqueado", ""])
        if final_metric_rows:
            lines.extend(["| Métrica | Valor |", "| --- | ---: |"])
            lines.extend(f"| {name} | {float(value):.8g} |" for name, value in final_metric_rows)
        else:
            lines.append("Nenhum teste final foi configurado ou avaliado; as métricas acima são de desenvolvimento.")

    limitations = card_config.get("limitations")
    if isinstance(limitations, str):
        limitations = [limitations]
    if not isinstance(limitations, list):
        limitations = [
            "As métricas descrevem somente o protocolo de validação registrado; valide em dados externos antes de uso operacional.",
            "A qualidade da inferência depende de o lote respeitar a assinatura e a proveniência documentadas.",
        ]
    lines.extend(["", "## Limitações e cuidados", ""])
    lines.extend(f"- {item}" for item in limitations)

    context_keys = (
        "git_sha",
        "config_sha256",
        "config_fingerprint",
        "run_fingerprint",
        "dataset_sha256",
        "data_fingerprint",
        "schema_signature",
    )
    trace = {key: run_context[key] for key in context_keys if key in run_context}
    if trace:
        lines.extend(["", "## Rastreabilidade", "", "```json", json.dumps(trace, indent=2, ensure_ascii=False), "```"])
    notes = card_config.get("notes")
    if notes:
        lines.extend(["", "## Notas", "", str(notes)])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _relative_path(path: str | Path | None, root: Path) -> str | None:
    if path is None:
        return None
    return Path(os.path.relpath(Path(path).resolve(), root.resolve())).as_posix()


def _feature_list(features: Any) -> list[str]:
    """Flatten either legacy feature lists or the grouped config form."""

    if isinstance(features, (list, tuple)):
        return [feature for feature in features if isinstance(feature, str) and feature]
    if isinstance(features, dict):
        return [
            feature
            for group in features.values()
            if isinstance(group, (list, tuple))
            for feature in group
            if isinstance(feature, str) and feature
        ]
    return []


def _artifact_inventory(artifacts: dict[str, str], root: Path) -> dict[str, dict[str, Any]]:
    inventory = {}
    for name, path in artifacts.items():
        descriptor = _file_descriptor(path, root)
        if descriptor is not None:
            inventory[name] = descriptor
    return inventory


def _file_descriptor(path: str | Path | None, root: Path) -> dict[str, Any] | None:
    if path is None:
        return None
    file_path = Path(path)
    if not file_path.is_file():
        return None
    return {
        "path": _relative_path(file_path, root),
        "sha256": _sha256_file(file_path),
        "bytes": file_path.stat().st_size,
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
    return {
        key: value
        for key, value in config.items()
        if key not in {"model", "preflight_metadata"} and not key.startswith("_")
    }


def _public_result(result):
    if result is None:
        return None
    if isinstance(result, dict):
        return {
            key: _public_result(value)
            for key, value in result.items()
            if key != "pipeline" and not str(key).startswith("_")
        }
    if isinstance(result, list):
        return [_public_result(value) for value in result]
    if isinstance(result, tuple):
        return [_public_result(value) for value in result]
    return result


def _selected_model_descriptor(best_result: dict | None) -> dict | None:
    if best_result is None:
        return None
    return {
        "candidate_id": best_result.get("candidate_id"),
        "trial_id": best_result.get("trial_id"),
        "name": best_result.get("name", best_result.get("model")),
        "params": best_result.get("params", {}),
        "development_metrics": best_result.get("metrics", {}),
        "final_test_metrics": (best_result.get("final_test") or {}).get("metrics", {}),
    }
