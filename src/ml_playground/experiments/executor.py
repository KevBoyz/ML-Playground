"""Task-aware execution with reproducible evaluation protocols.

The original public entry point, :func:`run_experiment`, remains intentionally
small.  The implementation below owns the pieces that need to agree with one
another in a real training run: data roles, split planning, fold-local fitting,
and the boundary between development evidence and a blocked final test.
"""

from __future__ import annotations

from copy import deepcopy
from math import ceil
import time
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.model_selection import (
    GroupKFold,
    GroupShuffleSplit,
    KFold,
    RepeatedKFold,
    RepeatedStratifiedKFold,
    StratifiedGroupKFold,
    StratifiedKFold,
    TimeSeriesSplit,
    train_test_split,
)
from sklearn.preprocessing import PowerTransformer

from ml_playground.data.loader import auto_read
from ml_playground.evaluation.metrics import (
    compute_clustering_metrics,
    compute_metrics,
    normalize_metric_specs,
)
from ml_playground.models import get_model
from ml_playground.preprocessing.pipelines import build_model_pipeline


_HOLDOUT_METHODS = {"holdout", "group_holdout", "temporal_holdout"}
_TIME_METHODS = {"time_series", "time_series_split", "backtest", "temporal_holdout"}
_METHOD_ALIASES = {
    "stratifiedgroupkfold": "stratified_group_kfold",
    "stratified-group-kfold": "stratified_group_kfold",
    "groupkfold": "group_kfold",
    "group-shuffle-split": "group_holdout",
    "group_shuffle_split": "group_holdout",
    "timeseriessplit": "time_series",
    "time_series_split": "time_series",
    "time-series-split": "time_series",
    "time-series": "time_series",
    "timeseries": "time_series",
    "temporal": "temporal_holdout",
}


def run_experiment(config: dict) -> dict:
    """Run one model configuration for a supported task.

    Legacy configurations still use ``cross_validation`` and return their
    familiar ``metrics``/``folds`` fields.  A normalized ``evaluation`` section
    may additionally reserve an external or held-out final test; those metrics
    are stored under ``final_test`` and never replace development metrics.
    """

    started = time.perf_counter()
    task = config.get("task", "classification")
    if task == "clustering":
        result = _run_clustering(config)
    elif task in {"classification", "regression"}:
        result = _run_supervised(config, task)
    else:
        raise ValueError(f"Tarefa não suportada: {task}")

    model_config = config.get("model") or {"name": "nested_selection", "params": {}}
    result["status"] = "success"
    result["duration_seconds"] = time.perf_counter() - started
    result["model"] = model_config["name"]
    result["params"] = model_config.get("params", {})
    result["task"] = task
    return result


def fit_full_pipeline(config: dict):
    """Fit the selected deploy pipeline on development rows only by default.

    When an explicit final split was configured, the final-test rows remain out
    of this refit unless the caller deliberately sets
    ``evaluation.deployment_include_final_test``.  This prevents the old
    implicit full-data refit from silently making a held-out test part of the
    persisted artifact.
    """

    task = config.get("task", "classification")
    bundle = _load_data_bundle(config, task)
    if task == "clustering":
        return _fit_pipeline(config, bundle["X"], None)
    evaluation = _resolve_evaluation(config, bundle)
    development, final_test, _ = _partition_final_test(bundle, evaluation, config, task)
    include_final = bool(evaluation.get("deployment_include_final_test", False))
    if include_final and final_test is not None and final_test.get("y") is not None:
        development = _concat_bundles(development, final_test)
    return _fit_pipeline(config, development["X"], development["y"])


def _run_supervised(config: dict, task: str) -> dict:
    bundle = _load_data_bundle(config, task)
    evaluation = _resolve_evaluation(config, bundle)
    development, final_test, final_metadata = _partition_final_test(bundle, evaluation, config, task)
    metric_specs = config.get("metric_specs", config.get("metrics", []))

    protocol = evaluation["protocol"]
    if protocol == "nested_cv":
        result = _run_nested_cv(config, development, metric_specs, evaluation, task)
    else:
        splitter_config = _development_splitter_config(config, evaluation)
        result = _run_development_splits(
            config,
            development,
            metric_specs,
            splitter_config,
            task,
            split_role=_development_role(splitter_config),
        )

    evaluation_result = dict(result.pop("evaluation", {}))
    evaluation_result.update(
        {
            "protocol": protocol,
            "development_metric_origin": evaluation_result.get(
                "metric_origin", "development_cv"
            ),
            "final_test": {
                "configured": final_test is not None,
                "source": final_metadata.get("source") if final_metadata else None,
                "evaluated": False,
            },
        }
    )
    if final_metadata:
        result.setdefault("split_metadata", {}).setdefault("final_test", final_metadata)

    should_evaluate_final = bool(evaluation.get("evaluate_final_test", final_test is not None))
    if final_test is not None and should_evaluate_final:
        final_result = _evaluate_final_test(config, development, final_test, metric_specs, task)
        result["final_test"] = final_result
        evaluation_result["final_test"].update(
            {"evaluated": True, "metric_origin": "final_test"}
        )
    elif final_test is not None:
        evaluation_result["final_test"]["reason"] = "deferred_for_model_selection"

    result["evaluation"] = evaluation_result
    return result


def _resolve_evaluation(config: dict, bundle: dict) -> dict:
    """Compose the new evaluation contract with the legacy CV section."""

    raw = dict(config.get("evaluation") or {})
    legacy_cv = dict(config.get("cross_validation") or config.get("validation") or {})
    splitter = dict(raw.get("splitter") or raw.get("development_splitter") or legacy_cv)
    if "name" in splitter and "method" not in splitter:
        splitter["method"] = splitter["name"]
    if not splitter:
        splitter = {"method": "holdout"}

    final_test = raw.get("final_test")
    if final_test is None:
        final_test = {}
    elif not isinstance(final_test, dict):
        raise ValueError("evaluation.final_test deve ser um mapa")
    else:
        final_test = dict(final_test)

    has_external_test = bundle.get("test_X") is not None
    protocol = raw.get("protocol")
    if protocol is None:
        protocol = "development_cv_final_test" if has_external_test else "development"
    aliases = {
        "cv": "development",
        "development_cv": "development",
        "train_validation_test": "train_validation_test",
        "cv_final_test": "development_cv_final_test",
        "development_cv_final_test": "development_cv_final_test",
        "nested": "nested_cv",
    }
    protocol = aliases.get(protocol, protocol)
    if protocol not in {
        "development",
        "train_validation_test",
        "development_cv_final_test",
        "nested_cv",
    }:
        raise ValueError(f"Protocolo de avaliação não suportado: {protocol}")

    if has_external_test and "source" not in final_test:
        final_test["source"] = "path"
    if protocol == "train_validation_test" and "source" not in final_test:
        final_test["source"] = "split"
    if protocol == "development" and not has_external_test and not final_test:
        final_test["source"] = "none"
    final_test.setdefault("source", "none")

    return {
        **raw,
        "protocol": protocol,
        "splitter": splitter,
        "final_test": final_test,
    }


def _development_splitter_config(config: dict, evaluation: dict) -> dict:
    splitter = dict(evaluation.get("splitter") or {})
    if not splitter:
        splitter = dict(config.get("cross_validation") or config.get("validation") or {})
    if not splitter:
        splitter = {"method": "holdout"}
    if "name" in splitter and "method" not in splitter:
        splitter["method"] = splitter["name"]
    if evaluation["protocol"] == "train_validation_test" and not (
        config.get("cross_validation") or config.get("validation") or evaluation.get("splitter")
    ):
        splitter = {"method": "holdout", "test_size": evaluation.get("validation_size", 0.2)}
    return splitter


def _development_role(splitter_config: dict) -> str:
    """Name the evaluation evidence without conflating holdout and CV."""

    method = _normalise_method(splitter_config.get("method", splitter_config.get("name", "holdout")))
    return "development_validation" if method in _HOLDOUT_METHODS else "development_cv"


def _partition_final_test(bundle: dict, evaluation: dict, config: dict, task: str):
    """Reserve the final test before any development split is generated."""

    final_config = dict(evaluation.get("final_test") or {})
    source = str(final_config.get("source", "none")).lower()
    if source in {"none", "disabled", "false"}:
        return bundle, None, None
    if source in {"path", "external", "data.test", "test"}:
        if bundle.get("test_X") is None:
            raise ValueError("evaluation.final_test.source=path exige data.test com dataset válido")
        if bundle.get("test_y") is None:
            raise ValueError("O dataset de teste final deve conter a coluna target")
        final = _test_bundle(bundle)
        return bundle, final, {
            "version": 1,
            "source": "path",
            "split_role": "final_test",
            "row_ids": _json_values(final["row_ids"]),
            "n_test": len(final["X"]),
        }
    if source not in {"split", "holdout", "reserved_split"}:
        raise ValueError(f"Fonte de teste final não suportada: {source}")

    split_config = dict(final_config.get("splitter") or final_config)
    split_config.pop("source", None)
    split_config.setdefault("method", final_config.get("method", "holdout"))
    split_config.setdefault(
        "test_size",
        final_config.get("test_size", evaluation.get("final_test_size", 0.2)),
    )
    plan = _make_split_plan(bundle, split_config, config, task, split_role="final_test")
    if len(plan["splits"]) != 1:
        raise ValueError("O teste final por split deve usar uma única partição holdout")
    descriptor = plan["splits"][0]
    development = _subset_bundle(bundle, descriptor["train_idx"])
    final = _subset_bundle(bundle, descriptor["test_idx"])
    metadata = _public_split_metadata(descriptor, bundle, plan["metadata"])
    metadata["source"] = "split"
    return development, final, metadata


def _run_development_splits(
    config: dict,
    bundle: dict,
    metric_specs,
    split_config: dict,
    task: str,
    *,
    split_role: str,
) -> dict:
    plan = _make_split_plan(bundle, split_config, config, task, split_role=split_role)
    folds = []
    score_lists: dict[str, list[float]] = {}
    oof = _empty_oof()
    last_pipeline = None

    for descriptor in plan["splits"]:
        train_bundle = _subset_bundle(bundle, descriptor["train_idx"])
        test_bundle = _subset_bundle(bundle, descriptor["test_idx"])
        pipeline = _fit_pipeline(config, train_bundle["X"], train_bundle["y"])
        last_pipeline = pipeline
        y_pred, y_score, y_proba, class_labels = _predict_outputs(pipeline, test_bundle["X"])
        fold_metrics = compute_metrics(
            test_bundle["y"],
            y_pred,
            metric_specs,
            task=task,
            y_score=y_score,
            y_proba=y_proba,
            class_labels=class_labels,
        )
        for metric, value in fold_metrics.items():
            if np.isscalar(value):
                score_lists.setdefault(metric, []).append(float(value))
        predictions = _prediction_payload(
            test_bundle,
            y_pred,
            y_score,
            y_proba,
            class_labels,
        )
        fold = {
            "fold": descriptor["fold"],
            "fold_in_repeat": descriptor["fold_in_repeat"],
            "repeat": descriptor["repeat"],
            "split_id": descriptor["split_id"],
            "split_role": split_role,
            "n_train": len(train_bundle["X"]),
            "n_test": len(test_bundle["X"]),
            "metrics": _scalar_metrics(fold_metrics),
            "predictions": predictions,
        }
        folds.append(fold)
        _extend_oof(oof, predictions, descriptor, split_role)

    averages = {metric: float(np.mean(values)) for metric, values in score_lists.items() if values}
    stds = {metric: float(np.std(values, ddof=0)) for metric, values in score_lists.items() if values}
    split_metadata = {
        "version": 1,
        "development": {
            **plan["metadata"],
            "splits": [
                _public_split_metadata(descriptor, bundle, plan["metadata"])
                for descriptor in plan["splits"]
            ],
        },
    }
    is_holdout = len(plan["splits"]) == 1 and plan["metadata"]["method"] in _HOLDOUT_METHODS
    if is_holdout:
        only_fold = folds[0]
        return {
            "metrics": averages,
            "predictions": only_fold["predictions"],
            "pipeline": last_pipeline,
            "n_train": only_fold["n_train"],
            "n_test": only_fold["n_test"],
            "split_metadata": split_metadata,
            "evaluation": {
                "metric_origin": "development_validation",
                "aggregation": "single_holdout",
                "oof_semantics": "one validation prediction per held-out row",
            },
        }
    return {
        "metrics": averages,
        "metric_std": stds,
        "cv_scores": {metric: [float(value) for value in values] for metric, values in score_lists.items()},
        "folds": folds,
        "oof_predictions": oof,
        "split_metadata": split_metadata,
        "evaluation": {
            "metric_origin": "development_cv",
            "aggregation": "mean_over_folds",
            "oof_semantics": "one prediction per row, fold and repeat; repeats are not independent rows",
            "n_repeats": plan["metadata"]["n_repeats"],
        },
    }


def _evaluate_final_test(config: dict, development: dict, final_test: dict, metric_specs, task: str) -> dict:
    pipeline = _fit_pipeline(config, development["X"], development["y"])
    y_pred, y_score, y_proba, class_labels = _predict_outputs(pipeline, final_test["X"])
    metrics = compute_metrics(
        final_test["y"],
        y_pred,
        metric_specs,
        task=task,
        y_score=y_score,
        y_proba=y_proba,
        class_labels=class_labels,
    )
    return {
        "metrics": _scalar_metrics(metrics),
        "predictions": _prediction_payload(final_test, y_pred, y_score, y_proba, class_labels),
        "n_train": len(development["X"]),
        "n_test": len(final_test["X"]),
        "split_role": "final_test",
        # It is intentionally named separately to keep report serialization
        # from treating a final-test fitted object as a candidate-fold model.
        "pipeline": pipeline,
    }


def _run_nested_cv(config: dict, bundle: dict, metric_specs, evaluation: dict, task: str) -> dict:
    """Evaluate a candidate set with inner selection and outer scoring.

    Passing just ``model`` is valid (the inner selection then has one
    candidate).  Passing ``models`` or ``evaluation.candidates`` makes this a
    true model/parameter-selection nested CV run without requiring the runner
    to reuse outer-test evidence for global selection.
    """

    outer_config = dict(
        evaluation.get("outer_splitter") or evaluation.get("splitter") or {"method": "kfold"}
    )
    if "name" in outer_config and "method" not in outer_config:
        outer_config["method"] = outer_config["name"]
    inner_config = dict(evaluation.get("inner_splitter") or {})
    if not inner_config:
        inner_config = {"method": "kfold", "n_splits": 3, "shuffle": True}
    if "name" in inner_config and "method" not in inner_config:
        inner_config["method"] = inner_config["name"]

    outer_plan = _make_split_plan(bundle, outer_config, config, task, split_role="outer_cv")
    candidates = _nested_candidates(config, evaluation)
    if not candidates:
        raise ValueError("nested_cv exige model ou uma lista de candidatos")

    folds = []
    score_lists: dict[str, list[float]] = {}
    oof = _empty_oof()
    selected_candidates = []
    for descriptor in outer_plan["splits"]:
        outer_train = _subset_bundle(bundle, descriptor["train_idx"])
        outer_test = _subset_bundle(bundle, descriptor["test_idx"])
        inner_results = []
        for candidate in candidates:
            candidate_config = deepcopy(config)
            candidate_config["model"] = candidate
            try:
                inner_result = _run_development_splits(
                    candidate_config,
                    outer_train,
                    metric_specs,
                    inner_config,
                    task,
                    split_role="inner_cv",
                )
                inner_results.append(
                    {
                        "model": candidate["name"],
                        "params": candidate.get("params", {}),
                        "metrics": inner_result["metrics"],
                        "metric_std": inner_result.get("metric_std", {}),
                    }
                )
            except Exception as exc:
                inner_results.append(
                    {
                        "model": candidate["name"],
                        "params": candidate.get("params", {}),
                        "status": "error",
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
        selected = _select_nested_candidate(inner_results, config.get("selection") or {}, metric_specs)
        selected_candidates.append({"split_id": descriptor["split_id"], **selected})
        selected_config = deepcopy(config)
        selected_config["model"] = {"name": selected["model"], "params": selected.get("params", {})}
        pipeline = _fit_pipeline(selected_config, outer_train["X"], outer_train["y"])
        y_pred, y_score, y_proba, class_labels = _predict_outputs(pipeline, outer_test["X"])
        fold_metrics = compute_metrics(
            outer_test["y"],
            y_pred,
            metric_specs,
            task=task,
            y_score=y_score,
            y_proba=y_proba,
            class_labels=class_labels,
        )
        for metric, value in fold_metrics.items():
            if np.isscalar(value):
                score_lists.setdefault(metric, []).append(float(value))
        predictions = _prediction_payload(outer_test, y_pred, y_score, y_proba, class_labels)
        folds.append(
            {
                "fold": descriptor["fold"],
                "fold_in_repeat": descriptor["fold_in_repeat"],
                "repeat": descriptor["repeat"],
                "split_id": descriptor["split_id"],
                "split_role": "outer_cv",
                "n_train": len(outer_train["X"]),
                "n_test": len(outer_test["X"]),
                "metrics": _scalar_metrics(fold_metrics),
                "predictions": predictions,
                "selection": {"selected": selected, "inner_candidates": inner_results},
            }
        )
        _extend_oof(oof, predictions, descriptor, "outer_cv")

    return {
        "metrics": {metric: float(np.mean(values)) for metric, values in score_lists.items() if values},
        "metric_std": {metric: float(np.std(values, ddof=0)) for metric, values in score_lists.items() if values},
        "cv_scores": {metric: [float(value) for value in values] for metric, values in score_lists.items()},
        "folds": folds,
        "oof_predictions": oof,
        "selected_candidates": selected_candidates,
        "split_metadata": {
            "version": 1,
            "outer_cv": {
                **outer_plan["metadata"],
                "splits": [
                    _public_split_metadata(descriptor, bundle, outer_plan["metadata"])
                    for descriptor in outer_plan["splits"]
                ],
            },
        },
        "evaluation": {
            "metric_origin": "outer_cv",
            "aggregation": "mean_over_outer_folds",
            "oof_semantics": "one outer prediction per row, fold and repeat",
            "inner_selection": "per_outer_training_partition",
            "n_repeats": outer_plan["metadata"]["n_repeats"],
        },
    }


def _nested_candidates(config: dict, evaluation: dict) -> list[dict]:
    entries = evaluation.get("candidates", config.get("models"))
    if entries:
        from ml_playground.experiments.grid import build_model_grid

        return build_model_grid(entries)
    model = config.get("model")
    return [dict(model)] if model else []


def _select_nested_candidate(results: list[dict], selection: dict, metric_specs) -> dict:
    primary = selection.get("primary_metric")
    if primary is None:
        specs = normalize_metric_specs(metric_specs)
        if not specs:
            raise ValueError("nested_cv exige ao menos uma métrica")
        primary = specs[0]["id"]
    direction = selection.get("direction", _metric_direction(primary))
    candidates = [
        result
        for result in results
        if result.get("status", "success") == "success" and primary in result.get("metrics", {})
    ]
    if not candidates:
        raise ValueError(f"Nenhum candidato interno produziu a métrica {primary!r}")
    if direction == "minimize":
        ordered = sorted(
            candidates,
            key=lambda result: (
                result["metrics"][primary],
                result.get("metric_std", {}).get(primary, 0.0),
                result["model"],
                repr(sorted(result.get("params", {}).items())),
            ),
        )
    else:
        ordered = sorted(
            candidates,
            key=lambda result: (
                -result["metrics"][primary],
                result.get("metric_std", {}).get(primary, 0.0),
                result["model"],
                repr(sorted(result.get("params", {}).items())),
            ),
        )
    return {key: value for key, value in ordered[0].items() if key not in {"status", "error"}}


def _metric_direction(metric: str) -> str:
    return "minimize" if metric in {"mae", "mse", "rmse", "mape", "max_error", "davies_bouldin", "log_loss", "brier_score"} else "maximize"


def _make_split_plan(bundle: dict, split_config: dict, config: dict, task: str, *, split_role: str) -> dict:
    """Build index splits once, before any estimator is fitted."""

    split_config = dict(split_config or {})
    method = _normalise_method(split_config.get("method", split_config.get("name", "holdout")))
    method = _effective_method(method, split_config, config.get("data", {}), task)
    random_state = split_config.get("random_state", config.get("data", {}).get("random_state", 42))
    X, y = bundle["X"], bundle["y"]
    if y is None:
        raise ValueError("Split supervisionado exige target")
    if method in _HOLDOUT_METHODS:
        train_idx, test_idx = _holdout_indices(
            X,
            y,
            bundle.get("groups"),
            bundle.get("times"),
            split_config,
            config.get("data", {}),
            task,
            random_state,
            method,
        )
        descriptors = [
            _split_descriptor(train_idx, test_idx, 1, 1, 1, split_role, bundle, method)
        ]
        n_repeats = 1
    else:
        splitter = _build_splitter(
            split_config,
            config.get("data", {}),
            random_state,
            task,
            groups=bundle.get("groups"),
            times=bundle.get("times"),
            y=y,
        )
        split_y = _stratification_labels(
            y, split_config, config.get("data", {}), task, method, required=False
        )
        if _is_stratified_method(method) and split_y is None:
            raise ValueError(f"{method} exige labels de estratificação")
        groups = bundle.get("groups") if _is_group_method(method) else None
        base_splits = int(split_config.get("n_splits", 5))
        n_repeats = int(split_config.get("n_repeats", 3)) if method.startswith("repeated_") else 1
        descriptors = []
        for number, (train_idx, test_idx) in enumerate(splitter.split(X, split_y if split_y is not None else y, groups), start=1):
            repeat = (number - 1) // base_splits + 1
            fold_in_repeat = (number - 1) % base_splits + 1
            descriptors.append(
                _split_descriptor(
                    np.asarray(train_idx),
                    np.asarray(test_idx),
                    number,
                    fold_in_repeat,
                    repeat,
                    split_role,
                    bundle,
                    method,
                )
            )
        if not descriptors:
            raise ValueError("O splitter não produziu nenhuma partição")
    return {
        "splits": descriptors,
        "metadata": {
            "method": method,
            "random_state": random_state,
            "n_splits": len(descriptors),
            "n_repeats": n_repeats,
            "split_role": split_role,
        },
    }


def _holdout_indices(X, y, groups, times, split_config, data_config, task, random_state, method):
    test_size = float(split_config.get("test_size", data_config.get("test_size", 0.2)))
    if not 0 < test_size < 1:
        raise ValueError("test_size deve estar entre 0 e 1")
    indices = np.arange(len(X))
    if method == "group_holdout":
        group_values = _require_values(groups, "group_column", method)
        splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
        train_idx, test_idx = next(splitter.split(X, y, group_values))
        return np.asarray(train_idx), np.asarray(test_idx)
    if method == "temporal_holdout":
        return _temporal_holdout_indices(_require_values(times, "time_column", method), test_size)
    stratify = _stratification_labels(y, split_config, data_config, task, method, required=False)
    try:
        train_idx, test_idx = train_test_split(
            indices,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify,
        )
    except ValueError as exc:
        if stratify is not None:
            raise ValueError(f"Não foi possível estratificar o holdout: {exc}") from exc
        raise
    return np.asarray(train_idx), np.asarray(test_idx)


def _build_splitter(cv_config, data_config, random_state, task, *, groups=None, times=None, y=None):
    """Return a configured sklearn/custom splitter for development or outer CV."""

    method = _normalise_method(cv_config.get("method", cv_config.get("name", "kfold")))
    n_splits = int(cv_config.get("n_splits", 5))
    if n_splits < 2:
        raise ValueError("n_splits deve ser >= 2")
    shuffle = bool(cv_config.get("shuffle", True))
    seed = random_state if shuffle else None
    if method == "kfold" and _classification_stratified_default(cv_config, data_config, task):
        method = "stratified_kfold"

    if method == "kfold":
        return KFold(n_splits=n_splits, shuffle=shuffle, random_state=seed)
    if method == "stratified_kfold":
        return StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=seed)
    if method == "repeated_kfold":
        return RepeatedKFold(
            n_splits=n_splits,
            n_repeats=int(cv_config.get("n_repeats", 3)),
            random_state=random_state,
        )
    if method == "repeated_stratified_kfold":
        return RepeatedStratifiedKFold(
            n_splits=n_splits,
            n_repeats=int(cv_config.get("n_repeats", 3)),
            random_state=random_state,
        )
    if method == "group_kfold":
        _require_values(groups, "group_column", method)
        return GroupKFold(n_splits=n_splits)
    if method == "stratified_group_kfold":
        _require_values(groups, "group_column", method)
        return StratifiedGroupKFold(n_splits=n_splits, shuffle=shuffle, random_state=seed)
    if method in {"time_series", "backtest"}:
        time_values = _require_values(times, "time_column", method)
        return _OrderedTimeSeriesSplit(
            time_values,
            n_splits=n_splits,
            test_size=cv_config.get("test_size"),
            gap=int(cv_config.get("gap", 0)),
            max_train_size=cv_config.get("max_train_size"),
        )
    raise ValueError(f"Método de validação não suportado: {method}")


class _OrderedTimeSeriesSplit:
    """Apply ``TimeSeriesSplit`` to chronological order, not input row order."""

    def __init__(self, times, *, n_splits, test_size=None, gap=0, max_train_size=None):
        values = np.asarray(times)
        if len(values) == 0:
            raise ValueError("O split temporal exige ao menos uma data")
        if pd.isna(values).any():
            raise ValueError("time_column contém valores ausentes")
        self.times = values
        self.order = np.argsort(values, kind="stable")
        self._splitter = TimeSeriesSplit(
            n_splits=n_splits,
            test_size=test_size,
            gap=gap,
            max_train_size=max_train_size,
        )

    def split(self, X, y=None, groups=None):
        ordered_index = np.arange(len(self.order))
        for train_positions, test_positions in self._splitter.split(ordered_index):
            train_idx = self.order[train_positions]
            test_idx = self.order[test_positions]
            # TimeSeriesSplit can cut through equal timestamps.  Dropping only
            # tied training rows preserves the embargo guarantee; those rows
            # are never fitted against an equal-time test observation.
            min_test = self.times[test_idx].min()
            train_idx = train_idx[self.times[train_idx] < min_test]
            if not len(train_idx):
                raise ValueError("O corte temporal deixou o treino vazio; reduza n_splits")
            yield train_idx, test_idx

    def get_n_splits(self, X=None, y=None, groups=None):
        return self._splitter.get_n_splits(X, y, groups)


def _normalise_method(method: Any) -> str:
    if not isinstance(method, str):
        raise ValueError("O método de validação deve ser uma string")
    normalized = method.strip().lower().replace(" ", "_")
    return _METHOD_ALIASES.get(normalized, normalized)


def _effective_method(method: str, split_config: dict, data_config: dict, task: str) -> str:
    """Resolve documented defaults before recording the immutable split plan."""

    if method == "kfold" and _classification_stratified_default(split_config, data_config, task):
        return "stratified_kfold"
    return method


def _classification_stratified_default(cv_config, data_config, task):
    return bool(cv_config.get("stratified", data_config.get("stratified", task == "classification"))) and task == "classification"


def _is_group_method(method: str) -> bool:
    return method in {"group_kfold", "stratified_group_kfold"}


def _is_stratified_method(method: str) -> bool:
    return method in {"stratified_kfold", "repeated_stratified_kfold", "stratified_group_kfold"}


def _stratification_labels(y, split_config, data_config, task, method, *, required: bool):
    """Return labels only when stratification was explicitly requested."""

    requested = _is_stratified_method(method)
    declaration = split_config.get("stratify", split_config.get("stratification"))
    bins = split_config.get("stratify_bins", split_config.get("regression_stratify_bins"))
    if declaration is None:
        declaration = data_config.get("stratify", data_config.get("stratification"))
    if bins is None:
        bins = data_config.get("stratify_bins", data_config.get("regression_stratify_bins"))
    if bins is not None:
        requested = True
    if isinstance(declaration, bool):
        requested = requested or declaration
        declaration = None
    elif declaration is not None:
        requested = True

    if task == "classification":
        if method == "holdout":
            requested = bool(
                split_config.get("stratified", data_config.get("stratified", True))
            )
        if not requested:
            return None
        return np.asarray(y)
    if task != "regression" or not requested:
        return None

    if isinstance(declaration, dict):
        bins = declaration.get("n_bins", declaration.get("bins", bins))
        strategy = declaration.get("strategy", declaration.get("method", "quantile"))
        if strategy not in {"quantile", "qcut", "quantiles"}:
            raise ValueError("A estratificação de regressão aceita somente bins quantílicos")
    if bins is None:
        raise ValueError(
            "Estratificação de regressão exige bins declarados (stratify: {n_bins: ...})"
        )
    return _quantile_bins(y, int(bins))


def _quantile_bins(y, n_bins: int):
    if n_bins < 2:
        raise ValueError("stratify.n_bins deve ser >= 2")
    values = np.asarray(y, dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("A estratificação de regressão não aceita target ausente ou infinito")
    try:
        codes = pd.qcut(values, q=n_bins, labels=False, duplicates="drop")
    except ValueError as exc:
        raise ValueError(f"Não foi possível criar bins quantílicos: {exc}") from exc
    codes = np.asarray(codes)
    if len(np.unique(codes)) < 2:
        raise ValueError("A estratificação de regressão produziu menos de dois bins")
    return codes


def _temporal_holdout_indices(times, test_size: float):
    values = np.asarray(times)
    if pd.isna(values).any():
        raise ValueError("time_column contém valores ausentes")
    order = np.argsort(values, kind="stable")
    cut = len(order) - ceil(len(order) * test_size)
    if cut <= 0 or cut >= len(order):
        raise ValueError("O teste temporal precisa deixar linhas para treino e teste")
    # Never let the same timestamp live on both sides of the final boundary.
    while cut > 0 and values[order[cut - 1]] == values[order[cut]]:
        cut -= 1
    if cut == 0:
        raise ValueError("Não há corte temporal com timestamps distintos")
    return order[:cut], order[cut:]


def _require_values(values, role, method):
    if values is None:
        raise ValueError(f"{method} exige data.{role}")
    array = np.asarray(values)
    if len(array) == 0:
        raise ValueError(f"{method} exige valores em data.{role}")
    return array


def _split_descriptor(train_idx, test_idx, fold, fold_in_repeat, repeat, split_role, bundle, method):
    if len(np.intersect1d(train_idx, test_idx)):
        raise ValueError("O plano de split contém linhas em treino e avaliação")
    groups = bundle.get("groups")
    if _is_group_method(method) or method == "group_holdout":
        overlap = set(np.asarray(groups)[train_idx]).intersection(np.asarray(groups)[test_idx])
        if overlap:
            raise ValueError("O split por grupo vazou grupos entre treino e avaliação")
    times = bundle.get("times")
    if method in _TIME_METHODS:
        train_times, test_times = np.asarray(times)[train_idx], np.asarray(times)[test_idx]
        if np.max(train_times) >= np.min(test_times):
            raise ValueError("O split temporal contém treino no mesmo instante ou após o teste")
    return {
        "fold": fold,
        "fold_in_repeat": fold_in_repeat,
        "repeat": repeat,
        "split_id": f"{split_role}-r{repeat}-f{fold_in_repeat}",
        "split_role": split_role,
        "train_idx": np.asarray(train_idx, dtype=int),
        "test_idx": np.asarray(test_idx, dtype=int),
    }


def _public_split_metadata(descriptor, bundle, plan_metadata):
    train_idx, test_idx = descriptor["train_idx"], descriptor["test_idx"]
    metadata = {
        "split_id": descriptor["split_id"],
        "split_role": descriptor["split_role"],
        "fold": descriptor["fold"],
        "fold_in_repeat": descriptor["fold_in_repeat"],
        "repeat": descriptor["repeat"],
        "n_train": int(len(train_idx)),
        "n_test": int(len(test_idx)),
        "train_indices": [int(index) for index in train_idx],
        "test_indices": [int(index) for index in test_idx],
        "train_row_ids": _json_values(np.asarray(bundle["row_ids"])[train_idx]),
        "test_row_ids": _json_values(np.asarray(bundle["row_ids"])[test_idx]),
        "method": plan_metadata["method"],
        "random_state": plan_metadata["random_state"],
    }
    if bundle.get("groups") is not None:
        groups = np.asarray(bundle["groups"])
        metadata["n_train_groups"] = int(len(np.unique(groups[train_idx])))
        metadata["n_test_groups"] = int(len(np.unique(groups[test_idx])))
        metadata["group_overlap_count"] = int(
            len(set(groups[train_idx]).intersection(groups[test_idx]))
        )
    if bundle.get("times") is not None:
        times = np.asarray(bundle["times"])
        metadata["train_time_range"] = _time_range(times[train_idx])
        metadata["test_time_range"] = _time_range(times[test_idx])
    return metadata


def _time_range(values):
    return {"min": _json_value(np.min(values)), "max": _json_value(np.max(values))}


def _prediction_payload(bundle, y_pred, y_score, y_proba, class_labels):
    payload = {
        "row_ids": _json_values(bundle["row_ids"]),
        "y_true": _json_values(bundle.get("y")),
        "y_pred": _json_values(y_pred),
        "y_score": _json_values(y_score),
        "y_proba": _json_values(y_proba),
        "class_labels": _json_values(class_labels),
    }
    metadata = _json_metadata(bundle.get("metadata"))
    if metadata:
        payload["metadata"] = metadata
    return payload


def _empty_oof():
    return {
        "row_ids": [],
        "y_true": [],
        "y_pred": [],
        "y_score": [],
        "y_proba": [],
        "fold": [],
        "fold_in_repeat": [],
        "repeat": [],
        "split_id": [],
        "split_role": [],
        "metadata": {},
    }


def _extend_oof(oof, predictions, descriptor, split_role):
    n_rows = len(predictions["row_ids"])
    previous_rows = len(oof["row_ids"])
    for key in ("row_ids", "y_true", "y_pred"):
        oof[key].extend(predictions.get(key) or [])
    for key in ("y_score", "y_proba"):
        values = predictions.get(key)
        oof[key].extend([None] * n_rows if values is None else values)
    oof["fold"].extend([descriptor["fold"]] * n_rows)
    oof["fold_in_repeat"].extend([descriptor["fold_in_repeat"]] * n_rows)
    oof["repeat"].extend([descriptor["repeat"]] * n_rows)
    oof["split_id"].extend([descriptor["split_id"]] * n_rows)
    oof["split_role"].extend([split_role] * n_rows)
    incoming_metadata = predictions.get("metadata") or {}
    metadata = oof["metadata"]
    for name in set(metadata).union(incoming_metadata):
        if name not in metadata:
            metadata[name] = [None] * previous_rows
        values = incoming_metadata.get(name)
        if values is None:
            metadata[name].extend([None] * n_rows)
        else:
            metadata[name].extend(list(values))


def _run_clustering(config: dict) -> dict:
    bundle = _load_data_bundle(config, "clustering")
    X, row_ids = bundle["X"], bundle["row_ids"]
    pipeline = _fit_pipeline(config, X, None)
    estimator = pipeline.named_steps["model"]
    labels = getattr(estimator, "labels_", None)
    if labels is None:
        raise ValueError(f"O modelo {config['model']['name']} não expôs labels_")

    transformed = pipeline.named_steps["preprocessing"].transform(X)
    metrics, notes = compute_clustering_metrics(
        transformed,
        labels,
        config.get("metric_specs", config.get("metrics", [])),
        inertia=getattr(estimator, "inertia_", None),
    )
    constraints = _cluster_constraints(config, metrics)
    return {
        "metrics": metrics,
        "metric_notes": notes,
        "predictions": {
            "row_ids": _json_values(row_ids),
            "labels": _json_values(labels),
            "metadata": _json_metadata(bundle.get("metadata")),
        },
        "pipeline": pipeline,
        "n_samples": len(X),
        "cluster_selection": constraints,
        "split_metadata": {
            "version": 1,
            "internal": {"split_role": "clustering_internal", "n_samples": len(X)},
        },
        "evaluation": {
            "protocol": "internal_clustering",
            "metric_origin": "internal",
            "selection_constraints": constraints,
        },
    }


def _cluster_constraints(config, metrics):
    evaluation = config.get("evaluation") or {}
    requested = dict(
        evaluation.get("cluster_constraints", config.get("cluster_constraints", {})) or {}
    )
    violations = []
    if "max_noise_ratio" in requested and metrics.get("noise_ratio", 0.0) > requested["max_noise_ratio"]:
        violations.append("max_noise_ratio")
    if "min_cluster_size" in requested and metrics.get("cluster_size_min", 0) < requested["min_cluster_size"]:
        violations.append("min_cluster_size")
    if "min_clusters" in requested and metrics.get("cluster_count", 0) < requested["min_clusters"]:
        violations.append("min_clusters")
    if "max_clusters" in requested and metrics.get("cluster_count", 0) > requested["max_clusters"]:
        violations.append("max_clusters")
    return {"mode": "internal", "constraints": requested, "eligible": not violations, "violations": violations}


def _load_dataset(config: dict, task: str):
    """Backward-compatible tuple view of the normalized data bundle."""

    bundle = _load_data_bundle(config, task)
    return bundle["X"], bundle["y"], bundle["row_ids"]


def _load_data_bundle(config: dict, task: str) -> dict:
    """Use a preflight bundle when available, otherwise load legacy config."""

    prepared = (
        config.get("prepared_data")
        or config.get("_prepared_data")
        or config.get("data_bundle")
    )
    if prepared is None and config.get("contract_version"):
        from ml_playground.experiments.preflight import prepare_experiment_data

        prepared = prepare_experiment_data(config)
    if prepared is not None:
        return _bundle_from_prepared(prepared, config, task)
    return _bundle_from_config(config, task)


def _bundle_from_prepared(prepared: dict, config: dict, task: str) -> dict:
    if not isinstance(prepared, dict):
        raise TypeError("prepared_data deve ser um mapa")
    data_config = config["data"]
    frame = prepared.get("frame")
    feature_columns = prepared.get("feature_columns") or _feature_columns(data_config, frame, task)
    X = prepared.get("X")
    if X is None:
        X = _select_columns(frame, feature_columns)
    X = _as_pandas_frame(X, feature_columns)
    y = prepared.get("y")
    if y is None and task in {"classification", "regression"}:
        y = _column_values(frame, data_config["target"])
    row_ids = prepared.get("row_ids")
    if row_ids is None:
        row_ids = _role_values(frame, data_config.get("id_column"), len(X))
    metadata = prepared.get("metadata")
    if metadata is None:
        metadata = _metadata_values(frame, data_config.get("metadata_columns", []))
    groups = prepared.get("groups")
    if groups is None:
        groups = _role_values(frame, data_config.get("group_column"), None)
    times = prepared.get("times")
    if times is None:
        times = _role_values(frame, data_config.get("time_column"), None)

    test_frame = prepared.get("test_frame")
    test_X = prepared.get("test_X")
    if test_X is None and test_frame is not None:
        test_X = _select_columns(test_frame, feature_columns)
    test_y = prepared.get("test_y")
    if test_y is None and test_frame is not None and data_config.get("target") in _columns(test_frame):
        test_y = _column_values(test_frame, data_config["target"])
    test_row_ids = prepared.get("test_row_ids")
    if test_X is not None and test_row_ids is None:
        test_row_ids = _role_values(test_frame, data_config.get("id_column"), len(test_X))
    test_metadata = prepared.get("test_metadata")
    if test_metadata is None:
        test_metadata = _metadata_values(test_frame, data_config.get("metadata_columns", []))
    test_groups = prepared.get("test_groups")
    if test_groups is None:
        test_groups = _role_values(test_frame, data_config.get("group_column"), None)
    test_times = prepared.get("test_times")
    if test_times is None:
        test_times = _role_values(test_frame, data_config.get("time_column"), None)

    return _make_bundle(
        X,
        y,
        row_ids,
        metadata,
        groups,
        times,
        test_X=test_X,
        test_y=test_y,
        test_row_ids=test_row_ids,
        test_metadata=test_metadata,
        test_groups=test_groups,
        test_times=test_times,
        feature_columns=feature_columns,
        prepared=prepared,
        task=task,
    )


def _bundle_from_config(config: dict, task: str) -> dict:
    data_config = config["data"]
    frame = data_config.get("frame")
    if frame is None:
        frame = auto_read(data_config["path"], **dict(data_config.get("read_options") or {}))
    feature_columns = _feature_columns(data_config, frame, task)
    X = _select_columns(frame, feature_columns)
    y = _column_values(frame, data_config.get("target")) if task in {"classification", "regression"} else None
    test_config = data_config.get("test") or {}
    test_frame = test_config.get("frame")
    if test_frame is None and test_config.get("path"):
        test_frame = auto_read(test_config["path"], **dict(test_config.get("read_options") or {}))
    test_X = _select_columns(test_frame, feature_columns) if test_frame is not None else None
    test_y = None
    if test_frame is not None and task in {"classification", "regression"} and data_config.get("target") in _columns(test_frame):
        test_y = _column_values(test_frame, data_config["target"])
    return _make_bundle(
        X,
        y,
        _role_values(frame, data_config.get("id_column"), len(X)),
        _metadata_values(frame, data_config.get("metadata_columns", [])),
        _role_values(frame, data_config.get("group_column"), None),
        _role_values(frame, data_config.get("time_column"), None),
        test_X=test_X,
        test_y=test_y,
        test_row_ids=_role_values(test_frame, data_config.get("id_column"), len(test_X)) if test_X is not None else None,
        test_metadata=_metadata_values(test_frame, data_config.get("metadata_columns", [])),
        test_groups=_role_values(test_frame, data_config.get("group_column"), None),
        test_times=_role_values(test_frame, data_config.get("time_column"), None),
        feature_columns=feature_columns,
        task=task,
    )


def _make_bundle(
    X,
    y,
    row_ids,
    metadata,
    groups,
    times,
    *,
    test_X=None,
    test_y=None,
    test_row_ids=None,
    test_metadata=None,
    test_groups=None,
    test_times=None,
    feature_columns=None,
    prepared=None,
    task,
):
    X = _as_pandas_frame(X, feature_columns)
    y = None if y is None else np.asarray(y)
    if y is not None and len(X) != len(y):
        raise ValueError("X e y possuem números de linhas diferentes")
    if task in {"classification", "regression"} and y is None:
        raise ValueError("Target não encontrada no dataset")
    row_ids = np.arange(len(X)) if row_ids is None else np.asarray(row_ids)
    if len(row_ids) != len(X):
        raise ValueError("row_ids possui tamanho diferente de X")
    bundle = {
        "X": X,
        "y": y,
        "row_ids": row_ids,
        "metadata": _normalise_metadata(metadata, len(X)),
        "groups": _normalise_role_array(groups, len(X), "groups"),
        "times": _normalise_role_array(times, len(X), "times"),
        "feature_columns": list(X.columns),
    }
    if prepared:
        for key in (
            "profile",
            "test_profile",
            "data_fingerprint",
            "test_data_fingerprint",
            "schema_signature",
            "config_fingerprint",
        ):
            if key in prepared:
                bundle[key] = prepared[key]
    if test_X is not None:
        test_X = _as_pandas_frame(test_X, bundle["feature_columns"])
        if list(test_X.columns) != bundle["feature_columns"]:
            raise ValueError("As features do teste final não coincidem com as de desenvolvimento")
        test_y = None if test_y is None else np.asarray(test_y)
        if test_y is not None and len(test_y) != len(test_X):
            raise ValueError("test_X e test_y possuem números de linhas diferentes")
        test_row_ids = np.arange(len(test_X)) if test_row_ids is None else np.asarray(test_row_ids)
        if len(test_row_ids) != len(test_X):
            raise ValueError("test_row_ids possui tamanho diferente de test_X")
        bundle.update(
            {
                "test_X": test_X,
                "test_y": test_y,
                "test_row_ids": test_row_ids,
                "test_metadata": _normalise_metadata(test_metadata, len(test_X)),
                "test_groups": _normalise_role_array(test_groups, len(test_X), "test_groups"),
                "test_times": _normalise_role_array(test_times, len(test_X), "test_times"),
            }
        )
    else:
        bundle["test_X"] = None
    return bundle


def _feature_columns(data_config, frame, task):
    columns = _columns(frame)
    target = data_config.get("target")
    features = data_config.get("features")
    if isinstance(features, dict):
        features = [feature for group in features.values() for feature in group]
    if features is None:
        excluded = {
            target,
            data_config.get("id_column"),
            data_config.get("group_column"),
            data_config.get("time_column"),
            *data_config.get("metadata_columns", []),
        }
        features = [column for column in columns if column not in excluded]
    if not isinstance(features, list) or not features:
        raise ValueError("O experimento deve declarar ao menos uma feature")
    missing = sorted(set(features) - set(columns))
    if missing:
        raise ValueError(f"Features não encontradas no dataset: {missing}")
    if task in {"classification", "regression"} and target not in columns:
        raise ValueError(f"Target não encontrada no dataset: {target}")
    return features


def _columns(frame):
    if frame is None:
        return []
    return list(frame.columns)


def _select_columns(frame, columns):
    if frame is None:
        raise ValueError("Dados preparados não contêm frame ou X")
    if hasattr(frame, "select") and not isinstance(frame, pd.DataFrame):
        return frame.select(columns).to_pandas()
    return frame.loc[:, columns].copy()


def _column_values(frame, column):
    if frame is None or column is None or column not in _columns(frame):
        return None
    if hasattr(frame, "get_column"):
        return frame.get_column(column).to_numpy()
    return frame[column].to_numpy()


def _role_values(frame, column, fallback_length):
    if column and column in _columns(frame):
        return _column_values(frame, column)
    return np.arange(fallback_length) if fallback_length is not None else None


def _metadata_values(frame, columns):
    if frame is None or not columns:
        return {}
    missing = sorted(set(columns) - set(_columns(frame)))
    if missing:
        raise ValueError(f"Metadata columns não encontradas no dataset: {missing}")
    return {column: _column_values(frame, column) for column in columns}


def _as_pandas_frame(values, columns=None):
    if isinstance(values, pd.DataFrame):
        frame = values.copy()
    elif hasattr(values, "to_pandas"):
        frame = values.to_pandas()
    else:
        frame = pd.DataFrame(values, columns=columns)
    if columns is not None:
        missing = sorted(set(columns) - set(frame.columns))
        if missing:
            raise ValueError(f"Features não encontradas nos dados preparados: {missing}")
        frame = frame.loc[:, columns].copy()
    return frame.reset_index(drop=True)


def _normalise_metadata(metadata, length):
    if metadata is None:
        return {}
    if isinstance(metadata, pd.DataFrame):
        metadata = {column: metadata[column].to_numpy() for column in metadata.columns}
    elif hasattr(metadata, "to_pandas"):
        frame = metadata.to_pandas()
        metadata = {column: frame[column].to_numpy() for column in frame.columns}
    if not isinstance(metadata, dict):
        raise TypeError("metadata deve ser um mapa ou DataFrame")
    result = {}
    for name, values in metadata.items():
        array = np.asarray(values)
        if len(array) != length:
            raise ValueError(f"metadata.{name} possui tamanho diferente de X")
        result[name] = array
    return result


def _normalise_role_array(values, length, name):
    if values is None:
        return None
    array = np.asarray(values)
    if len(array) != length:
        raise ValueError(f"{name} possui tamanho diferente de X")
    if name in {"times", "test_times"} and not np.issubdtype(array.dtype, np.number):
        try:
            array = pd.to_datetime(array, errors="raise").to_numpy()
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} contém valores temporais inválidos") from exc
    return array


def _subset_bundle(bundle, indices):
    indices = np.asarray(indices, dtype=int)
    return {
        "X": bundle["X"].iloc[indices].reset_index(drop=True),
        "y": None if bundle.get("y") is None else np.asarray(bundle["y"])[indices],
        "row_ids": np.asarray(bundle["row_ids"])[indices],
        "metadata": {
            name: np.asarray(values)[indices]
            for name, values in (bundle.get("metadata") or {}).items()
        },
        "groups": None if bundle.get("groups") is None else np.asarray(bundle["groups"])[indices],
        "times": None if bundle.get("times") is None else np.asarray(bundle["times"])[indices],
        "feature_columns": bundle.get("feature_columns", list(bundle["X"].columns)),
    }


def _test_bundle(bundle):
    return {
        "X": bundle["test_X"],
        "y": bundle["test_y"],
        "row_ids": bundle["test_row_ids"],
        "metadata": bundle.get("test_metadata") or {},
        "groups": bundle.get("test_groups"),
        "times": bundle.get("test_times"),
        "feature_columns": bundle.get("feature_columns", list(bundle["test_X"].columns)),
    }


def _concat_bundles(left, right):
    if list(left["X"].columns) != list(right["X"].columns):
        raise ValueError("Não é possível refitar com schemas de features diferentes")
    metadata_names = set(left.get("metadata", {})).intersection(right.get("metadata", {}))
    return {
        "X": pd.concat([left["X"], right["X"]], ignore_index=True),
        "y": np.concatenate([left["y"], right["y"]]),
        "row_ids": np.concatenate([left["row_ids"], right["row_ids"]]),
        "metadata": {
            name: np.concatenate([left["metadata"][name], right["metadata"][name]])
            for name in metadata_names
        },
        "groups": _concat_optional(left.get("groups"), right.get("groups")),
        "times": _concat_optional(left.get("times"), right.get("times")),
        "feature_columns": list(left["X"].columns),
    }


def _concat_optional(left, right):
    if left is None or right is None:
        return None
    return np.concatenate([left, right])


def _fit_pipeline(config, X_train, y_train=None):
    model_config = config["model"]
    model = get_model(model_config["name"], model_config.get("params", {}))
    preprocessing = dict(config.get("preprocessing", {}) or {})
    # ``target`` is orchestration config, not a feature transformer branch.
    preprocessing.pop("target", None)
    pipeline = build_model_pipeline(preprocessing, X_train, model)
    target_transform = _target_transform_config(config)
    if y_train is None:
        pipeline.fit(X_train)
        return pipeline
    if target_transform is None:
        pipeline.fit(X_train, y_train)
        return pipeline
    if config.get("task", "classification") != "regression":
        raise ValueError("target_transform é suportado somente para regressão")
    transformed = _wrap_target_transform(pipeline, target_transform)
    transformed.fit(X_train, y_train)
    return transformed


def _target_transform_config(config):
    value = config.get("target_transform")
    if value is None:
        value = (config.get("evaluation") or {}).get("target_transform")
    if value is None:
        value = (config.get("preprocessing") or {}).get("target")
    if value is None or value is False or value == "none":
        return None
    if isinstance(value, str):
        return {"method": value}
    if not isinstance(value, dict):
        raise TypeError("target_transform deve ser string ou mapa")
    return dict(value)


def _wrap_target_transform(pipeline, config):
    method = str(config.get("method", config.get("name", "none"))).lower()
    if method in {"none", "identity"}:
        return pipeline
    if method == "log1p":
        return TransformedTargetRegressor(regressor=pipeline, func=np.log1p, inverse_func=np.expm1)
    if method in {"yeo_johnson", "yeo-johnson"}:
        return TransformedTargetRegressor(
            regressor=pipeline,
            transformer=PowerTransformer(method="yeo-johnson", standardize=bool(config.get("standardize", True))),
        )
    raise ValueError(f"Transformação de target não suportada: {method}")


def _predict_outputs(pipeline, X):
    y_pred = pipeline.predict(X)
    y_proba = None
    y_score = None
    if hasattr(pipeline, "predict_proba"):
        y_proba = pipeline.predict_proba(X)
        y_score = y_proba
    elif hasattr(pipeline, "decision_function"):
        y_score = pipeline.decision_function(X)
    class_labels = getattr(pipeline, "classes_", None)
    return y_pred, y_score, y_proba, class_labels


def _scalar_metrics(metrics):
    return {
        name: float(value) if np.isscalar(value) else np.asarray(value).tolist()
        for name, value in metrics.items()
    }


def _json_metadata(metadata):
    return {name: _json_values(values) for name, values in (metadata or {}).items()}


def _json_values(values):
    if values is None:
        return None
    return [_json_value(value) for value in np.asarray(values).tolist()]


def _json_value(value):
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, (np.datetime64, pd.Timestamp)):
        return str(value)
    if isinstance(value, np.ndarray):
        return [_json_value(item) for item in value.tolist()]
    if isinstance(value, list):
        return [_json_value(item) for item in value]
    if isinstance(value, tuple):
        return [_json_value(item) for item in value]
    if not isinstance(value, (list, dict, tuple, np.ndarray)):
        try:
            if bool(pd.isna(value)):
                return None
        except (TypeError, ValueError):
            pass
    return value


def _get_y_score(model, X):
    """Backward-compatible score helper used by existing callers."""

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X)
        return probabilities[:, 1] if probabilities.ndim == 2 and probabilities.shape[1] == 2 else probabilities
    if hasattr(model, "decision_function"):
        return model.decision_function(X)
    return None
