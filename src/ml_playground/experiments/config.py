"""Loading and validation of self-contained experiment directories."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Mapping

import yaml

from ml_playground.models import MODEL_METADATA, model_supports_task


REQUIRED_EXPERIMENT_FILES = (
    "experiment.yaml",
    "models.yaml",
    "preprocessing.yaml",
    "metrics.yaml",
)
OPTIONAL_EXPERIMENT_FILES = ("cross_validation.yaml", "views.yaml")
EXPERIMENT_FILES = REQUIRED_EXPERIMENT_FILES + OPTIONAL_EXPERIMENT_FILES
SUPPORTED_TASKS = {"classification", "regression", "clustering"}
SUPERVISED_TASKS = {"classification", "regression"}
DATA_CONTRACT_VERSION = 1
SUPPORTED_SCHEMA_MODES = {"permissive", "strict"}
PROVENANCE_FIELDS = ("recipe_ref", "recipe_revision", "source_description")
SUPPORTED_VALIDATION = {
    "holdout",
    "kfold",
    "stratified_kfold",
    "repeated_kfold",
    "repeated_stratified_kfold",
    "group_kfold",
    "stratified_group_kfold",
    "group_holdout",
    "time_series",
    "time_series_split",
    "backtest",
    "temporal_holdout",
}
_VALIDATION_ALIASES = {
    "stratifiedgroupkfold": "stratified_group_kfold",
    "stratified-group-kfold": "stratified_group_kfold",
    "groupkfold": "group_kfold",
    "group-shuffle-split": "group_holdout",
    "group_shuffle_split": "group_holdout",
    "timeseriessplit": "time_series",
    "time-series": "time_series",
    "timeseries": "time_series",
    "temporal": "temporal_holdout",
}
SUPPORTED_EVALUATION_PROTOCOLS = {
    "development",
    "development_cv",
    "train_validation_test",
    "development_cv_final_test",
    "cv_final_test",
    "nested_cv",
    "nested",
}
SUPPORTED_METRICS_BY_TASK = {
    "classification": {
        "accuracy",
        "precision",
        "precision_macro",
        "precision_weighted",
        "recall",
        "recall_macro",
        "recall_weighted",
        "f1",
        "f1_macro",
        "f1_weighted",
        "roc_auc",
        "log_loss",
        "balanced_accuracy",
        "average_precision",
        "pr_auc",
        "brier_score",
        "kappa",
        "mcc",
        "confusion_matrix",
    },
    "regression": {"mae", "mse", "rmse", "r2", "mape", "max_error"},
    "clustering": {
        "silhouette",
        "calinski_harabasz",
        "davies_bouldin",
        "inertia",
        "cluster_count",
        "noise_ratio",
        "cluster_size_min",
        "cluster_size_max",
    },
}
SUPPORTED_METRICS = set().union(*SUPPORTED_METRICS_BY_TASK.values())
SUPPORTED_MODELS = set(MODEL_METADATA)
SUPPORTED_VIEWS = {
    "model_comparison",
    "confusion_matrix",
    "roc_curve",
    "precision_recall_curve",
    "class_distribution",
    "decision_boundary",
    "probability_curve",
    "learning_curve",
    "validation_curve",
    "knn_neighbors_curve",
    "tree_structure",
    "feature_importance",
    "predicted_vs_actual",
    "fit_vs_feature",
    "residuals_vs_fitted",
    "residual_distribution",
    "qq_residuals",
    "scale_location",
    "residuals_vs_leverage",
    "coefficient_importance",
    "prediction_projection",
    "elbow_curve",
    "silhouette_curve",
    "k_distance",
    "cluster_scatter",
    "cluster_size",
    "cluster_profile_heatmap",
    "noise_outliers",
    "dendrogram",
    "pca_cluster_projection",
    "pca_explained_variance",
    "cluster_contingency",
    "cluster_model_comparison",
    "feature_distributions",
    "correlation_heatmap",
    "pairplot",
    "target_relationships",
    "missingness_summary",
}
SUPPORTED_VIEW_SCOPES = {"best", "candidates", "selected_models", "folds"}
_SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")


def compute_config_fingerprint(config: Mapping[str, Any]) -> str:
    """Hash the semantic experiment contract, excluding runtime-only payloads."""

    ignored_fields = {
        "_prepared_data",
        "config_fingerprint",
        "preflight_metadata",
        "run_fingerprint",
        "config_files",
        "experiment_dir",
        "project_root",
    }

    def canonicalize(value: Any) -> Any:
        if isinstance(value, Mapping):
            return {
                str(key): canonicalize(item)
                for key, item in value.items()
                if str(key) not in ignored_fields and not str(key).startswith("_")
            }
        if isinstance(value, (list, tuple)):
            return [canonicalize(item) for item in value]
        if isinstance(value, set):
            return sorted(canonicalize(item) for item in value)
        if isinstance(value, Path):
            return str(value)
        return value

    payload = json.dumps(
        canonicalize(config),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def discover_experiments(root: str | Path = "experiments") -> list[Path]:
    """Return named experiment folders that contain the primary YAML file."""

    root_path = Path(root).resolve()
    if not root_path.exists():
        raise FileNotFoundError(f"Pasta de experimentos não encontrada: {root_path}")
    if not root_path.is_dir():
        raise NotADirectoryError(f"O caminho de experimentos não é uma pasta: {root_path}")

    return sorted(
        folder
        for folder in root_path.iterdir()
        if folder.is_dir() and (folder / "experiment.yaml").exists()
    )


def load_experiment(
    experiment_dir: str | Path,
    *,
    project_root: str | Path | None = None,
) -> dict[str, Any]:
    """Load, compose and validate all YAMLs belonging to one experiment."""

    folder = Path(experiment_dir).resolve()
    if not folder.exists() or not folder.is_dir():
        raise FileNotFoundError(f"Experimento não encontrado: {folder}")

    project = Path(project_root).resolve() if project_root is not None else _infer_project_root(folder)
    documents = {filename: _read_yaml(folder / filename) for filename in REQUIRED_EXPERIMENT_FILES}
    documents.update(
        {
            filename: _read_optional_yaml(folder / filename)
            for filename in OPTIONAL_EXPERIMENT_FILES
        }
    )

    experiment_doc = documents["experiment.yaml"]
    experiment_name = experiment_doc.get("name", folder.name)
    _validate_name(experiment_name, "name")
    if experiment_name != folder.name:
        raise ValueError(
            "O campo 'name' deve ser igual ao nome da pasta: "
            f"'{experiment_name}' != '{folder.name}'"
        )

    task = experiment_doc.get("task")
    if task not in SUPPORTED_TASKS:
        raise ValueError(
            f"Tarefa inválida em {folder / 'experiment.yaml'}: {task!r}. "
            f"Disponíveis: {sorted(SUPPORTED_TASKS)}"
        )

    data = _validate_data(experiment_doc.get("data"), project, folder, task)
    models = _validate_models(documents["models.yaml"], task)
    preprocessing = _validate_preprocessing(documents["preprocessing.yaml"])
    metrics, metric_primary = _validate_metrics(documents["metrics.yaml"], task)
    validation = _validate_validation(documents["cross_validation.yaml"], task, data)
    views = _validate_views(documents["views.yaml"], task)

    selection = dict(experiment_doc.get("selection") or {})
    metric_ids = [_metric_id(metric) for metric in metrics]
    selection.setdefault("primary_metric", metric_primary or metric_ids[0])
    if selection["primary_metric"] not in metric_ids:
        raise ValueError(f"Métrica principal não está habilitada: {selection['primary_metric']}")
    selected_metric = next(
        metric for metric in metrics if _metric_id(metric) == selection["primary_metric"]
    )
    selection.setdefault("direction", _default_metric_direction(_metric_name(selected_metric)))
    if selection["direction"] not in {"maximize", "minimize"}:
        raise ValueError("selection.direction deve ser 'maximize' ou 'minimize'")

    outputs = dict(experiment_doc.get("outputs") or {})
    output_root = Path(outputs.get("root", "reports"))
    if not output_root.is_absolute():
        output_root = project / output_root
    outputs["root"] = str(output_root.resolve())
    outputs.setdefault("save_model", True)
    outputs.setdefault("save_predictions", True)
    outputs.setdefault("figures", True)

    contract_version = experiment_doc.get("contract_version", DATA_CONTRACT_VERSION)
    if not isinstance(contract_version, int) or isinstance(contract_version, bool) or contract_version < 1:
        raise ValueError("contract_version deve ser um inteiro positivo")

    config = {
        "experiment_name": experiment_name,
        "task": task,
        "data": data,
        "models": models,
        "preprocessing": preprocessing,
        "metrics": metrics,
        "cross_validation": validation,
        "views": views,
        "selection": selection,
        "outputs": outputs,
        "experiment_dir": str(folder),
        "project_root": str(project),
        "config_files": {
            filename.removesuffix(".yaml"): str(folder / filename)
            for filename, document in documents.items()
            if document is not None
        },
        "contract_version": contract_version,
        "provenance": _validate_provenance(experiment_doc.get("provenance")),
    }
    evaluation = _validate_evaluation(experiment_doc.get("evaluation"), task, data)
    if evaluation:
        config["evaluation"] = evaluation
    execution = _validate_execution(experiment_doc.get("execution"))
    if execution:
        config["execution"] = execution
    search = _validate_search(experiment_doc.get("search"))
    if search:
        config["search"] = search

    config["config_fingerprint"] = compute_config_fingerprint(config)
    return config


def _infer_project_root(folder: Path) -> Path:
    if folder.parent.name == "experiments":
        return folder.parent.parent
    return Path.cwd().resolve()


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Arquivo obrigatório não encontrado: {path}")
    return _parse_yaml(path)


def _read_optional_yaml(path: Path) -> dict[str, Any] | None:
    return _parse_yaml(path) if path.exists() else None


def _parse_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        document = yaml.safe_load(stream)
    if not isinstance(document, dict):
        raise ValueError(f"O YAML deve conter um mapa no topo: {path}")
    return document


def _validate_name(value: Any, field: str) -> None:
    if not isinstance(value, str) or not _SAFE_NAME.fullmatch(value):
        raise ValueError(f"{field} deve conter apenas letras, números, '_' ou '-': {value!r}")


def _validate_data(
    data: Any,
    project_root: Path,
    experiment_dir: Path,
    task: str,
) -> dict[str, Any]:
    if not isinstance(data, dict):
        raise ValueError(f"data ausente ou inválido em {experiment_dir / 'experiment.yaml'}")
    path = data.get("path")
    target = data.get("target")
    if not isinstance(path, str) or not path:
        raise ValueError("data.path é obrigatório")
    if task in SUPERVISED_TASKS and (not isinstance(target, str) or not target):
        raise ValueError(f"data.target é obrigatório para tarefa {task}")
    if task == "clustering" and target is not None:
        raise ValueError("data.target não é aceito para clusterização")

    features, feature_groups = _validate_features(data.get("features"))
    if task == "clustering" and features is None:
        raise ValueError("data.features é obrigatório para clusterização")

    result = dict(data)
    source = Path(path)
    resolved_source = (project_root / source).resolve() if not source.is_absolute() else source
    if not resolved_source.exists():
        raise FileNotFoundError(f"Dataset não encontrado: {resolved_source}")
    result["path"] = str(resolved_source)
    result["read_options"] = _validate_read_options(data.get("read_options"), "data.read_options")
    if features is not None:
        result["features"] = features
    if feature_groups:
        result["feature_groups"] = feature_groups

    result["id_column"] = _validate_optional_column_name(data.get("id_column"), "data.id_column")
    result["metadata_columns"] = _validate_column_list(
        data.get("metadata_columns", []), "data.metadata_columns", allow_empty=True
    )
    result["group_column"] = _validate_optional_column_name(
        data.get("group_column"), "data.group_column"
    )
    result["time_column"] = _validate_optional_column_name(
        data.get("time_column"), "data.time_column"
    )
    result["schema"] = _validate_schema(data.get("schema"))

    protected_columns = {
        column
        for column in (
            target,
            result["id_column"],
            result["group_column"],
            result["time_column"],
            *result["metadata_columns"],
        )
        if column is not None
    }
    if features is not None:
        role_features = sorted(set(features) & protected_columns)
        if role_features:
            raise ValueError(
                "data.features não pode incluir target, ID, metadados, grupo ou tempo: "
                f"{role_features}"
            )

    test = data.get("test")
    if test is not None:
        if not isinstance(test, dict):
            raise ValueError("data.test deve ser um mapa com path e read_options opcionais")
        test_path = test.get("path")
        if not isinstance(test_path, str) or not test_path:
            raise ValueError("data.test.path é obrigatório quando data.test é informado")
        test_source = Path(test_path)
        resolved_test = (
            (project_root / test_source).resolve()
            if not test_source.is_absolute()
            else test_source.resolve()
        )
        if not resolved_test.exists():
            raise FileNotFoundError(f"Dataset de teste não encontrado: {resolved_test}")
        result["test"] = {
            **test,
            "path": str(resolved_test),
            "read_options": _validate_read_options(
                test.get("read_options"), "data.test.read_options"
            ),
        }

    result.setdefault("random_state", 42)
    return result


def _validate_features(value: Any) -> tuple[list[str] | None, dict[str, list[str]]]:
    """Normalize either the legacy flat feature list or explicit feature groups."""

    if value is None:
        return None, {}
    if isinstance(value, list):
        return _validate_column_list(value, "data.features"), {}
    if not isinstance(value, dict) or not value:
        raise ValueError("data.features deve ser uma lista ou mapa não vazio de grupos")

    groups: dict[str, list[str]] = {}
    flattened: list[str] = []
    for group, columns in value.items():
        if not isinstance(group, str) or not group:
            raise ValueError("Os nomes de data.features devem ser strings não vazias")
        groups[group] = _validate_column_list(columns, f"data.features.{group}")
        flattened.extend(groups[group])
    if len(set(flattened)) != len(flattened):
        raise ValueError("Uma feature não pode pertencer a mais de um grupo")
    return flattened, groups


def _validate_optional_column_name(value: Any, field: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} deve ser uma string não vazia")
    return value


def _validate_column_list(value: Any, field: str, *, allow_empty: bool = False) -> list[str]:
    if (
        not isinstance(value, list)
        or (not value and not allow_empty)
        or not all(isinstance(column, str) and column for column in value)
        or len(set(value)) != len(value)
    ):
        qualifier = "uma lista" if allow_empty else "uma lista não vazia"
        raise ValueError(f"{field} deve ser {qualifier} de nomes únicos")
    return list(value)


def _validate_read_options(value: Any, field: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ValueError(f"{field} deve ser um mapa de opções do leitor")
    return dict(value)


def _validate_schema(value: Any) -> dict[str, Any]:
    if value is None:
        return {"mode": "permissive", "dtypes": {}}
    if isinstance(value, str):
        value = {"mode": value}
    if not isinstance(value, dict):
        raise ValueError("data.schema deve ser um mapa")

    mode = value.get("mode", "permissive")
    if mode not in SUPPORTED_SCHEMA_MODES:
        raise ValueError(
            f"data.schema.mode inválido: {mode!r}. Disponíveis: {sorted(SUPPORTED_SCHEMA_MODES)}"
        )
    declared_dtypes = value.get("dtypes", value.get("columns", {}))
    if declared_dtypes is None:
        declared_dtypes = {}
    if not isinstance(declared_dtypes, dict):
        raise ValueError("data.schema.dtypes deve ser um mapa de coluna para dtype")
    if not all(
        isinstance(column, str)
        and column
        and isinstance(dtype, str)
        and dtype
        for column, dtype in declared_dtypes.items()
    ):
        raise ValueError("data.schema.dtypes deve conter nomes de coluna e dtypes não vazios")

    return {"mode": mode, "dtypes": dict(declared_dtypes)}


def _validate_provenance(value: Any) -> dict[str, str]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError("provenance deve ser um mapa")
    result = {}
    for field in PROVENANCE_FIELDS:
        field_value = value.get(field)
        if field_value is None:
            continue
        if not isinstance(field_value, str) or not field_value.strip():
            raise ValueError(f"provenance.{field} deve ser uma string não vazia")
        result[field] = field_value
    return result


def _validate_models(document: dict[str, Any], task: str) -> list[dict[str, Any]]:
    models = document.get("models")
    if not isinstance(models, list) or not models:
        raise ValueError("models.yaml deve conter uma lista não vazia em 'models'")
    validated = []
    for index, entry in enumerate(models):
        if not isinstance(entry, dict) or not isinstance(entry.get("name"), str):
            raise ValueError(f"models[{index}] deve conter name")
        name = entry["name"]
        if name not in SUPPORTED_MODELS:
            raise ValueError(f"Modelo não suportado em models[{index}]: {name}")
        if not model_supports_task(name, task):
            raise ValueError(f"Modelo {name!r} não é compatível com a tarefa {task}")
        params = entry.get("params", {})
        if not isinstance(params, dict):
            raise ValueError(f"models[{index}].params deve ser um mapa")
        validated.append({"name": name, "params": params})
    return validated


def _validate_preprocessing(document: dict[str, Any]) -> dict[str, Any]:
    if not document:
        return {}
    for branch in ("numeric", "categorical"):
        value = document.get(branch)
        if value is not None and not isinstance(value, dict):
            raise ValueError(f"preprocessing.{branch} deve ser um mapa")
        if value and not isinstance(value.get("steps", []), list):
            raise ValueError(f"preprocessing.{branch}.steps deve ser uma lista")
    groups = document.get("groups")
    if groups is not None:
        if not isinstance(groups, dict) or not groups:
            raise ValueError("preprocessing.groups deve ser um mapa não vazio")
        for name, group in groups.items():
            if not isinstance(name, str) or not name or not isinstance(group, dict):
                raise ValueError("Cada grupo de preprocessing deve ter nome e mapa válidos")
            _validate_column_list(group.get("columns"), f"preprocessing.groups.{name}.columns")
            if not isinstance(group.get("steps", []), list):
                raise ValueError(f"preprocessing.groups.{name}.steps deve ser uma lista")
    target = document.get("target")
    if target is not None and not isinstance(target, (str, dict)):
        raise ValueError("preprocessing.target deve ser uma string ou mapa")
    return document


def _validate_metrics(
    document: dict[str, Any], task: str
) -> tuple[list[str | dict[str, Any]], str | None]:
    section = document.get(task, document)
    if isinstance(section, dict):
        names = section.get("names", section.get("metrics", []))
        primary = section.get("primary")
    else:
        names, primary = section, None
    if not isinstance(names, list) or not names:
        raise ValueError(f"metrics.yaml deve conter nomes para a tarefa {task}")

    normalized = [_normalize_metric(entry, index) for index, entry in enumerate(names)]
    metric_names = [metric["name"] for metric in normalized]
    metric_ids = [metric["id"] for metric in normalized]
    if len(set(metric_ids)) != len(metric_ids):
        raise ValueError("Os IDs de métricas devem ser únicos")
    unsupported = sorted(set(metric_names) - SUPPORTED_METRICS_BY_TASK[task])
    if unsupported:
        raise ValueError(f"Métricas não suportadas para {task}: {unsupported}")
    if primary is not None and primary not in metric_ids:
        raise ValueError("metrics.primary deve estar presente em metrics.names")

    if all(isinstance(entry, str) for entry in names):
        return list(names), primary
    return normalized, primary


def _normalize_metric(entry: Any, index: int) -> dict[str, Any]:
    if isinstance(entry, str) and entry:
        return {"name": entry, "id": entry, "params": {}}
    if not isinstance(entry, dict):
        raise ValueError(f"metrics[{index}] deve ser um nome ou mapa")

    name = entry.get("name", entry.get("id"))
    metric_id = entry.get("id", name)
    if not isinstance(name, str) or not name or not isinstance(metric_id, str) or not metric_id:
        raise ValueError(f"metrics[{index}] deve conter name/id não vazio")
    explicit_params = entry.get("params", {})
    if explicit_params is None:
        explicit_params = {}
    if not isinstance(explicit_params, dict):
        raise ValueError(f"metrics[{index}].params deve ser um mapa")
    flat_params = {
        key: value
        for key, value in entry.items()
        if key not in {"name", "id", "params"}
    }
    return {"name": name, "id": metric_id, "params": {**flat_params, **explicit_params}}


def _metric_id(metric: str | Mapping[str, Any]) -> str:
    return metric if isinstance(metric, str) else str(metric["id"])


def _metric_name(metric: str | Mapping[str, Any]) -> str:
    return metric if isinstance(metric, str) else str(metric["name"])


def _validate_validation(
    document: dict[str, Any] | None,
    task: str,
    data: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if task == "clustering":
        if document and document.get("method", "none") != "none":
            raise ValueError("cross_validation não é suportado para clusterização")
        return {"method": "none"}

    result = dict(document or {})
    method = result.get("method", result.get("name", "holdout"))
    if not isinstance(method, str):
        raise ValueError("cross_validation.method deve ser uma string")
    method = _VALIDATION_ALIASES.get(method.strip().lower().replace(" ", "_"), method)
    if method not in SUPPORTED_VALIDATION:
        raise ValueError(
            f"Método de validação inválido: {method!r}. Disponíveis: {sorted(SUPPORTED_VALIDATION)}"
        )
    if task == "regression" and method in {
        "stratified_kfold",
        "repeated_stratified_kfold",
        "stratified_group_kfold",
    } and not _has_regression_stratification(result):
        raise ValueError(
            f"{method} em regressão exige bins declarados em cross_validation.stratify"
        )
    result["method"] = method
    if method == "holdout":
        test_size = float(result.get("test_size", 0.2))
        if not 0 < test_size < 1:
            raise ValueError("cross_validation.test_size deve estar entre 0 e 1")
    elif int(result.get("n_splits", 5)) < 2:
        raise ValueError("cross_validation.n_splits deve ser >= 2")
    data = data or {}
    if method in {"group_kfold", "stratified_group_kfold", "group_holdout"} and not data.get("group_column"):
        raise ValueError(f"{method} exige data.group_column")
    if method in {"time_series", "time_series_split", "backtest", "temporal_holdout"} and not data.get("time_column"):
        raise ValueError(f"{method} exige data.time_column")
    return result


def _has_regression_stratification(config: Mapping[str, Any]) -> bool:
    declaration = config.get("stratify", config.get("stratification"))
    bins = config.get("stratify_bins", config.get("regression_stratify_bins"))
    if isinstance(declaration, Mapping):
        bins = declaration.get("n_bins", declaration.get("bins", bins))
    return bins is not None


def _validate_evaluation(value: Any, task: str, data: Mapping[str, Any]) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError("evaluation deve ser um mapa")
    result = dict(value)
    if task == "clustering":
        protocol = result.get("protocol")
        if protocol not in {None, "internal", "internal_clustering"}:
            raise ValueError("Clusterização usa avaliação interna; não configure protocolo supervisionado")
        return result
    protocol = result.get("protocol", "development")
    if protocol not in SUPPORTED_EVALUATION_PROTOCOLS:
        raise ValueError(
            f"evaluation.protocol inválido: {protocol!r}. "
            f"Disponíveis: {sorted(SUPPORTED_EVALUATION_PROTOCOLS)}"
        )
    result["protocol"] = protocol
    splitter = result.get("splitter", result.get("development_splitter"))
    if splitter is not None:
        if not isinstance(splitter, dict):
            raise ValueError("evaluation.splitter deve ser um mapa")
        validated_splitter = _validate_validation(splitter, task, data)
        result["splitter"] = validated_splitter
        result.pop("development_splitter", None)
    final_test = result.get("final_test")
    if final_test is not None:
        if not isinstance(final_test, dict):
            raise ValueError("evaluation.final_test deve ser um mapa")
        source = final_test.get("source", "none")
        if source not in {"none", "path", "external", "data.test", "test", "split", "holdout", "reserved_split"}:
            raise ValueError("evaluation.final_test.source deve ser path, split ou none")
        if source in {"path", "external", "data.test", "test"} and not data.get("test"):
            raise ValueError("evaluation.final_test.source=path exige data.test")
        result["final_test"] = dict(final_test)
    if "evaluate_final_test" in result and not isinstance(result["evaluate_final_test"], bool):
        raise ValueError("evaluation.evaluate_final_test deve ser booleano")
    return result


def _validate_execution(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError("execution deve ser um mapa")
    result = dict(value)
    if "n_jobs" in result and (not isinstance(result["n_jobs"], int) or result["n_jobs"] < 1):
        raise ValueError("execution.n_jobs deve ser um inteiro positivo")
    if "max_candidates" in result and (not isinstance(result["max_candidates"], int) or result["max_candidates"] < 1):
        raise ValueError("execution.max_candidates deve ser um inteiro positivo")
    if "max_wall_time_seconds" in result and float(result["max_wall_time_seconds"]) <= 0:
        raise ValueError("execution.max_wall_time_seconds deve ser positivo")
    if result.get("on_candidate_error", "continue") not in {"continue", "fail_fast"}:
        raise ValueError("execution.on_candidate_error deve ser continue ou fail_fast")
    return result


def _validate_search(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError("search deve ser um mapa")
    result = dict(value)
    if result.get("strategy", "grid") not in {"grid", "random"}:
        raise ValueError("search.strategy deve ser grid ou random")
    if "max_candidates" in result and (not isinstance(result["max_candidates"], int) or result["max_candidates"] < 1):
        raise ValueError("search.max_candidates deve ser um inteiro positivo")
    return result


def _validate_views(document: dict[str, Any] | None, task: str) -> dict[str, list[dict[str, Any]]]:
    document = document or {}
    source = document.get("views", document)
    if not isinstance(source, dict):
        raise ValueError("views.yaml deve conter um mapa 'views'")
    result: dict[str, list[dict[str, Any]]] = {"common": [], task: []}
    for section in ("common", task):
        entries = source.get(section, [])
        if not isinstance(entries, list):
            raise ValueError(f"views.{section} deve ser uma lista")
        for index, entry in enumerate(entries):
            if not isinstance(entry, dict) or not isinstance(entry.get("name"), str):
                raise ValueError(f"views.{section}[{index}] deve conter name")
            if entry["name"] not in SUPPORTED_VIEWS:
                raise ValueError(f"View não suportada: {entry['name']}")
            scope = entry.get("scope", "best")
            if scope not in SUPPORTED_VIEW_SCOPES:
                raise ValueError(f"Escopo de view inválido: {scope}")
            enabled = entry.get("enabled", True)
            if not isinstance(enabled, bool):
                raise ValueError(f"views.{section}[{index}].enabled deve ser booleano")
            params = entry.get("params", {})
            if not isinstance(params, dict):
                raise ValueError(f"views.{section}[{index}].params deve ser um mapa")
            result[section].append(
                {"name": entry["name"], "enabled": enabled, "scope": scope, "params": params}
            )
    return result


def _default_metric_direction(metric: str) -> str:
    return "minimize" if metric in {"mae", "mse", "rmse", "mape", "max_error", "davies_bouldin"} else "maximize"
