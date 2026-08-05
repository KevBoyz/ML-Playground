"""Loading and validation of self-contained experiment directories."""

import re
from pathlib import Path
from typing import Any

import yaml


EXPERIMENT_FILES = (
    "experiment.yaml",
    "models.yaml",
    "preprocessing.yaml",
    "metrics.yaml",
    "cross_validation.yaml",
)
SUPPORTED_TASKS = {"classification"}
SUPPORTED_VALIDATION = {
    "holdout",
    "kfold",
    "stratified_kfold",
    "repeated_kfold",
    "repeated_stratified_kfold",
}
SUPPORTED_MODELS = {
    "knn",
    "logistic_regression",
    "logistic",
    "svm",
    "decision_tree",
    "random_forest",
    "xgboost",
    "lightgbm",
}
SUPPORTED_METRICS = {
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
    "kappa",
    "mcc",
    "confusion_matrix",
}
_SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")


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

    project = (
        Path(project_root).resolve()
        if project_root is not None
        else _infer_project_root(folder)
    )
    documents = {
        filename: _read_yaml(folder / filename)
        for filename in EXPERIMENT_FILES
    }

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

    data = _validate_data(experiment_doc.get("data"), project, folder)
    models = _validate_models(documents["models.yaml"])
    preprocessing = _validate_preprocessing(documents["preprocessing.yaml"])
    metrics, metric_primary = _validate_metrics(documents["metrics.yaml"], task)
    validation = _validate_validation(documents["cross_validation.yaml"])

    selection = dict(experiment_doc.get("selection") or {})
    selection.setdefault("primary_metric", metric_primary or metrics[0])
    selection.setdefault("direction", "maximize")
    if selection["primary_metric"] not in metrics:
        raise ValueError(
            f"Métrica principal não está habilitada: {selection['primary_metric']}"
        )
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

    return {
        "experiment_name": experiment_name,
        "task": task,
        "data": data,
        "models": models,
        "preprocessing": preprocessing,
        "metrics": metrics,
        "cross_validation": validation,
        "selection": selection,
        "outputs": outputs,
        "experiment_dir": str(folder),
        "project_root": str(project),
        "config_files": {
            key.removesuffix(".yaml"): str(folder / key)
            for key in EXPERIMENT_FILES
        },
    }


def _infer_project_root(folder: Path) -> Path:
    if folder.parent.name == "experiments":
        return folder.parent.parent
    return Path.cwd().resolve()


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Arquivo obrigatório não encontrado: {path}")
    with path.open(encoding="utf-8") as stream:
        document = yaml.safe_load(stream)
    if not isinstance(document, dict):
        raise ValueError(f"O YAML deve conter um mapa no topo: {path}")
    return document


def _validate_name(value: Any, field: str) -> None:
    if not isinstance(value, str) or not _SAFE_NAME.fullmatch(value):
        raise ValueError(
            f"{field} deve conter apenas letras, números, '_' ou '-': {value!r}"
        )


def _validate_data(
    data: Any,
    project_root: Path,
    experiment_dir: Path,
) -> dict[str, Any]:
    if not isinstance(data, dict):
        raise ValueError(f"data ausente ou inválido em {experiment_dir / 'experiment.yaml'}")
    path = data.get("path")
    target = data.get("target")
    if not isinstance(path, str) or not path:
        raise ValueError("data.path é obrigatório")
    if not isinstance(target, str) or not target:
        raise ValueError("data.target é obrigatório")
    data = dict(data)
    source = Path(path)
    resolved_source = (project_root / source).resolve() if not source.is_absolute() else source
    if not resolved_source.exists():
        raise FileNotFoundError(f"Dataset não encontrado: {resolved_source}")
    data["path"] = str(resolved_source)
    data.setdefault("random_state", 42)
    return data


def _validate_models(document: dict[str, Any]) -> list[dict[str, Any]]:
    models = document.get("models")
    if not isinstance(models, list) or not models:
        raise ValueError("models.yaml deve conter uma lista não vazia em 'models'")
    validated = []
    for index, entry in enumerate(models):
        if not isinstance(entry, dict) or not isinstance(entry.get("name"), str):
            raise ValueError(f"models[{index}] deve conter name")
        if entry["name"] not in SUPPORTED_MODELS:
            raise ValueError(f"Modelo não suportado em models[{index}]: {entry['name']}")
        params = entry.get("params", {})
        if not isinstance(params, dict):
            raise ValueError(f"models[{index}].params deve ser um mapa")
        validated.append({"name": entry["name"], "params": params})
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
    return document


def _validate_metrics(
    document: dict[str, Any], task: str
) -> tuple[list[str], str | None]:
    section = document.get(task, document)
    if isinstance(section, dict):
        names = section.get("names", section.get("metrics", []))
        primary = section.get("primary")
    else:
        names, primary = section, None
    if not isinstance(names, list) or not names or not all(isinstance(x, str) for x in names):
        raise ValueError(f"metrics.yaml deve conter nomes para a tarefa {task}")
    unsupported = sorted(set(names) - SUPPORTED_METRICS)
    if unsupported:
        raise ValueError(f"Métricas não suportadas: {unsupported}")
    return names, primary


def _validate_validation(document: dict[str, Any]) -> dict[str, Any]:
    method = document.get("method", "holdout")
    if method not in SUPPORTED_VALIDATION:
        raise ValueError(
            f"Método de validação inválido: {method!r}. "
            f"Disponíveis: {sorted(SUPPORTED_VALIDATION)}"
        )
    result = dict(document)
    result["method"] = method
    if method == "holdout":
        test_size = float(result.get("test_size", 0.2))
        if not 0 < test_size < 1:
            raise ValueError("cross_validation.test_size deve estar entre 0 e 1")
    elif int(result.get("n_splits", 5)) < 2:
        raise ValueError("cross_validation.n_splits deve ser >= 2")
    return result
