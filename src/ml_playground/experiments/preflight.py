"""Pre-fit data contract orchestration for tabular experiments."""

from __future__ import annotations

import hashlib
import json
import platform
from importlib.metadata import PackageNotFoundError, version
from typing import Any, Mapping

import numpy as np
import polars as pl

from ml_playground.data.loader import auto_read
from ml_playground.data.validation import fingerprint_file, validate_data
from ml_playground.experiments.config import DATA_CONTRACT_VERSION, compute_config_fingerprint


SUPERVISED_TASKS = {"classification", "regression"}


def prepare_experiment_data(config: Mapping[str, Any]) -> dict[str, Any]:
    """Read declared sources once, enforce their contract, and return reusable views.

    The returned payload deliberately keeps in-memory frames and training arrays separate
    from ``preflight_metadata``. The former is safe to attach under a private runtime key;
    the latter is JSON-serializable evidence suitable for a manifest or report.
    """

    task = config.get("task", "classification")
    if task not in {"classification", "regression", "clustering"}:
        raise ValueError(f"Tarefa não suportada no preflight: {task!r}")
    data_config = config.get("data")
    if not isinstance(data_config, Mapping):
        raise ValueError("data é obrigatório para executar o preflight")

    target = data_config.get("target")
    if task in SUPERVISED_TASKS and (not isinstance(target, str) or not target):
        raise ValueError(f"data.target é obrigatório para tarefa {task}")
    if task == "clustering" and target is not None:
        raise ValueError("data.target não é aceito para clusterização")

    roles = _column_roles(data_config, target)
    development_frame = _read_source(data_config, "desenvolvimento")
    feature_columns = _resolve_feature_columns(development_frame, data_config, target, roles)
    schema = _schema_contract(data_config.get("schema"))
    development = _prepare_source(
        development_frame,
        source_config=data_config,
        source_name="desenvolvimento",
        task=task,
        target=target,
        feature_columns=feature_columns,
        roles=roles,
        schema=schema,
    )

    test = None
    test_config = data_config.get("test")
    if test_config is not None:
        if not isinstance(test_config, Mapping):
            raise ValueError("data.test deve ser um mapa com path e read_options opcionais")
        test_frame = _read_source(test_config, "teste externo")
        test = _prepare_source(
            test_frame,
            source_config=test_config,
            source_name="teste externo",
            task=task,
            target=target,
            feature_columns=feature_columns,
            roles=roles,
            schema=schema,
        )
        _validate_external_schema_compatibility(development, test, schema, target, roles)

    config_fingerprint = config.get("config_fingerprint") or compute_config_fingerprint(config)
    metadata = _preflight_metadata(
        config=config,
        config_fingerprint=config_fingerprint,
        development=development,
        test=test,
        feature_columns=feature_columns,
        roles=roles,
    )
    return {
        "frame": development["frame"],
        "test_frame": test["frame"] if test else None,
        "X": development["X"],
        "y": development["y"],
        "row_ids": development["row_ids"],
        "metadata": development["metadata"],
        "groups": development["groups"],
        "times": development["times"],
        "test_X": test["X"] if test else None,
        "test_y": test["y"] if test else None,
        "test_row_ids": test["row_ids"] if test else None,
        "test_metadata": test["metadata"] if test else None,
        "test_groups": test["groups"] if test else None,
        "test_times": test["times"] if test else None,
        "feature_columns": feature_columns,
        "row_id_column": development["row_id_column"],
        "row_id_stability": development["row_id_stability"],
        "test_row_id_column": test["row_id_column"] if test else None,
        "test_row_id_stability": test["row_id_stability"] if test else None,
        "profile": development["profile"],
        "test_profile": test["profile"] if test else None,
        "data_fingerprint": development["fingerprint"]["digest"],
        "test_data_fingerprint": test["fingerprint"]["digest"] if test else None,
        "schema_signature": development["profile"]["schema_signature"],
        "test_schema_signature": test["profile"]["schema_signature"] if test else None,
        "contract_version": metadata["contract_version"],
        "config_fingerprint": config_fingerprint,
        "run_fingerprint": metadata["run_fingerprint"],
        "preflight_metadata": metadata,
    }


def _read_source(source_config: Mapping[str, Any], source_name: str) -> pl.DataFrame:
    path = source_config.get("path")
    if not isinstance(path, str) or not path:
        raise ValueError(f"data.path é obrigatório para {source_name}")
    read_options = source_config.get("read_options", {})
    if not isinstance(read_options, Mapping):
        raise ValueError(f"read_options de {source_name} deve ser um mapa")
    return auto_read(path, read_options=read_options)


def _column_roles(data_config: Mapping[str, Any], target: str | None) -> dict[str, Any]:
    metadata_columns = data_config.get("metadata_columns", [])
    if metadata_columns is None:
        metadata_columns = []
    if not isinstance(metadata_columns, list) or not all(
        isinstance(column, str) and column for column in metadata_columns
    ):
        raise ValueError("data.metadata_columns deve ser uma lista de nomes de coluna")
    if len(set(metadata_columns)) != len(metadata_columns):
        raise ValueError("data.metadata_columns não pode repetir colunas")

    roles = {
        "target": target,
        "id_column": _optional_column(data_config.get("id_column"), "data.id_column"),
        "metadata_columns": list(metadata_columns),
        "group_column": _optional_column(data_config.get("group_column"), "data.group_column"),
        "time_column": _optional_column(data_config.get("time_column"), "data.time_column"),
    }
    return roles


def _optional_column(value: Any, field: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} deve ser uma string não vazia")
    return value


def _resolve_feature_columns(
    frame: pl.DataFrame,
    data_config: Mapping[str, Any],
    target: str | None,
    roles: Mapping[str, Any],
) -> list[str]:
    declared = data_config.get("features")
    if isinstance(declared, Mapping):
        declared = [column for columns in declared.values() for column in columns]
    if declared is None:
        excluded = {
            column
            for column in (
                target,
                roles["id_column"],
                roles["group_column"],
                roles["time_column"],
                *roles["metadata_columns"],
            )
            if column is not None
        }
        features = [column for column in frame.columns if column not in excluded]
    else:
        if not isinstance(declared, list) or not all(
            isinstance(column, str) and column for column in declared
        ):
            raise ValueError("data.features deve ser uma lista de nomes de coluna")
        if len(set(declared)) != len(declared):
            raise ValueError("data.features não pode repetir colunas")
        features = list(declared)

    protected = {
        column
        for column in (
            target,
            roles["id_column"],
            roles["group_column"],
            roles["time_column"],
            *roles["metadata_columns"],
        )
        if column is not None
    }
    collision = sorted(set(features) & protected)
    if collision:
        raise ValueError(
            "Features não podem incluir target, ID, metadados, grupo ou tempo: "
            f"{collision}"
        )
    if not features:
        raise ValueError("O experimento deve declarar ou resolver ao menos uma feature")
    return features


def _schema_contract(value: Any) -> dict[str, Any]:
    if value is None:
        return {"mode": "permissive", "dtypes": {}}
    if isinstance(value, str):
        value = {"mode": value}
    if not isinstance(value, Mapping):
        raise ValueError("data.schema deve ser um mapa")
    mode = value.get("mode", "permissive")
    if mode not in {"permissive", "strict"}:
        raise ValueError("data.schema.mode deve ser 'permissive' ou 'strict'")
    dtypes = value.get("dtypes", value.get("columns", {}))
    if dtypes is None:
        dtypes = {}
    if not isinstance(dtypes, Mapping) or not all(
        isinstance(column, str)
        and column
        and isinstance(dtype, str)
        and dtype
        for column, dtype in dtypes.items()
    ):
        raise ValueError("data.schema.dtypes deve ser um mapa de coluna para dtype")
    return {"mode": mode, "dtypes": dict(dtypes)}


def _prepare_source(
    frame: pl.DataFrame,
    *,
    source_config: Mapping[str, Any],
    source_name: str,
    task: str,
    target: str | None,
    feature_columns: list[str],
    roles: Mapping[str, Any],
    schema: Mapping[str, Any],
) -> dict[str, Any]:
    path = source_config["path"]
    required_columns = _required_columns(feature_columns, target, roles, task)
    missing = sorted(set(required_columns) - set(frame.columns))
    if missing:
        raise ValueError(f"{source_name}: colunas obrigatórias ausentes: {missing}")
    if frame.height == 0:
        raise ValueError(f"{source_name}: a tabela não contém linhas")

    _validate_schema(frame, source_name, required_columns, schema)
    profile = validate_data(
        frame,
        {
            "task": task,
            "target": target,
            "id_column": roles["id_column"],
        },
    )
    if task in SUPERVISED_TASKS and target is not None and frame[target].null_count() > 0:
        raise ValueError(f"{source_name}: o target '{target}' contém valores ausentes")

    fingerprint = fingerprint_file(path)
    row_ids, row_id_column = _resolve_row_ids(frame, roles["id_column"], fingerprint, source_name)
    if row_id_column == "source_row_id":
        profile["warnings"].append(
            "Nenhum id_column foi declarado; source_row_id depende da ordem da fonte"
        )
    metadata_columns = _metadata_columns(roles)
    return {
        "frame": frame,
        "X": frame.select(feature_columns).to_pandas(),
        "y": frame[target].to_numpy() if task in SUPERVISED_TASKS and target is not None else None,
        "row_ids": row_ids,
        "row_id_column": row_id_column,
        "row_id_stability": "stable_id" if row_id_column != "source_row_id" else "source_order_only",
        "metadata": frame.select(metadata_columns).to_pandas(),
        "groups": (
            frame[roles["group_column"]].to_numpy() if roles["group_column"] is not None else None
        ),
        "times": (
            frame[roles["time_column"]].to_numpy() if roles["time_column"] is not None else None
        ),
        "profile": profile,
        "fingerprint": fingerprint,
    }


def _required_columns(
    feature_columns: list[str],
    target: str | None,
    roles: Mapping[str, Any],
    task: str,
) -> list[str]:
    columns = list(feature_columns)
    if task in SUPERVISED_TASKS and target is not None:
        columns.append(target)
    columns.extend(
        column
        for column in (
            roles["id_column"],
            roles["group_column"],
            roles["time_column"],
            *roles["metadata_columns"],
        )
        if column is not None
    )
    return list(dict.fromkeys(columns))


def _validate_schema(
    frame: pl.DataFrame,
    source_name: str,
    required_columns: list[str],
    schema: Mapping[str, Any],
) -> None:
    declared_dtypes = schema["dtypes"]
    missing_dtypes = sorted(set(declared_dtypes) - set(frame.columns))
    if missing_dtypes:
        raise ValueError(f"{source_name}: schema declara colunas ausentes: {missing_dtypes}")
    mismatches = {
        column: {"expected": expected, "actual": str(frame.schema[column])}
        for column, expected in declared_dtypes.items()
        if not _dtype_matches(str(frame.schema[column]), expected)
    }
    if mismatches:
        raise ValueError(f"{source_name}: dtypes incompatíveis com data.schema: {mismatches}")

    if schema["mode"] == "strict":
        expected_columns = set(required_columns) | set(declared_dtypes)
        extras = sorted(set(frame.columns) - expected_columns)
        if extras:
            raise ValueError(
                f"{source_name}: data.schema.mode=strict não permite colunas extras: {extras}"
            )


def _dtype_matches(actual: str, expected: str) -> bool:
    actual_normalized = actual.lower().replace(" ", "")
    expected_normalized = expected.lower().replace(" ", "")
    if actual_normalized == expected_normalized:
        return True
    aliases = {
        "string": {"string", "utf8", "str"},
        "integer": {"integer", "int", "int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64"},
        "float": {"float", "float32", "float64"},
        "numeric": {"numeric", "integer", "int", "int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64", "float", "float32", "float64"},
        "boolean": {"boolean", "bool"},
        "date": {"date"},
        "datetime": {"datetime"},
    }
    actual_family = _dtype_family(actual_normalized)
    expected_family = _dtype_family(expected_normalized)
    if expected_normalized in aliases:
        return actual_family in aliases[expected_normalized]
    return expected_family == actual_family and expected_family in {"date", "datetime"}


def _dtype_family(dtype: str) -> str:
    if dtype.startswith(("string", "utf8")):
        return "string"
    if dtype.startswith("int") or dtype.startswith("uint"):
        return "integer"
    if dtype.startswith("float"):
        return "float"
    if dtype.startswith("bool"):
        return "boolean"
    if dtype.startswith("datetime"):
        return "datetime"
    if dtype.startswith("date"):
        return "date"
    return dtype


def _resolve_row_ids(
    frame: pl.DataFrame,
    id_column: str | None,
    fingerprint: Mapping[str, Any],
    source_name: str,
) -> tuple[np.ndarray, str]:
    if id_column is None:
        prefix = fingerprint["digest"][:16]
        return np.asarray([f"source_row_id:{prefix}:{index}" for index in range(frame.height)]), "source_row_id"

    identifiers = frame[id_column]
    null_count = identifiers.null_count()
    if null_count:
        raise ValueError(f"{source_name}: id_column '{id_column}' contém {null_count} valores ausentes")
    unique_count = identifiers.n_unique()
    if unique_count != frame.height:
        raise ValueError(
            f"{source_name}: id_column '{id_column}' deve ser único; "
            f"{unique_count} valores distintos para {frame.height} linhas"
        )
    return identifiers.to_numpy(), id_column


def _metadata_columns(roles: Mapping[str, Any]) -> list[str]:
    return list(
        dict.fromkeys(
            [
                *roles["metadata_columns"],
                *(column for column in (roles["group_column"], roles["time_column"]) if column),
            ]
        )
    )


def _validate_external_schema_compatibility(
    development: Mapping[str, Any],
    test: Mapping[str, Any],
    schema: Mapping[str, Any],
    target: str | None,
    roles: Mapping[str, Any],
) -> None:
    if schema["mode"] != "strict":
        return
    columns = [*development["X"].columns]
    if target is not None:
        columns.append(target)
    columns.extend(
        column
        for column in (roles["id_column"], roles["group_column"], roles["time_column"])
        if column is not None
    )
    columns.extend(roles["metadata_columns"])
    mismatches = {
        column: {
            "development": str(development["frame"].schema[column]),
            "test": str(test["frame"].schema[column]),
        }
        for column in dict.fromkeys(columns)
        if str(development["frame"].schema[column]) != str(test["frame"].schema[column])
    }
    if mismatches:
        raise ValueError(f"teste externo: schema incompatível com desenvolvimento: {mismatches}")


def _preflight_metadata(
    *,
    config: Mapping[str, Any],
    config_fingerprint: str,
    development: Mapping[str, Any],
    test: Mapping[str, Any] | None,
    feature_columns: list[str],
    roles: Mapping[str, Any],
) -> dict[str, Any]:
    sources = {
        "development": _source_metadata(development),
        "test": _source_metadata(test) if test else None,
    }
    identity_payload = {
        "config_fingerprint": config_fingerprint,
        "sources": {
            name: source["fingerprint"] if source else None
            for name, source in sources.items()
        },
        "schemas": {
            name: source["schema_signature"] if source else None
            for name, source in sources.items()
        },
    }
    run_fingerprint = hashlib.sha256(
        json.dumps(identity_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {
        "contract_version": config.get("contract_version", DATA_CONTRACT_VERSION),
        "package_version": _package_version(),
        "python_version": platform.python_version(),
        "config_fingerprint": config_fingerprint,
        "run_fingerprint": run_fingerprint,
        "feature_columns": list(feature_columns),
        "roles": {
            "id_column": roles["id_column"],
            "metadata_columns": _metadata_columns(roles),
            "group_column": roles["group_column"],
            "time_column": roles["time_column"],
            "target": roles["target"],
        },
        "provenance": dict(config.get("provenance") or {}),
        "sources": sources,
    }


def _source_metadata(source: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "fingerprint": dict(source["fingerprint"]),
        "schema_signature": source["profile"]["schema_signature"],
        "profile": source["profile"],
        "row_id_column": source["row_id_column"],
        "row_id_stability": source["row_id_stability"],
    }


def _package_version() -> str:
    try:
        return version("ml-playground")
    except PackageNotFoundError:
        return "unknown"
