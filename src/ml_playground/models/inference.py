"""Batch inference with the same schema contract used to train an artifact."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from ml_playground.models.serialization import load_model


class SchemaValidationError(ValueError):
    """Raised when an inference batch does not satisfy an artifact contract."""


class PredictionCapabilityError(ValueError):
    """Raised when a persisted estimator cannot score a new batch."""


@dataclass(frozen=True)
class BatchPredictionResult:
    """Predictions plus the validation evidence used to produce them."""

    frame: pd.DataFrame
    validation: dict[str, Any]
    metadata: dict[str, Any]
    output_path: str | None = None


def predict_batch(
    artifact_path: str | Path,
    data: Any,
    *,
    id_column: str | None = None,
    feature_columns: Iterable[str] | None = None,
    strict: bool = True,
    allow_extra_columns: bool | None = None,
    output_path: str | Path | None = None,
    output_format: str | None = None,
) -> BatchPredictionResult:
    """Validate and score a tabular batch with a persisted pipeline.

    The artifact metadata may carry a ``signature`` or ``schema`` dictionary
    containing ``features``, ``dtypes``, ``target``, ``id_column`` and
    ``id_unique``.  Older artifacts remain usable when sklearn exposes
    ``feature_names_in_``; otherwise callers must explicitly provide
    ``feature_columns`` rather than making an unsafe positional prediction.
    """

    pipeline, artifact_metadata = load_model(artifact_path)
    if artifact_metadata is None:
        metadata = {}
    elif isinstance(artifact_metadata, dict):
        metadata = dict(artifact_metadata)
    else:
        raise SchemaValidationError("Os metadados do artefato devem ser um mapa")
    frame = _as_dataframe(data)
    contract = _resolve_contract(pipeline, metadata, feature_columns)
    features, identifiers, validation = _validate_batch(
        frame,
        contract,
        id_column=id_column,
        strict=strict,
        allow_extra_columns=allow_extra_columns,
    )

    if not hasattr(pipeline, "predict"):
        raise PredictionCapabilityError("O artefato não expõe predict para novas linhas")

    predictions = np.asarray(pipeline.predict(features))
    if predictions.ndim != 1:
        raise PredictionCapabilityError("predict deve retornar exatamente uma previsão por linha")
    if len(predictions) != len(features):
        raise PredictionCapabilityError("A quantidade de previsões não corresponde ao lote")

    result = pd.DataFrame({validation["id_column"]: identifiers.to_numpy()})
    result["prediction"] = predictions
    _append_scores(result, pipeline, features)

    written_path = None
    if output_path is not None:
        written_path = _write_prediction_output(result, output_path, output_format)
    return BatchPredictionResult(
        frame=result,
        validation=validation,
        metadata=metadata,
        output_path=written_path,
    )


def _as_dataframe(data: Any) -> pd.DataFrame:
    if isinstance(data, pd.DataFrame):
        frame = data.copy()
    elif hasattr(data, "to_pandas"):
        frame = data.to_pandas()
    else:
        frame = pd.DataFrame(data)
    if frame.empty:
        raise SchemaValidationError("O lote de inferência não pode estar vazio")
    if frame.columns.has_duplicates:
        duplicates = frame.columns[frame.columns.duplicated()].tolist()
        raise SchemaValidationError(f"O lote possui colunas duplicadas: {duplicates}")
    return frame


def _resolve_contract(pipeline, metadata: dict[str, Any], feature_columns: Iterable[str] | None) -> dict[str, Any]:
    source_name = "metadata"
    source = metadata.get("signature")
    if not isinstance(source, dict):
        source = metadata.get("schema") or metadata.get("data_contract")
    if not isinstance(source, dict):
        source = {}
        source_name = "sklearn"

    expected_features = _feature_names(source.get("features"))
    if not expected_features:
        expected_features = _feature_names(metadata.get("features"))
    inferred = _pipeline_feature_names(pipeline)
    if not expected_features:
        expected_features = inferred

    if isinstance(feature_columns, str):
        raise SchemaValidationError("feature_columns deve ser uma sequência de nomes, não uma string")
    requested_features = list(feature_columns) if feature_columns is not None else None
    if requested_features is not None:
        if not requested_features or not all(isinstance(column, str) and column for column in requested_features):
            raise SchemaValidationError("feature_columns deve ser uma sequência não vazia de nomes")
        if len(set(requested_features)) != len(requested_features):
            raise SchemaValidationError("feature_columns contém nomes duplicados")
        if expected_features and requested_features != expected_features:
            raise SchemaValidationError(
                "feature_columns diverge da assinatura persistida do modelo"
            )
        expected_features = requested_features
        source_name = "caller"

    if not expected_features:
        raise SchemaValidationError(
            "O artefato não possui nomes de features. Informe feature_columns "
            "explicitamente ou salve o modelo com uma signature."
        )

    dtypes = source.get("dtypes", metadata.get("dtypes", {}))
    if not isinstance(dtypes, dict):
        dtypes = {}
    return {
        "source": source_name,
        "features": expected_features,
        "dtypes": dtypes,
        "target": source.get("target", metadata.get("target")),
        "id_column": source.get("id_column", metadata.get("id_column")),
        "id_unique": bool(source.get("id_unique", metadata.get("id_unique", False))),
        "allow_extra_columns": source.get(
            "allow_extra_columns",
            metadata.get("allow_extra_columns", True),
        ),
    }


def _feature_names(values: Any) -> list[str]:
    if not isinstance(values, (list, tuple, np.ndarray, pd.Index)):
        return []
    names = []
    for value in values:
        if isinstance(value, dict):
            value = value.get("name")
        if isinstance(value, str) and value:
            names.append(value)
    return names if len(names) == len(values) and len(set(names)) == len(names) else []


def _pipeline_feature_names(pipeline) -> list[str]:
    candidates = [pipeline]
    named_steps = getattr(pipeline, "named_steps", {})
    if isinstance(named_steps, dict):
        candidates.extend(named_steps.values())
    for candidate in candidates:
        names = _feature_names(getattr(candidate, "feature_names_in_", None))
        if names:
            return names
    return []


def _validate_batch(
    frame: pd.DataFrame,
    contract: dict[str, Any],
    *,
    id_column: str | None,
    strict: bool,
    allow_extra_columns: bool | None,
) -> tuple[pd.DataFrame, pd.Series, dict[str, Any]]:
    expected = contract["features"]
    missing = [column for column in expected if column not in frame.columns]
    if missing:
        raise SchemaValidationError(f"Features ausentes no lote: {missing}")

    selected_id = id_column if id_column is not None else contract.get("id_column")
    if selected_id is not None and selected_id not in frame.columns:
        raise SchemaValidationError(f"Coluna de ID ausente no lote: {selected_id}")
    if selected_id in {"prediction", "score"}:
        raise SchemaValidationError("A coluna de ID usa um nome reservado para a saída")

    target = contract.get("target")
    allowed = set(expected)
    if selected_id:
        allowed.add(selected_id)
    if isinstance(target, str):
        allowed.add(target)
    extras = [column for column in frame.columns if column not in allowed]
    extras_allowed = contract["allow_extra_columns"] if allow_extra_columns is None else allow_extra_columns
    if strict and not extras_allowed and extras:
        raise SchemaValidationError(f"Colunas não previstas na assinatura: {extras}")

    type_errors = _dtype_errors(frame, expected, contract.get("dtypes", {}))
    if strict and type_errors:
        raise SchemaValidationError("Tipos incompatíveis no lote: " + "; ".join(type_errors))

    if selected_id is None:
        identifiers = pd.Series(frame.index.to_numpy(), index=frame.index, name="row_id")
        output_id_column = "row_id"
        generated_id = True
    else:
        identifiers = frame[selected_id]
        if identifiers.isna().any():
            raise SchemaValidationError(f"A coluna de ID {selected_id!r} contém valores ausentes")
        if contract.get("id_unique") and identifiers.duplicated().any():
            raise SchemaValidationError(f"A coluna de ID {selected_id!r} deve ser única")
        output_id_column = selected_id
        generated_id = False

    return (
        frame.loc[:, expected].copy(),
        identifiers,
        {
            "contract_source": contract["source"],
            "expected_features": expected,
            "extra_columns": extras,
            "id_column": output_id_column,
            "generated_id": generated_id,
            "n_rows": len(frame),
            "strict": strict,
        },
    )


def _dtype_errors(frame: pd.DataFrame, features: list[str], dtypes: dict[str, Any]) -> list[str]:
    errors = []
    for feature in features:
        expected = dtypes.get(feature)
        if expected is None:
            continue
        if not _dtype_matches(frame[feature], str(expected)):
            errors.append(
                f"{feature}: esperado {expected}, recebido {frame[feature].dtype}"
            )
    return errors


def _dtype_matches(values: pd.Series, expected: str) -> bool:
    token = expected.casefold()
    dtype = values.dtype
    if any(part in token for part in ("int", "float", "number", "numeric", "double", "decimal")):
        return pd.api.types.is_numeric_dtype(dtype) and not pd.api.types.is_bool_dtype(dtype)
    if "bool" in token:
        return pd.api.types.is_bool_dtype(dtype)
    if any(part in token for part in ("date", "time", "datetime")):
        return pd.api.types.is_datetime64_any_dtype(dtype)
    if any(
        part in token
        for part in ("str", "string", "object", "utf8", "unicode", "category", "categorical")
    ):
        if isinstance(dtype, pd.CategoricalDtype):
            return values.dropna().astype("object").map(
                lambda value: isinstance(value, str)
            ).all()
        if pd.api.types.is_object_dtype(dtype):
            return values.dropna().map(lambda value: isinstance(value, str)).all()
        return pd.api.types.is_string_dtype(dtype)
    return str(dtype).casefold() == token


def _append_scores(result: pd.DataFrame, pipeline, features: pd.DataFrame) -> None:
    if hasattr(pipeline, "predict_proba"):
        probabilities = np.asarray(pipeline.predict_proba(features))
        if probabilities.ndim == 1:
            result["probability"] = probabilities
            return
        if probabilities.ndim != 2:
            raise PredictionCapabilityError("predict_proba retornou formato inválido")
        labels = _class_labels(pipeline, probabilities.shape[1])
        for index, label in enumerate(labels):
            column = _unique_column_name(result, f"probability_{_label_token(label, index)}")
            result[column] = probabilities[:, index]
        return

    if hasattr(pipeline, "decision_function"):
        scores = np.asarray(pipeline.decision_function(features))
        if scores.ndim == 1:
            result["decision_score"] = scores
        elif scores.ndim == 2:
            labels = _class_labels(pipeline, scores.shape[1])
            for index, label in enumerate(labels):
                column = _unique_column_name(result, f"decision_score_{_label_token(label, index)}")
                result[column] = scores[:, index]


def _class_labels(pipeline, width: int) -> list[Any]:
    labels = getattr(pipeline, "classes_", None)
    if labels is None:
        model = getattr(pipeline, "named_steps", {}).get("model")
        labels = getattr(model, "classes_", None)
    labels = list(labels) if labels is not None else []
    return labels if len(labels) == width else list(range(width))


def _label_token(label: Any, index: int) -> str:
    token = re.sub(r"[^A-Za-z0-9_]+", "_", str(label)).strip("_")
    return token or str(index)


def _unique_column_name(frame: pd.DataFrame, base: str) -> str:
    if base not in frame.columns:
        return base
    suffix = 2
    while f"{base}_{suffix}" in frame.columns:
        suffix += 1
    return f"{base}_{suffix}"


def _write_prediction_output(frame: pd.DataFrame, output_path: str | Path, output_format: str | None) -> str:
    path = Path(output_path)
    selected_format = (output_format or path.suffix.lstrip(".") or "csv").casefold()
    if selected_format not in {"csv", "parquet"}:
        raise ValueError("output_format deve ser 'csv' ou 'parquet'")
    if not path.suffix:
        path = path.with_suffix(f".{selected_format}")
    path.parent.mkdir(parents=True, exist_ok=True)
    if selected_format == "csv":
        frame.to_csv(path, index=False)
    else:
        frame.to_parquet(path, index=False)
    return str(path)
