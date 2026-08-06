"""Lightweight data-quality evidence used by experiment preflight."""

from __future__ import annotations

import hashlib
import math
from collections import Counter
from datetime import date, datetime
from pathlib import Path
from typing import Any, Mapping

import polars as pl


def check_missing(df: pl.DataFrame) -> pl.DataFrame:
    counts = []
    for col in df.columns:
        n = df[col].null_count()
        if n > 0:
            counts.append({"column": col, "null_count": n})
    return pl.DataFrame(counts, schema={"column": pl.Utf8, "null_count": pl.UInt32})


def check_dtypes(df: pl.DataFrame, expected: dict[str, Any]) -> dict[str, str]:
    result = {}
    for col, expected_type in expected.items():
        if col not in df.columns:
            result[col] = "ausente"
        else:
            actual = str(df[col].dtype)
            if actual != expected_type:
                result[col] = f"esperado {expected_type}, obtido {actual}"
    return result


def fingerprint_file(path: str | Path, *, chunk_size: int = 1_048_576) -> dict[str, Any]:
    """Build content identity for a declared source without loading it as a table."""

    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {source.resolve()}")
    if not source.is_file():
        raise ValueError(f"O caminho de dados deve apontar para um arquivo: {source}")

    digest = hashlib.sha256()
    with source.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return {
        "algorithm": "sha256",
        "digest": digest.hexdigest(),
        "size_bytes": source.stat().st_size,
    }


def schema_signature(df: pl.DataFrame) -> str:
    """Create an ordered, type-aware signature for the observed input schema."""

    payload = "\n".join(f"{column}\t{df.schema[column]}" for column in df.columns)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def validate_data(df: pl.DataFrame, config: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Profile one in-memory table without changing it or fitting any estimator."""

    options = dict(config or {})
    dtypes = {column: str(df.schema[column]) for column in df.columns}
    missing = {
        column: int(df[column].null_count())
        for column in df.columns
        if df[column].null_count() > 0
    }
    duplicate_count = df.height - df.unique(keep="first").height
    info: dict[str, Any] = {
        "rows": df.height,
        "cols": df.width,
        "columns": list(df.columns),
        "dtypes": dtypes,
        "schema": dtypes,
        "schema_signature": schema_signature(df),
        "missing": missing,
        "duplicates": duplicate_count,
        "cardinality": {
            column: int(df[column].n_unique())
            for column in df.columns
        },
        "warnings": [],
    }

    if df.height == 0:
        info["warnings"].append("A fonte não contém linhas")
    for column, nulls in missing.items():
        info["warnings"].append(f"{column}: {nulls} valores ausentes")
    if duplicate_count > 0:
        info["warnings"].append(f"{duplicate_count} linhas duplicadas")

    target = options.get("target")
    if target:
        if target not in df.columns:
            info["warnings"].append(f"Target configurado ausente: {target}")
        else:
            target_profile = _profile_target(df[target])
            info["target"] = target_profile
            if target_profile["missing"]:
                info["warnings"].append(
                    f"{target}: {target_profile['missing']} valores ausentes no target"
                )
            if options.get("task") == "classification":
                rare = [
                    item
                    for item in target_profile.get("distribution", [])
                    if item["count"] < 2
                ]
                if rare:
                    info["warnings"].append(
                        f"{target}: classes com menos de 2 linhas: "
                        f"{[item['value'] for item in rare]}"
                    )

    id_column = options.get("id_column")
    if id_column and id_column in df.columns:
        id_series = df[id_column]
        info["id"] = {
            "column": id_column,
            "missing": int(id_series.null_count()),
            "n_unique": int(id_series.n_unique()),
            "is_unique": int(id_series.n_unique()) == df.height,
        }
    elif id_column:
        info["warnings"].append(f"Coluna de ID configurada ausente: {id_column}")

    return info


def _profile_target(series: pl.Series) -> dict[str, Any]:
    values = series.drop_nulls().to_list()
    counts = Counter(_json_value(value) for value in values)
    unique_count = int(series.n_unique())
    profile: dict[str, Any] = {
        "column": series.name,
        "dtype": str(series.dtype),
        "missing": int(series.null_count()),
        "n_unique": unique_count,
    }
    if unique_count <= 20:
        profile["distribution"] = [
            {"value": value, "count": count}
            for value, count in sorted(counts.items(), key=lambda item: (-item[1], str(item[0])))
        ]
    elif series.dtype.is_numeric():
        profile["summary"] = {
            "min": _json_value(series.min()),
            "max": _json_value(series.max()),
            "mean": _json_value(series.mean()),
        }
    else:
        profile["summary"] = {"kind": "high_cardinality"}
    return profile


def _json_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if hasattr(value, "item"):
        return _json_value(value.item())
    return str(value)
