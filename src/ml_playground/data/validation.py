from typing import Any, Optional

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


def validate_data(df: pl.DataFrame, config: Optional[dict] = None) -> dict:
    info = {
        "rows": df.height,
        "cols": df.width,
        "columns": df.columns,
        "dtypes": {c: str(df[c].dtype) for c in df.columns},
        "missing": {},
        "duplicates": 0,
        "warnings": [],
    }

    for col in df.columns:
        nulls = df[col].null_count()
        if nulls > 0:
            info["missing"][col] = nulls
            info["warnings"].append(f"{col}: {nulls} valores ausentes")

    dup = df.height - df.unique(keep="first").height
    if dup > 0:
        info["duplicates"] = dup
        info["warnings"].append(f"{dup} linhas duplicadas")

    return info
