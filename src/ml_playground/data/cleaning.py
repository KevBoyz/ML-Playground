from typing import Any, Optional

import polars as pl


def fill_missing(
    df: pl.DataFrame,
    strategy: str = "mean",
    columns: Optional[list[str]] = None,
    fill_value: Optional[Any] = None,
) -> pl.DataFrame:
    cols = columns or df.columns
    result = df.clone()

    if strategy == "constant":
        for col in cols:
            if fill_value is not None:
                result = result.with_columns(pl.col(col).fill_null(fill_value))
        return result

    for col in cols:
        if strategy in ("mean", "median") and result[col].dtype in (
            pl.Float32,
            pl.Float64,
            pl.Int8,
            pl.Int16,
            pl.Int32,
            pl.Int64,
            pl.UInt8,
            pl.UInt16,
            pl.UInt32,
            pl.UInt64,
        ):
            val = result[col].mean() if strategy == "mean" else result[col].median()
            result = result.with_columns(pl.col(col).fill_null(val))
        elif strategy == "mode":
            mode_val = result[col].drop_nulls().mode()
            if mode_val.height > 0:
                result = result.with_columns(pl.col(col).fill_null(mode_val[0]))
        elif strategy == "forward":
            result = result.with_columns(pl.col(col).fill_null(strategy="forward"))
        elif strategy == "backward":
            result = result.with_columns(pl.col(col).fill_null(strategy="backward"))

    return result
