from pathlib import Path
from typing import Any

import polars as pl

SUPPORTED_EXTENSIONS = {".csv", ".parquet", ".xlsx", ".xls", ".ods"}


def _resolve_path(path: str | Path) -> Path:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {p.resolve()}")
    return p


def auto_read(path: str | Path, **kwargs: Any) -> pl.DataFrame:
    p = _resolve_path(path)
    ext = p.suffix.lower()
    if ext == ".csv":
        return pl.read_csv(p, **kwargs)
    elif ext in (".xlsx", ".xls", ".ods"):
        from polars import read_excel as _read_excel

        return _read_excel(p, **kwargs)
    elif ext == ".parquet":
        return pl.read_parquet(p, **kwargs)
    else:
        raise ValueError(f"Extensão não suportada: {ext}. Use: {SUPPORTED_EXTENSIONS}")
