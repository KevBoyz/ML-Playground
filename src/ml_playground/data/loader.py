from pathlib import Path
from typing import Any, Mapping

import polars as pl

SUPPORTED_EXTENSIONS = {".csv", ".parquet", ".xlsx", ".xls", ".ods"}


def _resolve_path(path: str | Path) -> Path:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {p.resolve()}")
    return p


def auto_read(
    path: str | Path,
    *,
    read_options: Mapping[str, Any] | None = None,
    **kwargs: Any,
) -> pl.DataFrame:
    """Dispatch an explicit set of reader options to the source format."""

    p = _resolve_path(path)
    if read_options is not None and not isinstance(read_options, Mapping):
        raise ValueError("read_options deve ser um mapa de opções do leitor")
    options = dict(read_options or {})
    options.update(kwargs)
    ext = p.suffix.lower()
    if ext == ".csv":
        return pl.read_csv(p, **options)
    elif ext in (".xlsx", ".xls", ".ods"):
        from polars import read_excel as _read_excel

        return _read_excel(p, **options)
    elif ext == ".parquet":
        return pl.read_parquet(p, **options)
    else:
        raise ValueError(f"Extensão não suportada: {ext}. Use: {SUPPORTED_EXTENSIONS}")
