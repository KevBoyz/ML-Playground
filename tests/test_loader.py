from pathlib import Path
import polars as pl
import pytest
from ml_playground.data.loader import auto_read, _resolve_path


def test_resolve_path_inexistente():
    with pytest.raises(FileNotFoundError):
        _resolve_path("data/nao_existe.csv")


def test_auto_read_csv(tmp_path: Path):
    csv = tmp_path / "data.csv"
    csv.write_text("x,y\n1,2\n")
    df = auto_read(csv)
    assert df.shape == (1, 2)


def test_auto_read_csv_com_kwargs(tmp_path: Path):
    csv = tmp_path / "data.csv"
    csv.write_text("a|b\n1|2\n")
    df = auto_read(csv, separator="|")
    assert df.shape == (1, 2)


def test_auto_read_parquet(tmp_path: Path):
    pq = tmp_path / "data.parquet"
    pl.DataFrame({"a": [1, 2]}).write_parquet(pq)
    df = auto_read(pq)
    assert df.shape == (2, 1)


def test_auto_read_extensao_invalida(tmp_path: Path):
    f = tmp_path / "data.txt"
    f.write_text("")
    with pytest.raises(ValueError, match="não suportada"):
        auto_read(f)
