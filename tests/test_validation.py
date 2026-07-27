import polars as pl
from ml_playground.data.validation import check_missing, check_dtypes, validate_data


def test_check_missing():
    df = pl.DataFrame({"a": [1, None], "b": [None, None]})
    result = check_missing(df)
    assert result.height == 2
    assert result.filter(pl.col("column") == "a")[0, "null_count"] == 1
    assert result.filter(pl.col("column") == "b")[0, "null_count"] == 2


def test_check_missing_sem_nulos():
    df = pl.DataFrame({"a": [1, 2], "b": [3, 4]})
    assert check_missing(df).height == 0


def test_check_dtypes():
    df = pl.DataFrame({"a": [1, 2], "b": ["x", "y"]})
    expected = {"a": "Int64", "b": "String"}
    result = check_dtypes(df, expected)
    assert result == {}


def test_check_dtypes_coluna_ausente():
    df = pl.DataFrame({"a": [1]})
    expected = {"b": "Int64"}
    result = check_dtypes(df, expected)
    assert result["b"] == "ausente"


def test_validate_data():
    df = pl.DataFrame({"a": [1, None], "b": ["x", "y"]})
    info = validate_data(df)
    assert info["rows"] == 2
    assert info["cols"] == 2
    assert "a" in info["missing"]
    assert len(info["warnings"]) >= 1
