import polars as pl
from ml_playground.data.cleaning import fill_missing


def test_fill_missing_mean():
    df = pl.DataFrame({"a": [1.0, None, 3.0]})
    result = fill_missing(df, strategy="mean")
    assert result["a"][1] == 2.0


def test_fill_missing_constant():
    df = pl.DataFrame({"a": [1, None, 3]})
    result = fill_missing(df, strategy="constant", fill_value=0)
    assert result["a"][1] == 0


def test_fill_missing_forward():
    df = pl.DataFrame({"a": [1, None, 3]})
    result = fill_missing(df, strategy="forward")
    assert result["a"][1] == 1
