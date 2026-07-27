import numpy as np
import pytest

from ml_playground.experiments.cache import PipelineCache


def test_cache_fits_and_returns_pipeline(tmp_path):
    cache = PipelineCache(cache_dir=str(tmp_path / "cache"))
    config = {"scaling": {"method": "standard"}}
    X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    pipe = cache.get_or_fit(config, X)
    assert pipe is not None
    result = pipe.transform(X)
    assert result.shape == X.shape


def test_cache_hits_on_second_call(tmp_path):
    cache = PipelineCache(cache_dir=str(tmp_path / "cache"))
    config = {"scaling": {"method": "minmax"}}
    X = np.array([[1.0, 2.0], [3.0, 4.0]])
    pipe1 = cache.get_or_fit(config, X)
    pipe2 = cache.get_or_fit(config, X)
    xform1 = pipe1.transform(X)
    xform2 = pipe2.transform(X)
    assert xform1 == pytest.approx(xform2)


def test_cache_clear(tmp_path):
    cache = PipelineCache(cache_dir=str(tmp_path / "cache"))
    config = {"scaling": {"method": "standard"}}
    X = np.array([[1.0, 2.0]])
    cache.get_or_fit(config, X)
    assert len(list(cache.cache_dir.iterdir())) > 0
    cache.clear()
    assert not cache.cache_dir.exists() or len(list(cache.cache_dir.iterdir())) == 0
