from ml_playground.experiments.grid import build_model_grid, expand_params


def test_expand_params_scalar():
    result = expand_params({"a": 1, "b": 2})
    assert result == [{"a": 1, "b": 2}]


def test_expand_params_list():
    result = expand_params({"a": [1, 2], "b": [3, 4]})
    assert len(result) == 4
    assert {"a": 1, "b": 3} in result
    assert {"a": 1, "b": 4} in result
    assert {"a": 2, "b": 3} in result
    assert {"a": 2, "b": 4} in result


def test_expand_params_mixed():
    result = expand_params({"a": [1, 2], "b": 3})
    assert len(result) == 2
    assert {"a": 1, "b": 3} in result
    assert {"a": 2, "b": 3} in result


def test_expand_params_single_list():
    result = expand_params({"a": [1, 2, 3]})
    assert len(result) == 3


def test_build_model_grid():
    models = [
        {"name": "rf", "params": {"n": [10, 20]}},
        {"name": "svm", "params": {"c": [0.1]}},
    ]
    grid = build_model_grid(models)
    assert len(grid) == 3  # 2 rf + 1 svm
    assert grid[0] == {"name": "rf", "params": {"n": 10}}
    assert grid[1] == {"name": "rf", "params": {"n": 20}}
    assert grid[2] == {"name": "svm", "params": {"c": 0.1}}


def test_build_model_grid_no_params():
    models = [{"name": "rf"}]
    grid = build_model_grid(models)
    assert len(grid) == 1
    assert grid[0] == {"name": "rf", "params": {}}
