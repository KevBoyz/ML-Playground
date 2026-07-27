import numpy as np
from ml_playground.preprocessing.pipelines import (
    build_pipeline,
    build_column_transformer,
)


def test_build_pipeline_sequential():
    config = [
        {"name": "scale", "category": "scaling", "method": "standard"},
    ]
    pipe = build_pipeline(config)
    X = np.array([[1.0, 2.0], [3.0, 4.0]])
    result = pipe.fit_transform(X)
    assert result.shape == (2, 2)


def test_build_pipeline_multiple_steps():
    config = [
        {"name": "impute", "category": "imputation", "method": "mean"},
        {"name": "scale", "category": "scaling", "method": "standard"},
        {
            "name": "pca",
            "category": "dimensionality",
            "method": "pca",
            "params": {"n_components": 2},
        },
    ]
    pipe = build_pipeline(config)
    X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    result = pipe.fit_transform(X)
    assert result.shape[1] == 2


def test_build_pipeline_skips_passthrough():
    config = [
        {"name": "noop", "category": "transformation", "method": "none"},
        {"name": "scale", "category": "scaling", "method": "standard"},
    ]
    pipe = build_pipeline(config)
    assert len(pipe.steps) == 1
    assert pipe.steps[0][0] == "scale"


def test_build_pipeline_all_passthrough():
    config = [
        {"name": "a", "category": "transformation", "method": "none"},
        {"name": "b", "category": "feature_selection", "method": "none"},
    ]
    pipe = build_pipeline(config)
    assert len(pipe.steps) == 1
    assert pipe.steps[0][1] == "passthrough"


def test_build_pipeline_flat_config():
    config = {
        "imputation": {"method": "mean"},
        "scaling": {"method": "standard"},
        "transformation": {"method": "none"},
    }
    pipe = build_pipeline(config)
    assert len(pipe.steps) == 2


def test_build_column_transformer():
    config = [
        {
            "name": "scale_num",
            "category": "scaling",
            "method": "standard",
            "columns": [0, 1],
        },
        {
            "name": "noop",
            "category": "transformation",
            "method": "none",
            "columns": [2],
        },
    ]
    ct = build_column_transformer(config)
    assert len(ct.transformers) == 1
    assert ct.transformers[0][0] == "scale_num"
