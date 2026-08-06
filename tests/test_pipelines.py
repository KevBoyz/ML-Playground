import numpy as np
import pandas as pd
import pytest
from ml_playground.preprocessing.pipelines import (
    build_pipeline,
    build_column_transformer,
    build_preprocessor,
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


def test_build_preprocessor_accepts_explicit_feature_groups():
    frame = pd.DataFrame(
        {
            "continuous": [1.0, 2.0, 3.0],
            "ordinal": [1, 2, 3],
            "category": ["a", "b", "a"],
        }
    )
    config = {
        "groups": {
            "continuous": {
                "columns": ["continuous"],
                "steps": [{"name": "scale", "category": "scaling", "method": "standard"}],
            },
            "ordinal": {"columns": ["ordinal"], "steps": []},
            "nominal": {
                "columns": ["category"],
                "steps": [{"name": "onehot", "category": "encoding", "method": "onehot"}],
            },
        }
    }

    preprocessor = build_preprocessor(config, frame)
    transformed = preprocessor.fit_transform(frame)

    assert transformed.shape == (3, 4)


def test_explicit_feature_groups_reject_column_overlap():
    frame = pd.DataFrame({"value": [1.0, 2.0]})
    config = {
        "groups": {
            "first": {"columns": ["value"], "steps": []},
            "second": {"columns": ["value"], "steps": []},
        }
    }

    with pytest.raises(ValueError, match="repetidas"):
        build_preprocessor(config, frame)
