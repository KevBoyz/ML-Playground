from pathlib import Path

import pytest
import yaml

from ml_playground.experiments.templates import initialize_experiment


def test_initialize_classification_experiment(tmp_path):
    directory = tmp_path / "churn"

    created = initialize_experiment(directory, task="classification")

    assert {path.name for path in created} == {
        "experiment.yaml",
        "models.yaml",
        "preprocessing.yaml",
        "metrics.yaml",
        "cross_validation.yaml",
        "views.yaml",
    }
    experiment = yaml.safe_load((directory / "experiment.yaml").read_text(encoding="utf-8"))
    assert experiment["name"] == "churn"
    assert experiment["data"]["id_column"] == "record_id"
    assert experiment["provenance"]["recipe_ref"]


def test_initialize_clustering_does_not_create_cross_validation(tmp_path):
    directory = tmp_path / "segments"

    initialize_experiment(directory, task="clustering")

    assert not (directory / "cross_validation.yaml").exists()


def test_initialize_refuses_to_overwrite_existing_contract(tmp_path):
    directory = tmp_path / "existing"
    initialize_experiment(directory, task="regression")

    with pytest.raises(FileExistsError):
        initialize_experiment(directory, task="regression")


def test_initialize_accepts_explicit_name(tmp_path):
    directory = tmp_path / "folder"

    initialize_experiment(directory, task="regression", name="pricing_v2")

    document = yaml.safe_load((directory / "experiment.yaml").read_text(encoding="utf-8"))
    assert document["name"] == "pricing_v2"
