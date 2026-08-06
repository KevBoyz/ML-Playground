import json
from pathlib import Path

import polars as pl
import pytest

from ml_playground.experiments.config import load_experiment
from ml_playground.experiments.preflight import prepare_experiment_data


def _write_contract_sources(tmp_path: Path) -> tuple[Path, Path]:
    development = tmp_path / "development.csv"
    external_test = tmp_path / "external_test.csv"
    pl.DataFrame(
        {
            "customer_id": [101, 102, 103, 104],
            "household_id": [1, 1, 2, 3],
            "event_at": ["2026-01-01", "2026-01-02", "2026-01-03", "2026-01-04"],
            "region": ["north", "north", "south", "west"],
            "age": [21.0, 32.0, 43.0, 54.0],
            "plan": ["basic", "plus", "basic", "pro"],
            "target": [0, 1, 0, 1],
        }
    ).write_csv(development, separator=";")
    pl.DataFrame(
        {
            "customer_id": [201, 202],
            "household_id": [4, 5],
            "event_at": ["2026-02-01", "2026-02-02"],
            "region": ["north", "south"],
            "age": [25.0, 35.0],
            "plan": ["basic", "pro"],
            "target": [0, 1],
        }
    ).write_csv(external_test, separator=";")
    return development, external_test


def _contract_config(development: Path, external_test: Path) -> dict:
    return {
        "task": "classification",
        "contract_version": 1,
        "data": {
            "path": str(development),
            "read_options": {"separator": ";"},
            "test": {"path": str(external_test), "read_options": {"separator": ";"}},
            "target": "target",
            "id_column": "customer_id",
            "metadata_columns": ["region"],
            "group_column": "household_id",
            "time_column": "event_at",
            "features": ["age", "plan"],
            "schema": {"mode": "strict"},
        },
        "provenance": {
            "recipe_ref": "recipes/churn.py",
            "recipe_revision": "abc123",
            "source_description": "fixture prepared table",
        },
    }


def test_prepare_experiment_data_reuses_read_contract_and_serializable_metadata(tmp_path: Path):
    development, external_test = _write_contract_sources(tmp_path)

    prepared = prepare_experiment_data(_contract_config(development, external_test))

    assert prepared["X"].columns.tolist() == ["age", "plan"]
    assert prepared["row_ids"].tolist() == [101, 102, 103, 104]
    assert prepared["groups"].tolist() == [1, 1, 2, 3]
    assert prepared["metadata"].columns.tolist() == ["region", "household_id", "event_at"]
    assert prepared["test_y"].tolist() == [0, 1]
    assert prepared["data_fingerprint"] != prepared["test_data_fingerprint"]
    assert prepared["row_id_stability"] == "stable_id"
    assert prepared["preflight_metadata"]["provenance"]["recipe_ref"] == "recipes/churn.py"
    json.dumps(prepared["preflight_metadata"])


def test_prepare_experiment_data_rejects_invalid_id_before_training(tmp_path: Path):
    development, external_test = _write_contract_sources(tmp_path)
    duplicate = pl.read_csv(development, separator=";").with_columns(
        pl.lit(101).alias("customer_id")
    )
    duplicate.write_csv(development, separator=";")

    with pytest.raises(ValueError, match="id_column 'customer_id' deve ser único"):
        prepare_experiment_data(_contract_config(development, external_test))


def test_strict_external_schema_rejects_dtype_drift(tmp_path: Path):
    development, external_test = _write_contract_sources(tmp_path)
    drifted = pl.read_csv(external_test, separator=";").with_columns(
        pl.lit("unknown").alias("age")
    )
    drifted.write_csv(external_test, separator=";")

    with pytest.raises(ValueError, match="schema incompatível"):
        prepare_experiment_data(_contract_config(development, external_test))


def test_prepare_experiment_data_generates_source_row_id_when_no_business_id(tmp_path: Path):
    development, _ = _write_contract_sources(tmp_path)
    config = _contract_config(development, tmp_path / "unused.csv")
    config["data"].pop("test")
    config["data"].pop("id_column")
    config["data"]["schema"] = {"mode": "permissive"}

    prepared = prepare_experiment_data(config)

    assert prepared["row_id_column"] == "source_row_id"
    assert prepared["row_id_stability"] == "source_order_only"
    assert prepared["row_ids"][0].startswith("source_row_id:")
    assert any("depende da ordem" in warning for warning in prepared["profile"]["warnings"])


def test_load_experiment_normalizes_new_data_contract(tmp_path: Path):
    development, external_test = _write_contract_sources(tmp_path)
    experiment = tmp_path / "experiments" / "contract_demo"
    experiment.mkdir(parents=True)
    (experiment / "experiment.yaml").write_text(
        "name: contract_demo\n"
        "task: classification\n"
        "contract_version: 1\n"
        "data:\n"
        f"  path: {development.name}\n"
        "  read_options: {separator: ';'}\n"
        "  test:\n"
        f"    path: {external_test.name}\n"
        "    read_options: {separator: ';'}\n"
        "  target: target\n"
        "  id_column: customer_id\n"
        "  metadata_columns: [region]\n"
        "  group_column: household_id\n"
        "  time_column: event_at\n"
        "  features:\n"
        "    numeric: [age]\n"
        "    categorical: [plan]\n"
        "  schema: {mode: strict}\n"
        "provenance:\n"
        "  recipe_ref: recipes/demo.py\n"
        "  recipe_revision: abc123\n"
        "  source_description: fixture\n"
        "evaluation: {protocol: development_cv_final_test}\n",
        encoding="utf-8",
    )
    (experiment / "models.yaml").write_text(
        "models:\n  - name: decision_tree\n    params: {max_depth: [2]}\n",
        encoding="utf-8",
    )
    (experiment / "preprocessing.yaml").write_text("{}\n", encoding="utf-8")
    (experiment / "metrics.yaml").write_text(
        "classification:\n  names: [accuracy, balanced_accuracy]\n  primary: balanced_accuracy\n",
        encoding="utf-8",
    )

    config = load_experiment(experiment, project_root=tmp_path)

    assert config["data"]["features"] == ["age", "plan"]
    assert config["data"]["feature_groups"] == {"numeric": ["age"], "categorical": ["plan"]}
    assert config["data"]["read_options"] == {"separator": ";"}
    assert config["data"]["schema"] == {"mode": "strict", "dtypes": {}}
    assert config["provenance"]["recipe_revision"] == "abc123"
    assert config["evaluation"]["protocol"] == "development_cv_final_test"
    assert len(config["config_fingerprint"]) == 64
