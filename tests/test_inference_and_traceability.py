import json
from pathlib import Path

import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ml_playground.experiments.report import REPORT_CONTRACT_VERSION, write_experiment_reports
from ml_playground.experiments.tracker import create_run
from ml_playground.models.inference import SchemaValidationError, predict_batch
from ml_playground.models.serialization import ARTIFACT_CONTRACT_VERSION, load_model, save_model


@pytest.fixture
def fitted_artifact(tmp_path: Path):
    features = pd.DataFrame(
        {
            "age": [22.0, 25.0, 35.0, 42.0, 51.0, 58.0],
            "income": [30.0, 34.0, 55.0, 70.0, 90.0, 110.0],
        }
    )
    target = [0, 0, 0, 1, 1, 1]
    pipeline = Pipeline(
        [("scaler", StandardScaler()), ("model", LogisticRegression(random_state=42))]
    ).fit(features, target)
    path = save_model(
        pipeline,
        tmp_path / "classifier.joblib",
        {
            "task": "classification",
            "signature": {
                "features": ["age", "income"],
                "dtypes": {"age": "float64", "income": "float64"},
                "target": "outcome",
                "id_column": "customer_id",
                "id_unique": True,
            },
            "provenance": {"dataset_sha256": "dataset-checksum"},
        },
    )
    return Path(path), pipeline


def test_save_model_adds_contract_without_losing_signature(fitted_artifact):
    path, _ = fitted_artifact

    _, metadata = load_model(path)

    assert metadata["artifact_contract_version"] == ARTIFACT_CONTRACT_VERSION
    assert metadata["signature"]["features"] == ["age", "income"]
    assert metadata["provenance"]["dataset_sha256"] == "dataset-checksum"
    assert metadata["model_class"].endswith("Pipeline")


def test_predict_batch_validates_signature_preserves_ids_and_writes_parquet(fitted_artifact, tmp_path):
    path, _ = fitted_artifact
    batch = pd.DataFrame(
        {
            "customer_id": [1001, 1002],
            "age": [28.0, 55.0],
            "income": [40.0, 100.0],
            "outcome": [0, 1],
        }
    )

    result = predict_batch(
        path,
        batch,
        output_path=tmp_path / "scores",
        output_format="parquet",
    )

    assert result.validation["id_column"] == "customer_id"
    assert result.validation["generated_id"] is False
    assert result.frame["customer_id"].tolist() == [1001, 1002]
    assert {"prediction", "probability_0", "probability_1"}.issubset(result.frame.columns)
    assert Path(result.output_path).is_file()
    assert len(pd.read_parquet(result.output_path)) == 2


def test_predict_batch_rejects_missing_feature_invalid_dtype_and_duplicate_id(fitted_artifact):
    path, _ = fitted_artifact

    with pytest.raises(SchemaValidationError, match="Features ausentes"):
        predict_batch(path, pd.DataFrame({"customer_id": [1], "age": [30.0]}))

    with pytest.raises(SchemaValidationError, match="Tipos incompatíveis"):
        predict_batch(
            path,
            pd.DataFrame({"customer_id": [1], "age": ["old"], "income": [50.0]}),
        )

    with pytest.raises(SchemaValidationError, match="deve ser única"):
        predict_batch(
            path,
            pd.DataFrame(
                {
                    "customer_id": [1, 1],
                    "age": [30.0, 31.0],
                    "income": [50.0, 51.0],
                }
            ),
        )


def test_predict_batch_uses_sklearn_feature_names_for_legacy_metadata(fitted_artifact, tmp_path):
    _, pipeline = fitted_artifact
    legacy_path = save_model(pipeline, tmp_path / "legacy.joblib")

    result = predict_batch(legacy_path, pd.DataFrame({"age": [30.0], "income": [50.0]}))

    assert result.validation["contract_source"] == "sklearn"
    assert result.validation["generated_id"] is True
    assert result.frame["row_id"].tolist() == [0]


def test_report_inventory_is_relative_checksummed_and_links_trials(tmp_path):
    model_path = tmp_path / "model.joblib"
    model_path.write_bytes(b"model artifact")
    config = {
        "experiment_name": "traceable_report",
        "task": "classification",
        "data": {"path": str(tmp_path / "dataset.csv"), "target": "target", "features": ["x"]},
        "cross_validation": {"method": "holdout", "test_size": 0.2},
        "selection": {"primary_metric": "accuracy", "direction": "maximize"},
        "outputs": {
            "root": str(tmp_path / "reports"),
            "save_predictions": True,
            "figures": False,
            "predictions_format": "both",
        },
        "views": {},
        "run_context": {"git_sha": "abc123", "dataset_sha256": "data-checksum"},
    }
    results = [
        {
            "name": "logistic",
            "params": {"C": 0.1},
            "status": "success",
            "metrics": {"accuracy": 0.75},
            "predictions": {
                "row_ids": [10, 11],
                "y_true": [0, 1],
                "y_pred": [0, 1],
                "y_score": [[0.8, 0.2], [0.3, 0.7]],
            },
        },
        {
            "name": "logistic",
            "params": {"C": 1.0},
            "status": "success",
            "metrics": {"accuracy": 1.0},
            "predictions": {
                "row_ids": [10, 11],
                "y_true": [0, 1],
                "y_pred": [0, 1],
                "y_score": [[0.9, 0.1], [0.1, 0.9]],
            },
        },
    ]

    artifacts = write_experiment_reports(
        config,
        {"results": results},
        "run_trace",
        best_result=results[1],
        model_path=str(model_path),
    )

    summary = pd.read_csv(artifacts["summary"])
    predictions = pd.read_csv(artifacts["predictions_csv"])
    manifest = json.loads(Path(artifacts["manifest"]).read_text(encoding="utf-8"))
    card = Path(artifacts["model_card"]).read_text(encoding="utf-8")

    assert summary["candidate_id"].nunique() == 2
    assert {"candidate_id", "trial_id", "split_id"}.issubset(predictions.columns)
    assert Path(artifacts["predictions_parquet"]).is_file()
    assert manifest["contract_version"] == REPORT_CONTRACT_VERSION
    assert manifest["paths"]["artifacts"]["summary"].startswith("metrics/")
    assert len(manifest["artifact_inventory"]["summary"]["sha256"]) == 64
    assert manifest["model_artifact"]["path"].startswith("../")
    assert "Candidate ID" in card
    assert "abc123" in card


def test_tracker_persists_optional_run_context_without_changing_config(tmp_path):
    config = {"model": "ridge", "run_context": {"git_sha": "deadbeef"}}

    run = create_run(config, base_dir=tmp_path)

    saved_config = json.loads((Path(run["run_dir"]) / "config.json").read_text(encoding="utf-8"))
    context = json.loads((Path(run["run_dir"]) / "run_context.json").read_text(encoding="utf-8"))
    assert saved_config == config
    assert context["git_sha"] == "deadbeef"
    assert context["run_id"] == run["run_id"]
    assert len(context["config_sha256"]) == 64


def test_tracker_promotes_preflight_identity_to_run_context(tmp_path):
    config = {
        "experiment_name": "traceable",
        "config_fingerprint": "config-fingerprint",
        "preflight_metadata": {
            "contract_version": 1,
            "run_fingerprint": "run-fingerprint",
            "sources": {
                "development": {
                    "fingerprint": {"algorithm": "sha256", "digest": "data-fingerprint"},
                    "schema_signature": "schema-fingerprint",
                }
            },
        },
    }

    run = create_run(config, base_dir=tmp_path)

    context = run["run_context"]
    assert context["config_fingerprint"] == "config-fingerprint"
    assert context["run_fingerprint"] == "run-fingerprint"
    assert context["data_fingerprint"] == "data-fingerprint"
    assert context["dataset_sha256"] == "data-fingerprint"
    assert context["schema_signature"] == "schema-fingerprint"
    assert context["data_contract_version"] == 1
