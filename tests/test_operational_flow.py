from pathlib import Path

import pandas as pd
import polars as pl

from ml_playground.experiments.runner import build_run_plan, run_grid


def _supervised_config(development: Path, final_test: Path, reports_root: Path | None = None) -> dict:
    outputs = {
        "save_model": False,
        "save_predictions": True,
        "figures": False,
        "predictions_format": "csv",
    }
    if reports_root is not None:
        outputs["root"] = str(reports_root)
    return {
        "experiment_name": "external_test_contract" if reports_root is not None else None,
        "project_root": str(development.parent),
        "task": "classification",
        "data": {
            "path": str(development),
            "test": {"path": str(final_test)},
            "target": "target",
            "id_column": "record_id",
            "features": ["signal"],
            "random_state": 42,
        },
        "evaluation": {
            "protocol": "development",
            "evaluate_final_test": True,
            "splitter": {"method": "stratified_kfold", "n_splits": 2, "shuffle": True, "random_state": 42},
        },
        "cross_validation": {"method": "stratified_kfold", "n_splits": 2, "shuffle": True, "random_state": 42},
        "models": [
            {"name": "dummy_classifier", "params": {"strategy": ["prior"]}},
            {"name": "decision_tree", "params": {"max_depth": [1], "random_state": [42]}},
        ],
        "preprocessing": {},
        "metrics": ["accuracy", "f1_macro"],
        "selection": {"primary_metric": "accuracy", "direction": "maximize"},
        "outputs": outputs,
        "search": {"strategy": "grid", "max_candidates": 2},
    }


def _write_sources(tmp_path: Path, *, invert_final_target: bool = False) -> tuple[Path, Path]:
    development = tmp_path / "development.csv"
    final_test = tmp_path / "final.csv"
    pl.DataFrame(
        {
            "record_id": list(range(12)),
            "signal": [0, 1, 2, 3, 4, 5, 10, 11, 12, 13, 14, 15],
            "target": [0] * 6 + [1] * 6,
        }
    ).write_csv(development)
    targets = [0, 0, 0, 1, 1, 1]
    if invert_final_target:
        targets = [1 - value for value in targets]
    pl.DataFrame(
        {
            "record_id": list(range(100, 106)),
            "signal": [0, 2, 4, 11, 13, 15],
            "target": targets,
        }
    ).write_csv(final_test)
    return development, final_test


def test_external_final_test_is_evaluated_only_after_candidate_selection(tmp_path):
    development, final_test = _write_sources(tmp_path)
    normal = run_grid(_supervised_config(development, final_test), write_reports=False)

    _, inverted_final = _write_sources(tmp_path, invert_final_target=True)
    inverted = run_grid(_supervised_config(development, inverted_final), write_reports=False)

    assert normal["best"]["name"] == inverted["best"]["name"] == "decision_tree"
    assert "final_test" in normal["best"]
    assert "final_test" in inverted["best"]
    assert normal["best"]["final_test"]["metrics"]["accuracy"] != inverted["best"]["final_test"]["metrics"]["accuracy"]
    assert sum("final_test" in result for result in normal["results"]) == 1


def test_run_plan_bounds_grid_and_report_separates_final_test_metrics(tmp_path):
    development, final_test = _write_sources(tmp_path)
    config = _supervised_config(development, final_test, tmp_path / "reports")
    config["search"] = {"strategy": "random", "random_state": 7, "max_candidates": 1}

    plan = build_run_plan(config)
    result = run_grid(config, write_reports=True)

    assert plan["search"]["candidate_count"] == 1
    assert plan["estimated_fits"]["development_total"] == 2
    summary = pd.read_csv(result["reports"]["summary"])
    predictions = pd.read_csv(result["reports"]["predictions"])
    assert "final_test_metric_accuracy" in summary.columns
    assert set(predictions["split_role"]) == {"final_test"}
