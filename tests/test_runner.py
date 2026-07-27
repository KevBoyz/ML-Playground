import polars as pl


def test_run_grid(tmp_path):
    from ml_playground.experiments.runner import run_grid

    df = pl.DataFrame(
        {
            "feat1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "target": [0, 0, 0, 1, 1, 1],
        }
    )
    csv_path = tmp_path / "data.csv"
    df.write_csv(csv_path)

    base_config = {
        "data": {
            "path": str(csv_path),
            "target": "target",
            "test_size": 0.3,
            "random_state": 42,
        },
        "preprocessing": {"scaling": {"method": "standard"}},
        "metrics": ["accuracy"],
    }
    models_config = [
        {"name": "logistic", "params": {"C": [0.1, 1.0], "random_state": 42}},
    ]
    result = run_grid(base_config, models_config)
    assert "results" in result
    assert len(result["results"]) == 2
    assert all(r["name"] == "logistic" for r in result["results"])
    assert all(
        "accuracy" in r.get("metrics", {}) or "error" in r for r in result["results"]
    )


def test_run_grid_catches_errors(tmp_path):
    from ml_playground.experiments.runner import run_grid

    base_config = {
        "data": {"path": str(tmp_path / "nonexistent.csv"), "target": "target"},
        "metrics": ["accuracy"],
    }
    models_config = [{"name": "logistic"}]
    result = run_grid(base_config, models_config)
    assert "error" in result["results"][0]
