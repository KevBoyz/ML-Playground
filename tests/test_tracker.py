import json
from pathlib import Path

from ml_playground.experiments.tracker import create_run


def test_create_run_returns_id_and_dir(tmp_path):
    result = create_run({"test": 1}, base_dir=str(tmp_path))
    assert "run_id" in result
    assert "run_dir" in result
    assert Path(result["run_dir"]).exists()


def test_create_run_saves_config(tmp_path):
    config = {"model": "rf", "params": {"n": 10}}
    result = create_run(config, base_dir=str(tmp_path))
    config_path = Path(result["run_dir"]) / "config.json"
    assert config_path.exists()
    saved = json.loads(config_path.read_text(encoding="utf-8"))
    assert saved == config


def test_create_run_unique_ids(tmp_path):
    r1 = create_run({"a": 1}, base_dir=str(tmp_path))
    r2 = create_run({"a": 2}, base_dir=str(tmp_path))
    assert r1["run_id"] != r2["run_id"]
