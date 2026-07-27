import json
import uuid
from datetime import datetime
from pathlib import Path


def create_run(config: dict, base_dir: str = "runs") -> dict:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"{timestamp}_{uuid.uuid4().hex[:6]}"
    run_dir = Path(base_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(
        json.dumps(config, indent=2, default=str, ensure_ascii=False),
        encoding="utf-8",
    )
    return {"run_id": run_id, "run_dir": str(run_dir)}
