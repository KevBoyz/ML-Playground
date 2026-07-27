import joblib
from datetime import datetime
from pathlib import Path


def save_model(model, path, metadata=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    artifact = {
        "model": model,
        "metadata": {
            "saved_at": datetime.now().isoformat(),
            **(metadata or {}),
        },
    }
    joblib.dump(artifact, path)
    return str(path)


def load_model(path):
    artifact = joblib.load(path)
    return artifact["model"], artifact.get("metadata")
