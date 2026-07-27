import hashlib
import json
from pathlib import Path

import joblib

from ml_playground.preprocessing.pipelines import build_pipeline


def _hash_config(config):
    raw = json.dumps(config, sort_keys=True, default=str)
    return hashlib.md5(raw.encode()).hexdigest()


class PipelineCache:
    def __init__(self, cache_dir="cache/preprocessing"):
        self.cache_dir = Path(cache_dir)

    def get_or_fit(self, config, X, y=None):
        key = _hash_config(config)
        path = self.cache_dir / f"{key}.joblib"
        if path.exists():
            return joblib.load(path)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        pipe = build_pipeline(config)
        pipe.fit(X, y)
        joblib.dump(pipe, path)
        return pipe

    def clear(self):
        if self.cache_dir.exists():
            for f in self.cache_dir.iterdir():
                f.unlink()
