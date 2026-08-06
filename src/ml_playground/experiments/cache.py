import hashlib
import json
from pathlib import Path
from typing import Any

import joblib

from ml_playground.preprocessing.pipelines import build_pipeline


def _cache_key(
    config: dict[str, Any],
    X: Any,
    y: Any = None,
    *,
    split_id: str | None = None,
    context: dict[str, Any] | None = None,
) -> str:
    """Build a cache identity that cannot cross data or training splits."""

    payload = {
        "config": config,
        "X": joblib.hash(X),
        "y": joblib.hash(y),
        "split_id": split_id,
        "context": context or {},
    }
    raw = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


class PipelineCache:
    """Cache fitted preprocessing only for an identical training partition.

    The cache is intentionally opt-in. Its identity includes the exact inputs,
    target and caller-provided split context, so a fitted transformer can never
    be reused from a different fold just because the YAML is the same.
    """

    def __init__(self, cache_dir="cache/preprocessing"):
        self.cache_dir = Path(cache_dir)

    def get_or_fit(self, config, X, y=None, *, split_id=None, context=None):
        key = _cache_key(config, X, y, split_id=split_id, context=context)
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
