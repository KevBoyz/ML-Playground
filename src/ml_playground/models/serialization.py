"""Persistence for fitted estimators and their inference contract."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import joblib


ARTIFACT_CONTRACT_VERSION = "2.0"


def _normalized_metadata(model, metadata):
    """Keep caller metadata intact while adding a stable artifact envelope."""

    normalized = dict(metadata or {})
    normalized.setdefault("artifact_contract_version", ARTIFACT_CONTRACT_VERSION)
    normalized.setdefault("saved_at", datetime.now(timezone.utc).isoformat())
    normalized.setdefault(
        "model_class",
        f"{type(model).__module__}.{type(model).__qualname__}",
    )

    # Older callers often supplied the pieces at the top level.  Preserve that
    # shape, but make a single signature available to safe inference consumers.
    if not isinstance(normalized.get("signature"), dict):
        schema = normalized.get("schema") or normalized.get("data_contract")
        features = normalized.get("features")
        if isinstance(schema, dict):
            features = features or schema.get("features")
        if features is not None:
            signature = {"features": features}
            for source in (schema, normalized):
                if not isinstance(source, dict):
                    continue
                for key in (
                    "dtypes",
                    "target",
                    "id_column",
                    "id_unique",
                    "allow_extra_columns",
                ):
                    if key in source:
                        signature.setdefault(key, source[key])
            normalized["signature"] = signature

    return normalized


def save_model(model, path, metadata=None):
    """Persist a fitted model with optional schema, signature and provenance.

    ``metadata`` remains deliberately open-ended.  Callers may include
    ``schema``, ``signature`` and ``provenance`` dictionaries; they are stored
    verbatim and supplemented only with artifact bookkeeping.
    """

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    artifact = {
        "model": model,
        "metadata": _normalized_metadata(model, metadata),
    }
    joblib.dump(artifact, path)
    return str(path)


def load_model(path):
    artifact = joblib.load(path)
    return artifact["model"], artifact.get("metadata")
