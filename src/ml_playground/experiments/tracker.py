import hashlib
import json
import platform
import subprocess
import uuid
from datetime import datetime, timezone
from pathlib import Path


RUN_CONTRACT_VERSION = "2.0"


def build_run_context(config: dict, *, created_at: str | None = None) -> dict:
    """Normalize the optional context that makes an experiment reproducible.

    The runner deliberately does not invent domain provenance.  It does, however,
    preserve context supplied by callers and add stable bookkeeping that can be
    checked against a report or a persisted model later.
    """

    supplied = config.get("run_context") or {}
    if not isinstance(supplied, dict):
        raise TypeError("run_context deve ser um mapa quando informado")

    # Runtime frames can be attached by the executor.  They are neither stable
    # nor useful as an experiment identity, so do not turn their repr into a
    # misleading configuration hash.
    hashable_config = {
        key: value
        for key, value in config.items()
        if not key.startswith("_") and key not in {"run_context", "preflight_metadata"}
    }
    serialized_config = json.dumps(
        hashable_config,
        sort_keys=True,
        default=str,
        ensure_ascii=False,
    )
    context = dict(supplied)
    context.setdefault("contract_version", RUN_CONTRACT_VERSION)
    context.setdefault(
        "created_at",
        created_at or datetime.now(timezone.utc).isoformat(),
    )
    context.setdefault(
        "config_sha256",
        hashlib.sha256(serialized_config.encode("utf-8")).hexdigest(),
    )

    for key in (
        "project_root",
        "experiment_dir",
        "config_files",
        "config_fingerprint",
        "run_fingerprint",
    ):
        if key in config:
            context.setdefault(key, config[key])

    provenance = config.get("provenance")
    if provenance is not None:
        context.setdefault("provenance", provenance)

    preflight = config.get("preflight_metadata")
    if isinstance(preflight, dict):
        context.setdefault("preflight", preflight)
        for key in ("config_fingerprint", "run_fingerprint", "contract_version"):
            if key in preflight:
                context.setdefault(
                    "data_contract_version" if key == "contract_version" else key,
                    preflight[key],
                )
        development = (preflight.get("sources") or {}).get("development")
        if isinstance(development, dict):
            fingerprint = development.get("fingerprint") or {}
            if isinstance(fingerprint, dict) and fingerprint.get("digest"):
                context.setdefault("data_fingerprint", fingerprint["digest"])
                context.setdefault("dataset_sha256", fingerprint["digest"])
            if development.get("schema_signature"):
                context.setdefault("schema_signature", development["schema_signature"])
    context.setdefault("python_version", platform.python_version())
    context.setdefault("platform", platform.platform())
    project_root = Path(config.get("project_root", Path.cwd()))
    lock_path = project_root / "uv.lock"
    if lock_path.is_file():
        context.setdefault("dependency_lock_sha256", _sha256_file(lock_path))
    git_context = _git_context(project_root)
    for key, value in git_context.items():
        context.setdefault(key, value)
    return context


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_context(project_root: Path) -> dict[str, str | bool]:
    """Collect optional Git evidence without making Git a runtime dependency."""

    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=project_root,
            text=True,
            capture_output=True,
            check=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=project_root,
            text=True,
            capture_output=True,
            check=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return {}
    return {"git_sha": revision, "git_dirty": bool(status.strip())}


def create_run(config: dict, base_dir: str = "runs", *, persist: bool = True) -> dict:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"{timestamp}_{uuid.uuid4().hex[:6]}"
    run_dir = Path(base_dir) / run_id
    run_context = build_run_context(config)
    run_context.setdefault("run_id", run_id)
    if persist:
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "config.json").write_text(
            json.dumps(config, indent=2, default=str, ensure_ascii=False),
            encoding="utf-8",
        )
        (run_dir / "run_context.json").write_text(
            json.dumps(run_context, indent=2, default=str, ensure_ascii=False),
            encoding="utf-8",
        )
    return {
        "run_id": run_id,
        "run_dir": str(run_dir) if persist else None,
        "contract_version": RUN_CONTRACT_VERSION,
        "run_context": run_context,
    }
