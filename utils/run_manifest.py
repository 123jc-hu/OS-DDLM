"""Small provenance records for completed experiment runs."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any, Mapping

import torch


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_revision() -> tuple[str | None, bool | None]:
    root = Path(__file__).resolve().parents[1]
    try:
        revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True).strip())
        return revision, dirty
    except (OSError, subprocess.CalledProcessError):
        return None, None


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def create_manifest(config: Mapping[str, Any], experiment_dir: Path) -> Path:
    revision, dirty = _git_revision()
    manifest_dir = experiment_dir / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = manifest_dir / f"run_{_timestamp()}.json"
    payload = {
        "status": "running",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": dict(config),
        "config_sha256": hashlib.sha256(json.dumps(dict(config), sort_keys=True, default=str).encode()).hexdigest(),
        "seed": config.get("random_seed", config.get("seed")),
        "git_revision": revision,
        "git_dirty": dirty,
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
        },
        "stdout_path": config.get("stdout_path"),
        "stderr_path": config.get("stderr_path"),
        "log_path": config.get("log_path"),
        "data_split": None,
        "outputs": {},
    }
    manifest_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return manifest_path


def finalize_manifest(
    manifest_path: Path,
    experiment_dir: Path,
    provenance: Mapping[str, Any],
    status: str,
    error: str | None = None,
) -> None:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["status"] = status
    payload["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    payload["data_split"] = provenance.get("data_split")
    payload["subjects"] = provenance.get("subjects", [])
    if error is not None:
        payload["error"] = error

    outputs = (
        list(experiment_dir.glob("results*.csv"))
        + list(experiment_dir.glob("selected_channels*.csv"))
        + [experiment_dir / "runtime_metrics.csv"]
    )
    for output in outputs:
        if output.exists():
            payload["outputs"][output.name] = {"sha256": file_sha256(output), "bytes": output.stat().st_size}
    manifest_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
