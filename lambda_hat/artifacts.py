# lambda_hat/artifacts.py
"""
Artifact storage system for lambda_hat.

Three-layer model:
- Store: Content-addressed immutable objects (SHA256-based)
- Experiments: Grouping, manifests, TensorBoard aggregation
- Scratch: Ephemeral working space (safe to delete)
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import socket
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

SCHEMA_VERSION = "1"


def _now_utc_iso() -> str:
    """Return current UTC time in ISO format."""
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _short_id(n: int = 6) -> str:
    """Generate a short random ID."""
    return uuid.uuid4().hex[:n]


def write_json(p: Path, obj: Dict[str, Any]) -> None:
    """Write JSON atomically using tmp file + rename."""
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    with tmp.open("w") as f:
        json.dump(obj, f, indent=2, sort_keys=True)
    tmp.replace(p)


def atomic_append_jsonl(p: Path, obj: Dict[str, Any]) -> None:
    """Append a single JSON line atomically."""
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    line = json.dumps(obj, sort_keys=True) + "\n"
    with tmp.open("w") as f:
        f.write(line)
    # Append atomically where possible; fall back to rename
    with open(p, "a") as dest:
        with tmp.open("r") as src:
            shutil.copyfileobj(src, dest)
    tmp.unlink(missing_ok=True)


def dir_sha256(path: Path) -> str:
    """
    Stable hash over relative paths + content.
    Ignores meta.json if present.
    """
    h = hashlib.sha256()
    base = path
    for root, dirs, files in os.walk(base):
        dirs.sort()
        files.sort()
        for fname in files:
            if fname == "meta.json":
                continue
            fp = Path(root) / fname
            rel = fp.relative_to(base).as_posix().encode()
            h.update(b"FILENAME:")
            h.update(rel)
            h.update(b"\0")
            with open(fp, "rb") as f:
                while True:
                    chunk = f.read(1024 * 1024)
                    if not chunk:
                        break
                    h.update(chunk)
    return h.hexdigest()


@dataclass
class Paths:
    """Root paths for artifact storage."""

    home: Path
    store: Path
    experiments: Path
    scratch: Path

    @staticmethod
    def from_env() -> "Paths":
        """Create Paths from environment variables."""
        home = Path(os.environ.get("LAMBDA_HAT_HOME", Path.cwd() / ".lambda_hat")).resolve()
        return Paths(
            home=home,
            store=Path(os.environ.get("LAMBDA_HAT_STORE", home / "store")).resolve(),
            experiments=Path(
                os.environ.get("LAMBDA_HAT_EXPERIMENTS", home / "experiments")
            ).resolve(),
            scratch=Path(os.environ.get("LAMBDA_HAT_SCRATCH", home / "scratch")).resolve(),
        )

    def ensure(self) -> None:
        """Create all root directories."""
        for p in [self.store, self.experiments, self.scratch]:
            p.mkdir(parents=True, exist_ok=True)


class ArtifactStore:
    """Content-addressed artifact storage using SHA256."""

    def __init__(self, root: Path):
        self.root = root
        (self.root / "objects" / "sha256").mkdir(parents=True, exist_ok=True)

    def _dest_for_hash(self, h: str) -> Path:
        """Return storage path for a given hash."""
        return self.root / "objects" / "sha256" / h[:2] / h[2:4] / h

    def put_dir(self, src_dir: Path, a_type: str, meta: Dict[str, Any]) -> str:
        """
        Copy a directory to store, content-addressed.
        Returns URN: urn:lh:<type>:sha256:<hash>
        """
        tmp = self.root / "tmp" / uuid.uuid4().hex
        if tmp.exists():
            shutil.rmtree(tmp)
        shutil.copytree(src_dir, tmp)
        h = dir_sha256(tmp)
        dest = self._dest_for_hash(h)
        payload = dest / "payload"
        meta_p = dest / "meta.json"

        if dest.exists():
            shutil.rmtree(tmp)
        else:
            payload.parent.mkdir(parents=True, exist_ok=True)
            tmp.rename(payload)
        # Write/ensure meta.json (first write wins)
        if not meta_p.exists():
            meta = {
                "schema": SCHEMA_VERSION,
                "type": a_type,
                "hash": {"algo": "sha256", "hex": h},
                "created": _now_utc_iso(),
                **meta,
            }
            write_json(meta_p, meta)
        urn = f"urn:lh:{a_type}:sha256:{h}"
        return urn

    def put_file(self, src_file: Path, a_type: str, meta: Dict[str, Any]) -> str:
        """
        Store a single file as an artifact.
        Returns URN.
        """
        tmp_dir = self.root / "tmp" / uuid.uuid4().hex
        payload = tmp_dir / "payload"
        payload.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_file, payload)
        return self.put_dir(tmp_dir, a_type, meta)

    def path_for(self, urn: str) -> Path:
        """
        Resolve URN to filesystem path.
        URN format: urn:lh:<type>:sha256:<hash>
        """
        parts = urn.split(":")
        assert (
            len(parts) >= 5 and parts[0] == "urn" and parts[1] == "lh" and parts[3] == "sha256"
        ), f"Invalid URN format: {urn}"
        h = parts[4]
        return self._dest_for_hash(h)


def safe_symlink(src: Path, dest: Path) -> None:
    """Create symlink, replacing existing if necessary."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        if dest.exists() or dest.is_symlink():
            dest.unlink()
    except FileNotFoundError:
        pass
    os.symlink(src, dest, target_is_directory=src.is_dir())


@dataclass
class RunContext:
    """Per-run directory context."""

    experiment: str
    algo: str
    run_id: str
    run_dir: Path
    tb_dir: Path
    logs_dir: Path
    parsl_dir: Path
    artifacts_dir: Path
    inputs_dir: Path
    scratch_dir: Path

    @staticmethod
    def create(
        experiment: Optional[str],
        algo: str,
        paths: Paths,
        tag: Optional[str] = None,
    ) -> "RunContext":
        """Create a new run context with computed paths."""
        experiment = experiment or os.environ.get("LAMBDA_HAT_DEFAULT_EXPERIMENT", "dev")
        ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        rid = f"{ts}-{algo}{('-' + tag) if tag else ''}-{_short_id()}"
        base = paths.experiments / experiment / "runs" / rid
        ctx = RunContext(
            experiment=experiment,
            algo=algo,
            run_id=rid,
            run_dir=base,
            tb_dir=base / "tb",
            logs_dir=base / "logs",
            parsl_dir=base / "parsl",
            artifacts_dir=base / "artifacts",
            inputs_dir=base / "inputs",
            scratch_dir=base / "scratch",
        )
        for d in [
            ctx.run_dir,
            ctx.tb_dir,
            ctx.logs_dir,
            ctx.parsl_dir,
            ctx.artifacts_dir,
            ctx.inputs_dir,
            ctx.scratch_dir,
        ]:
            d.mkdir(parents=True, exist_ok=True)
        # Keep a friendly symlink for TB aggregation
        tb_agg = paths.experiments / experiment / "tb" / rid
        safe_symlink(ctx.tb_dir, tb_agg)
        return ctx

    def write_run_manifest(self, payload: Dict[str, Any]) -> None:
        """Write run manifest to both run dir and experiment index."""
        payload = {
            "schema": SCHEMA_VERSION,
            "run_id": self.run_id,
            "experiment": self.experiment,
            "algo": self.algo,
            "host": socket.gethostname(),
            "created": _now_utc_iso(),
            **payload,
        }
        write_json(self.run_dir / "manifest.json", payload)
        # Also append to experiment-level index
        atomic_append_jsonl(self.run_dir.parent.parent / "manifest.jsonl", payload)
