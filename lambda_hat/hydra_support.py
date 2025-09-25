# Register custom Hydra/OmegaConf resolvers at import time.
# These are used inside YAML (e.g. to compute target_id).
from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
from omegaconf import OmegaConf, DictConfig, ListConfig

_REGISTERED = False


def _to_canonical(obj):
    # Convert DictConfig/ListConfig and mixed types to JSON-serializable.

    # FIX: Check if the object is an OmegaConf container before converting.
    if isinstance(obj, (DictConfig, ListConfig)):
        return OmegaConf.to_container(obj, resolve=True)

    # If it's a primitive type (int, float, str, bool, None), return it directly.
    return obj


def _sha256_json(*parts):
    payload = json.dumps(
        [_to_canonical(p) for p in parts], sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def _git_commit_sha():
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
            )
            .decode()
            .strip()
        )
    except Exception:
        return os.environ.get("LLC_CODE_SHA", "unknown")


def register_resolvers():
    global _REGISTERED
    if _REGISTERED:
        return
    OmegaConf.register_new_resolver(
        "fingerprint", lambda *parts: "tgt_" + _sha256_json(*parts)[:12]
    )
    OmegaConf.register_new_resolver("git_sha", lambda: _git_commit_sha())
    OmegaConf.register_new_resolver("hostname", lambda: platform.node())
    _REGISTERED = True


# Import-time registration ensures resolvers are available before Hydra composes config.
register_resolvers()
