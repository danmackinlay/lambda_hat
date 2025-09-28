# Register custom OmegaConf resolvers at import time.
# These are used inside YAML for metadata purposes.
from __future__ import annotations

import os
import platform
import subprocess

from omegaconf import OmegaConf

_REGISTERED = False


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
    OmegaConf.register_new_resolver("git_sha", lambda: _git_commit_sha())
    OmegaConf.register_new_resolver("hostname", lambda: platform.node())
    _REGISTERED = True


# Import-time registration ensures resolvers are available.
register_resolvers()
