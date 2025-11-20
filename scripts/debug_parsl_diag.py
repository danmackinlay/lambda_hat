#!/usr/bin/env python3
"""Minimal Parsl reproduction script for diagnostics hang debugging.

This script isolates the Parsl + diagnostics interaction to help diagnose
fork-after-threads deadlocks with JAX + matplotlib/arviz.

Usage:
    # With diagnostics enabled (may hang if spawn not working)
    JAX_ENABLE_X64=1 uv run python scripts/debug_parsl_diag.py

    # With diagnostics disabled (should always work)
    JAX_ENABLE_X64=1 LAMBDA_HAT_SKIP_DIAGNOSTICS=1 uv run python scripts/debug_parsl_diag.py
"""

import logging
import sys
from pathlib import Path

import parsl
from omegaconf import OmegaConf
from parsl import python_app

from lambda_hat.artifacts import Paths, RunContext
from lambda_hat.logging_config import configure_logging
from lambda_hat.parsl_cards import load_parsl_config_from_card

log = logging.getLogger(__name__)


@python_app
def debug_build_app(cfg_yaml, target_id, experiment):
    """Single build task that calls build_entry (which may run diagnostics)."""
    from lambda_hat.commands.build_cmd import build_entry

    return build_entry(cfg_yaml, target_id, experiment)


def main():
    """Run minimal Parsl test with single build task."""
    # Configure logging
    configure_logging()

    # Enable Parsl debug logging
    parsl.set_stream_logger(level=logging.DEBUG)

    log.info("=== Minimal Parsl Diagnostics Reproduction ===")

    # Initialize artifact system
    paths = Paths.from_env()
    paths.ensure()
    ctx = RunContext.create(experiment="debug_parsl", algo="debug_diag", paths=paths)

    # Load Parsl config (local.yaml with spawn mode)
    local_card = Path("config/parsl/local.yaml")
    if not local_card.exists():
        log.error("Local card not found: %s", local_card)
        sys.exit(1)

    log.info("Loading Parsl config from %s", local_card)
    parsl_cfg = load_parsl_config_from_card(local_card, [f"run_dir={ctx.parsl_dir}"])

    log.info("About to parsl.load(...)")
    parsl.load(parsl_cfg)
    log.info("parsl.load(...) returned successfully")

    # Load smoke config to get a minimal build configuration
    smoke_cfg = OmegaConf.load("config/smoke.yaml")
    experiment = smoke_cfg.get("experiment", "debug_parsl")

    # Compose a single build config (smoke has one target)
    from lambda_hat.workflow_utils import compose_build_cfg, target_id_for

    target_spec = list(smoke_cfg["targets"])[0]
    jax_x64 = bool(smoke_cfg.get("jax_enable_x64", True))
    build_cfg = compose_build_cfg(target_spec, jax_enable_x64=jax_x64)
    target_id = target_id_for(build_cfg)

    # Write temp config
    cfg_yaml_path = ctx.scratch_dir / "debug_build.yaml"
    cfg_yaml_path.parent.mkdir(parents=True, exist_ok=True)
    cfg_yaml_path.write_text(OmegaConf.to_yaml(build_cfg))

    log.info("Submitting single build task: target_id=%s", target_id)
    executor = "htex64" if jax_x64 else "htex32"

    future = debug_build_app(
        cfg_yaml=str(cfg_yaml_path),
        target_id=target_id,
        experiment=experiment,
        executor=executor,
    )

    log.info("Waiting for result...")
    try:
        result = future.result()
        log.info("✓ Build succeeded!")
        log.info("  URN: %s", result.get("urn"))
        log.info("  L0: %.6f", result.get("L0", 0.0))
    except Exception as e:
        log.error("✗ Build FAILED: %s", e)
        raise
    finally:
        log.info("Clearing Parsl...")
        parsl.clear()

    log.info("=== Test Complete ===")


if __name__ == "__main__":
    main()
