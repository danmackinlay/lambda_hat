#!/usr/bin/env python
"""Test if heavy imports work in Parsl HTEX workers.

This minimal test helps diagnose whether matplotlib/arviz imports
are actually lethal in Parsl workers, or if the workflow failures
are due to some other cause.
"""

import logging
import sys
from pathlib import Path

import parsl
from parsl import python_app

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger(__name__)


@python_app
def test_basic_imports_app():
    """Test basic imports (should always work)."""
    from lambda_hat.commands.debug_cmd import test_basic_imports
    return test_basic_imports()


@python_app
def test_analysis_module_app():
    """Test importing analysis module (no matplotlib yet)."""
    from lambda_hat.commands.debug_cmd import test_import_analysis_only
    return test_import_analysis_only()


@python_app
def test_full_diagnostics_app():
    """Test importing matplotlib/arviz (the heavy stuff)."""
    from lambda_hat.commands.debug_cmd import test_import_diagnostics
    return test_import_diagnostics()


def run_import_tests(config_path: Path):
    """Run import tests under Parsl with specified config.

    Args:
        config_path: Path to Parsl config YAML (e.g., config/parsl/local.yaml)
    """
    # Load Parsl config
    from lambda_hat.parsl_cards import load_parsl_config_from_card

    log.info("Loading Parsl config from %s", config_path)
    parsl_config = load_parsl_config_from_card(config_path)

    try:
        # Load Parsl
        log.info("Loading Parsl...")
        dfk = parsl.load(parsl_config)
        log.info("✓ Parsl loaded successfully")

        # Test 1: Basic imports
        log.info("\n=== Test 1: Basic imports (JAX, numpy, lambda_hat) ===")
        try:
            future = test_basic_imports_app()
            result = future.result(timeout=30)
            log.info("✓ SUCCESS: %s", result)
        except Exception as e:
            log.error("✗ FAILED: %s", e)
            log.exception("Traceback:")

        # Test 2: Analysis module
        log.info("\n=== Test 2: Import analysis module ===")
        try:
            future = test_analysis_module_app()
            result = future.result(timeout=30)
            log.info("✓ SUCCESS: %s", result)
        except Exception as e:
            log.error("✗ FAILED: %s", e)
            log.exception("Traceback:")

        # Test 3: Full diagnostics (matplotlib/arviz)
        log.info("\n=== Test 3: Import matplotlib/arviz ===")
        try:
            future = test_full_diagnostics_app()
            result = future.result(timeout=30)
            log.info("✓ SUCCESS: %s", result)
        except Exception as e:
            log.error("✗ FAILED: %s", e)
            log.exception("Traceback:")

        log.info("\n=== Import Tests Complete ===")

    finally:
        log.info("Cleaning up Parsl...")
        parsl.dfk().cleanup()
        parsl.clear()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/test_parsl_imports.py <parsl-config-yaml>")
        print("Example: python scripts/test_parsl_imports.py config/parsl/local.yaml")
        sys.exit(1)

    config_path = Path(sys.argv[1])
    if not config_path.exists():
        print(f"Error: Config file not found: {config_path}")
        sys.exit(1)

    run_import_tests(config_path)
