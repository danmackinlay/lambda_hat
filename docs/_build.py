#!/usr/bin/env python3
"""Build all generated documentation."""

import subprocess
import sys
from pathlib import Path


def run_generator(script_name):
    """Run a doc generator script."""
    script = Path(__file__).parent / script_name
    print(f"Running {script_name}...")
    result = subprocess.run([sys.executable, str(script)], capture_output=True, text=True)

    if result.returncode != 0:
        print(f"ERROR: {script_name} failed")
        print(result.stderr)
        return False

    print(result.stdout)
    return True


def main():
    """Generate all docs and run checks."""
    generators = ["_gen_cli.py", "_gen_config.py"]

    print("=" * 60)
    print("Building documentation")
    print("=" * 60)

    for gen in generators:
        if not run_generator(gen):
            sys.exit(1)

    print("\n" + "=" * 60)
    print("Running drift checks")
    print("=" * 60)

    if not run_generator("_checks.py"):
        sys.exit(1)

    print("\n✅ Documentation build complete")


if __name__ == "__main__":
    main()
