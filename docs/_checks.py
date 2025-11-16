#!/usr/bin/env python3
"""Documentation drift checks - fails CI if docs are out of sync."""

import sys
import tomllib
from pathlib import Path
import re


def check_blackjax_pin():
    """Verify BlackJAX pin matches between pyproject.toml and docs."""
    # Read pin from pyproject.toml
    pyproject = Path("pyproject.toml")
    with pyproject.open("rb") as f:
        data = tomllib.load(f)

    deps = data.get("project", {}).get("dependencies", [])
    blackjax_dep = None
    for dep in deps:
        if dep.startswith("blackjax"):
            blackjax_dep = dep
            break

    if not blackjax_dep:
        print("ERROR: BlackJAX not found in pyproject.toml dependencies")
        return False

    # Extract version (e.g., "blackjax==1.2.5" -> "1.2.5")
    match = re.search(r"blackjax\s*==\s*([0-9.]+)", blackjax_dep)
    if not match:
        print(f"ERROR: Cannot parse BlackJAX version from: {blackjax_dep}")
        return False

    pinned_version = match.group(1)
    print(f"✓ pyproject.toml pins blackjax=={pinned_version}")

    # Check if compatibility.md exists and mentions this version
    compat_doc = Path("docs/compatibility.md")
    if compat_doc.exists():
        content = compat_doc.read_text()
        if pinned_version not in content:
            print(f"ERROR: docs/compatibility.md does not mention blackjax=={pinned_version}")
            print("       Run: uv run python docs/_build.py")
            return False
        print(f"✓ docs/compatibility.md mentions blackjax=={pinned_version}")

    return True


def check_stale_paths():
    """Grep docs for forbidden stale paths."""
    forbidden = [
        "lambda_hat/sampling.py",  # Should reference CLI or sampling_runner.py
    ]

    docs_dir = Path("docs")
    errors = []

    for pattern in forbidden:
        for md_file in docs_dir.glob("*.md"):
            # Skip generated files
            if md_file.name in ["cli.md", "config.md"]:
                continue

            content = md_file.read_text()
            if pattern in content:
                errors.append(f"{md_file.name} references stale path: {pattern}")

    if errors:
        print("ERROR: Found stale path references:")
        for err in errors:
            print(f"  - {err}")
        return False

    print("✓ No stale path references found")
    return True


def main():
    """Run all checks."""
    checks = [
        ("BlackJAX pin consistency", check_blackjax_pin),
        ("Stale path references", check_stale_paths),
    ]

    failed = []
    for name, check_fn in checks:
        print(f"\nChecking: {name}")
        if not check_fn():
            failed.append(name)

    if failed:
        print(f"\n❌ {len(failed)} check(s) failed:")
        for name in failed:
            print(f"  - {name}")
        sys.exit(1)

    print("\n✅ All documentation checks passed")


if __name__ == "__main__":
    main()
