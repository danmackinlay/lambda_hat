#!/usr/bin/env python3
"""
Promote diagnostic plots from artifacts/ to assets/readme/ for README examples.

Usage:
    uv run python scripts/promote_readme_images.py                    # promote from latest run
    uv run python scripts/promote_readme_images.py artifacts/run-dir  # promote from specific run
"""

from pathlib import Path
import shutil
import sys

# Which files to copy (left = substring to match in artifacts, right = stable name in assets)
SELECT = [
    ("sgld_llc_running", "sgld_llc_running.png"),
    ("hmc_llc_running", "hmc_llc_running.png"),
    ("mclmc_llc_running", "mclmc_llc_running.png"),
    ("hmc_acceptance", "hmc_acceptance.png"),
    ("hmc_L_trace", "hmc_L_trace.png"),
    ("hmc_L_acf", "hmc_L_acf.png"),
    ("mclmc_energy_hist", "mclmc_energy_hist.png"),
]


def latest_run_dir(artifacts_dir: Path) -> Path:
    """Find the most recent timestamped run directory."""
    runs = [p for p in artifacts_dir.glob("*-*") if p.is_dir()]
    if not runs:
        raise SystemExit("No runs found under artifacts/")
    return sorted(runs)[-1]


def find_first_match(run_dir: Path, key: str) -> Path | None:
    """Find the first PNG file containing the key substring."""
    for p in sorted(run_dir.glob("*.png")):
        if key in p.name:
            return p
    return None


def main():
    root = Path(__file__).resolve().parents[1]
    artifacts = root / "artifacts"
    assets = root / "assets" / "readme"
    assets.mkdir(parents=True, exist_ok=True)

    # Allow overriding the source run dir on the CLI
    if len(sys.argv) > 1:
        run_dir = Path(sys.argv[1]).resolve()
    else:
        run_dir = latest_run_dir(artifacts)

    if not run_dir.exists():
        raise SystemExit(f"Run dir not found: {run_dir}")

    print(f"Promoting images from: {run_dir}")
    copied = 0

    for key, outname in SELECT:
        src = find_first_match(run_dir, key)
        if not src:
            print(f"  [skip] no match for '{key}'")
            continue

        dst = assets / outname
        shutil.copy2(src, dst)
        print(f"  copied {src.name} -> {dst.relative_to(root)}")
        copied += 1

    if copied == 0:
        print("No images copied. Did this run save plots? (save_plots=True)")
    else:
        print(f"Done. Copied {copied} images.")
        print(
            "Commit updated assets/readme/*.png and refresh README references if needed."
        )


if __name__ == "__main__":
    main()
