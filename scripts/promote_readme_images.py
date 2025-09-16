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
import re
from datetime import datetime

# Which files to copy (left = substring to match in artifacts, right = stable name in assets)
SELECT = [
    ("sgld_running_llc", "sgld_llc_running.png"),
    ("hmc_running_llc", "hmc_llc_running.png"),
    ("mclmc_running_llc", "mclmc_llc_running.png"),
    ("hmc_acceptance", "hmc_acceptance.png"),
    ("hmc_Ln_trace", "hmc_L_trace.png"),
    ("hmc_Ln_acf", "hmc_L_acf.png"),
    ("mclmc_energy_hist", "mclmc_energy_hist.png"),
]


def _has_needed_artifacts(p: Path) -> bool:
    """Valid if it contains metrics.json OR at least one of the expected PNGs."""
    if (p / "metrics.json").exists():
        return True
    pngs = [q.name for q in p.glob("*.png")]
    return any(key in name for key, _ in SELECT for name in pngs)


def latest_run_dir(artifacts_dir: Path) -> Path:
    """Pick the newest run (hash or timestamp dir, symlinks ok) that actually has artifacts."""
    if not artifacts_dir.exists():
        raise SystemExit(f"Artifacts directory not found: {artifacts_dir}")

    candidates: list[tuple[float, Path]] = []
    for p in artifacts_dir.iterdir():
        if not p.is_dir():
            continue
        # Follow symlink where possible
        try:
            q = p.resolve()
        except Exception:
            q = p
        if not _has_needed_artifacts(q):
            continue  # skip empty/aborted dirs

        # Prefer parsed timestamp when name is YYYYMMDD-HHMMSS; else use newest file mtime
        ts = None
        if re.fullmatch(r"\d{8}-\d{6}", p.name):
            try:
                ts = datetime.strptime(p.name, "%Y%m%d-%H%M%S").timestamp()
            except ValueError:
                ts = None
        if ts is None:
            mtimes = [f.stat().st_mtime for f in q.glob("*.png")]
            if (q / "metrics.json").exists():
                mtimes.append((q / "metrics.json").stat().st_mtime)
            ts = max(mtimes) if mtimes else q.stat().st_mtime
        candidates.append((ts, p))

    if not candidates:
        raise SystemExit("No runs with artifacts found under artifacts/")
    candidates.sort(key=lambda t: t[0])
    return candidates[-1][1]


def find_first_match(run_dir: Path, key: str) -> Path | None:
    """Find the first PNG file containing the key substring (follow symlinks)."""
    base = run_dir.resolve() if run_dir.exists() else run_dir
    for p in sorted(base.glob("*.png")):
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
