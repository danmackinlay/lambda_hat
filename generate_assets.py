import subprocess
import shutil
from pathlib import Path
import datetime
import time
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("generate_assets")


def run_hydra_sweep_and_collect_assets():
    """
    Launches a Hydra multirun sweep for 3 runs and collects generated plots.
    """
    ROOT_DIR = Path(__file__).parent.resolve()
    ASSET_DIR = ROOT_DIR / "assets" / "readme"
    ASSET_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Define the Hydra command
    # We use 'fast' configurations for quick asset generation.
    experiment_name = (
        f"asset_generation_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    # Hydra output path relative to the root directory
    output_dir = ROOT_DIR / "multirun" / experiment_name

    # Define the 3 runs (sweeping over seeds)
    seeds = [42, 43, 44]
    num_runs = len(seeds)

    command = [
        "python",
        str(ROOT_DIR / "lambda_hat/cli_main.py"),
        "--multirun",
        "sampler=fast",  # Use fast sampler settings
        "model=small",  # Use small model
        "data=small",  # Use small data
        f"seed={','.join(map(str, seeds))}",
        f"hydra.sweep.dir=multirun/{experiment_name}",  # Set specific output dir
    ]

    log.info(f"Running Hydra sweep: {' '.join(command)}")

    # 2. Run the command
    start_time = time.time()
    try:
        # Ensure we run from the root directory for correct path resolution
        subprocess.run(
            command, check=True, capture_output=True, text=True, cwd=ROOT_DIR
        )
        log.info(f"Sweep completed successfully in {time.time() - start_time:.2f}s.")
    except subprocess.CalledProcessError as e:
        log.error(
            f"Error running Hydra sweep:\nSTDOUT:\n{e.stdout}\nSTDERR:\n{e.stderr}"
        )
        return

    # 3. Collect assets
    log.info("--- Collecting Assets ---")

    # Hydra organizes multirun outputs by job number (0, 1, 2, ...)
    for job_num in range(num_runs):
        run_dir = output_dir / str(job_num)
        if not run_dir.exists():
            log.warning(f"Run directory not found: {run_dir}")
            continue

        log.info(f"Processing Job {job_num} (Seed {seeds[job_num]})")

        # Copy Trace Plot (visualization of ArviZ traces)
        trace_plot = run_dir / "llc_traces.png"
        if trace_plot.exists():
            target_path = ASSET_DIR / f"llc_traces_run_{job_num}.png"
            shutil.copy(trace_plot, target_path)
            log.info(f"Copied: {target_path}")

        # Copy Comparison Plot
        comparison_plot = run_dir / "llc_comparison.png"
        if comparison_plot.exists():
            target_path = ASSET_DIR / f"llc_comparison_run_{job_num}.png"
            shutil.copy(comparison_plot, target_path)
            log.info(f"Copied: {target_path}")

    log.info("Asset collection complete.")


if __name__ == "__main__":
    # Ensure we run from the root for correct Hydra path resolution
    if Path.cwd() != Path(__file__).parent.resolve():
        os.chdir(Path(__file__).parent.resolve())
    run_hydra_sweep_and_collect_assets()
