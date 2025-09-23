# cli package
# llc/cli.py
"""
Click-based CLI for LLC - thin routing only.
All business logic moved to command modules.
"""

from __future__ import annotations
import logging

import os
import click

from llc.cli.options import (
    run_shared_options,
    add_run_sampler_choice,
    sweep_shared_options,
    analyze_shared_options,
)


# ---------- Click CLI ----------


@click.group()
@click.option("--verbose", is_flag=True, help="Enable verbose logging.")
def cli(verbose: bool):
    """LLC command-line interface (Click)."""
    # Configure JAX and matplotlib environment for CLI usage
    os.environ.setdefault("JAX_ENABLE_X64", "true")
    os.environ.setdefault(
        "MPLBACKEND", "Agg"
    )  # headless backend for server environments

    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )


# ----- Commands -----


@cli.command()
@add_run_sampler_choice
@run_shared_options()
def run(**kwargs):
    """Run a single experiment (local / submitit / modal)."""
    from llc.commands.run_cmd import run_entry

    return run_entry(kwargs)


@cli.command("repeat")
@click.option("--cfg-json", type=click.Path(exists=True, dir_okay=False), help="Config JSON (payload) to replay.")
@click.option("--from-run", type=click.Path(exists=True, file_okay=False), help="Existing run dir (uses run_dir/config.json).")
@add_run_sampler_choice
@run_shared_options()
def repeat_cmd(cfg_json, from_run, **kwargs):
    """Repeat a run from a saved config JSON (or from an existing run dir)."""
    import json
    from llc.commands.run_cmd import run_entry

    if not cfg_json and not from_run:
        raise SystemExit("Provide --cfg-json or --from-run.")
    if from_run:
        cfg_json = os.path.join(from_run, "config.json")

    with open(cfg_json) as f:
        cfg = json.load(f)

    # Merge backend/gpu/submitit flags from kwargs on top of saved config
    cfg.update({k: v for k, v in kwargs.items() if v is not None})

    # Enforce single-sampler, if present
    if isinstance(cfg.get("samplers"), (list, tuple)) and len(cfg["samplers"]) == 1:
        kwargs["sampler"] = cfg["samplers"][0]

    return run_entry({**cfg, **kwargs})


@cli.command()
@sweep_shared_options()
def sweep(**kwargs):
    """Run a parameter sweep (local / submitit / modal)."""
    from llc.commands.sweep_cmd import sweep_entry

    return sweep_entry(kwargs)


@cli.command()
@click.argument("run_dir", type=click.Path(exists=True))
@analyze_shared_options()
def analyze(run_dir, **kwargs):
    """Post-hoc analysis on saved .nc files"""
    from llc.commands.analyze_cmd import analyze_entry

    return analyze_entry(run_dir, **kwargs)


@cli.command("promote-readme-images")
@click.argument(
    "run_dir", required=False, type=click.Path(exists=True, file_okay=False)
)
@click.option(
    "--root",
    type=click.Path(exists=True, file_okay=False),
    default=".",
    help="Repository root (default: .)",
)
def promote_readme_images_cmd(run_dir, root):
    """
    Copy a curated set of diagnostic PNGs from a run dir into assets/readme/.
    If RUN_DIR is omitted, pick the newest completed run under runs/.
    """
    from llc.commands.promote_cmd import promote_readme_images_entry

    return promote_readme_images_entry(run_dir, root)


@cli.command("pull-runs")
@click.argument("run_id", required=False)
@click.option(
    "--target",
    default="runs",
    show_default=True,
    type=click.Path(file_okay=False),
    help="Local target root",
)
def pull_artifacts_cmd(run_id, target):
    """
    Pull runs from Modal using the deployed SDK functions.
    If RUN_ID is omitted, discover the latest on the server.
    """
    from llc.commands.pull_cmd import pull_runs_entry

    return pull_runs_entry(run_id, target)


@cli.command("plot-sweep")
@click.option(
    "--csv",
    "csv_path",
    type=click.Path(exists=True, dir_okay=False),
    default="llc_sweep_results.csv",
    show_default=True,
    help="Sweep results CSV (from `llc sweep`).",
)
@click.option(
    "--out",
    "out_dir",
    type=click.Path(file_okay=False),
    default="sweep_plots",
    show_default=True,
    help="Directory to write PNGs.",
)
@click.option(
    "--size-col",
    type=click.Choice(["target_params", "n_data", "depth", "hidden"]),
    default="target_params",
    show_default=True,
    help="Which column to treat as 'problem size' on the x-axis.",
)
@click.option(
    "--samplers",
    default="sgld,sghmc,hmc,mclmc",
    help="Comma-separated subset of samplers to plot (default: all).",
)
@click.option(
    "--filters",
    default="",
    help='Comma-separated equality filters like "activation=relu,x_dist=gauss_iso".',
)
@click.option(
    "--logx/--no-logx",
    default=True,
    show_default=True,
    help="Log-scale for the size axis.",
)
@click.option("--overwrite", is_flag=True, help="Overwrite any existing PNGs.")
def plot_sweep_cmd(csv_path, out_dir, size_col, samplers, filters, logx, overwrite):
    """
    Plot sweep efficiency: ESS/sec and WNV vs problem size, plus a SE frontier.

    Examples:
      llc plot-sweep --csv llc_sweep_results.csv --filters "activation=relu,x_dist=gauss_iso"
      llc plot-sweep --size-col n_data --samplers hmc,mclmc
    """
    from llc.commands.plot_sweep_cmd import plot_sweep_entry

    return plot_sweep_entry(
        csv_path, out_dir, size_col, samplers, filters, logx, overwrite
    )


@cli.command("showcase-readme")
@click.option("--sampler", type=click.Choice(["sgld", "sghmc", "hmc", "mclmc"]), help="Run only this sampler instead of all.")
@run_shared_options()  # inherits backend/gpu/slurm/modal flags
def showcase_readme_cmd(**kwargs):
    """Run 'full' preset on chosen backend, generate plots, and refresh README images."""
    from llc.commands.showcase_cmd import showcase_readme_entry
    return showcase_readme_entry(**kwargs)


@cli.command("preview")
@click.argument("run_dir", type=click.Path(exists=True, file_okay=False))
def preview_cmd(run_dir):
    """(Re)generate the HTML preview for a run directory."""
    import json
    import os
    from llc.config import Config
    from llc.artifacts import generate_gallery_html

    cfg_path = os.path.join(run_dir, "config.json")
    metrics_path = os.path.join(run_dir, "metrics.json")
    cfg = Config(**json.load(open(cfg_path))) if os.path.exists(cfg_path) else Config()
    metrics = json.load(open(metrics_path)) if os.path.exists(metrics_path) else {}
    path = generate_gallery_html(run_dir, cfg, metrics)
    print(f"HTML preview: {path}")


if __name__ == "__main__":
    cli()
