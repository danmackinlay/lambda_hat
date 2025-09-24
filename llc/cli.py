import click
import json
import logging
import os
from dataclasses import replace

from .config import Config, apply_preset, override_config
from .sweep import load_study_yaml, run_sweep
from .exec_local import map_local

# Setup logging
def setup_logging(verbose: bool = False):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S"
    )
    # Quiet third-party libraries unless verbose
    if not verbose:
        logging.getLogger("matplotlib").setLevel(logging.WARNING)
        logging.getLogger("arviz").setLevel(logging.WARNING)

@click.group()
@click.option("-v", "--verbose", is_flag=True, help="Enable verbose logging")
def cli(verbose):
    setup_logging(verbose)

@cli.command()
@click.option("--sampler", type=click.Choice(["sgld", "sgnht", "hmc", "mclmc"]), required=True)
@click.option("--preset", type=click.Choice(["quick", "full"]))
@click.option("--gpu", type=str, help="GPU id (e.g. 0); omit for CPU")
@click.option("--save-plots", is_flag=True, help="Save diagnostic plots")
@click.option("--target", type=click.Choice(["mlp", "quadratic", "dln"]), help="Override target")
@click.option("--n-data", type=int, help="Override n_data")
@click.option("--chains", type=int, help="Override chains")
def run(sampler, preset, gpu, save_plots, target, n_data, chains):
    """Run a single sampler configuration."""
    cfg = Config(samplers=(sampler,))
    cfg = apply_preset(cfg, preset)

    # Apply overrides
    overrides = {}
    if save_plots:
        overrides["save_plots"] = True
    if target:
        overrides["target"] = target
    if n_data:
        overrides["n_data"] = n_data
    if chains:
        overrides["chains"] = chains

    if overrides:
        cfg = override_config(cfg, overrides)

    # Set GPU environment BEFORE importing JAX
    if gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
        os.environ["JAX_PLATFORMS"] = "cuda"
        os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    else:
        os.environ["JAX_PLATFORMS"] = "cpu"

    # Import AFTER env vars so JAX picks the right platform
    from .run import run_one
    result = run_one(cfg, save_artifacts=True, skip_if_exists=True)
    click.echo(f"Run completed: {result['run_dir']}")
    click.echo("Metrics:")
    click.echo(json.dumps(result["metrics"], indent=2))

@cli.command()
@click.option("--study", type=click.Path(exists=True), required=True, help="Study YAML file")
@click.option("--gpus", type=str, help="Comma-separated GPU ids, e.g. 0,1,2")
@click.option("--backend", type=click.Choice(["local", "slurm"]), default="local")
def sweep(study, gpus, backend):
    """Run a parameter sweep from study YAML file."""
    study_obj = load_study_yaml(study)

    # Expand to configs
    configs = []
    for p in study_obj.problems:
        for s in study_obj.samplers:
            for seed in study_obj.seeds:
                cfg = replace(
                    study_obj.base,
                    seed=seed,
                    samplers=(s.name,),
                    **p.overrides,
                    **s.overrides
                )
                configs.append(cfg)

    click.echo(f"Running {len(configs)} configurations...")

    if backend == "local":
        gpus_list = [int(x.strip()) for x in gpus.split(",")] if gpus else None
        results = map_local(configs, gpus=gpus_list)
        click.echo(f"Completed {len(results)} jobs successfully")
    else:
        from .exec_slurm import map_slurm
        results = map_slurm(configs)
        click.echo(f"Submitted {len(results)} SLURM jobs")

@cli.command()
@click.argument("run_dir", type=click.Path(exists=True))
@click.option("--which", type=str, default="all", help="Which sampler results to analyze")
@click.option("--plots", type=str, default="running_llc,rank,autocorr,energy,theta",
              help="Comma-separated list of plots to generate")
def analyze(run_dir, which, plots):
    """Analyze results from a run directory."""
    import glob
    import arviz as az
    import matplotlib.pyplot as plt
    from .analysis import fig_running_llc, fig_rank_llc, fig_autocorr_llc, fig_energy, fig_theta_trace

    # Create analysis output directory
    outdir = os.path.join(run_dir, "analysis")
    os.makedirs(outdir, exist_ok=True)

    # Parse requested plots
    requested = [p.strip() for p in plots.split(",") if p.strip()]

    # Find .nc files
    nc_files = glob.glob(os.path.join(run_dir, "*.nc"))
    if not nc_files:
        click.echo(f"No .nc files found in {run_dir}")
        return

    for nc_file in nc_files:
        sampler = os.path.basename(nc_file).replace(".nc", "")
        if which != "all" and sampler != which:
            continue

        click.echo(f"\n=== {sampler.upper()} ===")
        idata = az.from_netcdf(nc_file)
        summary = az.summary(idata, var_names=["llc"])
        click.echo(summary)

        # Generate requested plots
        if "running_llc" in requested:
            beta = float(idata.attrs.get("beta", 1.0))
            n = int(idata.attrs.get("n_data", 1))
            L0 = float(idata.attrs.get("L0", 0.0))
            fig = fig_running_llc(idata, n, beta, L0, f"{sampler.upper()} Running LLC")
            fig.savefig(os.path.join(outdir, f"{sampler}_running_llc.png"), dpi=150, bbox_inches='tight')
            plt.close(fig)

        if "rank" in requested:
            try:
                fig = fig_rank_llc(idata)
                fig.savefig(os.path.join(outdir, f"{sampler}_rank.png"), dpi=150, bbox_inches='tight')
                plt.close(fig)
            except Exception as e:
                click.echo(f"Warning: Could not generate rank plot for {sampler}: {e}")

        if "autocorr" in requested:
            try:
                fig = fig_autocorr_llc(idata)
                fig.savefig(os.path.join(outdir, f"{sampler}_autocorr.png"), dpi=150, bbox_inches='tight')
                plt.close(fig)
            except Exception as e:
                click.echo(f"Warning: Could not generate autocorr plot for {sampler}: {e}")

        if "energy" in requested:
            fig = fig_energy(idata)
            if fig:
                fig.savefig(os.path.join(outdir, f"{sampler}_energy.png"), dpi=150, bbox_inches='tight')
                plt.close(fig)

        if "theta" in requested:
            fig = fig_theta_trace(idata)
            if fig:
                fig.savefig(os.path.join(outdir, f"{sampler}_theta.png"), dpi=150, bbox_inches='tight')
                plt.close(fig)

    click.echo(f"\nPlots saved to {outdir}")

if __name__ == "__main__":
    cli()