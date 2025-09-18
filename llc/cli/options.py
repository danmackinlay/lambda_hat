"""Shared Click option decorators for CLI commands."""

import click


def run_shared_options():
    """Shared options for run command and commands that extend it."""
    def decorator(f):
        # Backend / basics
        f = click.option(
            "--backend",
            type=click.Choice(["local", "submitit", "modal"]),
            default="local",
            help="Execution backend.",
        )(f)
        f = click.option(
            "--preset",
            type=click.Choice(["quick", "full"]),
            default=None,
            help="Apply preset configuration.",
        )(f)
        # Control flags
        f = click.option(
            "--skip/--no-skip",
            "skip_if_exists",
            default=True,
            help="Skip run if results already exist.",
        )(f)
        f = click.option(
            "--no-artifacts",
            is_flag=True,
            default=False,
            help="Don't save artifacts (plots, data files, gallery).",
        )(f)
        # Plotting
        f = click.option(
            "--save-plots/--no-save-plots",
            "save_plots",
            default=None,
            help="Enable/disable saving diagnostic plots.",
        )(f)

        # Data parameters
        f = click.option("--n-data", type=int, help="Number of data points")(f)
        f = click.option("--seed", type=int, help="Random seed")(f)
        f = click.option(
            "--loss", type=click.Choice(["mse", "t_regression"]), help="Loss function"
        )(f)

        # Model architecture
        f = click.option("--depth", type=int, help="Number of hidden layers")(f)
        f = click.option("--width", type=int, help="Hidden layer width")(f)
        f = click.option(
            "--target-params", "target_params", type=int, help="Target parameter count"
        )(f)

        # Target selection
        f = click.option(
            "--target", type=click.Choice(["mlp", "quadratic"]), help="Target model"
        )(f)
        f = click.option(
            "--quad-dim", type=int, help="Parameter dimension for quadratic target"
        )(f)

        # SGLD parameters
        f = click.option("--sgld-steps", type=int, help="SGLD total steps")(f)
        f = click.option("--sgld-warmup", type=int, help="SGLD warmup steps")(f)
        f = click.option("--sgld-step-size", type=float, help="SGLD step size")(f)
        f = click.option("--sgld-batch-size", type=int, help="SGLD minibatch size")(f)
        f = click.option("--sgld-eval-every", type=int, help="SGLD evaluation frequency")(f)
        f = click.option("--sgld-thin", type=int, help="SGLD thinning factor")(f)

        # SGLD preconditioning
        f = click.option(
            "--sgld-precond",
            type=click.Choice(["none", "rmsprop", "adam"]),
            default=None,
            help="Diagonal preconditioning for SGLD",
        )(f)
        f = click.option("--sgld-beta1", type=float, help="Adam beta1 (first moment)")(f)
        f = click.option(
            "--sgld-beta2", type=float, help="RMSProp/Adam beta2 (second moment)"
        )(f)
        f = click.option("--sgld-eps", type=float, help="Preconditioner epsilon")(f)
        f = click.option(
            "--sgld-bias-correction/--no-sgld-bias-correction",
            "sgld_bias_correction",
            default=None,
            help="Enable/disable Adam bias correction",
        )(f)

        # HMC parameters
        f = click.option("--hmc-draws", type=int, help="HMC total draws")(f)
        f = click.option("--hmc-warmup", type=int, help="HMC warmup steps")(f)
        f = click.option("--hmc-eval-every", type=int, help="HMC evaluation frequency")(f)
        f = click.option("--hmc-thin", type=int, help="HMC thinning factor")(f)

        # MCLMC parameters
        f = click.option("--mclmc-draws", type=int, help="MCLMC total draws")(f)
        f = click.option("--mclmc-eval-every", type=int, help="MCLMC evaluation frequency")(
            f
        )
        f = click.option("--mclmc-thin", type=int, help="MCLMC thinning factor")(f)

        # Chain parameters
        f = click.option("--chains", type=int, help="Number of parallel chains")(f)
        f = click.option(
            "--use-batched-chains/--no-batched-chains",
            default=None,
            help="Use batched chain execution (vmap+scan) for speed",
        )(f)

        return f
    return decorator


def sweep_shared_options():
    """Shared options for sweep command."""
    def decorator(f):
        f = click.option(
            "--workers", type=int, default=0, help="Number of local workers (0/1=serial)"
        )(f)
        f = click.option(
            "--n-seeds",
            type=int,
            default=2,
            help="Number of random seeds per configuration",
        )(f)
        # Reuse run-shared too (includes backend)
        return run_shared_options()(f)
    return decorator


def analyze_shared_options():
    """Shared options for analyze command."""
    def decorator(f):
        f = click.option(
            "--which",
            type=click.Choice(["all", "sgld", "hmc", "mclmc"]),
            default="all",
            help="Which sampler(s) to analyze"
        )(f)
        f = click.option(
            "--plots",
            default="running_llc,rank,ess_evolution,ess_quantile,autocorr,energy,theta",
            help="Comma-separated list of plots to generate"
        )(f)
        f = click.option(
            "--out",
            type=click.Path(file_okay=False),
            default=None,
            help="Output directory (default: <run_dir>/analysis)"
        )(f)
        f = click.option(
            "--overwrite",
            is_flag=True,
            help="Overwrite existing plots"
        )(f)
        return f
    return decorator