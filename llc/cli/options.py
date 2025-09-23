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
        # GPU options
        f = click.option(
            "--gpu-mode",
            type=click.Choice(["off", "vectorized", "sequential"]),
            default="off",
            help="GPU execution on a single device: 'vectorized' vmaps chains; 'sequential' runs one chain at a time.",
        )(f)
        f = click.option(
            "--gpu-types",
            type=str,
            default="L40S",
            help="Comma-separated Modal GPU types to request (e.g. 'H100,A100,L40S'). Optional.",
        )(f)

        # Submitit/SLURM options
        f = click.option(
            "--slurm-partition",
            type=str,
            help="SLURM partition name (e.g. 'gpu', 'cpu'). Optional.",
        )(f)
        f = click.option(
            "--account",
            "slurm_account",
            type=str,
            help="SLURM account name (e.g. 'abc123'). Optional.",
        )(f)
        f = click.option(
            "--timeout-min",
            type=int,
            default=180,  # 3 hours to match Modal default
            help="Timeout in minutes for SLURM jobs.",
        )(f)
        f = click.option(
            "--cpus",
            type=int,
            default=4,
            help="CPUs per task for SLURM jobs.",
        )(f)
        f = click.option(
            "--mem-gb",
            type=int,
            default=16,
            help="Memory in GB for SLURM jobs.",
        )(f)
        f = click.option(
            "--slurm-signal-delay-s",
            type=int,
            default=120,
            help="Grace period in seconds before SLURM kills job.",
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
            help="Don't save run outputs (plots, data files).",
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
        f = click.option(
            "--sgld-eval-every", type=int, help="SGLD evaluation frequency"
        )(f)
        f = click.option("--sgld-thin", type=int, help="SGLD thinning factor")(f)

        # SGLD preconditioning
        f = click.option(
            "--sgld-precond",
            type=click.Choice(["none", "rmsprop", "adam"]),
            default=None,
            help="Diagonal preconditioning for SGLD",
        )(f)
        f = click.option("--sgld-beta1", type=float, help="Adam beta1 (first moment)")(
            f
        )
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
        f = click.option("--hmc-eval-every", type=int, help="HMC evaluation frequency")(
            f
        )
        f = click.option("--hmc-thin", type=int, help="HMC thinning factor")(f)

        # MCLMC parameters
        f = click.option("--mclmc-draws", type=int, help="MCLMC total draws")(f)
        f = click.option(
            "--mclmc-eval-every", type=int, help="MCLMC evaluation frequency"
        )(f)
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
            "--workers",
            type=int,
            default=0,
            help="Number of local workers (0/1=serial)",
        )(f)
        f = click.option(
            "--n-seeds",
            type=int,
            default=2,
            help="Number of random seeds per configuration",
        )(f)
        # Study inputs
        f = click.option(
            "--study",
            type=click.Path(exists=True, dir_okay=False),
            help="YAML file describing base/problem/sampler/seeds (overrides defaults).",
        )(f)
        f = click.option(
            "--sampler-grid",
            type=str,
            default=None,
            help='JSON list of sampler variants, e.g. \'[{"name":"sgld","overrides":{"sgld_precond":"adam"}}]\'',
        )(f)
        f = click.option(
            "--problem-grid",
            type=str,
            default=None,
            help='JSON list of problems, e.g. \'[{"name":"dim_10k","overrides":{"target_params":10000}}]\'',
        )(f)
        # Reuse run-shared too (includes backend)
        return run_shared_options()(f)

    return decorator


def analyze_shared_options():
    """Shared options for analyze command."""

    def decorator(f):
        f = click.option(
            "--which",
            type=click.Choice(["all", "sgld", "sghmc", "hmc", "mclmc"]),
            default="all",
            help="Which sampler(s) to analyze",
        )(f)
        f = click.option(
            "--plots",
            default="running_llc,rank,ess_evolution,ess_quantile,autocorr,energy,theta",
            help="Comma-separated list of plots to generate",
        )(f)
        f = click.option(
            "--out",
            type=click.Path(file_okay=False),
            default=None,
            help="Output directory (default: <run_dir>/analysis)",
        )(f)
        f = click.option("--overwrite", is_flag=True, help="Overwrite existing plots")(
            f
        )
        return f

    return decorator


def _sampler_choice():
    # Kept separate so we can reuse later if needed
    return click.Choice(["sgld", "sghmc", "hmc", "mclmc"])

def add_run_sampler_choice(f):
    """Opt-in sampler selector for `llc run` (single-sampler intent)."""
    import click
    return click.option(
        "--sampler",
        type=_sampler_choice(),
        help="Sampler to run (enforces single-sampler runs).",
    )(f)
