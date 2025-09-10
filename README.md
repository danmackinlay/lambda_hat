# Local Learning Coefficient Sampler Benchmarks

This repo contains code to **estimate Local Learning Coefficients (LLCs)** for small neural networks using **stochastic gradient Langevin dynamics (SGLD)**, **Hamiltonian Monte Carlo (HMC)**, and **Microcanonical Langevin Monte Carlo (MCLMC)**, all implemented via [BlackJAX](https://github.com/blackjax-devs/blackjax/tree/1.2.5).

---

## Motivation

In [Singular Learning Theory (SLT)](https://singularlearningtheory.com), the _Local Learning Coefficient (LLC)_ quantifies the *effective local dimensionality* of a model around a trained optimum. The LLC is crucial for understanding the geometry of singular loss surfaces, which differ fundamentally from the quadratic approximations that standard Bayesian Laplace methods assume.
The recent [*From Global to Local: A Scalable Benchmark for Local Posterior Sampling* (Hitchcock & Hoogland, 2024)](file-9pNmXEB8xGTwKS1evcvu5F) uses _deep linear networks (DLNs)_ as ground truth, because those admit analytic LLC values.

We might fail to be persuaded by those;
linear nets are a very special case.
Our research agenda is to see how well SGLD (and alternative SGMCMC methods) track local geometry in *nonlinear* models (ReLU, GeLU, etc.) where analytic LLCs aren’t available.
To do that responsibly, we first need to ground-truth against a sampler we trust (HMC) on small models with ~10k parameters — large enough to show interesting degeneracies, but still small enough that HMC is (barely) feasible.

Ultimately, we want to devise and evaluate new sampling algorithms for singular neural nets. This repo is the foundation: it gives us side-by-side SGLD, HMC, and MCLMC runs with consistent LLC estimation and diagnostics.

## What’s inside

- `main.py` — end-to-end pipeline:

  - Flexible **MLP model** with configurable depth, widths, activation (ReLU, tanh, GeLU, identity for deep-linear).
  - **Teacher–student data generator** with parametric input distributions (isotropic Gaussian, anisotropic, mixture of Gaussians, low-dim manifolds, heavy-tailed).
  - Noise models: Gaussian, heteroscedastic, Student-t, outliers.
  - **ERM training** to locate the empirical minimizer \(w^\*\); the local Gaussian prior is centered at \(w^\*\).
  - Tempered local posterior (\(\beta = \beta_0/\log n\) by default) + Gaussian localization (\(\gamma = d / r^2\) if `prior_radius` given).
  - **Samplers**:
    - **SGLD** (unadjusted stochastic gradient Langevin dynamics, with minibatching, online LLC evaluation, optional RMSProp/Adam preconditioning to come).
    - **HMC** (full-batch, with BlackJAX `window_adaptation` to tune step size + diagonal mass).
    - **MCLMC** (microcanonical Langevin Monte Carlo, unadjusted, with automatic tuning of step size and momentum decoherence length `L` using [the official BlackJAX tuner](https://blackjax-devs.github.io/sampling-book/algorithms/mclmc.html)).
  - **Online LLC estimator** (Def. 3.1 in the paper) computed during sampling, using occasional full-batch loss evaluations.
  - **Diagnostics via [ArviZ](https://python.arviz.org/):**
    - Running \(\hat\lambda_t\) (per-chain + pooled).
    - Trace, autocorrelation, ESS, and \(\hat R\) for \(L_n(w)\).
    - Optional trace/rank plots for a tiny subset or random projection of θ (memory-safe).
    - HMC acceptance-rate histogram; MCLMC energy-change histogram.
  - **Work-normalized variance (WNV)** metrics: variance of LLC estimate × (wall-clock time or gradient-equivalent count).

---

## Example Diagnostics

These are examples from a recent run (promoted to `assets/readme/`).

### Running LLC Estimates

![SGLD running LLC](assets/readme/sgld_llc_running.png)
*SGLD: Running Local Learning Coefficient estimates showing convergence over sampling iterations*

![HMC running LLC](assets/readme/hmc_llc_running.png)
*HMC: Running LLC estimates with multiple chains and pooled estimate*

![MCLMC running LLC](assets/readme/mclmc_llc_running.png)
*MCLMC: Running LLC estimates showing sampling efficiency*

### MCMC Diagnostics

![HMC L_n Traces](assets/readme/hmc_L_trace.png)
*HMC: Trace plots of loss function values L_n across chains*

![HMC Autocorrelation](assets/readme/hmc_L_acf.png)
*HMC: Autocorrelation function for L_n showing mixing properties*

![HMC Acceptance](assets/readme/hmc_acceptance.png)
*HMC: Acceptance rate distribution across chains*

![MCLMC Energy Changes](assets/readme/mclmc_energy_hist.png)
*MCLMC: Energy change distribution showing microcanonical dynamics*

---

## Installation

We use [uv](https://docs.astral.sh/uv/) for dependency management:

```bash
uv sync                    # Core dependencies only
```

### Optional backends

For distributed computing, install the appropriate backend:

```bash
uv sync --extra slurm      # SLURM/submitit support
uv sync --extra modal      # Modal serverless support
uv sync --all-extras       # Both backends
```

Or with pip:
```bash
pip install llc[slurm]     # SLURM support
pip install llc[modal]     # Modal support
```

---

## Visualization and Artifact Management

The system includes comprehensive visualization saving capabilities for systematic analysis and documentation:

### Automatic Run Organization

- **Timestamped Directories**: Each run creates `artifacts/YYYYMMDD-HHMMSS/` directories
- **Deterministic Naming**: Plots use consistent `<sampler>_<plotname>.png` format
- **Run Manifest**: `manifest.txt` contains complete configuration and runtime statistics
- **Documentation**: `README_snippet.md` provides formatted run summaries

### Saved Diagnostic Plots

For each sampler (SGLD, HMC, MCLMC), the system saves:

- `*_llc_running.png` - Running LLC estimates over time
- `*_L_trace.png` - Trace plots of loss function values
- `*_L_acf.png` - Autocorrelation function plots
- `*_L_ess.png` - Effective sample size plots
- `*_L_rhat.png` - R-hat convergence diagnostics
- `*_theta_trace.png` - Parameter trace plots (subset)
- `*_theta_rank.png` - Rank plots for parameters

Additional sampler-specific plots:
- `hmc_acceptance.png` - HMC acceptance rate histogram
- `mclmc_energy_hist.png` - MCLMC energy change distribution

### Configuration Options

Enable visualization saving in your config:

```python
from main import Config, main

cfg = Config(
    save_plots=True,              # Save all diagnostic plots
    save_manifest=True,           # Generate run manifest
    save_readme_snippet=True,     # Create documentation snippet
    artifacts_dir="artifacts",    # Base directory (default)
    auto_create_run_dir=True,     # Create timestamped subdirs
)

main(cfg)
```

### Makefile Targets

The included Makefile provides convenient shortcuts:

```bash
make run-save       # Run with visualization saving enabled
make diag           # Quick test run with plots
make clean          # Remove all artifacts
make artifacts      # Create artifacts directory
make promote-readme # Copy plots from latest run to README assets
```

### Updating README Examples

To refresh the example plots in the README:

```bash
make diag                    # Run a quick diagnostic with plots
make promote-readme         # Copy selected plots to assets/readme/
git add assets/readme       # Stage the new images
git commit -m "refresh README examples"
```

The promotion script automatically selects key diagnostic plots and copies them to stable filenames in `assets/readme/`. No manual file management needed.

---

## Usage

### Run a single experiment

```bash
uv run python main.py
```

* Uses the default `Config` (`in_dim=32`, `target_params≈10k`, ReLU MLP).
* Trains to ERM, centers prior at $w^\*$, runs **SGLD** then **HMC** then **MCLMC**.
* Prints LLC estimates, ESS/$\hat R$, acceptance stats, WNV.
* Shows ArviZ convergence plots (trace, autocorr, ESS, R̂) plus running LLC curves.

### Run a sweep

```bash
uv run python main.py sweep
```

* Iterates over depth/width/activation/data/noise settings (see `sweep_space()`),
* logs per-run LLC results and saves to `llc_sweep_results.csv`.

#### Distributed sweeps

**Local (default):**
```bash
uv run python main.py sweep --backend=local --workers=4
```

**SLURM cluster:**
```bash
uv run python main.py sweep --backend=submitit \
  --partition=gpu --gpus=1 --timeout-min=60 \
  --save-artifacts --artifacts-dir=/shared/llc_results
```

**Modal serverless:**

```bash
uv run modal volume create llc-artifacts
uv run python main.py sweep --backend=modal \
  --modal-timeout-s=3600 --save-artifacts
```

Use `--save-artifacts` to generate full diagnostic plots and data. Artifacts are saved locally (local backend), to shared storage (SLURM), or Modal volumes (modal backend).

#### Retrieving Modal artifacts

After running sweeps on Modal, download results locally:

```bash
# List available runs
modal volume ls llc-artifacts

# Download specific run
modal volume get llc-artifacts /artifacts/20250909-172233 ./artifacts/20250909-172233

# Download all results
modal volume get llc-artifacts /artifacts ./artifacts
```

---

## Roadmap

* **Preconditioned SGLD**: RMSProp-SGLD / Adam-SGLD (per Hitchcock & Hoogland’s findings).
* **Adjusted MCLMC**: MH-corrected variant with adaptive step-size (target accept ≈0.9).
* **Trans-dimensional moves**: exploring SLT’s *blow-ups* and richer sampler designs.
* **Scaling studies**: push toward larger ReLU/GELU networks, beyond HMC’s limit, to stress-test SGLD/MCLMC.
* **Better LLC error estimation**: block bootstrap on $L_n$ traces, multi-chain variance combination.

---

## Notes on BlackJAX API (v1.2.5)

To prevent confusion across docs vs release:

* **SGLD**

  * Public API: `sgld = blackjax.sgld(grad_fn)`
  * Step signature: `new_position = sgld.step(rng_key, position, minibatch, step_size)`
  * Source: [sgld.py (1.2.5)](https://github.com/blackjax-devs/blackjax/blob/1.2.5/blackjax/sgmcmc/sgld.py#L38-L47)
    (see `step_fn` → returns `kernel(...)` → returns `new_position`).

* **HMC**

  * Use `blackjax.hmc` with `blackjax.window_adaptation`.
  * `HMCInfo` fields include `acceptance_rate` (flat attribute).
  * Source: [hmc.py (1.2.5)](https://github.com/blackjax-devs/blackjax/blob/1.2.5/blackjax/mcmc/hmc.py#L330-L334).

* **MCLMC**

  * See [Sampling Book MCLMC example](https://blackjax-devs.github.io/sampling-book/algorithms/mclmc.html).
  * Tune `(L, step_size)` with `blackjax.mclmc_find_L_and_step_size`, then build `blackjax.mclmc(logdensity_fn, L, step_size)`.
  * Integrators available in [integrators module (1.2.5)](https://github.com/blackjax-devs/blackjax/tree/1.2.5/blackjax/mcmc/integrators) (e.g. `isokinetic_mclachlan`).
  * `MCLMCInfo` has `energy_change` field (see [mclmc.py (1.2.5)](https://github.com/blackjax-devs/blackjax/blob/1.2.5/blackjax/mcmc/mclmc.py)).

⚠️ **Docs drift warning**: the online blackjax docs default to `main`.
They may show `acceptance_probability` for HMC or a different SGLD step signature. Always cross-check the [1.2.5 tag source](https://github.com/blackjax-devs/blackjax/tree/1.2.5) when in doubt.

---

## License

MIT.
