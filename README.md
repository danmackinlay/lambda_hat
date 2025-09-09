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

## Installation

We use [uv](https://docs.astral.sh/uv/) instead of pip:

```bash
uv sync
```

GPU versions of libraries can occasionally be annoying with `uv`. We may switch to `pip`.
For now, adjust the JAX/`jaxlib` version for GPU if needed.

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
