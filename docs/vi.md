# Variational Inference (VI)

Variational Inference provides a fast, scalable alternative to markov Chain Monte Carlo (MCMC) for estimating the Local Learning Coefficient.

---

## Overview

**What is VI?**

VI approximates the tempered local posterior using a parameterized variational family $q_θ(w)$, optimized to maximize the Evidence Lower Bound (ELBO), where $θ$ denotes the variational parameters.

**Why use VI for LLC?**

- **Speed**: 10-100× faster than MCMC samplers
- **Scalability**: Efficient minibatch gradients
- **Exploration**: Fast hyperparameter sweeps and model selection
- **Initial estimates**: Validate experimental setup before running expensive MCMC

**When to use VI**:

- Model selection and architecture search
- Hyperparameter tuning (find optimal $\gamma, \beta$ and sampler settings)
- Initial LLC estimates before MCMC validation
- Resource-constrained settings

**When to use MCMC instead**:

- Gold-standard LLC estimates for publication
- Highly multimodal posteriors (though Flow VI may help)
- Need for rigorous convergence diagnostics

---

## Available VI Algorithms

Lambda-Hat provides two VI implementations:

### [MFA VI (Mixture of Factor Analyzers)](./vi_mfa.md)

**Default and recommended.**

- Mixture of low-rank Gaussians centered at $w^*$
- $M$ components, each with rank-$r$ covariance
- STL + Rao-Blackwellized gradients
- HVP control variate for variance reduction
- Robust, well-tested and fast

**When to use**: Default choice for most problems.

### [Flow VI (Normalizing Flows)](./vi_flow.md)

**Experimental, more expressive.**

- Normalizing flows with manifold-plus-noise construction
- RealNVP, MAF, or NSF architectures
- Vmap-compatible (PRNG issues resolved Nov 2025)
- Requires `uv sync --extra flowvi`

**When to use**: When MFA capacity is insufficient or posteriors are complex/multimodal.

---

## Quick Start

**Run workflow with VI** (builds targets and runs all samplers, including VI):
```bash
uv run lambda-hat workflow sample \
  --config config/experiments.yaml \
  --backend local
```

This command:

- Builds all targets defined in the config (Stage A)
- Runs all samplers defined in the config (Stage B), which typically includes:
  - **HMC** (Hamiltonian Monte Carlo)
  - **MCLMC** (Microcanonical Langevin Monte Carlo)
  - **SGLD** (Stochastic Gradient Langevin Dynamics)
  - **VI** (Variational Inference **MFA by default** see `lambda_hat/conf/sample/sampler/vi.yaml`)

To test samplers you may...

**Configure VI hyperparameters** in your experiments config (`config/experiments.yaml`):

All VI configuration options set via `overrides` under `sampler.vi`:

**Example configurations:**

```yaml
samplers:
  # Basic MFA VI with custom capacity
  - name: vi
    overrides:
      M: 8
      r: 2
      steps: 5000
      whitening_mode: "adam"
      tensorboard: true

  # Flow VI with custom architecture
  - name: vi
    overrides:
      algo: "flow"
      M: 8
      r: 2
      flow_depth: 6
      flow_type: "maf"
```

**Note**: Flow VI requires `uv sync --extra flowvi`. See [Flow VI documentation](./vi_flow.md) for details.

---

## Core Concepts

### Evidence Lower Bound (ELBO)

VI maximizes the ELBO:

$$
  \begin{align}
    \mathsf{ELBO}(q_θ) &= \mathbb{𝔼}_{q_θ(w)}[\log p(w)] - \mathbb{𝔼}_{q_θ(w)}[\log q_θ(w)] \\
                       &= \mathbb{𝔼}_{q_θ(w)}[\log p(w)] + \mathcal{H}(q_θ)
  \end{align}
$$

where:

- $p(w)$ is the tempered Gaussian tethered posterior,  $\exp(-β L(w) - γ ||w - w^*||²)$
- $q_θ(w)$ is the parameterized variational approximation ($θ$ is the variational parameter)
- $\mathcal{H}(q_θ)$ is the entropy of $q_θ$

**Key property**: $\mathsf{ELBO} ≤ \log Z$ (true log partition function)

### Tethering/Localization

All VI methods tether/localize around $w^*$ (ERM optimal point) via the quadratic penalty $||w - w^*||^2$ in the exponential:

$$p(w) = \exp( -\beta L(w) - \gamma ||w - w^*||² + \mathrm{const})$$

where:

- $\beta$ is inverse temperature (controls sampling temperature)
- $\gamma$ is a tethering strength (for the Gaussian tether around w*)
- $L(w)$ loss function (MSE, Gaussian likelihood, etc.)

**Tuning γ**:

- Try ${1e-4, 1e-3, 1e-2}$
- Smaller $\gamma$ corresponds to a wider posterior (more exploration)
- Larger $\gamma$ corresponds to a tighter posterior (stronger localization)
- Set via `posterior.gamma`, **not** `sampler.vi.gamma`

### Temperature β

Controls the sampling temperature:

- $\beta = 1 / \log(n)$ (default, `beta_mode: "1_over_log_n"`)
- $\beta = 1$ (full posterior, `beta_mode: "manual"`, `beta0: 1.0`)
- Larger $\beta$ begets sharper posterior (colder)
- Smaller $\beta$ begets smoother posterior (hotter)

---

## Running VI

### Single VI Run

```bash
# Build target first (if not already built) -- Stage A
uv run lambda-hat build \
  --config-yaml config/experiments.yaml \
  --target-id <target_id>

# Run VI samplers -- Stage B
uv run lambda-hat sample \
  --config-yaml config/experiments.yaml \
  --target-id <target_id>
```

### Via Workflow (Recommended)

```bash
# Run full workflow (builds targets + runs samplers)
uv run lambda-hat workflow sample --backend local

# With promotion (generate galleries)
uv run lambda-hat workflow sample --backend local  --promote
```

### Example Configuration

```yaml
# config/experiments.yaml
targets:
  - { model: base, data: base, teacher: _null, seed: 43 }

samplers:
  - name: vi
    overrides:
      M: 8
      r: 2
      steps: 5000
      whitening_mode: "adam"
      lr_schedule: "cosine"
      tensorboard: true
    seed: 54321

posterior:
  gamma: 0.001
  beta_mode: "1_over_log_n"
```

---

## Configuring VI

All VI configuration parameters are set via `overrides` under `sampler.vi` or, for posterior parameters, under `posterior`.

### Sampler Parameters (`sampler.vi`)

#### Algorithm Selection

| Parameter | Default | Description |
|-----------|---------|-------------|
| `algo` | `"mfa"` | VI algorithm: `"mfa"` (mixture of factor analyzers) or `"flow"` (normalizing flows) |

#### Core Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `M` | 8 | Number of mixture components (MFA) or latent dim (Flow) |
| `r` | 2 | Rank per component (MFA) or latent rank (Flow) |
| `r_per_component` | null | Heterogeneous rank budgets (list[int] of length M) |
| `steps` | 5000 | Optimization steps |
| `batch_size` | 256 | Minibatch size |
| `lr` | 0.01 | Learning rate (Adam optimizer) |
| `eval_every` | 50 | Trace recording frequency |
| `eval_samples` | 64 | MC samples for final LLC estimate |
| `dtype` | `"float32"` | Precision: `"float32"` or `"float64"` |

#### Optimization

| Parameter | Default | Description |
|-----------|---------|-------------|
| `lr_schedule` | null | LR schedule: `"cosine"` \| `"linear_decay"` |
| `lr_warmup_frac` | 0.05 | Fraction of steps for warmup |

#### Geometry Whitening

| Parameter | Default | Description |
|-----------|---------|-------------|
| `whitening_mode` | `"none"` | Geometry whitening: `"none"` \| `"rmsprop"` \| `"adam"` |
| `whitening_decay` | 0.99 | EMA decay for gradient moment accumulation |

#### Stability & Regularization

| Parameter | Default | Description |
|-----------|---------|-------------|
| `clip_global_norm` | 5.0 | Gradient clipping threshold (null to disable) |
| `alpha_temperature` | 1.0 | Softmax temperature on mixture weights (MFA) |
| `alpha_dirichlet_prior` | null | Dirichlet(α) prior on mixture weights (MFA) |
| `entropy_bonus` | 0.0 | Add λ × H(q) to ELBO for exploration |

#### Diagnostics

| Parameter | Default | Description |
|-----------|---------|-------------|
| `tensorboard` | false | Enable TensorBoard logging |

#### Flow-Specific Parameters

Only used when `algo: "flow"`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `d_latent` | 8 | Latent dimension for normalizing flow |
| `sigma_perp` | 1e-3 | Orthogonal noise scale for manifold-plus-noise map |
| `flow_depth` | 4 | Number of flow transformations (coupling layers) |
| `flow_hidden` | `[64, 64]` | Hidden layer sizes for flow network |
| `flow_type` | `"realnvp"` | Flow architecture: `"realnvp"` \| `"maf"` \| `"nsf_ar"` |

### Posterior Parameters (`posterior`)

Configure via `posterior` block (**not** `sampler.vi`):

| Parameter | Default | Description |
|-----------|---------|-------------|
| `gamma` | 0.001 | Localizer strength (Gaussian tether around w*) |
| `beta_mode` | `"1_over_log_n"` | Inverse temperature schedule |
| `beta0` | null | Manual β override (if `beta_mode = "manual"`) |
| `loss` | `"mse"` | Loss type: `"mse"` \| `"gaussian"` \| `"student"` |

---

## TensorBoard Monitoring

Enable real-time optimization diagnostics:

```yaml
samplers:
  - name: vi
    overrides:
      tensorboard: true
```

**Launch TensorBoard**:
```bash
tensorboard --logdir runs/targets/<tgt>/run_vi_<rid>/diagnostics/tb
```

### Key Metrics (Shared Across VI Methods)

**ELBO decomposition**:
- `vi/elbo` — Total ELBO (maximize this)
- `vi/elbo_like` — Expected log likelihood term
- `vi/logq` — Negative entropy term

**Localization**:
- `vi/radius2` — Localization radius (mean, quantiles)
- Track to verify posterior stays near w*

**Optimization**:
- `vi/grad_norm` — Gradient magnitude (watch for spikes)
- `vi/cumulative_fge` — Work-normalized progress

**Convergence**:
- ELBO should increase steadily then plateau
- `grad_norm` should decrease and stabilize
- `radius2` should stabilize at reasonable value

### Diagnosis

**Slow convergence** (flat ELBO):

- Increase capacity (M or r)
- Try learning rate schedule (`lr_schedule: "cosine"`)
- Check if γ is too large (try smaller)

**Training instability** (spiking grad_norm):

- Reduce learning rate
- Increase `clip_global_norm`
- Use `whitening_mode: "adam"`

**Poor LLC estimates**:

- Insufficient capacity (increase M or r)
- Try different VI algorithm (MFA ↔ Flow)
- Compare with HMC ground truth

---

## Work Tracking

VI reports work metrics for fair comparison with MCMC:

**Function-Gradient Evaluations (FGE)**:

- 1 VI step = 1 FGE (one minibatch gradient)
- Whitening pre-pass cost included in `total_fge`
- Cumulative FGE: `batch_size / n_data` per step

**Output metrics**:

- `total_fge` — Total function-gradient evaluations
- `n_full_loss` — MC samples for λ̂ estimation
- `n_minibatch_grads` — Optimization steps
- `sampler_flavour` — Algorithm used ("mfa" or "flow")

**Comparison with MCMC**:

- VI: ~5000 steps × (batch_size/n_data) FGE ≈ 1000 FGE for batch_size=256, n=1280
- HMC: ~1000 draws × 1 full gradient per draw ≈ 1000 FGE
- SGLD: ~20000 steps × (batch_size/n_data) FGE ≈ 4000 FGE for batch_size=256, n=1280

---

## Output Structure

VI produces the same output format as MCMC samplers:

```
runs/targets/tgt_abc123/run_vi_rid123/
├── trace.nc              # ArviZ-compatible NetCDF trace
├── analysis.json         # Metrics (llc_mean, llc_std, ESS, etc.)
└── diagnostics/
    ├── trace.png         # Trace plots
    ├── llc_convergence_combined.png
    └── tb/               # TensorBoard logs (if enabled)
```

**Key metrics in `analysis.json`**:

- `lambda_hat` — LLC estimate (mean)
- `lambda_hat_std` — LLC standard error
- `ess` — Effective sample size
- `total_fge` — Work metric (function-gradient evaluations)

---

## Implementation Notes

### Precision

VI operates in `float32` by default:
- Sufficient for LLC estimation
- ~2× faster than float64
- Numerical stability features (clipping, ridge regularization)
- Can switch to `float64` via `dtype: float64` (rarely needed)

### HVP Usage

**HVP (Hessian-Vector Product)** is used selectively:

- **Not used** during optimization (JIT stability issues)
- **Used only at evaluation time** for control variate
- Reduces LLC estimate variance without biasing the estimate
- Optimization uses RMSProp/Adam diagonal whitening instead

### Return Structure

VI returns standard sampling results compatible with other samplers:

```python
{
    "lambda_hat": jnp.array(...),     # LLC estimate (scalar)
    "traces": {                       # Per-iteration metrics
        "elbo": jnp.array(...),
        "grad_norm": jnp.array(...),
        # ...
    },
    "extras": {                       # Final evaluation metrics
        "Eq_Ln": jnp.array(...),
        "Ln_wstar": jnp.array(...),
        "cv_info": { ... },           # Control variate diagnostics
    },
    "timings": {                      # Wall-clock times
        "adaptation": 0.0,            # VI has no adaptation phase
        "sampling": float(...),       # Training + evaluation time
        "total": float(...),
    },
    "work": {                         # FGE accounting
        "n_full_loss": int(...),
        "n_minibatch_grads": int(...),
        # ...
    },
}
```

---

## Algorithm Comparison

| Feature | MFA VI | Flow VI |
|---------|--------|---------|
| **Status** | Default, robust | Experimental |
| **Family** | Mixture of Gaussians | Normalizing flows |
| **Expressiveness** | Medium | High |
| **Speed** | Fast | Medium |
| **Complexity** | Low | High |
| **Multimodal** | Limited | Better |
| **Requirements** | None | `--extra flowvi` |
| **Vmap** | ✅ Compatible | ✅ Compatible (Nov 2025 fix) |

**When to use MFA**: Default choice, fastest, works well for most problems

**When to use Flow**: Complex posteriors, suspected multimodality, MFA capacity insufficient

---

## See Also

- [MFA VI](./vi_mfa.md) — Mixture of factor analyzers (default algorithm)
- [Flow VI](./vi_flow.md) — Normalizing flows (experimental algorithm)
- [Configuration Reference](./config.md) — Complete YAML schema
- [Samplers](./samplers.md) — Comparison with MCMC methods
- [Workflows](./workflows.md) — Running VI in sweeps with Parsl
- [Experiments Guide](./experiments.md) — Composing configs and overrides
