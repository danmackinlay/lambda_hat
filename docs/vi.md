# Variational Inference (VI)

Variational Inference provides a fast, scalable alternative to MCMC for estimating the Local Learning Coefficient.

---

## Overview

**What is VI?**
VI approximates the tempered local posterior using a parameterized variational family q(w), optimized to maximize the Evidence Lower Bound (ELBO).

**Why use VI for LLC?**
- **Speed**: 10-100× faster than MCMC samplers
- **Scalability**: Efficient minibatch gradients
- **Exploration**: Fast hyperparameter sweeps and model selection
- **Initial estimates**: Validate experimental setup before running expensive MCMC

**When to use VI**:
- Model selection and architecture search
- Hyperparameter tuning (find optimal γ, β, sampler settings)
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

- Mixture of low-rank Gaussians centered at w*
- M components, each with rank-r covariance
- STL + Rao-Blackwellized gradients
- HVP control variate for variance reduction
- Robust, well-tested, fast

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

**Basic usage** (uses MFA by default):
```bash
uv run lambda-hat workflow llc --backend local
```

**Custom VI settings** in `config/experiments.yaml`:
```yaml
samplers:
  - name: vi
    overrides:
      M: 8                      # Number of components (MFA) or latent dim (Flow)
      r: 2                      # Rank per component (MFA) or latent rank (Flow)
      steps: 5000               # Optimization steps
      whitening_mode: "adam"    # Geometry preconditioning
      tensorboard: true         # Enable real-time diagnostics
```

**Enable Flow VI**:
```yaml
samplers:
  - name: vi
    overrides:
      sampler_flavour: "flow"  # Use Flow instead of MFA
      M: 8
      r: 2
```

---

## Core Concepts

### Evidence Lower Bound (ELBO)

VI maximizes the ELBO:
```
ELBO(q) = 𝔼_q[log p(w)] - 𝔼_q[log q(w)]
        = 𝔼_q[log p(w)] + H(q)
```

where:
- `p(w)` = tempered local posterior ∝ exp(-β L(w) - γ ||w - w*||²)
- `q(w)` = variational approximation
- `H(q)` = entropy of q

**Key property**: ELBO ≤ log Z (true log partition function)

### Localization

All VI methods localize around w* (ERM optimum) via:
```
log p(w) = -β L(w) - γ ||w - w*||² + const
```

where:
- β = inverse temperature (controls sampling temperature)
- γ = localizer strength (Gaussian tether around w*)
- L(w) = loss function (MSE, Gaussian likelihood, etc.)

**Tuning γ**:
- Try `{1e-4, 1e-3, 1e-2}`
- Smaller γ = wider posterior (more exploration)
- Larger γ = tighter posterior (stronger localization)
- Must be set via `posterior.gamma`, **not** `sampler.vi.gamma`

### Temperature β

Controls the sampling temperature:
- β = 1 / log(n) (default, `beta_mode: "1_over_log_n"`)
- β = 1 (full posterior, `beta_mode: "manual"`, `beta0: 1.0`)
- Larger β = sharper posterior (colder)
- Smaller β = smoother posterior (hotter)

---

## Configuration

### Shared VI Parameters

Common to all VI algorithms (configure under `sampler.vi`):

#### Core Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `M` | 8 | Number of mixture components (MFA) or latent dim (Flow) |
| `r` | 2 | Rank per component (MFA) or latent rank (Flow) |
| `steps` | 5000 | Optimization steps |
| `batch_size` | 256 | Minibatch size |
| `lr` | 0.01 | Learning rate (Adam optimizer) |
| `eval_every` | 50 | Trace recording frequency |
| `eval_samples` | 64 | MC samples for final LLC estimate |
| `dtype` | float32 | Precision (`float32` or `float64`) |

#### Whitening & Stability

| Parameter | Default | Description |
|-----------|---------|-------------|
| `whitening_mode` | `"none"` | Geometry whitening: `"none"` \| `"rmsprop"` \| `"adam"` |
| `whitening_decay` | 0.99 | EMA decay for gradient moment accumulation |
| `clip_global_norm` | 5.0 | Gradient clipping threshold (null to disable) |

#### Optimization

| Parameter | Default | Description |
|-----------|---------|-------------|
| `lr_schedule` | null | LR schedule: `"cosine"` \| `"linear_decay"` |
| `lr_warmup_frac` | 0.05 | Fraction of steps for warmup |

#### Diagnostics

| Parameter | Default | Description |
|-----------|---------|-------------|
| `tensorboard` | false | Enable TensorBoard logging |

### Posterior Parameters

Configure via `posterior` block (**not** `sampler.vi`):

| Parameter | Default | Description |
|-----------|---------|-------------|
| `gamma` | 0.001 | Localizer strength (Gaussian tether around w*) |
| `beta_mode` | `"1_over_log_n"` | Inverse temperature schedule |
| `beta0` | null | Manual β override (if beta_mode = `"manual"`) |
| `loss` | `"mse"` | Loss type: `"mse"` \| `"gaussian"` \| `"student"` |

---

## Running VI

### Single VI Run

```bash
# Build target first (if not already built)
uv run lambda-hat build --config-yaml config/experiments.yaml --target-id tgt_abc123

# Run VI sampler
uv run lambda-hat sample --config-yaml config/experiments.yaml --target-id tgt_abc123
```

### Via Workflow (Recommended)

```bash
# Run full workflow (builds targets + runs samplers)
uv run lambda-hat workflow llc --backend local

# With promotion (generate galleries)
uv run lambda-hat workflow llc --backend local  --promote
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
