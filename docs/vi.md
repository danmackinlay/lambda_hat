# Variational Inference

Variational Inference (VI) provides a fast, scalable alternative to MCMC for estimating the Local Learning Coefficient.

---

## Quick Start

**At a glance**:
- Fast LLC estimates for model selection and hyperparameter sweeps
- Two algorithms: **MFA** (default) or **Flow** (requires `--extra flowvi`)
- STL gradients + Rao-Blackwellized control variates for variance reduction
- Float32 stability with optional TensorBoard monitoring

**Run VI**:
```bash
# Simple usage (uses defaults from config)
uv run lambda-hat workflow llc --local

# With custom VI settings in config/experiments.yaml
samplers:
  - name: vi
    overrides:
      M: 8
      r: 2
      steps: 5000
      whitening_mode: "adam"
```

---

## Algorithms

### MFA (Mixture of Factor Analyzers)

**Default VI algorithm.** Approximates the tempered local posterior using a mixture of low-rank Gaussians centered at w* (ERM optimum).

**Variational family**: M components, each with rank-r covariance.

**Features**:
- STL (sticking-the-landing) gradients
- Rao-Blackwellized gradients for reduced variance
- Diagonal geometry whitening (RMSProp/Adam modes)
- HVP control variate for LLC variance reduction (evaluation-time only)

**When to use**: Default choice for VI; robust and well-tested.

---

### Flow (Normalizing Flows)

**Experimental.** Uses normalizing flows via manifold-plus-noise construction.

**Requirements**: `uv sync --extra flowvi`

**Architectures**:
- RealNVP coupling flow (default)
- MAF (Masked Autoregressive Flow)
- NSF (Neural Spline Flow)

**Features**:
- Low-rank latent space with orthogonal noise
- Vmap-compatible (PRNG key issues resolved Nov 2025)
- More expressive than MFA for complex posteriors

**Current limitations**:
- HVP control variate deferred to future work
- More experimental than MFA

**When to use**: When MFA capacity is insufficient or you suspect multimodal posteriors.

**Configuration**:
```yaml
samplers:
  - name: vi
    overrides:
      sampler_flavour: "flow"  # Enable flow VI
      M: 8                     # Number of flow components
      r: 2                     # Latent rank
```

See `docs/vi_normalizing_flow.md` for detailed flow theory and implementation notes.

---

## Configuration

VI parameters live under `sampler.vi` in `config/experiments.yaml`. The localizer γ (gamma) comes from `posterior.gamma`, **not** from VI config.

### Core Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `M` | 8 | Number of mixture components |
| `r` | 2 | Rank budget per component |
| `steps` | 5000 | Optimization steps |
| `batch_size` | 256 | Minibatch size |
| `lr` | 0.01 | Learning rate (Adam optimizer) |
| `eval_every` | 50 | Trace recording frequency |
| `eval_samples` | 64 | MC samples for final LLC estimate |
| `dtype` | float32 | Precision (`float32` or `float64`) |

### Whitening & Stability

| Parameter | Default | Description |
|-----------|---------|-------------|
| `whitening_mode` | `"none"` | Geometry whitening: `"none"` \| `"rmsprop"` \| `"adam"` |
| `whitening_decay` | 0.99 | EMA decay for gradient moment accumulation |
| `clip_global_norm` | 5.0 | Gradient clipping threshold (null to disable) |
| `alpha_temperature` | 1.0 | Softmax temperature on mixture weights |

### Advanced Options

| Parameter | Default | Description |
|-----------|---------|-------------|
| `r_per_component` | null | Heterogeneous rank budgets (list[int] of length M) |
| `alpha_dirichlet_prior` | null | Dirichlet(α₀) prior on mixture weights |
| `lr_schedule` | null | LR schedule: `"cosine"` \| `"linear_decay"` |
| `lr_warmup_frac` | 0.05 | Fraction of steps for warmup |
| `entropy_bonus` | 0.0 | Add λ * H(q) to ELBO for exploration |
| `tensorboard` | false | Enable TensorBoard logging |

### Posterior Parameters

Configure via `posterior` (not `sampler.vi`):

| Parameter | Default | Description |
|-----------|---------|-------------|
| `gamma` | 0.001 | Localizer strength (Gaussian tether around w*) |
| `beta_mode` | `"1_over_log_n"` | Inverse temperature schedule |
| `beta0` | null | Manual β override (if beta_mode = `"manual"`) |
| `loss` | `"mse"` | Loss type: `"mse"` \| `"gaussian"` \| `"student"` |

---

## Hyperparameter Examples

### Example 1: Basic VI run

```yaml
targets:
  - { model: base, data: base, teacher: _null, seed: 43 }

samplers:
  - name: vi
    overrides:
      M: 8
      r: 2
      whitening_mode: "rmsprop"
      lr_schedule: "cosine"

posterior:
  gamma: 0.001
```

### Example 2: VI hyperparameter sweep

```yaml
targets:
  - { model: base, data: base, teacher: _null, seed: 43 }

samplers:
  # Baseline
  - name: vi
    overrides: { M: 4, r: 1, whitening_mode: "none", lr: 0.01 }
    seed: 1001

  # More expressive
  - name: vi
    overrides: { M: 8, r: 2, whitening_mode: "rmsprop", lr: 0.01 }
    seed: 1002

  # Aggressive geometry + entropy
  - name: vi
    overrides:
      M: 8
      r: 2
      whitening_mode: "adam"
      entropy_bonus: 0.1
      lr_schedule: "cosine"
    seed: 1003

posterior:
  gamma: 0.001
```

---

## TensorBoard Monitoring

Enable real-time optimization diagnostics:

```yaml
samplers:
  - name: vi
    overrides:
      tensorboard: true
      M: 8
      r: 2
```

**Launch TensorBoard**:
```bash
tensorboard --logdir runs/targets/<tgt>/run_vi_<rid>/diagnostics/tb
```

**Key metrics**:
- `vi/elbo`, `vi/elbo_like`, `vi/logq`: ELBO decomposition
- `vi/radius2`: Localization radius (quantiles + mean)
- `vi/pi_entropy`, `vi/pi_min`, `vi/pi_max`: Mixture weight health
- `vi/grad_norm`: Gradient diagnostics
- `vi/cumulative_fge`: Work-normalized progress

**Diagnosis**:
- **Low `pi_entropy` / high `pi_max`**: Mixture collapse → add `entropy_bonus` or `alpha_dirichlet_prior`
- **Spiking `grad_norm`**: Reduce `lr` or increase `clip_global_norm`
- **Flat `elbo`**: Increase `M`, `r`, or switch to `lr_schedule: "cosine"`

---

## Tuning Guide

### Robust starting point

```yaml
whitening_mode: "adam"
clip_global_norm: 5.0
M: 8
r: 2
lr: 0.01
lr_schedule: "linear_decay"
lr_warmup_frac: 0.05
posterior:
  gamma: 0.001
```

### Capacity tuning

1. **Underfitting** (poor radius matching, high KL): Increase `r` first (cheapest), then `M`
2. **Mixture collapse** (low `pi_entropy`): Add `entropy_bonus: 0.1` or `alpha_dirichlet_prior: 2.0`
3. **Noisy optimization**: Keep `whitening_mode: adam`, try `lr_schedule: "cosine"` with larger warmup

### Gamma (γ) tuning

- γ controls localization tightness around w*
- Try `{1e-4, 1e-3, 1e-2}`; pick smallest value with stable ELBO and radius
- **Must** be set via `posterior.gamma`, not `sampler.vi.gamma`

---

## Implementation Notes

### Work Tracking

VI reports `total_fge` (function-gradient evaluations) for fair comparison with MCMC:
- 1 VI step = 1 FGE (minibatch gradient)
- Whitening pre-pass cost included in `total_fge`
- Cumulative FGE: `batch_size / n_data` per step (minibatch accounting)

### Precision

- VI operates in `float32` by default with numerical stability features
- Can switch to `float64` via `dtype: float64` (slower, rarely needed)

### HVP Usage

- HVP-diagonal whitening is **not** implemented for the optimization loop (JIT stability)
- HVP is used **only at evaluation time** for the control variate (variance reduction)
- Optimization uses RMSProp/Adam diagonal whitening (`whitening_mode: rmsprop|adam`)

---

## See Also

- [Configuration Reference](./config.md) — Complete YAML schema and defaults
- [Samplers](./samplers.md) — Comparison with MCMC samplers
- [Workflows](./workflows.md) — Running VI in sweeps with Parsl
- `docs/vi_mfa.md` — Detailed MFA implementation notes
- `docs/vi_normalizing_flow.md` — Detailed flow theory and JAX/vmap learnings
