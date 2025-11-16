# Variational Inference (VI) for Local Learning Coefficient Estimation

Variational Inference provides a fast, scalable alternative to MCMC for estimating the Local Learning Coefficient. VI approximates the tempered local posterior using a mixture of factor analyzers and optimizes the Evidence Lower Bound (ELBO).

## Quick Reference

**Variational family**: Mixture of M factor analyzers, each with low-rank covariance centered at w* (ERM optimum).

**Key features**:
- STL (sticking-the-landing) + Rao-Blackwellized gradients
- HVP control variate for LLC variance reduction (evaluation-time only)
- Diagonal geometry whitening (RMSProp/Adam modes)
- Float32 numerical stability (clipping, ridge regularization, normalization)
- TensorBoard integration for real-time diagnostics

**Typical use**: Fast LLC estimates for model selection, hyperparameter sweeps, or initial exploration before running MCMC.

---

## Configuration

VI hyperparameters are controlled via `config/experiments.yaml` under `samplers[].overrides`. The localizer γ (gamma) comes from `posterior.gamma`, not from VI config.

### Core Parameters (sampler.vi)

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

### Whitening & Stability (sampler.vi)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `whitening_mode` | `"none"` | Geometry whitening: `"none"` \| `"rmsprop"` \| `"adam"` |
| `whitening_decay` | 0.99 | EMA decay for gradient moment accumulation |
| `clip_global_norm` | 5.0 | Gradient clipping threshold (null to disable) |
| `alpha_temperature` | 1.0 | Softmax temperature on mixture weights |

### Advanced Options (sampler.vi)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `r_per_component` | null | Heterogeneous rank budgets (list[int] of length M) |
| `alpha_dirichlet_prior` | null | Dirichlet(α₀) prior on mixture weights |
| `lr_schedule` | null | LR schedule: `"cosine"` \| `"linear_decay"` |
| `lr_warmup_frac` | 0.05 | Fraction of steps for warmup |
| `entropy_bonus` | 0.0 | Add λ * H(q) to ELBO for exploration |
| `tensorboard` | false | Enable TensorBoard logging |

###  Posterior Parameters (posterior)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `gamma` | 0.001 | Localizer strength (Gaussian tether around w*) |
| `beta_mode` | `"1_over_log_n"` | Inverse temperature schedule |
| `beta0` | null | Manual β override (if beta_mode = `"manual"`) |
| `loss` | `"mse"` | Loss type: `"mse"` \| `"gaussian"` \| `"student"` |

**Important**: γ (gamma) must be configured via `posterior.gamma`, not `sampler.vi.gamma`. See "Hyperparameter Experiments" below for examples.

---

## Hyperparameter Experiments: Model vs VI

There are three knob surfaces for exploring LLC behavior:

1. **Target model** (depth, width, activation, loss)
2. **Posterior temperature / localization** (`posterior.beta0`, `posterior.beta_mode`, `posterior.gamma`)
3. **VI family / optimizer** (`sampler.vi.*`)

All are driven from `config/experiments.yaml`.

### Example 1: Single model, single VI run

```yaml
store_root: "runs"
jax_enable_x64: true

targets:
  - { model: base, data: base, teacher: _null, seed: 43 }

samplers:
  - name: vi
    overrides:
      M: 8
      r: 2
      steps: 5000
      batch_size: 256
      lr: 0.01
      whitening_mode: "rmsprop"
      lr_schedule: "cosine"
      lr_warmup_frac: 0.05
      entropy_bonus: 0.1
    seed: 54321

posterior:
  gamma: 0.001  # Localizer strength
  beta_mode: "1_over_log_n"
  loss: "mse"
```

**Commands**:
```bash
uv run lambda-hat workflow llc --local
```

**Results**: `runs/targets/<tgt>/run_vi_<rid>/analysis.json` contains `llc_mean`, `llc_std`, `ess`, `wnv`.

### Example 2: Sweeping model hyperparameters with fixed VI

```yaml
targets:
  - { model: small, data: base, teacher: _null, seed: 41 }
  - { model: base,  data: base, teacher: _null, seed: 42 }
  - model: base
    data: base
    teacher: _null
    seed: 43
    overrides:
      model: { depth: 6, target_params: 20000, activation: tanh }

samplers:
  - name: vi
    overrides: { M: 8, r: 2, whitening_mode: "rmsprop" }
    seed: 54321

posterior:
  gamma: 0.001
```

**Workflow**: N targets × 1 sampler → N VI runs with same VI settings, different architectures.

### Example 3: Sweeping VI hyperparameters for a fixed target

```yaml
targets:
  - { model: base, data: base, teacher: _null, seed: 43 }

samplers:
  # Baseline
  - name: vi
    overrides: { M: 4, r: 1, whitening_mode: "none", lr: 0.01 }
    seed: 1001

  # More expressive mixture
  - name: vi
    overrides: { M: 8, r: 2, whitening_mode: "rmsprop", lr: 0.01 }
    seed: 1002

  # Aggressive geometry + entropy bonus
  - name: vi
    overrides:
      M: 8
      r: 2
      whitening_mode: "adam"
      lr: 0.005
      entropy_bonus: 0.1
      lr_schedule: "cosine"
      lr_warmup_frac: 0.1
    seed: 1003

posterior:
  gamma: 0.001
```

**Workflow**: 1 target × M samplers → M independent VI runs. Compare `analysis.json` outputs (llc_mean vs total_fge / elapsed_time) to evaluate VI hyperparameter sensitivity.

### Example 4: Sweeping localizer γ (gamma)

To vary γ per VI run, you must create separate config files with different `posterior.gamma` values:

**`config/exp_gamma_small.yaml`**:
```yaml
targets:
  - { model: base, data: base, teacher: _null, seed: 43 }
samplers:
  - { name: vi, overrides: { M: 8, r: 2 }, seed: 54321 }
posterior:
  gamma: 0.0001  # Small localizer
```

**`config/exp_gamma_large.yaml`**:
```yaml
targets:
  - { model: base, data: base, teacher: _null, seed: 43 }
samplers:
  - { name: vi, overrides: { M: 8, r: 2 }, seed: 54321 }
posterior:
  gamma: 0.01  # Large localizer
```

**Commands**:
```bash
uv run lambda-hat workflow llc --local --config config/exp_gamma_small.yaml
uv run lambda-hat workflow llc --local --config config/exp_gamma_large.yaml
```

---

## TensorBoard Integration

Enable real-time optimization monitoring:

```yaml
samplers:
  - name: vi
    overrides:
      tensorboard: true
      M: 8
      r: 2
    seed: 54321
```

**Launch TensorBoard**:
```bash
tensorboard --logdir runs/targets/<tgt>/run_vi_<rid>/diagnostics/tb
```

**Key metrics**:
- `vi/elbo`, `vi/elbo_like`, `vi/logq`: ELBO decomposition
- `vi/radius2`: Localization radius (quantiles + mean)
- `vi/pi_entropy`, `vi/pi_min`, `vi/pi_max`: Mixture weight health
- `vi/grad_norm`, `vi/A_col_norm_max`: Gradient diagnostics
- `vi/D_sqrt_min/max/med`: Diagonal variance spread
- `vi/cumulative_fge`: Work-normalized progress

**Interpreting**:
- **Low `pi_entropy` / high `pi_max`**: Mixture collapse → add `entropy_bonus` or `alpha_dirichlet_prior`
- **Spiking `grad_norm`**: Reduce `lr` or increase `clip_global_norm`
- **Flat `elbo`**: Increase `M`, `r`, or switch to `lr_schedule: "cosine"`

---

## Tuning Guide

### Start point (robust defaults)
- `whitening_mode: adam`, `clip_global_norm: 5.0`
- `M: 8`, `r: 2`
- `lr: 0.01`, `lr_schedule: "linear_decay"`, `lr_warmup_frac: 0.05`
- `posterior.gamma: 0.001`

### Capacity tuning
1. **Underfitting** (poor radius matching, high KL): Increase `r` first (cheapest), then `M`
2. **Mixture collapse** (low `pi_entropy`): Add `entropy_bonus: 0.1` or `alpha_dirichlet_prior: 2.0`
3. **Noisy optimization**: Keep `whitening_mode: adam`, try `lr_schedule: "cosine"` with larger warmup

### Gamma (γ) tuning
- γ controls localization tightness around w*
- Try `{1e-4, 1e-3, 1e-2}`; pick smallest value with stable ELBO and radius
- Must be set via `posterior.gamma`, not `sampler.vi.gamma`

---

## Implementation Notes

**Where HVP whitening is used**:
- HVP-diagonal whitening is **not** implemented for the optimization loop (JIT stability)
- HVP is used **only at evaluation time** for the control variate (variance reduction)
- Optimization uses RMSProp/Adam diagonal whitening (`whitening_mode: rmsprop|adam`)

**Precision**:
- VI operates in `float32` by default with numerical stability features
- Can switch to `float64` via `dtype: float64` (slower, rarely needed)

**Work tracking**:
- VI reports `total_fge` (function-gradient evaluations) for fair comparison with MCMC
- 1 VI step = 1 FGE (minibatch gradient)
- Whitening pre-pass cost included in `total_fge`

---

## See Also

- `docs/configuration.md` - Full experiments.yaml schema and override patterns
- `docs/sweeps.md` - Advanced sweep patterns (mixed precision, ablations)
- `docs/output_management.md` - Artifact layout and content-addressed IDs
