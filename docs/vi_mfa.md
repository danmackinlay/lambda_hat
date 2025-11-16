# MFA VI (Mixture of Factor Analyzers)

MFA approximates the tempered local posterior using a mixture of low-rank Gaussians centered at w* (ERM optimum).

For general VI concepts and usage, see [Variational Inference](./vi.md).
For Flow VI, see [Flow VI](./vi_flow.md).

---

## Overview

**Default VI algorithm.** Provides a good balance of expressiveness and computational efficiency for LLC estimation.

**Variational family**: M components, each with rank-r covariance:
```
q(w) = Σᵢ πᵢ 𝒩(w | μᵢ, Dᵢ + UᵢUᵢᵀ)
```

where:
- M = number of mixture components
- r = rank of low-rank factor per component
- μᵢ centered at w* (ERM optimum)
- Dᵢ = diagonal covariance
- UᵢUᵢᵀ = low-rank factor (r-dimensional)

---

## Key Features

### STL (Sticking-the-Landing) Gradients
- Detaches gradient flow through samples for entropy term
- Reduces variance significantly as q → p
- Unbiased gradient estimator

### Rao-Blackwellized Gradients
- Analytically integrates out certain variables
- Reduces MC variance in ELBO gradient estimates
- Particularly effective for mixture weights

### Diagonal Geometry Whitening
- RMSProp or Adam preconditioning for parameter updates
- Adapts to local geometry without full HVP computation
- Modes: `"none"` | `"rmsprop"` | `"adam"`

### HVP Control Variate
- Used **only at evaluation time** for LLC variance reduction
- Not used during optimization (JIT stability)
- Constructs quadratic approximation around w* for variance reduction

---

## Configuration

MFA-specific parameters (in addition to [shared VI config](./vi.md#configuration)):

### Capacity Control

| Parameter | Default | Description |
|-----------|---------|-------------|
| `M` | 8 | Number of mixture components |
| `r` | 2 | Rank budget per component |
| `r_per_component` | null | Heterogeneous rank budgets (list[int] of length M) |

### Mixture Regularization

| Parameter | Default | Description |
|-----------|---------|-------------|
| `alpha_temperature` | 1.0 | Softmax temperature on mixture weights |
| `alpha_dirichlet_prior` | null | Dirichlet(α₀) prior on mixture weights |
| `entropy_bonus` | 0.0 | Add λ * H(q) to ELBO for exploration |

---

## Tuning Guide

### Starting Point (Robust Defaults)

```yaml
samplers:
  - name: vi
    overrides:
      M: 8
      r: 2
      whitening_mode: "adam"
      clip_global_norm: 5.0
      lr: 0.01
      lr_schedule: "linear_decay"
      lr_warmup_frac: 0.05

posterior:
  gamma: 0.001
```

### Capacity Tuning

**Underfitting** (poor radius matching, high KL divergence):
1. Increase `r` first (cheapest: 2 → 4 → 8)
2. If still insufficient, increase `M` (4 → 8 → 16)
3. Consider heterogeneous ranks: `r_per_component: [8, 4, 4, 2, 2, 1, 1, 1]`

**Overfitting** (numerical issues, training instability):
1. Decrease `r` or `M`
2. Increase `clip_global_norm`
3. Add regularization via `alpha_dirichlet_prior`

### Mixture Health

**Mixture collapse** (low `pi_entropy`, high `pi_max` in TensorBoard):

Symptoms:
- Only 1-2 components have significant weight
- `pi_entropy` drops below log(M/2)
- Some components never get used

Fixes:
1. Add `entropy_bonus: 0.1` (encourages exploration)
2. Use `alpha_dirichlet_prior: 2.0` (symmetric Dirichlet prior)
3. Increase `alpha_temperature: 1.5` (softer mixture selection)
4. Reduce M (fewer components, less collapse risk)

### Optimization Stability

**Noisy gradients** (spiking `grad_norm` in TensorBoard):
1. Keep `whitening_mode: "adam"` (most stable)
2. Reduce `lr` (0.01 → 0.005 → 0.001)
3. Increase `clip_global_norm` (5.0 → 10.0)
4. Use `lr_schedule: "cosine"` with larger warmup

**Slow convergence** (flat `elbo` curve):
1. Increase capacity (M or r)
2. Try `lr_schedule: "cosine"`
3. Check if γ (gamma) is too large (try smaller values)

---

## Hyperparameter Examples

### Example 1: Basic MFA Run

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

### Example 2: MFA Capacity Sweep

```yaml
targets:
  - { model: base, data: base, teacher: _null, seed: 43 }

samplers:
  # Low capacity
  - name: vi
    overrides: { M: 4, r: 1, whitening_mode: "none" }
    seed: 1001

  # Medium capacity (default)
  - name: vi
    overrides: { M: 8, r: 2, whitening_mode: "rmsprop" }
    seed: 1002

  # High capacity + regularization
  - name: vi
    overrides:
      M: 16
      r: 4
      whitening_mode: "adam"
      entropy_bonus: 0.1
      alpha_dirichlet_prior: 2.0
    seed: 1003

posterior:
  gamma: 0.001
```

### Example 3: Mixture Collapse Prevention

If you observe mixture collapse in initial runs:

```yaml
samplers:
  - name: vi
    overrides:
      M: 8
      r: 2
      whitening_mode: "adam"
      entropy_bonus: 0.1              # Encourage exploration
      alpha_dirichlet_prior: 2.0      # Symmetric prior
      alpha_temperature: 1.2          # Softer selection
      lr_schedule: "cosine"
      lr_warmup_frac: 0.1             # Longer warmup

posterior:
  gamma: 0.001
```

---

## TensorBoard Diagnostics

MFA-specific metrics (see [VI Overview](./vi.md#tensorboard-monitoring) for general VI metrics):

**Mixture health**:
- `vi/pi_entropy` — Mixture weight entropy (should be ≈ log(M) for healthy mixture)
- `vi/pi_min` — Minimum mixture weight (watch for near-zero components)
- `vi/pi_max` — Maximum mixture weight (watch for collapse to single component)

**Interpretation**:
- Healthy: `pi_entropy ≈ log(M)`, `pi_max ≈ 1/M`, all components used
- Collapse: `pi_entropy ≪ log(M)`, `pi_max → 1`, only few components active

**Component-level diagnostics**:
- `vi/A_col_norm_max` — Maximum column norm in factor matrix
- `vi/D_sqrt_min/max/med` — Diagonal variance spread
- `vi/radius2` — Per-component localization radius

---

## Implementation Notes

### MFA-Specific Details

**Mixture weights**:
- Optimized in simplex via softmax parameterization
- Temperature parameter allows soft vs hard selection
- Optional Dirichlet prior for regularization

**Low-rank factors**:
- Parameterized as unconstrained matrices U ∈ ℝ^(d×r)
- Covariance: Σᵢ = Dᵢ + UᵢUᵢᵀ (diagonal + low-rank)
- Diagonal D ensures full-rank even if U is degenerate

**Localization**:
- All component means μᵢ are offsets from w*
- Localization prior γ applies to offset magnitudes
- Prevents components from drifting too far from MAP

### Computational Cost

**Per VI step**:
- O(M × r × d) for low-rank updates
- O(M × d) for diagonal updates
- O(M × batch_size × d) for likelihood gradients

**Comparison to MCMC**:
- ~10-100× faster than HMC for similar LLC accuracy
- Suitable for initial exploration and hyperparameter sweeps
- Use HMC for high-accuracy gold standard

---

## When to Use MFA VI

**Good fit**:
- Fast LLC estimates for model selection
- Hyperparameter sweeps (many configurations)
- Initial exploration before running MCMC
- Unimodal or near-unimodal posteriors
- Problems where capacity M ≤ 16, r ≤ 8 is sufficient

**Poor fit**:
- Highly multimodal posteriors (consider Flow VI)
- Need for very high accuracy (use HMC instead)
- Posterior has long-range correlations MFA can't capture

---

## See Also

- [Variational Inference Overview](./vi.md) — General VI concepts and shared config
- [Flow VI](./vi_flow.md) — Alternative VI algorithm using normalizing flows
- [Configuration Reference](./config.md) — Complete YAML schema
- [Samplers](./samplers.md) — Comparison with MCMC methods
