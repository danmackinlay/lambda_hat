# Variational Inference (VI) Configuration and Implementation

Variational Inference provides a fast, scalable alternative to MCMC sampling for estimating the Local Learning Coefficient. VI approximates the tempered local posterior using a mixture of factor analyzers and optimizes the Evidence Lower Bound (ELBO).

## Implementation Details

The VI implementation is in `lambda_hat/variational.py` and integrated into `lambda_hat/sampling.py`. Key features:

- **Variational family**: Mixture of factor analyzers with equal means at $w^*$
- **Covariance structure**: $\Sigma_m = D + K_m K_m^T$ where $K_m = D^{1/2} A_m$ (algebraic whitening)
- **Shared diagonal $D$**: Ensures positive-definiteness and enables Woodbury identities for $O(Mdr)$ complexity
- **Low-rank factors**: Rank budget $r$ per component captures subspace anisotropy
- **Gradients**: STL (sticking-the-landing) for continuous parameters, Rao-Blackwellized for mixture weights
- **Precision**: Operates in `float32` by default with numerical stability features

## Usage

VI is configured via `config/experiments.yaml` using the sampler configuration system.

```yaml
# Example: Running VI with specific settings in config/experiments.yaml
samplers:
  - name: vi
    overrides:
      M: 8              # Number of mixture components
      r: 2              # Rank budget per component
      steps: 5000       # Optimization steps
      batch_size: 256   # Minibatch size
      lr: 0.01          # Learning rate (Adam optimizer)
      gamma: 0.001      # Localizer strength
      eval_every: 50    # Record traces every N steps
      eval_samples: 64  # MC samples for final estimate
```

Then execute:
```bash
uv run snakemake -j 4
```

## Configuration Options

### Core Parameters
- `M`: Number of mixture components (default: 8)
  - Start with 8; increase if underfitting (poor radius matching or high KL)
- `r`: Rank budget per component (default: 2)
  - Controls low-rank subspace dimensionality; start with 1-2, increase only if diagnostics suggest underfitting
- `steps`: Total optimization steps (default: 5000)
- `batch_size`: Minibatch size for gradient estimation (default: 256)
- `lr`: Learning rate for Adam optimizer (default: 0.01)

### Localization and Evaluation
- `gamma`: Localizer strength (default: 0.001)
  - Controls Gaussian tether around $w^*$: try $\gamma \in \{10^{-4}, 10^{-3}, 10^{-2}\}$
  - Pick smallest value with stable ELBO and radius
- `eval_every`: Frequency of trace recording (default: 50)
  - Records full-dataset loss every N steps
- `eval_samples`: MC samples for final LLC estimate (default: 64)

### Precision and Stability
- `dtype`: Precision (default: `float32`)
  - VI includes float32 stability features (D_sqrt clipping, ridge regularization, column normalization)

### Diagnostics (Stage 2)
- `tensorboard`: Enable TensorBoard logging (default: `false`)
  - Writes optimization metrics to `runs/.../diagnostics/tb/` for visualization
  - Logs 15+ scalar metrics every step (ELBO, radius, gradient norms, mixture stats, etc.)
  - Negligible overhead (~1% of runtime)
  - See "TensorBoard Integration" section for usage guide

### Geometry Whitening (Stage 1)
- `whitening_mode`: Whitening method (default: `"none"`)
  - Options: `"none"` (identity), `"rmsprop"` (second moment), `"adam"` (first + second moment)
  - Estimates diagonal preconditioner from minibatch gradients at $w^*$ before optimization
  - Helps reduce ELBO/radius spikes on anisotropic problems
- `whitening_decay`: EMA decay for gradient moment accumulation (default: 0.99)
  - Higher values (closer to 1.0) = slower adaptation, more stable estimates
  - Lower values = faster adaptation, noisier estimates
- `use_whitening`: Deprecated (use `whitening_mode` instead)

### Advanced Configuration (Stage 3)
- `r_per_component`: Per-component rank budgets (default: `None`)
  - Optional list of integers of length M specifying heterogeneous ranks
  - Example: `[1, 2, 2, 3]` for M=4 components with different ranks
  - Enables fine-grained control over model capacity per component
  - If `None`, all components use rank `r`
- `mixture_cap`: Upper bound on M (default: `None`)
  - Reserved for future pruning implementation
- `prune_threshold`: Component pruning threshold (default: 1e-3)
  - Drop mixture components with weight π < threshold
  - Reserved for future pruning implementation
- `alpha_dirichlet_prior`: Dirichlet prior on mixture weights (default: `None`)
  - Symmetric Dirichlet(α₀, ..., α₀) prior discourages collapse
  - Try α₀ ∈ {1.0, 2.0, 5.0}; larger values encourage uniformity
  - If `None`, no prior (maximum likelihood on π)
- `lr_schedule`: Learning rate schedule (default: `None`)
  - Options: `None` (constant), `"cosine"` (cosine decay with warmup), `"linear_decay"` (linear decay)
  - Cosine schedule often improves final convergence quality
  - Linear decay useful for finite-horizon optimization
- `lr_warmup_frac`: Fraction of steps for LR warmup (default: 0.05)
  - Used with `lr_schedule="cosine"` to gradually ramp up learning rate
  - Helps stabilize early optimization
- `entropy_bonus`: Entropy bonus λ (default: 0.0)
  - Adds λ * H(q) to ELBO to encourage exploration
  - Try λ ∈ {0.01, 0.1, 0.5} if mixture weights collapse
  - Higher values produce more uniform π distributions

## Mathematical Formulation

### Variational Family

The VI approximation uses a mixture of factor analyzers:

$$
q_\phi(w) = \sum_{m=1}^M \pi_m \mathcal{N}(w^*, \Sigma_m)
$$

where all components share mean $w^*$ (the ERM solution) and have low-rank plus diagonal covariance:

$$
\Sigma_m = D + K_m K_m^T, \quad K_m = D^{1/2} A_m
$$

- $D = \text{diag}(\text{softplus}(\rho))^2$: Shared diagonal across components
- $A_m \in \mathbb{R}^{d \times r}$: Low-rank factors in algebraically whitened form
- $\pi = \text{softmax}(\alpha)$: Mixture weights

### ELBO Objective

VI maximizes the local tempered ELBO:

$$
\mathcal{L}(\phi) = \mathbb{E}_{q_\phi}\left[-n\beta L_n(w) - \frac{\gamma}{2}\|w-w^*\|^2\right] + \mathsf{H}(q_\phi)
$$

where:
- $\beta = 1/\log n$: Inverse temperature (hot posterior)
- $\gamma$: Localizer strength
- $\mathsf{H}(q_\phi)$: Entropy of the variational distribution

### Gradient Estimators

**Continuous parameters** ($\rho$, $A$): STL pathwise gradients
- Reparameterize: $w = w^* + K_m z + D^{1/2} \varepsilon$ where $z \sim \mathcal{N}(0, I_r)$, $\varepsilon \sim \mathcal{N}(0, I_d)$
- Backprop only through $-n\beta L_n(w) - \frac{\gamma}{2}\|w-w^*\|^2$ (stop-grad for $\log q$)
- Unbiased with low variance

**Mixture weights** ($\alpha$): Rao-Blackwellized score gradient
- $g_{\alpha_j} = (\text{payoff} - \text{baseline}) (r_j(w) - \pi_j)$
- Uses responsibilities $r_j(w) = \pi_j p_j(w) / q(w)$ for variance reduction

## Numerical Stability (Float32)

VI includes several stability features for robust float32 operation:

1. **D_sqrt clipping**: Constrain $D^{1/2} \in [10^{-4}, 10^2]$ to prevent under/overflow
2. **Ridge regularization**: Cholesky on $C = I(1+\epsilon) + A^T A$ with enhanced ridge ($\epsilon = 10^{-5}$)
3. **Column normalization**: Clip factor columns to max norm 10.0 to prevent explosion
4. **Log stability**: Use $\log(D^{1/2} + \epsilon)$ instead of $\log(D^{1/2})$

### Stage 1 Stability Enhancements

Additional stability guards (from Stage 1 VI plan):

5. **Gradient clipping**: Clip global gradient norm via optax.clip_by_global_norm
   - Controlled by `clip_global_norm` config parameter (default: 5.0, null to disable)
   - Prevents gradient explosions during optimization
6. **Temperature-adjusted softmax**: Mixture weights use $\text{softmax}(\alpha / T)$ with temperature $T$
   - Controlled by `alpha_temperature` config parameter (default: 1.0)
   - Higher temperature (> 1.0) produces more uniform distributions (prevents component collapse)
   - Lower bound at $T = 0.5$ for numerical stability

These features ensure responsibilities stay in $[0,1]$ and prevent NaN/Inf propagation.

## Diagnostics

### Core Trace Metrics

VI traces include:
- `Ln`: Full-dataset loss (recorded every `eval_every` steps)
- `energy`: ELBO value (optimization objective)
- `acceptance_rate`: Always 1.0 (VI produces IID samples)
- `cumulative_fge`: Cumulative full-gradient equivalents (work tracking)

Analysis metrics (in `analysis.json`):
- `llc_mean`: LLC estimate $\hat{\lambda} = n\beta(\mathbb{E}_q[L_n] - L_n(w^*))$
- `ess`: Effective sample size (equals total draws for IID samples)
- `r_hat`: Gelman-Rubin diagnostic (NaN for IID samples)
- `wnv`: Work-normalized variance (efficiency metric)

### Enhanced Diagnostics (Stage 2)

VI also provides 15+ diagnostic metrics for monitoring optimization quality:

**ELBO components:**
- `elbo`: Total ELBO (objective value)
- `elbo_like`: Likelihood term only (target fitting)
- `logq`: Entropy term (variational distribution complexity)

**Geometry metrics:**
- `radius2`: Mean squared distance from $w^*$ in whitened coordinates
- `resp_entropy`: Responsibility entropy (detects component collapse)
- `cumulative_fge`: Cumulative full-gradient equivalents (work tracking)

**Control variate diagnostics:**
- `Eq_Ln_mc`: Raw MC estimate of $\mathbb{E}_q[L_n]$
- `Eq_Ln_cv`: Control-variate-corrected estimate
- `variance_reduction`: Variance reduction factor from CV

**Mixture statistics:**
- `pi_min`, `pi_max`, `pi_entropy`: Mixture weight distribution (detects collapse)

**Covariance diagnostics:**
- `D_sqrt_min`, `D_sqrt_max`, `D_sqrt_med`: Diagonal variance scaling

**Optimization dynamics:**
- `grad_norm`: Global gradient magnitude
- `A_col_norm_max`: Factor matrix column norms (tracks low-rank subspace)

All metrics are:
- Exported to ArviZ `sample_stats` for unified analysis
- Available for TensorBoard logging (opt-in, see below)
- Recorded every `eval_every` optimization steps

### TensorBoard Integration (Stage 2)

Enable real-time monitoring of VI optimization with TensorBoard:

**1. Enable TensorBoard in config:**
```yaml
samplers:
  - name: vi
    overrides:
      tensorboard: true  # Enable TensorBoard logging
      M: 8
      r: 2
      steps: 5000
      # ... other settings ...
```

**2. Run Snakemake workflow:**
```bash
uv run snakemake -j 4
```

**3. Launch TensorBoard:**
```bash
tensorboard --logdir runs/targets/TARGET_ID/run_vi_HASH/diagnostics/tb
```

**4. Open browser:**
Navigate to `http://localhost:6006` to view live metrics.

**What gets logged:**
- **Optimization progress**: ELBO, likelihood, entropy over training steps
- **Geometry evolution**: Radius, responsibility entropy, mixture statistics
- **Gradient dynamics**: Gradient norms, factor matrix growth
- **Work tracking**: Cumulative FGEs, variance reduction from control variates
- **Final estimates**: LLC, $\mathbb{E}_q[L_n]$, L_0 reference loss

**Interpreting TensorBoard plots:**

- **Converged ELBO**: Should plateau smoothly (no spikes or NaNs)
  - Spikes → reduce `lr` or enable whitening
  - Monotonic decrease → good (maximizing ELBO = minimizing negative ELBO)
- **Stable radius**: Should stabilize near expected local posterior scale
  - Growing radius → increase `gamma` (stronger localization)
  - Zero radius → decrease `gamma` (over-constrained)
- **Non-degenerate pi**: `pi_entropy` should stay > 0 (no component collapse)
  - Dropping entropy → reduce `M` or increase `alpha_temperature`
- **Gradient norms**: Should decay as optimization progresses
  - Exploding → enable `clip_global_norm` or reduce `lr`
- **Variance reduction > 1**: Control variate improves MC estimate quality
  - Higher is better (CV reducing variance effectively)

**Cost:** TensorBoard logging adds negligible overhead (~1% of runtime).

## Hyperparameter Tuning

### Choosing $\gamma$
- Start with $\gamma = 10^{-3}$
- Too small → unstable ELBO, divergent samples
- Too large → over-constrained, poor radius matching
- Monitor radius $\mathbb{E}_q\|w-w^*\|^2$ and ELBO stability

### Choosing $M$ and $r$
- Start with $M=8$, $r=1$
- Increase $r$ to 2-4 only if diagnostics suggest underfitting
- Larger $M$ helps with multimodality but increases cost ($O(Mdr)$ per step)

### Choosing `steps` and `lr`
- Default `steps=5000`, `lr=0.01` work well for most problems
- Monitor ELBO convergence; increase `steps` if not plateaued
- Reduce `lr` if ELBO is noisy/unstable

### When to Use Whitening

**Use `whitening_mode="none"` (default) when:**
- Problem has roughly isotropic gradients
- ELBO and radius traces are already smooth
- Debugging or first exploratory runs

**Use `whitening_mode="rmsprop"` when:**
- Seeing large spikes in ELBO or radius early in optimization
- Problem has anisotropic geometry (very different parameter scales)
- Gradients have high variance across dimensions

**Use `whitening_mode="adam"` when:**
- RMSProp whitening helps but still seeing instability
- Problem has both scale and directional anisotropy
- Extra cost (~500-1000 minibatch gradients) is acceptable

**How to tune `whitening_decay`:**
- Default 0.99 works well for most problems
- Increase to 0.995 for noisier gradients (smoother EMA)
- Decrease to 0.95 for very smooth gradients (faster adaptation)
- Must be in range [0.9, 0.999]

### Stage 3 Advanced Tuning

**Per-component ranks (`r_per_component`):**
- Use when different mixture components model different complexity
- Start with uniform ranks (default), increase selectively based on diagnostics
- Monitor `A_col_norm_max` to see which components need more capacity
- Example: `[1, 1, 2, 3]` gives higher-rank components more expressive power

**Dirichlet prior (`alpha_dirichlet_prior`):**
- Use when `pi_entropy` drops to near zero (component collapse)
- α₀ = 1.0: Uniform prior (no preference)
- α₀ > 1.0: Encourages uniformity (anti-collapse)
- α₀ < 1.0: Encourages sparsity (not recommended for VI)
- Monitor `pi_min` and `pi_entropy` to validate effectiveness

**LR schedules:**
- `"cosine"`: Best for long runs where you want gradual annealing
  - Use with `lr_warmup_frac=0.05` to stabilize early steps
  - Final LR decays to ~0, encouraging fine-grained convergence
- `"linear_decay"`: Simpler alternative, decays linearly to 0
  - Less commonly used than cosine
- Default (`None`): Constant LR, good for quick experiments

**Entropy bonus (`entropy_bonus`):**
- Use when mixture collapses to single component despite Dirichlet prior
- λ = 0.01: Mild exploration encouragement
- λ = 0.1: Moderate exploration (typical)
- λ = 0.5: Strong exploration (may hurt ELBO quality)
- Monitor `pi_entropy` to see if bonus is working

## Implementation Compliance

The VI implementation follows the design in `/plans/variational_inference.md`:

1. **Shared diagonal $D$**: Essential for PD, numerics, and Woodbury speed
2. **Algebraic whitening**: $K_m = D^{1/2} A_m$ stabilizes learning
3. **Geometry whitening**: ✓ Implemented in Stage 1 with RMSProp/Adam diagonal preconditioning
4. **Rank budget $r \geq 1$**: General implementation supports arbitrary rank
   - ✓ Stage 3 adds per-component rank budgets via masking
5. **STL + RB gradients**: Low-variance, scalable gradient estimators
   - ✓ Stage 3 adds Dirichlet prior and entropy bonus to mixture weight gradients
6. **Equal means**: All components centered at $w^*$ for strict locality
7. **Stability guards**: ✓ Stage 1 adds gradient clipping and temperature-adjusted softmax
8. **Advanced optimization**: ✓ Stage 3 adds LR schedules (cosine, linear decay)

## Comparison to MCMC

**Advantages of VI**:
- **Speed**: Near-SGD cost ($O(Mdr)$ per step vs full Hessian for HMC)
- **Scalability**: Minibatch gradients enable large datasets
- **No burn-in**: IID samples from converged $q$ require no warmup
- **Work efficiency**: Lower FGE count for comparable LLC accuracy

**Disadvantages of VI**:
- **Approximation error**: $q$ may not perfectly match true posterior $p$
- **Hyperparameter sensitivity**: Requires tuning $\gamma$, $M$, $r$
- **Bias**: Plug-in estimator incurs bias proportional to $\text{KL}(q \| p)$

**When to use VI**:
- Large networks ($d > 10^4$) where HMC is infeasible
- Quick exploratory runs to assess LLC magnitude
- Hyperparameter sweeps where speed matters

**When to use MCMC**:
- Small-to-medium networks where HMC is tractable
- Accuracy-critical applications requiring unbiased estimates
- Validation/ground-truth for VI results
