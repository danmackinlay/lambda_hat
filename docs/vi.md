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

### Precision
- `dtype`: Precision (default: `float32`)
  - VI includes float32 stability features (D_sqrt clipping, ridge regularization, column normalization)
- `use_whitening`: Enable geometry whitening (default: true)
  - Currently uses identity; future support for Adam/RMSProp-based whitening

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
2. **Ridge regularization**: Cholesky on $C = I(1+\epsilon) + A^T A$ instead of $I + A^T A$
3. **Column normalization**: Clip factor columns to max norm 10.0 to prevent explosion
4. **Log stability**: Use $\log(D^{1/2} + \epsilon)$ instead of $\log(D^{1/2})$

These features ensure responsibilities stay in $[0,1]$ and prevent NaN/Inf propagation.

## Diagnostics

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

## Implementation Compliance

The VI implementation follows the design in `/plans/variational_inference.md`:

1. **Shared diagonal $D$**: Essential for PD, numerics, and Woodbury speed
2. **Algebraic whitening**: $K_m = D^{1/2} A_m$ stabilizes learning
3. **Geometry whitening**: Infrastructure ready for Adam/RMSProp-based preconditioning
4. **Rank budget $r \geq 1$**: General implementation supports arbitrary rank
5. **STL + RB gradients**: Low-variance, scalable gradient estimators
6. **Equal means**: All components centered at $w^*$ for strict locality

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
