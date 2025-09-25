# Target Functions and Data Generators

## Target Types

The system supports different target functions:

### Neural Network Targets (Default)

```bash
uv run python -m lambda_hat run --preset=quick
```

**Model Configuration:**
- Configurable depth, widths, activation (ReLU, tanh, GeLU, identity for deep-linear)
- Small but non-trivial MLP model (~10k parameters by default)

**Data Generation:**
- Teacher–student data generator with parametric input distributions:
  - Isotropic Gaussian
  - Anisotropic Gaussian
  - Mixture of Gaussians
  - Low-dimensional manifolds
  - Heavy-tailed distributions

**Noise Models:**
- Gaussian noise
- Heteroscedastic noise
- Student-t noise
- Outliers

### Analytical Quadratic Target

```bash
uv run python -m lambda_hat run --target=quadratic --quad-dim=4
```

The quadratic target uses L_n(θ) = 0.5||θ||² for testing sampler correctness and factor-of-2 bug detection.

## Training Process

1. **ERM Training:** SGD to locate the empirical minimizer w*
2. **Local Gaussian Prior:** Centered at w*
3. **Tempered Posterior:** β = β₀/log n by default
4. **Gaussian Localization:** γ = d / r² if `prior_radius` given

This follows the standard approach for LLC estimation in Singular Learning Theory.