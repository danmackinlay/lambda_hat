# Target Functions and Data Generators

## Target Types

The system supports different target functions, defined by the `target` configuration parameter.

### Neural Network Targets (Default)

The default target is a Multi-Layer Perceptron (MLP) defined using Haiku (`target=mlp`).

```bash
# Run with default configuration
python train.py
```

**Model Configuration:**
The model architecture is defined under the `model` configuration group.

- Configurable `depth`, `widths`, `activation` (ReLU, tanh, GeLU, identity for deep-linear).
- The `target_params` setting allows inferring the widths to achieve a specific model size (e.g., ~10k parameters by default).

```bash
# Run with a small, deep linear network
python train.py model.target_params=500 model.depth=5 model.activation=identity
```

**Data Generation:**
Data is generated using a teacher–student setup, defined under the `data` configuration group.

- Parametric input distributions (`data.x_dist`):
  - `gauss_iso` (Isotropic Gaussian)
  - `gauss_aniso`, `mixture`, `lowdim_manifold`, `heavy_tail`.

**Noise Models:**

- Configurable noise (`data.noise_model`):
  - `gauss`, `hetero`, `student_t`, `outliers`.

### Analytical Quadratic Target

A simple quadratic target (L_n(θ) = 0.5||θ||²) is available for testing sampler correctness and debugging.

```bash
# Run with quadratic target
# Note: The dimension is controlled by quad_dim
python train.py target=quadratic quad_dim=4
```

## Training Process

The pipeline follows the standard approach for LLC estimation in Singular Learning Theory:

1. **ERM Training:** Optimization (Adam by default) is used to locate the empirical risk minimizer (w*). Configured under the `training` group.
2. **Local Gaussian Prior:** A Gaussian prior is centered at w*.
3. **Tempered Posterior:** The posterior is tempered, typically with β = β₀/log n. Configured under the `posterior` group.
4. **Gaussian Localization:** Gamma (γ) controls the strength of the localization around w*. Configured under the `posterior` group.