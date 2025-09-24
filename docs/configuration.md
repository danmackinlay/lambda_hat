# Configuration Details

This project uses Hydra for configuration. The configuration is structured and composable.

## Structure

The configuration is split into groups, defined in `conf/`:

- `config.yaml`: Main configuration and defaults.
- `model/`: Model architecture settings.
- `data/`: Data generation settings.
- `sampler/`: Sampler settings (HMC, SGLD, MCLMC parameters).

You can create new presets by adding YAML files to these directories.

## Key Configuration Groups

### Target (`target`)

Controls the type of target function for LLC estimation.

- `target=mlp`: Multi-Layer Perceptron (default) using Haiku
- `target=quadratic`: Simple quadratic target L_n(θ) = 0.5||θ||² for testing

**Example:**
```bash
python train.py target=quadratic quad_dim=4
```

### Model (`model.*`)

Controls the architecture of the student MLP (when `target=mlp`).

- `model.depth`: Number of hidden layers.
- `model.target_params`: Approximate total number of parameters (widths are inferred).
- `model.activation`: Activation function (e.g., `relu`, `tanh`, `gelu`, `identity` for deep-linear).

**Example:**
```bash
python train.py model.depth=4 model.target_params=5000 model.activation=identity
```

### Data (`data.*`)

Controls the synthetic data generation process using a teacher-student setup.

- `data.n_data`: Number of data points.
- `data.x_dist`: Input distribution:
  - `gauss_iso`: Isotropic Gaussian
  - `gauss_aniso`: Anisotropic Gaussian
  - `mixture`: Mixture of Gaussians
  - `lowdim_manifold`: Low-dimensional manifold
  - `heavy_tail`: Heavy-tailed distribution
- `data.noise_model`: Noise model:
  - `gauss`: Gaussian noise
  - `hetero`: Heteroscedastic noise
  - `student_t`: Student-t noise
  - `outliers`: Noise with outliers

**Example:**
```bash
python train.py data.n_data=10000 data.noise_model=student_t data.x_dist=mixture
```

### Training (`training.*`)

Controls the ERM (Empirical Risk Minimization) training process.

- `training.optimizer`: Optimizer (default: `adam`)
- `training.learning_rate`: Learning rate for optimization
- `training.erm_steps`: Number of training steps

### Posterior (`posterior.*`)

Controls the posterior configuration for LLC estimation.

- `posterior.loss`: Loss function (e.g., `mse`)
- `posterior.beta_mode`: Temperature schedule (e.g., `1_over_log_n`)
- `posterior.beta0`: Base temperature parameter
- `posterior.gamma`: Localization strength around w*

### Sampler (`sampler.*`)

Controls the parameters for the MCMC samplers.

- `sampler.chains`: Number of parallel chains.
- `sampler.hmc.draws`: Number of HMC draws.
- `sampler.hmc.warmup`: Number of HMC warmup steps.
- `sampler.sgld.steps`: Total number of SGLD steps.
- `sampler.sgld.step_size`: SGLD step size.
- `sampler.sgld.batch_size`: Minibatch size for SGLD.
- `sampler.mclmc.draws`: Number of MCLMC draws.

**Example:**
```bash
python train.py sampler.chains=8 sampler.hmc.draws=5000 sampler.sgld.step_size=1e-6
```