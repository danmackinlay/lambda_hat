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

### Model (`model.*`)

Controls the architecture of the student MLP.

- `model.depth`: Number of hidden layers.
- `model.target_params`: Approximate total number of parameters (widths are inferred).
- `model.activation`: Activation function (e.g., `relu`, `tanh`).

**Example:**
```bash
python train.py model.depth=4 model.target_params=5000
```

### Data (`data.*`)

Controls the synthetic data generation process.

- `data.n_data`: Number of data points.
- `data.x_dist`: Input distribution (e.g., `gauss_iso`).
- `data.noise_model`: Noise model (e.g., `gauss`, `student_t`).

**Example:**

```bash
python train.py data.n_data=10000 data.noise_model=student_t
```

### Sampler (`sampler.*`)

Controls the parameters for the MCMC samplers.

- `sampler.chains`: Number of parallel chains.
- `sampler.hmc.draws`: Number of HMC draws.
- `sampler.sgld.steps`: Total number of SGLD steps.

**Example:**

```bash
python train.py sampler.chains=8 sampler.hmc.draws=5000
```