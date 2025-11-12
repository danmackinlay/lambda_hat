# Configuration Details

This project uses **OmegaConf** to parse YAML configuration files, orchestrated by **Snakemake**.
Configs are merged manually in the `Snakefile`.

## How Configs Are Composed

- **Presets** live under `lambda_hat/conf/` (e.g., `model/base.yaml`, `data/small.yaml`).
- The **`Snakefile`** merges presets to build two types of configs:
  - **Stage A** (target build): `model` + `data` + `teacher` + `training`
  - **Stage B** (sampling run): `sampler` + `posterior` + target reference
- **User experiments** are defined in `config/experiments.yaml`

## User-Facing Experiments

Define parameter sweeps in `config/experiments.yaml`:

```yaml
store_root: "runs"
jax_enable_x64: true

targets:
  - { model: small, data: small, teacher: _null, seed: 42 }
  - { model: base,  data: base,  teacher: _null, seed: 43 }
  - { model: wide,  data: large, teacher: _null, seed: 44,
      overrides: { training: { steps: 10000 } } }

samplers:
  - { name: hmc }
  - { name: sgld, overrides: { step_size: 1e-6, eval_every: 50 }, seed: 12345 }
  - { name: mclmc, overrides: { draws: 5000 } }
  - { name: vi, overrides: { M: 8, r: 2, gamma: 0.001 }, seed: 54321 }
```

Then run the full pipeline:

```bash
# View the DAG
uv run snakemake -n

# Execute locally with 4 cores
uv run snakemake -j 4

# Execute on HPC with a profile
uv run snakemake --profile slurm -j 100
```

## Configuration Groups

### Model Presets (`lambda_hat/conf/model/`)

Controls neural network architecture for target building.

Available presets:
- `small.yaml`: Small networks for quick testing
- `base.yaml`: Standard architecture
- `wide.yaml`: Wider networks (if exists)

Key parameters:
- `depth`: Number of hidden layers
- `target_params`: Approximate total number of parameters (widths are inferred)
- `activation`: Activation function (`relu`, `tanh`, `gelu`, `identity`)
- `bias`: Whether to use bias terms

**Example new preset** (`lambda_hat/conf/model/deep.yaml`):
```yaml
depth: 6
target_params: 50000
activation: relu
bias: true
```

### Data Presets (`lambda_hat/conf/data/`)

Controls synthetic data generation using teacher-student setup.

Available presets:
- `small.yaml`: Small datasets for testing
- `base.yaml`: Standard dataset size

Key parameters:
- `n_data`: Number of data points
- `x_dist`: Input distribution (`gauss_iso`, `gauss_aniso`, `mixture`, `lowdim_manifold`, `heavy_tail`)
- `noise_model`: Noise model (`gauss`, `hetero`, `student_t`, `outliers`)
- `noise_scale`: Noise level

**Example new preset** (`lambda_hat/conf/data/large.yaml`):
```yaml
n_data: 100000
x_dist: mixture
mixture_k: 8
noise_model: student_t
noise_scale: 0.05
```

### Sampler Presets (`lambda_hat/conf/sample/sampler/`)

Controls MCMC sampler parameters.

Available presets:
- `hmc.yaml`: Hamiltonian Monte Carlo
- `sgld.yaml`: Stochastic Gradient Langevin Dynamics
- `mclmc.yaml`: MCLMC sampler

**HMC parameters:**
- `draws`: Number of samples to draw
- `warmup`: Number of warmup steps
- `num_integration_steps`: Leapfrog steps per sample
- `step_size`: Step size (tuned automatically if `adapt_step_size: true`)
- `dtype`: Precision (`float64` recommended)

**SGLD parameters:**
- `steps`: Total number of gradient steps
- `warmup`: Number of warmup steps
- `batch_size`: Minibatch size
- `step_size`: Step size (needs manual tuning)
- `dtype`: Precision (`float32` for efficiency)

### Teacher Presets (`lambda_hat/conf/teacher/`)

Controls teacher network architecture (for teacher-student data generation).

- `_null.yaml`: No teacher (direct data generation)
- Custom teacher configs can specify architecture

## Adding New Configurations

### 1. Create New Presets

Add a new file under the appropriate directory:

```bash
# New model architecture
echo "depth: 8
target_params: 100000
activation: gelu" > lambda_hat/conf/model/huge.yaml

# New data configuration
echo "n_data: 200000
x_dist: heavy_tail
noise_model: outliers" > lambda_hat/conf/data/challenging.yaml
```

### 2. Use in Experiments

Reference the new presets in `config/experiments.yaml`:

```yaml
targets:
  - { model: huge, data: challenging, seed: 99 }
```

### 3. Override Specific Parameters

Use the `overrides` section for one-off parameter changes:

```yaml
targets:
  - { model: base, data: base, seed: 42,
      overrides: {
        model: { activation: identity },
        training: { steps: 20000, learning_rate: 0.0001 }
      }}

samplers:
  - { name: sgld,
      overrides: {
        step_size: 5e-7,
        batch_size: 128,
        eval_every: 200
      }}
```

## Target and Run IDs

- **Target ID**: Content-addressed hash of Stage A config (model + data + training + seed)
- **Run ID**: Hash of Stage B config (sampler + hyperparameters)
- Same config → same ID → deterministic reproducibility

Example artifacts:
```
runs/targets/tgt_abc123456789/meta.json                    # Target metadata
runs/targets/tgt_abc123456789/run_hmc_xy789abc/analysis.json # HMC run results
runs/targets/tgt_abc123456789/run_sgld_mn456def/analysis.json # SGLD run results
```

## Precision Control

Set precision globally in `config/experiments.yaml`:

```yaml
jax_enable_x64: true   # Use float64 (recommended for HMC/MCLMC)
jax_enable_x64: false  # Use float32 (faster, good for SGLD)
```

Individual samplers can override with `dtype` parameter in their preset files.