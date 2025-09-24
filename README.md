# Local Learning Coefficient Estimation with Hydra

In [Singular Learning Theory (SLT)](https://singularlearningtheory.com), the **Local Learning Coefficient (LLC)** quantifies the effective local dimensionality of a model around a trained optimum.

This repository provides a clean, standardized implementation using:
- **Hydra** for configuration management and experiment orchestration
- **Haiku** for modern JAX neural network definitions
- **BlackJAX** for state-of-the-art MCMC sampling
- **ArviZ** for diagnostics and analysis

---

## Features

✨ **Radical Simplification**: Replaced custom CLI framework with industry-standard Hydra
🏗️ **Modern Architecture**: Uses Haiku for neural networks, eliminating manual parameter management
🎯 **Clean Sampling**: Transparent JAX/BlackJAX loops with `jax.lax.scan`
📊 **Comprehensive Analysis**: Automatic LLC metrics, ESS computation, and visualization
🔧 **Composable Configuration**: Hierarchical YAML configs with easy overrides

---

## Quick Start

### Installation

```bash
# Create environment
uv venv --python 3.12 && source .venv/bin/activate

# Install dependencies
uv sync --extra cpu  # For CPU/MPS
# or
uv sync --extra cuda12  # For CUDA 12.x
```

### Basic Usage

```bash
# Run with default configuration
python train.py

# Use preset configurations
python train.py sampler=fast model=small data=small

# Override specific parameters
python train.py model.target_params=1000 data.n_data=5000

# Run parameter sweeps
python train.py --multirun sampler=base,fast model=small,base
```

---

## Configuration System

The new Hydra-based system uses structured, composable configurations:

### Directory Structure
```
conf/
├── config.yaml          # Main configuration with defaults
├── model/
│   ├── base.yaml        # Standard model (10K params)
│   └── small.yaml       # Small model (50 params)
├── data/
│   ├── base.yaml        # Standard dataset (20K points)
│   └── small.yaml       # Small dataset (100 points)
└── sampler/
    ├── base.yaml        # Production sampling settings
    └── fast.yaml        # Quick test settings
```

### Configuration Examples

**Default Configuration** (`conf/config.yaml`):
```yaml
defaults:
  - model: base
  - data: base
  - sampler: base

target: mlp
seed: 42

training:
  optimizer: adam
  learning_rate: 0.001
  erm_steps: 5000

posterior:
  loss: mse
  beta_mode: "1_over_log_n"
  beta0: 1.0
```

**Model Configurations** (`conf/model/small.yaml`):
```yaml
in_dim: 4
out_dim: 1
depth: 2
target_params: 50
activation: relu
```

**Sampler Configurations** (`conf/sampler/fast.yaml`):
```yaml
chains: 2
sgld:
  steps: 100
  warmup: 20
hmc:
  draws: 50
  warmup: 20
```

---

## Samplers

The system supports three main sampling algorithms:

### SGLD (Stochastic Gradient Langevin Dynamics)
- **Best for**: Large datasets, fast approximate sampling
- **Precision**: float32 for memory efficiency
- **Features**: Minibatching, optional preconditioning

### HMC (Hamiltonian Monte Carlo)
- **Best for**: High-quality samples, reliable diagnostics
- **Precision**: float64 for numerical stability
- **Features**: Automatic step size adaptation, mass matrix tuning

### MCLMC (Microcanonical Langevin Monte Carlo)
- **Best for**: Cutting-edge sampling efficiency
- **Precision**: float64
- **Features**: Advanced integrators, energy variance targeting

---

## Output and Analysis

Hydra automatically manages output directories with timestamped folders:

```
outputs/
└── 2024-09-24/
    └── 14-30-45/
        ├── config.yaml           # Full resolved configuration
        ├── target_info.json      # Model and data information
        ├── run_info.json         # Timing and execution details
        ├── metrics_summary.csv   # LLC statistics table
        ├── summary.txt           # Human-readable summary
        ├── llc_traces.png        # Trace plots for all samplers
        └── llc_comparison.png    # Comparative bar chart
```

### Key Metrics

For each sampler, the system computes:
- **LLC Mean**: Primary quantity of interest
- **LLC Standard Deviation**: Uncertainty quantification
- **Effective Sample Size (ESS)**: Sampling efficiency
- **R-hat**: Convergence diagnostic across chains

---

## Architecture Overview

The new codebase follows a clean, modular design:

```
llc/
├── models.py          # Haiku neural network definitions
├── data.py            # Dataset generation (unchanged)
├── losses.py          # Loss functions adapted for Haiku
├── training.py        # ERM optimization with Optax
├── targets.py         # Target construction and setup
├── posterior.py       # Posterior density construction
├── sampling.py        # Clean BlackJAX sampling loops
├── analysis.py        # LLC computation and metrics
├── artifacts.py       # Output saving and management
└── config.py          # Structured configuration schemas
```

### Key Design Principles

1. **Standardization**: Use proven tools (Hydra, Haiku) instead of custom frameworks
2. **Transparency**: Clear JAX code with `jax.lax.scan` loops instead of opaque abstractions
3. **Modularity**: Each module has a single, well-defined responsibility
4. **Type Safety**: Structured configs with dataclass validation

---

## Migration from Legacy CLI

The legacy `llc` command-line interface has been completely replaced. Here's how to translate common workflows:

### Old → New Command Mapping

| Legacy Command | New Hydra Command |
|---|---|
| `llc run --sampler sgld --preset=quick` | `python train.py sampler=fast` |
| `llc sweep --n-data 1000,5000 --sampler hmc` | `python train.py --multirun data.n_data=1000,5000` |
| `llc run --target-params 500 --depth 2` | `python train.py model.target_params=500 model.depth=2` |

### Configuration Migration

Legacy configurations can be translated to the new YAML format:

```python
# Old: Monolithic Config object
cfg = Config(
    target_params=1000,
    n_data=5000,
    sgld_steps=1000,
    hmc_draws=500
)

# New: Structured Hydra config
# conf/experiment/my_experiment.yaml
defaults:
  - base_config

model:
  target_params: 1000
data:
  n_data: 5000
sampler:
  sgld:
    steps: 1000
  hmc:
    draws: 500
```

---

## Advanced Usage

### Custom Experiments

Create experiment-specific configs:

```yaml
# conf/experiment/large_scale.yaml
defaults:
  - base_config
  - model: large

data:
  n_data: 100000
sampler:
  chains: 8
  sgld:
    steps: 50000
    batch_size: 512
```

Run with: `python train.py +experiment=large_scale`

### SLURM Integration

Use Hydra's submitit launcher for cluster execution:

```bash
python train.py --config-path conf --config-name config \
  hydra/launcher=submitit_slurm \
  hydra.launcher.timeout_min=240 \
  hydra.launcher.cpus_per_task=4
```

### Parameter Sweeps

Run comprehensive parameter sweeps:

```bash
python train.py --multirun \
  model=small,base \
  data.noise_scale=0.01,0.1,1.0 \
  sampler.chains=2,4,8
```

---

## Troubleshooting

### Common Issues

**Import Errors**: Ensure all dependencies are installed:
```bash
uv sync --extra cpu  # or --extra cuda12
```

**Configuration Errors**: Validate config structure:
```bash
python -c "from llc.config import setup_config; setup_config()"
```

**Memory Issues**: Use smaller configurations:
```bash
python train.py model=small data=small sampler=fast
```

### Performance Tips

1. **Use appropriate precision**: SGLD uses float32, HMC/MCLMC use float64
2. **Tune batch sizes**: Larger batches for SGLD on GPU, smaller for CPU
3. **Monitor adaptation**: Check HMC warmup convergence
4. **Profile runs**: Use `cfg.profile_adaptation=True` for timing

---

## Contributing

The new architecture makes contributions much easier:

1. **Add new samplers**: Implement in `sampling.py` following existing patterns
2. **Add new models**: Extend Haiku modules in `models.py`
3. **Add new configs**: Create YAML files in appropriate `conf/` subdirectories
4. **Add new analysis**: Extend functions in `analysis.py`

All changes should maintain the clean, functional style and comprehensive type hints.

---

## Citation

If you use this code in your research, please cite:

```bibtex
@software{llc_hydra,
  title={Local Learning Coefficient Estimation with Hydra},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/estimating_llc_hydra}
}
```