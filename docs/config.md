# Configuration Reference

**Auto-generated from `lambda_hat/conf/**/*.yaml`** — do not edit by hand.

Use this page to look up exact YAML defaults and schemas.
For a conceptual guide on composing experiments, see [Experiments Guide](./experiments.md).

This page lists all configuration options with their default values.
Configuration files use Hydra/OmegaConf for composition and interpolation.

## data

### `data/base.yaml`

```yaml
# @package _global_.data
n_data: 20000
x_dist: gauss_iso
cov_decay: 0.95
mixture_k: 4
mixture_spread: 2.0
x_dim_latent: 2
noise_model: gauss
noise_scale: 0.1
hetero_scale: 0.1
student_df: 4.0
outlier_frac: 0.05
outlier_scale: 2.0
```

### `data/small.yaml`

```yaml
# @package _global_.data
n_data: 100
x_dist: gauss_iso
cov_decay: 0.95
mixture_k: 4
mixture_spread: 2.0
x_dim_latent: 2
noise_model: gauss
noise_scale: 0.1
hetero_scale: 0.1
student_df: 4.0
outlier_frac: 0.05
outlier_scale: 2.0
```

## model

### `model/base.yaml`

```yaml
# @package _global_.model
in_dim: 32
out_dim: 1
depth: 3
widths: null
activation: relu
bias: true
layernorm: false
init: he
target_params: 10000
hidden: 300
```

### `model/small.yaml`

```yaml
# @package _global_.model
in_dim: 4
out_dim: 1
depth: 2
widths: null
activation: relu
bias: true
layernorm: false
init: he
target_params: 50
hidden: 32
```

## root

### `promote.yaml`

```yaml
# Asset promotion configuration
runs_root: runs
samplers: sgld,hmc,mclmc,vi
outdir: assets
plot_name: trace.png
```

### `workflow.yaml`

```yaml
# lambda_hat/conf/workflow.yaml
# @package _global_

# Unified Workflow Configuration for N x M Sweeps

# --- Shared Config ---
jax:
  enable_x64: true

# Ensure artifacts root is stable across single runs and multiruns
store:
  root: runs

runtime:
  seed: 12345  # Used for sampler PRNG splitting (Stage B Seed)
  code_sha: ${git_sha:}
  hostname: ${hostname:}

# Misc (toplevel Config fields)
use_tqdm: true
progress_update_every: 50
profile_adaptation: true

# --- Stage A: Target Definition (Used for Fingerprinting) ---
target:
  # Seed for data generation and NN training (Stage A Seed)
  seed: 42

# --- Stage B: Sampler Configuration ---
sampler:
  chains: 4
  # name will be overridden by sample/sampler group

posterior:
  # Defaults compatible with existing structure
  loss: mse
  beta_mode: 1_over_log_n
  beta0: 1.0
  gamma: 1.0

# Training block (used for Stage A target building)
training:
  optimizer: adam
  learning_rate: 0.001
  steps: 5000
  early_stop_tol: 1e-6
  batch_size: null  # Use full batch by default

# Teacher configuration is loaded via config group defaults (teacher: _null)
```

## sample

### `sample/base.yaml`

```yaml
# Two-stage: Stage B (sample) — OmegaConf config

jax:
  enable_x64: true

# Ensure artifacts root is stable across single runs and multiruns
store:
  root: runs

# Target selection is provided via CLI (--target-id). Keep this file agnostic.

runtime:
  seed: 12345  # used to split per-chain keys downstream
  code_sha: ${git_sha:}

sampler:
  chains: 4

posterior:
  loss: mse
  beta_mode: 1_over_log_n
  beta0: 1.0
  prior_radius: null
  gamma: 1.0
```

## sample / sampler

### `sample/sampler/hmc.yaml`

```yaml
# @package _global_.sampler
name: hmc
hmc:
  # Reduced draws to reach ~10k FGEs (1000 draws * 10 steps = 10000 FGEs)
  draws: 1000
  warmup: 200
  num_integration_steps: 10
  step_size: 0.01
  dtype: float64  # Explicit precision for HMC
  adapt_step_size: true
  target_acceptance: 0.8
```

### `sample/sampler/mclmc.yaml`

```yaml
# @package _global_.sampler
name: mclmc
mclmc:
  # Reduced draws to reach ~10k FGEs (1000 draws * (1.0/0.1) = 10000 FGEs)
  draws: 1000
  L: 1.0
  step_size: 0.1
  dtype: float64
  diagonal_preconditioning: false
  # Reduced adaptation steps slightly
  num_steps: 500
  frac_tune1: 0.1
  frac_tune2: 0.1
  frac_tune3: 0.1
  desired_energy_var: 0.0005
  trust_in_estimate: 1.0
  num_effective_samples: 150.0
  integrator: isokinetic_mclachlan
```

### `sample/sampler/sgld.yaml`

```yaml
# @package _global_.sampler
name: sgld
sgld:
  # Increased steps to reach ~10k FGEs (781250 steps * (256/20000) = 10000 FGEs)
  steps: 781250
  warmup: 50000
  batch_size: 256
  step_size: 0.000001
  dtype: float32
  precond: none
  beta1: 0.9
  beta2: 0.999
  eps: 0.00000001
  bias_correction: true
  # Increased thinning due to large step count
  eval_every: 100
```

### `sample/sampler/vi.yaml`

```yaml
# @package _global_.sampler
name: vi
vi:
  # Algorithm selection
  algo: mfa         # "mfa" (mixture of factor analyzers) or "flow"

  # Variational family
  M: 8              # number of mixture components
  r: 2              # rank budget per component

  # Optimization
  steps: 5000       # optimization steps
  batch_size: 256   # minibatch size
  lr: 0.01          # learning rate
  eval_every: 50    # how often to record metrics

  # Evaluation
  eval_samples: 64  # MC samples for final estimate

  # Precision
  dtype: float32

  # Whitening (Stage 1)
  whitening_mode: none      # "none"|"rmsprop"|"adam" - geometry whitening
  whitening_decay: 0.99     # EMA decay for gradient moments

  # Stability (Stage 1)
  clip_global_norm: 5.0     # gradient clipping threshold (null to disable)
  alpha_temperature: 1.0    # softmax temperature on mixture weights

  # Diagnostics (Stage 2)
  tensorboard: false        # enable TensorBoard logging (requires tensorboardX)

  # Advanced Configuration (Stage 3)
  r_per_component: null     # heterogeneous rank budgets (list[int] of length M)
  alpha_dirichlet_prior: null  # Dirichlet(α) prior on mixture weights
  lr_schedule: null         # "cosine"|"linear_decay" learning rate schedule
  lr_warmup_frac: 0.05      # fraction of steps for warmup
  entropy_bonus: 0.0        # add λ * H(q) to ELBO for exploration

  # Flow-specific Configuration (only used when algo="flow")
  d_latent: 8               # latent dimension for normalizing flow
  sigma_perp: 1e-3          # orthogonal noise scale for manifold-plus-noise map
  flow_depth: 4             # number of flow transformations (coupling layers)
  flow_hidden: [64, 64]     # hidden layer sizes for flow network
  flow_type: realnvp        # flow architecture: "realnvp"|"maf"|"nsf_ar"
```

## target

### `target/base.yaml`

```yaml
# OmegaConf config

jax:
  enable_x64: true

target:
  seed: 42

# Ensure artifacts root is stable across single runs and multiruns
store:
  root: runs

runtime:
  code_sha: ${git_sha:}
  hostname: ${hostname:}

# Training block (used by your ERM)
training:
  optimizer: adam
  learning_rate: 0.001
  steps: 5000
  early_stop_tol: 1e-6
  batch_size: null  # Use full batch by default

# Teacher configuration is loaded via config group defaults (teacher: _null)
```

## target / data

### `target/data/medium.yaml`

```yaml
data:
  n_data: 1000
  x_dist: gauss_iso
  cov_decay: 0.95
  noise_model: gauss
  noise_scale: 0.1
  x_dim_latent: 4
```

### `target/data/small.yaml`

```yaml
data:
  n_data: 100
  x_dist: gauss_iso
  cov_decay: 0.95
  noise_model: gauss
  noise_scale: 0.1
  x_dim_latent: 2
```

## target / model

### `target/model/medium.yaml`

```yaml
model:
  name: mlp
  target_params: 10000  # Approximate target number of parameters
  depth: 3
  width_factor: 1.2    # Multiplier for hidden layer width
  activation: relu
  use_bias: true
  layer_norm: false
  final_activation: none
```

### `target/model/small.yaml`

```yaml
model:
  name: mlp
  target_params: 1000  # Approximate target number of parameters
  depth: 2
  width_factor: 1.0    # Multiplier for hidden layer width
  activation: relu
  use_bias: true
  layer_norm: false
  final_activation: none
```

## teacher

### `teacher/_null.yaml`

```yaml
# Empty teacher config (default = no teacher)
{}
```

### `teacher/base.yaml`

```yaml
# Base teacher configuration
depth: 3
widths: null          # optional; if null we infer
activation: relu
dropout_rate: 0.1
# optional size drivers (choose at most one; both null is allowed -> fallback):
target_params: 100
hidden: null
```

### `teacher/small.yaml`

```yaml
# Small teacher configuration
depth: 2
widths: null          # optional; if null we infer
activation: relu
dropout_rate: 0.0
# optional size drivers (choose at most one; both null is allowed -> fallback):
target_params: 50
hidden: null
```
