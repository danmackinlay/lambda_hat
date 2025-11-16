# Workflows

Orchestrating experiments with Parsl, parameter sweeps, and artifact management.

---

## Quick Start

**At a glance**:
- Parsl orchestrates N targets × M samplers in parallel
- Run locally with `--local` or on SLURM clusters with `--parsl-card`
- Optional promotion creates asset galleries
- Artifacts stored in content-addressed system with automatic deduplication

**Local execution**:
```bash
# Basic workflow
uv run lambda-hat workflow llc --local

# With promotion (galleries)
uv run lambda-hat workflow llc --local --promote
```

**HPC execution**:
```bash
# SLURM cluster with A100 GPUs
uv run lambda-hat workflow llc --parsl-card config/parsl/slurm/gpu-a100.yaml
```

---

## Parsl Orchestration

### Execution Model

Parsl manages parallel execution through Python futures:

1. **Stage A** (Build): Train N targets in parallel
2. **Stage B** (Sample): Run M samplers per target (waits for target via `inputs=[target_future]`)
3. **Stage C** (Promote): Optional gallery generation (if `--promote` flag used)

**Example**: 2 targets × 3 samplers = 6 sampling jobs + 2 target builds

---

### Local Execution

**Default profile**: ThreadPoolExecutor with up to 8 workers.

```bash
# Run locally (uses up to 8 CPU cores)
uv run lambda-hat workflow llc --local

# With promotion
uv run lambda-hat workflow llc --local --promote
```

**Control parallelism**: Edit `config/parsl/local.yaml`:
```yaml
max_workers: 8  # Cap at 8 cores
```

**Monitoring**:
```
=== Stage A: Building Targets ===
  Submitting build for tgt_abc123 (model=small, data=small)

=== Stage B: Running Samplers ===
  Submitting hmc for tgt_abc123 (run_id=ab12cd34)

=== Waiting for 1 sampling runs to complete ===
  [1/1] completed
```

**Log files**:
```
logs/
├── build_target/
│   └── tgt_abc123.log
└── run_sampler/
    └── tgt_abc123_hmc_ab12cd34.log
```

---

### SLURM Cluster Execution

**Parsl cards** define executor configs for SLURM clusters.

**Available profiles**:
- `config/parsl/slurm/cpu.yaml` — CPU nodes
- `config/parsl/slurm/gpu-a100.yaml` — A100 GPU nodes

**Run on SLURM**:
```bash
# Use A100 GPU profile
uv run lambda-hat workflow llc --parsl-card config/parsl/slurm/gpu-a100.yaml

# Customize with overrides
uv run lambda-hat workflow llc --parsl-card config/parsl/slurm/cpu.yaml \
    --set walltime=04:00:00 --set gpus_per_node=2
```

**Parsl card structure** (`config/parsl/slurm/gpu-a100.yaml`):
```yaml
partition: gpu
nodes_per_block: 1
cores_per_node: 16
mem_per_node: 64
gpus_per_node: 1
walltime: "02:00:00"
scheduler_options: "#SBATCH --gres=gpu:a100:1"
```

**SLURM monitoring**:
```bash
squeue -u $USER              # Check job queue
scancel <job_id>             # Cancel job
tail -f logs/run_sampler/*   # Watch logs
```

---

## Parameter Sweeps

Lambda-Hat automatically generates N × M experiment matrices from `config/experiments.yaml`.

### Basic Sweep

**Define in `config/experiments.yaml`**:
```yaml
targets:
  - { model: small, data: small, teacher: _null, seed: 42 }
  - { model: base,  data: base,  teacher: _null, seed: 43 }
  - { model: wide,  data: large, teacher: _null, seed: 44 }

samplers:
  - { name: hmc }
  - { name: sgld, overrides: { step_size: 1e-6 } }
  - { name: mclmc, overrides: { draws: 5000 } }
  - { name: vi, overrides: { M: 8, r: 2 } }
```

**Result**: 3 targets × 4 samplers = **12 sampling jobs** + 3 target builds = **15 total jobs**

**Execute**:
```bash
uv run lambda-hat workflow llc --local
```

---

### Sweep Patterns

#### 1. Model Architecture Sweeps

Compare different network architectures:

```yaml
targets:
  # Using presets
  - { model: small, data: base, seed: 42 }
  - { model: base,  data: base, seed: 42 }
  - { model: wide,  data: base, seed: 42 }

  # Using overrides
  - { model: base, data: base, seed: 42,
      overrides: { model: { depth: 8, activation: gelu } } }
```

**Use case**: Understand how architecture affects LLC estimates.

---

#### 2. Sampler Hyperparameter Sweeps

Test robustness across sampler settings:

```yaml
targets:
  - { model: base, data: base, seed: 43 }

samplers:
  - { name: sgld, overrides: { step_size: 1e-7 }, seed: 1001 }
  - { name: sgld, overrides: { step_size: 1e-6 }, seed: 1002 }
  - { name: sgld, overrides: { step_size: 1e-5 }, seed: 1003 }
```

**Use case**: Find optimal sampler hyperparameters.

---

#### 3. Precision Sweeps

Compare float32 vs float64:

```yaml
# config/exp_f32.yaml
jax_enable_x64: false
targets: [...]

# config/exp_f64.yaml
jax_enable_x64: true
targets: [...]
```

Run separately:
```bash
uv run lambda-hat workflow llc --local --config config/exp_f32.yaml
uv run lambda-hat workflow llc --local --config config/exp_f64.yaml
```

---

## Optuna Hyperparameter Optimization

**Automated hyperparameter tuning** using Bayesian optimization:

```bash
# Optimize locally
uv run lambda-hat workflow optuna --config config/optuna_demo.yaml --local

# Optimize on SLURM
uv run lambda-hat workflow optuna --config config/optuna_demo.yaml \
    --parsl-card config/parsl/slurm/cpu.yaml
```

**How it works**:
1. Computes HMC reference LLC for each problem (high-quality baseline)
2. Optimizes method hyperparameters (SGLD/VI/MCLMC) to minimize `|LLC - LLC_ref|`
3. Uses Optuna's TPE sampler for Bayesian search
4. Results written to `results/optuna_trials.parquet`

**Use cases**:
- Find optimal hyperparameters for your problem class
- Compare methods under fair time budgets
- Automate parameter tuning instead of manual sweeps

See [Optuna Workflow documentation](./optuna_workflow.md) for detailed configuration.

---

## Artifact Management

### Artifact Layout

**Standard workflow** (`lambda-hat workflow llc`):
```
runs/
└── targets/
    ├── _catalog.jsonl               # Registry of all targets
    └── tgt_abc123/
        ├── meta.json                # Config, dimensions, L0
        ├── data.npz                 # Training data
        ├── params.npz               # Trained parameters
        ├── _runs.jsonl              # Manifest of Stage-B runs
        ├── run_hmc_ab12cd34/
        │   ├── trace.nc             # ArviZ trace
        │   ├── analysis.json        # Metrics
        │   └── diagnostics/
        │       ├── trace.png
        │       └── rank.png
        ├── run_sgld_ef567890/
        └── run_mclmc_gh901234/
```

**Optuna workflow** (`lambda-hat workflow optuna`):
```
artifacts/
├── problems/
│   └── p_abc123/
│       └── ref.json                 # HMC reference LLC
└── runs/
    └── p_abc123/
        └── vi/
            ├── r_def456/            # One trial
            │   ├── manifest.json    # Trial hyperparameters
            │   └── metrics.json     # Trial results
            └── r_ghi789/

results/
├── optuna_trials.parquet            # All trials aggregated
└── studies/
    └── optuna_llc/
        └── p_abc123:vi.pkl          # Optuna study (for resume)
```

---

### CLI Tools

**List experiments and runs**:
```bash
lambda-hat artifacts ls
```

**Garbage collect old artifacts**:
```bash
# Default TTL: 30 days
lambda-hat artifacts gc

# Custom TTL
lambda-hat artifacts gc --ttl-days 7
```

**Point TensorBoard at runs**:
```bash
lambda-hat artifacts tb --experiment my_experiment
```

---

### Promotion

**Promotion** copies plots from run directories into stable locations for galleries.

**Enable promotion**:
```bash
uv run lambda-hat workflow llc --local --promote
```

**Specify which plots**:
```bash
uv run lambda-hat workflow llc --local --promote \
    --promote-plots trace.png,llc_convergence_combined.png
```

**Manual promotion**:
```bash
# Create gallery with newest run per sampler
lambda-hat promote gallery \
  --runs-root runs \
  --samplers sgld,hmc,mclmc \
  --outdir runs/promotion \
  --snippet-out runs/promotion/gallery.md

# Copy specific plots
lambda-hat promote single \
  --runs-root runs \
  --samplers sgld \
  --outdir figures \
  --plot-name running_llc.png
```

---

## Reproducibility

**Same target ID = identical data and parameters**

**Content addressing**: Targets use deterministic IDs based on:
- Model config (depth, width, activation)
- Data config (n_data, n_features, seed)
- Training config (steps, optimizer)

**Precision guard**: Mismatch between target build (`jax_enable_x64`) and sampling run → error

**Metadata**: Package versions and code SHA recorded in `meta.json`

---

## See Also

- [Configuration Reference](./config.md) — Complete YAML schema
- [CLI Reference](./cli.md) — All command-line options
- [Samplers](./samplers.md) — Available sampling algorithms
- [Optuna Workflow](./optuna_workflow.md) — Hyperparameter optimization details
- `docs/output_management.md` — Detailed artifact system architecture
