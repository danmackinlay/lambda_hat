# Parallel Execution with Parsl

This project uses **Parsl** for parallel orchestration, enabling efficient execution locally or on HPC clusters through Python-native DAG execution.

## Overview

Parsl manages parallel execution through explicit dependency graphs using Python futures. For Lambda-Hat:

- **Target building** (Stage A) runs in parallel across different target configurations
- **Sampling runs** (Stage B) run in parallel across different samplers and targets
- **Each sampler** automatically waits for its target to build (via `inputs=[target_future]`)
- **Promotion** (Stage C) is optional and runs after all sampling completes

**Example execution flow:**
1. Build targets `tgt_abc123` and `tgt_def456` in parallel
2. Once targets are ready, run HMC and SGLD sampling in parallel on each target
3. Total: 2 targets × 2 samplers = 4 sampling jobs, plus 2 target building jobs
4. Optionally promote results to galleries (if `--promote` flag used)

---

## Local Parallel Execution

### Basic Usage

Run the workflow locally using ThreadPoolExecutor:

```bash
# Run locally (uses up to 8 CPU cores by default)
uv run python flows/parsl_llc.py --local

# With promotion enabled
uv run python flows/parsl_llc.py --local --promote
```

### Controlling Parallelism

Edit `parsl_config_local.py` to control the number of parallel workers:

```python
# Adjust max_workers
max_workers = min(os.cpu_count() or 4, 8)  # Cap at 8 cores
```

For very small tests, you can reduce this to 1 or 2 workers to simplify debugging.

### Monitoring Progress

The workflow prints progress messages as it runs:

```
=== Stage A: Building Targets ===
  Submitting build for tgt_abc123 (model=small, data=small)

=== Stage B: Running Samplers ===
  Submitting hmc for tgt_abc123 (run_id=ab12cd34)

=== Waiting for 1 sampling runs to complete ===
  [1/1] completed
```

### Log Files

All logs are written to `logs/` directory:

```
logs/
├── build_target/
│   └── tgt_abc123.log        # Build stdout
│   └── tgt_abc123.err        # Build stderr
└── run_sampler/
    └── tgt_abc123_hmc_ab12cd34.log   # Sample stdout
    └── tgt_abc123_hmc_ab12cd34.err   # Sample stderr
```

Check these files if jobs fail or behave unexpectedly.

---

## HPC Cluster Execution (SLURM)

### Basic Usage

Run the workflow on a SLURM cluster:

```bash
# Submit workflow to SLURM
uv run python flows/parsl_llc.py --parsl-config parsl_config_slurm.py

# With promotion
uv run python flows/parsl_llc.py --parsl-config parsl_config_slurm.py --promote
```

**How it works:**
1. You run the workflow **once** on a login/head node
2. Parsl submits `sbatch` jobs to SLURM automatically
3. Each job runs a target build or sampler run
4. Parsl manages dependencies between jobs
5. Results are aggregated when all jobs complete

### Configuring Resources

Edit `parsl_config_slurm.py` to adjust resources for your cluster:

```python
SlurmProvider(
    partition="normal",         # Change to "gpu", "debug", etc.
    nodes_per_block=1,          # Nodes per SLURM job
    max_blocks=50,              # Max concurrent SLURM jobs
    walltime="04:00:00",        # 4 hours per job
    cores_per_node=1,           # CPUs per node
    mem_per_node=8,             # GB of RAM
    scheduler_options="",       # Add custom SBATCH directives
)
```

**Common adjustments:**

| Setting | When to Change | Example |
|---------|----------------|---------|
| `partition` | Different queue | `partition="gpu"` for GPU nodes |
| `walltime` | Longer jobs | `walltime="24:00:00"` for large models |
| `max_blocks` | More parallelism | `max_blocks=100` for big sweeps |
| `mem_per_node` | Large datasets | `mem_per_node=32` for bigger networks |

### Environment Setup

The `worker_init` section runs on each SLURM node before executing tasks:

```python
worker_init="""
module load python || true
source ~/.bashrc || true
export PATH="$HOME/.local/bin:$PATH"
""".strip()
```

Customize this for your cluster:
- Load required modules (`module load cuda/12.1`)
- Activate conda environments
- Set environment variables

### Monitoring SLURM Jobs

While the workflow runs, you can monitor SLURM jobs from another terminal:

```bash
# Check job queue
squeue -u $USER

# Check specific job details
scontrol show job <jobid>

# Cancel all jobs (if needed)
scancel -u $USER
```

### Parsl Run Directory

Parsl creates a `parsl_runinfo/` directory with execution metadata:

```
parsl_runinfo/
├── <run_id>/
│   ├── submit_scripts/        # Generated SLURM scripts
│   ├── block_logs/            # SLURM job logs
│   └── parsl.log              # Parsl internal logs
```

This is useful for debugging SLURM submission issues.

---

## Troubleshooting

### Local Execution Issues

**Problem**: Jobs fail with "dependency failure"

**Solution**: Check the build log first - sampling jobs depend on successful target builds:
```bash
cat logs/build_target/tgt_*.err
```

**Problem**: "Too many open files" error

**Solution**: Reduce max_workers in `parsl_config_local.py`:
```python
max_workers = 2  # Lower for resource-constrained systems
```

### SLURM Execution Issues

**Problem**: Jobs never start / stuck in pending

**Solution**: Check SLURM job status:
```bash
squeue -u $USER --start  # See estimated start times
sinfo                    # Check partition availability
```

**Problem**: SLURM jobs fail immediately

**Solution**:
1. Check `parsl_runinfo/<run_id>/block_logs/` for SLURM errors
2. Verify partition name is correct in `parsl_config_slurm.py`
3. Check resource limits (walltime, memory) are valid for your cluster

**Problem**: "Module not found" errors in SLURM jobs

**Solution**: Update `worker_init` in `parsl_config_slurm.py`:
```python
worker_init="""
module load python/3.11
source ~/myenv/bin/activate  # or your env path
""".strip()
```

### JAX/GPU Issues on SLURM

**For GPU jobs**, update the SLURM config:

```python
SlurmProvider(
    partition="gpu",
    scheduler_options="#SBATCH --gres=gpu:1",  # Request 1 GPU
    worker_init="""
        module load cuda/12.1
        export JAX_PLATFORMS=cuda
    """.strip()
)
```

---

## Performance Tips

### Local Execution

1. **Start small**: Test with `--local` and 1-2 targets before scaling up
2. **Use float32**: Set `jax_enable_x64: false` in config for faster local testing
3. **Monitor resources**: Use `htop` to watch CPU/memory usage

### SLURM Execution

1. **Right-size jobs**: Don't request excessive resources if not needed
2. **Use debug partition**: For quick tests, use debug/short queue if available
3. **Check fairshare**: Smaller `max_blocks` may get scheduled faster
4. **Estimate walltime**: Add 20-30% buffer to avoid premature job kills

### Result Aggregation

Results are written to `results/llc_runs.parquet` at the end:

```bash
# View results
python -c "import pandas as pd; print(pd.read_parquet('results/llc_runs.parquet'))"

# Or use your favorite analysis tool
jupyter notebook  # Load and analyze parquet file
```

---

## Scaling Guidelines

| Scale | Targets | Samplers | Total Jobs | Recommended Config |
|-------|---------|----------|------------|-------------------|
| Small | 1-2 | 1-2 | 2-4 | `--local` |
| Medium | 5-10 | 2-4 | 20-40 | `--local` or small cluster |
| Large | 20-50 | 3-5 | 100-250 | SLURM with `max_blocks=50` |
| Very Large | 100+ | 5+ | 500+ | SLURM with `max_blocks=100+` |

**Memory estimates** (per job):
- Small model (d=50): ~2-4 GB
- Base model (d=500): ~8-16 GB
- Large model (d=5000): ~32-64 GB

---

## Comparison with Snakemake

If you're familiar with Snakemake, here are the key differences:

| Aspect | Snakemake | Parsl |
|--------|-----------|-------|
| DAG Definition | Implicit (file-based rules) | Explicit (Python futures) |
| Parallelism | `-j N` flag | Config `max_workers`/`max_blocks` |
| HPC Submission | Profiles (YAML) | Provider config (Python) |
| Dependency | File existence | Future completion |
| Debugging | Check logs + Snakemake DAG | Check logs + Python tracebacks |

**Why Parsl?**
- Python-native: all logic in Python, easier to reason about
- Dynamic workflows: easier to add conditional logic
- Explicit dependencies: clearer what depends on what
- Better for parameter sweeps: just Python loops over configs

---

## See Also

- [Configuration Guide](./configuration.md) - How to define experiment sweeps
- [Output Management](./output_management.md) - Understanding the artifact layout
- [Parsl Documentation](https://parsl.readthedocs.io/) - Official Parsl docs
