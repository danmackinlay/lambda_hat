# Parallel Execution with Snakemake

This project uses **Snakemake** for parallel orchestration, enabling efficient execution locally or on HPC clusters without any Hydra launchers.

## Overview

Snakemake automatically identifies parallelizable jobs in the DAG and can execute them concurrently based on dependency constraints. For Lambda-Hat:

- **Target building** (Stage A) can run in parallel across different target configurations
- **Sampling runs** (Stage B) can run in parallel across different samplers and targets
- **Within each target**, all samplers can run concurrently (they only depend on the target being built)

## Local Parallel Execution

Use the `-j` flag to specify the number of parallel jobs:

```bash
# Run with 4 parallel jobs
uv run snakemake -j 4

# Run with as many jobs as CPU cores
uv run snakemake -j

# Preview the DAG without execution
uv run snakemake -n --dag | dot -Tpng > dag.png
```

**Example execution flow:**
1. Build targets `tgt_abc123` and `tgt_def456` in parallel (if resources allow)
2. Once targets are ready, run HMC and SGLD sampling in parallel on each target
3. Total: 2 targets × 2 samplers = 4 sampling jobs, plus 2 target building jobs

## HPC Cluster Execution

Snakemake supports HPC execution through **profiles**. Profiles define how jobs are submitted to schedulers like SLURM, PBS, or SGE.

### Setting Up SLURM Profile

1. **Install snakemake-profiles** (if not already available):
```bash
# Install the cookiecutter and create a SLURM profile
pip install cookiecutter
cookiecutter https://github.com/Snakemake-Profiles/slurm.git
```

2. **Configure your profile** in `~/.config/snakemake/slurm/config.yaml`:
```yaml
cluster:
  mkdir -p logs/{rule} &&
  sbatch
    --account={resources.account}
    --partition={resources.partition}
    --qos={resources.qos}
    --cpus-per-task={resources.cpus}
    --mem={resources.mem_mb}
    --time={resources.time}
    --job-name=snakemake-{rule}-{wildcards}
    --output=logs/{rule}/{rule}-{wildcards}-%j.out
    --error=logs/{rule}/{rule}-{wildcards}-%j.err
    --parsable

default-resources:
  - cpus=1
  - mem_mb=4000
  - time="01:00:00"
  - account="your_account"
  - partition="your_partition"
  - qos="normal"

resources:
  - build_target:
      cpus: 4
      mem_mb: 8000
      time: "02:00:00"
  - run_sampler:
      cpus: 8
      mem_mb: 16000
      time: "04:00:00"
```

3. **Run with the profile**:
```bash
# Submit up to 100 parallel jobs to SLURM
uv run snakemake --profile slurm -j 100

# Or specify profile path directly
uv run snakemake --cluster-config ~/.config/snakemake/slurm/config.yaml -j 100
```

### Resource Specification in Snakefile

You can add resource requirements directly to rules in the `Snakefile`:

```python
rule build_target:
    input: cfg = rules.cfg_build.output.cfg
    output: meta = f"{STORE}/targets/{{tid}}/meta.json"
    resources:
        cpus=4,
        mem_mb=8000,
        time="02:00:00",
        partition="gpu"  # For GPU-requiring targets
    shell: """
        JAX_ENABLE_X64={JAX64} uv run python -m lambda_hat.entrypoints.build_target \
          --config-yaml {input.cfg} --target-id {params.tid} --target-dir {params.tdir}
    """

rule run_sampler:
    input:
        meta = rules.build_target.output.meta,
        cfg = rules.cfg_sample.output.cfg
    output: analysis = f"{STORE}/targets/{{tid}}/run_{{sampler}}_{{rid}}/analysis.json"
    resources:
        cpus=lambda wildcards: 16 if wildcards.sampler == "hmc" else 4,
        mem_mb=lambda wildcards: 32000 if wildcards.sampler == "hmc" else 8000,
        time="06:00:00"
    shell: """
        JAX_ENABLE_X64={JAX64} uv run python -m lambda_hat.entrypoints.sample \
          --config-yaml {input.cfg} --target-id {params.tid} --run-dir {params.rdir}
    """
```

## GPU Management

For GPU-enabled runs:

### Local GPU
```bash
# Limit to 1 job to avoid GPU conflicts
CUDA_VISIBLE_DEVICES=0 uv run snakemake -j 1
```

### HPC GPU
Configure GPU resources in your profile:

```yaml
# In your SLURM profile
resources:
  - run_sampler:
      cpus: 8
      mem_mb: 16000
      time: "04:00:00"
      gres: "gpu:1"  # Request 1 GPU
      partition: "gpu"
```

Or use additional SLURM parameters:
```yaml
cluster: |
  sbatch
    --account={resources.account}
    --partition={resources.partition}
    --gres=gpu:{resources.gpus}
    --cpus-per-task={resources.cpus}
    --mem={resources.mem_mb}
    --time={resources.time}
```

## Common Patterns

### Partial Execution

Run only specific stages:

```bash
# Build all targets but don't run samplers
uv run snakemake "runs/targets/*/meta.json" -j 4

# Run only HMC samplers
uv run snakemake "runs/targets/*/run_hmc_*/analysis.json" -j 8

# Run everything for one specific target
uv run snakemake "runs/targets/tgt_abc123456789/run_*/analysis.json" -j 4
```

### Forcing Reruns

```bash
# Force rebuild of all targets
uv run snakemake --forcerun build_target -j 4

# Force rerun of SGLD only
uv run snakemake --forcerun "runs/targets/*/run_sgld_*/analysis.json" -j 8

# Force rerun everything downstream of a specific target
uv run snakemake --forcerun runs/targets/tgt_abc123456789/meta.json -j 4
```

### Resource-Aware Scheduling

```bash
# Limit memory usage (useful on shared systems)
uv run snakemake -j 4 --resources mem_mb=32000

# Limit to specific partitions
uv run snakemake --profile slurm -j 100 --default-resources partition=normal
```

## Monitoring and Logging

### Job Status
```bash
# Check job status
uv run snakemake --summary
uv run snakemake --detailed-summary

# Dry run with details
uv run snakemake -n -r  # Show reasoning for each job
```

### Log Management
- **Local**: Logs go to `logs/` directory as specified in rules
- **SLURM**: Logs go to paths specified in your profile (typically `logs/{rule}/`)
- **Real-time monitoring**: Use `tail -f logs/build_target/*.log` to watch progress

### Failed Jobs
```bash
# Rerun only failed jobs
uv run snakemake --rerun-incomplete -j 4

# Get failed job details
snakemake --detailed-summary | grep FAILED
```

## Performance Tips

1. **Target building is expensive**: Consider building targets once and reusing them across sampler sweeps
2. **I/O can be bottleneck**: On HPC, use local scratch space for temporary files
3. **Memory management**: HMC/MCLMC need more RAM than SGLD; adjust resources accordingly
4. **JAX compilation**: First run per node may be slower due to XLA compilation caching

Example optimized workflow:
```bash
# Build targets first with fewer, larger jobs
uv run snakemake "runs/targets/*/meta.json" -j 2 --resources cpus=8

# Then run samplers with more parallelism
uv run snakemake -j 16 --resources cpus=2
```