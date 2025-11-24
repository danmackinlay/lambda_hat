Yes, I’d revise that plan given where you are now and what you actually care about (GPU Slurm, embarrassingly parallel, good UX, Option D in place).

Let me give you a **target design** that’s explicitly GPU-aware and UX-friendly, then concrete steps.

---

## 1. Reframe the architecture (what matters now)

Given:

* Option D is in and working: **precision is per-task** via `cfg.jax.enable_x64`, not via executor selection.
* On Slurm, jobs are **process-isolated**; you’re going to use `nodes_per_block=1` and short walltimes.
* You care about **GPU vs CPU**, not “float32 vs float64 executors”.
* Sampler runs are **embarrassingly parallel**; you’re penalised for long jobs, not for many jobs.

So:

1. Stop thinking about “local vs Slurm” as a deep design problem. Treat them as **just different Parsl cards**.
2. Make GPU vs CPU a **card choice**, not something the workflow has to infer.
3. Give the user a **single CLI knob** like `--backend=local|slurm-cpu|slurm-gpu` and hide everything else.
4. Tune Slurm cards for **many short single-node, single-GPU blocks** that Parsl fills with your embarrassingly parallel tasks.

---

## 2. Slurm + GPU design

### 2.1. Cards: explicit CPU vs GPU

You already have:

* `config/parsl/slurm/cpu.yaml`
* `config/parsl/slurm/gpu-a100.yaml`

Refine them so they both follow this pattern:

```yaml
type: slurm
label: htex_slurm

partition: <cpu-or-gpu-partition>
nodes_per_block: 1
init_blocks: 0
min_blocks: 0
max_blocks: 200     # lots of small blocks
walltime: "00:20:00"  # or whatever fits “one sampler target” scale

cores_per_node: 4      # or 8, doesn’t matter much if you do 1 worker
mem_per_node: 64
gpus_per_node: 0|1     # 0 in cpu.yaml, 1 in gpu-a100.yaml
gpu_type: null|a100    # as needed

retries: 1
run_dir: parsl_runinfo
max_workers: 1         # 1 worker per node/block – good for embarrassingly parallel

worker_init: |
  module load python || true
  source ~/.bashrc || true
  export PATH="$HOME/.local/bin:$PATH"
  export JAX_DEFAULT_PRNG_IMPL=threefry2x32
  export MPLBACKEND=Agg
  # GPU card only:
  # module load cuda/12.1 || true
  # export JAX_PLATFORMS=cuda
```

Key points:

* **nodes_per_block=1**: each Slurm job is a single node; you’re not doing multi-node parallelism.
* **max_workers=1** / low `cores_per_node`: each node runs a single worker process, so one Parsl task at a time → exactly what you want for “one job per compute node”.
* **walltime short**: each block is a short job. If you have many tasks, Parsl/HTEX will spin up many such blocks as the backlog grows. Slurm is happy: many short jobs, no long hogs.

You can tweak `max_blocks` and `walltime` once you see typical sampler runtime.

### 2.2. GPUs: don’t overthink it

For GPU Slurm:

* `gpus_per_node: 1`, `gpu_type: a100` (already in your file).
* `worker_init` loads CUDA and sets `JAX_PLATFORMS=cuda`. That’s enough; Option D still handles x64 vs x32.
* If you want to be nice to the cluster, add `XLA_PYTHON_CLIENT_PREALLOCATE=false` too to avoid JAX grabbing the whole GPU by default.

You do **not** need separate executors per sampler or per dtype.

---

## 3. UX: make backend selection trivial

Right now your CLI for workflows is:

```bash
lambda-hat workflow llc --config ... --local
# or
lambda-hat workflow llc --config ... --parsl-card config/parsl/slurm/gpu-a100.yaml
```

That’s annoying. Replace it with:

### 3.1. New CLI flag: `--backend`

Add to `workflow llc` in `lambda_hat/cli.py`:

```python
@workflow.command("llc")
@click.option(
    "--backend",
    type=click.Choice(["local", "slurm-cpu", "slurm-gpu"]),
    default="local",
    show_default=True,
    help="Execution backend: local HTEX, Slurm CPU, or Slurm GPU.",
)
# keep --config, --experiment, --promote, etc.
def workflow_sample(config, experiment, backend, ...):
    ...
```

Then inside that command:

```python
from lambda_hat.parsl_cards import load_parsl_config_from_card
from lambda_hat.artifacts import Paths, RunContext

paths_early = Paths.from_env()
paths_early.ensure()
exp_config = OmegaConf.load(config)
experiment_name = experiment or exp_config.get("experiment") or "dev"
ctx_early = RunContext.create(experiment=experiment_name, algo="parsl_llc", paths=paths_early)

# Map backend -> card path
if backend == "local":
    card_path = Path("config/parsl/local.yaml")
elif backend == "slurm-cpu":
    card_path = Path("config/parsl/slurm/cpu.yaml")
elif backend == "slurm-gpu":
    card_path = Path("config/parsl/slurm/gpu-a100.yaml")
else:
    raise click.ClickException(f"Unknown backend {backend!r}")

if not card_path.exists():
    raise click.ClickException(f"Parsl card not found: {card_path}")

parsl_cfg = load_parsl_config_from_card(card_path, [f"run_dir={ctx_early.parsl_dir}"])
```

And delete the previous `--local` / `--parsl-card` logic in that command. You can keep `workflow optuna` as-is for now, or give it the same `--backend` pattern later.

From the user’s perspective:

* Local dev:
  `uv run lambda-hat workflow llc --backend local --config config/smoke.yaml`
* CPU Slurm:
  `uv run lambda-hat workflow llc --backend slurm-cpu --config config/experiments.yaml`
* GPU Slurm:
  `uv run lambda-hat workflow llc --backend slurm-gpu --config config/experiments.yaml`

No agonising about cards.

If you want serious ergonomics, read a default backend from env:

```python
default_backend = os.getenv("LAMBDA_HAT_BACKEND", "local")
@click.option("--backend", ..., default=default_backend, show_default=True)
```

So on the cluster you can just:

```bash
export LAMBDA_HAT_BACKEND=slurm-gpu
uv run lambda-hat workflow llc --config config/experiments.yaml
```

---

## 4. Embarrassingly parallel runs: align workflow + Slurm

Your Stage B is already “target × sampler × seed → independent sampler run.” The improvements above make Slurm *see* many small jobs:

* `nodes_per_block=1`, `walltime` short
* `max_workers=1`
* `max_blocks` high

That combines well with your Option D model:

* Each sampler task is a separate Parsl task.
* HTEX requests as many blocks as you have runnable tasks (up to `max_blocks`), each block running exactly one worker executing one task at a time.
* Slurm happily schedules many short GPU jobs.

For very large grids, you might bump `max_blocks` to the expected concurrency limit (e.g. if your queue allows you ~100 GPUs at once, set `max_blocks=100`).

You do **not** need to change the Parsl app definitions; `run_sampler_app` already takes care of per-run configs.

---

## 5. Checklist of revisions vs your current plan

Compared to the text you pasted:

* Drop the whole “dual local HTEX with address fiddling” complexity as a *design target*; you now only need robust `local.yaml` with `address=127.0.0.1` and spawn. The subtle ZMQ / VPN discussion is still useful, but that’s a separate bugfix, not core design.
* Stop talking about “Slurm executor” vs “HTEX executor” as if you needed both. In practice, you’ll have a **single HTEX** plus a **SlurmProvider** with `nodes_per_block=1`.
* For “let Slurm schedule many small jobs”, you now achieve that by:

  * short Slurm blocks,
  * `nodes_per_block=1`,
  * high `max_blocks`,
  * one worker per node.
* UX plan becomes: **backend switch**, not “remember which card path you like”.

---

If you want, I can now write a concrete “coding agent plan” to implement:

* `--backend` wiring,
* updated `local.yaml`, `cpu.yaml`, `gpu-a100.yaml`,
* and the minimal doc tweaks so that everything is internally consistent.
