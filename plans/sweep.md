
## 1. Minimal working recipe

### 1.1. Understand the moving pieces

Configs are composed as:

* Stage A (build target): `model` + `data` + `teacher` + `training`.
* Stage B (sample): `sampler` + `posterior` + **reference to a built target**.

User-facing sweeps live in `config/experiments.yaml`:

```yaml
experiment: "dev"
jax_enable_x64: true

targets:
  - { model: small, data: small, teacher: _null, seed: 42 }
  - { model: base,  data: base,  teacher: _null, seed: 43,
      overrides: { training: { steps: 10000 } } }

samplers:
  - { name: hmc }
  - { name: mclmc }
  - { name: sgld, overrides: { step_size: 1e-6, eval_every: 50 }, seed: 12345 }
  - { name: vi, overrides: { algo: mfa, M: 8, r: 2, gamma: 0.001 }, seed: 54321 }
```



Teachers live under `lambda_hat/conf/teacher/` (`_null.yaml`, `base.yaml`, `small.yaml`, etc.), and are wired into the data generator as the “true” regression function.

Crucially: **teacher ≠ target**; the *target* is the student model + dataset + ERM solution, while `teacher` only controls the generative side.

---

### 1.2. Create a teacher–student mismatch sweep

Goal: a sweep that:

* Uses multiple samplers: HMC, MCLMC, SGLD, VI.
* Uses **mismatched teacher/student** so LLC is non-trivial. Docs already give the canonical example:

```yaml
targets:
  - model: base
    data: base
    teacher: small
    seed: 123
```

#### Step 1 — Make a dedicated experiments file

From repo root:

```bash
cp config/experiments.yaml config/experiments_teacher_sweep.yaml
```

Edit `config/experiments_teacher_sweep.yaml` to something like:

```yaml
experiment: "teacher_sweep"
jax_enable_x64: true  # HMC/MCLMC want float64; SGLD/VI handle casts internally

targets:
  # 0. Matched baseline (for comparison)
  - { model: small, data: small, teacher: _null, seed: 42 }

  # 1. Teacher smaller than student (mismatched)
  - { model: base, data: base, teacher: small, seed: 101 }

  # 2. Teacher larger than student (mismatched)
  - { model: small, data: base, teacher: base, seed: 102 }

samplers:
  # Full-batch MCMC (double precision)
  - { name: hmc }

  # Microcanonical variant
  - { name: mclmc }

  # Minibatch SGLD, tuned for ~10k FGEs by default conf/sampler/sgld.yaml
  - { name: sgld,
      seed: 2001,
      overrides: { step_size: 1e-6, batch_size: 256, eval_every: 100 } }

  # Variational inference (MFA)
  - { name: vi,
      seed: 2002,
      overrides: {
          algo: mfa,
          M: 8,
          r: 2,
          whitening_mode: adam,
          steps: 5000,
          eval_every: 50
      }}
```

This is just the existing `experiments.yaml` pattern, plus:

* a matched baseline, and
* two **mismatched** `(model, teacher)` choices.

The mismatched pattern (“teacher small, student base”) is exactly what `docs/problems.md` describes as the interesting SLT case.

If you want more exotic teachers (e.g. deeper or with dropout), add extra presets under `lambda_hat/conf/teacher/*.yaml` and reference them by name here; `experiments.md` already shows how to add new presets.

#### Step 2 — Run the sweep (local)

Commands:

```bash
# From repo root; ensure deps installed
uv sync --extra cpu

# Run the N × M sweep
uv run lambda-hat workflow llc \
  --config config/experiments_teacher_sweep.yaml \
  --backend local
```

This uses the standard workflow entrypoint described in README and `docs/workflows.md`.

You’ll get:

* 3 targets × 4 samplers = 12 sampling jobs + 3 builds. The “N × M jobs from `experiments.yaml`” behaviour is the same as the documented basic sweep.

Artifacts go under the experiment namespace, e.g.:

* `artifacts/experiments/teacher_sweep/targets/tgt_*/...`
* `artifacts/experiments/teacher_sweep/runs/*hmc*`, `*mclmc*`, `*sgld*`, `*vi*`.

#### Step 3 — Run on SLURM (optional)

Using the existing Parsl cards:

```bash
# CPU cluster
uv run lambda-hat workflow llc \
  --config config/experiments_teacher_sweep.yaml \
  --parsl-card config/parsl/slurm/cpu.yaml

# A100 GPU profile (if you want JAX-on-GPU for SGLD/VI)
uv run lambda-hat workflow llc \
  --config config/experiments_teacher_sweep.yaml \
  --parsl-card config/parsl/slurm/gpu-a100.yaml
```

#### Step 4 — Diagnostics & galleries (optional but useful)

After the sweep finishes:

```bash
# Generate diagnostics (trace, rank, LLC convergence, etc.) for all runs
uv run lambda-hat diagnose-experiment --experiment teacher_sweep --mode light

# Generate target-level teacher diagnostics for all mismatched targets
uv run lambda-hat diagnose-targets --experiment teacher_sweep

# Promote plots into a gallery
uv run lambda-hat workflow llc \
  --config config/experiments_teacher_sweep.yaml \
  --backend local  --promote
```

The workflows/docs already describe diagnose / promote; this just reuses them on your new experiment.

---

## 2. Draft HOWTO text for `README.md`

Drop something like this under the existing “Workflow” or “Config” sections.

````markdown
## HOWTO: Sweep over samplers with mismatched teacher / student

This repo’s experiment grid is defined in `config/experiments.yaml`:
each `targets:` row is a problem `(model, data, teacher, seed)`,
and each `samplers:` row is a sampling method/config. The workflow
`lambda-hat workflow llc` runs the full N × M grid in parallel.

To get **non-trivial LLC values** from a mismatched teacher–student
setup, create a dedicated experiments file:

```bash
cp config/experiments.yaml config/experiments_teacher_sweep.yaml
````

Edit `config/experiments_teacher_sweep.yaml`:

```yaml
experiment: "teacher_sweep"
jax_enable_x64: true  # HMC/MCLMC in float64; SGLD/VI cast internally

targets:
  # Matched baseline
  - { model: small, data: small, teacher: _null, seed: 42 }

  # Teacher smaller than student (mismatched)
  - { model: base, data: base, teacher: small, seed: 101 }

  # Teacher larger than student (mismatched)
  - { model: small, data: base, teacher: base, seed: 102 }

samplers:
  - { name: hmc }
  - { name: mclmc }
  - { name: sgld,
      seed: 2001,
      overrides: { step_size: 1e-6, batch_size: 256, eval_every: 100 } }
  - { name: vi,
      seed: 2002,
      overrides: {
          algo: mfa,
          M: 8,
          r: 2,
          whitening_mode: adam,
          steps: 5000,
          eval_every: 50
      }}
```

Then run the sweep:

```bash
# Local
uv run lambda-hat workflow llc \
  --config config/experiments_teacher_sweep.yaml \
  --backend local

# SLURM (examples)
uv run lambda-hat workflow llc \
  --config config/experiments_teacher_sweep.yaml \
  --parsl-card config/parsl/slurm/cpu.yaml
```

This produces 3 targets × 4 samplers = 12 sampling runs, plus
3 target builds, all recorded under
`artifacts/experiments/teacher_sweep/`. Use
`lambda-hat diagnose-experiment` and `lambda-hat diagnose-targets`
to generate diagnostics for the runs and for the teacher–student
mismatch plots.

````

---

## 3. Draft additions for `docs/experiments.md`

You already document basic sweeps and sampler-hyperparameter sweeps.:contentReference[oaicite:20]{index=20} :contentReference[oaicite:21]{index=21}

Add a subsection under “Sweep Patterns”:

```markdown
#### 4. Teacher–student mismatch sweeps (non-trivial LLC)

To probe **singular / misspecified** behaviour, use a teacher network
that does *not* match the student architecture. See
[problems.md](./problems.md) for a detailed explanation of teacher vs
student and the resulting data-generating process.

Example (`config/experiments_teacher_sweep.yaml`):

```yaml
experiment: "teacher_sweep"
jax_enable_x64: true

targets:
  # Matched baseline
  - { model: small, data: small, teacher: _null, seed: 42 }

  # Mismatched: teacher smaller than student
  - { model: base, data: base, teacher: small, seed: 101 }

  # Mismatched: teacher larger than student
  - { model: small, data: base, teacher: base, seed: 102 }

samplers:
  - { name: hmc }
  - { name: mclmc }
  - { name: sgld,
      seed: 2001,
      overrides: { step_size: 1e-6, batch_size: 256, eval_every: 100 } }
  - { name: vi,
      seed: 2002,
      overrides: { algo: mfa, M: 8, r: 2, whitening_mode: adam }}
````

Run the full grid:

```bash
uv run lambda-hat workflow llc \
  --config config/experiments_teacher_sweep.yaml \
  --backend local
```

Use `lambda-hat diagnose-experiment` to generate sampler diagnostics,
and `lambda-hat diagnose-targets` to regenerate teacher comparison plots
for all targets in the experiment.

````

---

## 4. Draft addition for `docs/problems.md`

You already explain matched vs mismatched teacher–student, with an
example `(model: base, teacher: small)`.:contentReference[oaicite:24]{index=24}

Add a short “recipe” block near that example:

```markdown
### Recipe: non-trivial LLC via mismatched teacher

To force LLC to see a **misspecified model class**, pick different
presets for `model` and `teacher` in `config/experiments.yaml`:

```yaml
targets:
  - model: base
    data: base
    teacher: small
    seed: 123
````

Here:

* The **true DGP** uses `teacher/small.yaml` (shallower, often with dropout),
* The **student** uses `model/base.yaml` (larger architecture), and
* The samplers see only the student + dataset produced by the teacher.

Add this row to `targets:` alongside your baseline `teacher: _null` rows,
then reference it from `samplers:` as usual (HMC, MCLMC, SGLD, VI).
The standard `lambda-hat workflow llc` will build the target and run all
samplers on the mismatched problem.

```

---

## 5. Optional tweak for `docs/samplers.md` or `docs/workflows.md`

You could add a one-liner cross-link:

- In `docs/samplers.md` “Choosing a sampler”, add:

> For teacher–student mismatch experiments that stress singular behaviour,
> see the “Teacher–student mismatch sweeps” recipe in
> [experiments.md](./experiments.md).

- In `docs/workflows.md` under “Parameter Sweeps”, add a bullet:

> * Teacher–student mismatch: configure `model` and `teacher` presets
>   differently in `config/experiments.yaml` (see `docs/problems.md`).
