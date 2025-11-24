For a smoke test, I’d go ruthlessly small and boring: one tiny problem, all samplers, aggressively cut-down steps so the whole grid runs in a couple of minutes and still exercises the full pipeline.

### 1. Config: `config/experiments_smoke.yaml`

Create a new file `config/experiments_smoke.yaml`:

```yaml
experiment: "smoke_all_samplers"
jax_enable_x64: true  # simplest: run everything in float64

# One tiny problem: small model, small data, no teacher
targets:
  - model: small
    data: small
    teacher: _null
    seed: 0
    overrides:
      data:
        n_train: 128
        n_test: 64
      training:
        steps: 200        # very short ERM run
        eval_every: 50

# All samplers, each in "toy" mode
samplers:
  # Full-batch HMC
  - name: hmc
    seed: 1
    overrides:
      num_warmup: 50
      num_samples: 50
      thinning: 1
      max_tree_depth: 5    # if supported; otherwise drop

  # Microcanonical LMC
  - name: mclmc
    seed: 2
    overrides:
      num_warmup: 50
      num_samples: 50
      thinning: 1
      trajectory_length: 1.0

  # Minibatch SGLD
  - name: sgld
    seed: 3
    overrides:
      steps: 500          # total SGLD steps
      batch_size: 64
      step_size: 1e-5
      eval_every: 100     # only a few checkpoints

  # Variational inference (mean-field / low-rank)
  - name: vi
    seed: 4
    overrides:
      algo: mfa
      M: 4                # mixture components
      r: 1                # rank
      whitening_mode: adam
      steps: 1000
      eval_every: 100
```

Notes/intent:

* Single target ⇒ any failure is clearly plumbing/sampler, not interaction between multiple problems.
* Tiny dataset and short training ⇒ fast build stage.
* For each sampler: 10–100-ish “units of work” so you see at least one diagnostics point and a couple of checkpoints, but runtime stays short.
* You can tighten even further if you want this to be something you run on every commit.

If your actual sampler config keys differ (e.g. `num_samples_total` vs `num_samples`, `n_steps` vs `steps`), keep the structure and rename the knobs; the idea is “small counts everywhere”.

### 2. Commands to run the smoke test

From repo root:

```bash
# Ensure deps are installed
uv sync --extra cpu

# Run the smoke experiment locally
uv run lambda-hat workflow sample \
  --config config/experiments_smoke.yaml \
  --backend local
```

Optional quick diagnostics:

```bash
# Check everything produced sensible outputs
uv run lambda-hat diagnose-experiment \
  --experiment smoke_all_samplers \
  --mode light

# (if you like) validate the target itself
uv run lambda-hat diagnose-targets \
  --experiment smoke_all_samplers
```

### 3. bonus

And if you want to squeeze more value out of it:

- Make this the default CI job: changes that break only one sampler or only the diagnostic stack will fail quickly.
- Add a variant with `jax_enable_x64: false` and a GPU Parsl card to sanity-check the mixed-precision / accelerator path separately.
