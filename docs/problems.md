## Target Problems, Teachers, and Data Generation

This project works with **synthetic regression problems** built from three YAML blocks:

- `model`: the **student** neural network we will train and later sample around
- `data`: the input distribution and noise model
- `teacher`: an optional **teacher** network used only to generate synthetic labels

Each row in `config/experiments.yaml` under `targets:` picks a triple `(model, data, teacher)` plus a Stage‑A seed:

```yaml
targets:
  - { model: small, data: small, teacher: _null, seed: 42 }
  - { model: base,  data: base,  teacher: _null, seed: 43 }
```

The Stage‑A workflow (`lambda-hat workflow llc`) turns each such row into a **target artifact** in `runs/targets/<target_id>/`. That artifact is the thing Stage‑B samplers (HMC, SGLD, MCLMC, VI) see as “the problem”.

---

### Terminology

**Problem / target problem**

> An abstract specification: `(model preset, data preset, teacher preset, target.seed, training config)`.

This is what you write in `config/experiments.yaml` or an Optuna problem list.

**Target**

> The concrete artifact built from a problem: dataset, ERM solution, and metadata on disk.

Targets live under `runs/targets/tgt_<hash>/` and are identified by a **content‑addressed target ID** that hashes the Stage‑A config (model + data + teacher + training + target seed).

**Teacher**

> A neural network (or simple function) that defines the true regression function used to generate synthetic labels.

The teacher is configured under `lambda_hat/conf/teacher/*.yaml` and built by `build_teacher` in `lambda_hat/data.py`. If you set `teacher: _null`, we fall back to using the **student model architecture** as the teacher (same `model` config, independent parameters).

**Student / model**

> The neural network we actually train (ERM) and around which we define the local posterior.

This is configured via `lambda_hat/conf/model/*.yaml` and instantiated by `build_mlp` in `lambda_hat/nn_eqx.py`. It defines the parameter space where LLC is measured.

---

### True vs inferred data‑generation processes

There are two distinct stories:

1. **True data‑generation process (synthetic world)**
   This is implemented in `lambda_hat/data.py::make_dataset` and depends on `data`, `teacher`, and `target.seed`. Given a problem config:

   1. Sample inputs (X) from the chosen `x_dist` (e.g. `gauss_iso`, `mixture`, `lowdim_manifold`, `heavy_tail`) using `sample_X`.
   2. Build the **teacher network** with `build_teacher(key, cfg)`:

      * If `teacher` preset is non‑empty (e.g. `teacher/base.yaml`), its `depth`, `activation`, and sizing hints (`target_params` or `hidden`) define a separate architecture.
      * If `teacher: _null`, the teacher **reuses the student architecture** (`model` block) but still gets its own random parameters.
   3. Compute noise‑free teacher outputs (f_\text{teacher}(X)).
   4. Optionally apply **dropout at the teacher level** if `teacher.dropout_rate > 0` (e.g. `teacher/base.yaml` uses `dropout_rate: 0.1`).
   5. Add observation noise according to `data.noise_model` (`gauss`, `hetero`, `student_t`, `outliers`) and `noise_scale` via `add_noise(...)`.

   This defines the **true regression model**

   $$
   X \sim p_X(\text{data config}),\quad
   Y \mid X \sim p(Y \mid X; \text{teacher config}, \text{noise model}).
   $$

2. **Inferred model / local posterior (what samplers see)**
   Stage‑A then trains the **student model** (f_\theta) on the generated dataset ((X, Y)) by ERM (`lambda_hat/training.py::train_erm`). The ERM solution (w_0) (stored as `params0.eqx`) is treated as the center of the local posterior.

   Stage‑B samplers see only:

   * the dataset ((X, Y)) in `data.npz`
   * the ERM parameters `params0.eqx`
   * the posterior configuration (`loss`, `beta_mode`, `gamma`, etc.)

   The **tempered local posterior** ( \pi(w) ) is defined in `lambda_hat/posterior.py` and the methodology doc as

   $$
   \pi(w) \propto \exp\left(-\frac{\gamma}{2}|w-w_0|^2 - n\beta L_n(w)\right),
   $$

   where (L_n) is the **student** loss, not the teacher loss. LLC is computed as

   $$
   \hat\lambda \approx n\beta(\overline{L_n(w)} - L_n(w_0)),
   $$

   using samples from this local posterior.

So:

* The **true world** is defined by `data + teacher + target.seed`.
* The **inference world** is defined by `model + training + posterior + sampler`, conditioned on the dataset produced above.

They only coincide when the student architecture matches the teacher (`teacher: _null` or you manually align them).

---

### How well specified is a problem?

Given:

* `model` preset (`lambda_hat/conf/model/*.yaml`)
* `data` preset (`lambda_hat/conf/data/*.yaml`)
* `teacher` preset (`lambda_hat/conf/teacher/*.yaml`)
* Stage‑A training block (`lambda_hat/conf/workflow.yaml::training`)
* `target.seed`

the synthetic problem is **fully determined up to the code SHA** recorded in `meta.json`. The target artifact records:

* the merged Stage‑A config (model, data, teacher, training, target seed)
* dimensions / parameter counts
* ERM loss (L_0) and other metrics
* the precise code SHA and versions used

This means:

* **Re-running Stage‑A with the same config and code** reproduces the **identical dataset and ERM solution** (content‑addressed `target_id`).
* Any ambiguity in “what problem am I actually solving?” comes from *visibility*, not from under‑specification: the DGP lives in `data.py` + `teacher` YAML, while the inference model lives in `model` + `posterior` + `sampler`.

This doc is meant to close that visibility gap.

---

### Teacher presets vs model presets

* `model/*.yaml` controls the **student**:

  * `in_dim`, `out_dim`
  * `depth`
  * sizing via `target_params` or `hidden`
  * `activation`, `bias`, `layernorm`, `init`

* `teacher/*.yaml` optionally overrides the **teacher**:

  * `depth`, `activation`, `dropout_rate`
  * sizing via `target_params` or `hidden`
  * if both `target_params` and `hidden` are null and `widths` is null, we **fallback to model sizing** (teacher and student architectures match)

If you use `teacher: _null`, the teacher config is effectively empty and the data is generated by directly using the student architecture as the teacher. This is the simplest “well‑specified” case where the inference model matches the true regression function class.

---

### Examples

**Matched teacher–student (default)**

```yaml
targets:
  - model: small
    data: small
    teacher: _null
    seed: 42
```

* Teacher and student share architecture (`model/small.yaml`); parameters are different random draws.
* True DGP: `X ~ gauss_iso`, `Y = f_teacher(X) + Gaussian noise`.
* Inference: train the same architecture `f_θ` to approximate `f_teacher` under MSE; sample around ERM solution.

**Mismatched teacher–student**

```yaml
targets:
  - model: base
    data: base
    teacher: small
    seed: 123
```

* Teacher uses `teacher/small.yaml` (shallower, different width and dropout) while student uses `model/base.yaml` (larger architecture).
* True DGP is “small + dropout + noise”.
* Student is a potentially over‑parameterized model trying to fit that DGP.
* LLC now probes a **mismatched** model class, which is exactly where singular learning theory is interesting.

**Quadratic targets**

For `target: quadratic`, Stage‑A bypasses the teacher and dataset entirely and constructs an analytic quadratic loss:

$$
L_n(w) = \frac{1}{2}|w|^2
$$

in `quad_dim` dimensions, with a Gaussian localizer. This is mainly for baseline LLC sanity checks and comparisons to the neural‑network case.

---

### Summary

* A **problem** is fully specified by `(model, data, teacher, training, target.seed)` + code SHA.
* The **true DGP** is `data + teacher + target.seed`.
* The **inference model** is `model + posterior + sampler`, centered at the ERM solution learned from the synthetic data.
* Teacher presets allow you to decouple “ground‑truth function” from “inference architecture”; setting `teacher: _null` collapses back to a well‑specified teacher–student model.
