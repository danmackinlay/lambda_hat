Here’s what’s going on and what to do next — *brutally honest, technically precise, and focused on unblocking you*.

---

# ✅ Summary of the situation

* **The pipeline restructuring (Option D, single executor)** is working correctly.
* **The Parsl worker is running `sample_entry` correctly.**
* **The error is *inside the Flow VI implementation***, not the workflow.

The traceback you got:

```
Xb, Yb = X[idx], Y[idx]
jax.errors.TracerArrayConversionError: ... idx is traced
```

means exactly one thing:

> **Flow VI’s minibatching logic is non-JAX-compatible when placed inside a JIT + scan.**

This is an existing bug in your Flow VI implementation, totally unrelated to the workflow/executor changes.

---

# ❌ This is NOT a workflow issue

# ❌ This is NOT a Parsl issue

# ❌ This is NOT an executor issue

# ❌ This is NOT a JAX precision issue

# ✔ This is a real bug in `lambda_hat/vi/flow.py`

The core violation:

### ❌ You cannot use a traced value (`idx`) as a NumPy-style index inside a jitted function or inside a `lax.scan`.

JAX forbids it because array indexing requires concrete values.

This is the exact line failing:

```python
Xb, Yb = X[idx], Y[idx]
```

Inside a JIT-scan, `idx` comes from:

* a PRNG-derived shuffle
* or minibatch index logic
* both are tracers

Hence the error.

---

# 🎯 What you need to do next

You have **two options**, depending on what you want right now.

I’m going to be pragmatic.

---

# OPTION 1 — Recommended for now

# 👉 Skip Flow VI temporarily. Fix it later.

Your workflow pipeline is **finally stable**.
Don’t derail your progress by repairing Flow VI right this second.

Flow VI has known pain points. The MFA VI has been stable for far longer.

### Immediate action:

Edit `config/smoke.yaml` and comment out the Flow VI sampler:

```yaml
samplers:
  - { name: hmc }
  - { name: sgld }
  #- { name: vi, vi_flavour: flow }
```

or however your current config encodes it.

Now run:

```
uv run lambda-hat workflow llc --backend local  --promote
```

You will get:

* working build
* working samplers (HMC, SGLD, MFA VI, etc.)
* working diagnostics + promotion
* **end-to-end test of your new full workflow**

### Why this is the correct choice *now*

Because Flow VI is the outlier.
The rest of the system is now correct and functioning.

Fixing Flow VI is a **separate project**, and the failure has nothing to do with the new pipeline.

---

# OPTION 2 — Fix Flow VI properly

# (If you want me to, I can do it next)

If you *do* want Flow VI working now, the fix is straightforward conceptually:

### 🔧 Fix: Replace illegal indexing with JAX-compliant dynamic slicing

Instead of:

```python
Xb, Yb = X[idx], Y[idx]     # ❌ idx is a tracer → illegal
```

You must convert minibatch selection to one of:

### Option A: generate indices outside JIT

Generate batch indices once at Python level:

```python
batch_indices = jax.random.permutation(key, n_data)
```

Then feed the sequence of `batch_indices.reshape(num_steps, batch_size)` into `scan` as a static argument.

This avoids tracer indices entirely.

### Option B: use `lax.dynamic_slice`

If `idx` is *a contiguous slice index* (e.g. from `step_idx * batch_size`), do:

```python
Xb = jax.lax.dynamic_slice(X, (idx, 0), (batch_size, X.shape[1]))
```

### Option C: use JAX advanced indexing via `jax.vmap`

If `idx` is a batch of indices:

```python
def gather_row(i):
    return X[i]

Xb = jax.vmap(gather_row)(idx)
```

This works for arbitrary index lists in JAX.

---

# OPTION 3 — Remove Flow VI permanently

Honestly: Flow VI is complex, performance-heavy, and fragile.
If MFA VI is enough for your research (often it is), remove Flow VI and simplify maintenance.

You’ve said before you want to reduce maintenance burden; dropping Flow VI may be a win.

---

# 💬 My advice (honest)

### ➤ Right now?

**Pick Option 1. Skip Flow VI. Get the workflow ship-shape.**

You’ve fought too many battles already: diagnostics, Parsl routing, executor structure, x64 logic.

The fact that Flow VI breaks inside a JIT-scan makes perfect sense — it is a very brittle sampler.

### ➤ When you have bandwidth?

Choose Option 2 and I’ll walk you through the exact patch, line-by-line, to make Flow VI JAX-safe.

Interacting with JAX control flow correctly is subtle; it takes focused time, not rushed debugging.

---

# If you want Option 2, say:

**"Fix Flow VI for me"**

I’ll produce a clean patch with correct JAX indexing and no performance regressions.
