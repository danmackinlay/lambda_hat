# Flow VI vmap Compatibility Issues - Status Report

**Date**: 2025-11-15 (Updated: All issues resolved)
**Context**: Implementing QA team's PRNG key fix for FlowVI + Parsl workflows
**Status**: ✅ **FULLY RESOLVED** - Flow VI now works with Parsl workflows

---

## Summary

All vmap compatibility issues have been **successfully resolved**. Flow VI now works correctly with Parsl workflows and multi-chain execution. All fixes implemented per QA team's guidance.

---

## Progress: What's Fixed

### ✅ Issue 1: PRNG Key Format (RESOLVED)

**Original Error:**
```
TypeError: JAX encountered invalid PRNG key data: expected key_data.shape to
end with (4,); got shape=(2,) for impl=PRNGImpl(...name='rbg', tag='rbg')
```

**Fix Implemented (per QA team recommendations):**
1. Created `lambda_hat/utils/rng.py` with `ensure_typed_key()` helper
2. Set `jax.config.update("jax_default_prng_impl", "threefry2x32")` in `lambda_hat/__init__.py`
3. Added `ensure_typed_key()` call at entry to `run_vi()` in `sampling.py`
4. Updated Parsl configs to set `JAX_DEFAULT_PRNG_IMPL=threefry2x32` in worker_init
5. Removed legacy key conversion from `flow.py` (no conversions inside vmap)

**Result:** Flow no longer crashes on PRNG key format issues. Keys are properly normalized before vmapping.

**Files Modified:**
- `lambda_hat/__init__.py` - Global JAX PRNG config
- `lambda_hat/utils/rng.py` - NEW: Host-side key normalization
- `lambda_hat/sampling.py` - Call ensure_typed_key before vmap
- `lambda_hat/vi/flow.py` - Removed int(rng_key[0]) conversion
- `parsl_config_local.py` - Set JAX_DEFAULT_PRNG_IMPL env var
- `parsl_config_slurm.py` - Added export to worker_init

---

## Remaining Issues: What's Broken

### ✅ Issue 2: Float Conversion in Vmap (RESOLVED)

**Error Encountered:**
```python
File "lambda_hat/vi/flow.py", line 622, in run
    "lambda_hat": float(lambda_hat),
                  ^^^^^^^^^^^^^^^^^
jax.errors.ConcretizationTypeError: Abstract tracer value encountered where
concrete value is expected: traced array with shape float32[]
```

**Location:** `lambda_hat/vi/flow.py:622`

**Root Cause:** Flow's `run()` method was calling `float(lambda_hat)` to convert JAX array to Python float. This requires concretization, which fails inside vmap.

**Fix Applied:**
```python
# Before (broken):
return {"lambda_hat": float(lambda_hat), ...}

# After (fixed):
return {"lambda_hat": lambda_hat, ...}  # Keep as JAX array (vmap-compatible)
```

**Rationale:** MFA returns JAX arrays directly without conversion. Flow should match this interface.

**Status:** ✅ Fixed, no longer crashes

---

### ✅ Issue 3: FlowJAX Parameters Contain Non-Data Objects (RESOLVED)

**Error Encountered:**
```python
File "lambda_hat/sampling.py", line 873, in run_vi
    results = jax.vmap(run_one_chain)(chain_keys)
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
TypeError: Output from batched function <PjitFunction of <function softplus at
0x105ed76a0>> with type <class 'jaxlib._jax.PjitFunction'> is not a valid JAX type
```

**Root Cause:** Flow's return dictionary contains FlowJAX flow parameters (from `init_injective_lift()` or optimizer state) that include **function references** (like `jax.nn.softplus`). JAX cannot vmap over functions.

**What Flow Returns:**
```python
return {
    "lambda_hat": jnp.array(...),           # ✅ OK
    "traces": {
        "llc": jnp.array(...),              # ✅ OK
        "grad_norm": jnp.array(...),        # ✅ OK
        ...
    },
    "extras": {
        "E_L": jnp.array(...),              # ✅ OK
        "L0": jnp.array(...),               # ✅ OK
        # Problem: likely here or in traces
        "flow_params": ???,                 # ❌ Contains PjitFunction references
    },
    "timings": {...},                       # ✅ OK (plain dicts)
    "work": {...},                          # ✅ OK (plain dicts)
}
```

**Hypothesis:** The `extras` dict or `traces` dict contains FlowJAX-specific state (flow parameters, bijector functions) that have function references embedded. These are not pure data and cannot be vmapped.

**Evidence:**
- MFA works fine with vmap (returns only JAX arrays and plain Python dicts)
- Flow unit tests pass (no vmap, direct calls)
- Error specifically mentions `PjitFunction` type, not data

---

## Investigation Needed

### Question 1: Where is softplus coming from?

Softplus isn't in `lambda_hat/vi/flow.py` but appears in the error. It must be embedded in FlowJAX flow parameters.

**To investigate:**
```python
# At end of flow.py run() method, before return:
import jax
print("Checking return types:")
for k, v in extras.items():
    print(f"  extras[{k!r}]: {type(v)}")
    if hasattr(v, '__dict__'):
        print(f"    -> contains: {v.__dict__.keys()}")

for k, v in traces.items():
    print(f"  traces[{k!r}]: {type(v)}, shape={getattr(v, 'shape', '?')}")
```

**Expected finding:** One of the returned objects contains a FlowJAX `Flow` instance or bijector with function attributes.

---

### Question 2: What does MFA return that Flow doesn't?

**MFA extras (from mfa.py:943):**
```python
extras = {
    "Eq_Ln": jnp.array(...),
    "Ln_wstar": jnp.array(...),
    "cv_info": {
        "Eq_Ln_mc": jnp.array(...),
        "Eq_Ln_cv": jnp.array(...),
        "variance_reduction": jnp.array(...),
    },
}
```

All pure data - no functions, no FlowJAX objects.

**Flow extras (from flow.py:606):**
```python
extras = {
    "E_L": E_L,
    "L0": L0,
    "Eq_Ln_mc": E_L,  # Flow has no HVP CV
    "Eq_Ln_cv": E_L,
    "variance_reduction": 1.0,
}
```

Looks similar... but **check if E_L or L0 contain unexpected types**.

---

## Solution Options

### Option A: Strip Non-Data from Flow Returns (Clean but Complex)

Modify `flow.py` to ensure `extras` and `traces` contain **only**:
- JAX arrays (DeviceArray/Array)
- Plain Python scalars (int, float, bool)
- Plain Python containers (dict, list) of the above

**Implementation:**
```python
def sanitize_for_vmap(obj):
    """Remove non-data objects (functions, Equinox modules, etc.)."""
    if isinstance(obj, (jnp.ndarray, jax.Array)):
        return obj  # JAX arrays are fine
    if isinstance(obj, (int, float, bool, str)):
        return obj  # Python scalars are fine
    if isinstance(obj, dict):
        return {k: sanitize_for_vmap(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(sanitize_for_vmap(v) for v in obj)
    # Drop anything else (functions, Equinox modules, etc.)
    return None

# In flow.py run() method:
extras_clean = sanitize_for_vmap(extras)
traces_clean = sanitize_for_vmap(traces)
return {"lambda_hat": lambda_hat, "traces": traces_clean, "extras": extras_clean, ...}
```

**Pros:** Fixes the root cause, Flow becomes vmap-compatible
**Cons:** May discard useful debugging info; requires testing to ensure nothing critical is lost

---

### Option B: Disable Vmap for Flow (Simple Workaround)

Run Flow chains **sequentially** instead of in parallel via vmap.

**Implementation in `sampling.py`:**
```python
# Run VI fitting and estimation for each chain
chain_keys = jax.random.split(rng_key, num_chains)

# Unified dispatch: all algorithms go through registry
algo = vi.get(config.algo)

if config.algo == "flow":
    # Flow algorithm: run chains sequentially (no vmap) due to non-data returns
    def run_one_chain_sequential(chain_key):
        return algo.run(
            rng_key=chain_key,
            loss_batch_fn=loss_batch_fn,
            loss_full_fn=loss_full_fn,
            wstar_flat=params_flat,
            unravel_fn=unravel_fn,
            data=data,
            n_data=n_data,
            beta=beta,
            gamma=gamma,
            vi_cfg=config,
        )

    # Sequential execution
    results_list = [run_one_chain_sequential(k) for k in chain_keys]

    # Manually stack results (replicate vmap structure)
    lambda_hats = jnp.stack([r["lambda_hat"] for r in results_list])
    all_traces = {k: jnp.stack([r["traces"][k] for r in results_list])
                  for k in results_list[0]["traces"].keys()}
    all_extras = results_list[0]["extras"]  # Take first chain's extras
    algo_timings = results_list[0]["timings"]  # Take first chain's timings

else:
    # MFA and other algorithms: standard vmapped execution
    def run_one_chain(chain_key):
        result = algo.run(...)
        return result

    # Vmap across chains
    results = jax.vmap(run_one_chain)(chain_keys)
    lambda_hats = results["lambda_hat"]
    all_traces = results["traces"]
    all_extras = results["extras"][0]  # Take first chain
    algo_timings = results["timings"][0]
```

**Pros:**
- Simple, drop-in workaround
- Preserves all Flow output (no data loss)
- Matches QA team's Option 2 recommendation

**Cons:**
- Slower for multi-chain Flow runs (chains run serially)
- Special-case logic (code smell)
- Doesn't scale well if num_chains > 1

---

### Option C: Run Single Chain for Flow (Simplest)

Force `num_chains=1` when using Flow algorithm.

**Implementation in `sampling_runner.py` or `sampling.py`:**
```python
if config.sampler.name == "vi" and config.sampler.vi.algo == "flow":
    num_chains = 1  # Flow doesn't support multi-chain vmap
```

**Pros:**
- Minimal code changes
- No vmap issues at all

**Cons:**
- Loses multi-chain diagnostics (Rhat, ESS cross-chain)
- Doesn't match MFA/HMC/SGLD interface
- User-unfriendly (silent behavior change)

---

## Recommended Path Forward

**Short-term (this PR):**
Implement **Option B** (disable vmap for Flow). This unblocks Parsl workflows while preserving all functionality.

**Medium-term (follow-up issue):**
Investigate **Option A** (sanitize returns) by:
1. Adding debug prints to identify which objects contain functions
2. Determining if those objects are essential or can be stripped
3. Testing that sanitized Flow still produces correct LLC estimates

**Long-term (if needed):**
Engage with FlowJAX maintainers about vmap-compatible serialization of flow parameters.

---

## Files to Review

**Modified (PRNG fix complete):**
- ✅ `lambda_hat/__init__.py`
- ✅ `lambda_hat/utils/rng.py`
- ✅ `lambda_hat/sampling.py`
- ✅ `lambda_hat/vi/flow.py`
- ✅ `parsl_config_local.py`
- ✅ `parsl_config_slurm.py`

**Needs modification (vmap workaround):**
- ⚠️ `lambda_hat/sampling.py` - Add Flow special case for vmap
- ⚠️ `lambda_hat/vi/flow.py` - OR sanitize return values

---

## Test Case

**Minimal reproduction:**
```bash
# Create test config
cat > test_flow_vmap.yaml <<EOF
store_root: "test_flow_vmap"
jax_enable_x64: false
targets:
  - { model: small, data: small, teacher: _null, seed: 42 }
samplers:
  - { name: vi, overrides: { algo: flow, steps: 10, d_latent: 4 }, seed: 100 }
EOF

# Run (will fail on Issue 3)
uv run python flows/parsl_llc.py --local --config test_flow_vmap.yaml

# Check error log
cat logs/run_sampler/tgt_*/run_vi_*/stderr
```

**Expected error (Issue 3):**
```
TypeError: Output from batched function <PjitFunction of <function softplus...
```

---

## Questions for QA Team / Experts

1. **Does FlowJAX support serializing flow parameters to pure data?**
   (i.e., can we extract just the numerical arrays and discard function references?)

2. **Is there a FlowJAX API to "freeze" a trained flow for inference only?**
   (Frozen flows might not need bijector functions in their state)

3. **Should Flow algorithm support multi-chain execution?**
   - If yes: Need to implement Option A (sanitize) or refactor Flow's return structure
   - If no: Option C (force num_chains=1) is acceptable

4. **Performance impact of sequential chains?**
   - If num_chains typically = 4, and Flow training takes ~30s/chain, sequential = 120s vs vmap = ~30s
   - Is this acceptable for Flow workflows, or is parallelism critical?

---

## Next Steps

**For immediate merge:**
1. Implement Option B (disable vmap for Flow)
2. Update `docs/flow_prng_issue.md` to note PRNG is fixed, vmap still WIP
3. Test MFA + Flow both work with Parsl (sequential chains)
4. Commit with message: "fix(flow): PRNG keys + sequential chain workaround"

**For follow-up:**
1. Create Issue: "Flow VI: Investigate vmap-compatible return structure"
2. Add detailed investigation from this doc
3. Tag FlowJAX maintainers if needed

---

## Final Solution Implemented

Per QA team's refined guidance (12-step checklist), all vmap issues were resolved by:

1. **Removed conflicting PRNG config**: Deleted `jax.config.update("jax_default_prng_impl", "rbg")` from `flow.py` line 46
2. **Removed non-data objects from returns**: Replaced `"final_dist": dist_final` (FlowJAX module) with pure JAX arrays
3. **Unified extras structure with MFA**: Standardized return format to match MFA's interface
   - Added `Eq_Ln`, `Ln_wstar`, `cv_info` structure
   - All values are JAX arrays wrapped with `jnp.asarray()` for type safety
4. **Added vmap-safety guard**: Validation in `sampling.py:872-880` to catch future non-data returns
5. **Fixed Python casts**: Removed `float(n_data)` from `flow.py:330`
6. **Fixed timings format**: Changed from `{"train": ..., "eval": ...}` to `{"adaptation": 0.0, "sampling": ..., "total": ...}` to match MFA

**Test Results After Fix:**
- ✅ Flow VI smoke test with Parsl: PASSED
- ✅ VI integration tests (4/4): PASSED
- ✅ Full test suite (33/34): PASSED (1 unrelated column naming failure)

**Files Modified:**
- `lambda_hat/vi/flow.py`: Removed rbg config, removed `final_dist`, unified extras, fixed timings
- `lambda_hat/sampling.py`: Added vmap-safety guard (already had PRNG normalization)
- `lambda_hat/__init__.py`: Global threefry2x32 config (from earlier PRNG fix)
- `lambda_hat/utils/rng.py`: Host-side key normalization (from earlier PRNG fix)
- Parsl configs: JAX_DEFAULT_PRNG_IMPL env var (from earlier PRNG fix)

---

**Last updated:** 2025-11-15
**Status:** ✅ **ALL ISSUES RESOLVED** - Flow VI production-ready with Parsl
