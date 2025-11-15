# Flow VI PRNG Key Format Issue (RESOLVED)

**Status**: ✅ **FULLY RESOLVED** - Flow VI now works with Parsl workflows
**Affects**: Flow algorithm only (MFA unaffected)
**Severity**: None (all issues fixed)
**Created**: 2025-11-15 during variational_snakemake merge
**Resolved**: 2025-11-15 via typed threefry2x32 keys + vmap-safe returns (QA team guidance)
**Remaining**: None - see flow_vmap_issues.md for complete resolution details

---

## Summary

The Flow-based VI algorithm fails when executed via Parsl workflows due to PRNG key format incompatibilities between JAX legacy keys and FlowJAX's RBG key requirements, specifically when keys are traced inside `jax.vmap`.

## Error Signature

```
TypeError: JAX encountered invalid PRNG key data: expected key_data.shape to
end with (4,); got shape=(2,) for impl=PRNGImpl(...name='rbg', tag='rbg')
```

**Location**: `lambda_hat/vi/flow.py:463` during `jax.random.split(rng_key, 3)` inside vmapped execution

## Root Cause Analysis

### Key Format Evolution in JAX

JAX has two PRNG key formats:
1. **Legacy format**: `shape=(2,)` uint32 array (Threefry PRNG)
2. **RBG format**: `shape=(4,)` uint32 array (default in JAX 0.7+)

FlowJAX (v17.2.1) requires typed RBG keys and fails when receiving legacy keys.

### The Vmap Problem

In `lambda_hat/sampling.py:868`, VI algorithms are executed via:
```python
results = jax.vmap(run_one_chain)(chain_keys)
```

When `run_one_chain` calls `algo.run(rng_key=chain_key, ...)`, the `rng_key` inside the vmapped function becomes a **traced/abstract value**. Any operation requiring concretization (like `int(rng_key[0])`) fails with `ConcretizationTypeError`.

### Original Variational_Snakemake Solution (Doesn't Work with Vmap)

Commits `0f4b5fc` and `ad70cbf` in variational_snakemake added legacy key conversion:
```python
if hasattr(rng_key, "shape") and rng_key.shape == (2,):
    seed = int(rng_key[0])  # ← FAILS inside vmap (ConcretizationTypeError)
    rng_key = jax.random.key(seed)
```

This works in non-vmapped contexts (e.g., unit tests) but breaks under vmap.

### Why MFA Works But Flow Doesn't

- **MFA**: Uses standard JAX random operations that tolerate both key formats
- **Flow**: FlowJAX library internally validates key format and requires RBG

## Attempted Fixes (All Failed)

### Attempt 1: Remove legacy conversion logic
```python
# lambda_hat/vi/flow.py:461-463 (CURRENT STATE)
# Modern JAX (0.7.1+) handles key format conversion automatically
key_init, key_train, key_eval = jax.random.split(rng_key, 3)
```
**Result**: Still fails - JAX doesn't auto-convert legacy → RBG when required by downstream libs

### Attempt 2: Convert keys at run_vi entry point
```python
# lambda_hat/sampling.py:775-780 (CURRENT STATE)
key_data = jax.random.key_data(rng_key)
if key_data.shape[-1] == 2:
    key_data = jnp.concatenate([key_data, key_data], axis=-1)  # Pad to (4,)
rng_key = jax.random.wrap_key_data(key_data)
```
**Result**: Failed - padding doesn't create valid RBG keys, just corrupts legacy keys

### Attempt 3: Try-except with fallback
Attempted exception-based detection but still required concretization for conversion.

## Test Results

### Passing
- ✅ `tests/test_vi_integration.py` - All 4 tests pass (no vmap, direct algo calls)
- ✅ MFA with Parsl workflow - Full end-to-end success
- ✅ Flow unit tests (when not vmapped)

### Failing
- ❌ Flow with Parsl workflow
- ❌ Flow smoke test: `uv run python flows/parsl_llc.py --local --config test_flow_smoke.yaml`

**Error log**: `logs/run_sampler/tgt_*_vi_*.err`

## Potential Solutions (Future Work)

### Option 1: Ensure RBG keys from source
Convert all PRNG keys to RBG format at the earliest entry point (e.g., `sampling_runner.py` or `entrypoints/sample.py`) before they reach `run_vi`. This avoids vmap issues by converting before tracing.

**Pros**: Clean, centralized fix
**Cons**: May affect other samplers if they rely on legacy key format

### Option 2: Disable vmap for Flow
Run Flow algorithm without vmap (single chain or sequential chains):
```python
if config.algo == "flow":
    # Run chains sequentially (no vmap)
    results = [run_one_chain(k) for k in chain_keys]
else:
    # Standard vmapped execution
    results = jax.vmap(run_one_chain)(chain_keys)
```

**Pros**: Simple workaround
**Cons**: Slower for multi-chain runs; special-case logic

### Option 3: FlowJAX key handling wrapper
Create a vmap-safe key conversion wrapper inside Flow's `run()` method using JAX's `jax.lax.cond` for conditional key wrapping (trace-compatible).

**Pros**: Isolated to Flow implementation
**Cons**: Complex, may still hit FlowJAX internal validation

### Option 4: Upstream fix in FlowJAX
Request FlowJAX to support legacy key formats or provide a robust conversion utility.

**Pros**: Fixes root cause
**Cons**: External dependency, timeline uncertain

## Recommended Next Steps

1. **Immediate**: Document Flow as experimental/broken with Parsl in CLAUDE.md and README
2. **Short-term**: Implement Option 1 (centralized RBG conversion) in `sampling_runner.py`
3. **Long-term**: Test with newer FlowJAX versions for improved key handling

## References

- JAX PRNG design: https://jax.readthedocs.io/en/latest/jep/263-prng.html
- FlowJAX key requirements: Commits `0f4b5fc`, `ad70cbf`, `2abee91` in variational_snakemake
- Vmap tracing constraints: https://docs.jax.dev/en/latest/errors.html#jax.errors.ConcretizationTypeError

## Related Files

- `lambda_hat/vi/flow.py` - Flow algorithm implementation
- `lambda_hat/sampling.py:735-868` - `run_vi()` function with vmap
- `tests/test_vi_integration.py` - Passing tests (no vmap)
- `test_flow_smoke.yaml` - Minimal failing test case
- `logs/run_sampler/tgt_*_vi_*.err` - Error logs from failed runs

---

**Last updated**: 2025-11-15
**Next review**: When attempting to fix or when FlowJAX updates
