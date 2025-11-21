# Executor Routing Problem

**Date**: 2025-11-20
**Status**: Workaround implemented; proper solution needed
**Related Files**: `lambda_hat/workflows/parsl_llc.py`, `config/parsl/local.yaml`

## Executive Summary

The Lambda-Hat workflow attempted to implement dynamic executor routing to run float64 tasks on `htex64` and float32 tasks on `htex32`. This feature was non-functional due to incorrect Parsl API usage. The `executor=...` parameter was being passed to Parsl app functions, but Parsl was not intercepting it, resulting in `TypeError: got an unexpected keyword argument 'executor'`.

**Immediate Fix**: Removed executor routing; all tasks now run on any available executor.
**Impact**: Works but suboptimal (no precision-based executor affinity).

## Background

### Why Dual Executors?

The workflow processes mixed-precision tasks:
- **float64**: HMC, MCLMC (MCMC samplers requiring double precision)
- **float32**: SGLD, VI (stochastic methods using single precision)

Dual executors were intended to optimize resource usage:
- `htex64`: Configured with `JAX_ENABLE_X64=1` for float64
- `htex32`: Default JAX precision for float32

### Attempted Implementation

```python
# parsl_llc.py:297-302 (NOW REMOVED)
future = build_target_app(
    cfg_yaml=str(cfg_yaml_path),
    target_id=tid,
    experiment=experiment,
    executor=build_executor,  # ❌ This doesn't work
)
```

**Problem**: The `@python_app` decorator defines:
```python
@python_app
def build_target_app(cfg_yaml, target_id, experiment):
    ...  # No 'executor' parameter
```

Parsl did not intercept the `executor` kwarg before passing it to the function.

## Root Cause Analysis

### What We Tried

1. **Passing `executor` at call time** (current attempt):
   ```python
   future = my_app(..., executor="htex64")
   ```
   - Expected: Parsl intercepts and routes to specified executor
   - Actual: `TypeError: my_app() got an unexpected keyword argument 'executor'`

2. **Why it failed**:
   - Parsl 2025.11.10 may not support dynamic executor selection via call-time kwargs
   - Or requires specific configuration/API that wasn't used
   - The parameter gets passed directly to the Python function instead of being intercepted by Parsl

### Parsl Executor Routing Options (Research Needed)

#### Option A: Static Declaration in Decorator
```python
@python_app(executors=['htex64'])
def float64_app(...):
    ...

@python_app(executors=['htex32'])
def float32_app(...):
    ...
```
**Pros**: Simple, explicit, guaranteed to work in Parsl 2024+
**Cons**: Requires duplicate app definitions for each executor

**Implementation sketch**:
```python
# Define separate apps for each precision
@python_app(executors=['htex64'])
def build_target_app_64(cfg_yaml, target_id, experiment):
    from lambda_hat.commands.build_cmd import build_entry
    return build_entry(cfg_yaml, target_id, experiment)

@python_app(executors=['htex32'])
def build_target_app_32(cfg_yaml, target_id, experiment):
    from lambda_hat.commands.build_cmd import build_entry
    return build_entry(cfg_yaml, target_id, experiment)

# Route at call site
if jax_x64:
    future = build_target_app_64(...)
else:
    future = build_target_app_32(...)
```

#### Option B: Call-Time Routing (If Supported)
```python
@python_app  # Must allow dynamic routing somehow
def my_app(...):
    ...

future = my_app(..., executor="htex64")  # Parsl intercepts
```
**Pros**: One app definition, flexible routing
**Cons**: Not working in current Parsl version or requires special setup

**Research needed**:
- Check Parsl 2025.11.10 documentation for executor selection API
- Search for `executor` parameter in Parsl test suite
- Verify if `python_app()` decorator needs special configuration

#### Option C: Factory Pattern
```python
def create_app_for_executor(executor_label):
    @python_app(executors=[executor_label])
    def app(...):
        ...
    return app

build_app_64 = create_app_for_executor("htex64")
build_app_32 = create_app_for_executor("htex32")
```
**Pros**: Programmatic, scales to many executors
**Cons**: Complex, may confuse Parsl's internal tracking

#### Option D: Single Executor with Runtime JAX Config
```python
# Remove dual executors, use one HTEX
# Set JAX precision inside tasks based on config
@python_app
def my_app(cfg_yaml, ...):
    import jax
    cfg = load_config(cfg_yaml)
    if cfg["dtype"] == "float64":
        jax.config.update("jax_enable_x64", True)
    # ... rest of task
```
**Pros**: Simplest, no Parsl routing needed
**Cons**: May have performance implications; not resource-isolated

## Current Workaround

**File**: `lambda_hat/workflows/parsl_llc.py`

**Build tasks** (line 297):
```python
# Submit build job (uses artifact system via command modules)
# Note: executor routing disabled for now - both executors support all dtypes
log.info(
    "  Submitting build for %s (model=%s, data=%s)",
    tid, t["model"], t["data"],
)
future = build_target_app(
    cfg_yaml=str(cfg_yaml_path),
    target_id=tid,
    experiment=experiment,
    # executor=build_executor,  # REMOVED
)
```

**Sample tasks** (line 340):
```python
# Submit sampling job (uses artifact system via command modules)
# Note: executor routing disabled for now - both executors support all dtypes
log.info(
    "  Submitting %s for %s (run_id=%s, dtype=%s)",
    sampler_name, tid, rid, dtype,
)
future = run_sampler_app(
    cfg_yaml=str(cfg_yaml_path),
    target_id=tid,
    experiment=experiment,
    inputs=[target_futures[tid]],  # Dependency: wait for target build
    # executor=executor,  # REMOVED
)
```

**Trade-off**: Works correctly but without precision-based optimization. Both executors are configured with `JAX_ENABLE_X64=1`, so float32 and float64 tasks both work, just without affinity.

## Requirements for Proper Solution

1. **Dynamic routing**: Select executor at task submission time based on sampler dtype
2. **Minimal code duplication**: Avoid defining separate apps for each executor if possible
3. **Type safety**: Ensure float64 tasks get `JAX_ENABLE_X64=1` environment
4. **Parsl compatibility**: Work with Parsl 2025.11.10+ (current pinned version)
5. **Maintainability**: Clear, readable code that future developers can understand

## Recommended Next Steps

### Phase 1: Research (1-2 hours)
1. **Check Parsl documentation**:
   - Review Parsl 2025.11.10 release notes for executor routing
   - Search documentation for "executor selection", "dynamic routing"
   - Check if `python_app()` supports `executors` parameter

2. **Test Parsl APIs**:
   - Create minimal test script with Option A (static decorator)
   - Test if Option B works with proper Parsl configuration
   - Benchmark performance difference vs workaround

3. **Consult Parsl community**:
   - Search Parsl GitHub issues for executor routing examples
   - Check Parsl test suite for patterns

### Phase 2: Decision (design review)
Based on research findings:
- If Option B works: Document proper API usage and implement
- If Option A only: Evaluate code duplication cost vs performance gain
- If neither works well: Consider Option D (single executor) + benchmark

### Phase 3: Implementation
- Update `parsl_llc.py` with chosen solution
- Add tests to verify executor routing
- Document pattern for future workflow development
- Consider generalizing pattern if we add more executors

## Related Code Locations

### Workflow Logic
- **`lambda_hat/workflows/parsl_llc.py`**:
  - Lines 84-93: `get_executor_for_dtype()` helper (currently unused)
  - Lines 96-115: `get_sampler_dtype()` helper (still used for logging)
  - Lines 297-302: Build task submission (executor routing removed)
  - Lines 340-346: Sample task submission (executor routing removed)

### Parsl Configuration
- **`config/parsl/local.yaml`**:
  ```yaml
  htex64:
    worker_init: "export JAX_ENABLE_X64=1; export MPLBACKEND=Agg"
    max_workers: 4

  htex32:
    worker_init: "export MPLBACKEND=Agg"
    max_workers: 4
  ```

### App Definitions
- **`lambda_hat/workflows/parsl_llc.py:123-144`**:
  ```python
  @python_app
  def build_target_app(cfg_yaml, target_id, experiment):
      ...

  @python_app
  def run_sampler_app(cfg_yaml, target_id, experiment, inputs=None):
      ...
  ```

## Performance Considerations

### Current Workaround Impact
With executor routing disabled:
- **float32 tasks** (SGLD, VI) run on either executor: minimal impact since both have x64 enabled
- **float64 tasks** (HMC, MCLMC) run on either executor: works fine
- **Load balancing**: Parsl distributes tasks evenly across both executors

**Estimated overhead**: <5% (both executors can handle both precisions)

### With Proper Routing
- **float32 → htex32**: Potentially faster due to less memory overhead
- **float64 → htex64**: No change
- **Resource utilization**: Better separation of precision-specific workloads

**Expected gain**: 5-15% faster for float32 tasks (needs benchmarking)

## Testing Plan

When implementing solution:

1. **Unit test executor selection logic**:
   ```python
   def test_get_executor_for_dtype():
       assert get_executor_for_dtype("float64") == "htex64"
       assert get_executor_for_dtype("float32") == "htex32"
   ```

2. **Integration test with Parsl**:
   - Submit tasks with different dtypes
   - Verify they land on correct executors via Parsl monitoring
   - Check task completion and correctness

3. **Benchmark comparison**:
   - Run smoke test with/without routing
   - Compare total walltime and per-task timing
   - Verify float32 speedup hypothesis

## References

- Parsl documentation: https://parsl.readthedocs.io/
- Parsl GitHub: https://github.com/Parsl/parsl
- Parsl version pinned in `pyproject.toml`: `parsl>=2025.11.10`
- Related commit: ff65b51 "Update documentation to reflect dual-HTEX local executor"

## Decision Record

**Date**: 2025-11-20
**Decided**: Use workaround (no executor routing) until proper solution researched
**Decided by**: Engineering (via debugging session 2025-11-20)
**Rationale**: Unblock workflow execution; optimize later with proper Parsl API
**Review date**: TBD (assign to design team)
**Follow-up**: Research Parsl 2025.11.10 executor selection patterns and propose implementation plan
