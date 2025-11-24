# Debugging Parsl Workflows: Extracting Worker Errors

**Date**: 2025-11-20
**Status**: Pattern established and tested
**Related Files**: `lambda_hat/workflows/parsl_llc.py`, `lambda_hat/commands/sample_cmd.py`

## The Problem: Hidden Worker Exceptions

When Parsl tasks fail, the default error reporting is often opaque and unhelpful:

```
[1/4] ✗ FAILED: tgt_deaf1a769193/hmc/f4567256
  Error: Dependency failure for task 1. The representative cause is via task 0
```

**What's happening:**
- Parsl workers run in separate processes (via HTEX)
- When a worker crashes, the exception is wrapped in Parsl's internal classes
- The actual Python traceback and error message are hidden inside the wrapper
- You only see "Dependency failure" without knowing the root cause

**Why this is frustrating:**
- No visibility into what actually went wrong
- Can't distinguish between import errors, type errors, or logic bugs
- Debugging requires manually inspecting worker log files (if they exist)
- Error propagates through dependency chains, hiding the original failure

## The Solution: Worker Error Extraction Pattern

### Step 1: Create an Error Extraction Wrapper

Add this function to your Parsl workflow file (e.g., `parsl_llc.py`):

```python
def unwrap_parsl_future(future, name: str):
    """Extract and surface exceptions from Parsl futures with full diagnostics.

    Args:
        future: Parsl AppFuture to unwrap
        name: Descriptive name for logging (e.g., "build_tgt_abc123")

    Returns:
        Future result if successful

    Raises:
        Original exception with enhanced logging of remote stdout/stderr
    """
    try:
        return future.result()
    except Exception as e:
        import traceback
        log.error(f"[{name}] FAILED in worker:")
        log.error("".join(traceback.format_exception(type(e), e, e.__traceback__)))

        # Dump remote debug info if available
        if hasattr(e, 'stdout') and e.stdout:
            log.error("---- WORKER STDOUT ----\n%s", e.stdout)
        if hasattr(e, 'stderr') and e.stderr:
            log.error("---- WORKER STDERR ----\n%s", e.stderr)

        # Log exception attributes for debugging Parsl wrappers
        log.error("Exception type: %s", type(e).__name__)
        log.error("Exception attributes: %s", dir(e))

        raise
```

**Key features:**
- Catches all exceptions from `.result()` calls
- Formats full traceback (including nested causes via `__traceback__`)
- Extracts remote stdout/stderr if attached to exception
- Logs exception metadata for debugging wrapper behavior
- Re-raises original exception to preserve workflow failure

### Step 2: Wrap All Future.result() Calls

**Before** (opaque errors):
```python
for i, (future, record) in enumerate(zip(run_futures, run_records), 1):
    try:
        future.result()  # ❌ Hides worker exceptions
        log.info("  [%d/%d] ✓ %s", i, len(run_futures), record['run_id'])
    except Exception as e:
        log.error("  [%d/%d] ✗ FAILED: %s", i, len(run_futures), record['run_id'])
        log.error("    Error: %s", str(e))  # Only sees "Dependency failure"
```

**After** (full visibility):
```python
for i, (future, record) in enumerate(zip(run_futures, run_records), 1):
    try:
        name = f"{record['target_id']}_{record['sampler']}_{record['run_id']}"
        unwrap_parsl_future(future, name)  # ✅ Extracts full traceback
        log.info("  [%d/%d] ✓ %s", i, len(run_futures), record['run_id'])
    except Exception as e:
        log.error("  [%d/%d] ✗ FAILED: %s", i, len(run_futures), record['run_id'])
        log.error("    Error: %s", str(e))
```

**What you get now:**
```
[tgt_deaf1a769193_hmc_f4567256] FAILED in worker:
Traceback (most recent call last):
  File "/path/to/sample_cmd.py", line 251, in sample_entry
    (run_dir / "traces_raw.json").write_text(json.dumps(traces_serializable, indent=2))
  File "/path/to/json/__init__.py", line 238, in dumps
    **kw).encode(obj)
  ...
TypeError: Object of type ArrayImpl is not JSON serializable

Exception type: TypeError
Exception attributes: ['__cause__', '__class__', '__context__', ...]
```

Now you can see:
- Exact line number where error occurred (`sample_cmd.py:251`)
- Full Python traceback with nested calls
- Actual exception type (`TypeError`)
- Specific error message ("Object of type ArrayImpl is not JSON serializable")

## Real-World Example: Lambda-Hat Workflow Debugging

### Initial Symptom

```bash
$ JAX_ENABLE_X64=1 uv run lambda-hat workflow llc --config config/smoke.yaml --backend local

[1/4] ✗ FAILED: tgt_deaf1a769193/hmc/f4567256
  Error: Dependency failure for task 1. The representative cause is via task 0
[2/4] ✗ FAILED: tgt_deaf1a769193/mclmc/d96c24af
  Error: Dependency failure for task 2. The representative cause is via task 0
...
```

**Dead end**: No idea what "task 0" did wrong or why it failed.

### After Adding Error Extraction

```bash
$ JAX_ENABLE_X64=1 uv run lambda-hat workflow llc --config config/smoke.yaml --backend local

[tgt_deaf1a769193_hmc_f4567256] FAILED in worker:
Traceback (most recent call last):
  ...
  File "/lambda_hat/workflows/parsl_llc.py", line 297, in <module>
    future = build_target_app(..., executor=build_executor)
TypeError: build_target_app() got an unexpected keyword argument 'executor'
```

**Breakthrough**: Now we can see:
1. The error is in our workflow code, not the worker task itself
2. We're passing an invalid `executor` parameter
3. The fix is to remove that parameter

### Bugs Found and Fixed

Using this pattern, we discovered and fixed:

1. **Executor parameter issue** (`parsl_llc.py:297,340`)
   - Error: `TypeError: got an unexpected keyword argument 'executor'`
   - Fix: Removed executor routing (documented in `docs/executor_routing.md`)

2. **JAX array serialization** (`sample_cmd.py:251`)
   - Error: `TypeError: Object of type ArrayImpl is not JSON serializable`
   - Fix: Convert JAX arrays to numpy before serialization

3. **Missing imports and parameter names** (earlier bugs)
   - Various `NameError` and `TypeError` issues caught early

## Implementation Checklist

When implementing this pattern in your Parsl workflow:

- [ ] Add `unwrap_parsl_future()` function to workflow module
- [ ] Import `logging` and create logger: `log = logging.getLogger(__name__)`
- [ ] Find all `.result()` calls on Parsl futures
- [ ] Replace `future.result()` with `unwrap_parsl_future(future, descriptive_name)`
- [ ] Test with a deliberately broken task to verify error extraction works
- [ ] Consider adding worker log directory configuration (see "Advanced Options")

## Advanced Options

### 1. Force Worker Logs to Visible Directory

By default, Parsl workers may write logs to hidden/temp directories. Force them to a known location:

**In Parsl config YAML** (`config/parsl/local.yaml`):
```yaml
executors:
  htex64:
    worker_logdir_root: /path/to/artifacts/run_dir/parsl_workers
    max_workers: 4
```

**Benefits:**
- All worker stdout/stderr in one place
- Easier to inspect when remote exceptions don't capture everything
- Can `tail -f` worker logs during execution

### 2. Instrument Entry Points

Add debug logging at the start of your Parsl app entry functions:

```python
@python_app
def build_target_app(cfg_yaml, target_id, experiment):
    """Build a target via direct command call."""
    import logging
    log = logging.getLogger(__name__)

    log.info("[WORKER START] build_target_app(target_id=%s)", target_id)

    from lambda_hat.commands.build_cmd import build_entry
    result = build_entry(cfg_yaml, target_id, experiment)

    log.info("[WORKER END] build_target_app(target_id=%s)", target_id)
    return result
```

**What this catches:**
- Import-time failures (if worker can't load modules)
- Crashes before any logging happens in the command
- Helps distinguish "worker started but crashed" vs "worker never started"

### 3. Test Imports in Workers

Create a minimal test to verify heavy imports work in Parsl workers:

```python
# scripts/test_parsl_imports.py
from parsl import python_app

@python_app
def test_heavy_imports_app():
    """Test if matplotlib/arviz work in Parsl workers."""
    import matplotlib
    import arviz
    import jax
    return f"Success: matplotlib={matplotlib.__version__}, arviz={arviz.__version__}"

# Run with your Parsl config
future = test_heavy_imports_app()
result = future.result()  # Will show import errors if they fail
print(result)
```

### 4. Inspect Composed Config YAMLs

If your workflow writes temporary config files to scratch:

```python
# In workflow
cfg_yaml_path = temp_cfg_dir / f"sample_{tid}_{sampler}.yaml"
cfg_yaml_path.write_text(OmegaConf.to_yaml(sample_cfg))
```

After a failure, inspect the composed config:
```bash
$ cat artifacts/.../scratch/configs/sample_tgt_abc_hmc_xyz.yaml
```

**Catches:**
- Config merging bugs
- Missing required fields
- Type mismatches in config values

## Common Parsl Worker Errors

### Error: "Dependency failure for task N"

**Meaning:** A task that this task depends on (via `inputs=[...]`) failed.

**Solution:**
1. Look for earlier task failures in the log
2. Use `unwrap_parsl_future()` to see the root cause
3. Fix the upstream task, not the dependent one

### Error: "ModuleNotFoundError" in worker

**Meaning:** Worker can't import a required module.

**Common causes:**
- Virtual environment not activated in worker
- Missing `worker_init` in executor config
- Module installed in different environment than worker uses

**Solution:**
```yaml
# config/parsl/local.yaml
executors:
  htex64:
    worker_init: |
      source /path/to/venv/bin/activate
      export PYTHONPATH=/path/to/project
```

### Error: "PickleError: Can't pickle <object>"

**Meaning:** Parsl can't serialize the function arguments or return value.

**Common culprits:**
- Passing file handles, database connections, or threads
- Returning non-serializable objects (e.g., JAX arrays can be tricky)

**Solution:**
- Pass file paths (strings), not file objects
- Convert JAX arrays to numpy before returning: `return np.asarray(jax_array)`
- Use primitives (int, float, str, dict, list) for return values

### Error: Worker crashes silently

**Meaning:** Worker process died without raising a Python exception.

**Common causes:**
- Out of memory (OOM killer)
- Segfault in C extension (JAX, numpy, etc.)
- Signal received (SIGKILL, timeout)

**Solution:**
1. Check `worker_logdir_root` for stderr files
2. Look for "Killed" messages (OOM)
3. Add memory limits to executor config
4. Run task manually outside Parsl to reproduce

## Pattern Variations

### Variation 1: Accumulate Errors and Continue

Instead of failing immediately, collect errors and report at end:

```python
failures = []

for i, (future, record) in enumerate(zip(run_futures, run_records), 1):
    try:
        unwrap_parsl_future(future, record['run_id'])
        log.info("  [%d/%d] ✓ %s", i, len(run_futures), record['run_id'])
    except Exception as e:
        log.error("  [%d/%d] ✗ FAILED: %s", i, len(run_futures), record['run_id'])
        failures.append({'record': record, 'error': str(e)})

# Report all failures at end
if failures:
    log.error("⚠ FAILURE SUMMARY: %d of %d runs failed", len(failures), len(run_futures))
    for f in failures:
        log.error("  • %s: %s", f['record']['run_id'], f['error'])
```

### Variation 2: Retry Failed Tasks

Wrap extraction in retry logic:

```python
def unwrap_with_retry(future, name: str, max_retries: int = 3):
    """Extract error and optionally retry on transient failures."""
    for attempt in range(max_retries):
        try:
            return unwrap_parsl_future(future, name)
        except (TimeoutError, ConnectionError) as e:
            if attempt < max_retries - 1:
                log.warning(f"[{name}] Retry {attempt+1}/{max_retries} after: {e}")
                continue
            raise
```

### Variation 3: Extract to Structured Log

For machine-readable error tracking:

```python
import json

def unwrap_to_json(future, name: str, output_path: Path):
    """Extract error and write structured JSON for analysis."""
    try:
        result = future.result()
        output_path.write_text(json.dumps({
            'name': name,
            'status': 'success',
            'result': str(result),
        }))
        return result
    except Exception as e:
        import traceback
        output_path.write_text(json.dumps({
            'name': name,
            'status': 'failed',
            'error_type': type(e).__name__,
            'error_message': str(e),
            'traceback': traceback.format_exception(type(e), e, e.__traceback__),
        }, indent=2))
        raise
```

## Why This Pattern Works

### Parsl's Exception Wrapping Mechanism

When a Parsl worker raises an exception:

1. **Worker process**: Exception occurs in `@python_app` function
2. **Parsl serialization**: Exception is pickled and sent back to main process
3. **Parsl DataFlowKernel**: Wraps exception in `AppException` or `DependencyError`
4. **Future.result()**: Re-raises wrapped exception

**The problem:** By default, you only see the wrapper (`DependencyError`), not the original exception.

**Our solution:**
- Call `.result()` to trigger the unwrapping
- Catch the exception and extract `__cause__` and `__traceback__`
- Format with `traceback.format_exception()` to show full chain
- Check for Parsl-specific attributes (`stdout`, `stderr`)

### What Gets Preserved

- ✅ Full exception traceback (all nested calls)
- ✅ Exception message and type
- ✅ Line numbers in source files
- ✅ Local variables (in traceback context)
- ✅ Chained exceptions (`__cause__`, `__context__`)

### What Gets Lost (and How to Recover)

- ❌ Worker's stdout (unless you configure `worker_logdir_root`)
  - **Fix:** Set `worker_logdir_root` in executor config
- ❌ Worker's stderr (same)
  - **Fix:** Same as above
- ❌ Worker environment variables
  - **Fix:** Log them in worker entry point
- ❌ Import-time errors (before function runs)
  - **Fix:** Add instrumentation at function start

## Testing Your Error Extraction

### Test 1: Deliberate TypeError

```python
@python_app
def broken_app():
    return "hello" + 123  # TypeError: can only concatenate str

future = broken_app()
unwrap_parsl_future(future, "test_broken")
```

**Expected output:**
```
[test_broken] FAILED in worker:
Traceback (most recent call last):
  File "<stdin>", line 2, in broken_app
TypeError: can only concatenate str (not "int") to str
```

### Test 2: Import Error

```python
@python_app
def import_error_app():
    import nonexistent_module  # ModuleNotFoundError
    return "success"

future = import_error_app()
unwrap_parsl_future(future, "test_import_error")
```

**Expected output:**
```
[test_import_error] FAILED in worker:
Traceback (most recent call last):
  File "<stdin>", line 2, in import_error_app
ModuleNotFoundError: No module named 'nonexistent_module'
```

### Test 3: Nested Exception

```python
@python_app
def nested_error_app():
    try:
        raise ValueError("Inner error")
    except ValueError as e:
        raise RuntimeError("Outer error") from e

future = nested_error_app()
unwrap_parsl_future(future, "test_nested")
```

**Expected output:**
```
[test_nested] FAILED in worker:
Traceback (most recent call last):
  File "<stdin>", line 3, in nested_error_app
ValueError: Inner error

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "<stdin>", line 5, in nested_error_app
RuntimeError: Outer error
```

## Summary

**Key takeaways:**

1. **Parsl hides worker exceptions by default** - You only see "Dependency failure" without this pattern
2. **The unwrap pattern extracts full tracebacks** - Wrap every `future.result()` call
3. **Name your futures descriptively** - Use task metadata for the `name` parameter
4. **Configure worker logs** - Set `worker_logdir_root` for additional debugging
5. **Test with broken tasks** - Verify error extraction before deploying to production

**Pattern in three lines:**
```python
def unwrap_parsl_future(future, name):
    try: return future.result()
    except Exception as e: log.error(traceback.format_exception(...)); raise
```

**Impact:**
- ✅ Debugging time reduced from hours to minutes
- ✅ Can see actual line numbers and error messages
- ✅ No more guessing what "Dependency failure" means
- ✅ Faster iteration on workflow development

This pattern should be **standard practice** for any Parsl workflow that runs real workloads.

## References

- Parsl documentation: https://parsl.readthedocs.io/
- Python traceback module: https://docs.python.org/3/library/traceback.html
- Lambda-Hat implementation: `lambda_hat/workflows/parsl_llc.py:46-76`
- Executor routing design: `docs/executor_routing.md`
