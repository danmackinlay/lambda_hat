# VI Critical Issue: No Samples Written to Trace

**Date**: 2025-11-13
**Status**: BLOCKING - Production VI broken
**Severity**: Critical

## Executive Summary

VI sampler runs without errors but produces **empty trace files** (no samples), resulting in degenerate LLC estimates (all zeros). This affects both production and all tests.

---

## Evidence

### 1. Empty Trace File
```bash
$ uv run python -c "import xarray as xr; ds = xr.open_dataset('runs/targets/tgt_ec9030f732ba/run_vi_65008785/trace.nc'); print(ds)"

KeyError: "No variable named 'lambda'. Variables on the dataset include []"
```

### 2. Degenerate Analysis
```json
{
  "llc_mean": 0.0,
  "llc_std": 0.0,
  "llc_min": 0.0,
  "llc_max": 0.0,
  "r_hat": NaN,
  "wnv": 0.0
}
```

### 3. VI Completes Successfully
- Runtime: 8.4s (normal)
- No Python errors/exceptions
- Log shows "wrote trace.nc & analysis.json"
- But trace.nc is empty!

---

## Investigation Timeline

### Initial Problem: QA Numerical Improvements Broke VI

Commits afb3122 and 3ed6012 introduced aggressive numerical hardening:
- Ridge increased from ~1e-6 to 1e-4 (100x)
- max_norm reduced from 10.0 to 3.0
- Added lax.Precision.HIGHEST

**Result**: VI over-regularized, collapsed variational family

### Revert Attempt

Reverted afb3122 and 3ed6012 to restore original numerics.

**Result**: VI still broken! LLC still 0.0, trace.nc still empty.

### Root Cause Discovery

VI was **already broken before QA fixes**. The breakage likely occurred in:

- **355e57c**: "fix(vi): make VI PyTree-aware and fix dtype consistency in float32 mode"
- **be0ba39**: "fix(vi): unpack minibatch tuple when calling loss function"

These commits modified core VI sampling/trace writing logic.

---

## Current Symptoms

1. **Production VI**: Runs fast (8.5s) but produces empty traces
2. **Quadratic tests**: All 3 fail with NaN/finite assertions
3. **MLP tests**:
   - 2/3 fail with NaN
   - 1/3 produces finite values but variance reduction slightly over threshold (2.269 vs 2.0)

---

## Hypothesis

The PyTree-aware refactoring (355e57c) likely broke the trace writing mechanism. Possible causes:

1. **Sample format mismatch**: VI generates samples in PyTree format but trace writer expects flat arrays
2. **Missing flatten step**: Samples never get flattened before writing to NetCDF
3. **Silent failure**: NetCDF write fails silently, produces empty file

---

## Action Items

### Immediate (Blocking)
1. **Bisect commits** to find exact breakage point (355e57c vs be0ba39)
2. **Debug trace writing** in `lambda_hat/sampling.py` for VI sampler
3. **Verify sample generation**: Add debug logging to confirm VI produces samples

### Short-term
1. **Fix sample writing** to handle PyTree format correctly
2. **Re-test** production VI and all test suites
3. **Reconsider QA numerics** once basic VI is working again

### Long-term
1. **Add trace validation**: Catch empty traces early with assertions
2. **Improve error handling**: Make NetCDF write failures visible
3. **Test coverage**: Add smoke test that verifies trace contains data

---

## Files Affected

- `lambda_hat/variational.py` - VI implementation (PyTree refactoring)
- `lambda_hat/sampling.py` - Trace writing (likely broken)
- `tests/test_vi_*.py` - All VI tests failing

---

## Commits

- **ffe980b**: Revert QA numerical improvements (current HEAD)
- **d1fb329**: Add MLP-based VI tests
- **afb3122**: QA numerical improvements (reverted, but not the root cause)
- **3ed6012**: QA feedback implementation (reverted)
- **355e57c**: PyTree-aware refactoring ⚠️ **SUSPECT**
- **be0ba39**: Minibatch unpacking fix ⚠️ **SUSPECT**

---

## Next Steps

**Priority**: Find and fix trace writing breakage before any other VI work.

The numerical improvements can be revisited once basic VI functionality is restored.
