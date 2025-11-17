# Logging Migration Plan - Remaining Print Statements

**Status**: Phase 1 complete (core modules), Phase 2 in progress
**Created**: 2025-11-17
**Total Remaining**: 106 print statements across 8 files

## Phase 1: Core Modules ✅ COMPLETE

Already migrated in commit `0a28f3b`:
- ✅ `lambda_hat/logging_config.py` - Created centralized logging module
- ✅ `lambda_hat/commands/sample_cmd.py` - Sample entrypoint (3 print statements)
- ✅ `lambda_hat/commands/build_cmd.py` - Build entrypoint (5 print statements)
- ✅ `lambda_hat/workflows/parsl_llc.py` - Partial migration (9 print statements converted)
- ✅ `lambda_hat/analysis.py` - Analysis diagnostics (3 print statements)
- ✅ `pyproject.toml` - Added Ruff T201 rule to ban print() in library code

## Phase 2: Remaining Modules

### Priority 1: Workflows (69 statements, ~65% of total)

#### `lambda_hat/workflows/parsl_optuna.py` - 38 print statements
**Purpose**: Optuna hyperparameter optimization workflow
**User-facing**: Yes (CLI output for optimization progress)
**Priority**: Medium

**Recommendation**:
```python
import logging
from lambda_hat.logging_config import configure_logging

log = logging.getLogger(__name__)

def main():
    configure_logging()  # At entrypoint
    log.info("=== Optuna Workflow Configuration ===")
    log.info("Problems: %d", len(problems))
    log.info("Methods: %s", methods)
    # etc.
```

**Migration effort**: 1-2 hours (many progress prints, consider using tqdm for progress bars)

**Print statement types**:
- Configuration summary (lines 31-40)
- Stage announcements (lines 43, 60)
- Progress updates (lines 45, 47, 55, 67, 69, 71, 75, 79, 83, 87)
- Results logging (lines 89, 93, 97, 101, 105, 109, 113, 117)

#### `lambda_hat/workflows/parsl_llc.py` - 31 print statements (partially done)
**Purpose**: Main LLC workflow (N targets × M samplers)
**User-facing**: Yes (primary CLI workflow)
**Priority**: HIGH

**Status**: Partially migrated (9 statements done in Phase 1)
**Remaining**: 31 statements in `run_workflow()` function

**Recommendation**:
```python
# Already has: configure_logging() at entrypoint ✅
# Need to migrate run_workflow() body:

log.info("=== Stage A: Building Targets ===")
log.info("  Submitting build for %s (model=%s, data=%s)", target_id, model, data)
log.info("=== Stage B: Running Samplers ===")
log.info("  Submitting %s for %s (run_id=%s)", sampler, target_id, run_id)
log.info("=== Waiting for %d sampling runs to complete ===", len(run_futures))
log.info("  [%d/%d] ✓ %s", i, total, run_key)
log.error("  [%d/%d] ✗ FAILED: %s", i, total, run_key)
log.info("Wrote %d rows to %s", len(df), output_path)
```

**Migration effort**: 1 hour (straightforward INFO/ERROR replacements)

**Print statement types**:
- Stage announcements (lines 193, 225, 268, 353)
- Task submissions (lines 195-198, 227-234)
- Progress updates (lines 270-297)
- Results summary (lines 355-387)

### Priority 2: Runners (22 statements, ~21% of total)

#### `lambda_hat/runners/run_method.py` - 13 print statements
**Purpose**: Optuna trial runner (method evaluation)
**User-facing**: Indirect (called by parsl_optuna)
**Priority**: Medium

**Recommendation**:
```python
import logging
log = logging.getLogger(__name__)

# Called from Parsl worker - logging already configured by parent
log.info("=== Optuna Trial: %s on %s ===", method_name, problem_id)
log.info("Trial %d: hyperparams=%s", trial_number, params)
log.info("Result: LLC=%.4f (target=%.4f)", llc_final, llc_ref)
log.error("Trial %d FAILED: %s", trial_number, error)
```

**Migration effort**: 30 minutes

**Print statement types**:
- Trial headers (lines 45-50)
- Hyperparameter logging (lines 55-70)
- Progress/results (lines 85-110)
- Error reporting (lines 115-120)

#### `lambda_hat/runners/hmc_reference.py` - 9 print statements
**Purpose**: HMC reference computation for Optuna
**User-facing**: Indirect (called by parsl_optuna)
**Priority**: Medium

**Recommendation**:
```python
import logging
log = logging.getLogger(__name__)

log.info("=== HMC Reference Computation ===")
log.info("Problem: %s", problem_id)
log.info("Target: %s", target_id)
log.info("HMC config: draws=%d, warmup=%d", draws, warmup)
log.info("Reference LLC: %.6f ± %.6f", llc_mean, llc_std)
```

**Migration effort**: 20 minutes

**Print statement types**:
- Configuration display (lines 30-45)
- Progress updates (lines 50-65)
- Results summary (lines 70-85)

### Priority 3: Commands & Utilities (15 statements, ~14% of total)

#### `lambda_hat/commands/artifacts_cmd.py` - 7 print statements
**Purpose**: Artifact management CLI (gc, ls)
**User-facing**: Yes (CLI output)
**Priority**: LOW

**Recommendation**:
```python
import logging
from lambda_hat.logging_config import configure_logging

log = logging.getLogger(__name__)

def gc_entry(ttl_days: int):
    configure_logging()  # At entrypoint
    log.info("GC removed %d unreachable objects (older than %dd)", removed, ttl_days)

def ls_entry():
    configure_logging()  # At entrypoint
    log.info("[%s]", exp_name)
    log.info("  %s", run_id)
```

**Migration effort**: 15 minutes

**Print statement types**:
- GC summary (line 91)
- List output (lines 102, 106, 109, 114, 119, 123)

#### `lambda_hat/commands/promote_cmd.py` - 3 print statements
**Purpose**: Promotion command entrypoint
**User-facing**: Yes (CLI output)
**Priority**: LOW

**Recommendation**:
```python
import logging
from lambda_hat.logging_config import configure_logging

log = logging.getLogger(__name__)

def promote_entry(config_yaml: str, ...):
    configure_logging()  # At entrypoint
    log.info("Promoting results from %s", runs_root)
    log.info("Samplers: %s", samplers)
```

**Migration effort**: 10 minutes

#### `lambda_hat/promote/core.py` - 3 print statements
**Purpose**: Promotion core logic
**User-facing**: Indirect (called by promote_cmd)
**Priority**: LOW

**Recommendation**:
```python
import logging
log = logging.getLogger(__name__)

log.info("Promoted %s -> %s", src, dst)
log.info("[gallery] %s: %s -> %s", sampler, src, dst)
log.info("[gallery] Wrote README snippet -> %s", md_snippet_out)
```

**Migration effort**: 10 minutes

#### `lambda_hat/training.py` - 2 print statements
**Purpose**: Neural network training loop
**User-facing**: Indirect (called by build_cmd)
**Priority**: LOW

**Recommendation**:
```python
import logging
log = logging.getLogger(__name__)

# Training progress (consider using jax.debug.print for JIT compatibility)
log.info("Step %d/%d, Loss: %.6f", step, total_steps, loss)
```

**Migration effort**: 5 minutes

**Note**: If these prints are inside JIT-compiled functions, use `jax.debug.print()` instead.

## Migration Sequence (Recommended Order)

### Sprint 1: High-Impact Workflows (3-4 hours)
1. ✅ **parsl_llc.py** - Finish remaining 31 statements (already has logging configured)
2. **parsl_optuna.py** - Convert 38 statements (add configure_logging at entrypoint)

### Sprint 2: Runner Infrastructure (1 hour)
3. **run_method.py** - Convert 13 statements (inherits logging from parent)
4. **hmc_reference.py** - Convert 9 statements (inherits logging from parent)

### Sprint 3: Commands & Utilities (40 minutes)
5. **artifacts_cmd.py** - Convert 7 statements (add configure_logging at entrypoints)
6. **promote_cmd.py** - Convert 3 statements (add configure_logging at entrypoint)
7. **promote/core.py** - Convert 3 statements (inherits from promote_cmd)
8. **training.py** - Convert 2 statements (inherits from build_cmd, watch for JIT)

## Testing Strategy

After each sprint:

```bash
# Check no new print() violations in migrated files
uv run ruff check lambda_hat/workflows/parsl_llc.py --select T201

# Run smoke tests to ensure logging works
JAX_ENABLE_X64=1 uv run pytest tests/test_smoke_workflow.py -v

# Verify log output quality
LOG_LEVEL=INFO uv run lambda-hat workflow llc --local
LOG_LEVEL=DEBUG uv run lambda-hat workflow llc --local
```

## Special Considerations

### 1. JAX JIT Functions (training.py)
If print statements are inside `@jax.jit` functions, use `jax.debug.print()`:

```python
@jax.jit
def train_step(params, batch):
    # WRONG: log.info() won't work inside JIT
    # RIGHT:
    jax.debug.print("step: loss={loss}", loss=loss_value)
    return params
```

### 2. Parsl Workers (runners/*)
Worker functions inherit logging config from parent process:

```python
# Parent (parsl_optuna.py main)
configure_logging()

# Worker (run_method.py)
log = logging.getLogger(__name__)  # Just use it, already configured
log.info("Trial starting...")
```

### 3. Progress Bars vs Logging
For long-running loops with many progress updates, consider `tqdm` instead of logging:

```python
from tqdm import tqdm

for trial in tqdm(trials, desc="Optuna trials"):
    # Only log important events
    log.info("Trial %d: best_value=%.4f", trial.number, trial.value)
```

### 4. User-Facing Output
For commands like `artifacts ls` that are meant for human consumption, logging INFO level is appropriate. For machine-readable output, consider adding a `--json` flag.

## Verification Checklist

After migration is complete:

- [ ] All 106 print statements converted to logging
- [ ] `uv run ruff check lambda_hat/ --select T201` shows 0 violations
- [ ] All entrypoints call `configure_logging()`
- [ ] Worker functions use module-level loggers without reconfiguring
- [ ] JIT functions use `jax.debug.print()` where needed
- [ ] Smoke tests pass with `LOG_LEVEL=INFO`
- [ ] Debug output works with `LOG_LEVEL=DEBUG`
- [ ] No performance regression from logging overhead

## Future Enhancements (Out of Scope)

Once basic migration is complete, consider:

1. **Structured logging** - Add context filters for run_id, sampler, target_id
2. **JSON logging** - Set `LOG_FORMAT=json` for machine parsing
3. **Parsl queue handlers** - Use `start_mp_logging_queue()` for worker logging
4. **Progress tracking** - Integrate tqdm or custom progress logger
5. **Log aggregation** - Ship logs to centralized store (for SLURM runs)

## References

- Original plan: `plans/logging.md`
- Python logging HOWTO: https://docs.python.org/3/howto/logging.html
- JAX debugging: https://docs.jax.dev/en/latest/debugging.html
- Ruff T201 rule: https://docs.astral.sh/ruff/rules/print/

---

**Last Updated**: 2025-11-17
**Total Progress**: 6/14 modules complete (43%), 106 print statements remaining
