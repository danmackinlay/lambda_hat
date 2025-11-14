# VI Completion Progress

Tracking implementation of the comprehensive VI improvement plan from `finish_vi.md`. This plan is being executed in careful stages to maintain code quality and testability.

---

## ✅ Stage 0: Completed (Before this plan)
- ✅ Mixture of factor analyzers with STL + RB gradients
- ✅ HVP control variate for LLC estimation
- ✅ Float32 numerical stability (clipping, ridge, normalization)
- ✅ Test suite validates convergence
- ✅ Snakemake integration
- ✅ ArviZ-compatible trace outputs

---

## ✅ Stage 1: Whitening & Core Stability (COMPLETED)

**Goal:** Close the whitening TODO and add essential stability improvements

**Status:** COMPLETED

### Tasks
- ✅ **Config Extensions** (`lambda_hat/config.py`)
  - Add `whitening_mode: str = "none"` (options: "none"|"rmsprop"|"adam")
  - Add `whitening_decay: float = 0.99` (EMA decay for gradient moments)
  - Add `clip_global_norm: Optional[float] = 5.0` (gradient clipping)
  - Add `alpha_temperature: float = 1.0` (softmax temperature on mixture weights)

- ✅ **Config YAML** (`lambda_hat/conf/sample/sampler/vi.yaml`)
  - Add defaults for new fields

- ✅ **Whitening Pre-Pass** (`lambda_hat/sampling.py::run_vi`)
  - Compute diagonal preconditioner A_diag from minibatch gradients
  - Build EMA of squared gradients over ~500-1000 samples
  - Call `vi.make_whitener(A_diag)` instead of `vi.make_whitener(None)`

- ✅ **Stability Guards** (`lambda_hat/variational.py`)
  - Add gradient clipping via optax.clip_by_global_norm
  - Bound α_temperature >= 0.5 (implemented in softmax_with_temperature helper)
  - Add softmax-with-temperature for mixture logits
  - Enhance Cholesky failure recovery (increased default ridge from 1e-6 to 1e-5)

- ✅ **Tests** (`tests/test_vi_whitening.py`)
  - test_vi_whitening_rmsprop_stability
  - test_vi_whitening_adam_stability
  - test_vi_gradient_clipping
  - test_softmax_with_temperature
  - test_whitener_identity
  - test_whitener_diagonal
  - test_whitener_numerical_stability

- ✅ **Documentation** (`docs/vi.md`)
  - Replace "future support for whitening" with actual usage guide
  - Add when to use whitening
  - Add how to choose mode (rmsprop vs adam)
  - Add how to tune whitening_decay
  - Add Stage 1 stability enhancements section
  - Update implementation compliance section

**Actual Effort:** ~3-4 hours
**Lines Changed:** ~350 across 8 files (including tests)
**Risk:** LOW - All tests pass, imports successful

---

## 🔄 Stage 2: TensorBoard & Observability (CURRENT)

**Goal:** Make VI tunable and debuggable with rich diagnostics

**Status:** IN PROGRESS

### Planned Features
- **TensorBoard Scalars**
  - `vi/elbo`, `vi/elbo_like`, `vi/logq`
  - `vi/radius2` (mean and quantiles), `vi/resp_entropy`
  - `vi/pi_min`, `vi/pi_max`, `vi/pi_entropy`
  - `vi/D_sqrt_min`, `vi/D_sqrt_max`, `vi/D_sqrt_med`
  - `vi/A_col_norm_p95`, `vi/grad_norm`
  - `vi/cumulative_fge`, `vi/lr`
  - Final: `vi/Eq_Ln_mc`, `vi/Eq_Ln_cv`, `vi/variance_reduction`, `vi/L0`, `llc`

- **TensorBoard Images** (low frequency)
  - Mixture weights bar chart (π), sorted
  - Per-component "scree" plots (reveals rank usage)
  - Histogram of D^{1/2} (log scale)
  - Radius trace overlay

- **ArviZ Integration**
  - Export VI-specific traces to `sample_stats` (resp_entropy, elbo_like, logq)
  - Enables unified convergence diagnostics with MCMC samplers

- **Documentation**
  - TensorBoard quickstart guide
  - How to read diagnostics
  - Debugging common issues

**Files to Modify:**
- `lambda_hat/sampling.py::run_vi` (TB post-hoc writing)
- `lambda_hat/analysis.py` (sample_stats export)
- `docs/vi.md` (TB guide)

**Estimated Effort:** 1-2 days
**Lines Changed:** ~400
**Risk:** LOW - additive features, no behavior changes

---

## 🔲 Stage 3: Advanced Configuration (FUTURE)

**Goal:** Expose fine-grained control for power users

**Status:** NOT STARTED

### Planned Features

#### 3A: Per-Component Ranks
- `r_per_component: Optional[List[int]]` - heterogeneous rank budgets
- Mask-based implementation (keep arrays static for JIT)
- Test: rank mask correctness

#### 3B: Mixture Management
- `mixture_cap: Optional[int]` - upper bound for M
- `prune_threshold: float = 1e-3` - drop weak components
- `alpha_dirichlet_prior: Optional[float]` - discourage collapse
- Component pruning after eval_every steps
- Test: pruning monotonicity, anti-collapse

#### 3C: LR Schedules & Optimization
- `lr_schedule: Optional[str]` - "cosine"|"linear_decay"
- `lr_warmup_frac: float = 0.05`
- Optax schedule factory
- Test: schedule application

#### 3D: Entropy & Exploration
- `entropy_bonus: float = 0.0` - add λ * H(q) to ELBO
- Enhanced α_temperature controls
- Test: exploration behavior

**Files to Modify:**
- `lambda_hat/config.py` (8+ new fields)
- `lambda_hat/variational.py` (masks, pruning, prior gradient, schedules)
- `lambda_hat/conf/sample/sampler/vi.yaml`
- `tests/test_vi_advanced.py` (new file)

**Estimated Effort:** 2-3 days
**Lines Changed:** ~600
**Risk:** MEDIUM - more complex, potential for bugs

**Decision Point:** Implement selectively based on user feedback after Stage 2

---

## 🔲 Stage 4: HVP Whitening & Polish (FUTURE)

**Goal:** Research-grade features and final refinements

**Status:** NOT STARTED

### Planned Features

#### 4A: HVP Diagonal Whitening
- `whitening_mode: "hvp_diag"` option
- Hutchinson-style approximation: `diag(H) ≈ mean( (H v) ⊙ v )`
- Reuse existing HVP machinery from control variate
- Test: HVP whitening improves on anisotropic problems

#### 4B: Full Test Suite
- Comprehensive integration tests
- Benchmark suite (compare to HMC/SGLD)
- Performance regression tests

#### 4C: User Tuning Playbook
- Practical guide in docs/vi.md
- How to tune γ (localizer strength)
- How to tune M & r (mixture size & rank)
- When to use whitening modes
- Optimizer & schedule selection
- Preventing component collapse
- Improving control variate quality

#### 4D: Acceptance Criteria Validation
- Config backward compatibility verified
- Whitening reduces ELBO/radius spikes (quantified)
- TensorBoard events present
- No NaNs on test suite
- WNV improvement on benchmark targets

**Files to Modify:**
- `lambda_hat/variational.py` (HVP whitening)
- `tests/test_vi_hvp.py` (new file)
- `tests/test_vi_benchmarks.py` (new file)
- `docs/vi.md` (tuning playbook)

**Estimated Effort:** 1-2 days
**Lines Changed:** ~300
**Risk:** LOW - refinement and documentation

---

## Summary Statistics

### Overall Plan
- **Total Stages:** 4 (Stage 0 already complete)
- **Total Estimated Effort:** 4-9 days
- **Total Lines Changed:** ~1600
- **Files Modified:** ~12
- **New Test Files:** ~3
- **Risk:** LOW overall due to staged approach

### Current Status (after Stage 0)
- ✅ Lines: ~3500 (existing VI implementation)
- ✅ Tests: Basic convergence validated
- 🔲 TODO: Whitening (documented but not implemented)
- 🔲 TODO: TensorBoard diagnostics
- 🔲 TODO: Advanced configuration options

---

## Notes for Future Implementers

1. **Each stage is self-contained**: Can be implemented, tested, and merged independently
2. **Backward compatibility is maintained**: Old configs continue to work
3. **Tests guard each feature**: No feature without a test
4. **Documentation is updated incrementally**: Each stage updates relevant docs
5. **Progress is tracked**: This file is updated after each stage completes

## Change Log

- 2025-11-14: Created this progress tracking document (Stage 1 start)
- 2025-11-14: Completed Stage 1 (Whitening & Core Stability) - all tasks ✓
- 2025-11-14: Starting Stage 2 (TensorBoard & Observability)
