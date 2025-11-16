# Samplers

Lambda-Hat supports multiple sampling algorithms for Local Learning Coefficient estimation: HMC, MCLMC, SGLD, and VI.

---

## Supported Samplers

### HMC (Hamiltonian Monte Carlo)

**What it is**: Full-batch MCMC sampler using Hamiltonian dynamics.

**Configuration**: `sampler: hmc` in your YAML.

**Precision**: `float64`

**Key features**:
- Automatic warmup with `blackjax.window_adaptation`
- Tunes step size and mass matrix during adaptation
- Parallel chain execution via `jax.vmap`

**Run it**:
```bash
lambda-hat sample --config-yaml config/experiments.yaml --target-id tgt_abc123
```

See [Configuration Reference](./config.md) for all HMC options under `sample/sampler/hmc.yaml`.

---

### MCLMC (Microcanonical Langevin Monte Carlo)

**What it is**: Full-batch MCMC using microcanonical Langevin dynamics.

**Configuration**: `sampler: mclmc` in your YAML.

**Precision**: `float64`

**Key features**:
- Adaptation via `blackjax.mclmc_find_L_and_step_size`
- Control adaptation with `adaptation.num_steps` (set to 0 to disable)
- Parallel chain execution via `jax.vmap`

**Run it**:
```bash
lambda-hat sample --config-yaml config/experiments.yaml --target-id tgt_abc123
```

See [Configuration Reference](./config.md) for all MCLMC options under `sample/sampler/mclmc.yaml`.

---

### SGLD (Stochastic Gradient Langevin Dynamics)

**What it is**: Stochastic gradient MCMC with minibatching.

**Configuration**: `sampler: sgld` in your YAML.

**Precision**: `float32`

**Key features**:
- Efficient minibatch gradients
- Preconditioning options: vanilla (`none`), RMSProp (`rmsprop`), Adam (`adam`)
- Parallel chain execution via `jax.vmap`

**Preconditioning**:

| Mode | Description |
|------|-------------|
| `none` | Standard SGLD with fixed step size |
| `rmsprop` | Adaptive step size using second moment estimates |
| `adam` | Adaptive step size and drift using first and second moment estimates |

**Configuration example**:
```yaml
samplers:
  - name: sgld
    overrides:
      steps: 20000
      batch_size: 512
      step_size: 1e-7
      precond: "adam"
```

**Run it**:
```bash
lambda-hat workflow llc --local
```

**Implementation notes**: The SGLD loop correctly separates adaptation statistics (using only loss gradient) from the final drift (which includes localization), following Hitchcock and Hoogland (Appendix D.3).

See [Configuration Reference](./config.md) for all SGLD options under `sample/sampler/sgld.yaml`.

---

### VI (Variational Inference)

**What it is**: Fast variational approximation using mixture of factor analyzers or normalizing flows.

**Configuration**: `sampler: vi` in your YAML.

**Precision**: `float32` (configurable)

**Algorithms**:
- [MFA](./vi_mfa.md) — Mixture of factor analyzers (default)
- [Flow](./vi_flow.md) — Normalizing flows (requires `--extra flowvi`)

**Key features**:
- Much faster than MCMC for initial exploration
- STL gradients and HVP control variates for variance reduction
- TensorBoard integration for real-time diagnostics

**Run it**:
```bash
lambda-hat workflow llc --local
```

See [VI overview](./vi.md) for general concepts and shared configuration.

---

## Choosing a Sampler

| Sampler | Speed | Accuracy | Use case |
|---------|-------|----------|----------|
| **VI** | Fast | Good | Initial exploration, hyperparameter sweeps |
| **SGLD** | Medium | Good | Large datasets, efficiency-focused |
| **HMC** | Slow | High | Gold standard, small-medium problems |
| **MCLMC** | Slow | High | Alternative to HMC, sometimes more efficient |

---

## Configuration

All sampler options are defined in `lambda_hat/conf/sample/sampler/*.yaml` and can be overridden in `config/experiments.yaml`:

```yaml
samplers:
  - name: hmc
    overrides:
      draws: 5000
      warmup: 1000
  - name: sgld
    overrides:
      steps: 20000
      batch_size: 512
      precond: "adam"
```

See:
- [Configuration Reference](./config.md) — complete schema with defaults
- [CLI Reference](./cli.md) — command-line options
- [Workflows](./workflows.md) — running sweeps with multiple samplers
