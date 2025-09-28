# SGLD Configuration and Implementation

Stochastic Gradient Langevin Dynamics (SGLD) is a key sampler used in this project, primarily suited for large datasets due to its use of minibatching.

## Implementation Details

The SGLD loop is implemented in `lambda_hat/sampling.py`. It uses a custom kernel to handle minibatching efficiently within a `jax.lax.scan` loop. The sampler operates in `float32` precision by default for memory efficiency.

## Usage

SGLD is configured via `config/experiments.yaml` using the sampler configuration system.

```yaml
# Example: Running with specific SGLD settings in config/experiments.yaml
samplers:
  - name: sgld
    overrides:
      steps: 20000
      batch_size: 512
      step_size: 1e-7
```

Then execute:
```bash
uv run snakemake -j 4
```

## Configuration Options

- `steps`: Total number of SGLD steps (iterations).
- `warmup`: Number of warmup steps (defined in config, but currently unused by the SGLD loop implementation).
- `batch_size`: Size of the minibatch used in each iteration.
- `step_size`: The learning rate (step size) for the SGLD update.
- `dtype`: Precision (default: `float32`).

## Preconditioning (Implementation Status)

The configuration structure (`lambda_hat/config.py` and `conf/sampler/*.yaml`) includes fields for adaptive preconditioning (like Adam or RMSProp): `precond`, `beta1`, `beta2`, `eps`.

**Status:** Preconditioned SGLD variants (pSGLD) are **fully implemented** in the sampling loop (`lambda_hat/sampling.py`).

The implementation includes:
- **Vanilla SGLD** (`precond="none"`): Standard SGLD with fixed step size
- **RMSPropSGLD** (`precond="rmsprop"`): Adaptive step size using second moment estimates
- **AdamSGLD** (`precond="adam"`): Adaptive step size and drift using first and second moment estimates

### Implementation Compliance

Following Hitchcock and Hoogland (Appendix D.3), the implementation correctly separates:
1. **Adaptation statistics**: Updated using only the loss gradient (excluding localization prior)
2. **Final drift**: Combines both loss gradient and localization term for the parameter update

This ensures the preconditioning adapts to the data likelihood while maintaining proper localization around the ERM solution.