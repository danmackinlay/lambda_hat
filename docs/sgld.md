# SGLD Configuration and Implementation

Stochastic Gradient Langevin Dynamics (SGLD) is a key sampler used in this project, primarily suited for large datasets due to its use of minibatching.

## Implementation Details

The SGLD loop is implemented in `llc/sampling.py`. It uses a custom kernel to handle minibatching efficiently within a `jax.lax.scan` loop. The sampler operates in `float32` precision by default for memory efficiency.

## Usage

SGLD is run by default. You can configure its parameters via Hydra overrides under the `sampler.sgld` group.

```bash
# Example: Running with specific SGLD settings
python train.py sampler.sgld.steps=20000 \
                sampler.sgld.batch_size=512 \
                sampler.sgld.step_size=1e-7
```

## Configuration Options

- `steps`: Total number of SGLD steps (iterations).
- `warmup`: Number of warmup steps (defined in config, but currently unused by the SGLD loop implementation).
- `batch_size`: Size of the minibatch used in each iteration.
- `step_size`: The learning rate (step size) for the SGLD update.
- `dtype`: Precision (default: `float32`).

## Preconditioning (Note on Implementation Status)

The configuration structure (`llc/config.py` and `conf/sampler/*.yaml`) includes fields for adaptive preconditioning (like Adam or RMSProp): `precond`, `beta1`, `beta2`, `eps`.

**Status:** As of the current version, preconditioned SGLD (e.g., pSGLD, Adam-SGLD) is **not implemented** in the sampling loop (`llc/sampling.py`).

The standard SGLD algorithm is used regardless of the `sampler.sgld.precond` setting. These configuration options are placeholders for future development.