# BlackJAX Usage Notes

This repository uses BlackJAX (pinned to version 1.2.5) for MCMC sampling. The implementation focuses on clean, idiomatic JAX loops using `jax.lax.scan` within `llc/sampling.py`.

## Implementation Overview

### HMC (Hamiltonian Monte Carlo)

- **Implementation:** Uses `blackjax.hmc` along with `blackjax.window_adaptation` for automatic tuning of step size and the inverse mass matrix during the warmup phase.
- **Precision:** Runs in `float64`.
- **Parallelism:** Chains are run in parallel using `jax.vmap`.

### SGLD (Stochastic Gradient Langevin Dynamics)

- **Implementation:** A custom SGLD kernel is implemented directly in `run_sgld` to handle minibatching efficiently.
- **Precision:** Runs in `float32`.
- **Parallelism:** Chains are run in parallel using `jax.vmap`.
- **Preconditioning:** Supports standard SGLD, Adam-preconditioned SGLD (pSGLD), and RMSProp-preconditioned SGLD via the `precond` configuration parameter. See [SGLD Configuration](./sgld.md).

### MCLMC (Microcanonical Langevin Monte Carlo)

- **Implementation:** Uses `blackjax.mclmc`.
- **Precision:** Runs in `float64`.
- **Parallelism:** Chains are run in parallel using `jax.vmap`.
- **Adaptation:** Supports both fixed hyperparameters and automatic adaptation via `blackjax.mclmc_find_L_and_step_size`. Adaptation is controlled by the `adaptation.num_steps` configuration parameter (set to 0 to disable).

## ⚠️ Docs Drift Warning

The online BlackJAX documentation defaults to the `main` branch, which may differ from the pinned version 1.2.5 used here. When consulting BlackJAX documentation, ensure you are viewing the [1.2.5 tag source](https://github.com/blackjax-devs/blackjax/tree/1.2.5).