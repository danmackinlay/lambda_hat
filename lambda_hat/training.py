# llc/training.py
"""Training utilities for finding ERM solutions"""

from __future__ import annotations

from typing import TYPE_CHECKING, Tuple, Dict, Any
import jax
import jax.numpy as jnp
import optax
from jax import jit, value_and_grad

if TYPE_CHECKING:
    from .config import Config


def train_erm(
    loss_fn,
    params_init,
    cfg: Config,
    key: jax.random.PRNGKey
) -> Tuple[Dict[str, Any], Dict[str, float]]:
    """Train to find the empirical risk minimizer w*

    Args:
        loss_fn: Loss function that takes params and returns scalar loss
        params_init: Initial Haiku parameters
        cfg: Configuration object
        key: JRNG key for any randomness

    Returns:
        Tuple of (optimized_params, metrics_dict)
    """
    t_cfg = cfg.training

    # Setup optimizer
    if t_cfg.optimizer == "adam":
        opt = optax.adam(learning_rate=t_cfg.learning_rate)
    elif t_cfg.optimizer == "sgd":
        opt = optax.sgd(learning_rate=t_cfg.learning_rate)
    elif t_cfg.optimizer == "adamw":
        opt = optax.adamw(learning_rate=t_cfg.learning_rate)
    else:
        raise ValueError(f"Unknown optimizer: {t_cfg.optimizer}")

    opt_state = opt.init(params_init)
    params = params_init

    # Create update function
    @jit
    def update(params, opt_state):
        loss_val, grads = value_and_grad(loss_fn)(params)
        updates, opt_state = opt.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss_val

    # Training loop
    losses = []
    for step in range(t_cfg.erm_steps):
        params, opt_state, loss_val = update(params, opt_state)
        losses.append(float(loss_val))

        # Early stopping check
        if step > 100 and t_cfg.early_stop_tol is not None:
            recent_losses = losses[-20:]
            if len(recent_losses) == 20:
                std = jnp.std(jnp.array(recent_losses))
                if std < t_cfg.early_stop_tol:
                    print(f"Early stopping at step {step}, std={std:.2e}")
                    break

        # Logging
        if step % max(1, t_cfg.erm_steps // 10) == 0:
            print(f"Step {step}/{t_cfg.erm_steps}, Loss: {loss_val:.6f}")

    # Compute final metrics
    final_loss = float(loss_fn(params))
    metrics = {
        "final_loss": final_loss,
        "erm_steps": step + 1,
        "initial_loss": losses[0] if losses else 0.0,
    }

    return params, metrics