# lambda_hat/vi/flow.py
"""Normalizing flow VI algorithm using FlowJAX.

This module implements variational inference using normalizing flows via the
manifold-plus-noise construction:
    q(w) = δ(w - w*) ⊗ N(0, D^{-1}) ⊗ Flow_z ⊗ N(0, σ²I)
    w = w* + D^{-1/2} @ (U @ z + E_perp @ eps)

where:
- U is a d × d_latent orthonormal basis (random QR initialization)
- E_perp is the orthonormal complement (d × (d - d_latent))
- D^{-1/2} is the whitening preconditioner from RMSProp/Adam
- z ~ Flow(z) is a low-dimensional latent flow (RealNVP by default)
- eps ~ N(0, σ²I) is small orthogonal noise

The log-density uses a block-triangular Jacobian for efficient computation.

Design choices (per plan):
- Random orthonormal U via QR (not learned, not data-adaptive SVD)
- RealNVP coupling flow (fast sampling, good expressiveness)
- K=1 ELBO sample per gradient step (standard for scalability)
- NO HVP control variate initially (deferred to future work)

Requires: equinox, flowjax (install via `uv sync --extra flowvi`)
"""

import time
from typing import Any, Callable, Dict, Optional, Tuple

import jax
import jax.numpy as jnp
import optax

# Defer equinox/flowjax imports until actually used
# This allows the module to be imported without requiring flowvi dependencies
_FLOWJAX_AVAILABLE = False
_IMPORT_ERROR = None

try:
    import equinox as eqx
    import flowjax.bijections as fjx_bij
    import flowjax.distributions as fjx_dist
    import flowjax.flows as fjx_flows

    # FlowJAX 17.2.1 requires new-style typed JAX PRNG keys
    jax.config.update("jax_default_prng_impl", "rbg")

    _FLOWJAX_AVAILABLE = True
except ImportError as e:
    _IMPORT_ERROR = e
    # Create placeholder objects for type checking
    eqx = None  # type: ignore
    fjx_bij = None  # type: ignore
    fjx_dist = None  # type: ignore
    fjx_flows = None  # type: ignore

# === InjectiveLift: Manifold-plus-noise variational family ===


class InjectiveLift(eqx.Module):
    """Manifold-plus-noise variational distribution for flow VI.

    Maps low-dimensional latent flow to parameter space via:
        w = w* + D^{-1/2} @ (U @ z + E_perp @ eps)
    where z ~ flow(z), eps ~ N(0, σ²I)

    The log-density uses a block-triangular Jacobian:
        log q(w) = log q_flow(z) + log|det J|
    where log|det J| = sum(log D^{-1/2}) + d_perp * log(σ)

    Attributes:
        flow: FlowJAX flow in R^{d_latent}
        U: (d, d_latent) orthonormal basis (spans latent manifold)
        E_perp: (d, d - d_latent) orthonormal complement (orthogonal directions)
        D_inv_sqrt: (d,) whitening preconditioner inverse sqrt
        sigma_perp: scalar, orthogonal noise scale
        wstar: (d,) center (ERM solution)
    """

    flow: eqx.Module  # FlowJAX AbstractDistribution
    U: jax.Array  # (d, d_latent)
    E_perp: jax.Array  # (d, d_perp)
    D_inv_sqrt: jax.Array  # (d,)
    sigma_perp: float
    wstar: jax.Array  # (d,)

    def sample_and_log_prob(self, key: jax.random.PRNGKey) -> Tuple[jax.Array, jax.Array]:
        """Sample w and compute log q(w).

        Args:
            key: JAX random key

        Returns:
            (w, log_q): sampled parameter (d,) and log-density scalar
        """
        k1, k2 = jax.random.split(key)

        # Sample from latent flow: z ~ q_flow(z)
        z = self.flow.sample(k1, sample_shape=())  # (d_latent,)
        logq_z = self.flow.log_prob(z)  # scalar

        # Orthogonal noise: eps ~ N(0, sigma_perp^2 I)
        d_perp = self.E_perp.shape[1]
        if d_perp > 0:
            eps = jax.random.normal(k2, (d_perp,))  # (d_perp,)
            perp_component = self.E_perp @ (self.sigma_perp * eps)
        else:
            perp_component = 0.0

        # Lift to parameter space: w = w* + D^{-1/2}(U@z + E_perp@eps)
        w = self.wstar + self.D_inv_sqrt * (self.U @ z + perp_component)

        # Log-determinant of block-triangular Jacobian
        # |J| = prod(D^{-1/2}) * sigma_perp^{d_perp}
        logdet = jnp.sum(jnp.log(self.D_inv_sqrt))
        if d_perp > 0:
            logdet = logdet + d_perp * jnp.log(self.sigma_perp)

        logq_w = logq_z + logdet
        return w, logq_w


def init_injective_lift(
    key: jax.random.PRNGKey,
    wstar_flat: jnp.ndarray,
    d_latent: int,
    sigma_perp: float,
    D_inv_sqrt: jnp.ndarray,
    flow_type: str,
    flow_depth: int,
    flow_hidden: list[int],
) -> InjectiveLift:
    """Initialize flow variational family.

    Args:
        key: JAX random key
        wstar_flat: (d,) flattened ERM solution
        d_latent: latent dimension for flow
        sigma_perp: orthogonal noise scale
        D_inv_sqrt: (d,) whitening preconditioner inverse sqrt (from whitener)
        flow_type: "realnvp" | "maf" | "nsf_ar"
        flow_depth: number of coupling/autoregressive layers
        flow_hidden: hidden layer sizes for flow network

    Returns:
        InjectiveLift instance with random initialization
    """
    d = wstar_flat.size
    dtype = wstar_flat.dtype

    k_basis, k_flow = jax.random.split(key)

    # Initialize random orthonormal basis U via QR decomposition
    Q, _ = jnp.linalg.qr(jax.random.normal(k_basis, (d, d_latent), dtype=dtype))
    U = Q  # (d, d_latent)

    # Compute orthonormal complement E_perp
    d_perp = d - d_latent
    if d_perp > 0:
        # Generate full orthogonal matrix, take complement columns
        k_perp = jax.random.fold_in(k_basis, 1)
        Q_full, _ = jnp.linalg.qr(jax.random.normal(k_perp, (d, d), dtype=dtype))
        E_perp = Q_full[:, d_latent:]  # Last d_perp columns
    else:
        E_perp = jnp.zeros((d, 0), dtype=dtype)

    # Build latent flow using FlowJAX
    base_dist = fjx_dist.Normal(jnp.zeros(d_latent, dtype=dtype))

    if flow_type == "realnvp":
        # RealNVP coupling flow (default, fast sampling)
        flow = fjx_flows.coupling_flow(
            key=k_flow,
            base_dist=base_dist,
            transformer=None,  # Default to affine coupling
            flow_layers=flow_depth,
            nn_width=flow_hidden[0] if flow_hidden else 64,
            nn_depth=len(flow_hidden) if flow_hidden else 2,
        )
    elif flow_type == "maf":
        # Masked Autoregressive Flow
        flow = fjx_flows.masked_autoregressive_flow(
            key=k_flow,
            base_dist=base_dist,
            transformer=None,  # Default to affine
            flow_layers=flow_depth,
            nn_width=flow_hidden[0] if flow_hidden else 64,
            nn_depth=len(flow_hidden) if flow_hidden else 2,
        )
    elif flow_type == "nsf_ar":
        # Neural Spline Flow (autoregressive)
        # Create rational quadratic spline transformer
        transformer = fjx_bij.RationalQuadraticSpline(knots=8, interval=3)
        flow = fjx_flows.masked_autoregressive_flow(
            key=k_flow,
            base_dist=base_dist,
            transformer=transformer,
            flow_layers=flow_depth,
            nn_width=flow_hidden[0] if flow_hidden else 64,
            nn_depth=len(flow_hidden) if flow_hidden else 2,
        )
    else:
        raise ValueError(f"Unknown flow_type '{flow_type}'. Supported: 'realnvp', 'maf', 'nsf_ar'")

    return InjectiveLift(
        flow=flow,
        U=U,
        E_perp=E_perp,
        D_inv_sqrt=D_inv_sqrt,
        sigma_perp=sigma_perp,
        wstar=wstar_flat,
    )


# === ELBO optimization ===


def build_flow_elbo_step(
    loss_batch_fn: Callable,
    data: Tuple[jnp.ndarray, jnp.ndarray],
    wstar_flat: jnp.ndarray,
    unravel_fn: Callable[[jnp.ndarray], Any],
    n_data: int,
    beta: float,
    gamma: float,
    batch_size: int,
    optimizer: optax.GradientTransformation,
    clip_global_norm: Optional[float],
    flow_static: Any,
    lift_consts: dict,
) -> Callable:
    """Build one ELBO optimization step for flow VI.

    ELBO = E_q[log p(w) - log q(w)]
    where log p(w) = -nβ L_n(w) - γ/2 ||w - w*||²

    Uses pathwise gradient estimator with K=1 sample per step.

    Args:
        loss_batch_fn: (w_pytree, Xb, Yb) -> scalar loss
        data: (X, Y) full dataset
        wstar_flat: (d,) flattened ERM solution
        unravel_fn: flat array -> pytree converter
        n_data: total number of data points
        beta: inverse temperature
        gamma: localizer strength
        batch_size: minibatch size for stochastic ELBO
        optimizer: Optax optimizer
        clip_global_norm: gradient clipping threshold (None to disable)
        flow_static: static (non-array) part of flow from eqx.partition
        lift_consts: dict with U, E_perp, D_inv_sqrt, sigma_perp, wstar

    Returns:
        step_fn: (key, flow_params, opt_state, step_idx) -> ((flow_params, opt_state), metrics)
    """
    X, Y = data
    ref_dtype = wstar_flat.dtype

    beta_tilde = jnp.asarray(beta * n_data, dtype=ref_dtype)
    gamma_val = jnp.asarray(gamma, dtype=ref_dtype)

    @jax.jit
    def step_fn(key, flow_params, opt_state, step_idx):
        # Draw minibatch
        key_batch, key_elbo = jax.random.split(key)
        idx = jax.random.choice(key_batch, n_data, shape=(batch_size,), replace=True)
        Xb, Yb = X[idx], Y[idx]

        # Negative ELBO (for minimization)
        def neg_elbo(flow_params):
            # Rebuild flow from params + static
            flow = eqx.combine(flow_params, flow_static)

            # ==== Inlined InjectiveLift.sample_and_log_prob ====
            k1, k2 = jax.random.split(key_elbo)

            # Sample from latent flow: z ~ q_flow(z)
            z = flow.sample(k1, sample_shape=())  # (d_latent,)
            logq_z = flow.log_prob(z)  # scalar

            # Orthogonal noise: eps ~ N(0, sigma_perp^2 I)
            E_perp = lift_consts["E_perp"]
            d_perp = E_perp.shape[1]
            if d_perp > 0:
                eps = jax.random.normal(k2, (d_perp,))
                perp_component = E_perp @ (lift_consts["sigma_perp"] * eps)
            else:
                perp_component = 0.0

            # Lift to parameter space: w = w* + D^{-1/2}(U@z + E_perp@eps)
            w_flat = (
                lift_consts["wstar"] +
                lift_consts["D_inv_sqrt"] * (lift_consts["U"] @ z + perp_component)
            )

            # Log-determinant of block-triangular Jacobian
            logdet = jnp.sum(jnp.log(lift_consts["D_inv_sqrt"]))
            if d_perp > 0:
                logdet = logdet + d_perp * jnp.log(lift_consts["sigma_perp"])

            logq = logq_z + logdet
            # ==== end inline ====

            w = unravel_fn(w_flat)

            # Compute log p(w) = -nβ L_batch - γ/2 ||w - w*||²
            Ln_batch = loss_batch_fn(w, Xb, Yb)
            localizer = 0.5 * gamma_val * jnp.sum((w_flat - wstar_flat) ** 2)
            logp = -(beta_tilde * Ln_batch + localizer)

            # ELBO = logp - logq (return negative for minimization)
            return -(logp - logq)

        # Compute gradients w.r.t. flow params only
        loss_val, grads = jax.value_and_grad(neg_elbo)(flow_params)

        # Compute grad norm before clipping
        grad_norm = optax.global_norm(grads)

        # Optional gradient clipping
        if clip_global_norm is not None:
            clipped, _ = optax.clip_by_global_norm(clip_global_norm).update(
                grads, opt_state, flow_params
            )
            grads = clipped

        # Optax update
        updates, new_opt_state = optimizer.update(grads, opt_state, flow_params)
        new_flow_params = optax.apply_updates(flow_params, updates)

        # Metrics
        work_fge = jnp.asarray(batch_size / float(n_data), dtype=jnp.float64)
        metrics = {
            "elbo": -loss_val,  # Positive ELBO for logging
            "grad_norm": grad_norm,
            "work_fge": work_fge,  # FGE accounting (minibatch/n_data)
        }

        return (new_flow_params, new_opt_state), metrics

    return step_fn


# === Final LLC estimation ===


def estimate_lambda_final(
    key: jax.random.PRNGKey,
    dist: InjectiveLift,
    loss_full_fn: Callable,
    wstar_flat: jnp.ndarray,
    unravel_fn: Callable,
    n_data: int,
    beta: float,
    eval_samples: int,
) -> Tuple[float, Dict[str, jnp.ndarray]]:
    """Final Monte Carlo estimate of lambda_hat = nβ(E_q[L] - L_0).

    Args:
        key: JAX random key
        dist: Trained InjectiveLift distribution
        loss_full_fn: (w_pytree) -> scalar loss on full data
        wstar_flat: (d,) flattened ERM solution
        unravel_fn: flat array -> pytree
        n_data: total number of data points
        beta: inverse temperature
        eval_samples: number of MC samples

    Returns:
        (lambda_hat, extras): LLC estimate and diagnostic dict
    """
    # Draw eval_samples from q(w)
    keys = jax.random.split(key, eval_samples)

    def eval_one(k):
        w_flat, logq = dist.sample_and_log_prob(k)
        w = unravel_fn(w_flat)
        L = loss_full_fn(w)
        return {"L": L, "logq": logq}

    results = jax.vmap(eval_one)(keys)

    # Compute E_q[L_n(w)]
    L_samples = results["L"]  # (eval_samples,)
    E_L = jnp.mean(L_samples)
    L_std = jnp.std(L_samples)

    # Compute L_0 = L_n(w*)
    w0 = unravel_fn(wstar_flat)
    L0 = loss_full_fn(w0)

    # lambda_hat = nβ(E_L - L_0)
    lambda_hat = n_data * beta * (E_L - L0)

    extras = {
        "E_L": E_L,
        "L_std": L_std,
        "L0": L0,
        "logq_mean": jnp.mean(results["logq"]),
        "logq_std": jnp.std(results["logq"]),
    }

    return lambda_hat, extras


# === VIAlgorithm adapter ===


class _FlowAlgorithm:
    """Normalizing flow VI algorithm (FlowJAX-based).

    Implements VIAlgorithm protocol for flow-based variational inference.
    Uses manifold-plus-noise construction with RealNVP/MAF/NSF flows.
    """

    name = "flow"

    def run(
        self,
        rng_key: jax.random.PRNGKey,
        loss_batch_fn: Callable,
        loss_full_fn: Callable,
        wstar_flat: jnp.ndarray,
        unravel_fn: Callable[[jnp.ndarray], Any],
        data: Tuple[jnp.ndarray, jnp.ndarray],
        n_data: int,
        beta: float,
        gamma: float,
        vi_cfg: Any,
        whitener: Any = None,
    ) -> Dict[str, Any]:
        """Run flow VI algorithm.

        Args:
            rng_key: JAX random key
            loss_batch_fn: (w_pytree, Xb, Yb) -> scalar
            loss_full_fn: (w_pytree) -> scalar
            wstar_flat: (d,) flattened ERM solution
            unravel_fn: flat array -> pytree converter
            data: (X, Y) dataset
            n_data: number of data points
            beta: inverse temperature
            gamma: localizer strength
            vi_cfg: VIConfig instance
            whitener: Optional whitener (geometry transformation), defaults to identity

        Returns:
            Dict with keys:
                - lambda_hat: scalar LLC estimate
                - traces: dict of metric arrays (steps,)
                - extras: dict of final diagnostics
                - timings: dict of timing info
                - work: dict of computational work metrics
        """
        # Check if flowjax dependencies are available
        if not _FLOWJAX_AVAILABLE:
            raise ImportError(
                "Flow VI requires flowjax and equinox. Install with: uv sync --extra flowvi"
            ) from _IMPORT_ERROR

        # Convert legacy PRNGKey to new RBG key format if needed
        # (rng_key may come from jax.random.PRNGKey which has shape (2,), but RBG needs (4,))
        if hasattr(rng_key, "shape") and rng_key.shape == (2,):
            # Legacy key - extract seed and create RBG key
            # JAX legacy keys are uint32[2], first element is the seed
            seed = int(rng_key[0])
            rng_key = jax.random.key(seed)

        key_init, key_train, key_eval = jax.random.split(rng_key, 3)

        d = wstar_flat.size
        dtype_str = vi_cfg.dtype
        dtype = jnp.float32 if dtype_str == "float32" else jnp.float64

        # Cast data and wstar to target dtype
        X, Y = data
        X = X.astype(dtype)
        Y = Y.astype(dtype)
        wstar_flat = wstar_flat.astype(dtype)

        # Extract D_inv_sqrt from whitener (provided by run_vi's pre-pass)
        # If whitener is None, A_inv_sqrt will fail fast (no fallback)
        D_inv_sqrt = whitener.A_inv_sqrt.astype(dtype) if whitener is not None else jnp.ones(d, dtype=dtype)

        # Initialize InjectiveLift
        dist = init_injective_lift(
            key=key_init,
            wstar_flat=wstar_flat,
            d_latent=vi_cfg.d_latent,
            sigma_perp=vi_cfg.sigma_perp,
            D_inv_sqrt=D_inv_sqrt,
            flow_type=vi_cfg.flow_type,
            flow_depth=vi_cfg.flow_depth,
            flow_hidden=vi_cfg.flow_hidden,
        )

        # Partition flow into (params, static) to avoid carrying PjitFunctions through scan
        # Only partition inexact (float/complex) arrays for differentiation
        flow_params, flow_static = eqx.partition(
            dist.flow, lambda x: eqx.is_array(x) and jnp.issubdtype(x.dtype, jnp.inexact)
        )

        # Collect lifting constants (non-trainable, stay outside scan)
        lift_consts = {
            "U": dist.U,
            "E_perp": dist.E_perp,
            "D_inv_sqrt": dist.D_inv_sqrt,
            "sigma_perp": dist.sigma_perp,
            "wstar": dist.wstar,
        }

        # Build optimizer (reuse MFA's lr_schedule logic)
        if vi_cfg.lr_schedule == "cosine":
            lr_sched = optax.cosine_decay_schedule(
                init_value=vi_cfg.lr,
                decay_steps=vi_cfg.steps,
                alpha=0.0,
            )
        elif vi_cfg.lr_schedule == "linear_decay":
            lr_sched = optax.linear_schedule(
                init_value=vi_cfg.lr,
                end_value=0.0,
                transition_steps=vi_cfg.steps,
            )
        else:
            lr_sched = vi_cfg.lr

        # Warmup schedule
        warmup_steps = int(vi_cfg.lr_warmup_frac * vi_cfg.steps)
        if warmup_steps > 0:
            warmup_sched = optax.linear_schedule(
                init_value=0.0,
                end_value=vi_cfg.lr,
                transition_steps=warmup_steps,
            )
            lr_final = optax.join_schedules(
                schedules=[warmup_sched, lr_sched], boundaries=[warmup_steps]
            )
        else:
            lr_final = lr_sched

        optimizer = optax.adam(learning_rate=lr_final)
        opt_state = optimizer.init(flow_params)  # Initialize on flow_params only

        # Build ELBO step (now takes flow_static and lift_consts)
        step_fn = build_flow_elbo_step(
            loss_batch_fn=loss_batch_fn,
            data=(X, Y),
            wstar_flat=wstar_flat,
            unravel_fn=unravel_fn,
            n_data=n_data,
            beta=beta,
            gamma=gamma,
            batch_size=vi_cfg.batch_size,
            optimizer=optimizer,
            clip_global_norm=vi_cfg.clip_global_norm,
            flow_static=flow_static,
            lift_consts=lift_consts,
        )

        # Training loop with jax.lax.scan (carry only arrays)
        def scan_body(carry, step_idx):
            flow_params, opt_state, key, cumulative_fge = carry
            key, subkey = jax.random.split(key)
            (new_params, new_opt_state), metrics = step_fn(subkey, flow_params, opt_state, step_idx)
            # Accumulate FGE
            new_fge = cumulative_fge + metrics["work_fge"]
            metrics_with_fge = {**metrics, "cumulative_fge": new_fge}
            new_carry = (new_params, new_opt_state, key, new_fge)
            return new_carry, metrics_with_fge

        # Initialize and run scan (carry: flow_params, opt_state, rng, fge)
        t0 = time.time()
        carry_init = (flow_params, opt_state, key_train, jnp.array(0.0, dtype=jnp.float64))
        step_indices = jnp.arange(vi_cfg.steps)
        (flow_params_final, _, _, _), trace_dict = jax.lax.scan(scan_body, carry_init, step_indices)
        train_time = time.time() - t0

        # Rebuild final distribution from trained params
        flow_final = eqx.combine(flow_params_final, flow_static)
        dist_final = InjectiveLift(flow=flow_final, **lift_consts)

        # Final LLC estimation
        t0_eval = time.time()
        lambda_hat, eval_extras = estimate_lambda_final(
            key=key_eval,
            dist=dist_final,
            loss_full_fn=loss_full_fn,
            wstar_flat=wstar_flat,
            unravel_fn=unravel_fn,
            n_data=n_data,
            beta=beta,
            eval_samples=vi_cfg.eval_samples,
        )
        eval_time = time.time() - t0_eval

        # Flatten traces (from (steps,) arrays)
        traces = {k: v for k, v in trace_dict.items()}

        # Add flow-specific diagnostics to traces
        traces["d_latent"] = jnp.full(vi_cfg.steps, vi_cfg.d_latent, dtype=jnp.float32)
        traces["sigma_perp"] = jnp.full(vi_cfg.steps, vi_cfg.sigma_perp, dtype=jnp.float32)

        # cumulative_fge is already in traces from scan loop (proper minibatch accounting)

        # Extras
        extras = {
            **eval_extras,
            "final_dist": dist_final,  # Save trained distribution
        }

        # Timings
        timings = {
            "train": train_time,
            "eval": eval_time,
            "total": train_time + eval_time,
        }

        # Work metrics (harmonized with MFA structure)
        work = {
            "n_full_loss": vi_cfg.eval_samples,  # Final LLC evaluation samples
            "n_minibatch_grads": vi_cfg.steps,  # Number of minibatch gradient steps
            "sampler_flavour": "flow",
        }

        return {
            "lambda_hat": float(lambda_hat),
            "traces": traces,
            "extras": extras,
            "timings": timings,
            "work": work,
        }


def make_flow_algo() -> _FlowAlgorithm:
    """Factory function for flow VI algorithm."""
    return _FlowAlgorithm()


# Register the algorithm
from lambda_hat.vi import registry  # noqa: E402

registry.register("flow", make_flow_algo)
