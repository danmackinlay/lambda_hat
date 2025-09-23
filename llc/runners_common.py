"""Common utilities for sampler runners."""

import time
import jax


def estimate_sampling_time(total_time, warmup_time, Ln_full64, init_thetas, M):
    """Estimate pure sampling time by subtracting warmup and eval overhead."""
    Ln_vmapped = jax.jit(jax.vmap(Ln_full64))
    t0 = time.perf_counter()
    _ = Ln_vmapped(init_thetas).block_until_ready()
    per_call = time.perf_counter() - t0
    return max(0.0, total_time - warmup_time - per_call * float(M))
