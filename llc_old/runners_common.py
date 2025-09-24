"""Common utilities for sampler runners."""

import time
import jax


def estimate_sampling_time(total_time, warmup_time, Ln_full64_batched, init_thetas, M):
    """Estimate pure sampling time by subtracting warmup and eval overhead.
    Expects Ln_full64_batched: theta[C,d] -> L[C]."""
    t0 = time.perf_counter()
    _ = Ln_full64_batched(init_thetas).block_until_ready()
    per_call = time.perf_counter() - t0
    return max(0.0, total_time - warmup_time - per_call * float(M))
