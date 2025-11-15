"""Parsl configuration for local execution (testing and development).

Uses ThreadPoolExecutor for parallel execution on the local machine.
Suitable for small experiments and testing the workflow.
"""

import os

from parsl.config import Config
from parsl.executors import ThreadPoolExecutor

# Set JAX PRNG implementation to threefry2x32 for FlowJAX compatibility
# ThreadPoolExecutor shares process memory, so this affects all threads
# (SlurmProvider uses worker_init for this; see parsl_config_slurm.py)
os.environ.setdefault("JAX_DEFAULT_PRNG_IMPL", "threefry2x32")

# Use all available cores, but cap at 8 to avoid overwhelming the system
max_workers = min(os.cpu_count() or 4, 8)

config = Config(
    executors=[
        ThreadPoolExecutor(
            label="local_threads",
            max_threads=max_workers,
        )
    ],
    retries=1,
    run_dir="parsl_runinfo",
)
