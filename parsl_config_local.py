"""Parsl configuration for local execution (testing and development).

Uses ThreadPoolExecutor for parallel execution on the local machine.
Suitable for small experiments and testing the workflow.
"""

import os

from parsl.config import Config
from parsl.executors import ThreadPoolExecutor

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
