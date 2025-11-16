"""Parsl configuration for SLURM cluster execution.

Uses HighThroughputExecutor with SlurmProvider for HPC cluster job submission.
Configure partition, walltime, and resource limits below.
"""

from parsl.addresses import address_by_hostname
from parsl.config import Config
from parsl.executors import HighThroughputExecutor
from parsl.providers import SlurmProvider

config = Config(
    executors=[
        HighThroughputExecutor(
            label="htex_slurm",
            address=address_by_hostname(),
            max_workers=1,  # 1 worker per node (each task may be multi-threaded)
            provider=SlurmProvider(
                # Adjust partition based on your cluster configuration
                partition="normal",
                # Scheduler options for additional SLURM directives
                # Example: scheduler_options="#SBATCH --constraint=gpu"
                scheduler_options="",
                # Worker initialization (activate Python environment)
                worker_init="""
module load python || true
source ~/.bashrc || true
# Activate uv environment if needed
export PATH="$HOME/.local/bin:$PATH"
# Set JAX PRNG implementation for FlowJAX compatibility
export JAX_DEFAULT_PRNG_IMPL=threefry2x32
                """.strip(),
                # Resource allocation
                nodes_per_block=1,
                init_blocks=0,  # Start with 0 blocks, scale up on demand
                min_blocks=0,
                max_blocks=50,  # Maximum number of concurrent SLURM jobs
                walltime="01:59:00",  # 1 hour 59 minutes per job
                # Adjust cores per node based on your cluster
                cores_per_node=2,
                mem_per_node=64,  # GB
            ),
        )
    ],
    retries=1,
    run_dir="parsl_runinfo",
)
