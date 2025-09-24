# tiny Submitit wrapper — optional
def map_slurm(cfgs, *, partition="gpu", timeout_min=119, gpus_per_job=1, cpus=4, mem_gb=16):
    """Submit one job per cfg; each job uses 1 GPU if available."""
    # TODO: lift minimal submitit executor from your llc.execution.SubmititExecutor.
    raise NotImplementedError("SLURM execution not yet implemented")