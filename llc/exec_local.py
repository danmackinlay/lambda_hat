import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List
from .run import run_one

def _worker(cfg):
    # Single-GPU isolation if CUDA: let orchestrator set CUDA_VISIBLE_DEVICES per process
    return run_one(cfg, save_artifacts=True, skip_if_exists=True)

def map_local(cfgs: List, *, gpus: List[int] = None) -> List[dict]:
    """One cfg per task. If gpus provided, round-robin set CUDA_VISIBLE_DEVICES."""
    outs = [None] * len(cfgs)

    def submit_with_gpu(i, cfg):
        if gpus is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpus[i % len(gpus)])
            os.environ["JAX_PLATFORMS"] = "cuda"
            os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
        else:
            os.environ["JAX_PLATFORMS"] = "cpu"
        return _worker(cfg)

    max_workers = len(gpus) if gpus else None
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futs = {ex.submit(submit_with_gpu, i, c): i for i, c in enumerate(cfgs)}
        for f in as_completed(futs):
            outs[futs[f]] = f.result()

    return outs