# llc/execution.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable, Callable, List, Any, Optional
import os

@dataclass
class BaseExecutor:
    def map(self, fn: Callable[[Any], Any], items: Iterable[Any]) -> List[Any]:
        raise NotImplementedError

# ----------------- Local -----------------
class LocalExecutor(BaseExecutor):
    def __init__(self, workers: int = 0):
        self.workers = int(workers)

    def map(self, fn, items):
        items = list(items)
        if self.workers in (0, 1):
            return [fn(x) for x in items]  # serial & debuggable
        # light-weight parallelism without new deps
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=self.workers) as ex:
            return list(ex.map(fn, items))

# ----------------- Submitit / SLURM -----------------
class SubmititExecutor(BaseExecutor):
    def __init__(
        self,
        folder: str = "slurm_logs",
        timeout_min: int = 60,
        slurm_partition: Optional[str] = None,
        gpus_per_node: int = 0,
        cpus_per_task: int = 4,
        mem_gb: int = 16,
        name: str = "llc",
        # NEW: allow arbitrary slurm args (account/qos/constraint/etc.)
        slurm_additional_parameters: Optional[dict] = None,
        # (kept for backward-compat) still accept generic extras:
        additional_params: Optional[dict] = None,
    ):
        try:
            import submitit  # lazy
        except Exception as e:
            raise RuntimeError("submitit is not installed. `pip install submitit`.") from e

        self._submitit = submitit
        self.executor = submitit.AutoExecutor(folder=folder)
        base = dict(
            timeout_min=timeout_min,
            cpus_per_task=cpus_per_task,
            mem_gb=mem_gb,
            name=name,
        )
        if slurm_partition:
            base["slurm_partition"] = slurm_partition
        if gpus_per_node:
            base["gpus_per_node"] = gpus_per_node
        if additional_params:
            base.update(additional_params)

        # NEW: pass raw Slurm fields cleanly
        if slurm_additional_parameters:
            base["slurm_additional_parameters"] = slurm_additional_parameters

        self.executor.update_parameters(**base)

    def map(self, fn, items):
        # submitit needs a top-level callable; fn must be importable
        jobs = self.executor.map_array(fn, list(items))
        return [j.result() for j in jobs]

# ----------------- Modal (serverless) -----------------
class ModalExecutor(BaseExecutor):
    """
    Calls a pre-decorated Modal function with `.map`. You provide the function handle.
    This keeps modal-specific code out of your core pipeline.
    """
    def __init__(self, remote_fn, options: Optional[dict] = None):
        # options e.g. {"gpu": "L40S", "timeout": 60*60, "cpu": 8, "memory": "24Gi"}
        self.remote_fn = remote_fn.options(**options) if options else remote_fn

    def map(self, fn_ignored, items):
        # We ignore `fn` and call the remote Modal function directly.
        return list(self.remote_fn.map(list(items)))

# ----------------- factory -----------------
def get_executor(backend: str, **kwargs) -> BaseExecutor:
    backend = (backend or "local").lower()
    if backend == "local":
        return LocalExecutor(workers=kwargs.get("workers", 0))
    if backend == "submitit":
        return SubmititExecutor(**kwargs)
    if backend == "modal":
        remote_fn = kwargs.get("remote_fn")
        if remote_fn is None:
            raise ValueError("Modal backend requires `remote_fn=` (a decorated Modal function).")
        return ModalExecutor(remote_fn=remote_fn, options=kwargs.get("options"))
    raise ValueError(f"Unknown backend: {backend}")