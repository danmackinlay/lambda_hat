# llc/execution.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable, Callable, List, Any, Optional
import os
import time
import threading

@dataclass
class BaseExecutor:
    def map(self, fn: Callable[[Any], Any], items: Iterable[Any]) -> List[Any]:
        raise NotImplementedError

def _run_with_soft_timeout(fn, arg, seconds):
    """Runs fn(arg). If seconds>0 and time exceeds, raises TimeoutError."""
    if not seconds or seconds <= 0:
        return fn(arg)
    result = {}
    exc = {}

    def target():
        try:
            result["value"] = fn(arg)
        except BaseException as e:
            exc["err"] = e

    th = threading.Thread(target=target, daemon=True)
    th.start()
    th.join(seconds)
    if th.is_alive():
        raise TimeoutError(f"Task exceeded {seconds}s")
    if "err" in exc:
        raise exc["err"]
    return result.get("value")

# ----------------- Local -----------------
class LocalExecutor(BaseExecutor):
    def __init__(self, workers: int = 0, timeout_s: int | None = None):
        self.workers = int(workers)
        self.timeout_s = int(timeout_s) if timeout_s else 0

    def map(self, fn, items):
        items = list(items)
        if self.workers in (0, 1):
            out = []
            for it in items:
                out.append(_run_with_soft_timeout(fn, it, self.timeout_s))
            return out
        from concurrent.futures import ProcessPoolExecutor, as_completed
        outs = [None]*len(items)
        with ProcessPoolExecutor(max_workers=self.workers) as ex:
            futs = {ex.submit(fn, it): i for i, it in enumerate(items)}
            for fut in as_completed(futs):
                i = futs[fut]
                outs[i] = fut.result()  # process-local timeouts aren't enforced; keep simple
        return outs

# ----------------- Submitit / SLURM -----------------
class _SubmititExperiment:
    """Small checkpointable wrapper for submitit with timeout handling"""
    def __init__(self, payload):
        # payload is a dict with {"cfg": {...}}
        self.payload = payload

    def __call__(self):
        from llc.tasks import run_experiment_task
        return run_experiment_task(self.payload["cfg"])

    def checkpoint(self):
        # Requeue same work on preemption/timeout
        import submitit
        return submitit.helpers.DelayedSubmission(_SubmititExperiment(self.payload))

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
        slurm_signal_delay_s: int = 120,    # NEW: grace period before kill
    ):
        try:
            import submitit  # lazy import
        except Exception as e:
            raise RuntimeError(
                "submitit is not installed. Install extra: `uv sync --extra slurm` "
                "or `pip install llc[slurm]`"
            ) from e

        self._submitit = submitit
        self.executor = submitit.AutoExecutor(folder=folder)
        base = dict(
            timeout_min=timeout_min,
            cpus_per_task=cpus_per_task,
            mem_gb=mem_gb,
            name=name,
            slurm_signal_delay_s=slurm_signal_delay_s,  # get SIGTERM early
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
        jobs = self.executor.map_array(_SubmititExperiment, list(items))
        return [j.result() for j in jobs]

# ----------------- Modal (serverless) -----------------
class ModalExecutor(BaseExecutor):
    """
    Calls a pre-decorated Modal function with `.map`. You provide the function handle.
    This keeps modal-specific code out of your core pipeline.
    """
    def __init__(self, remote_fn, options: Optional[dict] = None):
        # Defensive check that modal is available
        try:
            import modal  # lazy import check
        except Exception as e:
            raise RuntimeError(
                "modal is not installed. Install extra: `uv sync --extra modal` "
                "or `pip install llc[modal]`"
            ) from e
        
        # options e.g. {"gpu": "L40S", "timeout": 60*60, "cpu": 8, "memory": "24Gi"}
        self.remote_fn = remote_fn.options(**options) if options else remote_fn

    def map(self, fn_ignored, items):
        # We ignore `fn` and call the remote Modal function directly.
        return list(self.remote_fn.map(list(items)))

# ----------------- factory -----------------
def get_executor(backend: str, **kwargs) -> BaseExecutor:
    backend = (backend or "local").lower()
    if backend == "local":
        return LocalExecutor(
            workers=kwargs.get("workers", 0),
            timeout_s=kwargs.get("timeout_s", None)
        )
    if backend == "submitit":
        return SubmititExecutor(**kwargs)
    if backend == "modal":
        remote_fn = kwargs.get("remote_fn")
        if remote_fn is None:
            raise ValueError("Modal backend requires `remote_fn=` (a decorated Modal function).")
        return ModalExecutor(remote_fn=remote_fn, options=kwargs.get("options"))
    raise ValueError(f"Unknown backend: {backend}")