# llc/execution.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable, Callable, List, Any, Optional
import threading
import os
import time


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


def _submitit_safe_call(fn, item):
    """Wrapper that ensures Submitit jobs always return structured dict with status."""
    import traceback

    t0 = time.time()
    stage = "start"
    try:
        out = fn(item)  # run_experiment_task(cfg_dict)
        if isinstance(out, dict):
            out["status"] = "ok"
            out["duration_s"] = time.time() - t0
            return out
        else:
            # Legacy fallback for non-dict returns
            return {
                "status": "ok",
                "result": out,
                "duration_s": time.time() - t0,
            }
    except Exception as e:
        return {
            "status": "error",
            "error_type": e.__class__.__name__,
            "error": str(e)[:2000],
            "traceback": "".join(traceback.format_exc())[-4000:],
            "duration_s": time.time() - t0,
        }


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

        outs = [None] * len(items)
        with ProcessPoolExecutor(max_workers=self.workers) as ex:
            futs = {ex.submit(fn, it): i for i, it in enumerate(items)}
            for fut in as_completed(futs):
                i = futs[fut]
                outs[i] = (
                    fut.result()
                )  # process-local timeouts aren't enforced; keep simple
        return outs


# ----------------- Submitit / SLURM -----------------
class _SubmititExperiment:
    """Small checkpointable wrapper for submitit with timeout handling"""

    def __init__(self, wrapper_fn, fn, item):
        # Store the wrapper, function and item to call
        self.wrapper_fn = wrapper_fn
        self.fn = fn
        self.item = item

    def __call__(self):
        return self.wrapper_fn(self.fn, self.item)

    def checkpoint(self):
        # Requeue same work on preemption/timeout
        import submitit

        return submitit.helpers.DelayedSubmission(
            _SubmititExperiment(self.wrapper_fn, self.fn, self.item)
        )


class SubmititExecutor(BaseExecutor):
    def __init__(
        self,
        folder: str = "slurm_logs",
        timeout_min: int = 119,
        slurm_partition: Optional[str] = None,
        gpus_per_node: int = 0,
        cpus_per_task: int = 4,
        mem_gb: int = 16,
        name: str = "llc",
        slurm_additional_parameters: Optional[dict] = None,
        # (kept for backward-compat) still accept generic extras:
        additional_params: Optional[dict] = None,
        slurm_signal_delay_s: int = 120,  # NEW: grace period before kill
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
        # Create wrapper jobs that call _submitit_safe_call(fn, item) for each item
        items_list = list(items)
        jobs = [
            self.executor.submit(_SubmititExperiment(_submitit_safe_call, fn, item))
            for item in items_list
        ]
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
            pass  # lazy import check
        except Exception as e:
            raise RuntimeError(
                "modal is not installed. Install extra: `uv sync --extra modal` "
                "or `pip install llc[modal]`"
            ) from e

        # Runtime `.options(...)` is not supported on modal.Function.
        # All resource/timeout/volume settings must be set on the decorator in modal_app.py.
        self.remote_fn = remote_fn
        self._options = options or {}
        self._hang_timeout = int(
            os.environ.get("LLC_MODAL_CLIENT_HANG_TIMEOUT_S", "119")
        )

        # (Optional) Only autoscaler hints are adjustable at runtime.
        ac = {
            k: self._options[k]
            for k in (
                "min_containers",
                "max_containers",
                "buffer_containers",
                "scaledown_window",
            )
            if k in self._options
        }
        if ac:
            try:
                self.remote_fn.update_autoscaler(**ac)
            except Exception:
                # Non-fatal; ignore autoscaler tweak failures.
                pass

    def _map_blocking(self, items):
        # Existing behavior
        return list(self.remote_fn.map(list(items)))

    def map(self, fn, items):
        # For Modal, we assume remote_fn has the same signature as fn
        # The caller should pass a remote_fn that matches the intended function
        if self._hang_timeout <= 0:
            return self._map_blocking(items)

        result_box, exc_box = {}, {}

        def runner():
            try:
                result_box["value"] = self._map_blocking(items)
            except BaseException as e:
                exc_box["err"] = e

        t = threading.Thread(target=runner, daemon=True)
        t.start()
        t.join(self._hang_timeout)

        if t.is_alive():
            # We're stuck in scheduling. Exit cleanly with guidance.
            raise RuntimeError(
                "Modal: call did not start within "
                f"{self._hang_timeout}s (likely scheduling stalled: out of funds or disabled billing). "
                "Set LLC_MODAL_CLIENT_HANG_TIMEOUT_S to adjust, or top up your Modal balance and retry."
            )
        if "err" in exc_box:
            msg = str(exc_box["err"]).lower()
            if any(
                k in msg
                for k in ["insufficient", "funds", "balance", "quota", "billing"]
            ):
                raise RuntimeError(
                    "Modal billing/quota error while scheduling or running. "
                    "Top up balance or enable auto-recharge, then retry."
                ) from exc_box["err"]
            raise exc_box["err"]
        return result_box["value"]


# ----------------- factory -----------------
def get_executor(backend: str, **kwargs) -> BaseExecutor:
    backend = (backend or "local").lower()
    if backend == "local":
        return LocalExecutor(
            workers=kwargs.get("workers", 0), timeout_s=kwargs.get("timeout_s", None)
        )
    if backend == "submitit":
        return SubmititExecutor(**kwargs)
    if backend == "modal":
        remote_fn = kwargs.get("remote_fn")
        if remote_fn is None:
            raise ValueError(
                "Modal backend requires `remote_fn=` (a decorated Modal function)."
            )
        return ModalExecutor(remote_fn=remote_fn, options=kwargs.get("options"))
    raise ValueError(f"Unknown backend: {backend}")
