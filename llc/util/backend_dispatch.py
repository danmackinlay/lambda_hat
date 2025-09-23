# llc/util/backend_dispatch.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence

from llc.util.backend_bootstrap import (
    select_jax_platform,
    validate_modal_gpu_types,
    pick_modal_remote_fn,
    schema_stamp,
)
from llc.execution import get_executor
from llc.util.modal_utils import extract_modal_runs_locally


@dataclass
class BackendOptions:
    backend: str  # "local" | "submitit" | "modal"
    gpu_mode: str  # "off" | "vectorized" | "sequential"
    gpu_types: str = ""  # e.g. "H100,A100,L40S"
    local_workers: int = 0
    # submitit
    slurm_partition: Optional[str] = None
    slurm_account: Optional[str] = None
    timeout_min: int = 119
    cpus: int = 4
    mem_gb: int = 16
    slurm_signal_delay_s: int = 120
    # modal
    modal_autoscaler_cap: int = 8
    modal_chunk_size: int = 16
    modal_auto_extract: bool = True  # download artifacts after each job


def _build_submitit_kwargs(opts: BackendOptions) -> Dict[str, Any]:
    gpus_per_node = 1 if opts.gpu_mode != "off" else 0
    kw = dict(
        gpus_per_node=gpus_per_node,
        timeout_min=opts.timeout_min,
        cpus_per_task=opts.cpus,
        mem_gb=opts.mem_gb,
        slurm_signal_delay_s=opts.slurm_signal_delay_s,
    )
    if opts.slurm_partition:
        kw["slurm_partition"] = opts.slurm_partition
    # Pass account via Submitit's slurm_additional_parameters
    if opts.slurm_account:
        kw["slurm_additional_parameters"] = {"account": opts.slurm_account}
    return kw


def _build_modal_options(n_jobs: int, cap: int) -> Dict[str, int]:
    maxc = min(cap, max(1, n_jobs))
    return {
        "max_containers": maxc,
        "min_containers": 0,
        "buffer_containers": max(1, maxc // 2),
    }


def prepare_payloads(
    cfgs: Sequence, *, save_artifacts: bool, skip_if_exists: bool, gpu_mode: str
) -> List[dict]:
    return [schema_stamp(cfg, save_artifacts, skip_if_exists, gpu_mode) for cfg in cfgs]


def run_jobs(
    *,
    cfg_payloads: List[dict],
    opts: BackendOptions,
    task_fn: Callable[[dict], dict],  # usually llc.tasks.run_experiment_task
) -> List[dict]:
    backend = (opts.backend or "local").lower()
    # Configure platform/env locally; remote backends do it server-side
    if backend == "local":
        select_jax_platform(opts.gpu_mode)

    # For Modal decorator GPU selection
    validate_modal_gpu_types(opts.gpu_types)

    if backend == "modal":
        from llc.modal_app import app, ping

        remote_fn = pick_modal_remote_fn(opts.gpu_mode)
        results: List[dict] = []
        with app.run():
            # preflight to fail fast on billing/scheduling issues
            try:
                ping.remote()
            except Exception as e:
                msg = str(e).lower()
                if any(
                    k in msg
                    for k in ["insufficient", "funds", "balance", "quota", "billing"]
                ):
                    raise SystemExit(
                        "Modal preflight failed: likely out of funds or billing disabled. "
                        "Top up balance or enable auto-recharge, then retry."
                    )
                raise
            # chunked map to avoid very long heartbeats
            modal_opts = _build_modal_options(
                len(cfg_payloads), opts.modal_autoscaler_cap
            )
            for i in range(0, len(cfg_payloads), opts.modal_chunk_size):
                batch = cfg_payloads[i : i + opts.modal_chunk_size]
                ex = get_executor(
                    backend="modal", remote_fn=remote_fn, options=modal_opts
                )
                batch_results = ex.map(task_fn, batch)
                if opts.modal_auto_extract:
                    for r in batch_results:
                        if r.get("status") != "error":
                            try:
                                extract_modal_runs_locally(r)
                            except Exception:
                                pass
                results.extend(batch_results)
        return results

    if backend == "submitit":
        ex = get_executor(backend="submitit", **_build_submitit_kwargs(opts))
        return ex.map(task_fn, cfg_payloads)

    if backend == "local":
        ex = get_executor(backend="local", workers=opts.local_workers)
        return ex.map(task_fn, cfg_payloads)

    raise ValueError(f"Unknown backend: {backend}")
