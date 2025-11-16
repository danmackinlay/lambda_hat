"""Parsl apps for Optuna hyperparameter optimization workflow.

These apps wrap the lambda_hat.runners functions for parallel execution
with Parsl. Each app is a @python_app that can be scheduled on remote workers.
"""

from parsl import python_app


@python_app
def compute_hmc_reference(
    problem_spec: dict, out_ref_json: str, budget_sec: int = 36000, seed: int = 42
):
    """Compute HMC reference LLC estimate (Parsl app).

    Args:
        problem_spec: Dict with problem configuration {model, data, teacher, seed}
        out_ref_json: Path to write reference JSON
        budget_sec: Time budget in seconds (default: 36000 = 10 hours)
        seed: Random seed (default: 42)

    Returns:
        dict: Reference metrics with llc_ref, se_ref, diagnostics
    """
    from lambda_hat.runners.hmc_reference import run_hmc_reference

    return run_hmc_reference(problem_spec, out_ref_json, budget_sec, seed)


@python_app
def run_method_trial(
    problem_spec: dict,
    method_cfg: dict,
    ref_llc: float,
    out_metrics_json: str,
    budget_sec: int = 6000,
    seed: int = None,
):
    """Run method trial with given hyperparameters (Parsl app).

    Args:
        problem_spec: Dict with problem configuration
        method_cfg: Dict with method name and hyperparameters
        ref_llc: Reference LLC from HMC
        out_metrics_json: Path to write metrics JSON
        budget_sec: Time budget in seconds (default: 6000 = 100 minutes)
        seed: Random seed (default: None, uses method_cfg or 12345)

    Returns:
        dict: Trial metrics with llc_hat, se_hat, runtime_sec, error, diagnostics
    """
    from lambda_hat.runners.run_method import run_method_trial as _run_trial

    return _run_trial(problem_spec, method_cfg, ref_llc, out_metrics_json, budget_sec, seed)
