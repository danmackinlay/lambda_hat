# lambda_hat/entrypoints/workflow.py
from __future__ import annotations

import hydra
from omegaconf import DictConfig

# Import the reusable logic function from the sample entrypoint
from lambda_hat.entrypoints.sample import run_sampling_logic


# This entrypoint uses the unified 'workflow' configuration.
@hydra.main(config_path="../conf", config_name="workflow", version_base=None)
def main_workflow(cfg: DictConfig) -> None:
    print("=== LLC: Workflow (Configuration-Driven Sampling) ===")

    # The logic is identical to the standard sampling entrypoint.
    run_sampling_logic(cfg)


if __name__ == "__main__":
    main_workflow()
