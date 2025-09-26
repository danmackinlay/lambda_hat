#!/usr/bin/env python3
"""
Entry point for lambda-hat command (Hydra-based LLC estimation).
"""

import hydra
from lambda_hat.config import Config
from lambda_hat.main import main as main_func


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: Config) -> None:
    """Entry point for lambda-hat command (Hydra-based LLC estimation)."""
    # Import here to avoid circular imports
    return main_func(cfg)


if __name__ == "__main__":
    main()
