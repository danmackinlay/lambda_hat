#!/usr/bin/env python3
"""
Entry point for lambda-hat command (Hydra-based LLC estimation).
"""

import hydra
from lambda_hat.config import Config, setup_config


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: Config) -> None:
    """Entry point for lambda-hat command (Hydra-based LLC estimation)."""
    # Import here to avoid circular imports
    from lambda_hat.entry import main as entry_main
    return entry_main(cfg)


if __name__ == "__main__":
    setup_config()
    main()