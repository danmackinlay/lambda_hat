import hydra
from pathlib import Path
from omegaconf import DictConfig
from lambda_hat.promote.core import promote


@hydra.main(config_path="../conf", config_name="promote", version_base=None)
def main_promote(cfg: DictConfig) -> None:
    samplers = [s.strip() for s in cfg.samplers.split(",") if s.strip()]
    promote(
        runs_root=Path(cfg.runs_root),
        samplers=samplers,
        outdir=Path(cfg.outdir),
        plot_name=cfg.plot_name,
    )


if __name__ == "__main__":
    main_promote()
