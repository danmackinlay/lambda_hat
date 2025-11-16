# lambda_hat/parsl_cards.py
"""Build Parsl configs from declarative YAML cards instead of executing Python files."""

from __future__ import annotations

import os
from pathlib import Path

from omegaconf import DictConfig, OmegaConf
from parsl.addresses import address_by_hostname
from parsl.config import Config
from parsl.executors import HighThroughputExecutor, ThreadPoolExecutor
from parsl.providers import SlurmProvider


def _scheduler_options_from_card(c: DictConfig) -> str:
    """Build SBATCH scheduler options from card fields."""
    lines = []
    if c.get("account"):
        lines.append(f"#SBATCH --account={c.account}")
    if c.get("qos"):
        lines.append(f"#SBATCH --qos={c.qos}")
    if c.get("constraint"):
        lines.append(f"#SBATCH --constraint={c.constraint}")
    gpn = int(c.get("gpus_per_node", 0) or 0)
    if gpn > 0:
        gres = "gpu"
        if c.get("gpu_type"):
            gres += f":{c.gpu_type}"
        gres += f":{gpn}"
        lines.append(f"#SBATCH --gres={gres}")
    extra = (c.get("scheduler_options_extra") or "").strip()
    if extra:
        lines.append(extra)
    return "\n".join(lines)


def build_parsl_config_from_card(card: DictConfig) -> Config:
    """Turn a small OmegaConf 'card' into a Parsl Config.

    Args:
        card: OmegaConf DictConfig with type: local|slurm and resource settings

    Returns:
        Parsl Config object ready to load

    Raises:
        ValueError: If card type is unknown
    """
    typ = str(card.get("type", "local")).lower()

    # Common: write Parsl run dir if provided
    run_dir = card.get("run_dir", "parsl_runinfo")

    if typ == "local":
        # Set JAX default behavior (mirrors parsl_config_local.py)
        os.environ.setdefault(
            "JAX_DEFAULT_PRNG_IMPL", card.get("jax_default_prng_impl", "threefry2x32")
        )
        max_threads = card.get("max_threads")
        if max_threads is None:
            import os as _os

            max_threads = min(_os.cpu_count() or 4, 8)
        return Config(
            executors=[
                ThreadPoolExecutor(
                    label=card.get("label", "local_threads"), max_threads=int(max_threads)
                )
            ],
            retries=int(card.get("retries", 1)),
            run_dir=run_dir,
        )

    if typ == "slurm":
        scheduler_options = _scheduler_options_from_card(card)
        provider = SlurmProvider(
            partition=card.get("partition", "normal"),
            scheduler_options=scheduler_options,
            worker_init=(
                card.get("worker_init")
                or """
module load python || true
source ~/.bashrc || true
export PATH="$HOME/.local/bin:$PATH"
export JAX_DEFAULT_PRNG_IMPL=threefry2x32
"""
            ).strip(),
            nodes_per_block=int(card.get("nodes_per_block", 1)),
            init_blocks=int(card.get("init_blocks", 0)),
            min_blocks=int(card.get("min_blocks", 0)),
            max_blocks=int(card.get("max_blocks", 50)),
            walltime=card.get("walltime", "01:59:00"),
            cores_per_node=int(card.get("cores_per_node", 2)),
            mem_per_node=int(card.get("mem_per_node", 64)),
        )
        htex = HighThroughputExecutor(
            label=card.get("label", "htex_slurm"),
            address=address_by_hostname(),
            max_workers=int(card.get("max_workers", 1)),
            provider=provider,
        )
        return Config(
            executors=[htex],
            retries=int(card.get("retries", 1)),
            run_dir=run_dir,
        )

    raise ValueError(f"Unknown parsl card type: {typ}")


def load_parsl_config_from_card(
    card_path: Path, dot_overrides: list[str] | None = None
) -> Config:
    """Load Parsl config from a YAML card with optional CLI overrides.

    Args:
        card_path: Path to YAML card file
        dot_overrides: OmegaConf dotlist overrides (e.g., ["walltime=04:00:00"])

    Returns:
        Parsl Config object

    Side effects:
        Writes resolved card to parsl_runinfo/selected_parsl_card.yaml for traceability
    """
    cfg = OmegaConf.load(card_path)
    if dot_overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(dot_overrides))

    # Persist the resolved card for reproducibility
    resolved = OmegaConf.to_yaml(cfg)
    runinfo = Path("parsl_runinfo")
    runinfo.mkdir(parents=True, exist_ok=True)
    (runinfo / "selected_parsl_card.yaml").write_text(resolved)

    return build_parsl_config_from_card(cfg)
