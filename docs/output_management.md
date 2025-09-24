# Output Management and Directory Structure

This project relies on Hydra to manage output directories for experiments. Hydra ensures that each run has a unique, isolated working directory, preventing conflicts and maintaining reproducibility.

## Directory Structure

By default, Hydra organizes outputs based on the date and time the experiment was launched.

### Single Runs

When running a single configuration:

```bash
uv run python train.py model=small
```

The outputs are saved in the `outputs/` directory:

```
outputs/
└── YYYY-MM-DD/
    └── HH-MM-SS/
        ├── config.yaml           # Resolved configuration for this run (saved by llc/artifacts.py)
        ├── metrics_summary.csv   # Results
        ├── .hydra/               # Hydra internal logs and configuration copies
        └── train.log             # Application log
```

The application code (`llc/artifacts.py`) saves artifacts relative to the current working directory, which Hydra sets to this unique output path.

### Multi-Run Sweeps

When running a parameter sweep using `--multirun` (or `-m`):

```bash
uv run python train.py -m model=small,base
```

The outputs are saved in the `multirun/` directory:

```
multirun/
└── YYYY-MM-DD/
    └── HH-MM-SS/
        ├── 0/                    # Output directory for Job 0 (model=small)
        ├── 1/                    # Output directory for Job 1 (model=base)
        ├── multirun.yaml         # Overview of the sweep parameters
        └── .hydra/
```

## Customizing Output Paths

You can customize the output directory structure using Hydra configuration overrides.

```bash
# Example: Organize by experiment name instead of timestamp
uv run python train.py +experiment_name=baseline_test \
    hydra.run.dir=outputs/\${experiment_name}/\${now:%Y-%m-%d_%H-%M-%S}
```

## Reproducibility and Rerunning Experiments

Hydra does not automatically cache results. If you run the same command twice, Hydra will create two separate output directories, and the experiment will be executed again.

To reproduce an experiment exactly, you can use the configuration saved in the output directory of a previous run.

```bash
# Rerun using the configuration from a previous run
# Note: Point config-dir to the .hydra subdirectory where the run's configs are stored
uv run python train.py --config-dir outputs/YYYY-MM-DD/HH-MM-SS/.hydra/
```

## Aggregating Results

After a multi-run sweep completes, the results (e.g., `metrics_summary.csv`) are scattered across the individual job directories (0/, 1/, ...). You will need to aggregate these results for analysis.

```python
# Example aggregation script snippet (Python/Pandas)
import pandas as pd
from pathlib import Path
from omegaconf import OmegaConf

def aggregate_results(sweep_dir):
    sweep_path = Path(sweep_dir)
    results = []

    for job_dir in sweep_path.iterdir():
        if job_dir.is_dir() and job_dir.name.isdigit():
            metrics_file = job_dir / 'metrics_summary.csv'
            # We load the config saved by llc/artifacts.py
            config_file = job_dir / 'config.yaml'

            if metrics_file.exists() and config_file.exists():
                # Load metrics (CSV is indexed by sampler name)
                metrics = pd.read_csv(metrics_file, index_col=0)
                metrics_flat = metrics.to_dict('index')

                # Load config using OmegaConf
                config = OmegaConf.load(config_file)

                # Extract key parameters (customize as needed)
                record = {
                    'job_id': job_dir.name,
                    'n_data': config.data.n_data,
                    'target_params': config.model.target_params,
                    'seed': config.seed,
                }

                # Add LLC means and ESS for each sampler
                for sampler, data in metrics_flat.items():
                    record[f'{sampler}_llc_mean'] = data.get('llc_mean')
                    record[f'{sampler}_ess'] = data.get('ess')

                results.append(record)

    if not results:
        return pd.DataFrame()

    df = pd.DataFrame(results).sort_values(by='job_id').reset_index(drop=True)
    return df

# Usage:
# df_summary = aggregate_results('multirun/YYYY-MM-DD/HH-MM-SS')
```