# CLI Reference

**Auto-generated from `lambda_hat/cli.py`** — do not edit by hand.

## Main Command

```text
Usage: cli [OPTIONS] COMMAND [ARGS]...

  Lambda-Hat: Neural network Bayesian inference toolkit.

  Unified CLI for building targets, running samplers, managing artifacts,
  promoting results, and orchestrating Parsl workflows.

Options:
  --help  Show this message and exit.

Commands:
  artifacts  Artifact management (GC, list, TensorBoard).
  build      Build target artifact (Stage A: train neural network).
  promote    Promote plots to galleries (Stage C).
  sample     Run sampler on target (Stage B: MCMC/VI inference).
  workflow   Parsl workflow orchestration.

```

## `lambda-hat build`

```text
Usage: cli build [OPTIONS]

  Build target artifact (Stage A: train neural network).

Options:
  --config-yaml PATH  Path to composed YAML config  [required]
  --target-id TEXT    Target ID string (e.g., tgt_abc123)  [required]
  --experiment TEXT   Experiment name (defaults from config then env)
  --help              Show this message and exit.

```

## `lambda-hat sample`

```text
Usage: cli sample [OPTIONS]

  Run sampler on target (Stage B: MCMC/VI inference).

Options:
  --config-yaml PATH  Path to composed YAML config  [required]
  --target-id TEXT    Target ID to sample from  [required]
  --experiment TEXT   Experiment name (defaults from config then env)
  --help              Show this message and exit.

```

## `lambda-hat artifacts`

```text
Usage: cli artifacts [OPTIONS] COMMAND [ARGS]...

  Artifact management (GC, list, TensorBoard).

Options:
  --help  Show this message and exit.

Commands:
  gc  Garbage collect unreachable artifacts.
  ls  List experiments and runs.
  tb  Show TensorBoard logdir for an experiment.

```

## `lambda-hat promote`

```text
Usage: cli promote [OPTIONS] COMMAND [ARGS]...

  Promote plots to galleries (Stage C).

Options:
  --help  Show this message and exit.

Commands:
  gallery  Generate gallery HTML of all targets.
  single   Promote plots for a single target.

```

## `lambda-hat workflow`

```text
Usage: cli workflow [OPTIONS] COMMAND [ARGS]...

  Parsl workflow orchestration.

Options:
  --help  Show this message and exit.

Commands:
  llc     Run N×M targets×samplers workflow.
  optuna  Run Bayesian hyperparameter optimization.

```
