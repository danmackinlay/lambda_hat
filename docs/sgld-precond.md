# Preconditioned SGLD Options

You can optionally enable diagonal preconditioning for SGLD to improve practical efficiency.

## Usage

```bash
# RMSProp/pSGLD-style
uv run python -m llc run --sgld-precond=rmsprop --sgld-beta2=0.999 --sgld-eps=1e-8

# Adam-preconditioned SGLD
uv run python -m llc run --sgld-precond=adam --sgld-beta1=0.9 --sgld-beta2=0.999 \
  --sgld-eps=1e-8 --sgld-bias-correction
```

## Important Note

These adaptive SGMCMC variants are heuristics: the per-parameter scale changes during sampling, so the chain is not strictly stationary for the target at all times. They are widely used in practice and often yield better mixing than non-adaptive versions. Set `--sgld-precond=none` if you prefer the plain kernel.

**Adam preconditioning implementation:** As of the latest version, Adam preconditioning now uses the bias-corrected first moment (`m̂`) for the drift term, aligning with the standard Adam optimizer. This ensures proper behavior especially in early iterations when the exponential moving averages are warming up.

## Configuration Options

- `--sgld-precond`: Choose `none` (default), `rmsprop`, or `adam`
- `--sgld-beta1`: Adam first-moment decay (default: 0.9)
- `--sgld-beta2`: RMSProp/Adam second-moment decay (default: 0.999)
- `--sgld-eps`: Numerical stabilizer (default: 1e-8)
- `--sgld-bias-correction` / `--no-sgld-bias-correction`: Adam bias correction (default: on)

## Sweeps with SGLD Variants

In sweeps, you can declare `sgld` variants with different `--sgld-precond` values in your study YAML, and each will run as its own job:

```yaml
samplers:
  - name: sgld
    overrides: {sgld_precond: none}
  - name: sgld
    overrides: {sgld_precond: rmsprop, sgld_beta2: 0.99}
  - name: sgld
    overrides: {sgld_precond: adam, sgld_beta1: 0.9, sgld_beta2: 0.999}
```

This makes it easy to compare different preconditioning strategies on the same problems.