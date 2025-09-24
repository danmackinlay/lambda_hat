# Local Learning Coefficient Estimation

## Motivation

The **Local Learning Coefficient (LLC)**, introduced in Singular Learning Theory, captures the local effective dimension of a model around an empirical risk minimizer $w_0$. It governs the asymptotic form of Bayesian generalization error in singular models.

Unlike traditional complexity measures, the LLC characterizes how fast the posterior contracts around $w_0$, providing insight into the local geometry of the loss landscape.

## Local posterior

We define the *local posterior* as
$$
\pi(w) \propto \exp\{-n \beta L_n(w)\} \,\exp\left\{-\tfrac{\gamma}{2}\|w-w_0\|^2\right\},
$$
with dataset size $n$, inverse-temperature $\beta$, and localization strength $\gamma$.

- $L_n(w)$: empirical risk (e.g., mean squared error)
- $w_0$: empirical risk minimizer (ERM solution)
- $\beta = 1/\log n$ is the standard choice from SLT
- $\gamma$ is set from a prior radius or fixed to localize around $w_0$

This distribution concentrates mass near the ERM optimum while tempering the likelihood with $\beta$. The balance between these terms determines the effective local dimension.

## Estimator

Our estimator for the LLC is
$$
\hat\lambda = n \beta \Big(\tfrac{1}{T}\sum_{t=1}^T L_n(w_t) - L_n(w_0)\Big),
$$
where $\{w_t\}_{t=1}^T$ are samples from the local posterior.

This is exactly Eq. (3.4)/(3.8) of Hitchcock & Hoogland (2025). The term $L_n(w_0)$ is the empirical risk at the optimum, computed once. The expectation $\mathbb{E}[L_n(w)]$ is estimated from MCMC samples.

### Intuition

The LLC measures how much the loss increases, on average, when we move away from $w_0$ according to the local posterior. Higher values indicate more effective parameters contributing to the model complexity.

## Diagnostics

We estimate uncertainty using MCMC diagnostics:

- **Effective sample size (ESS)**: Number of independent samples
- $\hat R$ **convergence check**: Should be $\leq 1.1$ for convergence
- **Efficiency metrics**: ESS/sec, ESS/FDE, wall-normalized variance

### ArviZ Integration

All samples are converted to ArviZ `InferenceData` format, enabling:

- Running LLC plots showing convergence
- Rank plots for mixing diagnostics
- Autocorrelation analysis
- Energy diagnostics (for HMC/MCLMC)
- Parameter trace plots

### Variance Estimation

The sampling variance of $\hat\lambda$ is estimated as:
$$
\mathrm{Var}(\hat\lambda) \approx \frac{\sigma^2}{\mathrm{ESS}},
$$
where $\sigma^2$ is the sample variance of the loss values and ESS accounts for autocorrelation.

## Targets and samplers

### Targets

- **Quadratic**: Analytical $L_n(\theta) = \frac{1}{2}\|\theta\|^2$ for validation
- **MLP**: Teacher-student multilayer perceptrons with ReLU activation
- **Deep Linear Networks (DLN)**: $f(x) = W_M \cdots W_1 x$ with known analytical properties

### Samplers

- **SGLD**: Stochastic Gradient Langevin Dynamics with optional preconditioning (Adam/RMSProp)
- **SGNHT**: Stochastic Gradient Nose-Hoover Thermostat with momentum and adaptive friction
- **HMC**: Hamiltonian Monte Carlo via BlackJAX with window adaptation
- **MCLMC**: Microcanonical Langevin Monte Carlo via BlackJAX 1.2.5 fractional API

Each sampler is tuned for the local posterior setting, with careful attention to step sizes and batch sizes for stochastic methods.

## Implementation Notes

### Precision

- **SGLD/SGNHT**: Use f32 for state, f64 for diagnostics (LLC evaluation)
- **HMC/MCLMC**: Use f64 throughout for stability
- All final LLC calculations performed in f64 for numerical accuracy

### Efficiency Metrics

We report several efficiency measures:

- **ESS/sec**: Effective samples per wall-clock second
- **ESS/FDE**: Effective samples per full-data-equivalent gradient computation
- **WNV**: Wall-clock normalized variance for fair sampler comparison

These metrics account for computational cost differences between samplers.

## References

- Hitchcock & Hoogland, *From Global to Local: A Scalable Benchmark for Local Posterior Sampling*, 2025.
- Aoyagi, *Local learning coefficients and Bayesian generalization error*, 2024.
- Watanabe, *Mathematical theory of Bayesian statistics*, CRC Press, 2018.
- Wei et al., *The implicit regularization of stochastic gradient flow for least squares*, ICML 2019.