# Methodology: LLC Estimation and SGLD

This document details the mathematical formulation of the Local Learning Coefficient (LLC) estimator $\hat{\lambda}$ and the implementation details of the Stochastic Gradient Langevin Dynamics (SGLD) algorithms used in this project, following the approach in Hitchcock and Hoogland.

## Local Learning Coefficient (LLC) Estimator ($\hat\lambda$)

In Singular Learning Theory (SLT), the LLC $\lambda(w_0)$ characterizes the volume scaling rate of the loss landscape near a minimum $w_0$. We estimate this quantity using the estimator $\hat\lambda(w_0)$.

### Tempered Local Posterior

We sample from the tempered local posterior distribution $\pi(w)$:

$$
\pi(w) \propto \exp\left(-\frac{\gamma}{2} \|w-w_0\|^2 - n\beta L_{n}(w)\right)
$$

Where:
- $n$ is the dataset size.
- $L_n(w)$ is the empirical negative log-likelihood (the loss function).
- $\beta$ is the inverse temperature.
- $w_0$ is the ERM solution (the center of localization).
- $\gamma$ controls the strength of the Gaussian localization prior.

### The Estimator $\hat{\lambda}$

The LLC estimator is defined as (Hitchcock and Hoogland, Eq 3.4; Lau et al., 2024):

$$
\hat{\lambda}(w_0) = n\beta(E_{w}^{\beta}[L_{n}(w)] - L_{n}(w_0))
$$

In practice, we approximate the expectation $E_{w}^{\beta}[L_{n}(w)]$ by the average loss $\overline{L}$ of the samples drawn from the posterior $\pi(w)$, and denote $L_n(w_0)$ as $L_0$.

$$
\hat{\lambda} \approx n\beta(\overline{L} - L_0)
$$

This calculation is implemented in `lambda_hat/analysis.py`.

## Stochastic Gradient Langevin Dynamics (SGLD) Algorithms

We implement SGLD and its preconditioned variants (pSGLD) using stochastic gradients $g_t = \hat\nabla L_n(w_t)$.

The general update rule for pSGLD at time $t$ is (element-wise):

$$
\Delta w_{t}[i] = -\frac{\epsilon_{t}[i]}{2} \left(\gamma(w_{t}[i]-w_{0}[i])+\tilde{\beta} \cdot \text{Drift}_{t}[i]\right) + \sqrt{\epsilon_{t}[i]}\eta_{t}[i]
$$

Where $\tilde{\beta} = n\beta$, $\eta_{t} \sim \mathcal{N}(0, I)$, and $\epsilon_{t}[i]$ is the adaptive step size.

### Implementation Compliance

Crucially, following the methodology in Hitchcock and Hoogland (Appendix D.3), the adaptive statistics (moments $m_t, v_t$) are updated using **only the gradient of the loss** $g_t$, excluding the localization term.

#### Vanilla SGLD (Algorithm 1)

- $\text{Drift}_{t} = g_t$
- $\epsilon_{t}[i] = \epsilon$ (fixed scalar step size)

#### RMSPropSGLD (Algorithm 3)

RMSPropSGLD adapts the step size using the second moment estimate $v_t$.

1. Update second moment: $v_{t} = b_2 v_{t-1} + (1-b_2) g_t^2$. (Note: $v_{-1}$ initialized to 1).
2. Bias correction: $\hat{v}_{t} = v_{t} / (1-b_2^t)$.
3. Calculate adaptive step size: $\epsilon_{t}[i] = \epsilon / (\sqrt{\hat{v}_{t}[i]} + a)$ (where $a$ is `eps`).
4. $\text{Drift}_{t} = g_t$.

#### AdamSGLD (Algorithm 2)

AdamSGLD adapts both the step size and the drift term using first ($m_t$) and second ($v_t$) moment estimates.

1. Update first moment: $m_{t} = b_1 m_{t-1} + (1-b_1) g_t$. (Note: $m_{-1}$ initialized to 0).
2. Update second moment: $v_{t}$. (Note: $v_{-1}$ initialized to 1).
3. Bias correction: $\hat{m}_{t} = m_{t} / (1-b_1^t)$, $\hat{v}_{t}$.
4. Calculate adaptive step size: $\epsilon_{t}[i]$.
5. $\text{Drift}_{t} = \hat{m}_t$.