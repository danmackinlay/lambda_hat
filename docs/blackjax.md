# BlackJAX API Notes

This repo is pinned to BlackJAX 1.2.5. Here are the key API details to prevent confusion across docs vs release.

## SGLD

* **Public API:** `sgld = blackjax.sgld(grad_fn)`
* **Step signature:** `new_position = sgld.step(rng_key, position, minibatch, step_size)`
* **Source:** [sgld.py (1.2.5)](https://github.com/blackjax-devs/blackjax/blob/1.2.5/blackjax/sgmcmc/sgld.py#L38-L47) (see `step_fn` → returns `kernel(...)` → returns `new_position`)

## HMC

* **Usage:** `blackjax.hmc` with `blackjax.window_adaptation`
* **Info fields:** `HMCInfo` includes `acceptance_rate` (flat attribute)
* **Source:** [hmc.py (1.2.5)](https://github.com/blackjax-devs/blackjax/blob/1.2.5/blackjax/mcmc/hmc.py#L330-L334)

## MCLMC

* **Usage:** See [Sampling Book MCLMC example](https://blackjax-devs.github.io/sampling-book/algorithms/mclmc.html)
* **Tuning API (BlackJAX 1.2.5):** Use fractional parameters with `blackjax.mclmc_find_L_and_step_size`:
  ```python
  L, step_size, info = blackjax.mclmc_find_L_and_step_size(
      mclmc_kernel=None,  # computed internally
      num_steps=total_adaptation_steps,
      state=initial_position,
      rng_key=rng_key,
      logdensity_fn=logdensity_fn,
      frac_tune1=0.1,  # fraction for phase 1
      frac_tune2=0.1,  # fraction for phase 2
      frac_tune3=0.1,  # fraction for phase 3
      desired_energy_var=5e-4,
      trust_in_estimate=1.0,
      num_effective_samples=150.0,
      diagonal_preconditioning=False,
      integrator=integrator_fn,
  )
  ```
* **Key constraint:** `frac_tune1 + frac_tune2 + frac_tune3 ≤ 1.0`
* **Deprecated:** `num_steps_tune1/2/3` parameters - use fractional API instead
* **Reference:** [MCLMC adaptation docs (1.2.5)](https://blackjax-devs.github.io/blackjax/autoapi/blackjax/adaptation/mclmc_adaptation/)
* **Integrators:** Available in [integrators module (1.2.5)](https://github.com/blackjax-devs/blackjax/tree/1.2.5/blackjax/mcmc/integrators) (e.g. `isokinetic_mclachlan`)
* **Info fields:** `MCLMCInfo` has `energy_change` field (see [mclmc.py (1.2.5)](https://github.com/blackjax-devs/blackjax/blob/1.2.5/blackjax/mcmc/mclmc.py))

## ⚠️ Docs Drift Warning

The online BlackJAX docs default to `main`. They may show `acceptance_probability` for HMC or a different SGLD step signature. Always cross-check the [1.2.5 tag source](https://github.com/blackjax-devs/blackjax/tree/1.2.5) when in doubt.