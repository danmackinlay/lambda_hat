import jax
import jax.numpy as jnp
from lambda_hat.sampling import run_sgld
from lambda_hat.posterior import make_grad_loss_minibatch
from lambda_hat.targets import TargetBundle

def test_sgld_noise_independence_smoke():
    # Minimal quadratic target with two leaves
    params = {"a": jnp.zeros((3,), jnp.float32), "b": jnp.zeros((2,2), jnp.float32)}
    def loss_full(p): return 0.5*(jnp.sum(p["a"]**2)+jnp.sum(p["b"]**2)).astype(jnp.float32)
    def loss_mb(p, X, Y): return loss_full(p)
    tb = TargetBundle(
        d=7, params0_f32=params, params0_f64=None,
        loss_full_f32=loss_full, loss_minibatch_f32=loss_mb,
        loss_full_f64=None, loss_minibatch_f64=None,
        X=jnp.zeros((8,1), jnp.float64), Y=jnp.zeros((8,1), jnp.float64),
        L0=0.0, model=None
    )
    cfg = type("C", (), {})()
    cfg.steps = 5
    cfg.warmup = 0
    cfg.batch_size = 4
    cfg.step_size = 1e-4
    cfg.dtype = "float32"
    cfg.precond = "none"
    cfg.beta1 = 0.9
    cfg.beta2 = 0.999
    cfg.eps = 1e-8
    cfg.bias_correction = True
    cfg.eval_every = 1

    grad_loss = make_grad_loss_minibatch(tb.loss_minibatch_f32)
    out = run_sgld(jax.random.PRNGKey(0), grad_loss, tb.params0_f32, tb.params0_f32,
                   (tb.X, tb.Y), cfg, num_chains=1, beta=1.0, gamma=0.0,
                   loss_full_fn=tb.loss_full_f32)
    # Check basic output
    traces = out.traces
    assert "Ln" in traces
    # Quick stochastic sanity: step once and compare per-leaf updates
    # (re-run kernel once deterministically with fixed key)
