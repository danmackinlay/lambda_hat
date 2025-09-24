from llc.config import Config
from llc.run import run_one

def test_smoke_quadratic_sgld():
    cfg = Config(
        target="quadratic",
        quad_dim=16,
        n_data=200,
        samplers=("sgld",),
        sgld_steps=50,
        sgld_warmup=10,
        chains=2,
        save_plots=False
    )
    out = run_one(cfg, save_artifacts=False, skip_if_exists=False)
    assert "sgld_llc_mean" in out["metrics"]
    assert "family_id" in out["metrics"]

def test_smoke_quadratic_hmc():
    cfg = Config(
        target="quadratic",
        quad_dim=8,
        n_data=100,
        samplers=("hmc",),
        hmc_draws=20,
        hmc_warmup=10,
        chains=2,
        save_plots=False
    )
    out = run_one(cfg, save_artifacts=False, skip_if_exists=False)
    assert "hmc_llc_mean" in out["metrics"]

def test_smoke_quadratic_mclmc():
    cfg = Config(
        target="quadratic",
        quad_dim=8,
        n_data=100,
        samplers=("mclmc",),
        mclmc_draws=20,
        mclmc_num_steps=200,  # Increase to help tuner
        mclmc_frac_tune1=0.2,  # Increase tuning fractions
        mclmc_frac_tune2=0.2,
        mclmc_frac_tune3=0.2,
        chains=2,
        save_plots=False
    )
    out = run_one(cfg, save_artifacts=False, skip_if_exists=False)
    assert "mclmc_llc_mean" in out["metrics"]

def test_smoke_quadratic_sgnht():
    cfg = Config(
        target="quadratic",
        quad_dim=8,
        n_data=200,
        samplers=("sgnht",),
        sgnht_steps=80,
        sgnht_warmup=10,
        chains=2,
        save_plots=False
    )
    out = run_one(cfg, save_artifacts=False, skip_if_exists=False)
    assert "sgnht_llc_mean" in out["metrics"]

def test_smoke_dln_sgld():
    cfg = Config(
        target="dln",
        in_dim=16,
        out_dim=4,
        n_data=256,
        dln_layers_min=2,
        dln_layers_max=2,
        dln_h_min=8,
        dln_h_max=8,
        samplers=("sgld",),
        sgld_steps=60,
        sgld_warmup=10,
        chains=2,
        save_plots=False
    )
    out = run_one(cfg, save_artifacts=False, skip_if_exists=False)
    assert "sgld_llc_mean" in out["metrics"]

if __name__ == "__main__":
    test_smoke_quadratic_sgld()
    test_smoke_quadratic_hmc()
    test_smoke_quadratic_mclmc()
    test_smoke_quadratic_sgnht()
    test_smoke_dln_sgld()
    print("All smoke tests passed!")