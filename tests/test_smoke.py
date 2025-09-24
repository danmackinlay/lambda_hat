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
        mclmc_num_steps=50,
        chains=2,
        save_plots=False
    )
    out = run_one(cfg, save_artifacts=False, skip_if_exists=False)
    assert "mclmc_llc_mean" in out["metrics"]

if __name__ == "__main__":
    test_smoke_quadratic_sgld()
    test_smoke_quadratic_hmc()
    test_smoke_quadratic_mclmc()
    print("All smoke tests passed!")