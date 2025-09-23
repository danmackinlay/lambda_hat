import numpy as np
import pytest
from llc.convert import to_idata
from llc.analysis import llc_point_se, fig_rank_llc


def test_to_idata_and_metrics():
    """Test basic conversion and metrics computation"""
    # Fake 3 chains, ragged Ln histories
    H = [
        np.linspace(1.0, 0.8, 120),
        np.linspace(1.05, 0.85, 100),
        np.linspace(0.95, 0.82, 110),
    ]
    idata = to_idata(
        Ln_histories=H,
        theta_thin=None,
        acceptance=None,
        energy=None,
        n=1000,
        beta=0.2,
        L0=0.7,
    )
    m = llc_point_se(idata)
    assert "llc_mean" in m and np.isfinite(m["llc_mean"])
    assert "ess_bulk" in m
    assert "llc_se" in m


def test_to_idata_with_theta():
    """Test conversion with theta samples"""
    H = [np.random.randn(200) + 1.0, np.random.randn(200) + 1.0]
    theta = [
        np.random.randn(200, 10),
        np.random.randn(200, 10),
    ]  # 2 chains, 200 draws, 10 dims
    idata = to_idata(
        Ln_histories=H,
        theta_thin=theta,
        acceptance=None,
        energy=None,
        n=500,
        beta=0.1,
        L0=0.0,
        max_theta_dims=5,
    )
    assert "theta" in idata.posterior
    assert idata.posterior["theta"].shape[-1] == 5  # limited to max_theta_dims


def test_to_idata_with_hmc_extras():
    """Test conversion with HMC acceptance and energy"""
    H = [np.random.randn(100) + 1.0, np.random.randn(100) + 1.0]
    acc = [np.random.uniform(0.5, 1.0, 100), np.random.uniform(0.5, 1.0, 100)]
    energy = [np.random.randn(100), np.random.randn(100)]
    idata = to_idata(
        Ln_histories=H,
        theta_thin=None,
        acceptance=acc,
        energy=energy,
        n=500,
        beta=0.1,
        L0=0.0,
    )
    assert "acceptance_rate" in idata.sample_stats
    assert "energy" in idata.sample_stats


def test_rank_plot():
    """Test rank plot generation"""
    H = [np.random.randn(200) + 1.0, np.random.randn(200) + 1.0]
    idata = to_idata(
        Ln_histories=H,
        theta_thin=None,
        acceptance=None,
        energy=None,
        n=500,
        beta=0.1,
        L0=0.0,
    )
    fig = fig_rank_llc(idata)
    assert fig is not None


def test_empty_histories():
    """Test that empty histories raise appropriate error"""
    with pytest.raises(ValueError, match="Ln_histories is empty"):
        to_idata(
            Ln_histories=[],
            theta_thin=None,
            acceptance=None,
            energy=None,
            n=500,
            beta=0.1,
            L0=0.0,
        )


def test_ragged_alignment():
    """Test that ragged arrays are properly aligned"""
    # Different length histories
    H = [
        np.random.randn(150) + 1.0,
        np.random.randn(100) + 1.0,
        np.random.randn(120) + 1.0,
    ]
    # Different length acceptance (should be aligned to shortest)
    acc = [
        np.random.uniform(0.5, 1.0, 150),
        np.random.uniform(0.5, 1.0, 80),
        np.random.uniform(0.5, 1.0, 120),
    ]

    idata = to_idata(
        Ln_histories=H,
        theta_thin=None,
        acceptance=acc,
        energy=None,
        n=500,
        beta=0.1,
        L0=0.0,
    )

    # Should be truncated to minimum length
    assert idata.posterior["llc"].shape[1] == 80  # min of acceptance lengths
    assert idata.sample_stats["acceptance_rate"].shape[1] == 80


def test_sghmc_smoke():
    """Test SGHMC can run without errors"""
    from llc.config import TEST_CFG
    from llc.pipeline import run_one
    from dataclasses import replace

    cfg = replace(TEST_CFG, samplers=("sghmc",), save_plots=False)
    out = run_one(cfg, save_artifacts=False, skip_if_exists=False)
    assert "sghmc_llc_mean" in out.metrics
    assert np.isfinite(out.metrics["sghmc_llc_mean"])


def test_quick_run_smoke():
    """Test that a quick run completes successfully with atomic runs"""
    from llc.config import TEST_CFG
    from llc.pipeline import run_one
    from dataclasses import replace

    # Test SGLD run (atomic)
    cfg_sgld = replace(
        TEST_CFG,
        samplers=("sgld",),
        save_plots=False,
        sgld_steps=100,
        sgld_warmup=20,
    )
    out1 = run_one(cfg_sgld, save_artifacts=False, skip_if_exists=False)
    assert "sgld_llc_mean" in out1.metrics
    assert np.isfinite(out1.metrics["sgld_llc_mean"])

    # Test SGHMC run (atomic)
    cfg_sghmc = replace(
        TEST_CFG,
        samplers=("sghmc",),
        save_plots=False,
        sghmc_steps=100,
        sghmc_warmup=20,
    )
    out2 = run_one(cfg_sghmc, save_artifacts=False, skip_if_exists=False)
    assert "sghmc_llc_mean" in out2.metrics
    assert np.isfinite(out2.metrics["sghmc_llc_mean"])
