import copy
import os
from unittest.mock import patch
from lambda_hat.cache import run_id, run_family_id


def _with(d, **kw):
    """Helper to create config copies with updates"""
    x = copy.deepcopy(d)
    x.update(kw)
    return x


def test_run_id_ignores_irrelevant_hmc_knobs_for_sgld():
    """Test that HMC-specific parameters don't affect SGLD run IDs"""
    cfg_base = {
        "samplers": ("sgld",),
        "seed": 42,
        "n_data": 1000,
        "sgld_step_size": 1e-4,
        "sgld_steps": 1000,
    }
    cfg1 = _with(cfg_base, hmc_draws=1000)
    cfg2 = _with(cfg_base, hmc_draws=2000)
    assert run_id(cfg1) == run_id(cfg2)


def test_run_id_ignores_irrelevant_sgld_knobs_for_hmc():
    """Test that SGLD-specific parameters don't affect HMC run IDs"""
    cfg_base = {
        "samplers": ("hmc",),
        "seed": 42,
        "n_data": 1000,
        "hmc_draws": 1000,
        "hmc_warmup": 500,
    }
    cfg1 = _with(cfg_base, sgld_step_size=1e-6)
    cfg2 = _with(cfg_base, sgld_step_size=1e-5)
    assert run_id(cfg1) == run_id(cfg2)


def test_run_id_ignores_irrelevant_mclmc_knobs_for_sgld():
    """Test that MCLMC-specific parameters don't affect SGLD run IDs"""
    cfg_base = {
        "samplers": ("sgld",),
        "seed": 42,
        "n_data": 1000,
        "sgld_step_size": 1e-4,
        "sgld_steps": 1000,
    }
    cfg1 = _with(cfg_base, mclmc_draws=1000, mclmc_num_steps=100)
    cfg2 = _with(cfg_base, mclmc_draws=2000, mclmc_num_steps=200)
    assert run_id(cfg1) == run_id(cfg2)


def test_run_id_sensitive_to_relevant_knobs():
    """Test that relevant sampler parameters DO affect run IDs"""
    cfg_base = {
        "samplers": ("sgld",),
        "seed": 42,
        "n_data": 1000,
        "sgld_steps": 1000,
    }
    cfg1 = _with(cfg_base, sgld_step_size=1e-4)
    cfg2 = _with(cfg_base, sgld_step_size=1e-5)
    # These should be different because sgld_step_size affects SGLD
    assert run_id(cfg1) != run_id(cfg2)


def test_family_id_ignores_sampler_and_code_version():
    """Test that family IDs ignore sampler choice and code version"""
    cfg_base = {
        "seed": 42,
        "n_data": 1000,
        "model": "quadratic",
        "data": "synthetic",
    }

    cfg_hmc = _with(cfg_base, samplers=("hmc",), hmc_draws=1000)
    cfg_sgld = _with(cfg_base, samplers=("sgld",), sgld_step_size=1e-6)

    # Mock different code versions
    with patch.dict(os.environ, {"LAMBDA_HAT_CODE_VERSION": "v1.0"}):
        fam1 = run_family_id(cfg_hmc)

    with patch.dict(os.environ, {"LAMBDA_HAT_CODE_VERSION": "v2.0"}):
        fam2 = run_family_id(cfg_sgld)

    assert fam1 == fam2


def test_family_id_ignores_all_sampler_specific_fields():
    """Test that family IDs ignore all sampler-specific parameters"""
    cfg_base = {
        "seed": 42,
        "n_data": 1000,
        "model": "quadratic",
        "data": "synthetic",
        "samplers": ("sgld",),
    }

    # Add various sampler-specific fields
    cfg1 = _with(cfg_base,
                sgld_step_size=1e-4, sgld_steps=1000,
                hmc_draws=500, hmc_warmup=100,
                mclmc_draws=1000, mclmc_num_steps=50,
                sghmc_step_size=1e-3, sghmc_temperature=1.0)

    cfg2 = _with(cfg_base,
                sgld_step_size=1e-5, sgld_steps=2000,
                hmc_draws=1000, hmc_warmup=200,
                mclmc_draws=2000, mclmc_num_steps=100,
                sghmc_step_size=1e-4, sghmc_temperature=2.0)

    # Family IDs should be the same despite different sampler parameters
    assert run_family_id(cfg1) == run_family_id(cfg2)


def test_family_id_sensitive_to_problem_parameters():
    """Test that family IDs are sensitive to problem parameters"""
    cfg_base = {
        "samplers": ("sgld",),
        "n_data": 1000,
        "model": "quadratic",
        "data": "synthetic",
    }

    cfg1 = _with(cfg_base, seed=42)
    cfg2 = _with(cfg_base, seed=43)

    # Different seeds should give different family IDs
    assert run_family_id(cfg1) != run_family_id(cfg2)

    cfg3 = _with(cfg_base, seed=42, n_data=2000)
    # Different data size should give different family IDs
    assert run_family_id(cfg1) != run_family_id(cfg3)


def test_multi_sampler_fallback():
    """Test that multi-sampler configs fall back gracefully in field stripping"""
    cfg_multi = {
        "samplers": ("sgld", "hmc"),  # Multi-sampler - should be unchanged
        "seed": 42,
        "sgld_step_size": 1e-4,
        "hmc_draws": 1000,
    }

    cfg_sgld = {
        "samplers": ("sgld",),  # Single sampler - should strip hmc fields
        "seed": 42,
        "sgld_step_size": 1e-4,
        "hmc_draws": 1000,
    }

    # Multi-sampler config should have different run_id than single-sampler
    # because it doesn't strip irrelevant fields
    assert run_id(cfg_multi) != run_id(cfg_sgld)


def test_unknown_sampler_fallback():
    """Test that unknown samplers don't cause errors"""
    cfg = {
        "samplers": ("unknown_sampler",),
        "seed": 42,
        "n_data": 1000,
        "sgld_step_size": 1e-4,
        "hmc_draws": 1000,
    }

    # Should not raise an error, just return a valid hash
    rid = run_id(cfg)
    fid = run_family_id(cfg)
    assert len(rid) == 12
    assert len(fid) == 12