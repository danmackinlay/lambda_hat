"""Tests for MCLMC parameter validation and BlackJAX 1.2.5 API alignment."""

import pytest
from llc.validation import validate_mclmc_config, validate_config_before_dispatch


def test_validate_mclmc_config_success():
    """Test that valid MCLMC configurations pass validation."""
    valid_config = {
        "samplers": ["mclmc"],
        "mclmc_num_steps": 2000,
        "mclmc_frac_tune1": 0.1,
        "mclmc_frac_tune2": 0.1,
        "mclmc_frac_tune3": 0.1,
        "mclmc_desired_energy_var": 5e-4,
        "mclmc_trust_in_estimate": 1.0,
        "mclmc_num_effective_samples": 150.0,
        "mclmc_diagonal_preconditioning": False,
        "mclmc_integrator": "isokinetic_mclachlan",
    }
    # Should not raise
    validate_mclmc_config(valid_config)


def test_validate_mclmc_config_deprecated_tune_steps():
    """Test that deprecated mclmc_tune_steps triggers clear error."""
    config_with_deprecated = {
        "samplers": ["mclmc"],
        "mclmc_tune_steps": 2000,  # Deprecated
        "mclmc_frac_tune1": 0.1,
    }

    with pytest.raises(ValueError) as exc_info:
        validate_mclmc_config(config_with_deprecated)

    error_msg = str(exc_info.value)
    assert "deprecated MCLMC parameters" in error_msg
    assert "mclmc_tune_steps" in error_msg
    assert "mclmc_num_steps" in error_msg
    assert "Migration guide" in error_msg
    assert "blackjax-devs.github.io" in error_msg


def test_validate_mclmc_config_deprecated_num_steps_tune():
    """Test that deprecated num_steps_tune1/2/3 triggers clear error."""
    config_with_deprecated = {
        "samplers": ["mclmc"],
        "mclmc_num_steps_tune1": 500,  # Deprecated
        "mclmc_num_steps_tune2": 500,  # Deprecated
        "mclmc_num_steps": 2000,
    }

    with pytest.raises(ValueError) as exc_info:
        validate_mclmc_config(config_with_deprecated)

    error_msg = str(exc_info.value)
    assert "mclmc_num_steps_tune1" in error_msg
    assert "mclmc_num_steps_tune2" in error_msg
    assert "mclmc_frac_tune1" in error_msg


def test_validate_mclmc_config_unknown_params():
    """Test that unknown MCLMC parameters trigger clear error."""
    config_with_unknown = {
        "samplers": ["mclmc"],
        "mclmc_num_steps": 2000,
        "mclmc_unknown_param": "value",  # Unknown
        "mclmc_another_bad_param": 123,  # Unknown
    }

    with pytest.raises(ValueError) as exc_info:
        validate_mclmc_config(config_with_unknown)

    error_msg = str(exc_info.value)
    assert "Unknown MCLMC parameters" in error_msg
    assert "mclmc_unknown_param" in error_msg
    assert "mclmc_another_bad_param" in error_msg
    assert "Valid MCLMC parameters" in error_msg


def test_validate_mclmc_config_invalid_fractions():
    """Test that invalid fractions trigger errors."""
    # Test negative fraction
    config_negative = {
        "samplers": ["mclmc"],
        "mclmc_num_steps": 2000,
        "mclmc_frac_tune1": -0.1,  # Invalid
    }

    with pytest.raises(ValueError) as exc_info:
        validate_mclmc_config(config_negative)

    assert "mclmc_frac_tune1 must be a number in [0.0, 1.0]" in str(exc_info.value)

    # Test fraction > 1
    config_too_large = {
        "samplers": ["mclmc"],
        "mclmc_num_steps": 2000,
        "mclmc_frac_tune2": 1.5,  # Invalid
    }

    with pytest.raises(ValueError) as exc_info:
        validate_mclmc_config(config_too_large)

    assert "mclmc_frac_tune2 must be a number in [0.0, 1.0]" in str(exc_info.value)


def test_validate_mclmc_config_fractions_sum_too_large():
    """Test that fractions summing > 1.0 trigger error."""
    config_sum_too_large = {
        "samplers": ["mclmc"],
        "mclmc_num_steps": 2000,
        "mclmc_frac_tune1": 0.5,
        "mclmc_frac_tune2": 0.4,
        "mclmc_frac_tune3": 0.3,  # Sum = 1.2 > 1.0
    }

    with pytest.raises(ValueError) as exc_info:
        validate_mclmc_config(config_sum_too_large)

    error_msg = str(exc_info.value)
    assert "fractions sum to 1.200 > 1.0" in error_msg
    assert "frac_tune1 + frac_tune2 + frac_tune3 ≤ 1.0" in error_msg


def test_validate_mclmc_config_invalid_num_steps():
    """Test that invalid num_steps triggers error."""
    config_zero_steps = {
        "samplers": ["mclmc"],
        "mclmc_num_steps": 0,  # Invalid
    }

    with pytest.raises(ValueError) as exc_info:
        validate_mclmc_config(config_zero_steps)

    assert "mclmc_num_steps must be a positive integer" in str(exc_info.value)

    config_negative_steps = {
        "samplers": ["mclmc"],
        "mclmc_num_steps": -100,  # Invalid
    }

    with pytest.raises(ValueError) as exc_info:
        validate_mclmc_config(config_negative_steps)

    assert "mclmc_num_steps must be a positive integer" in str(exc_info.value)


def test_validate_config_before_dispatch_mclmc():
    """Test that validate_config_before_dispatch calls MCLMC validation."""
    config_with_mclmc = {
        "samplers": ["mclmc"],
        "mclmc_tune_steps": 2000,  # Should trigger validation error
    }

    with pytest.raises(ValueError) as exc_info:
        validate_config_before_dispatch(config_with_mclmc)

    assert "deprecated MCLMC parameters" in str(exc_info.value)


def test_validate_config_before_dispatch_non_mclmc():
    """Test that validation skips non-MCLMC samplers."""
    config_sgld_only = {
        "samplers": ["sgld"],
        "sgld_step_size": 1e-6,
        "mclmc_tune_steps": 2000,  # Present but ignored since MCLMC not in samplers
    }

    # Should not raise (MCLMC validation skipped)
    validate_config_before_dispatch(config_sgld_only)


def test_validate_config_before_dispatch_string_sampler():
    """Test that validation works with string sampler (not tuple)."""
    config_string_sampler = {
        "samplers": "mclmc",  # String instead of tuple
        "mclmc_tune_steps": 2000,  # Should trigger validation error
    }

    with pytest.raises(ValueError) as exc_info:
        validate_config_before_dispatch(config_string_sampler)

    assert "deprecated MCLMC parameters" in str(exc_info.value)


def test_api_alignment_integration():
    """Integration test: ensure valid config matches BlackJAX 1.2.5 API."""
    # This configuration should match what we pass to blackjax.mclmc_find_L_and_step_size
    valid_config = {
        "samplers": ["mclmc"],
        "mclmc_num_steps": 1000,
        "mclmc_frac_tune1": 0.15,
        "mclmc_frac_tune2": 0.10,
        "mclmc_frac_tune3": 0.05,  # Sum = 0.30 < 1.0 ✓
        "mclmc_desired_energy_var": 1e-3,
        "mclmc_trust_in_estimate": 0.8,
        "mclmc_num_effective_samples": 200.0,
        "mclmc_diagonal_preconditioning": True,
        "mclmc_integrator": "isokinetic_velocity_verlet",
    }

    # Should pass validation
    validate_config_before_dispatch(valid_config)

    # Verify parameter names match BlackJAX 1.2.5 API
    expected_params = {
        "num_steps", "frac_tune1", "frac_tune2", "frac_tune3",
        "desired_energy_var", "trust_in_estimate", "num_effective_samples",
        "diagonal_preconditioning"
    }

    config_mclmc_params = {
        k.replace("mclmc_", "") for k in valid_config.keys()
        if k.startswith("mclmc_") and k != "mclmc_integrator"
    }

    # All expected parameters should be present (subset check)
    assert expected_params.issubset(config_mclmc_params)