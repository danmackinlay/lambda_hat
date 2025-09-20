"""Test backend dispatcher functionality."""

from llc.util.backend_dispatch import BackendOptions, prepare_payloads
from llc.config import TEST_CFG


def test_backend_options_creation():
    """Test that BackendOptions can be created with all parameters."""
    opts = BackendOptions(
        backend="local",
        gpu_mode="off",
        gpu_types="H100,A100",
        local_workers=2,
        slurm_partition="gpu",
        timeout_min=120,
        cpus=8,
        mem_gb=32,
        slurm_signal_delay_s=60,
        modal_autoscaler_cap=4,
        modal_chunk_size=8,
        modal_auto_extract=False,
    )

    assert opts.backend == "local"
    assert opts.gpu_mode == "off"
    assert opts.gpu_types == "H100,A100"
    assert opts.local_workers == 2
    assert opts.modal_autoscaler_cap == 4


def test_prepare_payloads():
    """Test that prepare_payloads creates proper payload dicts."""
    cfgs = [TEST_CFG]
    payloads = prepare_payloads(
        cfgs, save_artifacts=True, skip_if_exists=False, gpu_mode="vectorized"
    )

    assert len(payloads) == 1
    payload = payloads[0]

    # Check required fields are present
    assert "save_artifacts" in payload
    assert "skip_if_exists" in payload
    assert "config_schema" in payload
    assert "gpu_mode" in payload

    assert payload["save_artifacts"] is True
    assert payload["skip_if_exists"] is False
    assert payload["gpu_mode"] == "vectorized"


def test_backend_options_defaults():
    """Test BackendOptions with minimal parameters."""
    opts = BackendOptions(backend="modal", gpu_mode="sequential")

    assert opts.backend == "modal"
    assert opts.gpu_mode == "sequential"
    assert opts.gpu_types == ""
    assert opts.local_workers == 0
    assert opts.modal_autoscaler_cap == 8
    assert opts.modal_chunk_size == 16
    assert opts.modal_auto_extract is True