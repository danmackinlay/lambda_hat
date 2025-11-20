# lambda_hat/commands/debug_cmd.py
"""Debug commands for testing Parsl + heavy library interactions."""


def test_import_diagnostics():
    """Test if heavy imports work in current environment.

    Returns:
        str: Success message with library versions
    """
    import arviz
    import matplotlib

    return f"Success: matplotlib={matplotlib.__version__}, arviz={arviz.__version__}"


def test_import_analysis_only():
    """Test if just importing analysis module works.

    Returns:
        str: Success message
    """
    from lambda_hat import analysis

    return f"Success: analysis module loaded from {analysis.__file__}"


def test_basic_imports():
    """Test basic imports that should always work.

    Returns:
        str: Success message
    """
    import jax
    import numpy

    return f"Success: jax={jax.__version__}, numpy={numpy.__version__}"
