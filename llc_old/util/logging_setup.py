import logging

_NOISY = (
    # common chatty libs
    "jax", "jaxlib",
    "matplotlib", "PIL", "fontTools",
    "urllib3", "submitit",
)

def setup_logging(verbose_count: int, quiet_count: int, *, debug_thirdparty: bool = False) -> None:
    """
    Root stays at INFO (or ERROR with -q).
    Our package logger 'llc' gets DEBUG when -v is used, via its own handler.
    Third-party libs stay at WARNING unless --debug-thirdparty is set.
    """
    # Clear any existing llc handlers first
    llc_logger = logging.getLogger("llc")
    llc_logger.handlers.clear()

    # 1) Root handler (console) - DEBUG if debug_thirdparty + verbose, otherwise INFO/ERROR
    if debug_thirdparty and verbose_count >= 1:
        root_level = logging.DEBUG
    elif quiet_count >= 1:
        root_level = logging.ERROR
    else:
        root_level = logging.INFO

    logging.basicConfig(level=root_level, format="%(levelname)s: %(message)s", force=True)

    # 2) Third-party noise clamped unless explicitly requested
    if not debug_thirdparty:
        # In quiet mode, set noisy libs to ERROR; otherwise WARNING
        noise_level = logging.ERROR if quiet_count >= 1 else logging.WARNING
        for name in _NOISY:
            logging.getLogger(name).setLevel(noise_level)

    # 3) Our namespace logger
    if verbose_count >= 1 and not debug_thirdparty:
        # Special case: we want LLC DEBUG but not third-party DEBUG
        llc_logger.setLevel(logging.DEBUG)
        # Add a dedicated handler so our DEBUG bypasses the root INFO filter
        h = logging.StreamHandler()
        h.setLevel(logging.DEBUG)
        h.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
        llc_logger.addHandler(h)
        # Avoid double-printing via root
        llc_logger.propagate = False
    else:
        # Either not verbose, or debug_thirdparty=True (use root DEBUG)
        # No special handler; inherit root INFO/ERROR/DEBUG behavior
        llc_logger.setLevel(logging.NOTSET)
        llc_logger.propagate = True