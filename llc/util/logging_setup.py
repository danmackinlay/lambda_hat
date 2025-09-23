import logging

# Keep this list short; WARNING is enough to kill "findfont" spam.
_NOISY = (
    "matplotlib",               # includes matplotlib.font_manager
    "PIL",
    "fontTools",
    "urllib3",
    "submitit",
)

def setup_logging(verbose_count: int, quiet_count: int, *, debug_thirdparty: bool = False) -> None:
    """Root at INFO (default), DEBUG with -v, ERROR with -q.
    If DEBUG and not debug_thirdparty, clamp common noisy libs to WARNING.
    """
    if quiet_count >= 1:
        root = logging.ERROR
    elif verbose_count >= 1:
        root = logging.DEBUG
    else:
        root = logging.INFO

    logging.basicConfig(level=root, format="%(levelname)s: %(message)s")

    if root <= logging.DEBUG and not debug_thirdparty:
        for name in _NOISY:
            logging.getLogger(name).setLevel(logging.WARNING)