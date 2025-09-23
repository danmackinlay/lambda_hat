import logging
import os

NOISY_LOGGERS = [
    # Matplotlib noise
    "matplotlib", "matplotlib.font_manager", "matplotlib.pyplot",
    # Font stack
    "fontTools", "PIL",
    # Common chatty libs (tune as desired)
    "urllib3",
]

class DropMatplotlibFindfont(logging.Filter):
    """Filter out the very noisy 'findfont' DEBUG lines from matplotlib."""
    def filter(self, record: logging.LogRecord) -> bool:
        if record.name.startswith("matplotlib"):
            # Record msg can be a format string; make it safe
            try:
                msg = record.getMessage()
            except Exception:
                msg = str(record.msg)
            if "findfont" in msg:
                return False
        return True

def setup_logging(verbose_count: int, quiet_count: int) -> None:
    """Configure root logging level and tame third-party noise.
    - Root level: INFO (default), DEBUG with -v, ERROR with -q.
    - Third-party libs: kept at INFO unless LLC_DEBUG_THIRDPARTY=1.
    - Drop matplotlib 'findfont' spam unconditionally.
    """
    # Root level mapping
    if quiet_count >= 1:
        root_level = logging.ERROR
    elif verbose_count >= 1:
        root_level = logging.DEBUG
    else:
        root_level = logging.INFO

    logging.basicConfig(level=root_level, format="%(levelname)s: %(message)s")

    # Always add filter to root to drop findfont spam
    root_logger = logging.getLogger()
    root_logger.addFilter(DropMatplotlibFindfont())

    # Optionally let users see third-party DEBUG via env
    allow_thirdparty_debug = os.getenv("LLC_DEBUG_THIRDPARTY", "") == "1"
    if not allow_thirdparty_debug:
        for name in NOISY_LOGGERS:
            logging.getLogger(name).setLevel(max(logging.getLogger(name).level, logging.INFO))

    # Keep submitit aligned with us
    logging.getLogger("submitit").setLevel(root_level)