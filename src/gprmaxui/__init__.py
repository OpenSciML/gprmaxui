import logging
from rich.logging import RichHandler

# Define the logger for your library
logger = logging.getLogger(__name__)  # This becomes 'gprmaxui' when __init__.py is loaded as the package

# Optional: allow external code to configure level, default to WARNING
logger.setLevel(logging.WARNING)

# Prevent log propagation to the root logger to avoid duplicated output
logger.propagate = False

# Only add handler if not already added (avoid duplicate handlers in interactive sessions)
if not logger.handlers:
    handler = RichHandler(rich_tracebacks=False)
    formatter = logging.Formatter("%(message)s", datefmt="[%X]")
    handler.setFormatter(formatter)
    logger.addHandler(handler)

# Export metadata
__version__ = "0.1.0"
__all__ = ["GprMaxModel"]  # Import your public API symbols
from .gprmax_model import GprMaxModel