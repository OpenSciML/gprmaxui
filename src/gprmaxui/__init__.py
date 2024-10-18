from rich.logging import RichHandler
import logging

from .gprmax_model import GprMaxModel

FORMAT = "%(message)s"
logging.basicConfig(
    level=logging.INFO,
    format=FORMAT,
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=False)],
)

__version__ = "0.1.0"
__name__ = "gprmaxui"
