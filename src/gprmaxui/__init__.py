import logging
from rich.logging import RichHandler
from .geometry_commands import *
from .domain_commands import *
from .gprmax_model import *


FORMAT = "%(message)s"
logging.basicConfig(
    level=logging.INFO,
    format=FORMAT,
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=False)],
)

__version__ = "0.1.0"
__name__ = "gprmaxui"
