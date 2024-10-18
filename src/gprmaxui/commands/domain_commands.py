from __future__ import annotations


import typing
from pathlib import Path

from gprmaxui.commands.commands_parser import CommandParser, Command


@CommandParser.register("domain")
class DomainSize(Command):
    """
    A dataclass representing the size of a domain.
    The x, y, and z coordinates within the gpr model are the length, depth, and width, respectively.
    """

    x: float  # x
    y: float  # z
    z: float  # y


@CommandParser.register("dx_dy_dz")
class DomainResolution(Command):
    """
    A dataclass representing the resolution of a domain.
    Allows you to specify the discretization of space in the x , y and z directions respectively (i.e. Δ𝑥, Δ𝑦, Δ𝑧).
    """

    dx: float
    dy: float
    dz: float


@CommandParser.register("time_window")
class TimeWindow(Command):
    """
    A dataclass representing the time window of a simulation.
    Allows you to specify the total time of the simulation (i.e. 𝑇).
    """

    twt: typing.Union[float, int]



@CommandParser.register("title")
class Title(Command):
    """
    A dataclass representing the title of a simulation.
    Allows you to specify the title of the simulation.
    """

    title: str


@CommandParser.register("pml_cells")
class PMLCells(Command):
    """
    A dataclass representing the number of PML cells.
    Allows you to specify the number of PML cells.
    """

    n: int = None


@CommandParser.register("output_dir")
class OutputDir(Command):
    """
    A dataclass representing the output directory of a simulation.
    Allows you to specify the output directory of the simulation.
    """

    path: typing.Union[str, Path]
