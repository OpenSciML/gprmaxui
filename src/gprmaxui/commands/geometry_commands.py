from __future__ import annotations

import math
import typing

from pydantic import Field, BaseModel

from gprmaxui.commands.commands_parser import CommandParser, Command
from gprmaxui.commands.domain_commands import DomainSize


class DomainPoint(BaseModel):
    x: float
    y: float
    z: float

    def distance(self, other: DomainPoint):
        """
        Calculate the distance between two points in 3D space.
        :param other:
        :return:
        """
        return math.sqrt(
            (self.x - other.x) ** 2 + (self.y - other.y) ** 2 + (self.z - other.z) ** 2
        )


@CommandParser.register("material")
class Material(Command):
    """
    A dataclass representing the material of a simulation.
    Allows you to specify the material of the simulation.
    """

    permittivity: float = 0  # ε (permittivity)
    conductivity: float = 0  # σ (conductivity)
    permeability: float = 0  # μ (permeability)
    magconductivity: float = 0  # σm (magnetic conductivity)
    id: str = None  # name of the material
    color: str = Field(None, exclude=True)  # color of the material


@CommandParser.register("box")
class DomainBox(Command):
    """
    A dataclass representing the box of a simulation command.
    """

    x_min: float
    y_min: float
    z_min: float
    x_max: float
    y_max: float
    z_max: float
    material: str = None
    dielectric_smoothing: typing.Literal["y", "n"] = "n"

    def area(self):
        """
        Calculate the area of the box.
        :return:
        """
        return (self.x_max - self.x_min) * (self.y_max - self.y_min)

    def volume(self):
        """
        Calculate the volume of the box.
        :return:
        """
        return (
            (self.x_max - self.x_min)
            * (self.y_max - self.y_min)
            * (self.z_max - self.z_min)
        )

    def within(self, point: DomainPoint):
        return (
            self.x_min <= point.x <= self.x_max
            and self.y_min <= point.y <= self.y_max
            and self.z_min <= point.z <= self.z_max
        )

    def center(self):
        return DomainPoint(
            x=(self.x_min + self.x_max) / 2,
            y=(self.y_min + self.y_max) / 2,
            z=(self.z_min + self.z_max) / 2,
        )

    @classmethod
    def from_size(cls, pos: DomainPoint, sz: DomainSize, material: str):
        x_min = pos.x - sz.x / 2
        x_max = pos.x + sz.x / 2

        y_min = pos.y - sz.y / 2
        y_max = pos.y + sz.y / 2

        z_min = pos.z - sz.z / 2
        z_max = pos.z + sz.z / 2
        return cls(
            x_min=x_min,
            x_max=x_max,
            y_min=y_min,
            y_max=y_max,
            z_min=z_min,
            z_max=z_max,
            material=material,
        )


@CommandParser.register("sphere")
class DomainSphere(Command):
    """
    A dataclass representing the sphere of a simulation command.
    """

    cx: float
    cy: float
    cz: float
    radius: float
    material: str = None
    dielectric_smoothing: typing.Literal["y", "n"] = "n"

    def area(self):
        """
        Calculate the area of the sphere.
        :return:
        """
        return 4 * 3.141592653589793 * self.radius**2

    def volume(self):
        """
        Calculate the volume of the sphere.
        :return:
        """
        return 4 / 3 * 3.141592653589793 * self.radius**3


@CommandParser.register("cylinder")
class DomainCylinder(Command):
    """
    A dataclass representing the cylinder of a simulation command.
    """

    cx_min: float
    cy_min: float
    cz_min: float
    cx_max: float
    cy_max: float
    cz_max: float
    radius: float
    material: str = None
    dielectric_smoothing: typing.Literal["y", "n"] = "n"

    def area(self):
        """
        Calculate the area of the cylinder.
        :return:
        """
        return 2 * 3.141592653589793 * self.radius * (self.cz_max - self.cz_min)

    def volume(self):
        """
        Calculate the volume of the cylinder.
        :return:
        """
        return 3.141592653589793 * self.radius**2 * (self.cz_max - self.cz_min)


@CommandParser.register("geometry_view")
class GeometryView(Command):
    """
    A dataclass representing the geometry view of a simulation command.
    """

    x_min: float
    y_min: float
    z_min: float
    x_max: float
    y_max: float
    z_max: float
    dx: float
    dy: float
    dz: float
    filename: str = None
    resolution: typing.Literal["n", "f"] = "n"


@CommandParser.register("snapshot")
class SnapshotView(Command):
    """
    A dataclass representing the snapshot of a simulation.
    """

    x_min: float
    y_min: float
    z_min: float
    x_max: float
    y_max: float
    z_max: float
    dx: float
    dy: float
    dz: float
    t: typing.Union[float, int]
    filename: str = None