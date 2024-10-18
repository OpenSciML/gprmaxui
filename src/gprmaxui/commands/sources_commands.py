from __future__ import annotations

import typing

from gprmaxui.commands.commands_parser import CommandParser, Command, StackCommand


@CommandParser.register("waveform")
class Waveform(Command):
    """
    A dataclass representing the waveform of a simulation.
    Allows you to specify the waveform of the simulation.
    """

    wave_family: typing.Literal[
        "ricker",
        "gaussian",
        "gaussiandot",
        "gaussiandotnorm",
        "gaussiandotdot",
        "gaussiandotdotnorm",
        "gaussianprime",
        "gaussiandoubleprime",
        "sine",
        "contsine",
    ]
    amplitude: float
    frequency: float
    id: str = None


@CommandParser.register("hertzian_dipole")
class HertzianDipole(Command):
    """
    A dataclass representing the hertzian dipole of a simulation.

    Allows you to specify a current density term at an electric field location
    - the simplest excitation, often referred to as an additive or soft source.
    """

    polarization: typing.Literal["x", "y", "z"] = "z"
    x: float
    y: float
    z: float
    waveform: str = None


@CommandParser.register("magnetic_dipole")
class MagneticDipole(Command):
    """
    A dataclass representing the magnetic dipole of a simulation.
    This will simulate an infinitesimal magnetic dipole.
    This is often referred to as an additive or soft source.
    """

    polarization: typing.Literal["x", "y", "z"] = "x"
    x: float
    y: float
    z: float
    waveform: str = None


@CommandParser.register("voltage_source")
class VoltageSource(Command):
    """
    A dataclass representing the voltage source of a simulation.

    Allows you to introduce a voltage source at an electric field location.
    It can be a hard source if it’s resistance is zero, i.e.
    the time variation of the specified electric field component is prescribed,
    or if it’s resistance is non-zero it behaves as a resistive voltage source.
    It is useful for exciting antennas when the physical properties of the
    antenna are included in the model.
    """

    polarization: typing.Literal["x", "y", "z"] = "x"
    x: float
    y: float
    z: float
    resistance: float
    waveform: str = None


@CommandParser.register("rx")
class Rx(Command):
    """
    A dataclass representing the receiver of a simulation.

    Allows you to specify a receiver at an electric field location.
    These are locations where the values of the electric and magnetic field
    components over the number of iterations of the model will be saved to file.
    """

    x: float
    y: float
    z: float


class Tx(StackCommand):
    waveform: Waveform
    source: typing.Union[HertzianDipole, MagneticDipole, VoltageSource]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        unique_id = f"{self.waveform.wave_family}_{id(self.waveform)}"

        self.waveform.id = unique_id
        self.source.waveform = unique_id


@CommandParser.register("src_steps")
class SrcSteps(Command):
    """
    A dataclass representing the source steps of a simulation.

    Allows you to specify the number of steps in the source waveform.
    """

    dx: float
    dy: float
    dz: float


@CommandParser.register("rx_steps")
class RxSteps(Command):
    """
    A dataclass representing the receiver steps of a simulation.
    """

    dx: float
    dy: float
    dz: float


class TxRxPair(StackCommand):
    """
    A dataclass representing a transmitter-receiver pair of a simulation.
    """

    tx: Tx
    rx: Rx
    src_steps: SrcSteps
    rx_steps: RxSteps
