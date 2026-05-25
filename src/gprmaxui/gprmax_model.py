from __future__ import annotations

import math
import json
import logging
import os
import sys
import tempfile
import typing
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from io import StringIO
from pathlib import Path
from typing import Dict, List, Tuple, Union, Optional

import cv2
import h5py
import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
from PIL import Image
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from tqdm import tqdm

from gprmaxui.commands import *
from gprmaxui.plotter import PlotterDialog
from gprmaxui.utils import (
    get_output_data,
    rmdir,
    merge_model_files,
    is_integer_num,
    figure2image,
    round_value,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class VideoFrameTask:
    frame_index: int
    trace_idx: int
    iteration_idx: int
    output_folder: str
    snapshot_file: str
    geometry_file: str
    data_file: str
    frame_path: str
    dt: float
    dx: float
    tx_x: float
    tx_y: float
    tx_z: float
    rx_x: float
    rx_y: float
    rx_z: float
    rx_component: str
    cmap: str
    figsize: Tuple[float, float]


def _capture_stdout(callback) -> str:
    original_stdout = sys.stdout
    string_out = StringIO()
    try:
        sys.stdout = string_out
        callback()
    finally:
        sys.stdout = original_stdout
    return string_out.getvalue()


def _physical_cpu_count() -> int:
    try:
        import psutil

        count = psutil.cpu_count(logical=False)
    except Exception:
        count = os.cpu_count()
    return count or 1


def _gpu_count(gpu) -> int:
    if gpu is None:
        return 0
    if isinstance(gpu, (list, tuple, set)):
        return len(gpu)
    return 1


def _resolve_mpi_tasks(mpi, gpu, n_traces: int):
    if mpi == "auto":
        gpu_tasks = _gpu_count(gpu)
        worker_count = gpu_tasks if gpu_tasks > 1 else _physical_cpu_count()
        return max(1, min(n_traces + 1, worker_count + 1))
    if mpi is True:
        return max(1, min(n_traces + 1, _physical_cpu_count() + 1))
    return mpi


def _validate_positive_int(value, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _resolve_frame_workers(workers, task_count: int) -> int:
    if task_count < 1:
        return 1
    if workers == "auto":
        return max(1, min(task_count, max(1, (os.cpu_count() or 1) - 1), 4))
    if workers is None:
        return 1
    return _validate_positive_int(workers, "workers")


def _video_frame_indices(
    n_traces: int, n_iterations: int, frame_step: int
) -> List[Tuple[int, int, int]]:
    frame_index = 0
    frame_indices = []
    for trace_idx in range(n_traces):
        for iteration_idx in range(0, n_iterations, frame_step):
            frame_indices.append((frame_index, trace_idx, iteration_idx))
            frame_index += 1
    return frame_indices


def _render_video_frame(task: VideoFrameTask) -> Tuple[int, str]:
    outputdata = np.load(task.data_file, mmap_mode="r")

    fig, axes = plt.subplots(2, 1, figsize=task.figsize)
    plotter = pv.Plotter(off_screen=True)
    try:
        plotter.set_background("white")
        plotter.camera_position = "xy"
        plotter.add_axes()

        snapshot_grid = pv.read(task.snapshot_file)
        plotter.add_mesh(
            snapshot_grid,
            cmap=task.cmap,
            scalars="H-field",
            show_edges=False,
            show_scalar_bar=False,
        )

        geometry_grid = pv.read(task.geometry_file)
        plotter.add_mesh(
            geometry_grid, show_edges=False, show_scalar_bar=False, opacity=0.5
        )

        plotter.add_mesh(
            pv.Cube(
                center=(
                    task.tx_x + (task.trace_idx * task.dx),
                    task.tx_y,
                    task.tx_z,
                ),
                x_length=task.dx * 2,
                y_length=task.dx * 2,
                z_length=task.dx * 2,
            ),
            color="red",
        )

        plotter.add_mesh(
            pv.Cube(
                center=(
                    task.rx_x + (task.trace_idx * task.dx),
                    task.tx_y,
                    task.tx_z,
                ),
                x_length=task.dx * 2,
                y_length=task.dx * 2,
                z_length=task.dx * 2,
            ),
            color="blue",
        )
        plotter.camera.tight()
        snapshot_capture = Image.fromarray(plotter.screenshot(return_img=True))

        ax = axes[0]
        ax.imshow(snapshot_capture, aspect="auto")
        ax.set_xlabel("Trace")
        ax.set_ylabel("Time")
        ax.set_title(
            f"{task.rx_component} Snapshot at trace {task.trace_idx + 1} "
            f"and iteration {task.iteration_idx + 1}"
        )

        new_arr = np.full_like(outputdata, fill_value=np.nan)
        new_arr[:, : task.trace_idx] = outputdata[:, : task.trace_idx]
        new_arr[: task.iteration_idx, task.trace_idx] = outputdata[
            : task.iteration_idx, task.trace_idx
        ]
        new_arr_shape = new_arr.shape

        masked_array = np.ma.array(new_arr, mask=np.isnan(new_arr))
        frame_cmap = plt.cm.get_cmap(task.cmap).copy()
        frame_cmap.set_bad(color="white")

        ax = axes[1]
        ax.imshow(
            masked_array,
            extent=[0, new_arr_shape[1], new_arr_shape[0] * task.dt, 0],
            interpolation="nearest",
            aspect="auto",
            cmap=frame_cmap,
        )
        ax.set_xlabel("Trace")
        ax.set_ylabel("Time")
        ax.set_title(f"{task.rx_component} B-scan")
        plt.tight_layout()
        canvas = FigureCanvas(fig)
        canvas.draw()
        image_array = np.asarray(canvas.buffer_rgba())[..., :3]
        Image.fromarray(image_array).save(task.frame_path)
    finally:
        plt.close(fig)
        plotter.close()

    return task.frame_index, task.frame_path


def in_notebook() -> bool:
    """Check if running inside a Jupyter notebook."""
    try:
        from IPython import get_ipython

        if get_ipython() is None:
            return False
        shell = get_ipython().__class__.__name__
        return shell == "ZMQInteractiveShell"
    except (ImportError, NameError):
        return False


class GprMaxModelSchema(BaseModel):
    title: str
    output_folder: Path
    domain_size: DomainSize
    domain_resolution: DomainResolution
    time_window: TimeWindow
    source: Optional[TxRxPair]
    materials: List[Material]
    geometry: List[Union[DomainBox, DomainCylinder, DomainSphere]]

    class Config:
        arbitrary_types_allowed = True
        json_encoders = {Path: lambda v: v.as_posix()}


class GprMaxModel:
    """
    A class representing a GprMax model.
    """

    def __init__(
        self,
        title: str,
        domain_size: DomainSize,
        domain_resolution: DomainResolution,
        time_window: TimeWindow,
        output_folder: Path,
    ):
        """
        Initialize a GprMax model.

        Args:
            title (str): Title of the model.
            domain_size (DomainSize): Size of the domain.
            domain_resolution (DomainResolution): Resolution of the domain.
            time_window (TimeWindow): Time window of the simulation.
            output_folder (Path): Path to the output folder.

        The default behavior for the absorbing boundary conditions (ABC)
        is first order Complex Frequency Shifted (CFS) Perfectly Matched Layers (PML),
        with thicknesses of 10 cells on each of the six sides of the model domain.
        This can be altered by using the n_pmcells parameter.
        """
        self.title = Title(title=title)
        self.output_folder = output_folder

        self.domain_size = domain_size
        self.domain_resolution = domain_resolution
        self.time_window = time_window

        self.source = None
        self.materials: List[Material] = []
        self.geometry: List[Union[DomainSphere, DomainCylinder, DomainBox]] = []
        self.output_views = []

    def data(self, rx: int = 1) -> Dict[str, Tuple[np.ndarray, float]]:
        """
        Get the data from the simulation.

        Args:
            rx (int): Receiver number.

        Returns:
            Dict[str, Tuple[np.ndarray, float]]: A dictionary with the data for each component (Ex, Ey, Ez, Hx, Hy, Hz).
        """
        output_file = self.output_folder / "output_merged.out"
        f = h5py.File(output_file, "r")
        nrx = f.attrs["nrx"]
        f.close()
        # Check there are any receivers
        if nrx == 0:
            raise Exception(f"No receivers found in {output_file}")
        assert rx <= nrx, f"Receiver {rx} does not exist in {output_file}"

        rx_components = ["Ex", "Ey", "Ez", "Hx", "Hy", "Hz"]
        data = {}
        for i, rx_component in enumerate(rx_components):
            outputdata, dt = get_output_data(str(output_file), rx, rx_component)
            data[rx_component] = outputdata, dt
        return data

    def _compute_n_traces(self) -> int:
        """
        Compute the number of steps to perform in the simulation.

        Returns:
            int: Number of steps.
        """
        if isinstance(self.source, TxRxPair):
            rx_pos_x = self.source.rx.x
            distance_to_end = self.domain_size.x - rx_pos_x
            n = math.floor(distance_to_end / self.source.rx_steps.dx)
            return n
        raise NotImplementedError("Not yet implemented")

    def _compute_dt(self) -> float:
        """
        Compute the time window for the simulation when time is discrete and
        it is defined as the number of iterations.

        Returns:
            float: Time step in seconds.
        """
        dx = self.domain_resolution.dx
        dy = self.domain_resolution.dy
        dz = self.domain_resolution.dz

        nx = round_value(self.domain_size.x / dx)
        ny = round_value(self.domain_size.y / dy)
        nz = round_value(self.domain_size.z / dz)

        # speed of light in m/s
        c = 299792458.0

        if nx == 1:
            dt = 1 / (c * np.sqrt((1 / dy) * (1 / dy) + (1 / dz) * (1 / dz)))
        elif ny == 1:
            dt = 1 / (c * np.sqrt((1 / dx) * (1 / dx) + (1 / dz) * (1 / dz)))
        elif nz == 1:
            dt = 1 / (c * np.sqrt((1 / dx) * (1 / dx) + (1 / dy) * (1 / dy)))
        else:
            # time step in seconds
            dt = 1 / (c * np.sqrt((1 / dx) * (1 / dx) + (1 / dy) * (1 / dy)))
        return dt

    def _compute_time_window(self) -> float:
        """
        Compute the time window for the simulation.

        Returns:
            float: Time window.
        """
        dt = self._compute_dt()
        twt = self.time_window.twt
        if is_integer_num(twt):
            time_window = (twt - 1) * dt
        else:
            time_window = twt
        return time_window

    def _compute_n_iterations(self) -> int:
        """
        Compute the number of iterations for the simulation.

        Returns:
            int: Number of iterations.
        """
        dt = self._compute_dt()
        twt = self.time_window.twt
        if is_integer_num(twt):
            iterations = twt
        else:
            iterations = int(np.ceil(twt / dt)) + 1
        return iterations

    def _compute_num_cells(self) -> Tuple[int, int, int]:
        """
        Compute the number of cells for the simulation.

        Returns:
            Tuple[int, int, int]: A tuple with the number of cells in each direction (x, y, z).
        """
        dx = self.domain_resolution.dx
        dy = self.domain_resolution.dy
        dz = self.domain_resolution.dz
        nx = round_value(self.domain_size.x / dx)
        ny = round_value(self.domain_size.y / dy)
        nz = round_value(self.domain_size.z / dz)
        return nx, ny, nz

    def run(self, *args, **kwargs) -> "GprMaxModel":
        """
        Run the simulation.

        Returns:
            GprMaxModel: The current instance of the GprMaxModel.
        """
        try:
            from gprMax.gprMax import api
        except ImportError:
            raise ImportError(
                "gprMax installation not found. Please install gprMax following the instructions at https://docs.gprmax.com/en/latest/include_readme.html"
            )

        nx, ny, nz = self._compute_num_cells()
        if nx == 0 or ny == 0 or nz == 0:
            raise Exception(" requires at least one cell in every dimension")

        n_traces = kwargs.pop("n", None)
        assert n_traces is not None, "The n argument must be specified"
        assert self.source, "No source set for the model"

        if isinstance(n_traces, str) and n_traces == "auto":
            logger.warning(
                "The n argument is set to auto. The number of steps will be computed automatically."
                "Notice that this option assume that the tx is being moved along the x axis, and there is only "
                "one tx in the model."
            )
            n_traces = self._compute_n_traces()

        out_geometry = kwargs.pop("geometry", False)
        out_snapshots = kwargs.pop("snapshots", False)
        snapshot_stride = _validate_positive_int(
            kwargs.pop("snapshot_stride", 1), "snapshot_stride"
        )
        num_threads = kwargs.pop("num_threads", None)
        if num_threads is not None:
            num_threads = _validate_positive_int(num_threads, "num_threads")
        mpi = _resolve_mpi_tasks(kwargs.pop("mpi", False), kwargs.get("gpu"), n_traces)
        mpi_no_spawn = kwargs.pop("mpi_no_spawn", False)
        gpu = kwargs.pop("gpu", None)
        geometry_fixed = kwargs.pop("geometry_fixed", False)
        geometry_only = kwargs.get("geometry_only", False)

        # create output folder
        clear_output_folder = kwargs.pop("clear_output_folder", True)
        self._mkdir_output_folder(clear_output_folder)

        input_prefix = ""
        if num_threads is not None:
            input_prefix = f"{NumThreads(n=num_threads)}\n"

        out_geometry = out_geometry or geometry_only
        output_commands = ""
        if any([out_geometry, out_snapshots]):
            output_commands = _capture_stdout(
                lambda: self._print_outputs(
                    geometry=out_geometry,
                    snapshots=out_snapshots,
                    snapshot_stride=snapshot_stride,
                )
            )

        # Write the input file
        model_file = self.output_folder / "sim.in"
        with open(model_file, "w") as f:
            f.write(input_prefix + str(self) + output_commands)

        # Run the simulation
        api_kwargs = {
            "mpi": mpi,
            "mpi_no_spawn": mpi_no_spawn,
            "gpu": gpu,
            "geometry_fixed": geometry_fixed,
            **kwargs,
        }
        api(str(model_file), *args, n=n_traces, **api_kwargs)

        # generated output file
        output_file = self.output_folder / "output_merged.out"
        if not output_file.exists() and not geometry_only:
            merge_model_files(output_file.parent, output_file)

        return self

    def _print_outputs(
        self,
        geometry: bool = True,
        snapshots: bool = True,
        snapshot_stride: int = 1,
    ) -> None:
        """
        Print the outputs.

        Args:
            geometry (bool): Whether to print geometry outputs.
            snapshots (bool): Whether to print snapshot outputs.
            snapshot_stride (int): Iteration interval between snapshot outputs.
        """
        snapshot_stride = _validate_positive_int(snapshot_stride, "snapshot_stride")
        if geometry:
            GeometryView(
                x_min=0,
                y_min=0,
                z_min=0,
                x_max=self.domain_size.x,
                y_max=self.domain_size.y,
                z_max=self.domain_size.z,
                dx=self.domain_resolution.dx,
                dy=self.domain_resolution.dy,
                dz=self.domain_resolution.dz,
                filename="geometry",
                resolution="n",
            )()

            if snapshots:
                iterations = self._compute_n_iterations()
                for i in range(1, iterations, snapshot_stride):
                    SnapshotView(
                        x_min=0,
                        y_min=0,
                        z_min=0,
                        x_max=self.domain_size.x,
                        y_max=self.domain_size.y,
                        z_max=self.domain_size.z,
                        dx=self.domain_resolution.dx,
                        dy=self.domain_resolution.dy,
                        dz=self.domain_resolution.dz,
                        filename="snapshot" + str(i),
                        t=i,
                    )()

    def _print_model_header(self) -> None:
        """
        Print the model header.
        """
        self.title()
        self.domain_size()
        self.domain_resolution()
        self.time_window()

    def _print_model_materials(self) -> None:
        """
        Print the model materials.
        """
        # we need to set the id of the material to the key of the dictionary
        for material in self.materials:
            material()

    def _print_geometry(self) -> None:
        """
        Print the geometries.
        """
        for geometry in self.geometry:
            geometry()

    def _print_source(self) -> None:
        """
        Print the model sources.
        """
        self.source()

    def __str__(self) -> str:
        """
        Return the string representation of the GprMax model.

        Returns:
            str: String representation of the model.
        """
        string_out = StringIO()
        sys.stdout = string_out

        self._print_model_header()

        self._print_model_materials()

        self._print_source()

        self._print_geometry()

        # Restore the standard output
        sys.stdout = sys.__stdout__
        return string_out.getvalue()

    def register_materials(self, *args: Material) -> None:
        """
        Register materials to the GprMax model.

        Args:
            *args (Material): Materials to register.
        """
        assert all(
            isinstance(material, Material) for material in args
        ), "All materials must be instances of the Material class."
        for material in args:
            self.materials.append(material)

    def add_geometry(
        self, *args: Union[DomainSphere, DomainCylinder, DomainBox, GeometryObjectsRead]
    ) -> None:
        """
        Register geometries to the GprMax model.

        Args:
            *args (Union[DomainSphere, DomainCylinder, DomainBox]): Geometries to register.
        """
        assert all(
            isinstance(
                geometry, (DomainSphere, DomainCylinder, DomainBox, GeometryObjectsRead)
            )
            for geometry in args
        ), "All geometries must be instances of the Geometry class."
        for geometry in args:
            self.geometry.append(geometry)

    def set_source(self, source: TxRxPair) -> None:
        """
        Register sources to the GprMax model.

        Args:
            source (TxRxPair): Source to register.
        """
        assert isinstance(
            source, TxRxPair
        ), "The source must be an instance of the TxRxPair class."
        self.source = source

    def _mkdir_output_folder(self, clear_output_folder: bool = True) -> None:
        """
        Create the output folder for the simulation.

        Args:
            clear_output_folder (bool): Whether to clear the output folder if it exists.
        """
        output_folder = self.output_folder
        if output_folder.exists() and clear_output_folder:
            rmdir(output_folder)
        output_folder.mkdir(parents=True, exist_ok=True)

    def plot_data(self, rx: int = 1, **kwargs) -> Union[None, Image.Image]:
        """
        Plot the data.

        Args:
            rx (int): Receiver number.

        Returns:
            Union[None, Image.Image]: Image of the plot if return_image is True, otherwise None.
        """
        data = self.data(rx=rx)
        rx_components = data.keys()
        n_cols = kwargs.pop("n_cols", 2)
        n_rows = math.ceil(len(rx_components) / n_cols)
        fig = plt.figure(figsize=(10, 10), facecolor="w", edgecolor="w")
        for i, rx_component in enumerate(rx_components):
            ax = fig.add_subplot(n_rows, n_cols, i + 1)
            ax.set_title(rx_component)
            outputdata, dt = data[rx_component]
            ax.imshow(
                outputdata,
                extent=[0, outputdata.shape[1], outputdata.shape[0] * dt, 0],
                interpolation="nearest",
                aspect="auto",
                cmap="gray",
            )
            ax.set_xlabel("Trace number")
            ax.set_ylabel("Time [s]")
        plt.tight_layout()

        return_image = kwargs.pop("return_image", False)
        if return_image:
            return figure2image(fig)
        plt.show()

    def plot_geometry(self, **kwargs) -> Union[None, Image.Image]:
        """
        Plot the model geometry using PyVista.

        Args:
            return_image (bool, optional): If True, returns a PIL image instead of showing an interactive window.

        Returns:
            Union[None, Image.Image]: PIL Image if return_image is True, otherwise None.
        """
        self.run(clear_output_folder=False, geometry_only=True, n=1)
        geometry_file = self.output_folder / "geometry.vti"

        if not geometry_file.exists():
            raise FileNotFoundError(f"Geometry file not found: {geometry_file}")

        return_image = kwargs.pop("return_image", False)
        notebook_mode = in_notebook()

        # Choose plotter
        if return_image:
            plotter = pv.Plotter(off_screen=True)
        elif notebook_mode:
            pv.set_jupyter_backend("trame")
            plotter = pv.Plotter(notebook=True)
        else:
            from PySide6.QtWidgets import QApplication, QDialog

            app = QApplication.instance() or QApplication(sys.argv)
            plotter_dialog = PlotterDialog()
            plotter = plotter_dialog.plotter

        # Load and add geometry
        geometry_grid = pv.read(geometry_file)
        plotter.set_background("white")
        plotter.add_mesh(
            geometry_grid, show_edges=False, opacity=0.3, show_scalar_bar=False
        )

        # Add TX/RX cubes
        dx = self.domain_resolution.dx * 2
        tx = self.source.tx.source
        rx = self.source.rx

        plotter.add_mesh(
            pv.Cube(center=(tx.x, tx.y, tx.z), x_length=dx, y_length=dx, z_length=dx),
            color="red",
        )
        plotter.add_mesh(
            pv.Cube(center=(rx.x, rx.y, rx.z), x_length=dx, y_length=dx, z_length=dx),
            color="blue",
        )

        # Configure camera
        plotter.camera_position = "xy"
        plotter.camera.tight()
        plotter.add_axes()

        # Show or return image
        if return_image:
            image = plotter.screenshot(return_img=True)
            return Image.fromarray(image)
        elif notebook_mode:
            return plotter.show()  # or 'panel'
        else:
            if plotter_dialog.exec() == QDialog.DialogCode.Rejected:
                sys.exit(0)

    def plot_snapshot(
        self,
        rx=1,
        trace_idx: int = None,
        iteration_idx: int = None,
        rx_component: str = "Ez",
        cmap="jet",
        return_image=False,
    ):
        """
        Plot a snapshot of the model.
        :param trace_idx:
        :param iteration_idx:
        :param rx_component:
        :return:
        """
        data = self.data(rx=rx)
        assert rx_component in data.keys(), f"Invalid rx component {rx_component}"

        outputdata, dt = data[rx_component]

        n_iterations, n_traces = outputdata.shape
        if iteration_idx is None:
            iteration_idx = n_iterations - 1

        if trace_idx is None:
            trace_idx = n_traces - 1

        trace_idx -= 1
        iteration_idx -= 1

        new_arr = np.full_like(outputdata, fill_value=np.nan)
        new_arr[:, :trace_idx] = outputdata[:, :trace_idx]
        new_arr[:iteration_idx, trace_idx] = outputdata[:iteration_idx, trace_idx]
        new_arr_shape = new_arr.shape

        # Creat a mask
        masked_array = np.ma.array(new_arr, mask=np.isnan(new_arr))
        cmap = plt.cm.get_cmap(cmap).copy()
        cmap.set_bad(color="white")

        fig, axes = plt.subplots(2, 1, figsize=(5, 10))

        # make B-scan plot
        ax = axes[1]
        ax.imshow(
            masked_array,
            extent=[0, new_arr_shape[1], new_arr_shape[0] * dt, 0],
            interpolation="nearest",
            aspect="auto",
            cmap=cmap,
        )
        ax.set_xlabel("Trace")
        ax.set_ylabel("Time")
        ax.set_title(f"{rx_component} B-scan")

        # make H-FIELD plot
        plotter = pv.Plotter(off_screen=True)
        plotter.set_background("white")
        plotter.camera_position = "xy"
        plotter.add_axes()
        snapshot_folder = self.output_folder.joinpath(f"sim_snaps{trace_idx + 1}")
        snapshot_file = snapshot_folder.joinpath(f"snapshot{iteration_idx + 1}.vti")
        snapshot_grid = pv.read(snapshot_file)
        plotter.add_mesh(
            snapshot_grid,
            cmap=cmap,
            scalars="H-field",
            show_edges=False,
            show_scalar_bar=False,
        )

        geometry_file = self.output_folder.joinpath(f"geometry{trace_idx + 1}.vti")
        geometry_grid = pv.read(geometry_file)
        plotter.add_mesh(
            geometry_grid, show_edges=False, show_scalar_bar=False, opacity=0.5
        )

        source = self.source
        tx = source.tx.source
        rx = source.rx

        tx_x, tx_y, tx_z = tx.x, tx.y, tx.z
        rx_x, rx_y, rx_z = rx.x, rx.y, rx.z

        plotter.add_mesh(
            pv.Cube(
                center=(tx_x + (trace_idx * self.domain_resolution.dx), tx_y, tx_z),
                x_length=self.domain_resolution.dx * 2,
                y_length=self.domain_resolution.dx * 2,
                z_length=self.domain_resolution.dx * 2,
            ),
            color="red",
        )

        plotter.add_mesh(
            pv.Cube(
                center=(rx_x + (trace_idx * self.domain_resolution.dx), tx_y, tx_z),
                x_length=self.domain_resolution.dx * 2,
                y_length=self.domain_resolution.dx * 2,
                z_length=self.domain_resolution.dx * 2,
            ),
            color="blue",
        )
        plotter.camera.tight()
        snapshot_capture = plotter.screenshot(return_img=True)
        snapshot_capture = Image.fromarray(snapshot_capture)
        ax = axes[0]
        ax.imshow(snapshot_capture, aspect="auto")
        ax.set_xlabel("Trace")
        ax.set_ylabel("Time")
        ax.set_title(
            f"{rx_component} Snapshot at trace {trace_idx + 1} and iteration {iteration_idx + 1}"
        )

        if return_image:
            return figure2image(fig)

        plt.tight_layout()
        plt.show()

    def _resolve_geometry_file_for_trace(self, trace_idx: int) -> Path:
        geometry_file = self.output_folder.joinpath(f"geometry{trace_idx + 1}.vti")
        if geometry_file.exists():
            return geometry_file

        fixed_geometry_file = self.output_folder.joinpath("geometry1.vti")
        if fixed_geometry_file.exists():
            return fixed_geometry_file

        single_geometry_file = self.output_folder.joinpath("geometry.vti")
        if single_geometry_file.exists():
            return single_geometry_file

        return geometry_file

    def _build_video_frame_tasks(
        self,
        outputdata: np.ndarray,
        dt: float,
        rx_component: str,
        cmap: str,
        figsize: Tuple[float, float],
        frame_step: int,
        temp_path: Path,
        data_file: Path,
    ) -> List[VideoFrameTask]:
        n_iterations, n_traces = outputdata.shape
        source = self.source
        tx = source.tx.source
        rx = source.rx
        tasks = []

        for frame_index, trace_idx, iteration_idx in _video_frame_indices(
            n_traces, n_iterations, frame_step
        ):
            snapshot_file = self.output_folder.joinpath(
                f"sim_snaps{trace_idx + 1}",
                f"snapshot{iteration_idx + 1}.vti",
            )
            frame_path = temp_path.joinpath(f"frame_{frame_index:08d}.png")
            tasks.append(
                VideoFrameTask(
                    frame_index=frame_index,
                    trace_idx=trace_idx,
                    iteration_idx=iteration_idx,
                    output_folder=str(self.output_folder),
                    snapshot_file=str(snapshot_file),
                    geometry_file=str(self._resolve_geometry_file_for_trace(trace_idx)),
                    data_file=str(data_file),
                    frame_path=str(frame_path),
                    dt=float(dt),
                    dx=float(self.domain_resolution.dx),
                    tx_x=float(tx.x),
                    tx_y=float(tx.y),
                    tx_z=float(tx.z),
                    rx_x=float(rx.x),
                    rx_y=float(rx.y),
                    rx_z=float(rx.z),
                    rx_component=rx_component,
                    cmap=cmap,
                    figsize=figsize,
                )
            )

        return tasks

    def _validate_video_frame_inputs(
        self, tasks: List[VideoFrameTask], frame_step: int
    ) -> None:
        missing = []
        required_files = {
            path for task in tasks for path in (task.snapshot_file, task.geometry_file)
        }
        for filename in sorted(required_files):
            if not Path(filename).exists():
                missing.append(filename)

        if not missing:
            return

        examples = "\n".join(f"  - {filename}" for filename in missing[:10])
        remaining = len(missing) - min(len(missing), 10)
        if remaining:
            examples += f"\n  - ... and {remaining} more"
        raise FileNotFoundError(
            "Missing snapshot or geometry files required to render the video.\n"
            f"Run model.run(..., geometry=True, snapshots=True, snapshot_stride={frame_step}) "
            "before save_video(), or choose a frame_step that matches existing snapshots.\n"
            f"Missing files:\n{examples}"
        )

    def animation_frame_generator(
        self, rx=1, rx_component: str = "Ez", cmap="jet", figsize=(10, 10)
    ):
        """
        Generate frames for the animation of the model simulation.

        Args:
            rx (int): Receiver number.
            rx_component (str): Receiver component to plot.
            cmap (str): Colormap to use for the plots.
            figsize (tuple): Size of the figure.

        Yields:
            PIL.Image.Image: Image of the current frame.
        """
        data = self.data(rx=rx)
        assert rx_component in data.keys(), f"Invalid rx component {rx_component}"
        outputdata, dt = data[rx_component]

        plotter = pv.Plotter(off_screen=True)
        plotter.set_background("white")
        plotter.camera_position = "xy"
        plotter.add_axes()

        n_traces = self._compute_n_traces()
        n_iterations = self._compute_n_iterations()
        for trace_idx in tqdm(range(n_traces)):
            for iteration_idx in range(0, n_iterations, 10):
                fig, axes = plt.subplots(2, 1, figsize=figsize)

                snapshot_folder = self.output_folder.joinpath(
                    f"sim_snaps{trace_idx + 1}"
                )
                snapshot_file = snapshot_folder.joinpath(
                    f"snapshot{iteration_idx + 1}.vti"
                )
                snapshot_grid = pv.read(snapshot_file)
                plotter.add_mesh(
                    snapshot_grid,
                    cmap=cmap,
                    scalars="H-field",
                    show_edges=False,
                    show_scalar_bar=False,
                )

                geometry_file = self.output_folder.joinpath(
                    f"geometry{trace_idx + 1}.vti"
                )
                geometry_grid = pv.read(geometry_file)
                plotter.add_mesh(
                    geometry_grid, show_edges=False, show_scalar_bar=False, opacity=0.5
                )
                source = self.source
                tx = source.tx.source
                rx = source.rx

                tx_x, tx_y, tx_z = tx.x, tx.y, tx.z
                rx_x, rx_y, rx_z = rx.x, rx.y, rx.z

                plotter.add_mesh(
                    pv.Cube(
                        center=(
                            tx_x + (trace_idx * self.domain_resolution.dx),
                            tx_y,
                            tx_z,
                        ),
                        x_length=self.domain_resolution.dx * 2,
                        y_length=self.domain_resolution.dx * 2,
                        z_length=self.domain_resolution.dx * 2,
                    ),
                    color="red",
                )

                plotter.add_mesh(
                    pv.Cube(
                        center=(
                            rx_x + (trace_idx * self.domain_resolution.dx),
                            tx_y,
                            tx_z,
                        ),
                        x_length=self.domain_resolution.dx * 2,
                        y_length=self.domain_resolution.dx * 2,
                        z_length=self.domain_resolution.dx * 2,
                    ),
                    color="blue",
                )
                plotter.camera.tight()
                snapshot_capture = plotter.screenshot(return_img=True)
                snapshot_capture = Image.fromarray(snapshot_capture)
                ax = axes[0]
                ax.imshow(snapshot_capture, aspect="auto")
                ax.set_xlabel("Trace")
                ax.set_ylabel("Time")
                ax.set_title(
                    f"{rx_component} Snapshot at trace {trace_idx + 1} and iteration {iteration_idx + 1}"
                )

                # make B-scan plot
                new_arr = np.full_like(outputdata, fill_value=np.nan)
                new_arr[:, :trace_idx] = outputdata[:, :trace_idx]
                new_arr[:iteration_idx, trace_idx] = outputdata[
                    :iteration_idx, trace_idx
                ]
                new_arr_shape = new_arr.shape

                # Create a mask
                masked_array = np.ma.array(new_arr, mask=np.isnan(new_arr))
                cmap = plt.cm.get_cmap(cmap).copy()
                cmap.set_bad(color="white")

                ax = axes[1]
                ax.imshow(
                    masked_array,
                    extent=[0, new_arr_shape[1], new_arr_shape[0] * dt, 0],
                    interpolation="nearest",
                    aspect="auto",
                    cmap=cmap,
                )
                ax.set_xlabel("Trace")
                ax.set_ylabel("Time")
                ax.set_title(f"{rx_component} B-scan")
                plt.tight_layout()
                canvas = FigureCanvas(fig)
                canvas.draw()
                # Get the image data as a string buffer and save it to a file
                image_array = np.asarray(canvas.buffer_rgba())[..., :3]

                data_capture = Image.fromarray(image_array)
                yield data_capture

                # in each iteration, clear the plotters
                plt.close(fig)
                plotter.clear()

                del snapshot_capture
                del data_capture

        plotter.close()

    def save_video(
        self,
        output_file: typing.Union[str, Path] = "model.mp4",
        fps=180,
        rx=1,
        rx_component: str = "Ez",
        cmap="jet",
        figsize=(10, 10),
        frame_step: int = 10,
        workers: typing.Union[int, str, None] = "auto",
        temp_dir: typing.Union[str, Path, None] = None,
    ):
        """
        Save the model simulation as a video.

        Args:
            output_file (str or Path): Path to the output video file.
            fps (int): Frames per second for the video.
            rx (int): Receiver number.
            rx_component (str): Receiver component to plot.
            cmap (str): Colormap to use for the plots.
            figsize (tuple): Size of the figure.
            frame_step (int): Iteration interval between rendered frames.
            workers (int | str | None): Number of parallel render workers. Use "auto" to choose a conservative default.
            temp_dir (str | Path | None): Parent directory for temporary rendered frame files.
        """
        frame_step = _validate_positive_int(frame_step, "frame_step")
        data = self.data(rx=rx)
        assert rx_component in data.keys(), f"Invalid rx component {rx_component}"
        outputdata, dt = data[rx_component]

        temp_dir_path = Path(temp_dir) if temp_dir is not None else None
        if temp_dir_path is not None:
            temp_dir_path.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=temp_dir_path) as working_dir:
            working_path = Path(working_dir)
            data_file = working_path.joinpath("outputdata.npy")
            np.save(data_file, outputdata)

            tasks = self._build_video_frame_tasks(
                outputdata=outputdata,
                dt=dt,
                rx_component=rx_component,
                cmap=cmap,
                figsize=figsize,
                frame_step=frame_step,
                temp_path=working_path,
                data_file=data_file,
            )
            if not tasks:
                raise ValueError("No frames were generated for the requested video")

            self._validate_video_frame_inputs(tasks, frame_step)
            worker_count = _resolve_frame_workers(workers, len(tasks))
            output_file = str(output_file)
            vout = None

            try:
                if worker_count == 1:
                    rendered_frames = map(_render_video_frame, tasks)
                    iterator = tqdm(rendered_frames, total=len(tasks))
                else:
                    executor = ProcessPoolExecutor(max_workers=worker_count)
                    rendered_frames = executor.map(_render_video_frame, tasks)
                    iterator = tqdm(rendered_frames, total=len(tasks))

                try:
                    for expected_index, (frame_index, frame_path) in enumerate(
                        iterator
                    ):
                        if frame_index != expected_index:
                            raise RuntimeError(
                                f"Rendered frame order mismatch: expected {expected_index}, got {frame_index}"
                            )
                        with Image.open(frame_path) as curr_frame:
                            curr_frame = curr_frame.convert("RGB")
                            if vout is None:
                                cap_size = curr_frame.size
                                fourcc = cv2.VideoWriter_fourcc("m", "p", "4", "v")
                                vout = cv2.VideoWriter()
                                success = vout.open(
                                    output_file, fourcc, fps, cap_size, True
                                )
                                if not success:
                                    raise Exception(
                                        "Could not open video file for writing"
                                    )
                            vout.write(np.asarray(curr_frame))
                finally:
                    if worker_count != 1:
                        executor.shutdown(wait=True, cancel_futures=True)
            finally:
                if vout is not None:
                    vout.release()

    def to_json(
        self, path: Union[str, Path] = None, indent: int = 2
    ) -> Union[str, None]:
        """
        Export the GprMaxModel to JSON format using Pydantic serialization.

        Args:
            path (str or Path): If provided, writes JSON to file.
            indent (int): Indentation for pretty printing.

        Returns:
            str | None: JSON string if `path` is None, else None.
        """
        schema = GprMaxModelSchema(
            title=self.title.title,
            output_folder=self.output_folder,
            domain_size=self.domain_size,
            domain_resolution=self.domain_resolution,
            time_window=self.time_window,
            source=self.source,
            materials=self.materials,
            geometry=self.geometry,
        )

        json_str = schema.model_dump_json(indent=indent)

        if path:
            Path(path).write_text(json_str)
            return None
        return json_str

    @staticmethod
    def from_json(data: Union[str, Path, dict]) -> GprMaxModel:
        """
        Load a GprMaxModel from a JSON file path, JSON string, or Python dictionary.

        Args:
            data (Union[str, Path, dict]): The path to a JSON file, a JSON string, or a parsed dict.

        Returns:
            GprMaxModel: The reconstructed model.
        """
        # Step 1: Load JSON data
        if isinstance(data, Path) or (isinstance(data, str) and Path(data).exists()):
            with open(data, "r") as f:
                json_obj = json.load(f)
        elif isinstance(data, str):
            json_obj = json.loads(data)
        elif isinstance(data, dict):
            json_obj = data
        else:
            raise TypeError(
                "Unsupported input type. Must be a path, JSON string, or dict."
            )

        # Step 2: Validate with schema
        schema = GprMaxModelSchema(**json_obj)

        # Step 3: Build the actual model instance
        model = GprMaxModel(
            title=schema.title,
            domain_size=schema.domain_size,
            domain_resolution=schema.domain_resolution,
            time_window=schema.time_window,
            output_folder=schema.output_folder,
        )
        if schema.source:
            model.set_source(schema.source)
        model.register_materials(*schema.materials)
        model.add_geometry(*schema.geometry)

        return model
