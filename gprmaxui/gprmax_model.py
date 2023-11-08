import logging
import math
import sys
from io import StringIO
from pathlib import Path

import matplotlib
from PySide6.QtWidgets import QApplication, QDialog

import cv2
import h5py
import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
from PIL import Image
from gprMax.gprMax import api
from gprMax.utilities import round_value
from gprmaxui.domain_commands import *
from gprmaxui.geometry_commands import *
from gprmaxui.plotter import PlotterDialog
from gprmaxui.sources_commands import *
from gprmaxui.utils import (
    get_output_data,
    rmdir,
    merge_model_files,
    is_integer_num,
    figure2image,
)
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from tqdm import tqdm

logger = logging.getLogger("rich")


class GprMaxModel:
    """
    A dataclass representing a GprMax model.
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
        :param title: title of the model
        :param domain_size:  size of the domain
        :param domain_resolution: resolution of the domain
        :param time_window: time window of the simulation
        :param n_pmlcells: number of pml cells

        The default behaviour for the absorbing boundary conditions (ABC)
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
        self.materials = []
        self.geometry = []
        self.output_views = []

    def data(self, rx: int = 1):
        """
        Get the data from the simulation
        @param rx: receiver number
        :return: a dictionary with the data for each component (Ex, Ey, Ez, Hx, Hy, Hz)
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

    def _compute_n_traces(self):
        """
        Compute the number of steps to perform in the simulation
        :return:
        """
        if isinstance(self.source, TxRxPair):
            rx_pos_x = self.source.rx.x
            distance_to_end = self.domain_size.x - rx_pos_x
            n = math.floor(distance_to_end / self.source.rx_steps.dx)
            return n
        raise NotImplementedError("Not yet implemented")

    def _compute_dt(self):
        """
        Compute the t window for the simulation when t is discrete and
        it is defined as the number of iterations
        :return:
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

    def _compute_time_window(self):
        """
        Compute the time window for the simulation
        :return:
        """
        dt = self._compute_dt()
        twt = self.time_window.twt
        if is_integer_num(twt):
            time_window = (twt - 1) * dt
        else:
            time_window = twt
        return time_window

    def _compute_n_iterations(self):
        """
        Compute the number of iterations for the simulation
        :return:
        """
        dt = self._compute_dt()
        twt = self.time_window.twt
        if is_integer_num(twt):
            iterations = twt
        else:
            iterations = int(np.ceil(twt / dt)) + 1
        return iterations

    def _compute_num_cells(self):
        """
        Compute the number of cells for the simulation
        :return: a tuple with the number of cells in each direction (x, y, z)
        """
        dx = self.domain_resolution.dx
        dy = self.domain_resolution.dy
        dz = self.domain_resolution.dz
        nx = round_value(self.domain_size.x / dx)
        ny = round_value(self.domain_size.y / dy)
        nz = round_value(self.domain_size.z / dz)
        return nx, ny, nz

    def run(self, *args, **kwargs):
        """
        Run the simulation
        :return:
        """
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
        geometry_only = kwargs.get("geometry_only", False)

        # create output folder
        clear_output_folder = kwargs.pop("clear_output_folder", True)
        self._mkdir_output_folder(clear_output_folder)

        string_out = StringIO()
        sys.stdout = string_out
        out_geometry = out_geometry or geometry_only
        if any([out_geometry, out_snapshots]):
            self._print_outputs(geometry=out_geometry, snapshots=out_snapshots)
        sys.stdout = sys.__stdout__

        # Write the input file
        model_file = self.output_folder / "sim.in"
        with open(model_file, "w") as f:
            f.write(str(self) + string_out.getvalue())

        # Run the simulation
        api(str(model_file), n=n_traces, *args, **kwargs)

        # generated output file
        output_file = self.output_folder / "output_merged.out"
        if not output_file.exists() and not geometry_only:
            merge_model_files(output_file.parent, output_file)

        return self

    def _print_outputs(self, geometry=True, snapshots=True):
        """
        Print the outputs.
        :return:
        """
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
                for i in range(1, iterations):
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

    def _print_model_header(self):
        """
        Print the model header.
        :return:
        """
        self.title()
        self.domain_size()
        self.domain_resolution()
        self.time_window()

    def _print_model_materials(self):
        """
        Print the model materials.
        :return:
        """
        # we need to set the id of the material to the key of the dictionary
        for material in self.materials:
            material()

    def _print_geometry(self):
        """
        Print the geometries.
        :return:
        """
        for geometry in self.geometry:
            geometry()

    def _print_source(self):
        """
        Print the model sources.
        :return:
        """
        self.source()

    def __str__(self):
        """
        Return the string representation of the GPRMax model.
        :return: str
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

    def register_materials(self, *args):
        """
        Register materials to the GPRMax model.
        :param kwargs:
        :return:
        """
        assert all(
            isinstance(material, Material) for material in args
        ), "All materials must be instances of the Material class."
        for material in args:
            self.materials.append(material)

    def add_geometry(self, *args):
        """
        Register geometries to the GPRMax model.
        :param kwargs:
        :return:
        """
        assert all(
            isinstance(geometry, (DomainSphere, DomainCylinder, DomainBox))
            for geometry in args
        ), "All geometries must be instances of the Geometry class."
        for geometry in args:
            self.geometry.append(geometry)

    def set_source(self, source: TxRxPair):
        """
        Register sources to the GPRMax model.
        :param kwargs:
        :return:
        """
        assert isinstance(
            source, TxRxPair
        ), "The source must be an instance of the TxRxPair class."
        self.source = source

    def _mkdir_output_folder(self, clear_output_folder=True):
        """
        Create the output folder for the simulation.
        :return:
        """
        output_folder = self.output_folder
        if output_folder.exists() and clear_output_folder:
            rmdir(output_folder)
        output_folder.mkdir(parents=True, exist_ok=True)

    def plot_data(self, rx=1, ax=None, **kwargs):
        """
        plot the data
        :param rx: int
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

    def plot_geometry(self, **kwargs):
        """
        Plot the model geometry.
        """

        self.run(clear_output_folder=False, geometry_only=True, n=1)
        geometry_file = self.output_folder / "geometry.vti"

        return_image = kwargs.pop("return_image", False)
        if return_image:
            plotter = pv.Plotter(off_screen=True)
        else:
            app = QApplication(sys.argv)
            plotter_dialog = PlotterDialog()
            plotter = plotter_dialog.plotter

        plotter.set_background("white")
        geometry_grid = pv.read(geometry_file)
        plotter.add_mesh(
            geometry_grid, show_edges=False, opacity=0.3, show_scalar_bar=False
        )

        source = self.source
        tx = source.tx.source
        rx = source.rx

        tx_x, tx_y, tx_z = tx.x, tx.y, tx.z
        rx_x, rx_y, rx_z = rx.x, rx.y, rx.z
        plotter.add_mesh(
            pv.Cube(
                center=(tx_x, tx_y, tx_z),
                x_length=self.domain_resolution.dx * 2,
                y_length=self.domain_resolution.dx * 2,
                z_length=self.domain_resolution.dx * 2,
            ),
            color="red",
        )

        plotter.add_mesh(
            pv.Cube(
                center=(rx_x, rx_y, rx_z),
                x_length=self.domain_resolution.dx * 2,
                y_length=self.domain_resolution.dx * 2,
                z_length=self.domain_resolution.dx * 2,
            ),
            color="blue",
        )

        plotter.camera_position = "xy"
        plotter.camera.tight()
        plotter.add_axes()

        if return_image:
            plotter.camera.tight()
            image = plotter.screenshot(return_img=True)
            image = Image.fromarray(image)
            return image
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

    def animation_frame_generator(
        self, rx=1, rx_component: str = "Ez", cmap="jet", figsize=(10, 10)
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

                # Creat a mask
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
                image_buffer = canvas.tostring_rgb()
                image_width, image_height = canvas.get_width_height()
                # Create a PIL image from the string buffer
                image_array = np.frombuffer(image_buffer, dtype=np.uint8).reshape(
                    image_height, image_width, 3
                )
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
    ):
        """
        Save the model simulation as a video.
        """
        # Write the PIL Images to the VideoWriter object.
        for i, curr_frame in enumerate(
            self.animation_frame_generator(
                rx=rx, rx_component=rx_component, cmap=cmap, figsize=figsize
            )
        ):
            if i == 0:
                fps = fps
                cap_size = curr_frame.size
                fourcc = cv2.VideoWriter_fourcc(
                    "m", "p", "4", "v"
                )  # note the lower case
                vout = cv2.VideoWriter()
                success = vout.open(output_file, fourcc, fps, cap_size, True)
            vout.write(np.asarray(curr_frame))
        vout.release()
