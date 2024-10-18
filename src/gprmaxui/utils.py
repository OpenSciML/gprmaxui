import math
import os
import re
from pathlib import Path
from typing import List, Tuple, Union

import h5py
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from gprMax._version import __version__
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas


def rmdir(folder: Path) -> None:
    """
    Clear a folder recursively.

    Args:
        folder (Path): The folder to clear.
    """
    for child in folder.iterdir():
        if child.is_dir():
            rmdir(child)
        else:
            child.unlink()
    folder.rmdir()


def get_output_data(filename: str, rxnumber: int, rxcomponent: str) -> Tuple[np.ndarray, float]:
    """
    Gets B-scan output data from a model.

    Args:
        filename (str): Filename (including path) of output file.
        rxnumber (int): Receiver output number.
        rxcomponent (str): Receiver output field/current component.

    Returns:
        Tuple[np.ndarray, float]: Array of A-scans (B-scan data) and temporal resolution of the model.
    """
    f = h5py.File(filename, "r")
    nrx = f.attrs["nrx"]
    dt = f.attrs["dt"]

    if nrx == 0:
        raise Exception(f"No receivers found in {filename}")

    path = f"/rxs/rx{rxnumber}/"
    availableoutputs = list(f[path].keys())

    if rxcomponent not in availableoutputs:
        raise Exception(
            f"{rxcomponent} output requested to plot, but the available output for receiver 1 is {', '.join(availableoutputs)}"
        )

    outputdata = np.array(f[path + "/" + rxcomponent])
    f.close()

    return outputdata, dt


def is_integer_num(n: Union[int, float]) -> bool:
    """
    Check if a number is an integer.

    Args:
        n (Union[int, float]): The number to check.

    Returns:
        bool: True if n is an integer, False otherwise.
    """
    if isinstance(n, int):
        return True
    if isinstance(n, float):
        return n.is_integer()
    return False


def merge_model_files(output_folder: Path, output_file: Path) -> None:
    """
    Merge the output files from a simulation run into a single file.

    Args:
        output_folder (Path): The folder containing the output files.
        output_file (Path): The path to the merged output file.
    """
    out_files = list(output_folder.glob("*.out"))
    if len(out_files) == 0:
        raise ValueError(f"No output files found in {output_folder}")
    out_files.sort(key=lambda x: int(re.search(r"\d+", x.stem).group()))
    model_runs = len(out_files)

    with h5py.File(output_file, "w") as fout:
        for model in range(model_runs):
            fin = h5py.File(out_files[model], "r")
            nrx = fin.attrs["nrx"]

            if model == 0:
                fout.attrs["Title"] = fin.attrs["Title"]
                fout.attrs["gprMax"] = __version__
                fout.attrs["Iterations"] = fin.attrs["Iterations"]
                fout.attrs["dt"] = fin.attrs["dt"]
                fout.attrs["nrx"] = fin.attrs["nrx"]
                for rx in range(1, nrx + 1):
                    path = f"/rxs/rx{rx}"
                    grp = fout.create_group(path)
                    availableoutputs = list(fin[path].keys())
                    for output in availableoutputs:
                        grp.create_dataset(
                            output,
                            (fout.attrs["Iterations"], model_runs),
                            dtype=fin[path + "/" + output].dtype,
                        )

            for rx in range(1, nrx + 1):
                path = f"/rxs/rx{rx}/"
                availableoutputs = list(fin[path].keys())
                for output in availableoutputs:
                    fout[path + "/" + output][:, model] = fin[path + "/" + output][:]

            fin.close()


def mpl_plot(filename: str, outputdata: np.ndarray, dt: float, rxnumber: int, rxcomponent: str) -> plt.Figure:
    """
    Creates a plot (with matplotlib) of the B-scan.

    Args:
        filename (str): Filename (including path) of output file.
        outputdata (np.ndarray): Array of A-scans (B-scan data).
        dt (float): Temporal resolution of the model.
        rxnumber (int): Receiver output number.
        rxcomponent (str): Receiver output field/current component.

    Returns:
        plt.Figure: Matplotlib plot object.
    """
    (path, filename) = os.path.split(filename)

    fig = plt.figure(
        num=filename + " - rx" + str(rxnumber),
        figsize=(20, 10),
        facecolor="w",
        edgecolor="w",
    )

    plt.imshow(
        outputdata,
        extent=[0, outputdata.shape[1], outputdata.shape[0] * dt, 0],
        interpolation="nearest",
        aspect="auto",
        cmap="gray",
        vmin=-np.amax(np.abs(outputdata)),
        vmax=np.amax(np.abs(outputdata)),
    )
    plt.xlabel("Trace number")
    plt.ylabel("Time [s]")

    ax = fig.gca()
    ax.grid(which="both", axis="both", linestyle="-.")

    cb = plt.colorbar()
    if "E" in rxcomponent:
        cb.set_label("Field strength [V/m]")
    elif "H" in rxcomponent:
        cb.set_label("Field strength [A/m]")
    elif "I" in rxcomponent:
        cb.set_label("Current [A]")

    return plt


def stretch_arr(data_array: np.ndarray, num_std: float = 1.5) -> np.ndarray:
    """
    Stretch a numpy array to a specified number of standard deviations.

    Args:
        data_array (np.ndarray): The data array to stretch.
        num_std (float): The number of standard deviations to stretch to.

    Returns:
        np.ndarray: The stretched data array.
    """
    data_array = data_array.astype(np.float32)
    data_stdev = np.nanstd(data_array)
    data_mean = np.nanmean(data_array)
    data_max_new = data_mean + num_std * data_stdev
    data_min_new = data_mean - num_std * data_stdev
    data_array[data_array > data_max_new] = data_max_new
    data_array[data_array < data_min_new] = data_min_new
    data_max = np.nanmax(data_array)
    data_min = np.nanmin(data_array)
    data_range = data_max - data_min
    data_array = (data_array - data_min) / data_range
    return data_array


def plot_model(output_folder: Path, n_cols: int = 3) -> None:
    """
    Plot the output of a simulation run.

    Args:
        output_folder (Path): The folder containing the output files.
        n_cols (int): Number of columns in the plot grid.
    """
    output_file = output_folder / "output_merged.out"
    if not output_file.exists():
        merge_model_files(output_folder, output_file)
    f = h5py.File(output_file, "r")
    nrx = f.attrs["nrx"]
    f.close()
    if nrx == 0:
        raise Exception(f"No receivers found in {output_file}")

    rx_components = ["Ex", "Ey", "Ez", "Hx", "Hy", "Hz"]
    for rx in range(1, nrx + 1):
        nrows = math.ceil(len(rx_components) / n_cols)
        ncols = n_cols
        fig = plt.figure(figsize=(10, 10), facecolor="w", edgecolor="w")
        for i, rx_component in enumerate(rx_components):
            outputdata, dt = get_output_data(output_file, rx, rx_component)
            try:
                outputdata = stretch_arr(outputdata)
            except:
                pass
            ax = fig.add_subplot(nrows, ncols, i + 1)
            ax.set_title(rx_component + f" - ({outputdata.shape})")
            ax.imshow(
                outputdata,
                extent=[0, outputdata.shape[1], outputdata.shape[0] * dt, 0],
                interpolation="nearest",
                aspect="auto",
                cmap="gray",
            )
            ax.set_xlabel("Trace number")
            ax.set_ylabel("Time [s]")
    plt.show()


def concat_images_h(im_list: List[Image.Image], resample: int = Image.BICUBIC) -> Image.Image:
    """
    Concatenate images horizontally with multiple resize.

    Args:
        im_list (List[Image.Image]): List of images to concatenate.
        resample (int): Resample method.

    Returns:
        Image.Image: Concatenated image.
    """
    min_height = min(im.height for im in im_list)
    im_list_resize = [
        im.resize(
            (int(im.width * min_height / im.height), min_height), resample=resample
        )
        for im in im_list
    ]
    total_width = sum(im.width for im in im_list_resize)
    dst = Image.new("RGB", (total_width, min_height))
    pos_x = 0
    for im in im_list_resize:
        dst.paste(im, (pos_x, 0))
        pos_x += im.width
    return dst


def concat_images_v(im_list: List[Image.Image], resample: int = Image.BICUBIC) -> Image.Image:
    """
    Concatenate images vertically with multiple resize.

    Args:
        im_list (List[Image.Image]): List of images to concatenate.
        resample (int): Resample method.

    Returns:
        Image.Image: Concatenated image.
    """
    min_width = min(im.width for im in im_list)
    im_list_resize = [
        im.resize((min_width, int(im.height * min_width / im.width)), resample=resample)
        for im in im_list
    ]
    total_height = sum(im.height for im in im_list_resize)
    dst = Image.new("RGB", (min_width, total_height))
    pos_y = 0
    for im in im_list_resize:
        dst.paste(im, (0, pos_y))
        pos_y += im.height
    return dst


def make_images_grid_from_2dlist(im_list_2d: List[List[Image.Image]], resample: int = Image.BICUBIC) -> Image.Image:
    """
    Concatenate images in a 2D list/tuple of images, with multiple resize.

    Args:
        im_list_2d (List[List[Image.Image]]): 2D list of images to concatenate.
        resample (int): Resample method.

    Returns:
        Image.Image: Concatenated image.
    """
    im_list_v = [
        concat_images_h(im_list_h, resample=resample) for im_list_h in im_list_2d
    ]
    return concat_images_v(im_list_v, resample=resample)


def make_images_grid(images_list: List[Image.Image], num_cols: int, resample: int = Image.BICUBIC) -> Image.Image:
    """
    Make a grid of images.

    Args:
        images_list (List[Image.Image]): List of images.
        num_cols (int): Number of columns.
        resample (int): Resample method.

    Returns:
        Image.Image: Grid of images.
    """
    num_rows = math.ceil(len(images_list) / num_cols)
    images_list_2d = [
        images_list[i * num_cols : (i + 1) * num_cols] for i in range(num_rows)
    ]
    return make_images_grid_from_2dlist(images_list_2d, resample=resample)


def figure2image(fig: plt.Figure) -> Image.Image:
    """
    Convert a Matplotlib figure to a PIL Image.

    Args:
        fig (plt.Figure): Matplotlib figure.

    Returns:
        Image.Image: PIL Image.
    """
    canvas = FigureCanvas(fig)
    canvas.draw()
    image_buffer = canvas.tostring_rgb()
    image_width, image_height = canvas.get_width_height()
    image_array = np.frombuffer(image_buffer, dtype=np.uint8).reshape(
        image_height, image_width, 3
    )
    image = Image.fromarray(image_array)
    plt.close(fig)
    return image