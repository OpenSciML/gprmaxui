import math
import os
import re
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from gprMax._version import __version__
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas


def rmdir(folder: Path):
    """
    Clear a folder recursively.
    """
    for child in folder.iterdir():
        if child.is_dir():
            rmdir(child)
        else:
            child.unlink()
    folder.rmdir()


def get_output_data(filename, rxnumber, rxcomponent):
    """Gets B-scan output data from a model.

    Args:
        filename (string): Filename (including path) of output file.
        rxnumber (int): Receiver output number.
        rxcomponent (str): Receiver output field/current component.

    Returns:
        outputdata (array): Array of A-scans, i.e. B-scan data.
        dt (float): Temporal resolution of the model.
    """

    # Open output file and read some attributes
    f = h5py.File(filename, "r")
    nrx = f.attrs["nrx"]
    dt = f.attrs["dt"]

    # Check there are any receivers
    if nrx == 0:
        raise Exception("No receivers found in {}".format(filename))

    path = "/rxs/rx" + str(rxnumber) + "/"
    availableoutputs = list(f[path].keys())

    # Check if requested output is in file
    if rxcomponent not in availableoutputs:
        raise Exception(
            "{} output requested to plot, but the available output for receiver 1 is {}".format(
                rxcomponent, ", ".join(availableoutputs)
            )
        )

    outputdata = f[path + "/" + rxcomponent]
    outputdata = np.array(outputdata)
    f.close()

    return outputdata, dt


def is_integer_num(n):
    """
    Check if a number is an integer
    :param n: number
    :return:  True if n is an integer, False otherwise
    """
    if isinstance(n, int):
        return True
    if isinstance(n, float):
        return n.is_integer()
    return False


def merge_model_files(output_folder: Path, output_file: Path):
    """
    Merge the output files from a simulation run into a single file.
    """
    out_files = list(output_folder.glob("*.out"))
    if len(out_files) == 0:
        raise ValueError(f"No output files found in {output_folder}")
    out_files.sort(key=lambda x: int(re.search(r"\d+", x.stem).group()))
    model_runs = len(out_files)

    # Combined output file
    with h5py.File(output_file, "w") as fout:
        # Add positional data for rxs
        for model in range(model_runs):
            fin = h5py.File(out_files[model], "r")
            nrx = fin.attrs["nrx"]

            # Write properties for merged file on first iteration
            if model == 0:
                fout.attrs["Title"] = fin.attrs["Title"]
                fout.attrs["gprMax"] = __version__
                fout.attrs["Iterations"] = fin.attrs["Iterations"]
                fout.attrs["dt"] = fin.attrs["dt"]
                fout.attrs["nrx"] = fin.attrs["nrx"]
                for rx in range(1, nrx + 1):
                    path = "/rxs/rx" + str(rx)
                    grp = fout.create_group(path)
                    availableoutputs = list(fin[path].keys())
                    for output in availableoutputs:
                        grp.create_dataset(
                            output,
                            (fout.attrs["Iterations"], model_runs),
                            dtype=fin[path + "/" + output].dtype,
                        )

            # For all receivers
            for rx in range(1, nrx + 1):
                path = "/rxs/rx" + str(rx) + "/"
                availableoutputs = list(fin[path].keys())
                # For all receiver outputs
                for output in availableoutputs:
                    fout[path + "/" + output][:, model] = fin[path + "/" + output][:]

            fin.close()

    # for file in out_files:
    #     file.unlink()


def mpl_plot(filename, outputdata, dt, rxnumber, rxcomponent):
    """Creates a plot (with matplotlib) of the B-scan.

    Args:
        filename (string): Filename (including path) of output file.
        outputdata (array): Array of A-scans, i.e. B-scan data.
        dt (float): Temporal resolution of the model.
        rxnumber (int): Receiver output number.
        rxcomponent (str): Receiver output field/current component.

    Returns:
        plt (object): matplotlib plot object.
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
    # plt.title('{}'.format(filename))

    # Grid properties
    ax = fig.gca()
    ax.grid(which="both", axis="both", linestyle="-.")

    cb = plt.colorbar()
    if "E" in rxcomponent:
        cb.set_label("Field strength [V/m]")
    elif "H" in rxcomponent:
        cb.set_label("Field strength [A/m]")
    elif "I" in rxcomponent:
        cb.set_label("Current [A]")

    # Save a PDF/PNG of the figure
    # savefile = os.path.splitext(filename)[0]
    # fig.savefig(path + os.sep + savefile + '.pdf', dpi=None, format='pdf',
    #             bbox_inches='tight', pad_inches=0.1)
    # fig.savefig(path + os.sep + savefile + '.png', dpi=150, format='png',
    #             bbox_inches='tight', pad_inches=0.1)

    return plt


def stretch_arr(data_array: np.ndarray, num_std: float = 1.5):
    """
    Stretch a numpy array to a specified number of standard deviations.
    :param data_array:
    :param num_std:
    :return:
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


def plot_model(output_folder: Path, n_cols=3):
    """
    Plot the output of a simulation run.
    """
    output_file = output_folder / "output_merged.out"
    if not output_file.exists():
        merge_model_files(output_folder, output_file)
    # Open output file and read number of outputs (receivers)
    f = h5py.File(output_file, "r")
    nrx = f.attrs["nrx"]
    f.close()
    # Check there are any receivers
    if nrx == 0:
        raise Exception("No receivers found in {}".format(output_file))

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
                # vmin=-np.amax(np.abs(outputdata)),
                # vmax=np.amax(np.abs(outputdata)),
            )
            ax.set_xlabel("Trace number")
            ax.set_ylabel("Time [s]")
    plt.show()


def concat_images_h(im_list, resample=Image.BICUBIC):
    """
    Concatenate images horizontally with multiple resize.
    :param im_list:
    :param resample:
    :return:
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


def concat_images_v(im_list, resample=Image.BICUBIC):
    """
    Concatenate images vertically with multiple resize.
    :param im_list:
    :param resample:
    :return:
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


def make_images_grid_from_2dlist(im_list_2d, resample=Image.BICUBIC):
    """
    Concatenate images in a 2D list/tuple of images, with multiple resize.
    :param im_list_2d:
    :param resample:
    :return:
    """
    im_list_v = [
        concat_images_h(im_list_h, resample=resample) for im_list_h in im_list_2d
    ]
    return concat_images_v(im_list_v, resample=resample)


def make_images_grid(images_list, num_cols, resample=Image.BICUBIC):
    """
    Make a grid of images.
    :param images_list: list of images
    :param num_cols: number of columns
    :param resample: resample method
    :return:
    """
    num_rows = math.ceil(len(images_list) / num_cols)
    images_list_2d = []
    for i in range(num_rows):
        images_list_2d.append(images_list[i * num_cols : (i + 1) * num_cols])
    return make_images_grid_from_2dlist(images_list_2d, resample=resample)


def figure2image(fig):
    # Create a canvas and render the figure onto it
    canvas = FigureCanvas(fig)
    canvas.draw()
    # Get the image data as a string buffer and save it to a file
    image_buffer = canvas.tostring_rgb()
    image_width, image_height = canvas.get_width_height()
    # Create a PIL image from the string buffer
    image_array = np.frombuffer(image_buffer, dtype=np.uint8).reshape(
        image_height, image_width, 3
    )
    image = Image.fromarray(image_array)
    plt.close(fig)
    return image
