import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tifffile
from scipy import ndimage as ndi
from skimage.filters import sobel
from skimage.segmentation import watershed

from spacec._shared.segmentation import format_CODEX
from spacec.plotting._style import SPACEcStyle, _add_scalebar, _nice_scalebar_length


def label_tissue(
    resized_im,
    lower_cutoff=0.012,
    upper_cutoff=0.025,
    savefig=False,
    showfig=True,
    output_dir="./",
    output_fname="",
):
    """
    Label the tissue in the given image.

    Parameters
    ----------
    resized_im : ndarray
        The resized image to label.
    lower_cutoff : float, optional
        Pixels in `resized_im` at or below this intensity are seeded as
        confident background markers for the watershed segmentation, by
        default 0.012.
    upper_cutoff : float, optional
        Pixels in `resized_im` at or above this intensity are seeded as
        confident tissue/foreground markers for the watershed segmentation,
        by default 0.025.
    savefig : bool, optional
        Whether to save the figure or not, by default False.
    showfig : bool, optional
        Whether to show the figure or not, by default True.
    output_dir : str, optional
        The directory to save the figure in, by default "./".
    output_fname : str, optional
        The filename to save the figure as, by default "".

    Returns
    -------
    DataFrame
        A DataFrame containing the labels from the segmentation.
    """
    # cut off
    elevation_map = sobel(resized_im)
    markers = np.zeros_like(resized_im, dtype=int)
    markers[resized_im <= lower_cutoff] = 1
    markers[resized_im >= upper_cutoff] = 2

    segmentation = watershed(elevation_map, markers)
    # plt.imshow(segmentation)
    # plt.title('Segmented tissues')
    # plt.show()

    segmentation = ndi.binary_fill_holes(segmentation - 1)
    # plt.imshow(segmentation)
    # plt.title('Segmented tissues with holes filled')
    # plt.show()

    # visualize initial identified segmented masks
    labeled_tissues, _ = ndi.label(segmentation)
    print(f"Identified {len(np.unique(labeled_tissues)) - 1} tissue pieces")
    fig, axs = plt.subplots(1, 1)
    axs.imshow(labeled_tissues)
    axs.set_title("Labeled tissues")

    if showfig:
        if savefig:
            fig.savefig(
                output_dir + output_fname + "_labeled_seg_tissue_plot.pdf",
                bbox_inches="tight",
            )
            plt.close(fig)
        else:
            plt.show()
    else:
        plt.close(fig)

    #######Non clustering option
    print("Saving the labels from the segmentation!")
    idx = np.nonzero(labeled_tissues)
    vals = labeled_tissues[idx]
    tissueframe = pd.DataFrame(vals, columns=["tissue"])
    tissueframe["y"] = idx[0]
    tissueframe["x"] = idx[1]
    tissueframe["region1"] = tissueframe["tissue"]

    return tissueframe


def save_labelled_tissue(
    filepath,
    tissueframe,
    region="region",  # this can be 'region1' if you didn't manually rename your tissue region
    padding=50,
    downscale_factor=64,
    output_dir="./",
    output_fname="",
    scalebar=True,
    um_per_px=None,
    scalebar_length_px=None,
    input_format="Multichannel",
    channel_file=None,
):
    """
    Save the labelled tissue from the given image.

    Parameters
    ----------
    filepath : str
        Path to the full-resolution image file (a directory of per-channel
        TIFFs when ``input_format="Channels"``). Should be the same file/
        directory that was passed to ``downscale_tissue``.
    tissueframe : DataFrame
        The DataFrame containing the labels from the segmentation.
    region : str, optional
        The region to group by, by default "region".
    padding : int, optional
        The padding to add to the extracted tissue, by default 50.
    downscale_factor : int, optional
        The factor to downscale the image by, by default 64.
    output_dir : str, optional
        The directory to save the image in, by default "./".
    output_fname : str, optional
        The filename to save the image as, by default "".
    scalebar : bool, optional
        Whether to draw a scalebar on the preview PNG, by default True.
    um_per_px : float, optional
        Micrometres per pixel at full resolution. When provided the scalebar
        label is shown in µm; otherwise the label is in px. By default None.
    scalebar_length_px : int, optional
        Explicit scalebar length in pixels. Auto-sized (~20 %% of image width)
        when None. By default None.
    input_format : str, optional
        The input_format used (either "CODEX", "Multichannel" or "Channels"), by
        default "Multichannel". Output mirrors the input: "Multichannel" writes
        one stacked crop per tissue piece (unchanged legacy behavior); "CODEX"
        writes one 4D `(cycles, H, W, images_per_cycle)` crop per tissue piece;
        "Channels" writes a directory of per-channel cropped TIFFs per tissue
        piece. This lets you enter the pipeline here at full resolution,
        skipping the optional downscale/label steps, without being forced
        through a Multichannel conversion.
    channel_file : str, optional
        Path to a text file listing channel names, one per line, in stack
        order. Required when `input_format="CODEX"`. Ignored for
        "Multichannel" (not needed, the whole stack is cropped as-is) and for
        "Channels" (channel names are taken from the filenames). Default is
        None.

    Returns
    -------
    None
    """
    tissueframe2 = tissueframe.groupby(region).agg([min, max])
    _style = SPACEcStyle()

    image_dict = None
    channel_names = None
    number_cycles = None
    images_per_cycle = None

    if input_format == "Multichannel":
        print("Reading in the qptiff file, might take awhile!")
        currim = tifffile.imread(filepath)
        img_h, img_w = currim.shape[1], currim.shape[2]
    elif input_format == "CODEX":
        if channel_file is None:
            raise ValueError("'channel_file' is required when input_format='CODEX'.")
        with open(channel_file, "r") as f:
            channel_names = f.read().splitlines()
        print("Reading in the CODEX file, might take awhile!")
        raw_image = tifffile.imread(filepath)
        number_cycles, images_per_cycle = raw_image.shape[0], raw_image.shape[3]
        img_h, img_w = raw_image.shape[1], raw_image.shape[2]
        image_dict, channel_names = format_CODEX(
            image=raw_image,
            channel_names=channel_names,
            input_format="CODEX",
        )
        del raw_image
    elif input_format == "Channels":
        print(f"Reading per-channel TIFFs from directory: {filepath}")
        image_dict, channel_names = format_CODEX(
            image=filepath,
            channel_names=None,
            input_format="Channels",
        )
        first_channel = next(iter(image_dict.values()))
        img_h, img_w = first_channel.shape
    else:
        raise ValueError(
            f"Invalid input_format: {input_format!r}. Choose from 'Multichannel', 'CODEX', 'Channels'."
        )

    if input_format != "Multichannel" and not image_dict:
        raise ValueError(
            f"Failed to load image data from {filepath!r} (input_format={input_format!r})."
        )

    for index, row in tissueframe2.iterrows():
        ymin = row["y"]["min"] * downscale_factor
        ymax = row["y"]["max"] * downscale_factor
        xmin = row["x"]["min"] * downscale_factor
        xmax = row["x"]["max"] * downscale_factor
        ymin = max(ymin - padding, 0)
        ymax = min(ymax + padding, img_h)
        xmin = max(xmin - padding, 0)
        xmax = min(xmax + padding, img_w)

        if input_format == "Multichannel":
            subim = currim[:, ymin:ymax, xmin:xmax]
            preview_im = subim[0]
            img_width_px = subim.shape[2]
        else:
            cropped = {
                name: arr[ymin:ymax, xmin:xmax] for name, arr in image_dict.items()
            }
            preview_im = cropped[channel_names[0]]
            img_width_px = preview_im.shape[1]

        fig, ax = plt.subplots()
        ax.imshow(preview_im)
        ax.set_title(f"Extracting tissue {index}: ")

        if scalebar:
            bar_px = (
                scalebar_length_px
                if scalebar_length_px is not None
                else _nice_scalebar_length(img_width_px)
            )
            _add_scalebar(
                ax, img_width_px, _style, scalebar_length_px=bar_px, um_per_px=um_per_px
            )

            png_outpath = os.path.join(
                output_dir, output_fname, f"reg00{index}_X01_Y01_Z01_preview.png"
            )
            fig.savefig(png_outpath, bbox_inches="tight", dpi=150)
            print(f"Saving preview with scalebar at {png_outpath}")

        plt.show()

        if input_format == "Multichannel":
            outpath = os.path.join(
                output_dir, output_fname, f"reg00{index}_X01_Y01_Z01.tif"
            )
            print(f"Saving tissue image at {outpath}")
            tifffile.imwrite(outpath, subim)
        elif input_format == "CODEX":
            cropped_h, cropped_w = preview_im.shape
            reconstructed = np.empty(
                (number_cycles, cropped_h, cropped_w, images_per_cycle),
                dtype=preview_im.dtype,
            )
            idx = 0
            for i in range(number_cycles):
                for n in range(images_per_cycle):
                    if idx >= len(channel_names):
                        break
                    reconstructed[i, :, :, n] = cropped[channel_names[idx]]
                    idx += 1
            outpath = os.path.join(
                output_dir, output_fname, f"reg00{index}_X01_Y01_Z01.tif"
            )
            print(f"Saving tissue image at {outpath}")
            tifffile.imwrite(outpath, reconstructed)
        else:  # Channels
            outdir = os.path.join(output_dir, output_fname, f"reg00{index}_X01_Y01_Z01")
            os.makedirs(outdir, exist_ok=True)
            print(f"Saving per-channel tissue images at {outdir}")
            for name, arr in cropped.items():
                tifffile.imwrite(os.path.join(outdir, f"{name}.tif"), arr)

        plt.close(fig)
