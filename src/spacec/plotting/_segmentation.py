import os
import pathlib
import random
import warnings
from typing import Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import skimage.io

from .._shared.overlay_utils import create_rgb_image, make_outline_overlay
from .._shared.segmentation import combine_channels, format_CODEX
from ._style import SPACEcStyle, _add_scalebar, _apply_style, _savefig


def segmentation_ch(
    file_name,  # image for segmentation
    channel_file,  # all channels used for staining
    output_dir,
    savefig=False,
    output_fname="",
    membrane_channel_list=None,  # channels used for membrane segmentation
    nuclei_channel="DAPI",
    input_format="Multichannel",
    style: Optional[Union[SPACEcStyle, str, pathlib.Path]] = None,
    scalebar: bool = True,
    scalebar_length_px=None,
    um_per_px=None,
    scalebar_show_text: bool = True,
    extra_seg_ch_list=None,
):
    """
    Plot the channel selected for segmentation.

    Parameters
    ----------
    file_name : str
        The path to the image file for segmentation.
    channel_file : str
        The path to the file containing all channels used for staining.
    output_dir : str
        The directory to save the output in.
    savefig : bool, optional
        Whether to save the figure, by default False.
    output_fname : str, optional
        The filename for the saved figure, by default "".
    membrane_channel_list : list, optional
        The channels used for membrane segmentation, by default None.
    nuclei_channel : str, optional
        The channel used for nuclei, by default "DAPI".
    input_format : str, optional
        The input_format used (either "CODEX", "Multichannel" or "Channels"), by default "Multichannel".
    extra_seg_ch_list : list, optional
        Deprecated alias for `membrane_channel_list`, kept only for backwards compatibility -- raises a
        `DeprecationWarning` and overrides `membrane_channel_list` if given. Will be removed in a future
        version.

    Returns
    -------
    None
    """
    if extra_seg_ch_list is not None:
        warnings.warn(
            "'extra_seg_ch_list' is deprecated and will be removed in a future version. "
            "Use 'membrane_channel_list' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        membrane_channel_list = extra_seg_ch_list

    _style = _apply_style(style)

    if input_format != "Channels":
        # Load the image
        img = skimage.io.imread(file_name)
        # Read channels and store as list
        with open(channel_file, "r") as f:
            channel_names = f.read().splitlines()
        # Function reads channels and stores them as a dictionary and returns the processed channel names
        image_dict, _ = format_CODEX(
            image=img,
            channel_names=channel_names,
            input_format=input_format,
        )
    else:
        # In Channels format, file_name is the directory containing individual channel files
        image_dict, _ = format_CODEX(
            image=file_name,
            channel_names=None,
            input_format=input_format,
        )

    if image_dict is None:
        print("Error: Failed to format image data")
        return

    # Check if nuclei_channel exists
    if nuclei_channel not in image_dict:
        print(f"Warning: Nuclei channel '{nuclei_channel}' not found in image data.")
        print(f"Available channels: {list(image_dict.keys())}")
        return

    # Combine channels for segmentation - FIX: only expect a single return value
    image_dict = combine_channels(
        image_dict, membrane_channel_list, new_channel_name="segmentation_channel"
    )

    # Check if segmentation channel was created
    if "segmentation_channel" not in image_dict:
        print("Error: Failed to create segmentation channel")
        return

    fig, ax = plt.subplots(1, 2, figsize=(15, 15))
    ax[0].imshow(image_dict[nuclei_channel])
    ax[1].imshow(image_dict["segmentation_channel"])
    ax[0].set_title("nuclei", fontsize=_style.font_size_medium)
    ax[1].set_title("membrane", fontsize=_style.font_size_medium)

    if scalebar:
        image_width_px = image_dict[nuclei_channel].shape[1]
        _add_scalebar(
            ax[0],
            image_width_px,
            _style,
            scalebar_length_px=scalebar_length_px,
            um_per_px=um_per_px,
            show_text=scalebar_show_text,
        )
        _add_scalebar(
            ax[1],
            image_width_px,
            _style,
            scalebar_length_px=scalebar_length_px,
            um_per_px=um_per_px,
            show_text=scalebar_show_text,
        )

    if savefig:
        _savefig(fig, output_dir, output_fname, "", _style)
        print(
            f"Saved figure to: {pathlib.Path(output_dir) / (output_fname + '.' + _style.save_format)}"
        )
    else:
        plt.show()


def show_masks(
    seg_output,
    nucleus_channel,
    membrane_channel_list=None,
    show_subsample=True,
    n=2,  # need to be at least 2
    tilesize=100,
    idx=0,
    rand_seed=1,
    style: Optional[Union[SPACEcStyle, str, pathlib.Path]] = None,
    scalebar: bool = True,
    scalebar_length_px=None,
    um_per_px=None,
    scalebar_show_text: bool = True,
    additional_channels=None,
):
    """
    Visualize the segmentation results of an image.

    Parameters
    ----------
    seg_output : dict
        The output from the segmentation process. It should contain 'image_dict' and 'masks'.
    nucleus_channel : str
        The name of the nucleus channel in the image_dict.
    membrane_channel_list : list of str, optional
        The names of membrane/additional channels to be combined with the nucleus channel for visualization.
    additional_channels : list, optional
        Deprecated alias for `membrane_channel_list`, kept only for backwards compatibility -- raises a
        `DeprecationWarning` and overrides `membrane_channel_list` if given. Will be removed in a future
        version.
    show_subsample : bool, optional
        Whether to show a subsample of the image. Default is True.
    n : int, optional
        The number of subsamples to show. Default is 2.
    tilesize : int, optional
        The size of the tiles for subsampling. Default is 100.
    idx : int, optional
        The index for displaying. Default is 0.
    rand_seed : int, optional
        The seed for the random number generator. Default is 1.

    Returns
    -------
    overlay_data : ndarray
        The overlay of the segmentation results on the RGB images.
    rgb_images : ndarray
        The RGB images.

    Raises
    ------
    ValueError
        If the image size is smaller than the tile size or if there are not enough tiles to display.

    """
    if additional_channels is not None:
        warnings.warn(
            "'additional_channels' is deprecated and will be removed in a future version. "
            "Use 'membrane_channel_list' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        membrane_channel_list = additional_channels

    _style = _apply_style(style)
    image_dict = seg_output["image_dict"]
    masks = seg_output["masks"]

    # Create a combined image stack
    # Assumes nuclei_image and membrane_image are numpy arrays of the same shape
    if len(masks.shape) == 2:
        masks = np.expand_dims(masks, axis=0)
        masks = np.expand_dims(masks, axis=0)
        masks = np.moveaxis(masks, 0, -1)

    if membrane_channel_list is not None:
        image_dict = combine_channels(
            image_dict, membrane_channel_list, new_channel_name="segmentation_channel"
        )
        nuclei_image = image_dict[nucleus_channel]
        add_chan_image = image_dict["segmentation_channel"]
        combined_image = np.stack([nuclei_image, add_chan_image], axis=-1)
        # Add an extra dimension for batch-style visualization utilities
        # Changes shape from (height, width, channels) to (1, height, width, channels)
        combined_image = np.expand_dims(combined_image, axis=0)
        # create rgb overlay of image data for visualization
        rgb_images = create_rgb_image(combined_image, channel_colors=["blue", "green"])
    else:
        nuclei_image = image_dict[nucleus_channel]
        combined_image = np.stack([nuclei_image], axis=-1)
        # Add an extra dimension for batch-style visualization utilities
        # Changes shape from (height, width, channels) to (1, height, width, channels)
        combined_image = np.expand_dims(combined_image, axis=0)
        # create rgb overlay of image data for visualization
        rgb_images = create_rgb_image(combined_image, channel_colors=["blue"])

    # create overlay of segmentation results
    overlay_data = make_outline_overlay(rgb_data=rgb_images, predictions=masks)

    # select index for displaying

    # plot the data
    fig, ax = plt.subplots(1, 2, figsize=(15, 15))
    ax[0].imshow(rgb_images[idx, ...])
    ax[1].imshow(overlay_data[idx, ...])
    ax[0].set_title("Raw data", fontsize=_style.font_size_medium)
    ax[1].set_title("Predictions", fontsize=_style.font_size_medium)

    if scalebar:
        image_width_px = rgb_images[idx].shape[1]
        _add_scalebar(
            ax[0],
            image_width_px,
            _style,
            scalebar_length_px=scalebar_length_px,
            um_per_px=um_per_px,
            show_text=scalebar_show_text,
        )
        _add_scalebar(
            ax[1],
            image_width_px,
            _style,
            scalebar_length_px=scalebar_length_px,
            um_per_px=um_per_px,
            show_text=scalebar_show_text,
        )

    plt.show()

    random.seed(rand_seed)
    if show_subsample:
        overlay_data = np.squeeze(overlay_data, axis=0)
        rgb_images = np.squeeze(rgb_images, axis=0)

        # Ensure the sizes are compatible for tile calculation
        if overlay_data.shape[0] < tilesize or overlay_data.shape[1] < tilesize:
            print("Image size is smaller than the tile size. Cannot display tiles.")
        else:
            # Calculate the number of tiles in x and y directions
            y_tiles, x_tiles = (
                overlay_data.shape[0] // tilesize,
                overlay_data.shape[1] // tilesize,
            )

            # Check if either x_tiles or y_tiles is zero
            if x_tiles == 0 or y_tiles == 0:
                print("Not enough tiles to display.")
            else:
                # Split images into tiles
                overlay_tiles = []
                grayscale_tiles = []
                for i in range(x_tiles):
                    for j in range(y_tiles):
                        x_start, y_start = i * tilesize, j * tilesize
                        overlay_tile = overlay_data[
                            y_start : y_start + tilesize, x_start : x_start + tilesize
                        ]
                        image_tile = rgb_images[
                            y_start : y_start + tilesize, x_start : x_start + tilesize
                        ]

                        overlay_tiles.append(overlay_tile)
                        grayscale_tiles.append(image_tile)

                # Randomly select n tiles
                random_indices = random.sample(range(len(overlay_tiles)), n)

                # Plot the tiles
                fig, axs = plt.subplots(n, 2, figsize=(10, 5 * n))
                for i, idx in enumerate(random_indices):
                    axs[i, 0].imshow(grayscale_tiles[idx])
                    axs[i, 0].axis("off")
                    axs[i, 1].imshow(overlay_tiles[idx])
                    axs[i, 1].axis("off")

                    axs[i, 0].set_title("Raw data", fontsize=_style.font_size_medium)
                    axs[i, 1].set_title("Predictions", fontsize=_style.font_size_medium)
                plt.tight_layout()
                plt.show()

    return overlay_data, rgb_images
