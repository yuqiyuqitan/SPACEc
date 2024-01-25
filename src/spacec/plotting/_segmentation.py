import random

import matplotlib.pyplot as plt
import numpy as np
import skimage.io
from deepcell.utils.plot_utils import create_rgb_image, make_outline_overlay
from skimage.measure import regionprops_table

from .._shared.segmentation import combine_channels, format_CODEX


# plot membrane channel selectd segmentation
def segmentation_ch(
    file_name,  # image for segmentation
    channel_file,  # all channels used for staining
    output_dir,  #
    extra_seg_ch_list=None,  # channels used for membrane segmentation
    nuclei_channel="DAPI",
    technology="Phenocycler",  # CODEX or Phenocycler --> This depends on the machine you are using and the resulting file format (see documentation above)
):
    # Load the image
    img = skimage.io.imread(file_name)

    # Read channels and store as list
    with open(channel_file, "r") as f:
        channel_names = f.read().splitlines()

    # Function reads channels and stores them as dictonary
    # (storing as dictionary allows to select specific channels by name)
    image_dict = format_CODEX(
        image=img, channel_names=channel_names, technology=technology
    )

    image_dict = combine_channels(
        image_dict, extra_seg_ch_list, new_channel_name="segmentation_channel"
    )

    fig, ax = plt.subplots(1, 2, figsize=(15, 15))
    ax[0].imshow(image_dict[nuclei_channel])
    ax[1].imshow(image_dict["segmentation_channel"])
    ax[0].set_title("nuclei")
    ax[1].set_title("membrane")
    plt.show()


def show_masks(
    seg_output,
    nucleus_channel,
    additional_channels=None,
    show_subsample=True,
    n=2,  # need to be at least 2
    tilesize=100,
    idx=0,
    rand_seed=1,
):
    image_dict = seg_output["image_dict"]
    masks = seg_output["masks"]

    # Create a combined image stack
    # Assumes nuclei_image and membrane_image are numpy arrays of the same shape
    if len(masks.shape) == 2:
        masks = np.expand_dims(masks, axis=0)
        masks = np.expand_dims(masks, axis=0)
        masks = np.moveaxis(masks, 0, -1)

    if additional_channels != None:
        image_dict = combine_channels(
            image_dict, additional_channels, new_channel_name="segmentation_channel"
        )
        nuclei_image = image_dict[nucleus_channel]
        add_chan_image = image_dict["segmentation_channel"]
        combined_image = np.stack([nuclei_image, add_chan_image], axis=-1)
        # Add an extra dimension to make it compatible with Mesmer's input requirements
        # Changes shape from (height, width, channels) to (1, height, width, channels)
        combined_image = np.expand_dims(combined_image, axis=0)
        # create rgb overlay of image data for visualization
        rgb_images = create_rgb_image(combined_image, channel_colors=["green", "blue"])
    else:
        nuclei_image = image_dict[nucleus_channel]
        combined_image = np.stack([nuclei_image], axis=-1)
        # Add an extra dimension to make it compatible with Mesmer's input requirements
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
    ax[0].set_title("Raw data")
    ax[1].set_title("Predictions")
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

                    axs[i, 0].set_title("Raw data")
                    axs[i, 1].set_title("Predictions")
                plt.tight_layout()
                plt.show()

    return overlay_data, rgb_images
