import os
import pathlib
import shutil
import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import skimage.io
import tensorflow as tf
from cellpose import io, models
from deepcell.applications import Mesmer
from deepcell.utils.plot_utils import create_rgb_image, make_outline_overlay
from skimage.measure import regionprops_table
from tensorflow.keras.models import load_model
from tqdm import tqdm

from .._shared.segmentation import (
    combine_channels,
    create_multichannel_tiff,
    format_CODEX,
)


def cell_segmentation(
    file_name,
    channel_file,
    output_dir,
    output_fname="",
    seg_method="mesmer",
    nuclei_channel="DAPI",
    input_format="Multichannel",  # CODEX or Multichannel --> This depends on the machine you are using and the resulting file format (see documentation above)
    membrane_channel_list=None,
    size_cutoff=0,  # quantificaition
    compartment="whole-cell",  # mesmer # segment whole cells or nuclei only
    plot_predictions=True,  # mesmer # plot segmentation results
    model="cyto3",  # cellpose
    use_gpu=True,  # cellpose
    cytoplasm_channel_list=None,  # celpose
    diameter=None,  # cellpose
    save_mask_as_png=False,  # cellpose
    model_path="./models",
    resize_factor=1,
    custom_model=False,
    differentiate_nucleus_cytoplasm=False,  # experimental
):
    """
    Perform cell segmentation on an image.
    Parameters
    ----------
    file_name : str
        The path to the image file.
    channel_file : str
        The path to the file containing the channel names.
    output_dir : str
        The directory where the output will be saved.
    output_fname : str, optional
        The name of the output file. Default is an empty string.
    seg_method : str
        The segmentation method to use. Options are 'mesmer' and 'cellpose'. Default is 'mesmer'.
    nuclei_channel : str
        The name of the nuclei channel. Default is 'DAPI'.
    input_format : str
        The input_format used to generate the image. Options are 'CODEX' and 'Multichannel'. Default is 'Multichannel'.
    membrane_channel_list : list of str, optional
        The names of the membrane channels.
    size_cutoff : int, optional
        The size cutoff for segmentation. Default is 0.
    compartment : str, optional
        The compartment to segment. Options are 'whole-cell' and 'nuclei'. Default is 'whole-cell'. This only applies to Mesmer.
    plot_predictions : bool, optional
        Whether to plot the segmentation results. Default is True.
    model : str, optional
        The model to use for segmentation. Default is 'tissuenet'. This only applies to Cellpose.
    use_gpu : bool, optional
        Whether to use GPU for segmentation. Default is True. This only applies to Cellpose.
    cytoplasm_channel_list : list of str, optional
        The names of the cytoplasm channels.
    diameter : int, optional
        The diameter of the cells. Default is None - if set to None the diameter is automatically defined. This only applies to Cellpose.
    save_mask_as_png : bool, optional
        Whether to save the segmentation mask as a PNG file. Default is False.
    model_path : str, optional
        The path to the model. Default is './models'.
    differentiate_nucleus_cytoplasm : bool, optional
        Whether to differentiate between nucleus and cytoplasm. Default is False.
    Returns
    -------
    dict
        A dictionary containing the original image ('img'), the segmentation masks ('masks'), and the image dictionary ('image_dict').
    """
    if use_gpu == True:
        gpus = tf.config.list_physical_devices("GPU")

        if gpus:
            print("GPU(s) available")
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)

    print("Create image channels!")

    # check input format
    if input_format not in ["CODEX", "Multichannel", "Channels"]:
        sys.exit(
            "Please provide a valid input format (Multichannel, Channels or CODEX)!"
        )

    if input_format != "Channels":
        # Load the image
        img = skimage.io.imread(file_name)
        # Read channels and store as list
        with open(channel_file, "r") as f:
            channel_names = f.read().splitlines()
        # Function reads channels and stores them as a dictionary (storing as a dictionary allows to select specific channels by name)
        image_dict = format_CODEX(
            image=img,
            channel_names=channel_names,  # file with list of channel names (see channelnames.txt)
            input_format=input_format,
        )
    else:
        image_dict = format_CODEX(
            image=file_name,
            channel_names=None,  # file with list of channel names (see channelnames.txt)
            input_format=input_format,
        )
        channel_names = list(image_dict.keys())
    # Generate image for segmentation
    if membrane_channel_list is not None:
        image_dict = combine_channels(
            image_dict, membrane_channel_list, new_channel_name="segmentation_channel"
        )  # combine channels for better segmentation. In this example we combine channels to get a membrane outline for all cells in the image
    segmentation_channels = [
        nuclei_channel,
        "segmentation_channel",
    ]  # replace with your actual channel names
    segmentation_image_dict = {
        channel: image_dict[channel]
        for channel in segmentation_channels
        if channel in image_dict
    }
    # Iterate over the dictionary
    for channel, img in segmentation_image_dict.items():
        # Calculate the new dimensions
        new_width = int(img.shape[1] * resize_factor)
        new_height = int(img.shape[0] * resize_factor)
        # Resize the image
        resized_img = cv2.resize(
            img, (new_width, new_height), interpolation=cv2.INTER_AREA
        )
        # Replace the original image with the resized image in the dictionary
        segmentation_image_dict[channel] = resized_img

    if differentiate_nucleus_cytoplasm == True:
        if membrane_channel_list == None:
            print(
                "Provide membrane channel for differentiation between nucleus and cytoplasm"
            )
            return
        else:
            if seg_method == "mesmer":
                print("Segmenting with Mesmer!")

                masks_nuclei = mesmer_segmentation(
                    nuclei_image=segmentation_image_dict[nuclei_channel],
                    membrane_image=None,
                    plot_predictions=plot_predictions,  # plot segmentation results
                    compartment="nuclear",
                    model_path=model_path,
                )  # segment whole cells or nuclei only

                masks_whole_cell = mesmer_segmentation(
                    nuclei_image=segmentation_image_dict[nuclei_channel],
                    membrane_image=segmentation_image_dict["segmentation_channel"],
                    plot_predictions=plot_predictions,  # plot segmentation results
                    compartment=compartment,
                    model_path=model_path,
                )  # segment whole cells or nuclei only
            else:
                print("Segmenting with Cellpose!")

                (
                    masks_nuclei,
                    flows,
                    styles,
                    input_image,
                    rgb_channels,
                ) = cellpose_segmentation(
                    image_dict=segmentation_image_dict,
                    output_dir=output_dir,
                    membrane_channel=None,
                    cytoplasm_channel=cytoplasm_channel_list,
                    nucleus_channel=nuclei_channel,
                    use_gpu=use_gpu,
                    model=model,
                    custom_model=custom_model,
                    diameter=diameter,
                    save_mask_as_png=save_mask_as_png,
                )

                (
                    masks_whole_cell,
                    flows,
                    styles,
                    input_image,
                    rgb_channels,
                ) = cellpose_segmentation(
                    image_dict=segmentation_image_dict,
                    output_dir=output_dir,
                    membrane_channel="segmentation_channel",
                    cytoplasm_channel=cytoplasm_channel_list,
                    nucleus_channel=nuclei_channel,
                    use_gpu=use_gpu,
                    model=model,
                    custom_model=custom_model,
                    diameter=diameter,
                    save_mask_as_png=save_mask_as_png,
                )

            # Remove single-dimensional entries from the shape of segmentation_masks
            masks_whole_cell = masks_whole_cell.squeeze()
            # Get the original dimensions of any one of the images
            original_height, original_width = image_dict[
                nuclei_channel
            ].shape  # or any other channel
            # Resize the masks back to the original size
            masks_whole_cell = cv2.resize(
                masks_whole_cell,
                (original_width, original_height),
                interpolation=cv2.INTER_NEAREST,
            )

            # Remove single-dimensional entries from the shape of segmentation_masks
            masks_nuclei = masks_nuclei.squeeze()
            # Get the original dimensions of any one of the images
            original_height, original_width = image_dict[
                nuclei_channel
            ].shape  # or any other channel
            # Resize the masks back to the original size
            masks_nuclei = cv2.resize(
                masks_nuclei,
                (original_width, original_height),
                interpolation=cv2.INTER_NEAREST,
            )

            # Create binary masks
            binary_masks_nuclei = masks_nuclei > 0
            binary_masks_whole_cell = masks_whole_cell > 0

            # Subtract the binary nuclei mask from the binary whole cell mask
            binary_masks_cytoplasm = binary_masks_whole_cell & ~binary_masks_nuclei

            # Now, if you want to get a labeled mask for the cytoplasm, you can use a function like `label` from `scipy.ndimage`
            from scipy.ndimage import label

            masks_cytoplasm, num_labels = label(binary_masks_cytoplasm)

            print("Quantifying features after segmentation!")
            print("Quantifying features nuclei")
            nuc = extract_features(
                image_dict=image_dict,  # image dictionary
                segmentation_masks=masks_nuclei,  # segmentation masks generated by cellpose
                channels_to_quantify=channel_names,  # list of channels to quantify (here: all channels)
                output_file=pathlib.Path(output_dir)
                / (
                    output_fname + "_" + seg_method + "_nuclei_result.csv"
                ),  # output path to store results as csv
                size_cutoff=size_cutoff,
            )  # size cutoff for segmentation masks (default = 0)
            print("Quantifying features cytoplasm")
            cyto = extract_features(
                image_dict=image_dict,  # image dictionary
                segmentation_masks=masks_cytoplasm,  # segmentation masks generated by cellpose
                channels_to_quantify=channel_names,  # list of channels to quantify (here: all channels)
                output_file=pathlib.Path(output_dir)
                / (
                    output_fname + "_" + seg_method + "_cytoplasm_result.csv"
                ),  # output path to store results as csv
                size_cutoff=size_cutoff,
            )  # size cutoff for segmentation masks (default = 0)

            print("Quantifying features whole cell")
            whole = extract_features(
                image_dict=image_dict,  # image dictionary
                segmentation_masks=masks_whole_cell,  # segmentation masks generated by cellpose
                channels_to_quantify=channel_names,  # list of channels to quantify (here: all channels)
                output_file=pathlib.Path(output_dir)
                / (
                    output_fname + "_" + seg_method + "_whole_cell_result.csv"
                ),  # output path to store results as csv
                size_cutoff=size_cutoff,
            )  # size cutoff for segmentation masks (default = 0)
            print("Done!")

            # remove
            out = [
                "x",
                "y",
                "eccentricity",
                "perimeter",
                "convex_area",
                "area",
                "axis_major_length",
                "axis_minor_length",
                "label",
            ]

            # keep metadata
            whole_meta = whole[out]

            # remove from nuc
            nuc = nuc.drop(out, axis=1)
            # add whole metadata to cyto
            nuc_save = pd.concat([nuc, whole_meta], axis=1)
            nuc_save.to_csv(
                output_dir
                + output_fname
                + "_"
                + seg_method
                + "_nuclei_intensities_result.csv"
            )
            # remove from cyto
            cyto = cyto.drop(out, axis=1)
            # add whole metadata to cyto
            cyto_save = pd.concat([cyto, whole_meta], axis=1)
            cyto_save.to_csv(
                output_dir
                + output_fname
                + "_"
                + seg_method
                + "_cytoplasm_intensities_result.csv"
            )

            whole.to_csv(
                output_dir
                + output_fname
                + "_"
                + seg_method
                + "_whole_cell_intensities_result.csv"
            )
            whole = whole.drop(out, axis=1)

            # add identifier to each column name
            nuc.columns = [str(col) + "_nuc" for col in nuc.columns]
            cyto.columns = [str(col) + "_cyto" for col in cyto.columns]
            whole.columns = [str(col) + "_whole" for col in whole.columns]

            # combine the dataframes and save as csv
            result = pd.concat([nuc, cyto, whole, whole_meta], axis=1)
            result = result.loc[:, ~result.columns.str.contains("Unnamed: 0")]
            result.to_csv(
                output_dir
                + output_fname
                + "_"
                + seg_method
                + "_segmentation_results_combined.csv"
            )

            return {
                "img": img,
                "masks": masks_whole_cell,
                "image_dict": image_dict,
                "masks_cytoplasm": masks_cytoplasm,
                "masks_nuclei": masks_nuclei,
            }

    else:
        if seg_method == "mesmer":
            print("Segmenting with Mesmer!")
            if membrane_channel_list is None:
                masks = mesmer_segmentation(
                    nuclei_image=segmentation_image_dict[nuclei_channel],
                    membrane_image=None,
                    plot_predictions=plot_predictions,  # plot segmentation results
                    compartment="nuclear",
                    model_path=model_path,
                )  # segment whole cells or nuclei only
            else:
                masks = mesmer_segmentation(
                    nuclei_image=segmentation_image_dict[nuclei_channel],
                    membrane_image=segmentation_image_dict["segmentation_channel"],
                    plot_predictions=plot_predictions,  # plot segmentation results
                    compartment=compartment,
                    model_path=model_path,
                )  # segment whole cells or nuclei only
        else:
            print("Segmenting with Cellpose!")
            if membrane_channel_list is None:
                masks, flows, styles, input_image, rgb_channels = cellpose_segmentation(
                    image_dict=segmentation_image_dict,
                    output_dir=output_dir,
                    membrane_channel=None,
                    cytoplasm_channel=cytoplasm_channel_list,
                    nucleus_channel=nuclei_channel,
                    use_gpu=use_gpu,
                    model=model,
                    custom_model=custom_model,
                    diameter=diameter,
                    save_mask_as_png=save_mask_as_png,
                )
            else:
                masks, flows, styles, input_image, rgb_channels = cellpose_segmentation(
                    image_dict=segmentation_image_dict,
                    output_dir=output_dir,
                    membrane_channel="segmentation_channel",
                    cytoplasm_channel=cytoplasm_channel_list,
                    nucleus_channel=nuclei_channel,
                    use_gpu=use_gpu,
                    model=model,
                    custom_model=custom_model,
                    diameter=diameter,
                    save_mask_as_png=save_mask_as_png,
                )
        # Remove single-dimensional entries from the shape of segmentation_masks
        masks = masks.squeeze()
        # Get the original dimensions of any one of the images
        original_height, original_width = image_dict[
            nuclei_channel
        ].shape  # or any other channel
        # Resize the masks back to the original size
        masks = cv2.resize(
            masks, (original_width, original_height), interpolation=cv2.INTER_NEAREST
        )
        print("Quantifying features after segmentation!")
        extract_features(
            image_dict=image_dict,  # image dictionary
            segmentation_masks=masks,  # segmentation masks generated by cellpose
            channels_to_quantify=channel_names,  # list of channels to quantify (here: all channels)
            output_file=pathlib.Path(output_dir)
            / (
                output_fname + "_" + seg_method + "_result.csv"
            ),  # output path to store results as csv
            size_cutoff=size_cutoff,
        )  # size cutoff for segmentation masks (default = 0)
        print("Done!")
        return {"img": img, "masks": masks, "image_dict": image_dict}


def extract_features(
    image_dict, segmentation_masks, channels_to_quantify, output_file, size_cutoff=0
):
    """
    Extract features from the given image dictionary and segmentation masks.

    Parameters
    ----------
    image_dict : dict
        Dictionary containing image data. Keys are channel names, values are 2D numpy arrays.
    segmentation_masks : ndarray
        2D numpy array containing segmentation masks.
    channels_to_quantify : list
        List of channel names to quantify.
    output_file : str
        Path to the output CSV file.
    size_cutoff : int, optional
        Minimum size of nucleus to consider. Nuclei smaller than this are ignored. Default is 0.

    Returns
    -------
    None
        The function doesn't return anything but writes the extracted features to a CSV file.

    """
    segmentation_masks = segmentation_masks.squeeze()

    # Count pixels for each nucleus
    _, counts = np.unique(segmentation_masks, return_counts=True)

    # Identify nucleus IDs above the size cutoff, excluding background (ID 0)
    nucleus_ids = np.where(counts > size_cutoff)[0][1:]

    # Filter out small objects from segmentation masks
    filterimg = np.where(
        np.isin(segmentation_masks, nucleus_ids), segmentation_masks, 0
    )

    # Extract morphological features
    props = regionprops_table(
        filterimg,
        properties=(
            "centroid",
            "eccentricity",
            "perimeter",
            "convex_area",
            "area",
            "axis_major_length",
            "axis_minor_length",
            "label",
        ),
    )
    props_df = pd.DataFrame(props)
    props_df.set_index(props_df["label"], inplace=True)

    # Pre-allocate array for mean intensities
    mean_intensities = np.empty((len(nucleus_ids), len(channels_to_quantify)))

    # For each channel, compute mean intensities for all labels using vectorized operations
    for idx, chan in enumerate(tqdm(channels_to_quantify, desc="Processing channels")):
        chan_data = image_dict[chan]
        labels_matrix = (
            np.isin(segmentation_masks, nucleus_ids).astype(int) * segmentation_masks
        )
        sum_per_label = np.bincount(labels_matrix.ravel(), weights=chan_data.ravel())[
            nucleus_ids
        ]
        count_per_label = np.bincount(labels_matrix.ravel())[nucleus_ids]
        mean_intensities[:, idx] = sum_per_label / count_per_label

    # Convert the array to a DataFrame
    mean_df = pd.DataFrame(
        mean_intensities, index=nucleus_ids, columns=channels_to_quantify
    )

    # Join with morphological features
    markers = mean_df.join(props_df)

    # rename column
    markers.rename(columns={"centroid-0": "y"}, inplace=True)
    markers.rename(columns={"centroid-1": "x"}, inplace=True)

    # Export to CSV
    markers.to_csv(output_file)

    return markers


def cellpose_segmentation(
    image_dict,
    output_dir,
    membrane_channel=None,
    cytoplasm_channel=None,
    nucleus_channel=None,
    use_gpu=True,
    model="cyto3",
    custom_model=False,
    diameter=None,
    save_mask_as_png=False,
):
    """
    Perform cell segmentation using CellPose.

    Parameters
    ----------
    image_dict : dict
        Dictionary containing image data. Keys are channel names, values are 2D numpy arrays.
    output_dir : str
        Directory where the output will be saved.
    membrane_channel : str, optional
        Name of the membrane channel. Default is None.
    cytoplasm_channel : str, optional
        Name of the cytoplasm channel. Default is None.
    nucleus_channel : str
        Name of the nucleus channel. Default is None.
    use_gpu : bool, optional
        Whether to use GPU for computation. Default is True.
    model : str, optional
        Model to use for segmentation. Default is "nuclei".
    pretrained_model : bool, optional
        Whether to use a pretrained model. Default is False.
    diameter : float, optional
        Diameter of cells. If None, it will be estimated automatically. Default is None. However, it is recommended to provide the diameter for better results.
    save_mask_as_png : bool, optional
        Whether to save the mask as a PNG file. Default is False.

    Returns
    -------
    masks : ndarray
        2D numpy array containing segmentation masks.
    flows : ndarray
        2D numpy array containing flows.
    styles : ndarray
        2D numpy array containing styles.
    input_image : ndarray
        2D numpy array containing the input image.
    rgb_channels : list
        List of RGB channels used for segmentation.

    """
    # Default values for rgb_channels
    rgb_channels = [0, 0]
    grey_channel = None

    # Check nucleus channel
    if not nucleus_channel:
        print("Please select a nucleus channel!")
        return

    # Channel selection logic
    if cytoplasm_channel:
        if membrane_channel:
            print(
                "CellPose expects only two channels as input. Selecting nucleus channel and cytoplasm channel. If you want to use nucleus and membrane instead set cytoplasm_channel to None."
            )
        else:
            print("Selecting nucleus and cytoplasm channel for segmentation.")
        rgb_channels = [2, 3]
    elif membrane_channel:
        print("Selecting nucleus and membrane channel for segmentation.")
        rgb_channels = [1, 3]

    # Run CellPose
    if membrane_channel is None and cytoplasm_channel is None:
        input_image = RGB_for_segmentation(
            membrane_channel, cytoplasm_channel, nucleus_channel, image_dict
        )
        # extract only the blue channel
        final_input = input_image[:, :, 2]

    else:
        input_image = RGB_for_segmentation(
            membrane_channel, cytoplasm_channel, nucleus_channel, image_dict
        )
        final_input = input_image.copy()

    masks, flows, styles = run_cellpose(
        final_input,
        output_dir=output_dir,
        use_gpu=use_gpu,
        model=model,
        custom_model=custom_model,
        diameter=diameter,
        rgb_channels=rgb_channels,
        save_mask_as_png=save_mask_as_png,
    )

    return masks, flows, styles, input_image, rgb_channels


def RGB_for_segmentation(
    membrane_channel, cytoplasm_channel, nucleus_channel, image_dict
):
    """
    Prepare an RGB image for segmentation from the given channels.

    Parameters
    ----------
    membrane_channel : str
        Name of the membrane channel.
    cytoplasm_channel : str
        Name of the cytoplasm channel.
    nucleus_channel : str
        Name of the nucleus channel.
    image_dict : dict
        Dictionary containing image data. Keys are channel names, values are 2D numpy arrays.

    Returns
    -------
    rgb_image : ndarray or None
        3D numpy array containing the RGB image for segmentation. If no valid channels are found, returns None.

    """
    # Initialize channels with None
    red_channel, green_channel, blue_channel = None, None, None

    # Check for membrane_channel
    if membrane_channel in image_dict:
        red_channel = image_dict[membrane_channel]

    # Check for cytoplasm_channel
    if cytoplasm_channel in image_dict:
        green_channel = image_dict[cytoplasm_channel]

    # Check for nucleus_channel
    if nucleus_channel in image_dict:
        blue_channel = image_dict[nucleus_channel]

    # If any of the channels is None, create an empty channel with the same shape as the others
    shape = None
    for channel in [red_channel, green_channel, blue_channel]:
        if channel is not None:
            shape = channel.shape
            break

    if red_channel is None and shape:
        red_channel = np.zeros(shape, dtype=np.uint8)

    if green_channel is None and shape:
        green_channel = np.zeros(shape, dtype=np.uint8)

    if blue_channel is None and shape:
        blue_channel = np.zeros(shape, dtype=np.uint8)

    # Stack the channels to create an RGB image
    if shape:  # Only if we found a valid shape
        rgb_image = np.dstack((red_channel, green_channel, blue_channel))
        return rgb_image
    else:
        # Handle the case where no valid channels are found
        return None


def run_cellpose(
    image,
    output_dir,
    use_gpu=True,
    model="nuclei",
    custom_model=False,
    diameter=None,
    rgb_channels=[0, 0],
    save_mask_as_png=False,
):
    """
    Run CellPose model on the given image.

    Parameters
    ----------
    image : ndarray
        2D or 3D numpy array containing the image data.
    output_dir : str
        Directory where the output will be saved.
    use_gpu : bool, optional
        Whether to use GPU for computation. Default is True.
    model : str, optional
        Model to use for segmentation. Default is "nuclei".
    pretrained_model : bool, optional
        Whether to use a pretrained model. Default is False.
    diameter : float, optional
        Diameter of cells. If None, it will be estimated automatically. Default is None.
    rgb_channels : list, optional
        List of RGB channels used for segmentation. Default is [0, 0].
    save_mask_as_png : bool, optional
        Whether to save the mask as a PNG file. Default is False.

    Returns
    -------
    masks : ndarray
        2D numpy array containing segmentation masks.
    flows : ndarray
        2D numpy array containing flows.
    styles : ndarray
        2D numpy array containing styles.

    """
    # IF ALL YOUR IMAGES ARE THE SAME TYPE, you can give a list with 2 elements
    # channels = [0,0] # IF YOU HAVE GRAYSCALE
    # channels = [2,3] # IF YOU HAVE G=cytoplasm and B=nucleus
    # channels = [2,1] # IF YOU HAVE G=cytoplasm and R=nucleus

    if custom_model == True:
        model_to_use = models.CellposeModel(model_type=model, gpu=use_gpu)
    else:
        model_to_use = models.Cellpose(
            model_type=model,
            gpu=use_gpu,
        )
    masks, flows, styles = model_to_use.eval(
        image, diameter=diameter, channels=rgb_channels
    )

    if save_mask_as_png == True:
        filename = output_dir + "/segmentation.png"
        io.save_to_png(image, masks, flows, filename)

    return masks, flows, styles


def load_mesmer_model(path):
    """
    Load the Mesmer model from the given path. If the model is not found, it is downloaded.

    Parameters
    ----------
    path : str
        Path where the Mesmer model is located or will be downloaded.

    Returns
    -------
    mesmer_pretrained_model : tensorflow.python.keras.engine.training.Model
        The loaded Mesmer model.

    """
    # check if folder Mesmer_model exist
    if not os.path.exists(os.path.join(path, "Mesmer_model")):
        os.makedirs(os.path.join(path, "Mesmer_model"))

        print("downloading Mesmer model")

        # download model from https://deepcell-data.s3-us-west-1.amazonaws.com/saved-models/MultiplexSegmentation-9.tar.gz
        url = "https://deepcell-data.s3-us-west-1.amazonaws.com/saved-models/MultiplexSegmentation-9.tar.gz"
        response = requests.get(url)

        # Ensure the download was successful
        response.raise_for_status()

        # Write the content of the response to a file in the 'Mesmer_model' directory
        with open(
            os.path.join(path, "Mesmer_model/MultiplexSegmentation.tar.gz"), "wb"
        ) as f:
            f.write(response.content)

        # unpack tar file
        shutil.unpack_archive(
            os.path.join(path, "Mesmer_model/MultiplexSegmentation.tar.gz"),
            os.path.join(path, "Mesmer_model/"),
        )

        # delete the .tar.gz file
        os.remove(os.path.join(path, "Mesmer_model/MultiplexSegmentation.tar.gz"))

        # move content of folder MultiplexSegmentation to Mesmer_model
        # shutil.move(os.path.join(path, 'Mesmer_model/MultiplexSegmentation'), os.path.join(path, 'Mesmer_model/'))

        # remove empty folder MultiplexSegmentation
        # os.rmdir(os.path.join(path, 'Mesmer_model/MultiplexSegmentation'))

        print("Mesmer model downloaded and unpacked")

    model_path = os.path.join(path, "Mesmer_model/MultiplexSegmentation")

    mesmer_pretrained_model = load_model(model_path, compile=False)

    return mesmer_pretrained_model


def mesmer_segmentation(
    nuclei_image,
    membrane_image,
    image_mpp=0.5,
    plot_predictions=True,
    compartment="whole-cell",  # or 'nuclear'
    model_path="./models",
):
    """
    Perform segmentation on the given images using the Mesmer model.

    Parameters
    ----------
    nuclei_image : ndarray
        2D numpy array containing the nuclei image data.
    membrane_image : ndarray
        2D numpy array containing the membrane image data.
    image_mpp : float, optional
        Microns per pixel for the images. Default is 0.5.
    plot_predictions : bool, optional
        Whether to plot the predictions. Default is True.
    compartment : str, optional
        Compartment to segment. Can be 'whole-cell' or 'nuclear'. Default is 'whole-cell'.
    model_path : str, optional
        Path where the Mesmer model is located. Default is './models'.

    Returns
    -------
    segmented_image : ndarray
        3D numpy array containing the segmentation results.

    """
    pathlib.Path(model_path).mkdir(parents=True, exist_ok=True)
    mesmer_pretrained_model = load_mesmer_model(model_path)

    # Initialize Mesmer application
    app = Mesmer(model=mesmer_pretrained_model)

    # Create a combined image stack
    # Assumes nuclei_image and membrane_image are numpy arrays of the same shape
    if membrane_image is None:
        # generate empty membrane image
        print("No membrane image provided. Nuclear segmentation only.")
        membrane_image = np.zeros_like(nuclei_image)
        combined_image = np.stack([nuclei_image, membrane_image], axis=-1)
    else:
        combined_image = np.stack([nuclei_image, membrane_image], axis=-1)

    # Add an extra dimension to make it compatible with Mesmer's input requirements
    # Changes shape from (height, width, channels) to (1, height, width, channels)
    combined_image = np.expand_dims(combined_image, axis=0)

    # Run the Mesmer model
    segmented_image = app.predict(
        combined_image, image_mpp=image_mpp, compartment=compartment
    )

    if plot_predictions == True:
        # create rgb overlay of image data for visualization
        rgb_images = create_rgb_image(combined_image, channel_colors=["green", "blue"])
        # create overlay of segmentation results
        overlay_data = make_outline_overlay(
            rgb_data=rgb_images, predictions=segmented_image
        )

        # select index for displaying
        idx = 0

        # plot the data
        # TODO: plotting and calculating should probably not be in the same figure
        fig, ax = plt.subplots(1, 2, figsize=(15, 15))
        ax[0].imshow(rgb_images[idx, ...])
        ax[1].imshow(overlay_data[idx, ...])
        ax[0].set_title("Raw data")
        ax[1].set_title("Predictions")
        plt.show()

    # The output will be a numpy array with the segmentation results
    return segmented_image
