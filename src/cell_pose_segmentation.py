import skimage.io
import numpy as np
import time, os, sys
from urllib.parse import urlparse
import matplotlib.pyplot as plt
import matplotlib as mpl

from cellpose import utils, io, core, models, metrics, plot
import pandas as pd
from skimage.measure import regionprops_table
from tqdm import tqdm
import os, shutil
from glob import glob
import random

def optimized_channels_to_list(image, channel_names, number_cycles, images_per_cycle, show_plots=False, stack=True):
    total_images = number_cycles * images_per_cycle
    image_list = [None] * total_images  # pre-allocated list
    image_dict = {}

    index = 0
    for i in range(number_cycles):
        cycle = image[i, :, :, :]

        for n in range(images_per_cycle):
            chan = i * 4 + n
            c = channel_names[chan]
            image2 = cycle[:, :, n]

            image_list[index] = image2
            index += 1

            image_dict[c] = image2

            if show_plots:
                plt.figure(figsize=(2, 2))
                plt.imshow(image2, interpolation='nearest', cmap='magma')
                plt.axis('off')
                cbar = plt.colorbar(shrink=0.5)
                cbar.ax.tick_params(labelsize=3)
                plt.title(c)
                plt.show()

    if stack:
        stacked_image = np.stack(image_list)
        return image_dict, stacked_image
    else:
        return image_dict

def RGB_for_segmentation(membrane_channel, cytoplasm_channel, nucleus_channel, image_dict):
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

def run_cellpose(image, output_dir, use_gpu = True, model = "nuclei", pretrained_model= False, diameter = None, rgb_channels = [0,0], save_mask_as_png = False):
# IF ALL YOUR IMAGES ARE THE SAME TYPE, you can give a list with 2 elements
# channels = [0,0] # IF YOU HAVE GRAYSCALE
# channels = [2,3] # IF YOU HAVE G=cytoplasm and B=nucleus
# channels = [2,1] # IF YOU HAVE G=cytoplasm and R=nucleus
    if pretrained_model == True:
        custom_model = models.CellposeModel(gpu=use_gpu, pretrained_model=model) # setup custom model
        masks, flows, styles = custom_model.eval(
            image, diameter=diameter, channels=rgb_channels, resample=False) # run model

    else:
        cellpose_model = models.Cellpose(gpu=use_gpu, model_type=model) # setup default model
        masks, flows, styles = cellpose_model.eval(image, diameter=diameter, channels=rgb_channels) # run model
    
    if save_mask_as_png == True:
        filename = output_dir + "/segmentation.png"
        io.save_to_png(image, masks, flows, filename)

    return masks, flows, styles

def cellpose_segmentation(image_dict, output_dir, membrane_channel=None, cytoplasm_channel=None, nucleus_channel=None, use_gpu=True, model="nuclei", pretrained_model=False, diameter=None, save_mask_as_png=False):
    
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
            print("CellPose expects only two channels as input. Selecting nucleus channel and cytoplasm channel. If you want to use nucleus and membrane instead set cytoplasm_channel to None.")
        else:
            print("Selecting nucleus and cytoplasm channel for segmentation.")
        rgb_channels = [2, 3]
    elif membrane_channel:
        print("Selecting nucleus and membrane channel for segmentation.")
        rgb_channels = [1, 3]
    
    
    # Run CellPose
    input_image = RGB_for_segmentation(membrane_channel, cytoplasm_channel, nucleus_channel, image_dict)
    
    masks, flows, styles = run_cellpose(input_image, output_dir=output_dir, use_gpu=use_gpu, model=model, pretrained_model=pretrained_model, diameter=diameter, rgb_channels=rgb_channels, save_mask_as_png=save_mask_as_png)
    
    return masks, flows, styles, input_image, rgb_channels


def extract_features(image_dict, segmentation_masks, channels_to_quantify, output_file, size_cutoff=0):
    # Count pixels for each nucleus
    _, counts = np.unique(segmentation_masks, return_counts=True)
    
    # Identify nucleus IDs above the size cutoff, excluding background (ID 0)
    nucleus_ids = np.where(counts > size_cutoff)[0][1:]
    
    # Filter out small objects from segmentation masks
    filterimg = np.where(np.isin(segmentation_masks, nucleus_ids), segmentation_masks, 0)
    
    # Extract morphological features
    props = regionprops_table(filterimg, properties=('centroid', 'eccentricity', 'perimeter', 'convex_area', 'area', 'axis_major_length', 'axis_minor_length', 'label'))
    props_df = pd.DataFrame(props)
    props_df.set_index(props_df["label"], inplace=True)
    
    # Pre-allocate array for mean intensities
    mean_intensities = np.empty((len(nucleus_ids), len(channels_to_quantify)))

    # For each channel, compute mean intensities for all labels using vectorized operations
    for idx, chan in enumerate(tqdm(channels_to_quantify, desc="Processing channels")):
        chan_data = image_dict[chan]
        labels_matrix = np.isin(segmentation_masks, nucleus_ids).astype(int) * segmentation_masks
        sum_per_label = np.bincount(labels_matrix.ravel(), weights=chan_data.ravel())[nucleus_ids]
        count_per_label = np.bincount(labels_matrix.ravel())[nucleus_ids]
        mean_intensities[:, idx] = sum_per_label / count_per_label
    
    # Convert the array to a DataFrame
    mean_df = pd.DataFrame(mean_intensities, index=nucleus_ids, columns=channels_to_quantify)
    
    # Join with morphological features
    markers = mean_df.join(props_df)
    
    # Export to CSV
    markers.to_csv(output_file)



from skimage.color import rgb2gray
from skimage import exposure

def overlay_masks_on_image(image, masks, gamma = 1.5):
    # Convert image to grayscale
    gray_image = rgb2gray(image)

    # Increase brightness using gamma correction
    # Adjust the gamma value as needed
    gray_image = exposure.adjust_gamma(gray_image, gamma=gamma)

    # Convert grayscale to RGB to overlay the masks
    overlay = np.stack([gray_image]*3, axis=-1)

    outlines = utils.masks_to_outlines(masks)


    # Overlay the mask as red outlines on the grayscale image
    overlay[outlines == 1] = [255, 0, 0]

    return overlay, gray_image


def check_segmentation(overlay, grayscale, n=10, tilesize = 1000):
    # Check the shapes of provided arrays
   # if overlay.shape != grayscale.shape:
    #    raise ValueError("The two images should have the same shape")
    
    # Calculate the number of tiles in x and y directions
    y_tiles, x_tiles = overlay.shape[0] // tilesize, overlay.shape[1] // tilesize
    
    # Split images into tiles
    overlay_tiles = []
    grayscale_tiles = []
    for i in range(x_tiles):
        for j in range(y_tiles):
            x_start, y_start = i * tilesize, j * tilesize
            overlay_tile = overlay[y_start:y_start+tilesize, x_start:x_start+tilesize]
            grayscale_tile = grayscale[y_start:y_start+tilesize, x_start:x_start+tilesize]
            overlay_tiles.append(overlay_tile)
            grayscale_tiles.append(grayscale_tile)

    # Randomly select n tiles
    random_indices = random.sample(range(len(overlay_tiles)), n)
    
    # Plot the tiles
    fig, axs = plt.subplots(n, 2, figsize=(10, 5*n))
    for i, idx in enumerate(random_indices):
        axs[i, 0].imshow(overlay_tiles[idx])
        axs[i, 0].axis('off')
        axs[i, 1].imshow(grayscale_tiles[idx], cmap="gray")
        axs[i, 1].axis('off')
    plt.tight_layout()
    plt.show()