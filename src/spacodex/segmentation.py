import pathlib
from turtle import mode
import skimage.io
import numpy as np
import time, os, sys
import requests
from urllib.parse import urlparse
import matplotlib.pyplot as plt
import matplotlib as mpl
import sys
from cellpose import utils, io, core, models, metrics, plot
import pandas as pd
from skimage.measure import regionprops_table
from tqdm import tqdm
import os, shutil
from glob import glob
import random
import numpy as np
from deepcell.applications import Mesmer
from deepcell.utils.plot_utils import create_rgb_image
from deepcell.utils.plot_utils import make_outline_overlay
from tensorflow.keras.models import load_model
from skimage.color import rgb2gray
from skimage import exposure

def format_CODEX(image, 
                 channel_names = None, 
                 number_cycles = None, 
                 images_per_cycle = None, 
                 #show_plots=False, 
                 stack=True, 
                 technology = "CODEX"):
    
    if technology == "CODEX":
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

                #if show_plots:
                #    plt.figure(figsize=(2, 2))
                #    plt.imshow(image2, interpolation='nearest', cmap='magma')
                #    plt.axis('off')
                #    cbar = plt.colorbar(shrink=0.5)
                #    cbar.ax.tick_params(labelsize=3)
                #    plt.title(c)
                #    plt.show()
                
        if stack:
            stacked_image = np.stack(image_list)
            return image_dict, stacked_image
        else:
            return image_dict
                
    if technology == "Phenocycler":
        image_dict = {}
        index = 0
        
        for i in range(len(channel_names)):
            image2 = image[i, :, :]
            image_dict[channel_names[i]] = image2
            index += 1

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
    if membrane_channel is None and cytoplasm_channel is None:
        input_image = RGB_for_segmentation(membrane_channel, cytoplasm_channel, nucleus_channel, image_dict)
        # extract only the blue channel
        final_input = input_image[:,:,2]
        
    else:
        input_image = RGB_for_segmentation(membrane_channel, cytoplasm_channel, nucleus_channel, image_dict)
        final_input = input_image.copy()
    
    masks, flows, styles = run_cellpose(final_input, output_dir=output_dir, use_gpu=use_gpu, model=model, pretrained_model=pretrained_model, diameter=diameter, rgb_channels=rgb_channels, save_mask_as_png=save_mask_as_png)
    
    return masks, flows, styles, input_image, rgb_channels


def extract_features(image_dict, segmentation_masks, channels_to_quantify, output_file, size_cutoff=0):
    segmentation_masks = segmentation_masks.squeeze() 
    
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
    
    # rename column 
    markers.rename(columns={'centroid-0': 'x'}, inplace=True)
    markers.rename(columns={'centroid-1': 'y'}, inplace=True)

    # Export to CSV
    markers.to_csv(output_file)



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
    
# combine multiple channels in one image and add as new image to image_dict with the name segmentation_channel
def combine_channels(image_dict, channel_list, new_channel_name):
    
    # Create empty image
    new_image = np.zeros((image_dict[channel_list[0]].shape[0], image_dict[channel_list[0]].shape[1]))
    
    # Add channels to image as maximum projection
    for channel in channel_list:
        new_image = np.maximum(new_image, image_dict[channel])
    
    # generate greyscale image
    new_image = np.uint8(new_image)
    
    # Add image to image_dict
    image_dict[new_channel_name] = new_image
    
    return image_dict


def load_mesmer_model(path):

    # check if folder Mesmer_model exist 
    if not os.path.exists(os.path.join(path, 'Mesmer_model')):
        os.makedirs(os.path.join(path, 'Mesmer_model'))

        print("downloading Mesmer model")

        # download model from https://deepcell-data.s3-us-west-1.amazonaws.com/saved-models/MultiplexSegmentation-9.tar.gz
        url = 'https://deepcell-data.s3-us-west-1.amazonaws.com/saved-models/MultiplexSegmentation-9.tar.gz'
        response = requests.get(url)

        # Ensure the download was successful
        response.raise_for_status()

        # Write the content of the response to a file in the 'Mesmer_model' directory
        with open(os.path.join(path, 'Mesmer_model/MultiplexSegmentation.tar.gz'), 'wb') as f:
            f.write(response.content)

        # unpack tar file
        shutil.unpack_archive(os.path.join(path, 'Mesmer_model/MultiplexSegmentation.tar.gz'), os.path.join(path, 'Mesmer_model/'))

        # delete the .tar.gz file
        os.remove(os.path.join(path, 'Mesmer_model/MultiplexSegmentation.tar.gz'))
        
        # move content of folder MultiplexSegmentation to Mesmer_model
        #shutil.move(os.path.join(path, 'Mesmer_model/MultiplexSegmentation'), os.path.join(path, 'Mesmer_model/'))
        
        # remove empty folder MultiplexSegmentation
        #os.rmdir(os.path.join(path, 'Mesmer_model/MultiplexSegmentation'))
        
        print("Mesmer model downloaded and unpacked")

    model_path = os.path.join(path, "Mesmer_model/MultiplexSegmentation")

    mesmer_pretrained_model = load_model(model_path, compile=False)

    return mesmer_pretrained_model

def mesmer_segmentation(nuclei_image, 
                              membrane_image, 
                              image_mpp=0.5, 
                              plot_predictions = True, 
                              compartment='whole-cell', # or 'nuclear'
                              model_path="./models"): 
    
    pathlib.Path(model_path).mkdir(parents=True, exist_ok=True)
    mesmer_pretrained_model = load_mesmer_model(model_path)

    # Initialize Mesmer application
    app = Mesmer(model=mesmer_pretrained_model)

    # Create a combined image stack
    # Assumes nuclei_image and membrane_image are numpy arrays of the same shape
    combined_image = np.stack([nuclei_image, membrane_image], axis=-1)

    # Add an extra dimension to make it compatible with Mesmer's input requirements
    # Changes shape from (height, width, channels) to (1, height, width, channels)
    combined_image = np.expand_dims(combined_image, axis=0)

    # Run the Mesmer model
    segmented_image = app.predict(combined_image, image_mpp=image_mpp, compartment=compartment)

    if plot_predictions == True:
        # create rgb overlay of image data for visualization
        rgb_images = create_rgb_image(combined_image, channel_colors=['green', 'blue'])
        # create overlay of segmentation results
        overlay_data = make_outline_overlay(rgb_data=rgb_images, predictions=segmented_image)
        
        # select index for displaying
        idx = 0

        # plot the data
        fig, ax = plt.subplots(1, 2, figsize=(15, 15))
        ax[0].imshow(rgb_images[idx, ...])
        ax[1].imshow(overlay_data[idx, ...])
        ax[0].set_title('Raw data')
        ax[1].set_title('Predictions')
        plt.show()
        
        # The output will be a numpy array with the segmentation results
    return segmented_image


# plot membrane channel selectd segmentation
def pl_segmentation_ch(file_name, # image for segmentation
                   channel_file, # all channels used for staining
                   output_dir, #
                   extra_seg_ch_list = None, # channels used for membrane segmentation
                   nuclei_channel = 'DAPI',
                   technology = 'Phenocycler' # CODEX or Phenocycler --> This depends on the machine you are using and the resulting file format (see documentation above)
                  ):
    # Load the image
    img = skimage.io.imread(file_name)

    # Read channels and store as list 
    with open(channel_file, 'r') as f:
        channel_names = f.read().splitlines()

    # Function reads channels and stores them as dictonary (storing as dictionary allows to select specific channels by name)
    image_dict = format_CODEX(image = img, 
                              channel_names = channel_names, 
                              technology = technology)

    image_dict = combine_channels(image_dict, extra_seg_ch_list, new_channel_name = 'segmentation_channel')
    fig, ax = plt.subplots(1,2, figsize=(15,15))
    ax[0].imshow(image_dict[nuclei_channel])
    ax[1].imshow(image_dict['segmentation_channel'])
    ax[0].set_title('nuclei')
    ax[1].set_title('membrane')
    plt.show()

# perform cell segmentation
def tl_cell_segmentation(file_name, 
                      channel_file,
                      output_dir,
                      output_fname = "",
                      seg_method ='mesmer', 
                      nuclei_channel = 'DAPI',
                      technology ='Phenocycler', # CODEX or Phenocycler --> This depends on the machine you are using and the resulting file format (see documentation above)
                      membrane_channel_list = None,
                      size_cutoff = 0, #quantificaition
                      compartment = 'whole-cell', # mesmer # segment whole cells or nuclei only
                      plot_predictions = True, # mesmer # plot segmentation results
                      model = "tissuenet", # cellpose
                      use_gpu = True, # cellpose
                      cytoplasm_channel_list = None, #celpose
                      pretrained_model= True, # cellpose 
                      diameter = None, #cellpose
                      save_mask_as_png = False #cellpose
                        ):

    print("Create image channels!")
    # Load the image
    img = skimage.io.imread(file_name)

    # Read channels and store as list 
    with open(channel_file, 'r') as f:
        channel_names = f.read().splitlines()
        
    # Function reads channels and stores them as a dictionary (storing as a dictionary allows to select specific channels by name)
    image_dict = format_CODEX(image = img, 
                            channel_names = channel_names, # file with list of channel names (see channelnames.txt)
                            technology = technology) 

    # Generate image for segmentation
    if membrane_channel_list is not None:
        image_dict = combine_channels(image_dict, membrane_channel_list, 
                                  new_channel_name = 'segmentation_channel') # combine channels for better segmentation. In this example we combine channels to get a membrane outline for all cells in the image
    
    if seg_method == 'mesmer':
        print("Segmenting with Mesmer!")
        if membrane_channel_list is None:
            print("Mesmer expects two-channel images as input, where the first channel must be a nuclear channel (e.g. DAPI) and the second channel must be a membrane or cytoplasmic channel (e.g. E-Cadherin).")
            sys.exit("Please provide any membrane or cytoplasm channel!")
        else:
            masks = mesmer_segmentation(nuclei_image = image_dict[nuclei_channel], 
                                                membrane_image = image_dict['segmentation_channel'], 
                                                plot_predictions = plot_predictions, # plot segmentation results
                                                compartment=compartment) # segment whole cells or nuclei only

    else:
        print("Segmenting with Cellpose!")
        if membrane_channel_list is None:
            masks, flows, styles, input_image, rgb_channels = cellpose_segmentation(image_dict = image_dict, 
                                                                                output_dir = output_dir, 
                                                                                membrane_channel = None, 
                                                                                cytoplasm_channel = cytoplasm_channel_list, 
                                                                                nucleus_channel = nuclei_channel, 
                                                                                use_gpu = use_gpu, 
                                                                                model = model, 
                                                                                pretrained_model= pretrained_model,
                                                                                diameter = diameter, 
                                                                                save_mask_as_png = save_mask_as_png)
        else:
            masks, flows, styles, input_image, rgb_channels = cellpose_segmentation(image_dict = image_dict, 
                                                                                output_dir = output_dir, 
                                                                                membrane_channel = "segmentation_channel", 
                                                                                cytoplasm_channel = cytoplasm_channel_list, 
                                                                                nucleus_channel = nuclei_channel, 
                                                                                use_gpu = use_gpu, 
                                                                                model = model, 
                                                                                pretrained_model= pretrained_model,
                                                                                diameter = diameter, 
                                                                                save_mask_as_png = save_mask_as_png)
        
    
    print("Quantifying features after segmentation!")
    extract_features(image_dict= image_dict, # image dictionary
                 segmentation_masks = masks, # segmentation masks generated by cellpose
                 channels_to_quantify = channel_names, # list of channels to quantify (here: all channels)
                 output_file = pathlib.Path(output_dir) / (output_fname + "_" + seg_method + "_result.csv"), # output path to store results as csv 
                 size_cutoff = size_cutoff) # size cutoff for segmentation masks (default = 0)
    
    print("Done!")

    return {'img':img, 'masks':masks, 'image_dict':image_dict}

def pl_show_masks(seg_output, 
                   nucleus_channel, 
                   additional_channels = None,
                   show_subsample = True,
                   n=2, # need to be at least 2
                  tilesize = 100,
                  idx = 0,
                  rand_seed = 1):

    image_dict = seg_output['image_dict']
    masks = seg_output['masks']
    
    
    # Create a combined image stack
    # Assumes nuclei_image and membrane_image are numpy arrays of the same shape
    if len(masks.shape) == 2:
        masks = np.expand_dims(masks, axis=0)
        masks = np.expand_dims(masks, axis=0)
        masks = np.moveaxis(masks, 0, -1)
        
    if additional_channels != None:
        image_dict = combine_channels(image_dict, additional_channels, 
                                      new_channel_name = 'segmentation_channel')
        nuclei_image = image_dict[nucleus_channel]
        add_chan_image = image_dict["segmentation_channel"]
        combined_image = np.stack([nuclei_image, add_chan_image], axis=-1)
        # Add an extra dimension to make it compatible with Mesmer's input requirements
        # Changes shape from (height, width, channels) to (1, height, width, channels)
        combined_image = np.expand_dims(combined_image, axis=0)
        # create rgb overlay of image data for visualization
        rgb_images = create_rgb_image(combined_image, channel_colors=['green', 'blue'])
    else:
        nuclei_image = image_dict[nucleus_channel]
        combined_image = np.stack([nuclei_image], axis=-1)
        # Add an extra dimension to make it compatible with Mesmer's input requirements
        # Changes shape from (height, width, channels) to (1, height, width, channels)
        combined_image = np.expand_dims(combined_image, axis=0)
        # create rgb overlay of image data for visualization
        rgb_images = create_rgb_image(combined_image, channel_colors=['blue'])
    
    # create overlay of segmentation results
    overlay_data = make_outline_overlay(rgb_data=rgb_images, predictions=masks)
            
    # select index for displaying
    

    # plot the data
    fig, ax = plt.subplots(1, 2, figsize=(15, 15))
    ax[0].imshow(rgb_images[idx, ...])
    ax[1].imshow(overlay_data[idx, ...])
    ax[0].set_title('Raw data')
    ax[1].set_title('Predictions')
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
            y_tiles, x_tiles = overlay_data.shape[0] // tilesize, overlay_data.shape[1] // tilesize

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
                        overlay_tile = overlay_data[y_start:y_start+tilesize, x_start:x_start+tilesize]
                        image_tile = rgb_images[y_start:y_start+tilesize, x_start:x_start+tilesize]

                        overlay_tiles.append(overlay_tile)
                        grayscale_tiles.append(image_tile)

                # Randomly select n tiles
                random_indices = random.sample(range(len(overlay_tiles)), n)

                # Plot the tiles
                fig, axs = plt.subplots(n, 2, figsize=(10, 5*n))
                for i, idx in enumerate(random_indices):
                    axs[i, 0].imshow(grayscale_tiles[idx])
                    axs[i, 0].axis('off')
                    axs[i, 1].imshow(overlay_tiles[idx])
                    axs[i, 1].axis('off')
                    
                    axs[i, 0].set_title('Raw data')
                    axs[i, 1].set_title('Predictions')
                plt.tight_layout()
                plt.show()

    return overlay_data, rgb_images

    
