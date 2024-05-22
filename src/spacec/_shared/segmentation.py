import numpy as np
import os
from skimage import io


def create_multichannel_tiff(input_dir, output_dir, output_filename):
    # Get a list of all TIFF files in the input directory
    tiff_files = [f for f in os.listdir(input_dir) if f.endswith(('.tiff', '.tif'))]

    # Create a list of channel names from the TIFF file names
    channel_names = [os.path.splitext(f)[0] for f in tiff_files]

    # Read the images into a list of numpy arrays
    images = [io.imread(os.path.join(input_dir, f)) for f in tiff_files]

    # Stack the images along the third axis to create a multi-channel image
    multi_channel_image = np.stack(images, axis=-1)

    # move last dimension to first position
    multi_channel_image = np.moveaxis(multi_channel_image, -1, 0)
    
    print(multi_channel_image.shape)

    # Save the multi-channel image as a new TIFF file in the output directory
    io.imsave(os.path.join(output_dir, output_filename), multi_channel_image)


    # save channel names to text file
    with open(os.path.join(output_dir, 'channelnames.txt'), 'w') as f:
        for item in channel_names:
            f.write("%s\n" % item)
        
    return channel_names

# combine multiple channels in one image and add as new image to image_dict with the name segmentation_channel
def combine_channels(image_dict, channel_list, new_channel_name):
    # Create empty image
    new_image = np.zeros(
        (image_dict[channel_list[0]].shape[0], image_dict[channel_list[0]].shape[1])
    )

    # Add channels to image as maximum projection
    for channel in channel_list:
        new_image = np.maximum(new_image, image_dict[channel])

    # generate greyscale image
    new_image = np.uint8(new_image)

    # Add image to image_dict
    image_dict[new_channel_name] = new_image

    return image_dict


def format_CODEX(
    image,
    channel_names=None,
    number_cycles=None,
    images_per_cycle=None,
    # show_plots=False,
    stack=True,
    input_format="Multichannel",
):
    if input_format == "CODEX":
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

                # if show_plots:
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

    if input_format == "Multichannel":
        image_dict = {}
        index = 0

        for i in range(len(channel_names)):
            image2 = image[i, :, :]
            image_dict[channel_names[i]] = image2
            index += 1

        return image_dict
    
    if input_format == "Channels":
        # Get a list of all TIFF files in the input directory
        tiff_files = [f for f in os.listdir(input_dir) if f.endswith(('.tiff', '.tif'))]

        # Create a list of channel names from the TIFF file names
        channel_names = [os.path.splitext(f)[0] for f in tiff_files]

        # Read the images into a list of numpy arrays
        images = [io.imread(os.path.join(image, f)) for f in tiff_files]

        # Stack the images along the third axis to create a multi-channel image
        multi_channel_image = np.stack(images, axis=-1)

        # move last dimension to first position
        multi_channel_image = np.moveaxis(multi_channel_image, -1, 0)
                
        image_dict = {}
        index = 0
        
        for i in range(len(channel_names)):
            image2 = multi_channel_image[i, :, :]
            image_dict[channel_names[i]] = image2
            index += 1

        return image_dict
