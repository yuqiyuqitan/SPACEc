import os
import pathlib

import numpy as np
import tifffile
from skimage import io


def create_multichannel_tiff(input_dir, output_dir, output_filename):
    """
    Create a multi-channel TIFF image by stacking individual TIFF files from a specified directory.

    This function searches for individual TIFF files in the provided input directory,
    sorts them alphabetically by filename, and reads each image file into memory.
    It then stacks these images along a new first dimension (representing channels)
    to create a multi-channel image. The resulting multi-channel TIFF image is saved
    to the specified output directory with the given output filename. Additionally, a
    text file containing the channel names (derived from the filenames) is created.

    Parameters
    ----------
    input_dir : str
        Directory containing the input TIFF files.
    output_dir : str
        Directory where the output TIFF file and the channel names text file will be saved.
    output_filename : str
        Name of the output multi-channel TIFF file (for example, 'image.tif').

    Returns
    -------
    list of str
        A sorted list of channel names derived from the filenames. If no TIFF files are found
        or an error occurs, an empty list is returned.

    Notes
    -----
    This function loads all input images into memory before stacking. For a large number of
    high-resolution images, this operation may be memory-intensive.
    """
    print(f"Creating multi-channel TIFF from directory: {input_dir}")
    try:
        # Retrieve and sort TIFF files (both .tiff and .tif) in the input directory.
        tiff_files = sorted(
            [f for f in os.listdir(input_dir) if f.lower().endswith((".tiff", ".tif"))]
        )
        if not tiff_files:
            print("Warning: No TIFF files found in input directory.")
            return []

        # Derive channel names by stripping file extensions.
        channel_names = [os.path.splitext(f)[0] for f in tiff_files]
        print(f"Found channels: {channel_names}")

        # Read each TIFF file using tifffile (this step may be memory intensive).
        images = [tifffile.imread(os.path.join(input_dir, f)) for f in tiff_files]

        # Stack images into a multi-channel image assuming each image has shape (Height, Width).
        multi_channel_image = np.stack(images, axis=0)
        print(f"Stacked image shape: {multi_channel_image.shape}")

        # Ensure the output directory exists or create it.
        output_path = pathlib.Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        output_filepath = output_path / output_filename
        channel_name_filepath = output_path / "channelnames.txt"

        # Save the multi-channel image with ImageJ compatible metadata.
        tifffile.imwrite(output_filepath, multi_channel_image, imagej=True)
        print(f"Saved multi-channel TIFF to: {output_filepath}")

        # Write channel names to a text file, one per line.
        with open(channel_name_filepath, "w") as f:
            for item in channel_names:
                f.write(f"{item}\n")
        print(f"Saved channel names to: {channel_name_filepath}")

        return channel_names

    except FileNotFoundError:
        print(f"Error: Input directory not found: {input_dir}")
        return []
    except Exception as e:
        print(f"Error creating multi-channel TIFF: {e}")
        return []


# combine multiple channels in one image and add as new image to image_dict with the name segmentation_channel
def combine_channels(image_dict, channel_list, new_channel_name):
    """
    Combine multiple channels into a single channel.

    Parameters
    ----------
    image_dict : dict
        Dictionary with channel names as keys and images as values.
    channel_list : list of str
        List of channel names to combine.
    new_channel_name : str
        Name of the new channel.

    Returns
    -------
    dict
        Updated dictionary with the new channel added.
    """
    if not channel_list:
        print("Warning: No channels provided to combine.")
        return image_dict

    valid_channels = [ch for ch in channel_list if ch in image_dict]
    if not valid_channels:
        print(
            f"Warning: None of the specified channels {channel_list} found in image_dict."
        )
        return image_dict

    print(
        f"Combining channels {valid_channels} into '{new_channel_name}' using max projection."
    )
    # Determine shape and dtype from the first valid channel
    ref_image = image_dict[valid_channels[0]]
    shape = ref_image.shape
    dtype = ref_image.dtype

    # Create empty image for the result
    new_image = np.zeros(shape, dtype=dtype)

    # Perform maximum projection
    for channel in valid_channels:
        if image_dict[channel].shape != shape:
            print(
                f"Warning: Shape mismatch for channel '{channel}' ({image_dict[channel].shape}). Skipping."
            )
            continue
        np.maximum(new_image, image_dict[channel], out=new_image)  # In-place maximum

    # Add the combined image to the dictionary
    image_dict[new_channel_name] = new_image
    return image_dict


def load_image_dictionary(file_name, channel_file, input_format, nuclei_channel):
    """
    Loads images and channel names based on the specified format using tifffile.

    Memory Consideration: Loading large 'Multichannel' or 'CODEX' files directly
    with tifffile.imread can consume significant memory. For very large files,
    consider using memory mapping (e.g., tifffile.imread(..., aszarr=True)) if needed
    (requires downstream code modification to handle Zarr arrays). 'Channels' format reads
    individual files, which can also be memory-intensive if there are many large files.

    Parameters:
        file_name (str or Path): Path to image file (Multichannel/CODEX) or directory (Channels).
        channel_file (str or Path): Path to file containing channel names (one per line). Used for Multichannel/CODEX.
        input_format (str): 'Multichannel', 'Channels', or 'CODEX'.
        nuclei_channel (str): Name of the nuclei channel (must exist in the loaded channels).

    Returns:
        tuple: (img_ref, image_dict, channel_names_list)
               img_ref (ndarray): Reference image (typically nuclei channel or the raw loaded stack).
               image_dict (dict): Dictionary mapping channel names to 2D NumPy arrays.
               channel_names_list (list): List of loaded channel names.
               Returns (None, None, None) on error.
    """
    print(f"--- Loading Image Data (Format: {input_format}) ---")
    img_ref = None
    image_dict = None
    channel_names_list = None

    try:
        if input_format == "Channels":
            # file_name is the directory path
            # format_CODEX now uses tifffile internally for 'Channels'
            image_dict, channel_names_list = format_CODEX(
                image=file_name,
                input_format=input_format,
            )
            if image_dict is None:
                raise ValueError("Image formatting failed.")  # Check format_CODEX error

            if nuclei_channel not in image_dict:
                raise ValueError(
                    f"Specified nuclei_channel '{nuclei_channel}' not found in loaded channels: {list(image_dict.keys())}"
                )
            img_ref = image_dict[nuclei_channel]  # Use nuclei as reference

        elif input_format in ["Multichannel", "CODEX"]:
            # file_name is the image file path
            if not os.path.exists(file_name):
                raise FileNotFoundError(f"Image file not found: {file_name}")
            if not os.path.exists(channel_file):
                raise FileNotFoundError(f"Channel file not found: {channel_file}")

            print(f"Loading image: {file_name}")
            # Memory intensive step for large files - use tifffile
            # Memory-intensive for large files; tifffile.imread(file_name, aszarr=True) would allow memory mapping
            loaded_img = io.imread(file_name)
            img_ref = loaded_img  # Store raw loaded image as reference
            print(f"Loaded image shape: {loaded_img.shape}")

            with open(channel_file, "r") as f:
                channel_names_from_file = f.read().splitlines()
            print(
                f"Loaded {len(channel_names_from_file)} channel names from: {channel_file}"
            )

            # Infer CODEX parameters from the image shape
            number_cycles = None
            images_per_cycle = None
            if input_format == "CODEX":
                if loaded_img.ndim != 4:
                    raise ValueError(
                        f"CODEX input image must be 4D. Got shape: {loaded_img.shape}"
                    )
                # Expected CODEX layout: (cycles, H, W, images_per_cycle)
                number_cycles = loaded_img.shape[0]
                images_per_cycle = loaded_img.shape[3]
                print(
                    f"Inferred CODEX params: Cycles={number_cycles}, Channels/Cycle={images_per_cycle}"
                )

            image_dict, channel_names_list = format_CODEX(
                image=loaded_img,
                channel_names=channel_names_from_file,
                number_cycles=number_cycles,
                images_per_cycle=images_per_cycle,
                input_format=input_format,
            )
            if image_dict is None:
                raise ValueError("Image formatting failed.")  # Check format_CODEX error

            if nuclei_channel not in image_dict:
                raise ValueError(
                    f"Specified nuclei_channel '{nuclei_channel}' not found in loaded channels: {list(image_dict.keys())}"
                )

        else:
            raise ValueError(f"Invalid input_format: {input_format}")

        print(f"Image dictionary created with {len(image_dict)} channels.")
        return img_ref, image_dict, channel_names_list

    except (FileNotFoundError, ValueError, MemoryError) as e:
        print(f"Error loading image dictionary: {e}")
        return None, None, None
    except Exception as e:
        print(f"An unexpected error occurred during image loading: {e}")
        return None, None, None


def format_CODEX(
    image,
    channel_names=None,
    number_cycles=None,  # Required for CODEX format
    images_per_cycle=None,  # Required for CODEX format
    input_format="Multichannel",
):
    """
    Formats image data into a dictionary based on the specified input format.
    Automatically detects number_cycles and images_per_cycle from image shape if not provided for CODEX format.

    Memory Consideration:
    - 'CODEX': Assumes 'image' (4D) is already loaded. Creates 2D slices, potentially duplicating data references.
    - 'Multichannel': Assumes 'image' (3D) is already loaded. Creates 2D slices.
    - 'Channels': Reads individual files using tifffile. Can be memory-intensive if many large files.

    Parameters:
        image (ndarray or str): Input image data (NumPy array for CODEX/Multichannel) or directory path (for Channels).
        channel_names (list, optional): List of channel names. Required for CODEX/Multichannel.
        number_cycles (int, optional): Number of cycles. If None for CODEX, inferred from image.shape[0].
        images_per_cycle (int, optional): Number of channels per cycle. If None for CODEX, inferred from image.shape[3].
        input_format (str): 'CODEX', 'Multichannel', or 'Channels'.

    Returns:
        tuple: (image_dict, channel_names_list)
               image_dict (dict): Dictionary mapping channel names to 2D NumPy arrays.
               channel_names_list (list): List of channel names processed.
               Returns (None, None) on error.
    """
    image_dict = {}
    processed_channel_names = []

    try:
        if input_format == "CODEX":
            if channel_names is None:
                raise ValueError("channel_names are required for CODEX format.")
            if image.ndim != 4:
                raise ValueError(
                    f"CODEX input image must be 4D. Got shape: {image.shape}"
                )

            # --- Auto-detect cycles and images_per_cycle if not provided ---
            inferred_params = False
            if number_cycles is None:
                number_cycles = image.shape[0]
                inferred_params = True
            if images_per_cycle is None:
                # Assuming CODEX format is (cycles, H, W, images_per_cycle)
                # Adjust index if your CODEX format is different (e.g., (cycles, images_per_cycle, H, W))
                images_per_cycle = image.shape[3]
                inferred_params = True

            if inferred_params:
                print(
                    f"Inferred CODEX parameters: Cycles={number_cycles}, Channels/Cycle={images_per_cycle} from image shape {image.shape}"
                )
            # --- End Auto-detection ---

            # Validate dimensions after potential inference
            if image.shape[0] != number_cycles or image.shape[3] != images_per_cycle:
                raise ValueError(
                    f"CODEX image shape {image.shape} incompatible with cycles={number_cycles}, images_per_cycle={images_per_cycle}. Expected (cycles, H, W, images_per_cycle)."
                )

            total_expected_images = number_cycles * images_per_cycle
            if len(channel_names) < total_expected_images:
                print(
                    f"Warning: Provided {len(channel_names)} channel names, but expected {total_expected_images} based on cycles/images_per_cycle."
                )
                # Truncate expected images to match available names
                total_expected_images = len(channel_names)

            print(
                f"Formatting CODEX image ({number_cycles} cycles, {images_per_cycle} channels/cycle)..."
            )
            idx = 0
            for i in range(number_cycles):
                for n in range(images_per_cycle):
                    if idx >= total_expected_images:
                        break  # Stop if we run out of channel names
                    channel_name = channel_names[idx]
                    # Extract the 2D image for this channel
                    img_slice = image[i, :, :, n]
                    image_dict[channel_name] = img_slice
                    processed_channel_names.append(channel_name)
                    idx += 1
                if idx >= total_expected_images:
                    break
            print(f"Formatted {len(image_dict)} CODEX channels.")
            return image_dict, processed_channel_names

        elif input_format == "Multichannel":
            if channel_names is None:
                raise ValueError("channel_names are required for Multichannel format.")
            if image.ndim != 3 or image.shape[0] != len(channel_names):
                raise ValueError(
                    f"Multichannel image shape {image.shape} incompatible with {len(channel_names)} channel names. Expected (channels, H, W)."
                )

            print(f"Formatting Multichannel image ({len(channel_names)} channels)...")
            for i, name in enumerate(channel_names):
                image_dict[name] = image[i, :, :]
            processed_channel_names = list(channel_names)  # Use provided names
            print(f"Formatted {len(image_dict)} Multichannel channels.")
            return image_dict, processed_channel_names

        elif input_format == "Channels":
            if not isinstance(image, (str, pathlib.Path)) or not os.path.isdir(image):
                raise ValueError(
                    "For 'Channels' format, 'image' must be a valid directory path."
                )

            print(f"Formatting 'Channels' from directory: {image}")
            # Get sorted list of TIFF files
            tiff_files = sorted(
                [f for f in os.listdir(image) if f.lower().endswith((".tiff", ".tif"))]
            )
            if not tiff_files:
                raise FileNotFoundError(f"No TIFF files found in directory: {image}")

            channel_names_from_files = [os.path.splitext(f)[0] for f in tiff_files]
            print(f"Found channels: {channel_names_from_files}")

            # Read images directly into dict using tifffile (Memory intensive if many large files)
            for i, f in enumerate(tiff_files):
                channel_name = channel_names_from_files[i]
                try:
                    img_path = os.path.join(image, f)
                    image_dict[channel_name] = io.imread(img_path)  # Use tifffile
                    processed_channel_names.append(channel_name)
                except Exception as e:
                    print(
                        f"Warning: Error reading file {f} {e}. Skipping channel '{channel_name}'."
                    )
            print(f"Formatted {len(image_dict)} channels from files.")
            return image_dict, processed_channel_names

        else:
            raise ValueError(
                f"Invalid input_format: {input_format}. Choose from 'CODEX', 'Multichannel', 'Channels'."
            )

    except Exception as e:
        print(f"Error during image formatting ({input_format}): {e}")
        return None, None
