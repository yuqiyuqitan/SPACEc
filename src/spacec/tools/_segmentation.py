import gc
import os
import pathlib
import sys
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import skimage
import skimage.io
import tifffile
from cellpose import io
from cellpose import models as cellpose_models  # fixed
from cellpose.core import use_gpu as cellpose_use_gpu
from IPython.display import clear_output
from scipy.ndimage import gaussian_filter, label
from skimage.measure import regionprops_table
from skimage.segmentation import relabel_sequential
from tqdm import tqdm
from tqdm.notebook import (
    tqdm as notebook_tqdm,  # Use notebook version for better display
)

from .._shared.overlay_utils import create_rgb_image, make_outline_overlay
from .._shared.segmentation import (
    combine_channels,
    create_multichannel_tiff,
    format_CODEX,
)


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
            # Consider tifffile.imread(file_name, aszarr=True) for memory mapping if needed later
            loaded_img = skimage.io.imread(file_name)
            img_ref = loaded_img  # Store raw loaded image as reference
            print(f"Loaded image shape: {loaded_img.shape}")

            with open(channel_file, "r") as f:
                channel_names_from_file = f.read().splitlines()
            print(
                f"Loaded {len(channel_names_from_file)} channel names from: {channel_file}"
            )

            # Determine CODEX parameters if needed (example, needs actual logic)
            number_cycles = None
            images_per_cycle = None
            if input_format == "CODEX":
                # Placeholder: Infer or require these parameters
                if loaded_img.ndim != 4:
                    raise ValueError(
                        f"CODEX input image must be 4D. Got shape: {loaded_img.shape}"
                    )
                # Assuming CODEX format is (cycles, H, W, images_per_cycle)
                # Adjust if your CODEX format is different (e.g., (cycles, images_per_cycle, H, W))
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


# --------------------------------------------------------------------------
# Preprocessing Utilities
# --------------------------------------------------------------------------


def setup_gpu(use_gpu=True, set_memory_growth=True):
    """Checks Cellpose GPU availability and returns effective GPU usage flag."""
    if set_memory_growth is not True:
        warnings.warn(
            "setup_gpu(set_memory_growth=...) has no effect after TensorFlow removal.",
            DeprecationWarning,
            stacklevel=2,
        )

    if not use_gpu:
        print("GPU usage explicitly disabled.")
        return False

    gpu_available = bool(cellpose_use_gpu())
    if gpu_available:
        if set_memory_growth:
            print(
                "GPU detected for Cellpose. set_memory_growth is ignored for non-TensorFlow backends."
            )
        else:
            print("GPU detected for Cellpose.")
        return True

    print("No GPU detected for Cellpose. Falling back to CPU.")
    return False


def prepare_segmentation_dict(image_dict, nuclei_channel, membrane_channel_list):
    """
    Prepares a dictionary containing only the channels needed for segmentation.
    Combines membrane channels if provided.

    Parameters:
        image_dict (dict): Full dictionary of images.
        nuclei_channel (str): Name of the nuclei channel.
        membrane_channel_list (list or None): List of membrane channel names to combine.

    Returns:
        tuple: (segmentation_dict, combined_membrane_channel_name)
               segmentation_dict (dict): Dictionary with 'nuclei_channel' and optionally 'segmentation_channel'.
               combined_membrane_channel_name (str or None): Name of the combined channel, or None.
    """
    if nuclei_channel not in image_dict:
        raise ValueError(
            f"Nuclei channel '{nuclei_channel}' not found in image dictionary."
        )

    segmentation_dict = {}
    # Make a copy to avoid modifying the original image_dict values if resizing/etc happens later
    segmentation_dict[nuclei_channel] = image_dict[nuclei_channel].copy()
    combined_membrane_channel_name = None

    if membrane_channel_list:
        combined_membrane_channel_name = "segmentation_channel"
        # Combine channels (creates a new array in the dict)
        # Pass a copy of image_dict to combine_channels to avoid modifying the original one
        segmentation_dict_temp = combine_channels(
            image_dict.copy(), membrane_channel_list, combined_membrane_channel_name
        )
        if combined_membrane_channel_name in segmentation_dict_temp:
            # Add the newly created combined channel to our segmentation_dict
            segmentation_dict[combined_membrane_channel_name] = segmentation_dict_temp[
                combined_membrane_channel_name
            ]
        else:
            print(
                f"Warning: Failed to create '{combined_membrane_channel_name}'. Proceeding without it."
            )
            combined_membrane_channel_name = None  # Reset if creation failed
        del segmentation_dict_temp  # Clean up temporary dict

    print(
        f"Segmentation dictionary prepared with channels: {list(segmentation_dict.keys())}"
    )
    return segmentation_dict, combined_membrane_channel_name


def resize_segmentation_images(seg_dict, resize_factor):
    """
    Resizes images within the segmentation dictionary using area interpolation.

    Memory Consideration: Creates new resized arrays, temporarily increasing memory usage.

    Parameters:
        seg_dict (dict): Dictionary containing images to resize.
        resize_factor (float): Factor by which to resize (e.g., 0.5 for half size).

    Returns:
        dict: Dictionary with resized images. Returns original dict if resize_factor is 1.
    """
    if resize_factor == 1:
        print("Resize factor is 1, skipping image resizing.")
        return seg_dict
    if resize_factor <= 0:
        raise ValueError("Resize factor must be positive.")

    print(f"Resizing segmentation images by factor: {resize_factor}")
    resized_seg_dict = {}
    for ch, im in seg_dict.items():
        if im is None:
            continue
        original_height, original_width = im.shape[:2]
        new_height = int(original_height * resize_factor)
        new_width = int(original_width * resize_factor)

        if new_height == 0 or new_width == 0:
            print(
                f"Warning: Resize factor {resize_factor} results in zero dimension for channel '{ch}'. Skipping resize."
            )
            resized_seg_dict[ch] = im.copy()  # Keep original
            continue

        # Use INTER_AREA for downsampling, INTER_LINEAR for upsampling (general purpose)
        interpolation = cv2.INTER_AREA if resize_factor < 1 else cv2.INTER_LINEAR
        resized_im = cv2.resize(
            im, (new_width, new_height), interpolation=interpolation
        )
        resized_seg_dict[ch] = resized_im
        print(f"  Resized channel '{ch}' from {im.shape} to {resized_im.shape}")
        # del im # Original image in seg_dict is not deleted here, caller should manage seg_dict
    gc.collect()
    return resized_seg_dict


def resize_mask(mask, target_shape_or_ref_img):
    """
    Resizes a segmentation mask to a target shape using nearest neighbor interpolation.

    Parameters:
        mask (ndarray): The segmentation mask to resize.
        target_shape_or_ref_img (tuple or ndarray): Target (height, width) or a reference image
                                                     from which to get the target shape.

    Returns:
        ndarray: The resized mask.
    """
    if mask is None:
        return None

    if isinstance(target_shape_or_ref_img, np.ndarray):
        target_height, target_width = target_shape_or_ref_img.shape[:2]
    elif (
        isinstance(target_shape_or_ref_img, tuple) and len(target_shape_or_ref_img) >= 2
    ):
        target_height, target_width = target_shape_or_ref_img[:2]
    else:
        raise ValueError(
            "target_shape_or_ref_img must be a NumPy array or a (height, width) tuple."
        )

    current_height, current_width = mask.shape[:2]

    if (current_height, current_width) == (target_height, target_width):
        return mask  # No resize needed

    # print(f"Resizing mask from {(current_height, current_width)} to {(target_height, target_width)}")
    # Use INTER_NEAREST for masks to preserve integer labels
    resized_mask = cv2.resize(
        mask, (target_width, target_height), interpolation=cv2.INTER_NEAREST
    )
    return resized_mask


# --------------------------------------------------------------------------
# Tiling Utilities
# --------------------------------------------------------------------------


def generate_tiles(image_shape, tile_size, tile_overlap):
    """
    Generate tile coordinates (y_start, y_end, x_start, x_end) with overlap.
    Ensures full image coverage and that overlap does not exceed tile_overlap.
    Tiles at the edges will be made smaller if necessary to fit the image dimensions.

    Parameters:
        image_shape (tuple): (height, width) of the full image.
        tile_size (int): Desired size of the square tile.
        tile_overlap (int): Maximum desired overlap between adjacent tiles.

    Returns:
        list: List of tuples, each defining a tile's coordinates (y_start, y_end, x_start, x_end).
    """
    height, width = image_shape
    tiles = []

    if height <= 0 or width <= 0:
        return tiles  # No tiles for empty image

    # Adjust tile size if image is smaller
    actual_tile_h = min(tile_size, height)
    actual_tile_w = min(tile_size, width)

    # Adjust overlap: must be less than tile size and non-negative
    # Ensure overlap is not greater than the dimension itself minus 1
    actual_overlap_h = max(0, min(tile_overlap, actual_tile_h - 1))
    actual_overlap_w = max(0, min(tile_overlap, actual_tile_w - 1))

    # Calculate step size based on adjusted tile size and overlap
    step_y = actual_tile_h - actual_overlap_h
    step_x = actual_tile_w - actual_overlap_w

    # Ensure step is at least 1 to prevent infinite loops
    step_y = max(1, step_y)
    step_x = max(1, step_x)

    # --- Generate Start Positions ---
    y_starts = []
    y = 0
    while y < height:
        y_starts.append(y)
        # If a tile starting at y reaches or passes the image height, stop.
        if y + actual_tile_h >= height:
            break
        y += step_y
        # Safety break if step is somehow non-positive (shouldn't happen)
        if y <= y_starts[-1]:
            print(
                f"Warning: Step calculation resulted in non-positive step_y ({step_y}). Breaking y loop."
            )
            break

    # If the last start position doesn't allow coverage to the edge, add a final start position.
    # This final start position aligns the tile's end with the image's end.
    if y_starts and y_starts[-1] + actual_tile_h < height:
        final_y_start = height - actual_tile_h
        # Add only if it's different from the last start position
        if final_y_start > y_starts[-1]:
            y_starts.append(final_y_start)
    elif not y_starts and height > 0:  # Handle case where height <= tile_size
        y_starts = [0]

    # Similar logic for x start positions
    x_starts = []
    x = 0
    while x < width:
        x_starts.append(x)
        if x + actual_tile_w >= width:
            break
        x += step_x
        if x <= x_starts[-1]:
            print(
                f"Warning: Step calculation resulted in non-positive step_x ({step_x}). Breaking x loop."
            )
            break

    if x_starts and x_starts[-1] + actual_tile_w < width:
        final_x_start = width - actual_tile_w
        if final_x_start > x_starts[-1]:
            x_starts.append(final_x_start)
    elif not x_starts and width > 0:  # Handle case where width <= tile_size
        x_starts = [0]

    # Ensure start lists are unique (can happen if added start coincides with a step)
    # Sorting is not strictly necessary but keeps order predictable
    y_starts = sorted(list(set(y_starts)))
    x_starts = sorted(list(set(x_starts)))

    # --- Create Tile Coordinates ---
    for y_start in y_starts:
        for x_start in x_starts:
            # Calculate end coordinates, ensuring they don't exceed image boundaries
            y_end = min(y_start + actual_tile_h, height)
            x_end = min(x_start + actual_tile_w, width)

            # Ensure tile has valid positive dimensions before adding
            if y_end > y_start and x_end > x_start:
                tiles.append((y_start, y_end, x_start, x_end))

    # Final check for uniqueness, although the generation logic should minimize duplicates
    unique_tiles = list(dict.fromkeys(tiles))

    # print(f"Generated {len(unique_tiles)} unique tiles for shape {image_shape} (Tile: {tile_size}, Max Overlap: {tile_overlap})")
    return unique_tiles


def display_tile_progress(
    tiles_info, completed_tiles_indices, image_shape, current_tile_index=None
):
    """
    Display an ASCII grid showing tile processing progress in Jupyter/IPython.
    Updates in-place.

    Parameters:
        tiles_info (list): List of tile coordinate tuples.
        completed_tiles_indices (set): Set of indices of completed tiles.
        image_shape (tuple): (height, width) of the full image.
        current_tile_index (int, optional): Index of the tile currently being processed.
    """
    try:
        clear_output(wait=True)  # Clears the output cell in Jupyter/IPython

        height, width = image_shape
        if not tiles_info:
            print("\n--- Tile Processing Progress ---")
            print("No tiles to process.")
            return

        # Determine grid dimensions based on unique start coordinates
        y_starts = sorted(list(set([y for y, _, _, _ in tiles_info])))
        x_starts = sorted(list(set([x for _, _, x, _ in tiles_info])))
        grid_height = len(y_starts)
        grid_width = len(x_starts)

        if grid_height == 0 or grid_width == 0:
            print("\n--- Tile Processing Progress ---")
            print("Could not determine grid dimensions.")
            return

        # Create a mapping from (y_start, x_start) to tile index for grid population
        # This assumes generate_tiles produces tiles in a somewhat grid-like order
        tile_coord_to_index = {}
        temp_grid_map = {}
        y_start_map = {y: i for i, y in enumerate(y_starts)}
        x_start_map = {x: i for i, x in enumerate(x_starts)}

        for idx, (y, _, x, _) in enumerate(tiles_info):
            if (y, x) not in tile_coord_to_index:
                tile_coord_to_index[(y, x)] = idx
            # Map grid position to tile index
            y_grid_idx = y_start_map.get(y)
            x_grid_idx = x_start_map.get(x)
            if y_grid_idx is not None and x_grid_idx is not None:
                temp_grid_map[(y_grid_idx, x_grid_idx)] = idx

        grid = [["□"] * grid_width for _ in range(grid_height)]  # Initialize grid

        for r in range(grid_height):
            for c in range(grid_width):
                tile_idx = temp_grid_map.get((r, c))
                if tile_idx is not None:
                    if tile_idx == current_tile_index:
                        grid[r][c] = "P"  # Processing
                    elif tile_idx in completed_tiles_indices:
                        grid[r][c] = "✓"  # Completed
                    # else: remains '□' (Pending)

        total_tiles = len(tiles_info)
        completed_count = len(completed_tiles_indices)
        progress_percent = (
            (completed_count / total_tiles * 100) if total_tiles > 0 else 0
        )

        print("\n--- Tile Processing Progress ---")
        print(
            f"Image Size: {width}x{height} | Grid: {grid_width}x{grid_height} | Total Tiles: {total_tiles}"
        )
        print(f"Completed: {completed_count}/{total_tiles} ({progress_percent:.1f}%)")
        if current_tile_index is not None:
            print(f"Processing Tile: {current_tile_index + 1}")

        # Print the grid
        if grid_width > 0 and grid_height > 0:
            # Limit grid display size for very large grids
            max_display_width = 80
            max_display_height = 40
            display_grid_width = min(grid_width, max_display_width)
            display_grid_height = min(grid_height, max_display_height)

            print("┌" + "─" * (display_grid_width * 2 - 1) + "┐")
            for r in range(display_grid_height):
                row_str = " ".join(grid[r][:display_grid_width])
                if grid_width > max_display_width:
                    row_str += " ..."  # Indicate truncation
                print("│" + row_str + "│")
            if grid_height > max_display_height:
                print("." * (display_grid_width * 2 + 1))  # Indicate truncation
            print("└" + "─" * (display_grid_width * 2 - 1) + "┘")
            print("Legend: P = Processing, ✓ = Completed, □ = Pending\n")
        else:
            print("Grid display skipped (invalid dimensions).")

    except Exception as e:
        # Avoid crashing the main process if display fails
        print(f"Warning: Failed to display tile progress: {e}")


# --------------------------------------------------------------------------
# Segmentation Models
# --------------------------------------------------------------------------


def cellpose_segmentation(
    image_dict,
    output_dir,  # Note: Currently only used if save_mask_as_png=True
    membrane_channel_name=None,  # Name of membrane channel in image_dict (e.g., 'segmentation_channel')
    cytoplasm_channel_name=None,  # Name of cytoplasm channel in image_dict
    nucleus_channel_name=None,  # Name of nucleus channel in image_dict
    use_gpu=True,
    model="cyto3",  # Default Cellpose model
    custom_model=False,  # Set to True if 'model' is a path to a custom model file
    diameter=None,  # Cell diameter estimate (recommended)
    save_mask_as_png=False,  # Save Cellpose overlay PNG
):
    """
    Perform cell segmentation using Cellpose. Handles channel selection for Cellpose input.

    Parameters:
        image_dict (dict): Dict with images needed for segmentation (e.g., nuclei, membrane/cyto).
        output_dir (str or Path): Base output directory (used for saving PNG).
        membrane_channel_name (str, optional): Key in image_dict for membrane channel.
        cytoplasm_channel_name (str, optional): Key in image_dict for cytoplasm channel.
        nucleus_channel_name (str): Key in image_dict for nucleus channel.
        use_gpu (bool): Whether to use GPU.
        model (str): Cellpose model name (e.g., 'cyto3', 'nuclei') or path to custom model.
        custom_model (bool): True if 'model' is a path.
        diameter (float, optional): Estimated cell diameter.
        save_mask_as_png (bool): If True, saves Cellpose's diagnostic PNG.

    Returns:
        tuple: (masks, flows, styles) from cellpose.eval() or (None, None, None) on error.
               'masks' is the 2D labeled mask array.
    """
    if not nucleus_channel_name or nucleus_channel_name not in image_dict:
        print(
            f"Error: Nucleus channel '{nucleus_channel_name}' not provided or not found in image_dict."
        )
        return None, None, None

    # Determine Cellpose channels argument based on provided channel names
    # Cellpose channel mapping: 0=gray, 1=red(membrane), 2=green(cyto), 3=blue(nucleus)
    channels_arg = [0, 0]  # Default to grayscale (nucleus only)
    input_image = None

    # Prepare the input image (can be 2D grayscale or 3D RGB-like)
    nucleus_img = image_dict[nucleus_channel_name]
    membrane_img = image_dict.get(membrane_channel_name)
    cytoplasm_img = image_dict.get(cytoplasm_channel_name)

    if cytoplasm_img is not None:
        print("Using Cytoplasm (Green=2) and Nucleus (Blue=3) channels for Cellpose.")
        channels_arg = [2, 3]
        # Create a 3D array [H, W, C] -> R=0, G=cyto, B=nuc
        input_image = np.stack(
            [np.zeros_like(nucleus_img), cytoplasm_img, nucleus_img], axis=-1
        )
    elif membrane_img is not None:
        print("Using Membrane (Red=1) and Nucleus (Blue=3) channels for Cellpose.")
        channels_arg = [1, 3]
        # Create a 3D array [H, W, C] -> R=memb, G=0, B=nuc
        input_image = np.stack(
            [membrane_img, np.zeros_like(nucleus_img), nucleus_img], axis=-1
        )
    else:
        print("Using only Nucleus channel (Grayscale=0) for Cellpose.")
        channels_arg = [0, 0]  # Grayscale mode
        input_image = nucleus_img  # Use the 2D nucleus image directly

    if input_image is None:
        print("Error: Failed to prepare input image for Cellpose.")
        return None, None, None

    # Run CellPose core function
    try:
        masks, flows, styles = run_cellpose(
            image=input_image,
            output_dir=output_dir,
            use_gpu=use_gpu,
            model=model,
            custom_model=custom_model,
            diameter=diameter,
            channels=channels_arg,
            save_mask_as_png=save_mask_as_png,
        )
        return masks, flows, styles
    except Exception as e:
        print(f"Error during Cellpose segmentation run: {e}")
        return None, None, None


def run_cellpose(
    image,
    output_dir,
    use_gpu=True,
    model="cyto3",
    custom_model=False,
    diameter=None,
    channels=[0, 0],
    save_mask_as_png=False,
):
    """
    Internal helper to initialize and run the Cellpose model evaluation.

    Parameters: (See cellpose_segmentation docstring)

    Returns:
        tuple: (masks, flows, styles) from model.eval().
    """
    print(
        f"Running Cellpose: model='{model}', custom={custom_model}, diameter={diameter}, channels={channels}, gpu={use_gpu}"
    )

    # Initialize model
    model_obj = None
    try:
        if custom_model:
            if not os.path.exists(model):
                raise FileNotFoundError(f"Custom Cellpose model not found at: {model}")
            # Use CellposeModel for custom models
            model_obj = cellpose_models.CellposeModel(
                pretrained_model=model, gpu=use_gpu
            )
            print(f"Loaded custom Cellpose model from: {model}")
        else:
            # Use Cellpose or CellposeModel based on model type (nuclei often needs Cellpose)
            # Check if model is 'nuclei' or similar that might require the base class
            if model in ["nuclei"]:  # Add other models if needed
                model_obj = cellpose_models.Cellpose(model_type=model, gpu=use_gpu)
            else:  # Default to CellposeModel for 'cyto', 'cyto2', 'cyto3' etc.
                model_obj = cellpose_models.CellposeModel(model_type=model, gpu=use_gpu)
            print(f"Initialized Cellpose model: {model}")
    except Exception as e:
        print(f"Error initializing Cellpose model '{model}': {e}")
        raise  # Re-raise error

    # Run evaluation
    masks, flows, styles, diams = None, None, None, None  # Initialize
    try:
        # The eval signature might differ slightly based on Cellpose version and class used
        if isinstance(model_obj, cellpose_models.Cellpose):
            masks, flows, styles, diams = model_obj.eval(
                image, diameter=diameter, channels=channels, do_3D=False
            )
            print(
                f"Cellpose segmentation complete. Found {np.max(masks)} objects. Est. diameter: {diams:.2f}"
            )
        elif isinstance(model_obj, cellpose_models.CellposeModel):
            # CellposeModel.eval might not return diameter directly in all versions
            eval_output = model_obj.eval(
                image, diameter=diameter, channels=channels, do_3D=False
            )
            if len(eval_output) == 4:  # Older versions might return diams
                masks, flows, styles, diams = eval_output
                print(
                    f"Cellpose segmentation complete. Found {np.max(masks)} objects. Est. diameter: {diams:.2f}"
                )
            elif (
                len(eval_output) == 3
            ):  # Newer versions might return only masks, flows, styles
                masks, flows, styles = eval_output
                # Try to get diameter from the object if possible (might not always be set post-eval)
                diams = getattr(
                    model_obj, "diam_labels", diameter
                )  # Use provided diameter if not found
                print(f"Cellpose segmentation complete. Found {np.max(masks)} objects.")
            else:
                raise TypeError(
                    f"Unexpected output from CellposeModel.eval: {eval_output}"
                )

        else:
            raise TypeError("Unsupported Cellpose model object type.")

    except Exception as e:
        print(f"Error during Cellpose model.eval: {e}")
        raise  # Re-raise error

    # Save output PNG if requested
    if save_mask_as_png:
        try:
            from cellpose import io as cellpose_io  # Use cellpose io module

            output_dir_path = pathlib.Path(output_dir)
            output_dir_path.mkdir(parents=True, exist_ok=True)
            # Construct filename
            base_fname = (
                f"cellpose_seg_{pathlib.Path(model).stem if custom_model else model}"
            )
            filename = output_dir_path / f"{base_fname}_seg.png"

            # Use cellpose_io.save_masks or masks_flows_to_seg depending on needs
            # masks_flows_to_seg creates the standard diagnostic image
            # Need diameter value (estimated or provided)
            effective_diameter = diams if diams is not None else diameter
            if effective_diameter is None:
                print("Warning: Diameter not available for saving PNG. Using default.")
                effective_diameter = 30.0  # Default for saving function if unknown

            # Ensure image is suitable for saving (e.g., rescale if needed, handle multi-channel input)
            # masks_flows_to_seg expects an image that can be displayed (e.g., uint8 or float scaled 0-1)
            # It might handle the input 'image' directly, but let's prepare a displayable version
            from cellpose import utils

            img_display = image  # Use the input image directly first
            if image.ndim == 3 and image.shape[-1] == 3:  # RGB-like input
                # Cellpose plotting utils often work with channel-first or specific channel indices
                # Let's try to reconstruct a displayable image based on channels_arg
                if channels == [2, 3]:  # Cyto(G), Nuc(B)
                    img_display = image[
                        :, :, [2, 1, 0]
                    ]  # BGR for display? Or use utils.format_image
                elif channels == [1, 3]:  # Memb(R), Nuc(B)
                    img_display = image[:, :, [2, 0, 1]]  # BGR for display?
                # Or just use the first channel if grayscale
            elif image.ndim == 2:
                img_display = image

            # Normalize image for display if it's not uint8
            if img_display.dtype != np.uint8:
                img_display = utils.normalize99(img_display) * 255
                img_display = img_display.astype(np.uint8)

            cellpose_io.masks_flows_to_seg(
                img_display, masks, flows, effective_diameter, filename, channels
            )
            print(f"Saved Cellpose segmentation overlay to: {filename}")

        except ImportError:
            print(
                "Warning: Could not import cellpose.io or cellpose.utils. Skipping saving PNG."
            )
        except Exception as e:
            print(f"Warning: Error saving Cellpose PNG: {e}")

    return masks, flows, styles


def load_mesmer_model(model_dir):
    """Deprecated compatibility shim retained for external callers."""
    print(
        "Mesmer model loading is no longer supported. "
        "SPACEc now uses Cellpose-only segmentation."
    )
    return None


def mesmer_segmentation(
    nuclei_image,
    membrane_image,  # Can be None for nuclear-only segmentation
    image_mpp=0.5,  # Microns per pixel - important for Mesmer performance
    plot_predictions=False,
    compartment="whole-cell",  # 'whole-cell' or 'nuclear'
    model_path="./models",  # Base directory for Mesmer model download/load
):
    """
    Backward-compatible alias for Mesmer requests, now fulfilled with Cellpose.
    """
    print(
        "Mesmer backend is deprecated and has been removed. "
        "Falling back to Cellpose for segmentation."
    )

    if nuclei_image.ndim != 2:
        print(f"Error: Nuclei image must be 2D, but got shape {nuclei_image.shape}")
        return None

    if membrane_image is not None and membrane_image.shape != nuclei_image.shape:
        print(
            f"Error: Nuclei ({nuclei_image.shape}) and membrane ({membrane_image.shape}) images must have the same shape."
        )
        return None

    if compartment not in {"whole-cell", "nuclear"}:
        print(
            f"Error: Invalid compartment: {compartment}. Choose 'whole-cell' or 'nuclear'."
        )
        return None

    seg_dict = {"_nuclei": nuclei_image}
    membrane_channel = None
    if compartment == "whole-cell" and membrane_image is not None:
        seg_dict["_membrane"] = membrane_image
        membrane_channel = "_membrane"

    try:
        segmented_mask, _, _ = cellpose_segmentation(
            image_dict=seg_dict,
            output_dir=pathlib.Path(model_path),
            membrane_channel_name=membrane_channel,
            cytoplasm_channel_name=None,
            nucleus_channel_name="_nuclei",
            use_gpu=True,
            model="cyto3" if membrane_channel else "nuclei",
            custom_model=False,
            diameter=None,
            save_mask_as_png=False,
        )
    except Exception as e:
        print(f"Error during Cellpose fallback for Mesmer alias: {e}")
        return None

    if segmented_mask is None:
        print("Warning: Cellpose fallback returned no mask for Mesmer alias.")
        return None

    segmented_mask = segmented_mask.astype(np.int32)
    if plot_predictions:
        try:
            if membrane_channel:
                combined_image = np.stack([nuclei_image, membrane_image], axis=-1)
                channel_colors = ["blue", "green"]
            else:
                combined_image = np.stack([nuclei_image], axis=-1)
                channel_colors = ["blue"]
            rgb_images = create_rgb_image(
                np.expand_dims(combined_image, axis=0), channel_colors=channel_colors
            )
            overlay_data = make_outline_overlay(
                rgb_data=rgb_images,
                predictions=np.expand_dims(segmented_mask, axis=0),
            )

            fig, ax = plt.subplots(1, 2, figsize=(15, 7))
            ax[0].imshow(rgb_images[0, ...])
            ax[0].axis("off")
            ax[0].set_title("Cellpose Input Preview")
            ax[1].imshow(overlay_data[0, ...])
            ax[1].axis("off")
            ax[1].set_title(f"Cellpose ({compartment})")
            plt.tight_layout()
            plt.show()
        except Exception as e:
            print(f"Warning: Error during fallback plotting: {e}")

    gc.collect()
    return segmented_mask


# --------------------------------------------------------------------------
# Stitching and Postprocessing
# --------------------------------------------------------------------------


def stitch_masks(tiles_info, tile_masks, full_shape, tile_overlap=32, sigma=128):
    """Stitch multiple segmentation masks from overlapping tiles with confidence-based blending.

    This function combines segmentation masks from multiple tiles into a single cohesive mask.
    It handles overlapping regions using a confidence-based approach and resolves conflicts
    between object labels across tile boundaries.

    Parameters
    ----------
    tiles_info : list of tuple
        List of tile coordinates, each tuple containing (y_start, y_end, x_start, x_end)
        defining the position of each tile in the full image.
    tile_masks : list of ndarray
        List of 2D integer-labeled segmentation masks, one for each tile.
        Must match the length of tiles_info. Each mask should have dimensions
        matching its corresponding tile coordinates.
    full_shape : tuple
        Shape (height, width) of the final stitched mask.
    tile_overlap : int, optional
        Overlap between adjacent tiles in pixels, by default 32.
        Used to create smooth transitions between tiles.
    sigma : float, optional
        Standard deviation for Gaussian smoothing of confidence maps,
        by default 128. Larger values create smoother transitions.

    Returns
    -------
    ndarray
        2D integer-labeled segmentation mask of shape full_shape.
        Background is 0, objects are labeled with consecutive positive integers.
        Returns zeros if no valid tiles are found.

    Notes
    -----
    Algorithm Steps:
    1. Validates input and initializes output arrays
    2. First pass: Prepares tiles and calculates confidence maps
        - Offsets labels to avoid conflicts
        - Creates confidence maps for overlap regions
    3. Second pass: Merges tiles with conflict resolution
        - Resolves overlapping objects using majority voting
        - Updates pixels based on confidence values
    4. Final relabeling to ensure consecutive labels

    Memory Optimization:
    - Uses caching for confidence maps of similar tile shapes
    - Processes tiles sequentially to limit memory usage
    - Cleans up intermediate arrays explicitly

    Performance Considerations:
    - Gaussian smoothing of confidence maps can be a bottleneck
    - Label conflict resolution scales with number of overlapping objects
    - Memory usage scales with full image size and tile overlap

    See Also
    --------
    scipy.ndimage.gaussian_filter : Used for confidence map smoothing
    skimage.morphology.label : Similar functionality for connected component labeling
    """

    print(
        f"Stitching {len(tiles_info)} masks with overlap={tile_overlap}, sigma={sigma}..."
    )
    start_time = time.time()  # Track stitching time

    # --- Input Validation ---
    if not tiles_info or not tile_masks or len(tiles_info) != len(tile_masks):
        print("Warning: Invalid input tiles_info or tile_masks. Returning empty mask.")
        return np.zeros(full_shape, dtype=np.int32)

    # Ensure tile_overlap is valid
    tile_overlap = max(0, int(tile_overlap))

    # --- Pre-allocation ---
    # Use float64 for accumulation to avoid potential overflow with large labels, then cast later if needed
    # Using int32 directly might be slightly faster if max_label doesn't exceed 2^31, but safer with int64 intermediate.
    # Let's stick to int32 as label counts rarely exceed this, and it matches skimage output.
    full_mask = np.zeros(full_shape, dtype=np.int32)
    # Confidence map stores the confidence of the label at each pixel
    confidence_map = np.zeros(full_shape, dtype=np.float32)

    # --- Confidence Map Cache ---
    confidence_cache = {}

    # --- First Pass: Prepare Tiles (Offset Labels and Get Confidence) ---
    print("  Preparing tiles (label offset and confidence)...")
    processed_tiles = []
    max_label = 0
    skipped_tiles = 0

    for i in range(len(tiles_info)):
        coords = tiles_info[i]
        tile_mask = tile_masks[i]

        # Skip if tile_mask is None or empty
        if tile_mask is None or tile_mask.size == 0 or not np.any(tile_mask):
            skipped_tiles += 1
            continue

        y_start, y_end, x_start, x_end = coords
        h, w = tile_mask.shape[:2]  # Use shape from actual mask

        # Ensure mask dimensions match coordinate dimensions
        if h != (y_end - y_start) or w != (x_end - x_start):
            print(
                f"Warning: Tile {i} mask shape {tile_mask.shape} mismatch with coords {coords}. Skipping."
            )
            skipped_tiles += 1
            continue

        # Copy mask to avoid modifying originals and offset labels
        tile_mask = tile_mask.copy()
        valid_mask_pixels = tile_mask > 0
        if max_label > 0 and np.any(valid_mask_pixels):
            tile_mask[valid_mask_pixels] += max_label

        current_max = tile_mask.max()
        max_label = max(max_label, current_max)

        # --- Calculate or Retrieve Confidence Map ---
        # Cache key based on actual tile shape
        cache_key = (
            h,
            w,
            tile_overlap,
            sigma,
            y_start > 0,
            y_end < full_shape[0],
            x_start > 0,
            x_end < full_shape[1],
        )
        if cache_key not in confidence_cache:
            # Create a confidence map: 1.0 in center, ramps down towards edges that overlap
            conf_local = np.ones((h, w), dtype=np.float32)
            overlap_y = min(
                tile_overlap, h // 2
            )  # Ensure overlap doesn't exceed half the tile dim
            overlap_x = min(tile_overlap, w // 2)

            # Create ramps (only if overlap > 0)
            if overlap_y > 0:
                ramp_y = np.linspace(0.0, 1.0, overlap_y, dtype=np.float32)
                if y_start > 0:  # Top edge needs ramp up
                    conf_local[:overlap_y, :] *= ramp_y[:, np.newaxis]
                if y_end < full_shape[0]:  # Bottom edge needs ramp down
                    conf_local[h - overlap_y :, :] *= ramp_y[::-1][:, np.newaxis]

            if overlap_x > 0:
                ramp_x = np.linspace(0.0, 1.0, overlap_x, dtype=np.float32)
                if x_start > 0:  # Left edge needs ramp up
                    conf_local[:, :overlap_x] *= ramp_x[np.newaxis, :]
                if x_end < full_shape[1]:  # Right edge needs ramp down
                    conf_local[:, w - overlap_x :] *= ramp_x[::-1][np.newaxis, :]

            # Apply Gaussian smoothing (potential bottleneck)
            # sigma/4 is used based on the original code's heuristic
            smooth_sigma = sigma / 4.0
            if smooth_sigma > 0:
                conf_local = gaussian_filter(
                    conf_local, sigma=smooth_sigma, mode="constant", cval=0.0
                )

            # Normalize confidence map (prevents issues if smoothing pushes max slightly > 1)
            max_conf = conf_local.max()
            if max_conf > 0:
                conf_local /= max_conf
            else:
                conf_local[:] = 0  # Avoid NaN if max is 0

            # Clip values to ensure they are within [0, 1] after filtering/normalization
            np.clip(conf_local, 0.0, 1.0, out=conf_local)

            confidence_cache[cache_key] = conf_local
        # --- End Confidence Map Calculation ---

        processed_tiles.append((coords, tile_mask, confidence_cache[cache_key]))

    if skipped_tiles > 0:
        print(f"  Skipped {skipped_tiles} empty or invalid tiles.")
    if not processed_tiles:
        print(
            "Warning: No valid tiles found to process after first pass. Returning empty mask."
        )
        return np.zeros(full_shape, dtype=np.int32)
    print(
        f"  Processed {len(processed_tiles)} tiles in first pass. Max label offset: {max_label}"
    )

    # --- Second Pass: Merge Tiles with Conflict Resolution ---
    print("  Merging tiles...")
    for coords, tile_mask, confidence in processed_tiles:
        y_start, y_end, x_start, x_end = coords

        # Get views into the full mask and confidence map
        region_mask = full_mask[y_start:y_end, x_start:x_end]
        region_conf = confidence_map[y_start:y_end, x_start:x_end]

        # --- Conflict Resolution ---
        # Identify pixels in the current tile that have labels (value > 0)
        tile_pixels_with_labels = tile_mask > 0

        # Find where these pixels overlap with existing labels in the full_mask region
        conflicting_pixels = tile_pixels_with_labels & (region_mask > 0)

        if np.any(conflicting_pixels):
            # Get labels from the tile and the existing region at conflicting pixels
            tile_labels_at_conflict = tile_mask[conflicting_pixels]
            region_labels_at_conflict = region_mask[conflicting_pixels]

            # Iterate through unique *tile* labels involved in conflicts
            unique_tile_labels_in_conflict = np.unique(tile_labels_at_conflict)

            for tile_L in unique_tile_labels_in_conflict:
                # Find where *this specific tile label* causes conflicts
                current_conflict_mask = conflicting_pixels & (tile_mask == tile_L)
                if not np.any(current_conflict_mask):
                    continue

                # Get the existing region labels that conflict with this tile_L
                overlapping_region_labels = region_mask[current_conflict_mask]

                # Find the most common *region* label overlapping with this *tile* label
                # Using bincount can be faster than unique if max label isn't excessively large
                try:
                    counts = np.bincount(overlapping_region_labels)
                    if counts.size > 0:
                        most_common_region_label = np.argmax(counts)
                        max_count = counts[most_common_region_label]

                        # If a region label is significantly present (>50% overlap),
                        # merge the current tile label (tile_L) into that region label.
                        # np.count_nonzero is faster than sum() for boolean arrays
                        if (
                            max_count > 0
                            and (max_count / np.count_nonzero(current_conflict_mask))
                            >= 0.5
                        ):
                            # Update the tile_mask *in place* for subsequent steps
                            tile_mask[tile_mask == tile_L] = most_common_region_label
                            # No need to break, continue checking other tile labels in conflict
                except (IndexError, ValueError) as e:
                    # Handle potential issues with bincount if labels are negative or too large
                    print(
                        f"Warning: Error during bincount conflict resolution for tile label {tile_L}: {e}. Falling back to slower unique."
                    )
                    unique_overlaps, counts = np.unique(
                        overlapping_region_labels, return_counts=True
                    )
                    if counts.size > 0:
                        max_idx = np.argmax(counts)
                        if (
                            counts[max_idx] / np.count_nonzero(current_conflict_mask)
                        ) >= 0.5:
                            tile_mask[tile_mask == tile_L] = unique_overlaps[max_idx]

        # --- Apply Update ---
        # Find pixels where the current tile has a label AND its confidence is higher
        # than the existing confidence in the region.
        update_mask = (tile_mask > 0) & (confidence > region_conf)

        # Apply the update using boolean indexing (generally efficient in NumPy)
        region_mask[update_mask] = tile_mask[update_mask]
        region_conf[update_mask] = confidence[update_mask]

    # Clean up intermediate list and cache
    del processed_tiles, confidence_cache
    gc.collect()

    # --- Final Relabeling ---
    print("  Relabeling final mask...")
    # Check if there are any labels before relabeling
    max_final_label = full_mask.max()
    if max_final_label > 0:
        # relabel_sequential ensures consecutive labels starting from 1
        full_mask, _, _ = relabel_sequential(full_mask)
        print(f"  Relabeling complete. Final max label: {full_mask.max()}")
    else:
        print("  Skipping relabeling as the final mask is empty.")

    end_time = time.time()
    print(f"Stitching finished in {end_time - start_time:.2f} seconds.")

    return full_mask


def remove_border_objects(mask):
    """
    Remove labeled objects in a mask that directly touch its borders.
    Assumes background is labeled as 0.
    """
    # Get unique labels along the four borders
    border_labels = set(np.unique(mask[0, :])).union(np.unique(mask[-1, :]))
    border_labels = border_labels.union(np.unique(mask[:, 0])).union(
        np.unique(mask[:, -1])
    )
    border_labels.discard(0)  # Keep background intact
    for lbl in border_labels:
        mask[mask == lbl] = 0
    return mask


# --------------------------------------------------------------------------
# Feature Extraction
# --------------------------------------------------------------------------


def extract_features(
    image_dict,
    segmentation_masks,
    channels_to_quantify,
    output_file,
    size_cutoff=0,
    # Tiling parameters for intensity calculation
    use_tiling_for_intensity=True,
    tile_size=2048,
    tile_overlap=128,
    memory_limit_gb=4,  # Approx. memory limit per channel before tiling intensity calc.
):
    """Extract morphological and intensity features from segmented images with memory optimization.

    This function performs feature extraction in multiple stages:
    1. Calculates morphological features using regionprops
    2. Filters objects based on size
    3. Creates a filtered mask
    4. Calculates mean intensities (with optional tiling)
    5. Combines and saves the features

    Parameters
    ----------
    image_dict : dict
        Dictionary mapping channel names to 2D image arrays. Images should be
        single-channel arrays with matching dimensions.
    segmentation_masks : ndarray
        2D integer-labeled segmentation mask. Background should be 0,
        objects labeled with consecutive positive integers.
    channels_to_quantify : list
        List of channel names in image_dict for intensity quantification.
        These names must exist as keys in image_dict.
    output_file : str or Path
        Path where the output CSV file will be saved.
    size_cutoff : int, optional
        Minimum object area in pixels to include in analysis, by default 0.
        Objects smaller than this are filtered out based on regionprops area.
    use_tiling_for_intensity : bool, optional
        Whether to enable tiled processing for intensity calculations,
        by default True. Recommended for large images.
    tile_size : int, optional
        Size of tiles in pixels for intensity calculation when tiling is used,
        by default 2048.
    tile_overlap : int, optional
        Overlap between adjacent tiles in pixels, by default 128.
        Prevents edge artifacts in tiled processing.
    memory_limit_gb : float, optional
        Memory threshold in GB per channel that triggers tiled processing,
        by default 4. Adjust based on available system memory.

    Returns
    -------
    pandas.DataFrame or None
        DataFrame containing extracted features if successful, with columns:
        - Morphological: 'label', 'y', 'x', 'area', 'eccentricity', 'perimeter',
          'convex_area', 'axis_major_length', 'axis_minor_length'
        - Intensity: mean intensity for each channel in channels_to_quantify
        Returns None if processing fails or no objects are found.

    Notes
    -----
    Memory Optimization:
    - Morphological features are calculated on the full mask at once
    - Intensity calculations can be tiled for large images
    - Intermediate results are explicitly cleaned up
    - Uses float64 for intensity calculations to maintain precision

    Performance Considerations:
    - Large masks may cause memory issues during morphological calculation
    - Tiling adds overhead but reduces peak memory usage
    - Consider reducing tile_size if memory errors occur
    - GPU memory is not used directly but temp arrays may impact GPU memory

    File Handling:
    - Creates output directory if it doesn't exist
    - Saves empty CSV if no objects are found
    - Attempts to save partial results (morphology only) on failure

    See Also
    --------
    skimage.measure.regionprops_table : Used for morphological feature extraction
    numpy.bincount : Used for efficient intensity calculation
    """
    print("--- Starting Feature Extraction ---")
    output_file = pathlib.Path(output_file)  # Ensure Path object

    if (
        segmentation_masks is None
        or np.prod(segmentation_masks.shape) == 0
        or np.max(segmentation_masks) == 0
    ):
        print(
            "Error: Segmentation mask is empty, None, or contains no labeled objects."
        )
        output_file.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame().to_csv(output_file, index=False)
        print(f"Created empty features file: {output_file}")
        return None

    segmentation_masks = segmentation_masks.squeeze().astype(np.int32)
    img_h, img_w = segmentation_masks.shape

    # Ensure size_cutoff is non-negative
    size_cutoff = max(0, size_cutoff)

    # --- 1. Calculate Morphological Features (on full mask before size filtering) ---
    print("Calculating morphological features...")
    props_df = None
    try:
        # Run regionprops on the original mask
        if np.max(segmentation_masks) > 0:
            props = regionprops_table(
                segmentation_masks,  # Use the original mask
                properties=(
                    "label",
                    "centroid",
                    "area",
                    "eccentricity",
                    "perimeter",
                    "convex_area",
                    "axis_major_length",
                    "axis_minor_length",
                ),
            )
            props_df = pd.DataFrame(props)
            props_df.rename(
                columns={"centroid-0": "y", "centroid-1": "x"}, inplace=True
            )
            print(f"Calculated initial morphology for {len(props_df)} objects.")

            # --- 2. Filter Small Objects based on regionprops area ---
            print(f"Filtering objects with area < {size_cutoff} pixels...")
            props_df = props_df[
                props_df["area"] >= size_cutoff
            ].copy()  # Filter based on area
            props_df.set_index("label", inplace=True)  # Set index after filtering

            if props_df.empty:
                print("No objects remaining after size filtering.")
                # Create empty file
                output_file.parent.mkdir(parents=True, exist_ok=True)
                pd.DataFrame().to_csv(output_file, index=False)
                print(f"Created empty features file: {output_file}")
                return None
            print(f"Found {len(props_df)} objects after size filtering.")
        else:
            print("No objects found in the initial mask for morphology calculation.")
            # Create empty file
            output_file.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame().to_csv(output_file, index=False)
            print(f"Created empty features file: {output_file}")
            return None

    except MemoryError as e:
        print(f"MemoryError calculating morphological features on full mask: {e}")
        print(
            "Consider using libraries designed for out-of-core morphology if this persists."
        )
        return None
    except Exception as e:
        print(f"Error calculating morphological features or filtering: {e}")
        return None

    # --- 3. Create Filtered Mask (filterimg) ---
    # This mask contains only the objects that passed the size filter.
    print("Creating filtered mask for intensity calculation...")
    filterimg = None  # Initialize
    max_label = 0  # Initialize
    final_nucleus_ids = props_df.index.to_numpy(
        dtype=np.int32
    )  # Get labels that passed filtering

    try:
        # Create a mapping array to zero out small labels based on props_df index
        original_max_label = np.max(segmentation_masks)
        filter_map = np.zeros(original_max_label + 1, dtype=segmentation_masks.dtype)
        # Only keep labels that are in the filtered props_df index
        valid_labels_in_mask = final_nucleus_ids[
            final_nucleus_ids <= original_max_label
        ]
        filter_map[valid_labels_in_mask] = valid_labels_in_mask  # Keep valid labels

        filterimg = filter_map[
            segmentation_masks
        ]  # Apply mapping to create filtered mask

        if len(final_nucleus_ids) > 0:
            # Use the max label from the *filtered* set for bincount minlength
            max_label = int(np.max(final_nucleus_ids))
        else:
            max_label = (
                0  # Should not happen if props_df wasn't empty, but safety check
            )

    except MemoryError:
        print(
            "MemoryError creating the filtered mask. Image/Mask might be too large for this step."
        )
        del props_df
        gc.collect()
        return None
    except Exception as e:
        print(f"Error creating filtered mask: {e}")
        del props_df
        gc.collect()
        return None

    # --- 4. Calculate Mean Intensities (Potentially Tiled, using filterimg) ---
    print("Calculating mean intensities...")
    mean_intensity_data = {}

    # Determine if tiling is needed for intensity calculation
    tiling_needed = False
    if channels_to_quantify:
        first_channel_name = channels_to_quantify[0]
        if first_channel_name in image_dict:
            dtype_size = image_dict[first_channel_name].dtype.itemsize
            estimated_gb_per_channel = (img_h * img_w * dtype_size) / (1024**3)
            tiling_needed = use_tiling_for_intensity and (
                estimated_gb_per_channel > memory_limit_gb
            )
        else:
            print(
                f"Warning: First channel '{first_channel_name}' for quantification not found in image_dict."
            )
            channels_to_quantify = []  # Avoid processing if first channel missing

    tiles_info = None
    if tiling_needed:
        print(
            f"Memory estimate ({estimated_gb_per_channel:.2f} GB) exceeds limit ({memory_limit_gb} GB). Using tiling for intensity."
        )
        tiles_info = generate_tiles(filterimg.shape, tile_size, tile_overlap)
        print(f"Generated {len(tiles_info)} tiles for intensity calculation.")
    elif channels_to_quantify:
        print("Processing intensities on full images.")
    else:
        print("Skipping intensity calculation (no valid channels specified).")

    # Process each channel using the 'filterimg'
    for chan in tqdm(channels_to_quantify, desc="Processing channels"):
        if chan not in image_dict:
            print(
                f"Warning: Channel '{chan}' not found in image_dict. Filling with NaN."
            )
            # Use props_df index length for consistency
            mean_intensity_data[chan] = np.full(len(props_df), np.nan)
            continue

        chan_data = None  # Ensure variable exists for finally block
        try:
            chan_data = image_dict[chan]
            if chan_data.shape != filterimg.shape:
                print(
                    f"Warning: Shape mismatch for channel '{chan}' ({chan_data.shape}) vs mask ({filterimg.shape}). Resizing channel."
                )
                chan_data = cv2.resize(
                    chan_data, (img_w, img_h), interpolation=cv2.INTER_LINEAR
                )
                if chan_data.shape != filterimg.shape:
                    raise ValueError(
                        f"Channel resize failed for '{chan}'. Expected {filterimg.shape}, got {chan_data.shape}."
                    )

            # Initialize sums and counts for this channel, ensure large enough for max_label from filtered set
            channel_sums = np.zeros(max_label + 1, dtype=np.float64)
            channel_counts = np.zeros(max_label + 1, dtype=np.int64)

            if tiling_needed and tiles_info:
                # --- Tiled Intensity Calculation ---
                for y_start, y_end, x_start, x_end in tiles_info:
                    try:
                        # Use filterimg for masking
                        tile_mask_view = filterimg[y_start:y_end, x_start:x_end]
                        tile_chan_view = chan_data[y_start:y_end, x_start:x_end]

                        if tile_mask_view.size == 0 or tile_chan_view.size == 0:
                            continue

                        # Use minlength=max_label + 1 based on filtered labels
                        tile_sums = np.bincount(
                            tile_mask_view.ravel(),
                            weights=tile_chan_view.ravel(),
                            minlength=max_label + 1,
                        )
                        tile_counts = np.bincount(
                            tile_mask_view.ravel(), minlength=max_label + 1
                        )

                        channel_sums += tile_sums
                        channel_counts += tile_counts
                        del tile_mask_view, tile_chan_view, tile_sums, tile_counts
                    except IndexError:
                        print(
                            f"Warning: Tile coordinates caused IndexError during intensity calculation for channel '{chan}'. Skipping tile."
                        )
                        continue
                    except Exception as tile_e:
                        print(
                            f"Warning: Error processing tile for channel '{chan}': {tile_e}. Skipping tile."
                        )
                        continue
            else:
                # --- Full Image Intensity Calculation ---
                if filterimg.size > 0 and chan_data.size > 0:
                    # Use filterimg here
                    channel_sums = np.bincount(
                        filterimg.ravel(),
                        weights=chan_data.ravel(),
                        minlength=max_label + 1,
                    )
                    channel_counts = np.bincount(
                        filterimg.ravel(), minlength=max_label + 1
                    )
                else:
                    print(
                        f"Warning: Empty filtered mask or channel data for '{chan}'. Skipping full image calculation."
                    )

            # Calculate mean intensity only for the labels present in props_df (final_nucleus_ids)
            # Ensure indices are valid before accessing sums/counts
            valid_indices_mask = (final_nucleus_ids >= 0) & (
                final_nucleus_ids <= max_label
            )
            valid_final_ids = final_nucleus_ids[valid_indices_mask]

            if len(valid_final_ids) == 0:
                print(
                    f"Warning: No valid labels found for channel '{chan}' after index check."
                )
                mean_intensity_data[chan] = np.full(
                    len(props_df), np.nan
                )  # Match props_df length
                continue

            # Get sums/counts only for the valid labels that passed size filtering
            counts_for_final_labels = channel_counts[valid_final_ids]
            sums_for_final_labels = channel_sums[valid_final_ids]

            # Initialize result array matching the length of props_df (filtered objects)
            mean_channel_intensity = np.full(len(props_df), np.nan, dtype=np.float64)

            # Calculate mean where counts > 0
            valid_counts_mask_local = counts_for_final_labels > 0
            # Ensure we only calculate for labels that actually had counts
            mean_values = np.full(
                np.sum(valid_counts_mask_local), np.nan
            )  # Initialize output for division
            np.divide(
                sums_for_final_labels[valid_counts_mask_local],
                counts_for_final_labels[valid_counts_mask_local],
                out=mean_values,
                where=valid_counts_mask_local,
            )  # Condition for division

            # Place calculated means into the correct positions in the result array
            # We need to map the results back based on the original index positions in final_nucleus_ids
            # Create a boolean mask aligned with final_nucleus_ids
            full_valid_mask = np.zeros(len(final_nucleus_ids), dtype=bool)
            full_valid_mask[valid_indices_mask] = valid_counts_mask_local

            mean_channel_intensity[full_valid_mask] = mean_values
            mean_intensity_data[chan] = mean_channel_intensity

        except MemoryError as e:
            print(f"\\nMemoryError processing channel '{chan}': {e}")
            mean_intensity_data[chan] = np.full(len(props_df), np.nan)
        except ValueError as e:
            print(f"\\nValueError processing channel '{chan}': {e}")
            mean_intensity_data[chan] = np.full(len(props_df), np.nan)
        except Exception as e:
            print(f"\\nError processing channel '{chan}': {e}")
            mean_intensity_data[chan] = np.full(len(props_df), np.nan)
        finally:
            if chan_data is not None:
                del chan_data
            gc.collect()

    # --- 5. Combine Features and Save ---
    print("Combining morphology and intensity features...")
    try:
        # filterimg no longer needed
        if filterimg is not None:
            del filterimg
        gc.collect()

        # Create DataFrame from intensity data using the filtered props_df index
        # Ensure the index matches props_df's index (which is 'label')
        mean_df = pd.DataFrame(mean_intensity_data, index=props_df.index)
        # mean_df.index.name = 'label' # Index already named 'label' from props_df

        # Join morphological features (already filtered) with mean intensities
        markers_df = props_df.join(
            mean_df, how="left"
        )  # props_df is already filtered by size

        if markers_df.isnull().values.any():
            nan_cols = markers_df.columns[markers_df.isnull().any()].tolist()
            print(
                f"Warning: Found NaN values in final features. Columns affected: {nan_cols}"
            )

        # Reorder columns (optional) - Use the globally defined morpho_cols
        present_morpho_cols = [col for col in morpho_cols if col in markers_df.columns]
        present_intensity_cols = [
            col for col in channels_to_quantify if col in markers_df.columns
        ]
        final_cols = present_morpho_cols + present_intensity_cols
        markers_df = markers_df.reindex(columns=final_cols)

        markers_df.reset_index(inplace=True)  # Make 'label' a column

        output_file.parent.mkdir(parents=True, exist_ok=True)
        markers_df.to_csv(output_file, index=False)
        print(
            f"Successfully saved features for {len(markers_df)} objects to {output_file}"
        )

    except Exception as e:
        print(f"Error combining features or saving CSV {output_file}: {e}")
        if props_df is not None:
            try:
                morpho_output_file = (
                    output_file.parent / f"{output_file.stem}_morphology_only.csv"
                )
                # Save the filtered props_df
                props_df.reset_index().to_csv(morpho_output_file, index=False)
                print(
                    f"Saved morphology-only features (after size filter) to {morpho_output_file}"
                )
            except Exception as save_e:
                print(f"Could not save morphology-only features: {save_e}")
        return None
    finally:
        if "props_df" in locals() and props_df is not None:
            del props_df
        if "mean_df" in locals() and mean_df is not None:
            del mean_df
        gc.collect()

    print("--- Feature Extraction Complete ---")
    return markers_df


# Make sure morpho_cols is defined (it was at the end of the first cell)
morpho_cols = [
    "y",
    "x",
    "area",
    "eccentricity",
    "perimeter",
    "convex_area",
    "axis_major_length",
    "axis_minor_length",
]


# --------------------------------------------------------------------------
# Main Segmentation Function
# --------------------------------------------------------------------------


def _perform_segmentation(
    seg_dict,  # Dict with channels needed for this tile/image
    seg_method,
    output_dir,  # For potential saving inside sub-functions
    nuclei_channel_name,
    membrane_channel_name,  # Name of combined membrane channel or None
    cytoplasm_channel_name,  # Name of cytoplasm channel or None (Cellpose only)
    compartment,  # 'whole-cell' or 'nuclear' (Mesmer)
    plot_predictions,  # For Mesmer plotting
    model_path,  # For Mesmer model download/load
    use_gpu,  # For Cellpose/TF
    model,  # Cellpose model name/path
    custom_model,  # Cellpose flag
    diameter,  # Cellpose diameter
    save_mask_as_png,  # Cellpose flag
    image_mpp=0.5,  # For Mesmer
):
    """Internal helper function to perform cell segmentation using either Mesmer or Cellpose.

    Parameters
    ----------
    seg_dict : dict
        Dictionary containing channel images needed for segmentation
    seg_method : str
        Segmentation method to use ('mesmer' or 'cellpose')
    output_dir : str or Path
        Directory for saving output files from segmentation
    nuclei_channel_name : str
        Name of the nuclei channel in seg_dict
    membrane_channel_name : str or None
        Name of the combined membrane channel in seg_dict, or None
    cytoplasm_channel_name : str or None
        Name of cytoplasm channel in seg_dict (Cellpose only), or None
    compartment : str
        Segmentation compartment for Mesmer ('whole-cell' or 'nuclear')
    plot_predictions : bool
        Whether to plot Mesmer predictions
    model_path : str or Path
        Path for Mesmer model download/loading
    use_gpu : bool
        Whether to use GPU acceleration
    model : str
        Cellpose model name or path to custom model
    custom_model : bool
        Whether model parameter is a path to custom Cellpose model
    diameter : float or None
        Expected cell diameter in pixels for Cellpose
    save_mask_as_png : bool
        Whether to save Cellpose overlay as PNG
    image_mpp : float, optional
        Microns per pixel for Mesmer, by default 0.5

    Returns
    -------
    ndarray or None
        2D integer-labeled segmentation mask if successful, None otherwise

    Notes
    -----
    The function handles both Mesmer and Cellpose segmentation methods:
    - For Mesmer: Requires nuclei image, optional membrane image
    - For Cellpose: Requires nuclei image, optional membrane/cytoplasm images

    The output mask is guaranteed to be 2D and of type int32 if successful.
    GPU memory is cleared after segmentation if GPU was used.
    """

    mask = None
    try:
        if seg_method == "mesmer":
            membrane_img = (
                seg_dict.get(membrane_channel_name) if membrane_channel_name else None
            )
            # Ensure nuclei image exists
            if nuclei_channel_name not in seg_dict:
                print(
                    f"Error: Nuclei channel '{nuclei_channel_name}' missing in seg_dict for Mesmer."
                )
                return None
            mask = mesmer_segmentation(
                nuclei_image=seg_dict[nuclei_channel_name],
                membrane_image=membrane_img,
                image_mpp=image_mpp,
                plot_predictions=plot_predictions,  # Plotting handled per-tile if called in loop
                compartment=compartment,
                model_path=model_path,
            )

        elif seg_method == "cellpose":
            # Note: cellpose_segmentation expects the *name* of the channels in seg_dict
            # It also needs the actual image data within seg_dict
            mask, _, _ = cellpose_segmentation(
                image_dict=seg_dict,  # Pass the dict containing tile images
                output_dir=output_dir,
                membrane_channel_name=membrane_channel_name,  # Pass name
                cytoplasm_channel_name=cytoplasm_channel_name,  # Pass name if available
                nucleus_channel_name=nuclei_channel_name,  # Pass name
                use_gpu=use_gpu,
                model=model,
                custom_model=custom_model,
                diameter=diameter,
                save_mask_as_png=save_mask_as_png,
            )
        else:
            print(f"Error: Unsupported segmentation method: {seg_method}")
            return None

        if mask is None:
            print(f"Warning: {seg_method} returned None.")
            return None

        # Ensure mask is 2D and integer type
        if mask.ndim > 2:
            mask = np.squeeze(mask)
        if mask.ndim != 2:
            print(
                f"Warning: Segmentation produced unexpected mask dimension {mask.ndim}. Expected 2D."
            )
            return None
        mask = mask.astype(np.int32)
        return mask

    except Exception as e:
        print(f"Error during _perform_segmentation ({seg_method}): {e}")
        # Print traceback for debugging
        import traceback

        traceback.print_exc()
        return None
    finally:
        # Clean up Python memory
        if use_gpu and (seg_method == "mesmer" or seg_method == "cellpose"):
            try:
                gc.collect()
            except Exception as clear_e:
                print(f"Warning: Error clearing session memory: {clear_e}")


def cell_segmentation(
    file_name,  # Path to image file or directory
    channel_file,  # Path to channel names file (if not input_format=="Channels")
    output_dir,  # Base directory for outputs
    output_fname="",  # Basename for output files
    seg_method="cellpose",  # 'cellpose' (preferred) or 'mesmer' (compat alias)
    nuclei_channel="DAPI",  # Name of the nucleus channel
    input_format="Multichannel",  # 'Multichannel', 'Channels', 'CODEX'
    membrane_channel_list=None,  # List of channel names for membrane/whole-cell seg
    cytoplasm_channel_list=None,  # List of channel names for cytoplasm (Cellpose only)
    size_cutoff=0,  # Min object size for feature extraction
    compartment="whole-cell",  # Mesmer: 'whole-cell' or 'nuclear'. Cellpose: Ignored.
    plot_predictions=False,  # Plot Mesmer predictions
    model="cyto3",  # Cellpose model name or path
    use_gpu=True,  # Use GPU if available
    diameter=None,  # Cellpose cell diameter estimate
    save_mask_as_png=False,  # Save Cellpose overlay PNGs
    model_path="./models",  # Path for Mesmer model download/load
    resize_factor=1,  # Factor to resize images before segmentation
    custom_model=False,  # True if 'model' is a path to a custom Cellpose model
    differentiate_nucleus_cytoplasm=False,  # Perform separate Nuc/Whole seg
    tile_size=4096,  # Tile size for segmentation
    tile_overlap=128,  # Overlap between segmentation tiles
    tiling_threshold=5000,  # Use tiling if H and W exceed this threshold
    image_mpp=0.5,  # Microns per pixel (primarily for Mesmer)
    stitch_sigma=64,  # Sigma for Gaussian blending during stitching
    remove_tile_border_objects=True,  # Remove objects touching tile borders before stitching
    # Feature extraction parameters embedded
    feature_tile_size=4096,
    feature_tile_overlap=128,
    feature_memory_limit_gb=8,
    set_memory_growth=True,
):
    """Perform cell segmentation using Cellpose with optional tiling and feature extraction.

    This function implements a complete segmentation pipeline including image loading,
    preprocessing, segmentation, mask stitching, and feature extraction. It handles large
    images through tiling and provides memory-optimized processing.

    Parameters
    ----------
    file_name : str or Path
        Path to input image file or directory (Multichannel = multichannel TIFF, Channels = single-channel TIFFs in a directory, CODEX = CODEX format with channels, cycles, y, x)
    channel_file : str or Path
        Path to channel names file (ignored if input_format=="Channels")
    output_dir : str or Path
        Base directory for output files
    output_fname : str, optional
        Basename for output files, by default auto-generated
    seg_method : {'cellpose', 'mesmer'}, optional
        Segmentation algorithm to use, by default 'cellpose'. ``'mesmer'`` is
        retained as a compatibility alias mapped to Cellpose.
    nuclei_channel : str, optional
        Name of the nuclei channel, by default 'DAPI'
    input_format : {'Multichannel', 'Channels', 'CODEX'}, optional
        Format of input data, by default 'Multichannel'
    membrane_channel_list : list of str, optional
        Channel names for membrane/whole-cell segmentation
    cytoplasm_channel_list : list of str, optional
        Channel names for cytoplasm (Cellpose only)
    size_cutoff : int, optional
        Minimum object size in pixels for feature extraction
    compartment : {'whole-cell', 'nuclear'}, optional
        Segmentation compartment for Mesmer (ignored by Cellpose)
    plot_predictions : bool, optional
        Whether to plot Mesmer predictions
    model : str, optional
        Model name or path for Cellpose
    use_gpu : bool, optional
        Whether to use GPU acceleration
    diameter : float, optional
        Expected cell diameter for Cellpose in pixels (setting a value is recommended to speed up segmentation significantly - if you are unsure you can measure the average cell diameter in ImageJ)
    save_mask_as_png : bool, optional
        Save Cellpose overlay as PNG
    model_path : str or Path, optional
        Path for Mesmer model download/load
    resize_factor : float, optional
        Factor to resize images before segmentation
    custom_model : bool, optional
        Whether 'model' is a path to custom Cellpose model
    differentiate_nucleus_cytoplasm : bool, optional
        Perform separate nuclear and whole-cell segmentation
    tile_size : int, optional
        Size of tiles for segmentation in pixels
    tile_overlap : int, optional
        Overlap between adjacent tiles in pixels
    tiling_threshold : int, optional
        Image size threshold to enable tiling
    image_mpp : float, optional
        Microns per pixel (for Mesmer)
    stitch_sigma : float, optional
        Sigma for Gaussian blending during stitching
    remove_tile_border_objects : bool, optional
        Remove objects touching tile borders
    feature_tile_size : int, optional
        Tile size for feature extraction
    feature_tile_overlap : int, optional
        Overlap for feature extraction tiles
    feature_memory_limit_gb : float, optional
        Memory limit per channel for feature extraction
    set_memory_growth : bool, optional
        Enable TensorFlow memory growth

    Returns
    -------
    dict or None
        Dictionary containing:
            - 'img_ref': Reference image
            - 'image_dict': Channel images
            - 'masks': Primary segmentation mask
            - 'masks_nuclei': Nuclear mask (if differentiated)
            - 'masks_cytoplasm': Cytoplasm mask (if differentiated)
            - 'features': DataFrame of extracted features
            - 'features_nuclei/cytoplasm/whole_cell': Region-specific features
            - 'features_combined': Combined features from all regions
        Returns None on critical error

    Notes
    -----
    Memory optimization strategies:
    - Tiling for large image segmentation
    - Memory-efficient feature extraction
    - Optional GPU memory growth
    - Cleanup of intermediate arrays

    The pipeline includes:
    1. Image loading and preprocessing
    2. Segmentation (tiled or full image)
    3. Mask post-processing and stitching
    4. Feature extraction and combination
    """

    results = {}
    output_dir = pathlib.Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- 1. Initial Setup ---
    print("--- Initializing Segmentation Pipeline ---")
    use_gpu = setup_gpu(use_gpu, set_memory_growth)  # Setup GPU and update status
    if not output_fname:
        base = (
            pathlib.Path(file_name).stem
            if input_format != "Channels"
            else pathlib.Path(file_name).name
        )
        output_fname = f"{base}_{seg_method}"
    print(f"Output basename: {output_fname}")
    print(f"Segmentation method: {seg_method}")
    print(f"Differentiate Nucleus/Cytoplasm: {differentiate_nucleus_cytoplasm}")

    # --- 2. Load Images ---
    # Memory bottleneck potential here for very large files
    img_ref, image_dict, channel_names = load_image_dictionary(
        file_name, channel_file, input_format, nuclei_channel
    )
    if image_dict is None:
        return None  # Error during loading
    results["img_ref"] = img_ref  # Store reference (could be large)
    results["image_dict"] = image_dict  # Holds all channel arrays (potentially large)
    original_shape = image_dict[nuclei_channel].shape  # Shape before any resizing

    # --- 3. Prepare Images for Segmentation ---
    print("\n--- Preparing Segmentation Inputs ---")
    # Creates a dict with only nuclei and optionally combined membrane channel
    segmentation_dict_prep, combined_membrane_channel_name = prepare_segmentation_dict(
        image_dict, nuclei_channel, membrane_channel_list
    )
    # Check if cytoplasm channel exists for Cellpose (if needed later)
    # Assuming cytoplasm_channel_list contains only one name for simplicity here
    cytoplasm_channel_name = (
        cytoplasm_channel_list[0] if cytoplasm_channel_list else None
    )
    if cytoplasm_channel_name and cytoplasm_channel_name not in image_dict:
        print(
            f"Warning: Specified cytoplasm channel '{cytoplasm_channel_name}' not found."
        )
        cytoplasm_channel_name = None  # Disable if not found

    # Resize if necessary (creates new, potentially large, arrays)
    segmentation_dict_resized = resize_segmentation_images(
        segmentation_dict_prep, resize_factor
    )
    del segmentation_dict_prep
    gc.collect()  # Clean up intermediate dict

    full_shape_seg = segmentation_dict_resized[
        nuclei_channel
    ].shape  # Shape used for segmentation
    print(f"Image shape for segmentation: {full_shape_seg}")

    # --- 4. Determine Tiling for Segmentation ---
    use_tiling_seg = (
        tile_size is not None
        and full_shape_seg[0] > tiling_threshold
        and full_shape_seg[1] > tiling_threshold
    )
    if use_tiling_seg:
        print(
            f"\n--- Tiling Enabled for Segmentation (Threshold: {tiling_threshold}px, Tile: {tile_size}, Overlap: {tile_overlap}) ---"
        )
        tiles_info = generate_tiles(full_shape_seg, tile_size, tile_overlap)
    else:
        print("\n--- Processing Full Image for Segmentation ---")
        tiles_info = None

    # --- 5. Perform Segmentation ---
    final_masks = {}  # Store final, full-sized masks

    try:
        if differentiate_nucleus_cytoplasm:
            print("\n--- Segmentation Mode: Differentiate Nucleus/Cytoplasm ---")
            if not combined_membrane_channel_name:
                print("Error: Membrane channels must be provided for differentiation.")
                return None

            # Define tasks: Nuclear and Whole-Cell
            segmentation_tasks = {
                "nuclei": {
                    "compartment": "nuclear",
                    "membrane": None,
                    "cytoplasm": None,
                },
                "whole_cell": {
                    "compartment": "whole-cell",
                    "membrane": combined_membrane_channel_name,
                    "cytoplasm": cytoplasm_channel_name,
                },
            }
            task_results_resized = {}  # Store masks at segmentation resolution

            for task_name, task_params in segmentation_tasks.items():
                print(f"\n--- Running {task_name.upper()} Segmentation ---")
                task_tile_masks = []
                completed_tiles_indices = set()

                if use_tiling_seg:
                    print(f"Processing {len(tiles_info)} tiles for {task_name}...")
                    for idx, (y_start, y_end, x_start, x_end) in enumerate(tiles_info):
                        display_tile_progress(
                            tiles_info,
                            completed_tiles_indices,
                            full_shape_seg,
                            current_tile_index=idx,
                        )
                        # Crop tile from the *resized* segmentation dict
                        tile_seg_dict = {
                            ch: im[y_start:y_end, x_start:x_end]
                            for ch, im in segmentation_dict_resized.items()
                        }

                        tile_mask = _perform_segmentation(
                            seg_dict=tile_seg_dict,
                            seg_method=seg_method,
                            output_dir=output_dir / f"tile_{idx}_{task_name}",
                            nuclei_channel_name=nuclei_channel,
                            membrane_channel_name=task_params["membrane"],
                            cytoplasm_channel_name=task_params[
                                "cytoplasm"
                            ],  # Pass cyto name
                            compartment=task_params["compartment"],
                            plot_predictions=plot_predictions,
                            model_path=model_path,
                            use_gpu=use_gpu,
                            model=model,
                            custom_model=custom_model,
                            diameter=diameter,
                            save_mask_as_png=save_mask_as_png,
                            image_mpp=image_mpp,
                        )

                        if tile_mask is None:
                            print(
                                f"Warning: Tile {idx} segmentation failed for {task_name}. Skipping tile."
                            )
                            # Append None or an empty mask? Append None for stitching robustness.
                            task_tile_masks.append(None)
                        else:
                            # Optional: Resize mask back to tile's input size if needed (shouldn't be necessary if seg returns correct size)
                            # tile_mask_resized = resize_mask(tile_mask, tile_seg_dict[nuclei_channel].shape)
                            if remove_tile_border_objects:
                                tile_mask = remove_border_objects(tile_mask)
                            task_tile_masks.append(tile_mask)

                        completed_tiles_indices.add(idx)
                        del tile_seg_dict, tile_mask
                        gc.collect()  # Clean up tile data
                        # time.sleep(0.01) # Small delay for display update

                    display_tile_progress(
                        tiles_info, completed_tiles_indices, full_shape_seg
                    )  # Show final progress
                    print(f"\nStitching {task_name} masks...")
                    stitched_mask = stitch_masks(
                        tiles_info,
                        task_tile_masks,
                        full_shape_seg,
                        tile_overlap,
                        sigma=stitch_sigma,
                    )
                    task_results_resized[task_name] = stitched_mask
                    del task_tile_masks, stitched_mask
                    gc.collect()

                else:  # No tiling for segmentation
                    print(f"Processing full image for {task_name}...")
                    full_mask = _perform_segmentation(
                        seg_dict=segmentation_dict_resized,  # Use the resized dict
                        seg_method=seg_method,
                        output_dir=output_dir / f"full_{task_name}",
                        nuclei_channel_name=nuclei_channel,
                        membrane_channel_name=task_params["membrane"],
                        cytoplasm_channel_name=task_params["cytoplasm"],
                        compartment=task_params["compartment"],
                        plot_predictions=plot_predictions,
                        model_path=model_path,
                        use_gpu=use_gpu,
                        model=model,
                        custom_model=custom_model,
                        diameter=diameter,
                        save_mask_as_png=save_mask_as_png,
                        image_mpp=image_mpp,
                    )
                    if full_mask is None:
                        raise RuntimeError(
                            f"Full image segmentation failed for {task_name}."
                        )
                    task_results_resized[task_name] = full_mask
                    del full_mask
                    gc.collect()

            # Resize final masks back to original image size
            print("Resizing final masks to original image shape...")
            final_masks["masks_nuclei"] = resize_mask(
                task_results_resized["nuclei"], original_shape
            )
            final_masks["masks"] = resize_mask(
                task_results_resized["whole_cell"], original_shape
            )  # 'masks' holds whole-cell

            # Calculate Cytoplasm Mask
            print("Calculating cytoplasm masks...")
            if (
                final_masks["masks_nuclei"] is not None
                and final_masks["masks"] is not None
            ):
                binary_masks_nuclei = final_masks["masks_nuclei"] > 0
                binary_masks_whole_cell = final_masks["masks"] > 0
                binary_masks_cytoplasm = binary_masks_whole_cell & (
                    ~binary_masks_nuclei
                )
                # Relabel cytoplasm mask
                masks_cytoplasm_labeled, num_labels = label(binary_masks_cytoplasm)
                final_masks["masks_cytoplasm"] = masks_cytoplasm_labeled.astype(
                    np.int32
                )
                print(f"Created cytoplasm mask with {num_labels} labeled objects.")
                del (
                    binary_masks_nuclei,
                    binary_masks_whole_cell,
                    binary_masks_cytoplasm,
                    masks_cytoplasm_labeled,
                )
            else:
                print(
                    "Warning: Could not calculate cytoplasm mask due to missing nuclei or whole-cell mask."
                )
                final_masks["masks_cytoplasm"] = None

            del task_results_resized
            gc.collect()

        else:  # Standard (non-differentiated) segmentation
            print("\n--- Segmentation Mode: Standard ---")
            current_membrane_channel = combined_membrane_channel_name
            current_compartment = compartment if current_membrane_channel else "nuclear"
            if not current_membrane_channel:
                print(
                    "Performing nuclear-only segmentation (no membrane channels provided)."
                )

            task_tile_masks = []
            completed_tiles_indices = set()
            primary_mask_resized = None  # Mask at segmentation resolution

            if use_tiling_seg:
                print(f"Processing {len(tiles_info)} tiles...")
                for idx, (y_start, y_end, x_start, x_end) in enumerate(tiles_info):
                    display_tile_progress(
                        tiles_info,
                        completed_tiles_indices,
                        full_shape_seg,
                        current_tile_index=idx,
                    )
                    tile_seg_dict = {
                        ch: im[y_start:y_end, x_start:x_end]
                        for ch, im in segmentation_dict_resized.items()
                    }

                    tile_mask = _perform_segmentation(
                        seg_dict=tile_seg_dict,
                        seg_method=seg_method,
                        output_dir=output_dir / f"tile_{idx}",
                        nuclei_channel_name=nuclei_channel,
                        membrane_channel_name=current_membrane_channel,
                        cytoplasm_channel_name=cytoplasm_channel_name,
                        compartment=current_compartment,
                        plot_predictions=plot_predictions,
                        model_path=model_path,
                        use_gpu=use_gpu,
                        model=model,
                        custom_model=custom_model,
                        diameter=diameter,
                        save_mask_as_png=save_mask_as_png,
                        image_mpp=image_mpp,
                    )

                    if tile_mask is None:
                        print(
                            f"Warning: Tile {idx} segmentation failed. Skipping tile."
                        )
                        task_tile_masks.append(None)
                    else:
                        if remove_tile_border_objects:
                            tile_mask = remove_border_objects(tile_mask)
                        task_tile_masks.append(tile_mask)

                    completed_tiles_indices.add(idx)
                    del tile_seg_dict, tile_mask
                    gc.collect()
                    # time.sleep(0.01)

                display_tile_progress(
                    tiles_info, completed_tiles_indices, full_shape_seg
                )
                print("\nStitching masks...")
                primary_mask_resized = stitch_masks(
                    tiles_info,
                    task_tile_masks,
                    full_shape_seg,
                    tile_overlap,
                    sigma=stitch_sigma,
                )
                del task_tile_masks
                gc.collect()

            else:  # No tiling for segmentation
                print("Processing full image...")
                primary_mask_resized = _perform_segmentation(
                    seg_dict=segmentation_dict_resized,
                    seg_method=seg_method,
                    output_dir=output_dir / "full",
                    nuclei_channel_name=nuclei_channel,
                    membrane_channel_name=current_membrane_channel,
                    cytoplasm_channel_name=cytoplasm_channel_name,
                    compartment=current_compartment,
                    plot_predictions=plot_predictions,
                    model_path=model_path,
                    use_gpu=use_gpu,
                    model=model,
                    custom_model=custom_model,
                    diameter=diameter,
                    save_mask_as_png=save_mask_as_png,
                    image_mpp=image_mpp,
                )
                if primary_mask_resized is None:
                    raise RuntimeError("Full image segmentation failed.")

            # Resize final mask back to original image size
            print("Resizing final mask to original image shape...")
            final_masks["masks"] = resize_mask(primary_mask_resized, original_shape)
            del primary_mask_resized
            gc.collect()

    except Exception as e:
        print(f"An error occurred during the segmentation stage: {e}")
        # Clean up potentially large intermediate data
        del segmentation_dict_resized
        if "task_results_resized" in locals():
            del task_results_resized
        if "task_tile_masks" in locals():
            del task_tile_masks
        gc.collect()
        return None  # Critical error

    # Clean up resized segmentation dictionary
    del segmentation_dict_resized
    gc.collect()

    # --- 6. Feature Extraction ---
    print("\n--- Extracting Features ---")
    # Use the *original* image_dict (full resolution) and *final, original-size* masks
    try:
        if differentiate_nucleus_cytoplasm:
            if all(
                k in final_masks and final_masks[k] is not None
                for k in ["masks_nuclei", "masks_cytoplasm", "masks"]
            ):
                print("Quantifying features for Nuclei, Cytoplasm, and Whole Cell...")
                features = {}
                for region, mask_key in [
                    ("nuclei", "masks_nuclei"),
                    ("cytoplasm", "masks_cytoplasm"),
                    ("whole_cell", "masks"),
                ]:  # 'masks' is whole_cell here
                    print(f"  Quantifying {region}...")
                    output_file = output_dir / f"{output_fname}_{region}_features.csv"
                    features[region] = extract_features(
                        image_dict=image_dict,
                        segmentation_masks=final_masks[mask_key],
                        channels_to_quantify=channel_names,
                        output_file=output_file,
                        size_cutoff=size_cutoff,
                        use_tiling_for_intensity=True,  # Enable intensity tiling
                        tile_size=feature_tile_size,
                        tile_overlap=feature_tile_overlap,
                        memory_limit_gb=feature_memory_limit_gb,
                    )
                    if features[region] is not None:
                        print(f"  Saved {region} features to {output_file}")
                        results[f"features_{region}"] = features[region]
                    else:
                        print(f"  Feature extraction failed for {region}.")

                # Combine features if all parts were successful
                if all(
                    f in features and features[f] is not None
                    for f in ["nuclei", "cytoplasm", "whole_cell"]
                ):
                    print("Combining features...")
                    try:
                        # Use whole_cell features as the base for metadata and labels
                        base_features = features["whole_cell"].copy()
                        base_features.set_index(
                            "label", inplace=True
                        )  # Ensure label is index

                        # Prepare intensity columns with suffixes
                        nuc_int = (
                            features["nuclei"]
                            .set_index("label")
                            .drop(columns=morpho_cols, errors="ignore")
                        )
                        cyto_int = (
                            features["cytoplasm"]
                            .set_index("label")
                            .drop(columns=morpho_cols, errors="ignore")
                        )
                        # whole_int = base_features.drop(columns=morpho_cols, errors='ignore') # Already have whole cell intensities

                        nuc_int.columns = [f"{col}_nuc" for col in nuc_int.columns]
                        cyto_int.columns = [f"{col}_cyto" for col in cyto_int.columns]
                        # whole_int.columns = [f"{col}_whole" for col in whole_int.columns] # Rename base intensities

                        # Join based on label index
                        combined = base_features.join([nuc_int, cyto_int], how="left")
                        combined.reset_index(inplace=True)  # Make label a column again

                        # Save combined features
                        combined_output_file = (
                            output_dir / f"{output_fname}_combined_features.csv"
                        )
                        combined.to_csv(combined_output_file, index=False)
                        print(f"Saved combined features to {combined_output_file}")
                        results["features_combined"] = combined
                    except Exception as e:
                        print(f"Warning: Failed to combine features: {e}")
                else:
                    print("Skipping feature combination due to missing feature sets.")
            else:
                print(
                    "Warning: Missing required masks for differentiated feature extraction."
                )

        else:  # Standard segmentation
            if "masks" in final_masks and final_masks["masks"] is not None:
                print("Quantifying features for segmented objects...")
                output_file = output_dir / f"{output_fname}_features.csv"
                features_df = extract_features(
                    image_dict=image_dict,
                    segmentation_masks=final_masks["masks"],
                    channels_to_quantify=channel_names,
                    output_file=output_file,
                    size_cutoff=size_cutoff,
                    use_tiling_for_intensity=True,  # Enable intensity tiling
                    tile_size=feature_tile_size,
                    tile_overlap=feature_tile_overlap,
                    memory_limit_gb=feature_memory_limit_gb,
                )
                if features_df is not None:
                    print(f"Saved features to {output_file}")
                    results["features"] = features_df
                else:
                    print("Feature extraction failed.")
            else:
                print("Warning: No final mask found for feature extraction.")

    except Exception as e:
        print(f"An error occurred during feature extraction: {e}")
        # Continue to return masks even if features fail

    # --- 7. Final Return ---
    print("\n--- Segmentation Pipeline Complete ---")
    results.update(
        final_masks
    )  # Add final masks ('masks', 'masks_nuclei', 'masks_cytoplasm')
    # Clean up large image_dict before returning? Optional.
    # del results['image_dict']; gc.collect()
    return results
