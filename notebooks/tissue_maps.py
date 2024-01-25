# %% [markdown]
# # Install TissUUmaps

# %%
# Create a conda environment with TissUUmaps installed or update an your existing environment

# conda create -y -n tissuumaps_env -c conda-forge python=3.9
# conda activate tissuumaps_env
# conda install -c conda-forge libvips pyvips openslide-python
# pip install "TissUUmaps[full]" # this step seems to need to do within jupyter notebook for it to work

# %% [markdown]
# ## Using TissUUmaps within Jupyter notebook
# Interactive visualization via TissUUmaps might be informative during multiple steps of the analysis. Apart from the general function provided with the TissUUmaps Python package, we provide specific functions that automatically phrase the input during multiple steps of the analysis.

# %%
from skimage.segmentation import find_boundaries


def masks_to_outlines_scikit_image(masks):
    """get outlines of masks as a 0-1 array

    Parameters
    ----------------

    masks: int, 2D or 3D array
        size [Ly x Lx] or [Lz x Ly x Lx], 0=NO masks; 1,2,...=mask labels

    Returns
    ----------------

    outlines: 2D or 3D array
        size [Ly x Lx] or [Lz x Ly x Lx], True pixels are outlines

    """
    if masks.ndim > 3 or masks.ndim < 2:
        raise ValueError(
            "masks_to_outlines takes 2D or 3D array, not %dD array" % masks.ndim
        )

    if masks.ndim == 3:
        outlines = np.zeros(masks.shape, bool)
        for i in range(masks.shape[0]):
            outlines[i] = find_boundaries(masks[i], mode="inner")
        return outlines
    else:
        return find_boundaries(masks, mode="inner")


# %%
import os
import pathlib
import pickle

import numpy as np

# import tissuumaps
import tissuumaps.jupyter as tj
from skimage.io import imsave


def tm_prepare_input(
    adata,
    images_pickle_path,
    directory,
    region_column="unique_region",
    region="",
    xSelector="x",
    ySelector="y",
    color_by="celltype_fine",
    keep_list=None,
    include_masks=True,
    open_viewer=True,
    add_UMAP=True,
):
    segmented_matrix = adata.obs

    with open(images_pickle_path, "rb") as f:
        seg_output = pickle.load(f)

    image_dict = seg_output["image_dict"]
    masks = seg_output["masks"]

    if keep_list == None:
        keep_list = [region_column, xSelector, ySelector, color_by]

    print("Preparing TissUUmaps input...")

    cache_dir = pathlib.Path(directory) / region
    cache_dir.mkdir(parents=True, exist_ok=True)

    # only keep columns in keep_list
    segmented_matrix = segmented_matrix[keep_list]

    if add_UMAP == True:
        # add UMAP coordinates to segmented_matrix
        segmented_matrix["UMAP_1"] = adata.obsm["X_umap"][:, 0]
        segmented_matrix["UMAP_2"] = adata.obsm["X_umap"][:, 1]

    csv_paths = []
    # separate matrix by region and save every region as single csv file
    region_matrix = segmented_matrix.loc[segmented_matrix[region_column] == region]

    region_matrix.to_csv(cache_dir / (region + ".csv"))
    csv_paths.append(cache_dir / (region + ".csv"))

    # generate subdirectory for images
    image_dir = cache_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    (image_dir / ".tissuumaps").mkdir(parents=True, exist_ok=True)

    image_list = []
    # save every image as tif file in image directory from image_dict. name by key in image_dict
    for key, image in image_dict.items():
        file_path = os.path.join(image_dir, f"{key}.tif")
        file_path_out = os.path.join(image_dir, ".tissuumaps", f"{key}.tif")
        imsave(file_path_out, image, check_contrast=False)
        image_list.append(file_path)

    if include_masks == True:
        # select first item from image_dict as reference image
        reference_image = list(image_dict.values())[0]

        # make reference image black by setting all values to 0
        reference_image = np.zeros_like(reference_image)

        # make the reference image rgb. Add empty channels
        if len(reference_image.shape) == 2:
            reference_image = np.expand_dims(reference_image, axis=-1)
            reference_image = np.repeat(reference_image, 3, axis=-1)

        # remove last dimension from masks
        masks_3d = np.squeeze(masks)
        outlines = masks_to_outlines_scikit_image(masks_3d)

        reference_image[outlines == True] = [255, 0, 0]

        file_path = os.path.join(image_dir, "masks.jpg")

        # save black pixel as transparent
        reference_image = reference_image.astype(np.uint8)

        imsave(file_path, reference_image)
        image_list.append(file_path)

    print(image_list)

    if open_viewer == True:
        print("Opening TissUUmaps viewer...")
        tj.loaddata(
            images=image_list,
            csvFiles=[str(p) for p in csv_paths],
            xSelector=xSelector,
            ySelector=ySelector,
            keySelector=color_by,
            nameSelector=color_by,
            colorSelector=color_by,
            piechartSelector=None,
            shapeSelector=None,
            scaleSelector=None,
            fixedShape=None,
            scaleFactor=1,
            colormap=None,
            compositeMode="source-over",
            boundingBox=None,
            port=5100,
            host="localhost",
            height=900,
            tmapFilename=region + "_project",
            plugins=[
                "Plot_Histogram",
                "Points2Regions",
                "Spot_Inspector",
                "Feature_Space",
                "ClassQC",
            ],
        )

    return image_list, csv_paths


# %% [markdown]
# # Instructions:
#
# To use the TissUUmaps viewer you need:
# - A working env with TissUUmaps installed
# - A pickle file that contains the segmentation output and images
# - An AnnData object containing the currently used single cell data
#
# The *tm_prepare_input* function reads the named content for one region. For that, the user has to provide a region column and a region name. The pickle file has to match the specified region.
# The function creates a folder that contains all necessary input files that are needed to launch the TissUUmaps session. Additionally, the function can launch the TissUUmaps session. If the session is launched from the function a tmap file is created in the input directory that allows to open the session again (both from jupyter and the standalone viewer app).
# Alternatively, the function can be used to prepare the directory and the viewer can be launched separately to modify the display options in jupyter as well as host ports etc.
#
# If the Jupyter viewer is too small (might be a problem on small monitors), the user can use the link (displayed if function is executed) to display TissUUmaps in the browser.

# %% [markdown]
# # Testing

# %%
import pathlib

root_path = pathlib.Path("..")
data_path = root_path / "data"  # where the data is stored

# where you want to store the output
output_path = root_path / "_out"
output_path.mkdir(exist_ok=True, parents=True)

# %%
import scanpy as sc

# Loading the denoise/filtered anndata from notebook 3 [cell type or cluster annotation is necessary for the step]
adata = sc.read(output_path / "adata_nn_demo_annotated_cn.h5ad")

# %%
adata.obs.head()

# %% [markdown]
# ## Integrated use

# %%
image_list, csv_paths = tm_prepare_input(
    adata,
    images_pickle_path=output_path / "seg_output_tonsil1.pickle",
    directory=output_path / "tmp/tm_prepare_input/cache2",
    region_column="unique_region",
    region="reg010",
    xSelector="x",
    ySelector="y",
    color_by="celltype",
    keep_list=None,
    open_viewer=True,
)

print("TEST")

# %% [markdown]
# ## Open TM viewer separate from preparing the data

# %%
xSelector = "x"
ySelector = "y"
color_by = "celltype"

# %%
image_list, csv_paths = tm_prepare_input(
    adata,
    images_pickle_path=output_path / "seg_output_tonsil1.pickle",
    directory=output_path / "tmp/tm_prepare_input/cache2",
    region_column="unique_region",
    region="reg010",
    xSelector=xSelector,
    ySelector=ySelector,
    color_by=color_by,
    keep_list=None,
    open_viewer=False,
)

# %%
tj.loaddata(
    images=image_list,
    csvFiles=csv_paths,
    xSelector=xSelector,
    ySelector=ySelector,
    keySelector=None,
    nameSelector=None,
    colorSelector=color_by,
    piechartSelector=None,
    shapeSelector=None,
    scaleSelector=None,
    fixedShape=None,
    scaleFactor=1,
    colormap=None,
    compositeMode="source-over",
    boundingBox=None,
    port=5100,
    host="localhost",
    height=900,
    tmapFilename="_project",
    plugins=[
        "Plot_Histogram",
        "Points2Regions",
        "Spot_Inspector",
        "Feature_Space",
        "ClassQC",
    ],
)
