import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tifffile
from scipy import ndimage as ndi
from skimage.filters import sobel
from skimage.segmentation import watershed


def label_tissue(
    resized_im,
    lower_cutoff=0.012,
    upper_cutoff=0.025,
    savefig=False,
    showfig=True,
    output_dir="./",
    output_fname="",
):
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
        else:
            plt.show()

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
):
    tissueframe2 = tissueframe.groupby(region).agg([min, max])
    print("Reading in the qptiff file, might take awhile!")
    currim = tifffile.imread(filepath)

    for index, row in tissueframe2.iterrows():
        ymin = row["y"]["min"] * downscale_factor
        ymax = row["y"]["max"] * downscale_factor
        xmin = row["x"]["min"] * downscale_factor
        xmax = row["x"]["max"] * downscale_factor
        ymin = max(ymin - padding, 0)
        ymax = min(ymax + padding, currim.shape[1])
        xmin = max(xmin - padding, 0)
        xmax = min(xmax + padding, currim.shape[2])
        subim = currim[:, ymin:ymax, xmin:xmax]
        outpath = os.path.join(
            output_dir, output_fname, f"reg00{index}_X01_Y01_Z01.tif"
        )
        plt.imshow(subim[0])
        plt.title(f"Extracting tissue {index}: ")
        plt.show()
        print(f"Saving tissue image at {outpath}")
        tifffile.imwrite(outpath, subim)
