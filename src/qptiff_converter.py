import tifffile
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from skimage.segmentation import watershed
from skimage.transform import rescale, resize, downscale_local_mean
from scipy import ndimage as ndi
from skimage.filters import sobel
from sklearn.cluster import AgglomerativeClustering

def hf_downscale_tissue(
    file_path,
    DNAslice = 0,
    downscale_factor = 64,
    sigma = 5.0,
    padding = 50,
    savefig = False,
    showfig = True,
    output_dir = './',
    output_fname = ""

):
    print("Reading in the qptiff file, might take awhile!")
    currim = tifffile.imread(file_path)
    nucim = currim[DNAslice]
    del currim
    print(f'Loaded nuclear image of dimension (Y,X) = {nucim.shape}')
    resized_im = resize(nucim, (nucim.shape[0] // downscale_factor, nucim.shape[1] // downscale_factor), anti_aliasing=True)
    resized_im = ndi.gaussian_filter(resized_im, sigma = sigma)
    
    if showfig:
        fig, axs = plt.subplots(1, 2)
        axs[0].imshow(nucim)
        axs[0].set_title('Nuclear image')
        axs[1].hist(resized_im)
        axs[1].set_title('Marker expression level histogram')
        if savefig:
            fig.savefig(output_dir + output_fname +"_raw_tissue_plot.pdf", bbox_inches="tight")
        else:
            plt.show()
    print("returning scaled down image!")
    return resized_im

def tl_label_tissue (resized_im,
                    lower_cutoff = 0.012,
                    upper_cutoff = 0.025,
                    savefig = False,
                    showfig = True,
                    output_dir = './',
                    output_fname = ""):
    # cut off 
    elevation_map = sobel(resized_im)
    markers = np.zeros_like(resized_im)
    markers[resized_im <= lower_cutoff] = 1
    markers[resized_im >= upper_cutoff] = 2

    segmentation = watershed(elevation_map, markers)
    #plt.imshow(segmentation)
    #plt.title('Segmented tissues')
    #plt.show()

    segmentation = ndi.binary_fill_holes(segmentation - 1)
    #plt.imshow(segmentation)
    #plt.title('Segmented tissues with holes filled')
    #plt.show()

    #visualize initial identified segmented masks
    labeled_tissues, _ = ndi.label(segmentation)
    print(f'Identified {len(np.unique(labeled_tissues)) - 1} tissue pieces')
    fig, axs = plt.subplots(1, 1)
    axs.imshow(labeled_tissues)
    axs.set_title('Labeled tissues')

    if showfig:
        if savefig:
            fig.savefig(output_dir + output_fname +"_labeled_seg_tissue_plot.pdf", bbox_inches="tight")
        else:
            plt.show()

    #######Non clustering option
    print("Saving the labels from the segmentation!")
    idx = np.nonzero(labeled_tissues)
    vals = labeled_tissues[idx]
    tissueframe = pd.DataFrame(vals, columns = ['tissue'])
    tissueframe['y'] = idx[0]
    tissueframe['x'] = idx[1]
    tissueframe['region1']=tissueframe['tissue']
    
    return tissueframe

def pl_tissue_lables(tissueframe, region = 'region1'):
    centroids = tissueframe.groupby('tissue').mean()
    fig, ax = plt.subplots()
    ax.scatter(centroids['x'], centroids['y'])
    ax.invert_yaxis()
    plt.gca().set_aspect('equal', adjustable='box')
    
    for i, txt in enumerate(centroids.index):
        ax.annotate(txt, (list(centroids['x'])[i], list(centroids['y'])[i]))
    
    plt.title('Tissue piece labels')
    plt.show()
    
    fig, ax = plt.subplots()
    ax.scatter(centroids['x'], centroids['y'])
    ax.invert_yaxis()
    plt.gca().set_aspect('equal', adjustable='box')
    
    for i, txt in enumerate(centroids[region]):
        ax.annotate(int(txt), (list(centroids['x'])[i], list(centroids['y'])[i]))
    
    plt.title('Region labels')
    plt.show()

def tl_save_labelled_tissue(filepath, 
                            tissueframe,
                            region = 'region', # this can be 'region1' if you didn't manually rename your tissue region
                            padding = 50,
                            downscale_factor = 64,
                            output_dir = './',
                            output_fname = ""):
    tissueframe2 = tissueframe.groupby(region).agg([min, max])
    print("Reading in the qptiff file, might take awhile!")
    currim = tifffile.imread(filepath)

    for index, row in tissueframe2.iterrows():
        ymin = row['y']['min'] * downscale_factor
        ymax = row['y']['max'] * downscale_factor
        xmin = row['x']['min'] * downscale_factor
        xmax = row['x']['max'] * downscale_factor
        ymin = max(ymin - padding, 0)
        ymax = min(ymax + padding, currim.shape[1])
        xmin = max(xmin - padding, 0)
        xmax = min(xmax + padding, currim.shape[2])
        subim = currim[:, ymin:ymax, xmin:xmax]
        outpath = os.path.join(output_dir, output_fname, f'reg00{index}_X01_Y01_Z01.tif')
        plt.imshow(subim[0])
        plt.title(f'Extracting tissue {index}: ')
        plt.show()
        print(f'Saving tissue image at {outpath}')
        tifffile.imwrite(outpath, subim)