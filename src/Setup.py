# Setup.py 

# This file contains all information to run the scCODEX tutorial 
# If you want to run the tutorial notebooks with your own data you need to change the paths in this file



#################### CHANGE PATHS HERE ####################
base_dir = '/home/alexander/Desktop/scCODEX_tutorial/' # Path to base directory
raw_images_path = os.join(base_dir, "raw_images/")  # Path to raw images
segmented_csv_path = '/home/alexander/Desktop/scCODEX_tutorial/segmented_csv/' # Path to segmented csv files
adata_path = '/home/alexander/Desktop/scCODEX_tutorial/output/' # Path to output folder
plot_output_path = '/home/alexander/Desktop/scCODEX_tutorial/output/plots/' # Path to output folder for plots



#################### DO NOT CHANGE ANYTHING BELOW THIS LINE ####################

# if adata_path does not exist, create it
if not os.path.exists(adata_path):
    os.makedirs(adata_path)
    
# if plot_output_path does not exist, create it
if not os.path.exists(plot_output_path):
    os.makedirs(plot_output_path)


# Import packages
import anndata as ad
import pandas as pd
import numpy as np
import time
import sys