# Setup.py 

# This file contains all information to run the scCODEX tutorial 
# If you want to run the tutorial notebooks with your own data you need to change the paths in this file



#################### CHANGE PATHS HERE ####################
import os
base_dir = '/Volumes/Tim_D260/Projects/Stanford_scCODEX/' # Path to base directory
raw_images_path = os.path.join(base_dir, "raw_images/")  # Path to raw images
segmented_csv_path = os.path.join(base_dir, "segmented_csv/") # Path to segmented csv files
adata_path = os.path.join(base_dir, "adata/") # Path to output folder
output_dir = '/Volumes/Tim_D260/Projects/Stanford_scCODEX/Results/' # Path to output folder for plots
custom_functions_path = '/Users/timnoahkempchen/Desktop/SAP5/src' # Path to custom functions

silence_warnings = True # If True, warnings will be silenced

#################### DO NOT CHANGE ANYTHING BELOW THIS LINE ####################

# Import packages
import time
import sys
import anndata as ad
import pandas as pd
import numpy as np

# if adata_path does not exist, create it
if not os.path.exists(adata_path):
    os.makedirs(adata_path)
    
# if plot_output_path does not exist, create it
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# detect path of this file
path = os.path.dirname(os.path.abspath(__file__))


# silence warnings
if silence_warnings == True:
    import warnings
    warnings.filterwarnings('ignore')
