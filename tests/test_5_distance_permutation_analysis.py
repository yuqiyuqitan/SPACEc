# # %% [markdown]
# # # SPACEc:Distance permutation analysis

# # %% [markdown]
# # ## Set up environment

# # %%
# import os
# import time
# import sys
# import anndata as ad
# import pandas as pd
# import numpy as np
# import warnings
# warnings.filterwarnings('ignore')

# data_dir = '/Users/yuqitan/Nolan Lab Dropbox/Yuqi Tan/analysis_pipeline/demo_data/' # where the data is stored
# code_dir = '/Users/yuqitan/Nolan Lab Dropbox/Yuqi Tan/analysis_pipeline/demo_data/pipeline_test_112023/src' # current where the code is stored, this will be replaced by pip install soon
# output_dir = '/Users/yuqitan/Nolan Lab Dropbox/Yuqi Tan/analysis_pipeline/demo_data/output_112023/' #where you want to store the output

# if not os.path.exists(output_dir): # check if output path exist - if not generate the path
#     os.makedirs(output_dir)
    
# sys.path.append(code_dir) 

# from helperfunctions_hf import * # Helper functions - used by other functions to execute steps like table formatting etc. KEY: hf
# from preprocessing_pp import * # Preprocessing functions - to normalize and prepare data for further analysis KEY: pp
# from tools_tl import * # tools - perform calculation on the data KEY: tl
# from plot_pl import * # plotting functions - used to visualize results KEY: pl

# sc.settings.set_figure_params(dpi=80, facecolor='white')

# # %%
# # Load data
# adata = sc.read(output_dir + "adata_nn_demo_annotated_cn.h5ad")
# adata

# # %% [markdown]
# # ## 5.1 Identify potential interactions

# # %%
# distance_pvals = tl_identify_interactions_ad(adata = adata , 
#                                                 id = "index", 
#                                                 x_pos = "x", 
#                                                 y_pos = "y", 
#                                                 cell_type = "celltype", 
#                                                 region = "unique_region",
#                                                 num_iterations=100,
#                                                 num_cores=10, 
#                                                 min_observed = 10,
#                                                 comparison = 'condition')
# distance_pvals.head()

# # %%
# # Identify significant cell-cell interactions
# # dist_table_filt is a simplified table used for plotting
# # dist_data_filt contains the filtered raw data with more information about the pairs
# dist_table_filt, dist_data_filt = tl_filter_interactions(distance_pvals = distance_pvals,
#                                             pvalue = 0.05,
#                                             logfold_group_abs = 0.1)

# print(dist_table_filt.shape)
# dist_data_filt.head()

# # %%
# pl_dumbbell(data = dist_table_filt, figsize=(10,10), colors = ['#DB444B', '#006BA2'])

# # %%



