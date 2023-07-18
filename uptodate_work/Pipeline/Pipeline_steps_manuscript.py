#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 13 13:18:03 2023

@author: timnoahkempchen
"""
#import standard packages
import os
import pandas as pd
import sys
import matplotlib.pyplot as plt
import scanpy as sc
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

sys.path.append('/Users/timnoahkempchen/Desktop/SAP2/Docker/Container_new') # This code is only needed if you want to load functions from a non-default directory

from SAP_helperfunctions_hf import * # Helper functions - used by other functions to execute steps like table formatting etc. KEY: hf
from SAP_preprocessing_pp import * # Preprocessing functions - to normalize and prepare data for further analysis KEY: pp
from SAP_tools_tl import * # tools - perform calculation on the data KEY: tl
from SAP_plot_pl import * # plotting functions - used to visualize results KEY: pl

# define color dictionarys for downstream visualizations
color_dic_cells = {'Tumor PDL1+ MHCI+': "#21f0b6",
 'CD8+ T cell PD1+': "#2a8665",
 'Tumor TYRP1+': "#79c6c1",
 'DC': "#2a538a",
 'Macrophage': "#daa4f9",
 'Tumor': "#7e39c2",
 'Endothelial CD106+': "#9ab9f9",
 'CD8+ T cell': "#9e3678",
 'DC TCF7+': "#f36ad5",
 'Epiehtlial': "#9f04fc",
 'CD4+ T cell': "#5ac230",
 'Epithelial': "#b7d165",
 'Tumor Ki67+': "#6d3918",
 'CD4+ Treg': "#efaa79",
 'Neutrophil': "#9f2114",
 'Endothelial': "#fd5917",
 'NK': "#fe1d66",
 'Macrophage PDL1+': "#f7767d",
 'APC MHCIIhi': "#fbbd13",
 'Macrophage CD86+': "#748d13",
 'Lymphatic': '#00944F',
 'CD86+ Macrophage': '#636A86',
 'Macrophage CD169+': '#7A50F7',
 'Tumor CD117hi': '#24B93B',
 'Lymphatic Ly6Chi': '#9BF29D',
 'NK cell KLRG1hi': '#829824',
 'B cell': '#F2A3F0'}

color_dic_day = {3: '#ffe74c', 
                 1: '#ff5964', 
                 5: '#38618c'}

# Downstream analysis 
## Preprocessing of data (Step 1-5)
###############################################################################################################################################################################

##############################################################
### Load segmented data (Step 1)

# Specify 1) the path of you segmentation output and 2) a path to store all output. Skip to the **Step 3** if your dataset is already normalized. 
data_dir = "/Users/yuqitan/Nolan Lab Dropbox/Yuqi Tan/Collaborations/Hendrik/processed_data/14_05_23_Breast_70066_risk/Scan1/CVcol_2023_01_09_DAPI_3px/fcs/"
output_dir = "/Users/yuqitan/Nolan Lab Dropbox/Yuqi Tan/Collaborations/Hendrik/processed_data/14_05_23_Breast_70066_risk/"
if not os.path.exists(output_dir): # check if output path exist - if not generate the path
    os.makedirs(output_dir)
    
# this is loading the segmentation output 
df = pp_read_data(path=data_dir + 'compensated/',\
               reg_list=[], nuc_1 = 1) #this is only useful when there are multiple regions. it also helps to format the heading
df.shape # this tells you the dimension of your data
df.head() # show the first five rows of the data

##############################################################
### Normalization & xy correction (Step 2)

# This is to normalize the data 
dfz = pp_format(data=df, 
                list_out=['first_index', 'cell_id','tile_num','z', 'x_tile','y_tile', 'size','DAPI'],
                list_keep = ['region','x','y','region_num',],
                method = "zscore") # ["zscore", "double_zscore", "MinMax", "ArcSin"]

#examine unique region per tissue to decide whether or not we need x y correction
pl_catplot(dfz,X = 'x',Y='y',
           exp = 'region_num', hue = 'region_num',invert_y=True,size = 1,figsize=8)

# (Optional) XY corrrection for CODEX data from non-fusion machine
# since this data is generated from fusion and has one tissue per slide
# the xy correction is not nesssary 
# df_cor = pp_xycorr(data=dfz_56, y_rows=2, x_columns=1, X_pix=13000, Y_pix=11000)


##############################################################
### Noise removal (Step 3)

# get the column index for the last antibody 
dfz.columns
dfz.columns.get_loc('ICOS')

# This function helps to figure out what the cut-off should be
# This is to remove top 1 % of all cells that are highly expressive for all antibodies
pl_zcount_thres(dfz = dfz, 
                col_num = 41, # last antibody index
                cut_off=0.01, # top 1% of cells
                count_bin=50) # adjust histogram visualization

df_nn,cc = pp_remove_noise(df=dfz, 
                           col_num=41, # this is the column index that has the last protein feature
                           z_sum_thres=47, # number obtained from the function above
                           z_count_thres=34 # number obtained from the function above
                          )

df_nn.to_csv(output_dir + "df_nn_NBT_230223_NBT_70129.csv")


##############################################################
### Clustering of data for cell type annotation (Step 4)

adata = hf_makeAnndata(df_nn = df_nn,
                       col_sum = 41, # this is the column index that has the last protein feature
                       nonFuncAb_list = ['GranzymeB', 'Tbet','PD-L1', 'PD1', 'IDO-I', 'ICOS'] # remove the antibodies that are not working
                      )

# no need to reassign the adata
tl_clustering(adata, 
              clustering='leiden', # can choose louvian
              n_neighbors=10,
              res = 1,
              reclustering = False # if true, no computing the neighbors
             ) 

# visualization of clustering with UMAP
sc.pl.umap(adata, color = 'leiden_1') 

# reclustering with different resolution
tl_test_clustering_resolutions(adata, clustering='leiden', n_neighbors=10, resolutions=[0.4, 0.5, 0.6]) # function cycles through list of resolutions 

#look at the marker gene expression
sc.pl.dotplot(adata, adata.var.index.to_list(), 'leiden_0.4')

# Plot data using its original spatial coordinates
df_nn['leiden_0.4'] = adata.obs['leiden_0.4'] # extract data frame from anndata object
ax = sns.scatterplot(x='x', y='y', data=df_nn, hue='leiden_0.4', s=1, alpha=0.5)
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
ax.invert_yaxis()

##############################################################
### Manual cell type annotation (Step 5)

# Generate dictionary to rename the clusters with cell types
cell_type_annotation_dic = {
    '0': 'Epithelial',
    '1': 'Stromal/blood vessels',
    '2': 'Stroma', 
    '3': 'Blood vessels',
    '4': 'Proliferating stroma',
    '5': 'M2 macrophage',
    '6': 'Early exhausted CD8 T cells',
    '7': "Overexposed junk",
    '8': 'monocyte/macrophage?',
    '9': 'Plasma cells',
    '10': 'Lymphatics',
    '11': 'Treg',
    '12': 'CD4+ (memory) T cells',
    '13': "DC",
    '14': 'Epithelial',
    '15': 'Neutrophil'
}

# Add annotation to anndata object
adata.obs['celltype'] = (
    adata.obs['leiden_0.4']
    .map(cell_type_annotation_dic)
    .astype('category')
)

# Add cell types to data frame
df_nn['celltype'] = adata.obs['celltype'].values

# Check the spatial position of the assigned cell types as quality control
pl_mono_cluster_spatial(df=df_nn, sample_col='region_num', cluster_col='leiden_0.4', figsize=(20, 20))

# After checking the clustering might be refined by subclustering. 
# subclustering cluster 0
sc.tl.leiden(adata, restrict_to=('leiden_0.4',['0']), resolution=0.13, key_added='leiden_0.4_subcluster_0')
sc.pl.umap(adata, color = 'leiden_0.4_subcluster_0') # visulize as UMAP
sc.pl.dotplot(adata, adata.var.index.to_list(), 'leiden_0.4_subcluster_0') # visulize marker expression as dotplot

# save anndata object
adata.write(output_dir+"NBT_230223_70129_adata.h5ad")


## Data analysis and visualization
###############################################################################################################################################################################

##############################################################
### Upload data to the downstream pipeline and generate basic visualizations (Step 6)

# specify file path to preprocessed data. If you use the SAP preprocessing module you can indicate the output of that module 
#loading data from file
df = pd.read_csv(output_dir + "df_nn_NBT_230223_NBT_70129.csv")
# or 
df = df_nn

#Get an overview - show column names as well as a preview of the dataframe.  
print(df.columns) # print all column names (features)
df.head() # show first five rows
print(df["celltype"].unique())

# Stacked Bar Plot
# Shows percentage of category per group. For example: major cell types per diagnosis 
ab = pl_stacked_bar_plot(data = df, # data frame to use 
                      per_cat = "original cell type", # column containing the categories that are used to fill the bar plot
                      grouping = "day harvested", # column containing a grouping variable (usually a condition or cell group)
                      output_dir = output_dir, 
                      sub_col = None, 
                      cell_list = df["original cell type"].unique(),  # list of cell types to plot 
                      norm = False, # logical value to decide if normalization per group is performed 
                      fig_sizing = (8,4), # numeric value to modify the figure size 
                      name_cat = "original cell type", 
                      col_order = None, # plotting order 
                      save_name = '/Percentage_cell_type_per_sample', # name for saved file 
                      color_dic = color_dic_cells # color dictionary 
                        ) 

ab = pl_stacked_bar_plot(data = df, # data frame to use 
                      per_cat = "Cell common", # column containing the categories that are used to fill the bar plot
                      grouping = "day harvested", # column containing a grouping variable (usually a condition or cell group)
                      output_dir = output_dir, 
                      sub_col = None, 
                      cell_list = df["Cell common"].unique(),  # list of cell types to plot 
                      norm = False, # logical value to decide if normalization per group is performed 
                      fig_sizing = (8,4), # numeric value to modify the figure size 
                      name_cat = "Cell common", 
                      col_order = None, # plotting order 
                      save_name = '/Percentage_common_cell_type_per_sample', # name for saved file 
                      color_dic = color_dic_cells # color dictionary 
                        ) 

# Arguments for swarm boxplot:
data = df # data frame to use 
grouping = "day harvested" # column containing a grouping variable (usually a condition or cell group)
sub_col = 'Cell common'
sub_list = ["CD8+ T cell"]
replicate_column = "unique_region"
output_dir = output_dir
norm = False
per_cat = "Cell common"
fig_sizing=(4,5) # numeric value to modify the figure size 
save_name = '/CD8+_per_sample'
color_dic = color_dic_day # color dictionary 
plot_order=None
flip=True

melt_per_plot = pl_swarm_box(data = data, 
                          grouping = grouping, 
                          replicate = replicate_column, 
                          sub_col = sub_col, 
                          sub_list = sub_list, 
                          per_cat = per_cat, 
                          norm=norm,
                          figure_sizing= fig_sizing, 
                          save_name=save_name, 
                          plot_order=plot_order, 
                          color_dic=color_dic, 
                          flip=flip,
                          output_dir = output_dir)


data = df # input data frame 
grouping = "day harvested" # column containing a grouping variable (usually a condition or cell group)
sub_list = df["Cell common"].unique()
rep = "unique_region"
per_categ = "Cell common"
sub_column = 'Cell common'
normalize = False
save = True # logical value to decide if the generated plot should be saved 
color_dic = color_dic_day # color dictionary 
fig_size = 8 # numeric value to modify the figure size 
plot_order = [1,3,5] # optional list to specify the plotting order
output_dir = output_dir # directory to save output

# Swarm Boxplot of Shannon diversity score
# calculate shannon diversity 
tt, test_results, res  = tl_Shan_div(data = data, 
                                     sub_l = sub_list, 
                                     group_com = grouping, 
                                     per_categ = per_categ, 
                                     rep = rep, 
                                     sub_column = sub_column, 
                                     normalize = normalize) 

# plot results of shannon diversity function
pl_Shan_div(tt, 
            test_results, 
            res, 
            grouping = grouping, 
            color_dic = color_dic, 
            sub_list = sub_list, 
            output_dir = output_dir, 
            save=save, 
            plot_order = plot_order,
            save_name= "/Shannon_Cell_common_all",
            fig_size=1.5)


##############################################################
### Analysis of cellular neighborhoods (Step 7)

# CN analysis
# Arguments for neighborhood analysis:
X = "x" # column containing the x coordinates 
Y = "y" # column containing the y coordinates 
reg = "unique_region" # column containg the unique regions 
cluster_col = "original cell type" # column which is used for clustering - typically cell types (to generate cellular neighborhoods)
# ks = [20] # k=n means it collects n nearest neighbors for each center cell
output_dir = output_dir
k = 10
# set a single value to generate neighborhoods with this value. If you want to generate the elbow plot supply the list with a range of ks
#n_neighborhoods = numbers = list(range(1, 41))# number of generated neighborhoods - value should be biologically meaningful 
n_neighborhoods = 10
#n_neighborhoods = range(1,40)
save_to_csv= True
plot_specific_neighborhoods = [2,4]

####### Neighborhood analysis 
df2 = pd.concat([df,pd.get_dummies(df[cluster_col])],1)
sum_cols = df2[cluster_col].unique()
values = df2[sum_cols].values

k_centroids = {}

# Previous version of function will be removed in future update 
# Maybe implement a function to test for the optimal neighborhood count


cells_df, k_centroids = tl_neighborhood_analysis_2(data = df2, 
                           values = values, 
                           sum_cols = sum_cols, 
                           X = X, 
                           Y = Y, 
                           reg = reg, 
                           cluster_col = cluster_col, 
                           k = k, 
                           n_neighborhoods = n_neighborhoods,
                           elbow=False) 

pl_neighborhood_analysis_2(data = cells_df, 
                           k_centroids = k_centroids,
                           values = values, 
                           sum_cols = sum_cols, 
                           X = X, 
                           Y = Y, 
                           reg = reg, 
                           output_dir = output_dir, 
                           k = k, 
                           plot_specific_neighborhoods = None,
                           size= 60)

pl_cell_type_composition_vis(data = cells_df, \
                          sample_column = "neighborhood10", \
                          cell_type_column = "original cell type", \
                          output_dir = output_dir)
    
pl_create_pie_charts(cells_df, "day harvested", "neighborhood10", show_percentages=False)


##############################################################
### Analysis of cellular communities (Step 8)

# Community analysis
# Arguments for community analysis:
data = cells_df
X = X
Y = Y
reg = "unique_region"
cluster_col_commun = "Neighborhood common"
# ks_commun = [10] # k=n means it collects n nearest neighbors for each center cell
output_dir = output_dir
k_commun = 100
n_communities_commun = 8
plot_specific_community = [2,4,5]
df2 = pd.concat([df,pd.get_dummies(df[cluster_col_commun])],1)
sum_cols = df2[cluster_col_commun].unique()
values = df2[sum_cols].values
k_centroids = {}


cells_df2, neighborhood_name, k_centroids = tl_community_analysis_2(data = data, 
                        values = values, 
                        sum_cols = sum_cols, 
                        X = X, 
                        Y = Y, 
                        reg = reg, 
                        cluster_col = cluster_col_commun, 
                        k = k_commun, 
                        n_neighborhoods = n_communities_commun,
                        elbow = False)

pl_community_analysis_2(data = cells_df2, 
                        values = values, 
                        sum_cols = sum_cols, 
                        output_dir = output_dir, 
                        X = X, 
                        Y = Y, 
                        reg = reg, 
                        save_path = None, 
                        k = k_commun, 
                        neighborhood_name = neighborhood_name,
                        k_centroids = k_centroids,
                        plot_specific_community = None, 
                        size=60)

# highlight specific cimmunities
reg_3_2 = cells_df2[cells_df2["unique_region"] == "3_2"]

pl_highlighted_dot(reg_3_2, 
                   x_col = "x", 
                   y_col = "y", 
                   group_col = 'community100', 
                   highlight_group = 3, 
                   highlight_color = "red", 
                   region_col= "unique_region",
                   subset_col=None, 
                   subset_list=None)


##############################################################
### Interactive visualization with TissUUmaps (Step 9)
import tissuumaps.jupyter as tj

# Replace 'directory_path' with the actual path to your chosen directory
directory_path = '/Volumes/homes/admin/Tim/Tissuemapstest/bestFocus/individual_channels'
tif_filepaths_list = hf_get_tif_filepaths(directory_path)

# Print the list of tif filepaths
for filepath in tif_filepaths_list:
    print(filepath)

# File path to csv file with xy positions and labels
csv_path = "/Volumes/homes/admin/Tim/Tissuemapstest/bestFocus/df_xy_cluster_RBT_170523_70066.csv"

# Start TissUUmaps viewer
tj.loaddata(images= tif_filepaths_list, 
                            csvFiles=[csv_path], 
                            xSelector='x', 
                            ySelector='y', 
                            keySelector="leiden_0.4", 
                            nameSelector=None, 
                            colorSelector=None, 
                            piechartSelector=None, 
                            shapeSelector=None, 
                            scaleSelector=None, 
                            fixedShape=None, 
                            scaleFactor=1, 
                            colormap=None, 
                            compositeMode='source-over', 
                            boundingBox=None, 
                            port=5100, 
                            host='localhost', 
                            height=1000, 
                            tmapFilename='_project', 
                            plugins=[])


##############################################################
### Cell type differentail enrichment analysis (Step 10)



##############################################################
### Canonical Correlation Analysis (CCA) (Step 11)



##############################################################
### Tensor Decomposition (Step 12)



##############################################################
### Distance permutation analysis (Step 13)
%reload_ext rpy2.ipython

%%R -i cells_df2

# load packages
library(tidyverse)
library(here)

source("triangulation_distances.R")

# Define columns of input data 
compare_condition_column <- "day harvested"
cell_index_column <- "index"
x_position_column <- "x"
y_position_column <- "y"
cell_type_column <- "Cell common"
region_column <- "unique_region"


# Define file path to save avg. distances 
filepath_avg_dist <- "Results"
dir.create(filepath_avg_dist)

# Define number of iterations for iterated distances
number_of_iterations <- 100
distance_threshold = 128
# Settings for Dumbbell plot
#pairs_for_comparisson_Dumbbell_plot <- c(unique(df_full$cell_type))

#################################################################################

df= cells_df2

colnames(df)[1] = "index"
cell_index_column <- "index"
df = df[,c(cell_index_column, x_position_column, y_position_column, compare_condition_column, region_column, cell_type_column)] 

metadata <- df %>%
  dplyr::select(compare_condition_column, region_column) %>%
  dplyr::distinct(.keep_all = TRUE)    

#################################################################################


triangulation_distances <- get_triangulation_distances(df_input = df,
                                                       id = cell_index_column,
                                                       x_pos = x_position_column,
                                                       y_pos = y_position_column,
                                                       cell_type = cell_type_column,
                                                       region = region_column,
                                                       calc_avg_distance = TRUE,
                                                       csv_output = filepath_avg_dist,
                                                       num_cores = 16)

head(triangulation_distances)

write.csv(triangulation_distances, paste0(filepath_avg_dist, "/", "triangulation_distances", ".csv"))

# Iterations
# In the iterated distances, distances are summarized per region for each iteration.
#Note: you don't need to shuffle the cell annotations yourself, it's done in the iteration for you



iterated_triangulation_distances <- iterate_triangulation_distances(df_input = df,
                                                                    id = cell_index_column,
                                                                    x_pos = x_position_column,
                                                                    y_pos = y_position_column,
                                                                    cell_type = cell_type_column,
                                                                    region = region_column,
                                                                    num_iterations = number_of_iterations,
                                                                    num_cores = 16)
#head(iterated_triangulation_distances)

write.csv(iterated_triangulation_distances, paste0(filepath_avg_dist, "/", "iterated_triangulation_distances","_",as.character(number_of_iterations),  ".csv"))  

#################################################################################
#triangulation_distances <- read_csv(paste0(filepath_avg_dist, "/", "triangulation_distances", ".csv"))
#iterated_triangulation_distances <- read_csv(paste0(filepath_avg_dist, "/", "iterated_triangulation_distances","_",as.character(number_of_iterations),  ".csv"))
#################################################################################
#### Further analysis triangulation - modified
names(metadata)[names(metadata) == compare_condition_column] <- "treatment"
# Reformat observed dataset
observed_distances <- triangulation_distances %>%
  # Append metadata
  dplyr::left_join(metadata,
    by = c("unique_region")
  ) %>%
  dplyr::filter(distance <= distance_threshold) %>%
  # Calculate the average distance to every cell type for each cell
  dplyr::group_by(celltype1_index, celltype1, celltype2, treatment, unique_region) %>%
  dplyr::summarize(mean_per_cell = mean(distance)) %>%
  dplyr::ungroup() %>%
  # Calculate the average distance between cell type to cell type on a per group basis
  dplyr::group_by(celltype1, celltype2, treatment) %>%
  dplyr::summarize(
    observed = list(mean_per_cell),
    observed_mean = mean(unlist(observed), na.rm = TRUE)
  ) %>%
  dplyr::ungroup()

# Reformat exepcted dataset
expected_distances <- iterated_triangulation_distances %>%
  # Append metadata
  dplyr::left_join(metadata,
    by = c("unique_region")
  ) %>%
  dplyr::filter(mean_dist <= distance_threshold) %>%
  # Calculate expected mean distance and list values
  dplyr::group_by(celltype1, celltype2, treatment) %>%
  dplyr::summarize(
    expected = list(mean_dist),
    expected_mean = mean(mean_dist, na.rm = TRUE)
  ) %>%
  dplyr::ungroup()

# drop comparisons with low numbers of observations 
# This step was implemented to reduce the influence of rare cell types - usually, these tend to dominate statistics as small changes are already highly significant 
logical_list_observed <- list()
for (i in 1:nrow(observed_distances)) {
  print(i)
  vec <- observed_distances[i, "observed"]

  if (length(unlist(vec)) > 10) {
    logical_list_observed[[as.character(i)]] <- TRUE
  } else {
    logical_list_observed[[as.character(i)]] <- FALSE
  }
}

list_observed <- unlist(logical_list_observed)  

observed_distances$keep <- list_observed

observed_distances <- observed_distances %>% filter(keep == TRUE)

# Perform the same filtering on the expected distances
# drop comparisons with low numbers of observations 
# This step was implemented to reduce the influence of rare cell types - usually, these tend to dominate statistics as small changes are already highly significant 
logical_list_expected <- list()
for (i in 1:nrow(expected_distances)) {
  print(i)
  vec <- expected_distances[i, "expected"]

  if (length(unlist(vec)) > 10) {
    logical_list_expected[[as.character(i)]] <- TRUE
  } else {
    logical_list_expected[[as.character(i)]] <- FALSE
  }
}

list_observed <- unlist(logical_list_expected)  

expected_distances$keep <- list_observed

expected_distances <- expected_distances %>% filter(keep == TRUE)

# Calculate pvalues and log fold differences
distance_pvals <- expected_distances %>%
  dplyr::left_join(observed_distances,
    by = c("celltype1", "celltype2", "treatment")
  ) %>%
  # Calculate wilcoxon test between observed and expected distances
  dplyr::group_by(celltype1, celltype2, treatment) %>%
  dplyr::mutate(pvalue = wilcox.test(unlist(expected), unlist(observed), exact = FALSE)$p.value) %>%
  dplyr::ungroup() %>%
  dplyr::select(-observed, -expected) %>%
  # Calculate log fold enrichment
  dplyr::mutate(
    logfold_group = log2(observed_mean / expected_mean),
    interaction = paste0(celltype1, " --> ", celltype2)
  )

# Get order of plot by magnitude of logfold differences between groups
intermed <- distance_pvals %>%
  dplyr::select(interaction, treatment, logfold_group) %>%
  tidyr::spread(key = treatment, value = logfold_group)

#intermed$difference <- (intermed[, condition_2] - intermed[, condition_1])

#ord <- (intermed %>%
 # dplyr::filter(!is.na(difference)) %>%
  #dplyr::arrange(condition_1))$interaction

# Assign interaction order
#distance_pvals$interaction <- factor(distance_pvals$interaction,
 # levels = ord
#)

distance_pvals <- write_csv(distance_pvals, paste0(specific_output, "/distance_pvals.csv"))

# general filtering before analysis of the results
distance_pvals_sig <- distance_pvals %>%
  filter(pvalue < 0.05) %>% # keep only significant results
  filter(celltype1 != celltype2) %>% # compare only different cell types 
  filter(!is.na(observed_mean)) %>% # ignore columns without observation
#  filter(celltype1 != "unknown") %>% # drop cells of type unknown
 # filter(celltype2 != "unknown") %>%
#  filter(celltype1 != "noise") %>% # drop cells of type noise 
 # filter(celltype2 != "noise") %>% 
  filter(!is.na(treatment))
  
#################################################################################

distance_pvals_sig <- distance_pvals_sig %>% 
  filter(!is.na(interaction)) %>% 
  filter(!is.na(logfold_group))
distance_pvals_sig$abs_logfold <- abs(distance_pvals_sig$logfold_group)
distance_pvals_sig_filt <- distance_pvals_sig %>% filter(abs_logfold >= 0.5)
distance_pvals_sig_filt <- distance_pvals_sig_filt[!duplicated(t(apply(distance_pvals_sig_filt[c("celltype1", "celltype2")], 1, sort))), ]

distance_pvals_interesting <- distance_pvals[distance_pvals$interaction %in% distance_pvals_sig_filt$interaction, ]

distance_pvals_interesting <- distance_pvals_interesting %>% filter(!is.na(treatment))

distance_pvals_interesting <- distance_pvals[distance_pvals$interaction %in% comb_filt$interaction, ]
distance_pvals_interesting <- distance_pvals_interesting %>% filter(!is.na(treatment))

#################################################################################

# Pairs to plot in Dumbell plot
pair_to = unique(distance_pvals_sig_filt$interaction)

# Colors used in Dumbell plot 
colors = c("#8de4d3", "#a21636", "#94ea5b")

#################################################################################

# Dumbbell plot
data <- distance_pvals %>%
  dplyr::filter(!is.na(interaction))

distance_pvals$pairs <- paste0(distance_pvals$celltype1, "_", distance_pvals$celltype2)
distance_pvals_sub <- distance_pvals[distance_pvals$interaction %in% pair_to, ]

distance_pvals_sub <- distance_pvals_sub %>% filter(!is.na(treatment)) %>% arrange(logfold_group) %>% mutate(interaction = factor(interaction, unique(interaction)))
#distance_pvals_sub$interaction <- factor(distance_pvals_sub$interaction, levels=unique(distance_pvals_sub$interaction))


distance_pvals_sub_filt <- distance_pvals_sub #%>% filter(!treatment == 5)
distance_pvals_sub_filt$treatment <- as.factor(distance_pvals_sub_filt$treatment)

ggplot2::ggplot(data = distance_pvals_sub_filt %>%
  dplyr::filter(!is.na(interaction))) +
  ggplot2::geom_vline(mapping = ggplot2::aes(xintercept = 0), linetype = "dashed") +
  ggplot2::geom_line(
    mapping = ggplot2::aes(x = logfold_group, y = interaction),
    na.rm = TRUE
  ) +
  ggplot2::geom_point(
    mapping = aes(x = logfold_group, y = interaction, fill = treatment, shape = treatment),
    size = 4, stroke = 0.5, na.rm = TRUE
  ) +
  ggplot2::scale_shape_manual(values = c(24, 22, 23)) +
  ggplot2::scale_fill_manual(values = colors) +
  ggplot2::theme_bw() +
  ggplot2::theme(
    panel.grid.major.x = element_blank(),
    panel.grid.minor.x = element_blank(),
    axis.text.y = element_text(size = 16),
    axis.text.x = element_text(size = 16, angle = 45, hjust = 1),
    axis.title.y = element_text(size = 16),
    axis.title.x = element_text(size = 16)
  )

ggsave(paste0(output_path, "dumbbell.pdf"))

#################################################################################

library(tidyverse)
library(igraph)
library(ggraph)
library(scales)
library(ggplot2)

pairs <- unique(distance_pvals_sig_filt$interaction)

distance_pvals_sub2 <- distance_pvals[distance_pvals$interaction %in% pairs, ]


distance_pvals_sub_grouped <- distance_pvals_sub2 %>% group_by(celltype1, celltype2)
distance_pvals_sub_grouped <- na.omit(distance_pvals_sub_grouped)

distance_pvals_sub2 <- na.omit(distance_pvals_sub2)

pairs <- unique(distance_pvals_sub2$interaction)

result_list_day1 <- list()
result_list_day3 <- list()
result_list_day5 <- list()

for (p in pairs) {
  distance_pvals_sub_filt_1 <- distance_pvals_sub2 %>%
    filter(interaction == p) %>%
    filter(treatment == 1)
  distance_pvals_sub_filt_2 <- distance_pvals_sub2 %>%
    filter(interaction == p) %>%
    filter(treatment == 3)
  distance_pvals_sub_filt_3 <- distance_pvals_sub2 %>%
    filter(interaction == p) %>%
    filter(treatment == 5)

  if (nrow(distance_pvals_sub_filt_1) > 0) {
    if (0 > distance_pvals_sub_filt_1$logfold_group) {
      direction <- "#3976AC"
    } else {
      direction <- "#C63D30"
    }

    df_res2 <- data.frame(
      celltype1 = distance_pvals_sub_filt_1$celltype1,
      celltype2 = distance_pvals_sub_filt_1$celltype2,
      logfold = distance_pvals_sub_filt_1$logfold_group,
      direction = direction
    )
    result_list_day1[[p]] <- df_res2
  }

  if (nrow(distance_pvals_sub_filt_2) > 0) {
    if (0 > distance_pvals_sub_filt_2$logfold_group) {
      direction <- "#3976AC"
    } else {
      direction <- "#C63D30"
    }

    df_res1 <- data.frame(
      celltype1 = distance_pvals_sub_filt_2$celltype1,
      celltype2 = distance_pvals_sub_filt_2$celltype2,
      logfold = distance_pvals_sub_filt_2$logfold_group,
      direction = direction
    )
    result_list_day3[[p]] <- df_res1
  }

  if (nrow(distance_pvals_sub_filt_3) > 0) {
    if (0 > distance_pvals_sub_filt_3$logfold_group) {
      direction <- "#3976AC"
    } else {
      direction <- "#C63D30"
    }

    df_res3 <- data.frame(
      celltype1 = distance_pvals_sub_filt_3$celltype1,
      celltype2 = distance_pvals_sub_filt_3$celltype2,
      logfold = distance_pvals_sub_filt_3$logfold_group,
      direction = direction
    )
    result_list_day5[[p]] <- df_res3
  }
}

graph_df_day1 <- as.data.frame(do.call(rbind, result_list_day1))
graph_df_day3 <- as.data.frame(do.call(rbind, result_list_day3))
graph_df_day5 <- as.data.frame(do.call(rbind, result_list_day5))

graph_list <- list(graph_df_day1, graph_df_day3, graph_df_day5)

for (x in 1:3) {
  graph_df <- graph_list[[x]]


  graph_df <- graph_df[!duplicated(t(apply(graph_df[c("celltype1", "celltype2")], 1, sort))), ]

  mat <- graph_df

  cci_control <- mat


  g <- graph_from_data_frame(data.frame(cci_control))
  E(g)$weights <- ifelse(cci_control$logfold == 0,
    1e-10, abs(cci_control$logfold)
  )

  color_dic_cells <- c('Tumor PDL1+ MHCI+' = "#21f0b6",
                     'CD8+ T cell PD1+' = "#2a8665",
                     'Tumor TYRP1+' = "#79c6c1",
                     'DC' = "#2a538a",
                     'Macrophage' = "#daa4f9",
                     'Tumor' = "#7e39c2",
                     'Endothelial CD106+' = "#9ab9f9",
                     'CD8+ T cell' = "#9e3678",
                     'DC TCF7+' = "#f36ad5",
                     'Epiehtlial' = "#9f04fc",
                     'CD4+ T cell' = "#5ac230",
                     'Epithelial' = "#b7d165",
                     'Tumor Ki67+' = "#6d3918",
                     'CD4+ Treg' = "#efaa79",
                     'Neutrophil' = "#9f2114",
                     'Endothelial' = "#fd5917",
                     'NK' = "#fe1d66",
                     'Macrophage PDL1+' = "#f7767d",
                     'APC MHCIIhi' = "#fbbd13",
                     'Macrophage CD86+' = "#748d13",
                     'Lymphatic' = '#00944F',
                     'CD86+ Macrophage' = '#636A86',
                     'Macrophage CD169+' = '#7A50F7',
                     'Tumor CD117hi' = '#24B93B',
                     'Lymphatic Ly6Chi' = '#9BF29D',
                     'NK cell KLRG1hi' = '#829824',
                     'B cell' = '#F2A3F0')

  
  
  V(g)$color <- color_dic_cells[unique(c(mat$celltype1, mat$celltype2))] 
  # pdf("figures/WilkEtAl/cellchat_CCI_network_byCondition_Control_network.pdf",
  #     width = 8,
  #     height = 6)


  radian.rescale <- function(x, start = 0, direction = 1) {
    c.rotate <- function(x) (x + start) %% (2 * pi) * direction
    c.rotate(scales::rescale(x, c(0, 2 * pi), range(x)))
  }

  lab.locs <- radian.rescale(x = 1:18, direction = -1, start = 0)

  plot(g,
    vertex.size = 17,
    vertex.color = V(g)$color,
    vertex.label.color = "black",
    vertex.label.cex = 1,
    #vertex.label.dist = 2.5,
    vertex.label.degree = lab.locs,
    edge.width = E(g)$weights * 15,
    edge.arrow.size = log(1 / E(g)$weights) / 80,
    edge.color = E(g)$direction,
    edge.curved = 0,
    asp = 0.9,
    layout = layout_in_circle,
    main = paste0(compare_condition_column, "_", unique(graph_df$group))
  )
  # dev.off()
}


##############################################################
### CN interface detection (Step 14)

# Spatial Context
data = cells_df2
col_list = data.columns
# Spatial context 
n_num = 75
ks=[n_num]
cluster_col = "Neighborhood common"
sum_cols=cells_df2[cluster_col].unique()
keep_cols = col_list
X='x'
Y='y'
#Neigh = Neighborhoods(cells_df2,ks,cluster_col,sum_cols,keep_cols,X,Y,reg=Reg,add_dummies=True)
#windows = Neigh.k_windows()
reg = "unique_region"

windows, sum_cols = tl_Create_neighborhoods(df = data,
                     n_num = n_num,
                     cluster_col = cluster_col,
                     X = X,
                     Y = Y,
                     regions = reg,
                     sum_cols = None,
                     keep_cols = None,
                     ks = [n_num])

w, k_centroids = tl_Chose_window_size(windows,
                      n_num = n_num,
                      n_neighborhoods = 10,
                      n2_name = 'neigh_ofneigh', sum_cols = sum_cols)

pl_Niche_heatmap(k_centroids, w, n_num, sum_cols)


names = cells_df2[cluster_col].unique()
colors = hf_generate_random_colors(n = len(names))

color_dic = hf_assign_colors(names, colors)

color_dic=color_dic
l=list(color_dic.keys())



plot_list = [ 'Vasculature' , 'Tumor','Immune Infiltrate']

pl_Barycentric_coordinate_projection(w, 
                                      plot_list = plot_list, 
                                      threshold = 70, 
                                      output_dir = output_dir, 
                                      save_name = save_name, 
                                      col_dic = color_dic,
                                      l = l,
                                      cluster_col = cluster_col,
                                      n_num = n_num,
                                      SMALL_SIZE = 14, 
                                      MEDIUM_SIZE = 16, 
                                      BIGGER_SIZE = 18,
                                      figsize=(6,5))


##############################################################
### CN spatial context analysis (Step 15)

simps, simp_freqs, simp_sums = tl_calculate_neigh_combs(w, 
                                                     l,
                                                     n_num, 
                                                     threshold = 0.85, 
                                                     per_keep_thres = 0.85)

g, tops, e0, e1 = tl_build_graph_CN_comb_map(simp_freqs)

pl_generate_CN_comb_map(graph = g, 
                     tops = tops, 
                     e0 = e0, 
                     e1 = e1, 
                     l = l,
                     simp_freqs = simp_freqs,
                     color_dic = color_dic)

pl_get_network(ttl_per_thres=0.9,
            comb_per_thres=0.005,
            neigh_sub = plot_list,
            save_name='All_comm',
            save_path = output_dir,
            sub_col = cluster_col, 
            color_dic = color_dic,
            windows = windows,
            n_num = n_num,
            l = l)


##############################################################
### Patch-based fragmentation analysis (Step 16)

data_path = "Z:/admin/Tim/Fragmentation_analysis_pub/Data/HuBMAP_gut/CODEX_HuBMAP_alldata_Dryad.csv"
voronoi_output_path = "Z:/admin/Tim/Fragmentation_analysis_pub/Mock_test_set/Results/Voronoi/"

df = pd.read_csv(data_path)
generate_voronoi_plots(df, voronoi_output_path)

image_folder = 'D:/Tim/doi_10.5061_dryad.76hdr7t1p__v3'
mask_output = "Z:/admin/Tim/Fragmentation_analysis_pub/Mock_test_set/Results/Masks/"

generate_masks_from_images(image_folder, mask_output, image_type = ".tif", filter_size = 5, threshold_value = 10)

filter_list = ['colon', 'B009Bt', 'SB']

df_info = generate_info_dataframe(df, voronoi_output_path, mask_output, filter_list)

folder_names = filtered_df['folder_names'].unique()
print(filtered_df)
print(folder_names)

voronoi_path, mask_path, region = hf_process_dataframe(df_info)

process_files(voronoi_path, mask_path, region)
    
df_info = df_info
output_dir_csv ="C:/Users/Tim/Downloads/Test_patch_20230712/csv"

results_df, contour_list = process_data(df_info, output_dir_csv)