#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 25 13:39:10 2023

@author: timnoahkempchen
"""
#############################################################
# Example pipeline 
#############################################################
df = pd.read_csv("/Users/timnoahkempchen/Desktop/SAP/Data/22_08_09_CellAtlas_only_Processed_Metadata_subsample10k.csv")
df.columns

# Shannon diversity
# Set filepaths 
input_filepath = "/Users/timnoahkempchen/Library/CloudStorage/GoogleDrive-timkem@stanford.edu/Meine Ablage/Datasets/data/22_08_09_CellAtlas_only_Processed_Metadata.csv"
output_filepath = "/Users/timnoahkempchen/Downloads/Output_test"

# mandatory 
## define column that defines replicates 
rep ='unique_region'

## specify column used to group the data 
group_com = 'community' # order can specify the order in which group_com is plotted on the x axis 
# used by helper function 
## a column name from the input dataframe, defines the category for which the percentage will be calculated
per_categ = 'Cell Type' 
## a column name from the input dataframe, defines a subset of the data to use
sub_column = 'Major Cell Cat'
## define string which is used to subset sub_column  
sub_l = ['Immune','Epithelial','Mesenchymal']

# optional
## Set specific colors for figure 
coloring = None
## Specify order for x axis 
ordering = None 
## Save figure - boolean 
save = True
## change figure size 
fig_size=8

# Stacked bar plot
per_cat = "Major Cell Cat"
grouping = 'consensus diagnosis'

norm=True
save_name= 'Major_subConsensus'

sub_col= 'Cell Type'
name_cat = 'Cell Type'
fig_sizing=(8,4)

pal_color=None
remove_leg=False
save_path = "/Users/timnoahkempchen/Downloads/TestFolder/"

# Swarm plot 
sub_list = ['CD4+ Treg']

cell_list = df["Major Cell Cat"].unique()

output_dir = "/Users/timnoahkempchen/Downloads/Trest_triang/Test_random"

#######
#order_tis = ['M1 Macrophage', 'M2 Macrophage']
#h_ordertu = ['Mucosa', 'Muscularis mucosa', 'Submucosa', 'Muscularis externa']
ab = stacked_bar_plot(data = df, per_cat = per_cat, grouping = grouping, sub_col= None,\
          cell_list = cell_list, norm=False,fig_sizing=fig_sizing, name_cat = per_cat,  \
                    col_order=None, save_name=save_name, pal_color=pal_color,) #h_order=h_ordertu,  pal_color=col_map
    
#######
total_neigh_st = swarm_box(data=df, grouping=grouping, replicate= rep,\
                           sub_col=sub_col, sub_list=sub_list, norm=False, per_cat= per_cat, \
                           figure_sizing=(1.5,3), save_name='sig_imm_cons', pal_color=None, h_order=None,\
                           flip=True)
    
#######
# use function 
# for different conditions
result, pval, tukey_tab = Shan_div(data1=df, sub_l = sub_l,\
        group_com = group_com, per_categ = per_categ,\
        rep=rep, sub_column=sub_column,normalize=True, save=save,\
                          coloring= coloring, fig_size=fig_size, ordering=None, output_filepath = output_filepath)
    
#######
cell_type_composition_vis(df, sample_column = "sample", cell_type_column = "Cell Type", output = output_dir)


#######
# Prepare dataframe for neighborhood analysis 
#Import Data
sample_column = "sample"
cell_type_column = "Cell Type"
region_column = "unique_region"
x_position_column = "x"
y_position_column = "y"
treatment_column ="consensus diagnosis"

ks = [20, 30, 35] # k=n means it collects n nearest neighbors for each center cell
cluster_col = "Cell Type"

cellhier_path = 'cellhier/'

k = 35
n_neighborhoods = 30
k_centroids = {}

df2 = pd.concat([df,pd.get_dummies(df[cluster_col])],1)
sum_cols = df2[cluster_col].unique()
values = df2[sum_cols].values

cells_df = neighborhood_analysis(df = df2, X = x_position_column, Y = y_position_column, reg = region_column, cluster_col = cluster_col, ks = ks, save_path = output_dir, k = k, n_neighborhoods = n_neighborhoods, save_to_csv= True)






