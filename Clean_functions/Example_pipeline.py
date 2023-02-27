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
#############################################################
# Functions from cell_type percent script 
#############################################################
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
# correlation analysis 
sub_l = ['Immune','Epithelial','Mesenchymal']
all_pair, sub_pari = corr_cell(data=df,  sub_l2=sub_l, per_categ='Cell Type', group2='Sub diagnosis', \
                               repl='unique_region',  sub_collumn= 'Major Cell Cat', cell='CD4+ Treg',\
                               normed=True, thres=0.7, cell2 = 'Endothelial CD36hi')

cor_mat, mp = cor_plot(data = df, group1 = 'Major Cell Cat',per_cat = 'Cell Type', sub_col= 'Major Cell Cat', sub_list= sub_l,norm=True, count=False, plot_scatter=False)

cell_type = 'Epithelial CK7+'
piar1 = all_pair.loc[all_pair['col1']==cell_type]
piar2 = all_pair.loc[all_pair['col2']==cell_type]
piar=pd.concat([piar1,piar2])
piar

pair_list = list(set(list(piar['col1'].unique())+list(piar['col2'].unique())))
pair_list

sl = pair_list

cor_subplot(mp=mp, sub_list=sl)

#############################################################
# Functions from neighborhood analysis script 
#############################################################

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

k = 35
n_neighborhoods = 30
k_centroids = {}

df2 = pd.concat([df,pd.get_dummies(df[cluster_col])],1)
sum_cols = df2[cluster_col].unique()
values = df2[sum_cols].values

cells_df = neighborhood_analysis(df = df2, X = x_position_column, Y = y_position_column, reg = region_column, cluster_col = cluster_col, ks = ks, save_path = output_dir, k = k, n_neighborhoods = n_neighborhoods, save_to_csv= True)

# This is yet another variation of the neighbourhood analysis function. The plots are a bit different but in general it is doing the same
cells_df = xycorr(df=cells,sample_col='date_array', y_rows=4, x_columns=4, X_pix=4032, Y_pix=3024)


#############################################################
# Cell Type Differential Enrichment 
#############################################################

ID_component1 = 'sample'
ID_component2 = 'Block type'
neighborhood_col = 'neigh_name'
group_col = 'Core Type'
group_dict = {'Dysplasia':0, 'Adenocarcinoma':1, "Barrett's Metaplasia":2, "Squamous":3}
cell_type_col = 'Cell Type'

cells2, ct_freq, all_freqs, pat_to_gp, neigh_num = cell_types_de_helper(df = cells_df, ID_component1 = ID_component1, ID_component2 = ID_component2, neighborhood_col = neighborhood_col, group_col = group_col, group_dict = group_dict, cell_type_col = cell_type_col)

neighborhood_col_number = 'neigh_num'
nbs = list(cells2['neigh_num'].unique())
patients = list(cells2['patients'].unique()) 
group = pd.Series(pat_to_gp)
cells = list(cells2['Cell Type'].unique())
#cells = ['Tumor','CD4+ Treg']
cells1 = cells.copy()
cells1.append('patients')
cells1

cell_types_de(ct_freq = ct_freq, all_freqs = all_freqs, neighborhood_num = neighborhood_col_number, nbs = nbs, patients = patients, group = group, cells = cells, cells1 = cells1, neigh_num = neigh_num)
