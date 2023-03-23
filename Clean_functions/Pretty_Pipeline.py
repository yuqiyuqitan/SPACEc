#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  8 12:53:53 2023

@author: timnoahkempchen
"""

# Arguments needed throughout analysis sorted by analysis step

#######################################################################################################################################################
# CHUNK_1: Basic User Input
#######################################################################################################################################################

#############################################################
# Filepaths 
#############################################################

input_file = "/Users/timnoahkempchen/Library/CloudStorage/GoogleDrive-timkem@stanford.edu/Meine Ablage/Datasets/data/22_08_09_CellAtlas_only_Processed_Metadata.csv"

output_dir = "/Users/timnoahkempchen/Downloads/Output_testnew/"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

#############################################################
# Reading the data 
#############################################################
df = pd.read_csv(input_file)

#############################################################
# Specify key column names 
#############################################################
sample_column = "sample" # Column specifies the analyzed samples. This could be an Identifier or a Case ID, etc. 
cell_type_column = "Cell Type" # Column specifies the respective cell type. This might be replaced with a broader cell types based on the specific analysis 
region_column = "unique_region" # This column should contain unique IDs for the respective regions 
X = "x" # Column containing coordinates on x axis 
Y = "y" # Column containing coordinates on y axis 
treatment_column ="consensus diagnosis" # Column containing comparisson, usally some kind of treatment/condition

#######################################################################################################################################################
# CHUNK_2: Basic Visualization
#######################################################################################################################################################

# This part of the pipeline aims to analize and visulize the basic cell type composition of the analyzed samples 

#############################################################
# Specify additional information
#############################################################
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


# Swarm plot 
sub_list = ['CD4+ Treg']
replicate_column = "unique_region"

cell_list = df["Major Cell Cat"].unique()

# correlation analysis 
sub_l = ['Immune','Epithelial','Mesenchymal']
save = True
## Set specific colors for figure 
coloring = None
## Specify order for x axis 
ordering = None 
## change figure size 
fig_size=8

#############################################################
# Functions from cell_type percent script 
#############################################################

####### Stacked Bar Plot
# Shows percentage of category per group. For example: major cell types per diagnosis 
ab = stacked_bar_plot(data = df, \
                      per_cat = per_cat, \
                      grouping = grouping, \
                      output_dir = output_dir, \
                      sub_col= None, \
                      cell_list = cell_list,\
                      norm=False, \
                      fig_sizing=fig_sizing, \
                      name_cat = per_cat, \
                      col_order=None, \
                      save_name= "stacked_bar", \
                      pal_color=pal_color) 
    
####### Swarm Boxplot 
total_neigh_st = swarm_box(data=df, \
                           grouping=grouping, \
                           replicate = replicate_column, \
                           sub_col=sub_col, \
                           sub_list=sub_list, \
                           output_dir = output_dir, \
                           norm=False, \
                           per_cat= per_cat, \
                           figure_sizing=(1.5,3), \
                           save_name='swarm_box', \
                           pal_color=None, \
                           h_order=None, \
                           flip=True)

####### Swarm Boxplot 
result, pval, tukey_tab = Shan_div(data1=df, \
                                   sub_l = sub_l, \
                                   group_com = grouping, \
                                   per_categ = per_cat, \
                                   rep = replicate_column, \
                                   sub_column=sub_col, \
                                   normalize=True, \
                                   save=save, \
                                   coloring= coloring, \
                                   fig_size=fig_size, \
                                   ordering=None, \
                                   output_dir = output_dir)
    
####### Correlation Analysis 

all_pair, sub_pari = corr_cell(data=df, \
                               sub_l2=sub_l, \
                               per_categ='Cell Type', \
                               group2='Sub diagnosis', \
                               repl='unique_region', \
                               sub_collumn= 'Major Cell Cat', \
                               cell='CD4+ Treg', \
                               normed=True, \
                               thres=0.7, \
                               cell2 = 'Endothelial CD36hi',
                               output_dir = output_dir, 
                               save_name = "cell1_cell2_cor")

cor_mat, mp = cor_plot(data = df, \
                       group1 = 'Major Cell Cat', \
                       per_cat = 'Cell Type', \
                       sub_col= 'Major Cell Cat', \
                       sub_list= sub_l, \
                       norm=True, \
                       count=False, \
                       plot_scatter=False)

cell_type = 'Epithelial CK7+'
piar1 = all_pair.loc[all_pair['col1']==cell_type]
piar2 = all_pair.loc[all_pair['col2']==cell_type]
piar=pd.concat([piar1,piar2])
piar

pair_list = list(set(list(piar['col1'].unique())+list(piar['col2'].unique())))
pair_list

sl = pair_list

cor_subplot(mp=mp, \
            sub_list=sl)


#######################################################################################################################################################
# CHUNK_3: Neighborhood and Community Analysis 
#######################################################################################################################################################

#############################################################
# Specify additional column names 
#############################################################

####### Neighborhood analysis 
ks = [20, 30, 35] # k=n means it collects n nearest neighbors for each center cell
cluster_col = "Cell Type"
k = 35
n_neighborhoods = 30
k_centroids = {}

####### Community analysis 
ks_commun = [20] # k=n means it collects n nearest neighbors for each center cell
cluster_col_commun = "neighborhood35"
k_commun = 20
n_communities_commun = 30
k_centroids = {}

#############################################################
# Neighborhood Analysis 
#############################################################

####### Visulize overall cell type composition
cell_type_composition_vis(df, \
                          sample_column = "sample", \
                          cell_type_column = "Cell Type", \
                          output_dir = output_dir)

####### Neighborhood analysis 
df2 = pd.concat([df,pd.get_dummies(df[cluster_col])],1)
sum_cols = df2[cluster_col].unique()
values = df2[sum_cols].values

cells_df = neighborhood_analysis(df = df2, \
                                 X = X, \
                                 Y = Y, \
                                 reg = region_column, \
                                 cluster_col = cluster_col, \
                                 ks = ks, \
                                 output_dir = output_dir, \
                                 k = k, \
                                 n_neighborhoods = n_neighborhoods, \
                                 save_to_csv= True, \
                                 plot_specific_neighborhoods = [2,4])
    
#############################################################
# Community Analysis 
#############################################################

cells_df2 = community_analysis(df = cells_df, \
                               X = X, \
                               Y = Y, \
                               reg = region_column, \
                               cluster_col = cluster_col_commun, \
                               ks = ks_commun, \
                               output_dir = output_dir, \
                               k = k_commun, \
                               n_neighborhoods = n_communities_commun, \
                               plot_specific_community = [2,4,5])


#######################################################################################################################################################
# CHUNK_4: Analysis dependent on neighborhood analysis 
#######################################################################################################################################################

#############################################################
# Specify additional column names 
#############################################################

####### Cell Type Differential Enrichment 
ID_component1 = 'sample'
ID_component2 = 'Block type'
neighborhood_col = 'neigh_name'
group_col = 'Core Type'
group_dict = {'Dysplasia':0, 'Adenocarcinoma':1, "Barrett's Metaplasia":2, "Squamous":3}
cell_type_col = 'Cell Type'

neighborhood_col_number = 'neigh_num'

#############################################################
# Cell Type Differential Enrichment 
#############################################################

cells2, ct_freq, all_freqs, pat_to_gp, neigh_num = cell_types_de_helper(df = cells_df, \
                                                                        ID_component1 = ID_component1, \
                                                                        ID_component2 = ID_component2, \
                                                                        neighborhood_col = neighborhood_col, \
                                                                        group_col = group_col, \
                                                                        group_dict = group_dict, \
                                                                        cell_type_col = cell_type_col)

nbs = list(cells2[neighborhood_col_number].unique())
patients = list(cells2['patients'].unique()) 
group = pd.Series(pat_to_gp)
cells = list(cells2['Cell Type'].unique())
#cells = ['Tumor','CD4+ Treg']

cells1 = cells.copy()
cells1.append('patients')
cells1

cell_types_de(ct_freq = ct_freq, \
              all_freqs = all_freqs, \
              neighborhood_num = neighborhood_col_number, \
              nbs = nbs, \
              patients = patients, \
              group = group, \
              cells = cells, \
              cells1 = cells1, \
              neigh_num = neigh_num)

#############################################################
# CCA
#############################################################

# Prepare IDs this could for example be the combination of patient ID and tissue type. Apart from that, the function assigns a number to each name from the neighborhood column
cells_df = prepare_neighborhood_df(cells_df, neighborhood_column = neighborhood_col, patient_ID_component1 = ID_component1, patient_ID_component2 = ID_component2) # this is a helper function 


# devide IDs/patients into groups
patient_to_group_dict = cells_df.loc[:,['patients',ID_component2]].drop_duplicates().set_index('patients').to_dict()[ID_component2]
group1_patients = [a for a,Sample_type in patient_to_group_dict.items() if Sample_type=="Biopsy"]
group2_patients = [a for a,Sample_type in patient_to_group_dict.items() if Sample_type=='Resection']

# Provide user feedback
print(group1_patients)

# select which neighborhoods and functional subsets
cns = list(cells_df['neigh_num'].unique())
subsets = ['CD4+ T cell']

#log (1e-3 +  neighborhood specific cell type frequency) of functional subsets) ('nsctf')
nsctf = np.log(1e-3 + cells_df.groupby(['patients','neigh_num'])[subsets].mean().reset_index().set_index(['neigh_num','patients']))

cca = CCA(n_components=1,max_iter = 5000)
func = pearsonr

# set number of permutation params
n_perms = 5000

# Run CCA
stats_group1, arr1 = Perform_CCA(cca, n_perms, nsctf, cns, subsets, group = group1_patients)

Visulize_CCA_results(CCA_results = stats_group1, save_fig = False, save_path = output_dir, save_name = "CCA_vis.png")


# Run CCA for group 2
stats_group2, arr2 = Perform_CCA(cca, n_perms, nsctf, cns, subsets, group = group2_patients)

Visulize_CCA_results(CCA_results = stats_group2, save_fig = False, save_path = output_dir, save_name = "CCA_vis.png")

#############################################################
# tensor decomposition 
#############################################################

# Prepare IDs this could for example be the combination of patient ID and tissue type. Apart from that, the function assigns a number to each name from the neighborhood column
cells_df = prepare_neighborhood_df(cells_df, neighborhood_column = neighborhood_col, patient_ID_component1 = ID_component1, patient_ID_component2 = ID_component2) # this is a helper function 

# devide IDs/patients into groups
patient_to_group_dict = cells_df.loc[:,['patients',ID_component2]].drop_duplicates().set_index('patients').to_dict()[ID_component2]
group1_patients = [a for a,Sample_type in patient_to_group_dict.items() if Sample_type=="Biopsy"]
group2_patients = [a for a,Sample_type in patient_to_group_dict.items() if Sample_type=='Resection']

# Provide user feedback
print(group1_patients)

list(cells_df['Coarse Cell'].unique())
list(cells_df['neigh_num'].unique())

# select the cts
cts = list(cells_df['Coarse Cell'].unique())
cts =['Macrophage CD169+',
 'CD4+ T cell',
 'DC',
 'Stromal',
#  'Tumor Ki67+',
#  'Tumor PDL1+ MHCI+',
#  'Tumor',
 'Macrophage',
 'Neutrophil',
 'NK',
 'CD8+ T cell PD1+',
 'CD8+ T cell',
 'CD4+ Treg',
 'B cell']

# select the cns
cns = list(cells_df['neigh_num'].unique())
#cns = [0, 1, 2, 3, 4, 5, 6]

###

# Build the tensors for each patient group
counts = cells_df.groupby(['patients','neigh_num','Coarse Cell']).size()

#initialize the tensors

dat1 = build_tensors(df = cells_df, group = group1_patients, cns = cns, cts = cts)
dat2 = build_tensors(df = cells_df, group = group2_patients, cns = cns, cts = cts)

###

# The following tries different numbers of CN modules/CT modules to determine suitable rank for decomposition

evaluate_ranks(dat1,2)
plt.show()
evaluate_ranks(dat2,2)
plt.show()

plot_modules_heatmap(dat1)
plot_modules_heatmap(dat2)

# Set a save path MOVE THIS TO TOP OF SCIPT COMBINE WITH OUTPUT 
save_path = '/Users/timnoahkempchen/Downloads/'

pal = sns.color_palette('bright',30) # Choose some random colors to demonstrate that function in working 
plot_modules_graphical(dat1, scale = 2, pal = pal, save_name = 'T cell')

#########################################################################################################
# Analysis dependent on community analysis 
#########################################################################################################

#############################################################
# Specify additional information
#############################################################

col_list = cells_df2.columns

# Spatial context 
n_num = 75
ks=[n_num]
cluster_col = 'community'
sum_cols=cells_df2[cluster_col].unique()
keep_cols = col_list
X='x'
Y='y'
Reg = 'unique_region'
Neigh = Neighborhoods(cells_df2,ks,cluster_col,sum_cols,keep_cols,X,Y,reg=Reg,add_dummies=True)
windows = Neigh.k_windows()


#Choose the windows size to continue with
w = windows[n_num]

n_neighborhoods=7
n2_name = 'neigh_ofneigh'
k_centroids = {}

km = MiniBatchKMeans(n_clusters = n_neighborhoods,random_state=0)
labels = km.fit_predict(w[sum_cols].values)
k_centroids[n_num] = km.cluster_centers_
w[n2_name] = labels


#############################################################
# Spatial context analysis 
#############################################################

n_num = 20

windows, sum_cols = Create_neighborhoods(df = cells_df,
                     n_num = n_num,
                     cluster_col = 'community',
                     X = 'x',
                     Y = 'y',
                     sum_cols = None,
                     keep_cols = None,
                     ks = [n_num])

w, k_centroids = Chose_window_size(windows,
                      n_num = n_num,
                      n_neighborhoods = 10,
                      n2_name = 'neigh_ofneigh', sum_cols = sum_cols)

Niche_heatmap(k_centroids, w)


names = cells_df2[cluster_col].unique()
colors = generate_random_colors(n = len(names))

color_dic = assign_colors(names, colors)

pal_color=color_dic
l=list(pal_color.keys())



plot_list = list_n = [ 'Atrophic Cardiac Enriched', "Inflamed Stroma", 'Inflamed CK7hi Epithelial']


Barycentric_coordinate_projection(windows, 
                                      plot_list = plot_list, 
                                      threshold = 10, 
                                      output_dir = output_dir, 
                                      save_name = save_name, 
                                      col_dic = color_dic, 
                                      SMALL_SIZE = 14, 
                                      MEDIUM_SIZE = 16, 
                                      BIGGER_SIZE = 18)

simps, simp_freqs, simp_sums = calculate_neigh_combs(w, 
                                                     l,
                                                     n_num, 
                                                     threshold = 0.85, 
                                                     per_keep_thres = 0.85)

g, tops, e0, e1 = build_graph_CN_comb_map(simp_freqs)

generate_CN_comb_map(graph = g, 
                     tops = tops, 
                     e0 = e0, 
                     e1 = e1, 
                     color_dic = color_dic)

get_network(ttl_per_thres=0.9,
            comb_per_thres=0.005,
            fig_size=(9.5,9),
            neigh_sub = plot_list,
            save_name='All_comm',
            sub_col = cluster_col, 
            color_dic = color_dic)

# Stats
simp_df_tissue1, simp_df_tissue2 = spatial_context_stats(data, n_num, total_per_thres = 0.9, \
                      comb_per_thres = 0.005, \
                      tissue_column = 'Block type',\
                      subset_list = ["Resection"],\
                      plot_order = ['Resection','Biopsy'],\
                      pal_tis = {'Resection':'blue','Biopsy':'orange'},\
                      patient_ID_component1 = ID_component1, \
                      patient_ID_component2 = ID_component2,\
                      subset_list_tissue1 = ["Resection"],\
                      subset_list_tissue2 = ["Biopsy"])
    
print(simp_df_tissue1["combination"].values)  
print(simp_df_tissue2["combination"].values)     
     
simp_df_tissue1 = simp_df_tissue1.set_index("combination")
simp_df_tissue2 = simp_df_tissue2.set_index("combination")
    
spatial_context_stats_vis(neigh_comb = (9,),
                              simp_df_tissue1 = simp_df_tissue1,
                              simp_df_tissue2 = simp_df_tissue2,
                              pal_tis = {'Resection': 'blue', 'Biopsy': 'orange'},
                              plot_order = ['Resection', 'Biopsy'])
    

#######################################################################################################################################################
# CHUNK_5: Analysis independent of neighborhood analysis 
#######################################################################################################################################################

#############################################################
# Cell distance
#############################################################

exclude_list = [] # Specify regions you want to exclude from the analysis 

df[cell_type_column].unique

cell_list = ['CD4+ T cell', 'Macrophage','CD8+ T cell', 'CD8+ T cell PD1+','Tumor PDL1+ MHCI+']
df_s = df.loc[df[cell_type_column].isin(cell_list)]
df_s

df_s[cell_type_column] = df_s[cell_type_column].astype('category')

df_s[cell_type_column] = df_s[cell_type_column].cat.set_categories(cell_list, ordered=True)

#Find only regions where all cell types are present
regions = df_s[region_column].unique()



for r in regions:
    df_sub = df_s[df_s[region_column]==r]
    for cell in df_sub[cell_type_column].unique():
        if len(df_sub.loc[df_sub[cell_type_column]==cell])<1:
            exclude_list.append(r)             
exclude_list = list(set(exclude_list))

#choose regions you want to run the analysis on
df_sub = df_s[~(df_s[region_column].isin(exclude_list))]
df_sub

#this code finds the smaller distance between every cell type of interest
#plots the results by region
#gathers all the data in the arrs object
regions = df_sub[region_column].unique()

arrs = []


for r in regions:
    print(r)
    df_sub = df_sub[df_sub[df_sub]==r]
    cls, dists = get_distances(df_sub_region, cell_list, celltype_column)
    plt.boxplot([
        np.nanmin(dists[(0,3)], axis=1),# here the 0 is for CD68 MACS, 4 is for EPi cells
        np.nanmin(dists[(2,3)], axis=1),# here the 0 is for SPP1 MACS, 4 is for EPi cells
        np.nanmin(dists[(2,3)], axis=1),
        np.nanmin(dists[(1,3)], axis=1),
    ], labels = names[:4])
    plt.xticks(rotation = 90) # Rotates X-Axis Ticks by 45-degrees
    plt.show()
    
    df_all = []
    for i in range(4):
        d = np.nanmin(dists[(i,4)], axis=1)
        df = pd.DataFrame({"dist": d, "type": [names[i]]*d.shape[0]  })
        df_all.append(df)
    df = pd.concat(df_all)
    df[region_column] = r
    arrs.append(df)

#plot all distances from all regions
df_all = pd.concat(arrs)

plt.boxplot([
    df_all[df_all["type"] == name]["dist"] for name in names[:4]
],labels = names[:4])
plt.xticks(rotation = 90)
plt.show()
