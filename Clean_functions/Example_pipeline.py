#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 25 13:39:10 2023

@author: timnoahkempchen
"""

#############################################################
# Example pipeline 
#############################################################
df = pd.read_csv("/Users/timnoahkempchen/Library/CloudStorage/GoogleDrive-timkem@stanford.edu/Meine Ablage/Datasets/data/22_08_09_CellAtlas_only_Processed_Metadata.csv")
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
X = "x"
Y = "y"
treatment_column ="consensus diagnosis"

ks = [20, 30, 35] # k=n means it collects n nearest neighbors for each center cell
cluster_col = "Cell Type"

k = 35
n_neighborhoods = 30
k_centroids = {}

df2 = pd.concat([df,pd.get_dummies(df[cluster_col])],1)
sum_cols = df2[cluster_col].unique()
values = df2[sum_cols].values

cells_df = neighborhood_analysis(df = df2, X = X, Y = Y, reg = region_column, cluster_col = cluster_col, ks = ks, save_path = output_dir, k = k, n_neighborhoods = n_neighborhoods, save_to_csv= True)

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

#############################################################
# Functions from community analysis script 
#############################################################

#######
# Prepare dataframe for neighborhood analysis 
#Import Data
sample_column = "sample"
cell_type_column = "neighborhood35"
region_column = "unique_region"
X = "x"
Y = "y"
treatment_column ="consensus diagnosis"

ks = [100] # k=n means it collects n nearest neighbors for each center cell
cluster_col = "neighborhood35"

k = 100
n_communities = 30
k_centroids = {}

cells_df2 = community_analysis(df = cells_df, X = X, Y = Y, reg = region_column, cluster_col = cluster_col, ks = ks, save_path = output_dir, k = k, n_neighborhoods = 30)


# Prepare annotations as dictionary 
annotations = {
    0: 'Atrophic Cardiac',
    1: 'Specialized',
    2: 'CK7hi and T cell',
    3: 'Inflamed Dysplasia',
    4: 'Atrophic Cardiac',
    5: 'Smooth Muscle',
    6: 'Smooth Muscle',
    7: 'Squamous',
    8: 'Oxnto-Cardiac/Specialized',
    9: 'Inflamed Mature Intestinal',
   
    10: 'Follicle',
    11: 'Specialized',
    12: 'Inflamed Dysplasia',
    13: 'Specialized',
    14: 'CK7hi and T cell',
    15: 'Inflamed Dysplasia',
    16: 'Inflamed Dysplasia',
    17: 'Squamous',
    18: 'Specialized',
    19: 'Inflamed Stroma',
   
    20: 'Smooth Muscle',
    21: 'Inflamed Stroma',
    22: 'Specialized',
    23: 'Atrophic Cardiac',
    24: 'Oxnto-Cardiac/Specialized',
    25: 'Inflamed Stroma',
    26: 'Inflamed Stroma',
    27: 'Follicle',
    28: 'Inflamed Mature Intestinal',
    29: 'Specialized',
}

annotate_communities(df = cells_df2, community_column = "community"+str(k), annotations)

# There is a lot more in Terms of visualization 

#############################################################
# CCA
#############################################################

#pat:treat
cells_df = prepare_neighborhood_df(cells_df, neighborhood_column = 'neighborhood35', patient_ID_component1 = 'sample', patient_ID_component2 = 'Block type') # this is a helper function 


#cells = pd.read_pickle('cells2_salil')
patient_to_group_dict = cells_df.loc[:,['patients',ID_component2]].drop_duplicates().set_index('patients').to_dict()[ID_component2]
group1_patients = [a for a,Sample_type in patient_to_group_dict.items() if Sample_type=="Biopsy"]
group2_patients = [a for a,Sample_type in patient_to_group_dict.items() if Sample_type=='Resection']
group1_patients

# Have a look at cell types 
cells_df['Cell Type'].unique()

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

# Visualization of CCA 
g1 = nx.Graph()
for cn_pair, cc in stats_group1.items():
    s,t = cn_pair
    obs, perms = cc
    p =np.mean(obs>perms)
    if p>0.9 :
        g1.add_edge(s,t, weight = p)
    
    
#pal = sns.color_palette('bright',10)
dash = {True: '-', False: ':'}
pos=nx.drawing.nx_pydot.pydot_layout(g1,prog='neato')
for k,v in pos.items():
    x,y = v
    plt.scatter([x],[y],c = [pal[k]], s = 300,zorder = 3)
    #plt.text(x,y, k, fontsize = 10, zorder = 10,ha = 'center', va = 'center')
    plt.axis('off')


atrs = nx.get_edge_attributes(g1, 'weight')    
for e0,e1 in g1.edges():
    p = atrs[e0,e1]
    plt.plot([pos[e0][0],pos[e1][0]],[pos[e0][1],pos[e1][1]], c= 'black',alpha = 3*p**3,linewidth = 3*p**3)

    
#plt.savefig(save_path+'T_cca_neigh_Tcell.png', format='png', dpi=300, transparent=True, bbox_inches='tight')


# Run CCA for group 2
stats_group2, arr2 = Perform_CCA(cca, n_perms, nsctf, cns, subsets, group = group2_patients)

# Visulize CAA for group 2
g2 = nx.Graph()
for cn_pair, cc in stats_group2.items():
    s,t = cn_pair
    obs, perms = cc
    p =np.mean(obs>perms)
    if p>0.9 :
        g2.add_edge(s,t, weight = p)
    
    
#pal = sns.color_palette('bright',10)
dash = {True: '-', False: ':'}
pos=nx.drawing.nx_pydot.pydot_layout(g2,prog='neato')
for k,v in pos.items():
    x,y = v
    plt.scatter([x],[y],c = [pal[k]], s = 300,zorder = 3)
    #plt.text(x,y+10, k, fontsize = 20, zorder = 10,ha = 'center', va = 'center')
    plt.axis('off')

atrs = nx.get_edge_attributes(g2, 'weight')    
for e0,e1 in g2.edges():
    p = atrs[e0,e1]
    plt.plot([pos[e0][0],pos[e1][0]],[pos[e0][1],pos[e1][1]], c= 'black',alpha = 3*p**3,linewidth = 3*p**3)
    

#plt.savefig("/Users/timnoahkempchen/Downloads/"+'2HC_cca_neigh_Tcell.png', format='png', dpi=300, transparent=True, bbox_inches='tight')

neigh_num

#############################################################
# tensor decomposition 
#############################################################

#pat:treat
cells_df2 = prepare_neighborhood_df(cells_df, neighborhood_column = 'neighborhood35', patient_ID_component1 = 'consensus diagnosis', patient_ID_component2 = 'unique_region') # this is a helper function 

#cells = pd.read_pickle('cells2_salil')
patient_to_group_dict = cells_df2.loc[:,['patients',ID_component2]].drop_duplicates().set_index('patients').to_dict()[ID_component2]
group1_patients = [a for a,Sample_type in patient_to_group_dict.items() if Sample_type=="Biopsy"]
group2_patients = [a for a,Sample_type in patient_to_group_dict.items() if Sample_type=='Resection']
group1_patients

cells_df2['patients'] = cells_df2['patients'].astype('category')
cells_df2['neigh_num'] = cells_df2['neigh_num'].astype('category')
cells_df2['Coarse Cell'] = cells_df2['Coarse Cell'].astype('category')

list(cells_df2['Coarse Cell'].unique())
list(cells_df2['neigh_num'].unique())

# select the cts
cts = list(cells_df2['Coarse Cell'].unique())
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
cns = list(cells_df2['neigh_num'].unique())
#cns = [0, 1, 2, 3, 4, 5, 6]

###

# Build the tensors for each patient group
counts = cells_df2.groupby(['patients','neigh_num','Coarse Cell']).size()

#initialize the tensors

dat1 = build_tensors(df = cells_df2, group = group1_patients, cns = cns, cts = cts)
dat2 = build_tensors(df = cells_df2, group = group2_patients, cns = cns, cts = cts)

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


#############################################################
# Cell distance
#############################################################

celltype_column = 'Cell Type'
region_column
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
        if len(df_sub_region.loc[df_sub_region[cell_type_column]==cell])<1:
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
    df_sub_region = df_sub[df_sub[region_column]==r]
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


########################################################################################################## HuBMAP Spatial contexts

