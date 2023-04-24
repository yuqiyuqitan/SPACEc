#!/usr/bin/env python
# coding: utf-8

# # 1) Basic User Input

# This section imports the required packages and contains user input which shuld be reviewed every time the pipeline is used. 

# In[1]:


import os
import pandas as pd
import sys
get_ipython().run_line_magic('matplotlib', 'inline')


# Usually, this filepath must be adusted only once to specify where the .py file with all functions is stored.

# In[2]:


# load custom functions
sys.path.insert(0, '/Users/timnoahkempchen/Desktop/SAP/Functions_grouped')

from  SAP_helperfunctions_hf import *
from  SAP_preprocessing_pp import *
from  SAP_tools_tl import *
from  SAP_plot_pl import *


# ### Session Info

# In[3]:


import session_info
session_info.show()


# ## 1.1) Filepaths 

# Specify where the input dataframe (df) is stored as well as a path to store all output. 
# 
# The input data frame has to fullfill some minimal requirements: 
# Data must be stored as pandas df (every row represents a cell)
# 1. The df must contain a __Sample ID__ for every cell
# 2. The df must contain one column each for __x__ and __y__ coordinates 
# 3. The df must contain one column specifying the __Cell Type__
# 4. The df must contain one column indicating the __Unique Region__ in which the cell was recorded 
# 5. The df should contain one column with an experimental condition or other type of __comparison__ 

############################
# Preprocessing
############################
path_csv_files = '../22_10_11_ST_56/CVcol_DAPI_3px/fcs/compensated/' 
output_pp_df = '../22_10_11_ST_56/result/df_nn_56_111822.csv'

df_56 = pp_read_data(path= path_csv_files,\
               reg_list=[], nuc_1 = 1)

dfz_56 = pp_format(data=df_56, list_out=['first_index', 'cell_id','tile_num','z', 'x_tile',
                 'y_tile', 'size','DAPI'],
                 list_keep = ['region','x','y','region_num',])

df_cor_56 = pp_xycorr(data=dfz_56, y_rows=2, x_columns=1, X_pix=13000, Y_pix=11000)

#modify figure size aesthetics for each neighborhood
plt.rcParams["legend.markerscale"] = 10
figs = pl_catplot(df_cor_56,X = 'Xcorr',Y='Ycorr',exp = 'array',
               hue = 'region_num',invert_y=True,size = 1,figsize=8)

#what is the differences between z_zum_thres and z_count_thres
#Add all and remove noise
df_nn_56,cc_56 = pp_remove_noise(df=df_cor_56, col_num=48, z_sum_thres=38, z_count_thres=38)

df_nn_56.to_csv(output_pp_df)

# In[6]:


input_file = "/Users/timnoahkempchen/Library/Mobile Documents/com~apple~CloudDocs/Uni/Master/Semester 3/Praktikum Nolan Lab/Python_pipeline_nolan_lab/Datasets/Confidential/22_04_08_ST_CODEX_CellClustered_JH/22_04_08_ST_CODEX_CellClustered_JH.csv"

output_dir = "/Users/timnoahkempchen/Library/Mobile Documents/com~apple~CloudDocs/Uni/Master/Semester 3/Praktikum Nolan Lab/Python_pipeline_nolan_lab/Datasets/Confidential/22_04_08_ST_CODEX_CellClustered_JH/Results/"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)


# ## 1.2) Reading the data

# In[7]:


df = pd.read_csv(input_file)


# In[19]:


df.columns


# In[18]:


df["unique_region"] = df['Sample'] + "_" + df['region_num']


# # 2) Basic Visualization & Analysis 

# ## 2.1) Color dictionary generator

# In[12]:


# provide a list with names colors are mapped on (usually cell types, categories or neighborhoods)
cell_names = df["Cell Type"].unique()
Sample_names = df["Sample"].unique()

# provide a list of colors (same length as names list) or generate a random collection of colors
cell_colors = hf_generate_random_colors(n = len(cell_names))
Sample_colors = hf_generate_random_colors(n = len(Sample_names))


# combine both lists into a dictionary 
color_dic_cells = hf_assign_colors(cell_names, cell_colors)
color_dic_Sample = hf_assign_colors(Sample_names, Sample_colors)


# ## 2.2) Generate visualizations

# **How does the function work:**
# 
# This is a Python function that generates a stacked bar plot of a given dataset. The function takes in several arguments including the data, column names for categorical variables, a grouping variable, a list of cells to plot, a directory to save the plot, and various plot customization options.
# 
# The function starts by checking if the data needs to be normalized based on a boolean parameter "norm". If "norm" is True, then the percentage of each cell type in the dataset is calculated based on the given grouping variable. If "norm" is False, the percentage is calculated across the entire dataset.
# 
# The function then pivots the resulting frequency table to create a tidy dataframe that can be used to generate the stacked bar plot. The function allows for customization of the order of columns and rows in the plot, as well as the color palette used.
# 
# Finally, the function generates the plot using the matplotlib library and saves it to a file if a save_name argument is provided. The function returns the resulting pivoted dataframe and the column order for the plot.
# 
# **What is the biological meaning of this analysis:**
# 
# The analysis shows a stacked bargraph for each specified group. Sections within the graph represent the percentage of cells that is present in this group. The user can choose to group data by different categorial variables (e.g. a clinical condition, a treatment). Percentages can be calculated for broader groups (what is the percentage of immune cells?) or single cell types (what is the percentage of CD4+ T cells).

# In[15]:


# Arguments for stacked bar plot:

data = df # data frame to use 

per_cat = "Cell Type" # column containing the categories that are used to fill the bar plot

grouping = 'Sample' # column containing a grouping variable (usually a condition or cell group)

norm = True

save_name = 'Percentage_cell_type_per_sample' # name for saved file 

sub_col= 'Cell Type'

name_cat = 'Cell Type'

fig_sizing = (8,4) # numeric value to modify the figure size 

color_dic = color_dic_cells # color dictionary 

remove_leg=False # removes legend if needed

cell_list = df["Cell Type"].unique()

col_order = ["Cntrl_d5", "2HC_d3", "2HC_d5", "T_d5"]


# In[17]:


####### Stacked Bar Plot
# Shows percentage of category per group. For example: major cell types per diagnosis 
ab = pl_stacked_bar_plot(data = df, 
                      per_cat = per_cat, 
                      grouping = grouping, 
                      output_dir = output_dir, 
                      sub_col = None, 
                      cell_list = cell_list,
                      norm = False, 
                      fig_sizing = fig_sizing, 
                      name_cat = per_cat, 
                      col_order = col_order, 
                      save_name = save_name, 
                      color_dic = color_dic) 


# **How does the function work:**
# 
# The swarm_box function takes in several parameters and returns a plot object. The purpose of this function is to create a box plot and swarm plot from the given data.
# 
# The function first checks if norm is True. If it is True, it subsets the data and computes the unique values of a given category column to get the percentage of cell type. If norm is False, it copies the entire data. The function then casts the category column to categorical type.
# 
# Next, the function computes the percentage of each category column by group and replicate. It converts column names to string type and resets the index. The function adds grouping and replicate to immune_list and subsets the data. It then melts the data frame and renames columns.
# 
# If col_in is not None, the function subsets melt_per_plot to include only those values. Otherwise, it does nothing. The function then orders the data by the average percentage of each category column.
# 
# If h_order is None, the function uses the unique values of the grouping column as the order. If color_dic is None, the function creates a figure with box plot and swarm plot for each category column or grouping column based on flip value.
# 
# If flip is True, the function creates a box plot with given parameters and a swarm plot with given parameters for grouping column. The function sets the transparency of box plot patches. If save_name is not None, the function saves the figure as a PNG file with given parameters.
# 
# In summary, the swarm_box function creates a box plot and swarm plot from the given data and returns a plot object. It has several parameters that allow for customization of the plot.
# 
# **What is the biological meaning of this analysis:**

# In[27]:


print(df["unique_region"].unique())


# In[20]:


# Arguments for swarm boxplot:

data = df # data frame to use 

grouping = 'Sample' # column containing a grouping variable (usually a condition or cell group)

sub_col = 'Cell Type'

sub_list = ['Treg']

replicate_column = "unique_region"

output_dir = output_dir

norm = False

per_cat = "Cell Type"

fig_sizing=(8,4) # numeric value to modify the figure size 

save_name = 'Boxplot_cell_type_per_sample'

color_dic = color_dic_Sample # color dictionary 

plot_order=None

flip=True


# In[23]:


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


# **How does the function work:**
# 
# This Python function calculates the Shannon Diversity of cell types within a sample and performs ANOVA analysis on the results. The function takes in several parameters, including the input data (data1), a list of subgroups within the data (sub_l), a grouping variable (group_com), a category variable (per_categ), a replicate variable (rep), a sub-column variable (sub_column), and a coloring variable (coloring). Other parameters include an output directory (output_dir), a normalization option (normalize), a save option (save), an ordering variable (ordering), and a figure size variable (fig_size).
# 
# First, the function calculates the Shannon Diversity using the per_only1() function on the input data and the other parameters passed to the function. It then aggregates the results by replicate and grouping variable and calculates the Shannon Diversity. The results are saved in a Pandas DataFrame (res) with the Shannon Diversity column multiplied by -1.
# 
# Next, the function performs ANOVA analysis on the results using the f_oneway() function from the scipy.stats module. The results are saved in the test_results variable. If the test results are significant (p-value < 0.05), the function performs pairwise Tukey's HSD post-hoc analysis using the pairwise_tukeyhsd() function from the statsmodels.stats.multicomp module. The results are saved in a Pandas DataFrame (table1) and plotted using a heatmap with the sns.heatmap() function from the seaborn module.
# 
# Finally, the function plots the results using a swarmplot and boxplot with the sns.swarmplot() and sns.boxplot() functions from the seaborn module. The plot is customized with various parameters, including the coloring variable, ordering variable, and fig_size variable. If the save option is enabled, the plot is saved in the output directory. The function returns three variables: the Shannon Diversity data (tt), the ANOVA test results (test_results), and the Tukey's HSD results (table1).
# 
# **Further information on per_only1 function:** 
# 
# The per_only1 function is a helper function that takes in a pandas DataFrame data, and several other arguments including grouping, replicate, sub_col, sub_list, per_cat, and norm.
# 
# The function first filters the DataFrame data based on the values in sub_col column that are in sub_list, and then calculates the percentage of each unique value in the per_cat column for the filtered DataFrame.
# 
# If norm is True, the function normalizes the percentage values by dividing them by the total count of each unique value in the per_cat column. Otherwise, it calculates the percentage without normalization.
# 
# The resulting DataFrame contains the grouping column, replicate column, per_cat column, and a column specifying the percentage. The grouping and replicate columns are specified by the arguments passed to the function, and the per_cat column is the column containing the cell types or categories of interest.
# 
# The melt_per_plot DataFrame returned by the function can be used for plotting or further analysis. An example of the output would be the percentage of CD4+ T cells in a unique region E08 assigned to community xxx.
# 
# **What is the biological meaning of this analysis:**
# 
# Estimates the diversity within a given community. The figure shows the Shannon Diversity Index for the specified groups as boxplot showing the replicates as individual datapoints. Typical replicates are the unique regions within the respective group. 
# 
# A higher score indicates a higher degree of diversity within the analyzed group. CAVE: rare cell types might be overrepresented in statistical evaluation. 

# In[ ]:


# Arguments for Shannon diversity plot:

data = df # input data frame 

sub_list = ['CD4+ Treg', "B"]

grouping = "Sample" # column containing a grouping variable (usually a condition or cell group)

per_categ = "Cell Type"

rep = "unique_region" # replicate column (usually the unique regions per sample)

sub_column = 'Cell Type'

normalize = False

save = True # logical value to decide if the generated plot should be saved 

color_dic = color_dic_Sample # color dictionary 

fig_size = 8 # numeric value to modify the figure size 

plot_order = None # optional list to specify the plotting order

output_dir = output_dir # directory to save output


# In[ ]:
####### Swarm Boxplot of Shannon diversity score
tt, test_results, res  = tl_Shan_div(data = data, 
                                     sub_l = sub_l, 
                                     group_com = group_com, 
                                     per_categ = per_categ, 
                                     rep = rep, 
                                     sub_column = sub_column, 
                                     normalize = normalize) 

pl_Shan_div(tt, 
            test_results, 
            res, 
            grouping = grouping, 
            color_dic = color_dic, 
            sub_list = sub_l, 
            output_dir = output_dir, 
            save=False, 
            plot_order=None, 
            fig_size=1.5)








#result, pval, tukey_tab = Shan_div(data = data, \
 #                                  sub_l = sub_l, \
#                                   group_com = group_com, \
#                                   per_categ = per_categ, \
#                                   rep = rep, \
#                                   sub_column=sub_column, \
#                                   normalize=normalize, \
#                                   save=save, \
#                                   coloring= coloring, \
#                                   fig_size=fig_size, \
#                                   ordering=ordering, \
 #                                  output_dir = output_dir)


# **How does the function work:**
# 
# **corr_cell**
# 
# This function performs a correlation analysis on a dataset and plots the results. The function takes several arguments:
# 
# - data: the input data for the analysis
# - sub_l2: a list of subcategories for grouping the data
# - per_categ: the category for calculating the percentage
# - group2: a grouping variable for the analysis
# - repl: a variable for replicates
# - sub_column: the name of the column that contains subcategory data
# - cell: the cell type for the analysis
# - output_dir: the directory where the output plot will be saved
# - save_name: the name for the output plot
# - thres: the threshold for correlation analysis
# - normed: whether to normalize the data (default is True)
# - cell2: an optional second cell type for the analysis
# 
# The function first calls the per_only1 function on the input data to calculate the percentage based on the specified categories, subcategories, and replicates. It then creates a pivot table from the resulting data, calculates the correlation matrix, and subsets the matrix based on the specified threshold and cell type.
# 
# The function then plots the correlation matrix using the cor_subplot function and saves the plot to the specified directory with the specified name. Finally, the function returns two sets of pairs: all pairs and pairs for the specified cell type.
# 
# **cor_plot**
# 
# The cor_plot function takes in a Pandas DataFrame data, and several parameters to create a heatmap and scatter plot of the correlation matrix between the columns in the DataFrame. The function can take in two grouping variables group1 and group2 and creates a heatmap of the correlation matrix between the percentage of each variable in per_cat. The sub_col parameter specifies the column to filter the data by the values in sub_list. If norm is True, the percentage values are normalized by the sum of the values in the column sub_col. If count is True, the count of each variable is used instead of the percentage. If plot_scatter is True, a scatter plot matrix is created with a regression line for each pair of variables.
# 
# If group2 is not specified, the correlation matrix is created between the percentage of each variable in per_cat for each unique value in group1. If count is True, the count of each variable is used instead of the percentage.
# 
# The function returns the correlation matrix cor_mat and the DataFrame used to create the heatmap and scatter plot mp.
# 
# **cor_subplot**
# 
# The function cor_subplot takes three required arguments mp, sub_list, and output_dir and one optional argument save_name.
# 
# The mp argument is a pandas DataFrame which contains data to be plotted in a correlation subplot. sub_list is a list of column names of the mp DataFrame for which the correlation matrix will be plotted. output_dir is a string which specifies the output directory where the figure will be saved.
# 
# The function first selects the columns specified in the sub_list argument from the mp DataFrame and creates a pairwise scatterplot matrix using Seaborn's pairplot function with kernel density estimates on the diagonal and linear regression fits on the lower triangle. The corner=True argument sets the diagonal axes to be drawn only once.
# 
# If save_name is not None, the function saves the figure to a file in the specified output_dir with the name save_name+'_corrplot.png'. The saved figure is in PNG format with a DPI of 300 and transparent background. The bbox_inches='tight' argument adjusts the figure size to remove any whitespace padding.
#     
# **What is the biological meaning of this analysis:**
#     

# In[ ]:


# Arguments correlation analysis: 

# corr_cell
data = df # a pandas DataFrame object that contains the data to be analyzed

sub_list = None # a list of subject IDs to be included in the analysis

per_categ = "Cell Type" # a string that represents the category column for the percentages in the data frame

rep = "unique_region" # a string that represents the replication column in the data frame

sub_column = "Cell Type" # a string that represents the subject ID column in the data frame

normed = True

save = True

coloring = None

fig_size = 8

thres = 0.7

ordering = None

output_dir = output_dir

cell = "CD4+ T cell" # a string that represents the cell type to be analyzed
cell2 =  'CD8+ T cell PD1+'

sub_column = "Cell Type"

group2 = 'Sample'

save_name = "cell1_cell2_cor"

# cor_plot
group1 = 'Cell Type'

per_cat = 'Cell Type'

sub_col = 'Cell Type'

sub_list = None

norm = True

count = False

plot_scatter = False

# Prepare subplot 
cell_type = 'CD169+ Macrophage'


# In[ ]:


####### Correlation Analysis 

all_pair, sub_pair = pl_corr_cell(data = data, 
                               sub_list2 = sub_list, 
                               per_categ = per_categ, 
                               group2 = group2, 
                               rep = rep, 
                               sub_column = sub_column, 
                               cell = cell, 
                               normed = normed, 
                               thres = thres, 
                               cell2 = cell2,
                               output_dir = output_dir, 
                               save_name = save_name)

cor_mat, mp = pl_cor_plot(data = data, \
                       group1 = group1, \
                       per_cat = per_cat, \
                       sub_col= sub_col, \
                       sub_list= sub_list, \
                       norm=True, \
                       count=False, \
                       plot_scatter=False)


piar1 = all_pair.loc[all_pair['col1']==cell_type]
piar2 = all_pair.loc[all_pair['col2']==cell_type]
piar=pd.concat([piar1,piar2])
piar

pair_list = list(set(list(piar['col1'].unique())+list(piar['col2'].unique())))
pair_list

sl = pair_list

pl_cor_subplot(mp=mp, \
            sub_list=sl, \
            output_dir = output_dir)


# # 3) Neighborhood and Community Analysis 

# ## 3.1) Neighborhood analysis 

# **How does the function work:**
# 
# I apologize for the confusion. You are correct, the third bar plot also shows stacked bars normalized to 100 percent.
# 
# Here's an updated description of the function:
# 
# This Python function is designed to visualize the composition of different cell types in a given dataset. The input to the function is a pandas dataframe (df) containing information about the samples and their cell types. The sample_column argument specifies the name of the column in the dataframe that contains sample identifiers, and cell_type_column specifies the name of the column containing the cell type information.
# 
# If an output_dir is specified, the function generates three different bar plots of the cell type composition. The first plot shows the absolute count of each cell type for each sample, where the bars are stacked. The second plot shows the percentage of each cell type in each sample. The third plot shows the absolute count of each cell type for each sample, where the bars are stacked and normalized to 100 percent.
# 
# All three plots are saved as PNG files in the specified output_dir.
# 
# If no output_dir is specified, the function prints a message saying that no output directory has been defined.
#     
# **What is the biological meaning of this analysis:**
# 
# These visualizations are ment to provide a general overview of the global composition of each individual sample. The function provides three different variations of bar graphs.
# 1. Stacked bar graph showing the absolut count of the category 
# 2. Bar graph that shows the counts per category side by side 
# 3. Stacked bar graph showing the percentage of each category per sample
#     

# In[28]:


# Arguments for composition visualization function:

data = df # data frame to use

sample_column = "Sample" # column that specifies the sample

cell_type_column = 'Cell Type' # column that specifies cell types - other variables will be accepted as well

output_dir = output_dir # directory to save output 


# In[29]:


####### Visulize overall cell type composition
pl_cell_type_composition_vis(data = df, \
                          sample_column = sample_column, \
                          cell_type_column = cell_type_column, \
                          output_dir = output_dir)


# **How does the function work:**
#     
# **What is the biological meaning of this analysis:**
# 
# Allocate every cell to a broader group based on its spatial context. Cellular Neighborhoods (CN) can be understood as common groups of cells which are shared by multiple regions (e.g. the "generic tumor neighborhood" is found in every tumor of dataset xxx). This code assings the neighborhoods, plots a spatial plot colored by neighborhoods and generates a heatmap that illustrates which celltype is found in the generated neighborhoods. 
# 
# **NOTE: This function modifies the data frame**
# 
# The function adds a column called neighborhoods_k (k is replaced with the value used for k). This column contains the ID of the assigned neighborhood for each cell. After executing the function, every neighborhood is represented by a number (starting with 0). In order to receive biological meaningful names, neigborhoods need to be annotated based on marker expression with the help of an expert or automated annotation approach. 

# In[30]:
########
'''
k is used to compute the neighborhoods of each cell by selecting the k nearest neighbors based on the Euclidean distance between the cells in the feature space. 
Once the neighborhoods have been computed, the n_neighborhoods parameter is used to cluster the neighborhoods using the MiniBatchKMeans algorithm. 
The result is that each neighborhood is assigned to one of n_neighborhoods clusters, allowing the identification of distinct cell types or niches within the tissue.

k reffers to the window size while n_neighborhoods reffers to the number of generated neighborhoods
'''

# Arguments for neighborhood analysis:

X = "x" # column containing the x coordinates 

Y = "y" # column containing the y coordinates 

reg = "unique_region" # column containg the unique regions 

cluster_col = "Cell Type" # column which is used for clustering - typically cell types (to generate cellular neighborhoods)

# ks = [20] # k=n means it collects n nearest neighbors for each center cell

output_dir = output_dir

k = 20

# set a single value to generate neighborhoods with this value. If you want to generate the elbow plot supply the list with a range of ks
n_neighborhoods = [5,7,8,9,10,11,12,13,14, 15,17, 20, 22, 25, 30] # number of generated neighborhoods - value should be biologically meaningful 
n_neighborhoods = 15

save_to_csv= True

plot_specific_neighborhoods = [2,4]


# In[31]:


####### Neighborhood analysis 
df2 = pd.concat([df,pd.get_dummies(df[cluster_col])],1)
sum_cols = df2[cluster_col].unique()
values = df2[sum_cols].values

k_centroids = {}

# Previous version of function will be removed in future update 
# Maybe implement a function to test for the optimal neighborhood count

#cells_df = neighborhood_analysis(data = df2, \
  #                               X = X, \
  #                               Y = Y, \
 #                                reg = reg, \
 #                                cluster_col = cluster_col, \
#                                 ks = ks, \
 #                                output_dir = output_dir, \
 #                                k = k, \
  #                               n_neighborhoods = n_neighborhoods, \
 #                                save_to_csv= True, \
  #                               plot_specific_neighborhoods = [2,4], \
  #                               values = values, \
 #                                sum_cols = sum_cols)



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
                           plot_specific_neighborhoods = None)



# I identified redundant elements and removed them 
#cells_df = neighborhood_analysis_2(data = df2, 
#                                 X = X, 
#                                 Y = Y, 
#                                 reg = reg, 
#                                 cluster_col = cluster_col, 
#                                 output_dir = output_dir, 
#                                 k = k, 
#                                 n_neighborhoods = n_neighborhoods, 
#                                 save_to_csv= True, 
#                                 plot_specific_neighborhoods = [2,4], 
#                                 values = values, 
#                                 sum_cols = sum_cols,
#                                 calc_silhouette_score = False)


# ## 3.2) Community Analysis 

# **How does the function work:**
#     
# **What is the biological meaning of this analysis:**
# 
# This analysis is very similar to the neighborhood analysis but classifies even broader groups. Instead of cell types neighborhoods are used for
#     
# **NOTE: This function modifies the data frame**
# 
# The function adds a column called communities_k (k is replaced with the value used for k). This column contains the ID of the assigned community for each cell. After executing the function, every community is represented by a number (starting with 0). In order to receive biological meaningful names, communities need to be annotated based on incorporated neighborhoods with the help of an expert or automated annotation approach. 

# In[32]:


# Arguments for community analysis:

data = cells_df

X = X

Y = Y

reg = "unique_region"

cluster_col_commun = "neighborhood20"

# ks_commun = [10] # k=n means it collects n nearest neighbors for each center cell

output_dir = output_dir

k_commun = 100

n_communities_commun = 12

plot_specific_community = [2,4,5]

values = values

sum_cols = sum_cols


# In[33]:


k_centroids = {}

#cells_df2 = community_analysis(data = cells_df, \
#                               X = X, \
#                               Y = Y, \
#                               reg = reg, \
#                               cluster_col = cluster_col_commun, \
 #                              ks = ks_commun, \
 #                              output_dir = output_dir, \
 #                              k = k_commun, \
#                               n_neighborhoods = n_communities_commun, \
#                               plot_specific_community = [2,4,5], \
#                               values = values, \
 #                              sum_cols = sum_cols)





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
                        plot_specific_community = None)








#cells_df2 = community_analysis_2(data = cells_df, \
#                               X = X, 
#                               Y = Y, 
#                               reg = reg, 
#                               cluster_col = cluster_col_commun, 
#                               output_dir = output_dir, 
#                               k = k_commun, 
#                               n_neighborhoods = n_communities_commun, 
#                               plot_specific_community = [2,4,5], 
#                               values = values, 
#                               sum_cols = sum_cols)





# Save progress 

cells_df2.to_csv("/Users/timnoahkempchen/Library/Mobile Documents/com~apple~CloudDocs/Uni/Master/Semester 3/Praktikum Nolan Lab/Python_pipeline_nolan_lab/Datasets/Confidential/22_04_08_ST_CODEX_CellClustered_JH/Results/preprocessed_df.csv") # save result for later

# # 4) Analysis dependent on neighborhood analysis 

# ## 4.1) Specify additional column names 

# ## 4.2) Cell Type Differential Enrichment 

# **How does the function work:**
# 
# **hf_cell_types_de_helper**
# 
# The hf_cell_types_de_helper function takes a pandas DataFrame (df), as well as several column names and dictionaries as inputs, and performs various data transformation operations on the input DataFrame to generate several outputs.
# 
# The function first reads in the input DataFrame df and generates a unique ID by concatenating two columns specified by the ID_component1 and ID_component2 inputs. It then creates a dictionary called neigh_num that maps each unique value in the neighborhood_col column of the input DataFrame to a unique integer index starting from 0. The function then creates a new column called group by mapping each value in the group_col column to its corresponding integer value from a dictionary called group_dict.
# 
# The function then creates a new dictionary called pat_dict that maps each unique value in the donor_tis column of the input DataFrame to a unique integer index starting from 0. The function assigns these integer indices to each patient in the donor_tis column, creating a new column called patients. The function then drops duplicates from the patients and group columns and creates a new dictionary called pat_to_gp that maps each patient to their corresponding group.
# 
# The function then groups the input DataFrame by patients and calculates the frequency of each value in the cell_type_col column for each patient, normalized by the total number of cells for that patient. The function stores the resulting DataFrame as ct_freq.
# 
# Finally, the function groups the input DataFrame by both patients and neigh_num, and calculates the frequency of each value in the cell_type_col column for each neighborhood and patient, normalized by the total number of cells for that patient and neighborhood. The function stores the resulting DataFrame as all_freqs.
# 
# The function returns several outputs, including the transformed input DataFrame (cells2), the ct_freq and all_freqs DataFrames, as well as the pat_to_gp and neigh_num dictionaries.
# 
# **cell_types_de**
# This function performs differential enrichment analysis of cell types between different neighborhoods in a tissue.
# 
# 
# The function takes in several inputs, including cell type frequencies, patient data, neighborhood data, and an output directory. The function first normalizes overall cell type frequencies and neighborhood-specific cell type frequencies, and then calculates differential enrichment for all cell subsets using linear regression. It creates a heatmap to visualize the changes in cell type frequencies across different neighborhoods, with asterisks indicating statistical significance (p < 0.05).
# 
# After correcting p-values for multiple testing, it generates a second heatmap that is sorted by the sum of absolute values of coefficients in each row and column. Finally, it saves the heatmap plot as a PNG file in the specified output directory and returns the sum of absolute values of coefficients in the sorted dataframe.
# 
# - ct_freq: a pandas DataFrame with cell type frequencies for all samples
# - all_freqs: a pandas DataFrame with cell type frequencies and neighborhood information for all samples
# - neighborhood_num: the name of the column in all_freqs that contains the neighborhood information
# - nbs: a list of integers representing the neighborhoods to analyze
# - patients: a list of patient IDs to include in the analysis
# - group: a pandas Series with group information (0 or 1) for each sample
# - cells: a list of cell type names to analyze
# cells1: a list of cell type names with neighborhood-specific information to analyze
# neigh_num: a dictionary mapping neighborhood numbers to names
# output_dir: a string representing the output directory for the generated plots
# The function then performs several steps:
# 
# Normalize the overall cell type frequencies and the neighborhood-specific cell type frequencies for the specified patient IDs using the normalize function.
# For each neighborhood in nbs, concatenate the normalized neighborhood-specific cell type frequencies for the specified patient IDs into a design matrix with a constant, group 0 or 1, and the normalized overall cell type frequencies. Then, fit a linear regression model to the neighborhood-specific cell type frequencies, where the neighborhood-specific cell type frequencies are the outcome and the design matrix is the predictor.
# Store the p-values and coefficients for the group coefficient in each linear regression model in a dictionary.
# Correct the p-values for multiple testing (this step is currently missing in the code).
# Create a heatmap of the coefficients for each cell type and neighborhood, using the sns.heatmap function.
# Sort the rows and columns of the heatmap by the absolute sum of the coefficients across neighborhoods and cell types.
# Add asterisks to the heatmap for any cell type and neighborhood combination where the p-value is less than 0.05.
# Save the heatmap as a PNG file in the specified output directory.
# Return the sum of the absolute values of the coefficients in the sorted heatmap.
# 
# 
# 
#     
# **What is the biological meaning of this analysis:**
# Identifies cell populations that statistical significantly differ between CNs
#     

# In[36]:


# Arguments for cell type differential enrichment analysis:

ID_component1 = 'Sample'

ID_component2 = 'region_num'

neighborhood_col = 'neighborhood20'

group_col = 'Sample'

group_dict = {'Cntrl_d5':0, '2HC_d3':1, "2HC_d5":2, "T_d5":3}

cell_type_col = 'Cell Type'

neighborhood_col_number = 'neigh_num'


# In[37]:


cells2, ct_freq, all_freqs, pat_to_gp, neigh_num = hf_cell_types_de_helper(df = cells_df, \
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





dat, pvals = tl_cell_types_de(ct_freq = ct_freq, 
                              all_freqs = all_freqs, 
                              neighborhood_num = neighborhood_col_number, 
                              nbs = nbs, 
                              patients = patients, 
                              group = group, 
                              cells = cells, 
                              cells1 = cells1)

pl_cell_types_de(data = dat,
                 pvals = pvals, 
                 neigh_num = neigh_num, 
                 output_dir = output_dir)









#cell_types_de(ct_freq = ct_freq, \
#              all_freqs = all_freqs, \
#              neighborhood_num = neighborhood_col_number, \
#              nbs = nbs, \
#              patients = patients, \
#              group = group, \
#              cells = cells, \
#              cells1 = cells1, \
#              neigh_num = neigh_num, \
#              output_dir = output_dir)


# ## 4.3) Canonical Correlation Analysis (CCA) 

# **How does the function work:**
#     
# **What is the biological meaning of this analysis:**
# 
# Neighborhoods/Communities influence each other. This communication can be described as correlation. Correlation can either be analyzed on the single cell level (occurence of cell type A in neighborhood 1 is negative/positive correlated with its occurence in neighborhood 2 - this can then also be compared between conditions e.g good/bad prognosis for cancer type xxx -> alteratiions in CN communication might be implecated in better/worse antitumoral immune responses etc...) or on the level of multiple celltypes.
# Canonical Correlation Analysis (CCA) looks at the frequency of multiple cell types (or neighborhoods). The idea is to build inter-CN communication networks. These networks can be visualized as graphs. 
#     

# In[38]:


# Arguments for CCA:

ID_component1 = 'region_num'

ID_component2 = 'Sample'

neighborhood_col = 'neighborhood20'


# In[39]:


# Prepare IDs this could for example be the combination of patient ID and tissue type. Apart from that, the function assigns a number to each name from the neighborhood column
cells_df = hf_prepare_neighborhood_df(cells_df, 
                                   neighborhood_column = neighborhood_col, 
                                   patient_ID_component1 = ID_component1, 
                                   patient_ID_component2 = ID_component2) # this is a helper function 


# In[67]:


# devide IDs/patients into groups
patient_to_group_dict = cells_df.loc[:,['patients',ID_component2]].drop_duplicates().set_index('patients').to_dict()[ID_component2]
group1_patients = [a for a,Sample in patient_to_group_dict.items() if Sample=="2HC_d5"]
#group2_patients = [a for a,Sample_type in patient_to_group_dict.items() if Sample_type=='Resection']


# In[68]:





# In[69]:


n_perms = 5000 # set number of permutation params

#subsets = ['CD4+ T cell']

subsets = None


# In[70]:


# Provide user feedback
print(group1_patients)

# select which neighborhoods and functional subsets
cns = list(cells_df['neigh_num'].unique())
print(cns)

#log (1e-3 +  neighborhood specific cell type frequency) of functional subsets) ('nsctf')
if subsets != None:
    nsctf = np.log(1e-3 + cells_df.groupby(['patients','neigh_num'])[subsets].mean().reset_index().set_index(['neigh_num','patients']))
    print(nsctf)
else:
    nsctf = np.log(1e-3 + cells_df.groupby(['patients','neigh_num']).mean().reset_index().set_index(['neigh_num','patients']))
    print(nsctf)

cca = CCA(n_components=1,max_iter = 5000)
func = pearsonr


# In[75]:


# Run CCA
stats_group = tl_Perform_CCA(cca = cca, 
                          n_perms = n_perms, 
                          nsctf = nsctf, 
                          cns = cns, 
                          subsets = subsets, 
                          group = group1_patients)


# In[72]:


# OPTIONAL

# print the name/number key for every neighborhood 
# Display DataFrames

# Using Zip method
ans = zip(cells_df.neighborhood20,cells_df.neigh_num)

# Converting it into list
ans = list(ans)

# Converting it into dictionary
ans = dict(ans)

# Display ans
print("Result of apply:\n",ans,"\n")


# In[73]:


# OPTIONAL

# This function provides a random color palette - If you want a specific color for your neighborhoods, just provide a list of color
# provide a list with names colors are mapped on (usually cell types, categories or neighborhoods)
neighb_names = cells_df['neigh_num'].unique()

# provide a list of colors (same length as names list) or generate a random collection of colors
neighb_colors = hf_generate_random_colors(n = len(neighb_names))


# combine both lists into a dictionary 
color_dic_neighb = hf_assign_colors(neighb_names, neighb_colors)


# In[77]:


# Visulize CCA
pl_Visulize_CCA_results(CCA_results = stats_group, 
                     save_fig = False, 
                     output_dir = output_dir, 
                     save_name = "CCA_vis.png", 
                     p_thresh = 0.05, 
                     colors = color_dic_neighb)


# ## 4.4) Tensor decomposition

# **How does the function work:**
# Description from original paper: 
# The tensor of CN-cell type distributions for each patient, with dimensions patients x cell types x CNs, was produced by computing the
# frequency of each cell type in each CN in the non-follicular compartments (i.e., all CNs except CN-5). This tensor was split along
# the patient direction by patient group (CLR and DII). Non-negative Tucker decomposition as implemented in the Tensorly Python package was applied to each tensor (Kossaifi et al., 2019). The ranks in each dimension (2,6,6) were selected by a visual elbow point
# method assessing the decomposition loss (Figure S6C). Several random-starts were utilized to ensure stability.
# The cell type modules correspond to the factors in cell-type space. The CN modules correspond to the factors in CN space. The
# interactions comprising a tissue module correspond to each 6x6 slice of the 2x6x6 core tensor.
#     
# **What is the biological meaning of this analysis:**
# From the original publication: 
# 
# We motivate our use of tensor methods for describing differences in the variation across patients’ joint CNCT
# compositions by discussing the limitations of traditional PCA for this purpose. One possibility for describing
# the differences, between patient groups, in variation across patients’ joint CN-CT compositions, would have been
# to first perform PCA (by flattening each patient’s 2D matrix to a 1D vector), and subsequently describe how the
# identified axes were different. However, this would have eliminated the information that CNs and CTs form two
# distinct but coupled views of the iTME. This coupling corresponds exactly to the fact that the underlying biological
# programs drive multiple distinct CTs to be found together in multiple distinct CNs. For example, multiple CTs
# might share combinations of cytokine receptors, and cytokine gradients might promote combinations of CNs.
# An example which illustrates how underlying biology could give rise to the tensor decomposition output is
# depicted as a schematic in the Figure below: (1) The tissue is formed by the interaction of CN ‘recruitment factors’
# (for example, cytokines) shared by multiple CNs to recruit cell types by interacting with cognate ‘cellular
# localization factors’ (for example, cytokine receptors) shared by multiple cell types (Panel 1, top aspect). The term
# factor should be viewed in a statistical sense and could represent more complicated programs than a single ligand
# or receptor. Different factors can interact to different extents (Panel 1, lower aspect). (2) Different interacting pairs
# of recruitment and localization factors are found together in the tissue, giving rise to the observed distribution of
# CNs and cell types (Panel 2). In the left region, the blue and red CNs share a recruitment factor (heart-shaped
# indentation), so share a common cell type (green) with a cognate localization factor (heart). In the right region, the
# orange and the gray cells share a localization factor (circle), so are found in multiple CNs. The green CN uses
# multiple recruitment factors, one shared with the yellow CN. Distinct interacting pairs of recruitment and
# localization factors co-occur across patients (red and blue found together, and yellow and green found together),
# each co-occurring collection of interacting pairs corresponding to a tissue module. These recruitment and
# localization factors are inferred from the tensor decomposition output, visualized as tissue modules comprised of
# CN modules and cell type (CT) modules, with interactions between them represented as edges (Panel 3). Note that
# there is a common collection of CT modules and CN modules that are present to different extents in each tissue
# module. The contribution of each CN module and CT module to each tissue module is represented by its shading
# (Panel 3). In tissue module 1 (top box), the CN module in the first row is interpreted as the recruitment factor with
# a circular indentation. This is because it contains yellow and green CNs, and there is a strong edge with the CT
# module containing the orange and grey cell types, and a weak edge with the CT module containing the blue cell
# type. The CN module with just the green CN (row 2) is interpreted as the recruitment factor with the square
# indentation. This is because that CN module does not contain any other CNs and has only one edge with one CT
# module containing the blue cell type. Since the red and green CNs are not found in the same patients, the CN module
# with the red and blue CNs and its cognate CT module with just the green cell type are faint in tissue module 1 and
# form tissue module 2. Note that the CN modules and the cell type modules are identified by their mutual
# dependence.
# 
# ![image.png](attachment:image.png)
# 
# Schematic illustrating the interpretation of the tensor decomposition output. (1) Legend of components: A CN
# module corresponds to a cell recruitment program utilized by the CNs comprising that module, and a CT module
# corresponds to a cell type localization program utilized by the cell types comprising that module. Different pairs of
# recruitment programs and localization programs interact to different strengths. (2) Different pairs of interacting
# recruitment programs and localization programs co-occur to form the tissue through balanced interactions between
# recruitment and localization factors. These combinations yield similar combinations of CNs and cell types within
# them across patients. (3) Graphical representation of tissue modules corresponding to combinations of interacting
# pairs, indicated by edges, of CN modules (left column) and CT modules (right column). CN modules and CT
# modules are common across both tissue modules. In each tissue module, the transparency of each CN module and
# CT module corresponds to the weight of the maximum edge of which it is part, i.e. indicating its contribution to
# that tissue module.

# In[ ]:


# Prepare IDs this could for example be the combination of patient ID and tissue type. Apart from that, the function assigns a number to each name from the neighborhood column
cells_df = hf_prepare_neighborhood_df(cells_df, 
                                   neighborhood_column = neighborhood_col, 
                                   patient_ID_component1 = ID_component1, 
                                   patient_ID_component2 = ID_component2) # this is a helper function 


# In[ ]:


# devide IDs/patients into groups
patient_to_group_dict = cells_df.loc[:,['patients',ID_component2]].drop_duplicates().set_index('patients').to_dict()[ID_component2]
group1_patients = [a for a,Sample_type in patient_to_group_dict.items() if Sample_type == 'Cntrl_d5']
# group2_patients = [a for a,Sample_type in patient_to_group_dict.items() if Sample_type=='Resection']

# Provide user feedback
print(group1_patients)
group1_patients = ['reg002_2HC_d5']

list(cells_df["Cell Type"].unique())
list(cells_df['neighborhood20'].unique())

# select the cts
cts = list(cells_df["Cell Type"].unique()) # In theory you could select all cell types 
#cts =['Macrophage CD169+',   # It is very likely that a user wants to select specific cell types 
# 'CD4+ T cell',
# 'DC',
# 'Stromal',
#  'Tumor Ki67+',
#  'Tumor PDL1+ MHCI+',
#  'Tumor',
# 'Macrophage',
# 'Neutrophil',
# 'NK',
# 'CD8+ T cell PD1+',
# 'CD8+ T cell',
# 'CD4+ Treg',
# 'B cell']



# select the cns
cns = list(cells_df['neigh_num'].unique())
#cns = [0, 1, 2, 3, 4, 5, 6]


# In[ ]:


# Build the tensors for each patient group
counts = cells_df.groupby(['patients','neigh_num','Cell Type']).size()

#initialize the tensors

dat1 = tl_build_tensors(df = cells_df, group = group1_patients, cns = cns, cts = cts, counts = counts)
#dat2 = build_tensors(df = cells_df, group = group2_patients, cns = cns, cts = cts)


# In[ ]:


# The following tries different numbers of CN modules/CT modules to determine suitable rank for decomposition

pl_evaluate_ranks(dat1,2)
plt.show()
#evaluate_ranks(dat2,2)
#plt.show()

pl_plot_modules_heatmap(dat1, cns, cts)
#plot_modules_heatmap(dat2, cns, cts)

# Set a save path MOVE THIS TO TOP OF SCIPT COMBINE WITH OUTPUT 
save_path = '/Users/timnoahkempchen/Downloads/'

color_dic = sns.color_palette('bright',30) # Choose some random colors to demonstrate that function in working 
pl_plot_modules_graphical(data = dat1, color_dic = color_dic, cns = cns, cts = cts, save_name = 'T cell', save_path = output_dir, scale = 0.4)


# # 5) Analysis dependent on community analysis 

# ## 5.1) Specify additional information

# In[ ]:


data = cells_df2

col_list = data.columns

# Spatial context 
n_num = 75
ks=[n_num]
cluster_col = 'community100'
sum_cols=cells_df2[cluster_col].unique()
keep_cols = col_list
X='x'
Y='y'
Reg = 'unique_region'
Neigh = Neighborhoods(cells_df2,ks,cluster_col,sum_cols,keep_cols,X,Y,reg=Reg,add_dummies=True)
windows = Neigh.k_windows()
reg = "unique_region"

#Choose the windows size to continue with
w = windows[n_num]

n_neighborhoods=7
n2_name = 'neigh_ofneigh'
k_centroids = {}

km = MiniBatchKMeans(n_clusters = n_neighborhoods,random_state=0)
labels = km.fit_predict(w[sum_cols].values)
k_centroids[n_num] = km.cluster_centers_
w[n2_name] = labels


# ## Spatial context analysis 

# **How does the function work:**
#     
# **What is the biological meaning of this analysis:**
# 
# The idea is to map where possible interactions might occur. The basic assumption is that local processes of CNs interact than two or more CNs contact. 
#     

# In[ ]:


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



plot_list = list_n = [ 0, 1, 7]


# In[ ]:


pl_Barycentric_coordinate_projection(w, 
                                      plot_list = plot_list, 
                                      threshold = 10, 
                                      output_dir = output_dir, 
                                      save_name = save_name, 
                                      col_dic = color_dic,
                                      l = l,
                                      cluster_col = cluster_col,
                                      n_num = n_num,
                                      SMALL_SIZE = 14, 
                                      MEDIUM_SIZE = 16, 
                                      BIGGER_SIZE = 18)


# **CN combination map**
# 
# ![image.png](attachment:image.png)
# 
# The graph shows differnet combinations of neightborhoods as well as single neighborhoods. The circles indicate the size of this specific modules and edges indicate the relationship of the individual CNs.
# 
# 
# 
# 
# 
# This is a Python function that plots a combination map of nodes and edges using the NetworkX and Matplotlib libraries. The function takes in a graph object called g, which is then used to calculate the positions of each node using the graphviz_layout function from the nx.drawing.nx_pydot module. The height variable is set to 8, which appears to determine the vertical spacing between nodes.
# 
# The function then sets the figure size to 40x20 using figsize(40,20), and loops through each node in the graph. For each node, the function determines its color based on the number of incoming edges and plots a scatter point with size determined by the value in the simp_freqs list that corresponds to the node. If the node is in the tops list, it is marked with an asterisk. The function then plots squares below the node for each element in the node, with color determined by the palt dictionary.
# 
# The function then loops through each edge in the graph and plots a line between the two nodes, with the color and thickness of the line determined by the number of incoming edges to the second node.
# 
# Finally, the function turns off the axis labels and displays the plot using plt.show(). The commented-out sections appear to contain code for adding additional information to the plot, such as profiles below each node and highlighting specific edges, but they are not currently being used.

# In[ ]:


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


# In[ ]:


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


# ### Statistics

# In[ ]:


simp_df_tissue1, simp_df_tissue2 = tl_spatial_context_stats(windows, 
                                                         n_num, 
                                                         total_per_thres = 0.9, \
                                                         comb_per_thres = 0.005, \
                                                         tissue_column = 'region',\
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
    
pl_spatial_context_stats_vis(neigh_comb = (9,),
                              simp_df_tissue1 = simp_df_tissue1,
                              simp_df_tissue2 = simp_df_tissue2,
                              pal_tis = {'Resection': 'blue', 'Biopsy': 'orange'},
                              plot_order = ['Resection', 'Biopsy'])


