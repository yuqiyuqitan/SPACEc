import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
import time
import sys
import matplotlib.pyplot as plt
import math
import os
from sklearn.cluster import MiniBatchKMeans
import seaborn as sns
import scanpy as sc
from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd



# helper function
'''
- data: a dataframe that the function will operate on
- grouping: a column name from the input dataframe, defines the groups for calculating the percentage
- replicate: a column name from the input dataframe, defines the replicate groups for calculating the percentage
- sub_col: a column name from the input dataframe, defines a subset of the data to use
- sub_list: a list of values that the function uses to filter the dataframe based on the sub_col
- per_cat: a column name from the input dataframe, defines the category for which the percentage will be calculated
- norm: an optional boolean parameter, if set to True, it will normalize the percentage calculation, if set to False, it will not normalize.
'''
def per_only1(data, grouping, replicate,sub_col, sub_list, per_cat, norm=True):
    
    #Find Percentage of cell type
    if norm==True:
        test1 = data.loc[data[sub_col].isin(sub_list)] #filters df for values by values in sub_list which are in the sub_col column 
        immune_list = list(test1[per_cat].unique()) #stores unique values for the per_cat column 
    else:
        test1=data.copy()
        immune_list = list(data.loc[data[sub_col].isin(sub_list)][per_cat].unique())
    
    test1[per_cat] = test1[per_cat].astype('category')
    test_freq = test1.groupby([grouping,replicate]).apply(lambda x: x[per_cat].value_counts(normalize = True,sort = False)*100) #group data by grouping variable and replicates, than applies the lambda function to count the frequency of each category in the per_cat column and normalizes by dividing by the total count.
    test_freq.columns = test_freq.columns.astype(str)
    test_freq.reset_index(inplace=True)
    immune_list.extend([grouping,replicate]) #adds grouping and replicate column to immune_list 
    test_freq1 = test_freq[immune_list] # subsets test_freq by immune_list

    melt_per_plot = pd.melt(test_freq1, id_vars=[grouping,replicate])#,value_vars=immune_list) #converts columns specified in id_vars into rows
    melt_per_plot.rename(columns={'value': 'percentage'}, inplace=True) #rename value to percentage 
    
    return melt_per_plot # returns a df which contains the group_column followed by the replicate column and the per category column, and a column specifying the percentage
    # Example: percentage CD4+ TCs in unique region E08 assigned to community xxx

# function
# calculates diversity of cell types within a sample 
def Shan_div(data1, sub_l, group_com, per_categ, rep, sub_column, coloring, output_filepath, normalize=True, save=False, \
             ordering=None, fig_size=1.5):
    #calculate Shannon Diversity
    tt = per_only1(data = data1, per_cat = per_categ, grouping = group_com,\
              sub_list=sub_l, replicate=rep, sub_col = sub_column, norm=normalize)
    tt['fraction']= tt['percentage']/100
    tt['Shannon']=tt['fraction']*np.log(tt['fraction'])
    tt.fillna(0,inplace=True)
    sdiv = tt.groupby([rep,group_com]).agg({'Shannon': 'sum'})
    res = sdiv.reset_index()
    res['Shannon Diversity'] = res['Shannon']*-1

    #Run Anova on results
    res_dict = {}
    for treat in list(res[group_com].unique()):
        res_dict[treat] = res.loc[res[group_com]==treat]['Shannon Diversity']

    treat_list = []
    if len(res_dict) > 1:
        for treat in res_dict.keys():
            treat_list.append(res_dict[treat])
        test_results=stats.f_oneway(*treat_list)[1]
    else:
        test_results=stats.f_oneway(res_dict[treat][0])[1]

        
    #Order by average
    if coloring is None:
        if ordering is None:
            plot_order = res.groupby(group_com).mean().reset_index().sort_values(by='Shannon Diversity')[group_com].to_list()    
        else:
            plot_order=ordering
        #Plot the swarmplot of results
        plt.figure(figsize=(fig_size,3))

        ax = sns.boxplot(data = res, x=group_com,  y='Shannon Diversity',  dodge=True, order=plot_order)
                        
        ax = sns.swarmplot(data = res, x=group_com, y='Shannon Diversity', dodge=True, order=plot_order,\
                        edgecolor='black',linewidth=1, color="white")
    
    else:
        if ordering is None:
            plot_order = res.groupby(group_com).mean().reset_index().sort_values(by='Shannon Diversity')[group_com].to_list()    
        else:
            plot_order=ordering
        #Plot the swarmplot of results
        plt.figure(figsize=(fig_size,3))

        ax = sns.boxplot(data = res, x=group_com,  y='Shannon Diversity',  dodge=True, order=plot_order, \
                        palette=coloring)
        ax = sns.swarmplot(data = res, x=group_com, y='Shannon Diversity', dodge=True, order=plot_order,\
                        edgecolor='black',linewidth=1, palette=coloring)

    for patch in ax.artists:
        r, g, b, a = patch.get_facecolor()
        patch.set_facecolor((r, g, b, .3))
    plt.xticks(rotation=90)
    plt.xlabel('')
    plt.ylabel('Shannon Diversity')
    plt.title('')
    sns.despine()
    if save==True:
        plt.savefig(output_filepath+sub_l[0]+'_Shannon.png', format='png', dpi=300, transparent=True, bbox_inches='tight')
    
    plt.show()
    if test_results < 0.05:
        plt.figure(figsize=(fig_size,fig_size))
        tukey = pairwise_tukeyhsd(endog=res['Shannon Diversity'],
                              groups=res[group_com],
                              alpha=0.05)
        tukeydf = pd.DataFrame(data=tukey._results_table.data[1:], columns=tukey._results_table.data[0])
        tukedf_rev = tukeydf.copy()
        tukedf_rev.rename(columns={'group1':'groupa','group2':'groupb'}, inplace=True)
        tukedf_rev.rename(columns={'groupa':'group2','groupb':'group1'}, inplace=True)
        tukedf_rev=tukedf_rev[tukeydf.columns]
        tukey_all = pd.concat([tukedf_rev,tukeydf])

        #Plot with tissue order preserved
        table1 = pd.pivot_table(tukey_all, values='p-adj', index=['group1'],
                            columns=['group2'])
        table1=table1[plot_order]
        table1=table1.reindex(plot_order)

        #plt.figure(figsize = (5,5))
        ax=sns.heatmap(table1, cmap='coolwarm',center=0.05,vmax=0.05)
        ax.set_title('Shannon Diversity') 
        ax.set_ylabel('')    
        ax.set_xlabel('')
        if save==True:    
            plt.savefig(output_filepath+sub_l[0]+'_tukey.png', format='png', dpi=300, transparent=True, bbox_inches='tight')
        plt.show()
    else:
        table1=False
    return tt, test_results, table1