#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 12 16:41:54 2023

@author: timnoahkempchen
"""

# load required packages 
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
import time
import sys
from sklearn.cluster import MiniBatchKMeans
import seaborn as sns
import plotnine
from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import math

from sklearn.cluster import MiniBatchKMeans
import scanpy as sc
from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import statsmodels.api as sm
from sklearn.cross_decomposition import CCA
import networkx as nx
from scipy.stats import pearsonr,spearmanr
from scipy.spatial.distance import cdist
#import graphviz
from tensorly.decomposition import non_negative_tucker
import tensorly as tl
import itertools
from functools import reduce
import os as os
from yellowbrick.cluster import SilhouetteVisualizer

##########################################################################################################
# pp
##########################################################################################################

##########################################################################################################
# tl
##########################################################################################################

# function
# calculates diversity of cell types within a sample 
def tl_Shan_div(data, sub_l, group_com, per_categ, rep, sub_column, normalize=True):
    #calculate Shannon Diversity
    tt = per_only1(data = data, per_cat = per_categ, grouping = group_com,\
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
        
    return tt, test_results, res


#############



def tl_neighborhood_analysis_2(data, values, sum_cols, X = 'x', Y = 'y', reg = 'unique_region', cluster_col = 'Cell Type', k = 35, n_neighborhoods = 30,  calc_silhouette_score = False):

    cells = data.copy()

    neighborhood_name = "neighborhood"+str(k)

    keep_cols = [X ,Y ,reg,cluster_col]

    n_neighbors = k

    cells[reg] = cells[reg].astype('str')

    #Get each region
    tissue_group = cells[[X,Y,reg]].groupby(reg)
    exps = list(cells[reg].unique())
    tissue_chunks = [(time.time(),exps.index(t),t,a) for t,indices in tissue_group.groups.items() for a in np.array_split(indices,1)] 

    tissues = [get_windows(job, n_neighbors, exps= exps, tissue_group = tissue_group, X = X, Y = Y) for job in tissue_chunks]

    #Loop over k to compute neighborhoods
    out_dict = {}
    
    for neighbors,job in zip(tissues,tissue_chunks):

        chunk = np.arange(len(neighbors))#indices
        tissue_name = job[2]
        indices = job[3]
        window = values[neighbors[chunk,:k].flatten()].reshape(len(chunk),k,len(sum_cols)).sum(axis = 1)
        out_dict[(tissue_name,k)] = (window.astype(np.float16),indices)
            
    windows = {}
    
    
    window = pd.concat([pd.DataFrame(out_dict[(exp,k)][0],index = out_dict[(exp,k)][1].astype(int),columns = sum_cols) for exp in exps],0)
    window = window.loc[cells.index.values]
    window = pd.concat([cells[keep_cols],window],1)
    windows[k] = window

    #Fill in based on above
    k_centroids = {}

    #producing what to plot
    windows2 = windows[k]
    windows2[cluster_col] = cells[cluster_col]
    
    if calc_silhouette_score != True:
        km = MiniBatchKMeans(n_clusters = n_neighborhoods,random_state=0)
        
        labels = km.fit_predict(windows2[sum_cols].values)
        k_centroids[k] = km.cluster_centers_
        cells[neighborhood_name] = labels
        
    else:  
        km = MiniBatchKMeans(n_clusters = n_neighborhoods,random_state=0)
        
        X = windows2[sum_cols].values
        
        labels = km.fit_predict(X)
        k_centroids[k] = km.cluster_centers_
        cells[neighborhood_name] = labels
        
        silhouette_score_res = silhouette_score(X, km.labels_, n_jobs = 4)
    
    if calc_silhouette_score != True:
        return(cells, k_centroids)
    else:
        return(cells, k_centroids, silhouette_score_res)
    
    
############

def tl_cell_types_de(ct_freq, all_freqs, neighborhood_num, nbs, patients, group, cells, cells1):
    
    # data prep
    # normalized overall cell type frequencies
    X_cts = normalize(ct_freq.reset_index().set_index('patients').loc[patients,cells])
    
    # normalized neighborhood specific cell type frequencies
    df_list = []
    
    for nb in nbs:
        cond_nb = all_freqs.loc[all_freqs[neighborhood_num]==nb,cells1].rename({col: col+'_'+str(nb) for col in cells}, axis = 1).set_index('patients')
        df_list.append(normalize(cond_nb))
    
    X_cond_nb = pd.concat(df_list, axis = 1).loc[patients]
    
    #differential enrichment for all cell subsets
    changes = {}
    #nbs =[0, 2, 3, 4, 6, 7, 8, 9]
    for col in cells:
        for nb in nbs:
            #build a design matrix with a constant, group 0 or 1 and the overall frequencies
            X = pd.concat([X_cts[col], group.astype('int'),pd.Series(np.ones(len(group)), index = group.index.values)], axis = 1).values
            if col+'_%d'%nb in X_cond_nb.columns:
                #set the neighborhood specific ct freqs as the outcome
                Y = X_cond_nb[col+'_%d'%nb].values
                X = X[~pd.isna(Y)]
                Y = Y[~pd.isna(Y)]
                #fit a linear regression model
                results = sm.OLS(Y,X).fit()
                #find the params and pvalues for the group coefficient
                changes[(col,nb)] = (results.pvalues[1], results.params[1])
            
    
    #make a dataframe with coeffs and pvalues
    dat = (pd.DataFrame(changes).loc[1].unstack())
    dat = pd.DataFrame(np.nan_to_num(dat.values),index = dat.index, columns = dat.columns).T.sort_index(ascending=True).loc[:,X_cts.columns]
    pvals = (pd.DataFrame(changes).loc[0].unstack()).T.sort_index(ascending=True).loc[:,X_cts.columns]
    
    #this is where you should correct pvalues for multiple testing 
    
    return dat, pvals
    
   
#######


def tl_community_analysis_2(data, values, sum_cols, X = 'x', Y = 'y', reg = 'unique_region', cluster_col = 'neigh_name', k = 100, n_neighborhoods = 30):
    
    cells = data.copy()

    neighborhood_name = "community"+str(k)

    keep_cols = [X ,Y ,reg,cluster_col]

    n_neighbors = k

    cells[reg] = cells[reg].astype('str')

    #Get each region
    tissue_group = cells[[X,Y,reg]].groupby(reg)
    exps = list(cells[reg].unique())
    tissue_chunks = [(time.time(),exps.index(t),t,a) for t,indices in tissue_group.groups.items() for a in np.array_split(indices,1)] 

    tissues = [get_windows(job, n_neighbors, exps= exps, tissue_group = tissue_group, X = X, Y = Y) for job in tissue_chunks]

    #Loop over k to compute neighborhoods
    out_dict = {}
    
    for neighbors,job in zip(tissues,tissue_chunks):

        chunk = np.arange(len(neighbors))#indices
        tissue_name = job[2]
        indices = job[3]
        window = values[neighbors[chunk,:k].flatten()].reshape(len(chunk),k,len(sum_cols)).sum(axis = 1)
        out_dict[(tissue_name,k)] = (window.astype(np.float16),indices)
            
    windows = {}
    
    
    window = pd.concat([pd.DataFrame(out_dict[(exp,k)][0],index = out_dict[(exp,k)][1].astype(int),columns = sum_cols) for exp in exps],0)
    window = window.loc[cells.index.values]
    window = pd.concat([cells[keep_cols],window],1)
    windows[k] = window

    #Fill in based on above
    k_centroids = {}
    
    
    #producing what to plot
    windows2 = windows[k]
    windows2[cluster_col] = cells[cluster_col]
    
    km = MiniBatchKMeans(n_clusters = n_neighborhoods,random_state=0)
    
    labels = km.fit_predict(windows2[sum_cols].values)
    k_centroids[k] = km.cluster_centers_
    cells[neighborhood_name] = labels
    
    return(cells, neighborhood_name, k_centroids)

   
#################


# CCA Analysis 

def tl_Perform_CCA(cca, n_perms, nsctf, cns, subsets, group):
    stats_group1 = {}
    for cn_i in cns:
        for cn_j in cns:
            if cn_i < cn_j:
                print(cn_i, cn_j)
                #concat dfs
                combined = pd.concat([nsctf.loc[cn_i].loc[nsctf.loc[cn_i].index.isin(group)],nsctf.loc[cn_j].loc[nsctf.loc[cn_j].index.isin(group)]], axis = 1).dropna(axis = 0, how = 'any')
                if combined.shape[0]>2:
                    if subsets != None:
                        x = combined.iloc[:,:len(subsets)].values
                        y = combined.iloc[:,len(subsets):].values
                    else: 
                        x = combined.values
                        y = combined.values

                    arr = np.zeros(n_perms)
                    #compute the canonical correlation achieving components with respect to observed data
                    ccx,ccy = cca.fit_transform(x,y)
                    stats_group1[cn_i,cn_j] = (pearsonr(ccx[:,0],ccy[:,0])[0],arr)
                    #initialize array for perm values
                    for i in range(n_perms):
                        idx = np.arange(len(x))
                        np.random.shuffle(idx)
                        # compute with permuted data
                        cc_permx,cc_permy = cca.fit_transform(x[idx],y)
                        arr[i] = pearsonr(cc_permx[:,0],cc_permy[:,0])[0]
    return stats_group1 

        

# tensor decomposition 




#######

def tl_build_tensors(df, group, cns, cts, counts):
    
    #initialize the tensors
    T1 = np.zeros((len(group),len(cns),len(cts)))
    
    for i,pat in enumerate(group):
        for j,cn in enumerate(cns):
            for k,ct in enumerate(cts):
                print(i, pat)
                print(j,cn)
                print(k,ct)
                T1[i,j,k] = counts.loc[(pat,cn,ct)]

    #normalize so we have joint distributions each slice
    dat1 =np.nan_to_num(T1/T1.sum((1,2), keepdims = True))
    
    return(dat1)  
##########################################################################################################
# pl 
##########################################################################################################

'''
data: a Pandas DataFrame containing the data to be plotted.
per_cat: a string representing the column name containing the categories to be plotted.
grouping: a string representing the column name used to group the data.
cell_list: a list of strings representing the categories to be plotted.
output_dir: a string representing the output directory to save the plot.
norm: a boolean value indicating whether to normalize the data or not (default: True).
save_name: a string representing the filename to save the plot (default: None).
col_order: a list of strings representing the order of the columns in the plot (default: None).
sub_col: a string representing the column name used to subset the data (default: None).
name_cat: a string representing the name of the category column in the plot (default: 'Cell Type').
fig_sizing: a tuple representing the size of the plot (default: (8,4)).
h_order: a list of strings representing the order of the categories in the plot (default: None).
pal_color: a dictionary containing color codes for the categories in the plot (default: None).
remove_leg: a boolean value indicating whether to remove the legend or not (default: False).

The function returns a Pandas DataFrame and a list of strings. The DataFrame contains the data used to create the plot, and the list of strings represents the order of the categories in the plot.
'''

def pl_stacked_bar_plot(data, per_cat, grouping, cell_list, output_dir,norm=True, save_name=None,\
              col_order=None, sub_col=None, name_cat = 'Cell Type',fig_sizing=(8,4),\
                     h_order=None, pal_color=None,remove_leg=False):
    
    #Find Percentage of cell type
    if norm==True:
        if sub_col is None:
            test1 = data.loc[data[per_cat].isin(cell_list)]
            sub_cell_list = list(test1[per_cat].unique())
        else:
            test1 = data.loc[data[sub_col].isin(cell_list)]
            sub_cell_list = list(test1[per_cat].unique())
    else:
        if sub_col is None:
            test1 = data.copy()
            sub_cell_list = list(data.loc[data[per_cat].isin(cell_list)][per_cat].unique())
        else:
            test1 = data.copy()
            sub_cell_list = list(data.loc[data[sub_col].isin(cell_list)][per_cat].unique())
            
    test1[per_cat] = test1[per_cat].astype('category')
    test_freq = test1.groupby(grouping).apply(lambda x: x[per_cat].value_counts(normalize = True,sort = False)*100)
    test_freq.columns = test_freq.columns.astype(str)
    
    ##### Can subset it here if I do not want normalized per the group
    test_freq.reset_index(inplace=True)
    sub_cell_list.append(grouping)
    test_freq = test_freq[sub_cell_list]
    melt_test = pd.melt(test_freq, id_vars=[grouping])#, value_vars=test_freq.columns)
    melt_test.rename(columns = {per_cat: name_cat, 'value':'percent'},  inplace = True)
    
    if norm==True:
        if col_order is None:
            bb = melt_test.groupby([grouping, per_cat]).sum().reset_index()
            col_order = bb.loc[bb[per_cat]==bb[per_cat][0]].sort_values(by='percent')[grouping].to_list()
    else:    
        if col_order is None:
            col_order = melt_test.groupby(grouping).sum().reset_index().sort_values(by='percent')[grouping].to_list()
    
    if h_order is None:
        h_order = list(melt_test[per_cat].unique()) 
    
    #Set up for plotting
    melt_test_piv = pd.pivot_table(melt_test, columns = [name_cat], index=[grouping], values=['percent'])
    melt_test_piv.columns = melt_test_piv.columns.droplevel(0)
    melt_test_piv.reset_index(inplace=True)
    melt_test_piv.set_index(grouping, inplace=True)
    melt_test_piv = melt_test_piv.reindex(col_order)
    melt_test_piv = melt_test_piv[h_order]
    
    #Get color dictionary 
    if pal_color is None:
        #first subplot
        ax1 = melt_test_piv.plot.bar(alpha = 0.8, linewidth=1,\
                                    figsize =fig_sizing, rot=90,stacked=True, edgecolor='black')

    else: 
        #first subplot
        ax1 = melt_test_piv.plot.bar(alpha = 0.8, linewidth=1, color=[pal_color.get(x) for x in melt_test_piv.columns],\
                                    figsize =fig_sizing, rot=90,stacked=True, edgecolor='black')

    for line in ax1.lines:
        line.set_color('black')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    if remove_leg==True:
        ax1.set_ylabel('')
        ax1.set_xlabel('')
    else:
        ax1.set_ylabel('percent')
    #ax1.spines['left'].set_position(('data', 1.0))
    #ax1.set_xticks(np.arange(1,melt_test.day.max()+1,1))
    #ax1.set_ylim([0, int(ceil(max(max(melt_test_piv.sum(axis=1)), max(tm_piv.sum(axis=1)))))])
    plt.xticks(list(range(len(list(melt_test_piv.index)))), list(melt_test_piv.index), rotation=90)
    lgd2 = ax1.legend(loc='center left', bbox_to_anchor=(1.0, 0.5), ncol=1, frameon=False)
    if save_name:
        plt.savefig(output_dir+save_name+'.png', format='png',\
                    dpi=300, transparent=True, bbox_inches='tight')
    return melt_test_piv, h_order  


#############

'''
data: pandas DataFrame containing the data to be plotted
grouping: name of the column containing the grouping variable for the swarm boxplot
replicate: name of the column containing the replicate variable for the swarm boxplot
sub_col: name of the column containing the subsetting variable for the swarm boxplot
sub_list: list of values to subset the data by
per_cat: name of the column containing the categorical variable for the swarm boxplot
output_dir: directory where the output plot will be saved
norm: boolean (default True) to normalize data by subsetting variable before plotting
figure_sizing: tuple (default (10,5)) containing the size of the output plot
save_name: name of the file to save the output plot (if output_dir is provided)
h_order: list of values to specify the order of the horizontal axis
col_in: list of values to subset the data by the per_cat column
pal_color: seaborn color palette for the boxplot and swarmplot
flip: boolean (default False) to flip the orientation of the plot
'''
# This function creates a box plot and swarm plot from the given data
# and returns a plot object.

def pl_swarm_box(data, grouping, replicate, sub_col, sub_list, per_cat, output_dir, norm=True,\
              figure_sizing=(10,5), save_name=None, h_order=None, col_in=None, \
              pal_color=None, flip=False):
       
    #Find Percentage of cell type
    test= data.copy()
    sub_list1 = sub_list.copy()
    
    if norm==True:
        test1 = test.loc[test[sub_col].isin(sub_list1)]
        immune_list = list(test1[per_cat].unique())
    else:
        test1=test.copy()
        immune_list = list(test.loc[test[sub_col].isin(sub_list1)][per_cat].unique())
    
    test1[per_cat] = test1[per_cat].astype('category')
    test_freq = test1.groupby([grouping,replicate]).apply(lambda x: x[per_cat].value_counts(normalize = True,sort = False)*100)
    test_freq.columns = test_freq.columns.astype(str)
    test_freq.reset_index(inplace=True)
    immune_list.extend([grouping,replicate])
    test_freq1 = test_freq[immune_list]

    melt_per_plot = pd.melt(test_freq1, id_vars=[grouping,replicate,])#,value_vars=immune_list)
    melt_per_plot.rename(columns={'value': 'percentage'}, inplace=True)
    
    if col_in:
        melt_per_plot = melt_per_plot.loc[melt_per_plot[per_cat].isin(col_in)]
    else:
        melt_per_plot = melt_per_plot
    
    #Order by average
    plot_order = melt_per_plot.groupby(per_cat).mean().reset_index().sort_values(by='percentage')[per_cat].to_list()

    if h_order is None:
        h_order = list(melt_per_plot[grouping].unique()) 
    
    
    #swarmplot to compare clustering
    plt.figure(figsize=figure_sizing)
    if flip==True:
        plt.figure(figsize=figure_sizing)
        if pal_color is None:
            ax = sns.boxplot(data = melt_per_plot, x=grouping,  y='percentage',  dodge=True, order=h_order)
            ax = sns.swarmplot(data = melt_per_plot, x=grouping, y='percentage', dodge=True,order=h_order,\
                           edgecolor='black',linewidth=1, color="white")
        else:
            ax = sns.boxplot(data = melt_per_plot, x=grouping,  y='percentage',  dodge=True,order=h_order, \
                         palette=pal_color)
            ax = sns.swarmplot(data = melt_per_plot, x=grouping, y='percentage', dodge=True,order=h_order,\
                           edgecolor='black',linewidth=1, palette=pal_color)
    
        for patch in ax.artists:
            r, g, b, a = patch.get_facecolor()
            patch.set_facecolor((r, g, b, .3))
        plt.xticks(rotation=90)
        plt.xlabel('')
        plt.ylabel('')
        plt.title(sub_list[0])
        sns.despine()
        
    else:
        if pal_color is None:
            ax = sns.boxplot(data = melt_per_plot, x=grouping,  y='percentage',  dodge=True, order=h_order)
            ax = sns.swarmplot(data = melt_per_plot, x=grouping, y='percentage', dodge=True,order=h_order,\
                           edgecolor='black',linewidth=1, color="white")
        else:
            ax = sns.boxplot(data = melt_per_plot, x=grouping,  y='percentage',  dodge=True,order=h_order, \
                         palette=pal_color)
            ax = sns.swarmplot(data = melt_per_plot, x=grouping, y='percentage', dodge=True,order=h_order,\
                           edgecolor='black',linewidth=1, palette=pal_color)
        for patch in ax.artists:
            r, g, b, a = patch.get_facecolor()
            patch.set_facecolor((r, g, b, .3))
        #ax.set_yscale(\log\)
        plt.xlabel('')
        handles, labels = ax.get_legend_handles_labels()
        plt.legend(handles[:len(melt_per_plot[grouping].unique())], labels[:len(melt_per_plot[grouping].unique())],\
                   bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.,frameon=False)
        plt.xticks(rotation=90)

        ax.set(ylim=(0,melt_per_plot['percentage'].max()+1))
        sns.despine()
    
    if output_dir:
        if save_name:
            plt.savefig(output_dir+save_name+'_swarm_boxplot.png', format='png', dpi=300, transparent=True, bbox_inches='tight')
        else:
            print("define save_name")
    else: 
        print("plot was not saved - to save the plot specify an output directory")
    return melt_per_plot




#############


def pl_Shan_div(tt, test_results, res, group_com, coloring, sub_l, output_dir, save=False, ordering=None, fig_size=1.5):      
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
        plt.savefig(output_dir+sub_l[0]+'_Shannon.png', format='png', dpi=300, transparent=True, bbox_inches='tight')
    
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
            plt.savefig(output_dir+sub_l[0]+'_tukey.png', format='png', dpi=300, transparent=True, bbox_inches='tight')
        plt.show()
    else:
        table1=False
        

#############

   
def pl_cell_type_composition_vis(data, sample_column = "sample", cell_type_column = "Cell Type", output_dir = None):
    
    if output_dir == None:
        print("You have defined no output directory!")
    
    #plotting option1
    #pd.crosstab(df['sample'], df['final_cell_types']).plot(kind='barh', stacked=True,figsize = (10,12))
    #plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
    #plt.show()
    

    #plotting option2
    ax = pd.crosstab(data[sample_column], data[cell_type_column]).plot(kind='barh', stacked=True,figsize = (10,10))
    ax.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
    fig = ax.get_figure()
    ax.set(xlabel='count')
    plt.savefig(output_dir +'/cell_types_composition_hstack.png', bbox_inches='tight')

    #plotting option1
    #pd.crosstab(df['sample'], df['final_cell_types']).plot(kind='barh', figsize = (10,10))
    #plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
    #plt.show()

    #plotting option2
    ax = pd.crosstab(data[sample_column], data[cell_type_column]).plot(kind='barh', stacked=False,figsize = (10,10))
    ax.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
    fig = ax.get_figure()
    ax.set(xlabel='count')
    plt.savefig(output_dir +'/cell_types_composition_hUNstack.png', bbox_inches='tight')

    # Cell type percentage 
    st = pd.crosstab(data[sample_column], data[cell_type_column])
    df_perc=(st/np.sum(st, axis = 1)[:,None])* 100
    df_perc
    #df_perc['sample'] = df_perc.index
    #df_perc

    tmp=st.T.apply(
    lambda x: 100 * x / x.sum()
    )

    ax = tmp.T.plot(kind='barh', stacked=True,figsize = (10,10))
    ax.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
    fig = ax.get_figure()
    ax.set(xlabel='percentage')
    plt.savefig(output_dir +'/cell_types_composition_perc_hstack.png', bbox_inches='tight')
    
    
##############


def pl_neighborhood_analysis_2(data, k_centroids, values, sum_cols, X = 'x', Y = 'y', reg = 'unique_region', output_dir = None, k = 35, plot_specific_neighborhoods = None):

    
    #modify figure size aesthetics for each neighborhood
    figs = catplot(data,X = X,Y=Y,exp = reg,hue = 'neighborhood'+str(k),invert_y=True,size = 5,)
 

    #Save Plots for Publication
    for n,f in enumerate(figs):
        f.savefig(output_dir+'neighborhood_'+str(k)+'_id{}.png'.format(n))

    #this plot shows the types of cells (ClusterIDs) in the different niches (0-9)
    k_to_plot = k
    niche_clusters = (k_centroids[k_to_plot])
    tissue_avgs = values.mean(axis = 0)
    fc = np.log2(((niche_clusters+tissue_avgs)/(niche_clusters+tissue_avgs).sum(axis = 1, keepdims = True))/tissue_avgs)
    fc = pd.DataFrame(fc,columns = sum_cols)
    s=sns.clustermap(fc, vmin =-3,vmax = 3,cmap = 'bwr')
    s.savefig(output_dir+"celltypes_perniche_"+"_"+str(k)+".png", dpi=600)

    if plot_specific_neighborhoods is True:
        #this plot shows the types of cells (ClusterIDs) in the different niches (0-9)
        k_to_plot = k
        niche_clusters = (k_centroids[k_to_plot])
        tissue_avgs = values.mean(axis = 0)
        fc = np.log2(((niche_clusters+tissue_avgs)/(niche_clusters+tissue_avgs).sum(axis = 1, keepdims = True))/tissue_avgs)
        fc = pd.DataFrame(fc,columns = sum_cols)
        s=sns.clustermap(fc.iloc[plot_specific_neighborhoods,:], vmin =-3,vmax = 3,cmap = 'bwr')
        s.savefig(output_dir+"celltypes_perniche_"+"_"+str(k)+".png", dpi=600)
    

##############


def pl_cell_types_de(dat, pvals, neigh_num, output_dir):
   
    #plot as heatmap
    f, ax = plt.subplots(figsize = (20,10))
    g = sns.heatmap(dat,cmap = 'bwr', vmin = -1, vmax = 1,cbar=False,ax = ax)
    for a,b in zip(*np.where (pvals<0.05)):
        plt.text(b+.5,a+.55,'*',fontsize = 20,ha = 'center',va = 'center')
    plt.tight_layout()
    
    inv_map = {v: k for k, v in neigh_num.items()}
    inv_map
    
    #plot as heatmap
    plt.style.use(['default'])
    #GENERAL GRAPH SETTINGs
    #font size of graph
    SMALL_SIZE = 14
    MEDIUM_SIZE = 16
    BIGGER_SIZE = 18
    
    #Settings for graph
    plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
    
    data_2 = dat.rename(index=inv_map)
    
    
    #Sort both axes
    sort_sum = data_2.abs().sum(axis=1).to_frame()
    sort_sum.columns = ['sum_col']
    xx = sort_sum.sort_values(by='sum_col')
    sort_x = xx.index.values.tolist()
    sort_sum_y = data_2.abs().sum(axis=0).to_frame()
    sort_sum_y.columns = ['sum_col']
    yy = sort_sum_y.sort_values(by='sum_col')
    sort_y = yy.index.values.tolist()
    df_sort = data_2.reindex(index = sort_x, columns =sort_y)
    
    
    f, ax = plt.subplots(figsize = (15,10))
    g = sns.heatmap(df_sort,cmap = 'bwr', vmin = -1, vmax = 1,cbar=True,ax = ax)
    for a,b in zip(*np.where (pvals<0.05)):
        plt.text(b+.5,a+.55,'*',fontsize = 20,ha = 'center',va = 'center')
    plt.tight_layout()
    
    f.savefig(output_dir+"tissue_neighborhood_coeff_pvalue_bar.png", format='png', dpi=300, transparent=True, bbox_inches='tight')
    
    df_sort.abs().sum()
    
    
##############
    

def pl_community_analysis_2(data, values, sum_cols, output_dir, neighborhood_name, k_centroids, X = 'x', Y = 'y', reg = 'unique_region', save_path = None, k = 100, plot_specific_community = None):
    
    output_dir2 = output_dir+"community_analysis/"
    if not os.path.exists(output_dir2):
        os.makedirs(output_dir2)
    
    cells = data.copy()
    
    #modify figure size aesthetics for each neighborhood
    plt.rcParams["legend.markerscale"] = 10
    figs = catplot(cells,X = X,Y=Y,exp = reg,
                   hue = neighborhood_name,invert_y=True,size = 1,figsize=8)
    
    #Save Plots for Publication
    for n,f in enumerate(figs):
        f.savefig(output_dir2+neighborhood_name+'_id{}.png'.format(n))
 
    if plot_specific_community is True:
        #this plot shows the types of cells (ClusterIDs) in the different niches (0-9)
        k_to_plot = k
        niche_clusters = (k_centroids[k_to_plot])
        tissue_avgs = values.mean(axis = 0)
        fc = np.log2(((niche_clusters+tissue_avgs)/(niche_clusters+tissue_avgs).sum(axis = 1, keepdims = True))/tissue_avgs)
        fc = pd.DataFrame(fc,columns = sum_cols)
        s=sns.clustermap(fc.iloc[plot_specific_community,:], vmin =-3,vmax = 3,cmap = 'bwr',figsize=(10,5))
        s.savefig(output_dir2+"celltypes_perniche_"+"_"+str(k)+".png", dpi=600)
    
    
    #this plot shows the types of cells (ClusterIDs) in the different niches (0-9)
    k_to_plot = k
    niche_clusters = (k_centroids[k_to_plot])
    tissue_avgs = values.mean(axis = 0)
    fc = np.log2(((niche_clusters+tissue_avgs)/(niche_clusters+tissue_avgs).sum(axis = 1, keepdims = True))/tissue_avgs)
    fc = pd.DataFrame(fc,columns = sum_cols)
    s=sns.clustermap(fc, vmin =-3,vmax = 3,cmap = 'bwr', figsize=(10,10))
    s.savefig(output_dir2+"celltypes_perniche_"+"_"+str(k)+".png", dpi=600)
    
    
###############


def pl_Visulize_CCA_results(CCA_results, save_path, save_fig = False, p_thresh = 0.1, save_name = "CCA_vis.png", colors = None):
    # Visualization of CCA 
    g1 = nx.petersen_graph()
    for cn_pair, cc in CCA_results.items():
        s,t = cn_pair
        obs, perms = cc
        p =np.mean(obs>perms)
        if p>p_thresh :
                g1.add_edge(s,t, weight = p)
    
    if colors != None:
        pal = colors
    else:
        pal = sns.color_palette('bright',50)
    
    pos=nx.nx_agraph.graphviz_layout(g1,prog='neato')
    for k,v in pos.items():
        x,y = v
        plt.scatter([x],[y],c = [pal[k]], s = 300,zorder = 3)
        #plt.text(x,y, k, fontsize = 10, zorder = 10,ha = 'center', va = 'center')
        plt.axis('off')
                
    for e0,e1 in g1.edges():
        if isinstance(g1.get_edge_data(e0, e1, default =0), int):
            p = g1.get_edge_data(e0, e1, default =0)
            p =p["weight"]
            print(p)
        else:
            p = g1.get_edge_data(0, 1, default =0)
            p =p["weight"]
            print(p)


        alpha = 3*p**1
        if alpha > 1:
            alpha = 1

        plt.plot([pos[e0][0],pos[e1][0]],[pos[e0][1],pos[e1][1]], c= 'black',alpha = alpha, linewidth = 3*p**3)
    if save_fig == True:
        plt.savefig(save_path + "/" + save_name, format='png', dpi=300, transparent=True, bbox_inches='tight') 
        
        

#######

def pl_plot_modules_heatmap(dat, cns, cts, figsize = (20,5), num_tissue_modules = 2, num_cn_modules = 5):
   # figsize(20,5)
    core, factors = non_negative_tucker(dat,rank=[num_tissue_modules,num_cn_modules,num_cn_modules],random_state = 32)
    plt.subplot(1,2,1)
    sns.heatmap(pd.DataFrame(factors[1],index = cns))
    plt.ylabel('CN')
    plt.xlabel('CN module')
    plt.title('Loadings onto CN modules')
    plt.subplot(1,2,2)
    sns.heatmap(pd.DataFrame(factors[2],index = cts))
    plt.ylabel('CT')
    plt.xlabel('CT module')
    plt.title('Loadings onto CT modules')
    plt.show()
    
    #figsize(num_tissue_modules*3,3)
    for p in range(num_tissue_modules):
        plt.subplot(1, num_tissue_modules, p+1)
        sns.heatmap(pd.DataFrame(core[p]))
        plt.title('tissue module {}, couplings'.format(p))
        plt.ylabel('CN module')
        plt.ylabel('CT module')
    plt.show()

#######

def pl_plot_modules_graphical(dat, cts, cns, num_tissue_modules = 2, num_cn_modules = 4, scale = 0.4, figsize = (1.5, 0.8), pal=None,save_name=None, save_path = None):
    
    core, factors = non_negative_tucker(dat,rank=[num_tissue_modules,num_cn_modules,num_cn_modules],random_state = 32)
    
    if pal is None:
        pal = sns.color_palette('bright',10)
    palg = sns.color_palette('Greys',10)
    
    figsize= (3.67*scale,2.00*scale)
    cn_scatter_size = scale*scale*45
    cel_scatter_size = scale*scale*15
    
    

    for p in range(num_tissue_modules):
        for idx in range(num_cn_modules):
            an = float(np.max(core[p][idx,:])>0.1) + (np.max(core[p][idx,:])<=0.1)*0.05
            ac = float(np.max(core[p][:,idx])>0.1) + (np.max(core[p][:,idx])<=0.1)*0.05

            cn_fac = factors[1][:,idx]
            cel_fac = factors[2][:,idx]

            cols_alpha = [(*pal[cn], an*np.minimum(cn_fac, 1.0)[i]) for i,cn in enumerate(cns)]
            cols = [(*pal[cn], np.minimum(cn_fac, 1.0)[i]) for i,cn in enumerate(cns)]
            cell_cols_alpha = [(0,0,0, an*np.minimum(cel_fac, 1.0)[i]) for i,_ in enumerate(cel_fac)]
            cell_cols = [(0,0,0, np.minimum(cel_fac, 1.0)[i]) for i,_ in enumerate(cel_fac)]
            
            plt.scatter(0.5*np.arange(len(cn_fac)), 5*idx + np.zeros(len(cn_fac)), c = cols_alpha, s = cn_scatter_size)
            offset = 9
            for i,k in enumerate(cns):
                plt.text(0.5*i, 5*idx, k,fontsize = scale*2,ha = 'center', va = 'center',alpha = an)

            plt.scatter(-4.2+0.25*np.arange(len(cel_fac))+offset, 5*idx + np.zeros(len(cel_fac)), c = cell_cols_alpha, s = 0.5*cel_scatter_size)#,vmax = 0.5,edgecolors=len(cell_cols_alpha)*[(0,0,0,min(1.0,max(0.1,2*an)))], linewidths= 0.05)
            
            
            rect = plt.Rectangle((-0.5,5*idx-2 ),4.5,4,linewidth=scale*scale*1,edgecolor='black',facecolor='none',zorder = 0,alpha = an,linestyle = '--')
            ax = plt.gca()
            ax.add_artist(rect)
            plt.scatter([offset-5],[5*idx],c = 'black', marker = 'D', s = scale*scale*5, zorder = 5,alpha = an)
            plt.text(offset-5,5*idx,idx,color = 'white',alpha = an, ha = 'center', va = 'center',zorder = 6,fontsize = 4.5)
            plt.scatter([offset-4.5],[5*idx],c = 'black', marker = 'D', s = scale*scale*5, zorder = 5,alpha = ac)
            plt.text(offset-4.5,5*idx,idx,color = 'white',alpha = ac, ha = 'center', va = 'center', zorder = 6,fontsize = 4.5)

            rect = plt.Rectangle((offset-4.5,5*idx-2 ),4.5,4,linewidth=scale*1,edgecolor='black',facecolor='none',zorder = 0, alpha = ac,linestyle = '-.')
            ax.add_artist(rect)

        for i,ct in enumerate(cts):
                plt.text(-4.2+offset+0.25*i, 27.5, ct, rotation = 45, color = 'black',ha = 'left', va = 'bottom',fontsize = scale*2,alpha = 1)
        for cn_i in range(num_cn_modules):
            for cel_i in range(num_cn_modules):
                plt.plot([-3+offset -2, -4+offset - 0.5],[5*cn_i, 5*cel_i], color = 'black', linewidth =2*scale*scale*1* min(1.0, max(0,-0.00+core[p][cn_i,cel_i])),alpha = min(1.0, max(0.000,-0.00+10*core[p][cn_i,cel_i])))#max(an,ac))



        plt.ylim(-5, 30)
        plt.axis('off')
        
        if save_name:
            plt.savefig(save_path+save_name+'_'+str(p)+'_tensor.png', format='png', dpi=300, transparent=True, bbox_inches='tight')

        plt.show()
        
        
#########


def pl_evaluate_ranks(dat, num_tissue_modules = 2):
    num_tissue_modules = num_tissue_modules+1
    pal = sns.color_palette('bright',10)
    palg = sns.color_palette('Greys',10)
    
    mat1 = np.zeros((num_tissue_modules,15))
    for i in range(2,15):
        for j in range(1,num_tissue_modules):
            # we use NNTD as described in the paper
            facs_overall = non_negative_tucker(dat,rank=[j,i,i],random_state = 2336)
            mat1[j,i] = np.mean((dat- tl.tucker_to_tensor((facs_overall[0],facs_overall[1])))**2)
    for j in range(1,num_tissue_modules):
        plt.plot(2+np.arange(13),mat1[j][2:],label = 'rank = ({},x,x)'.format(j))
        
    plt.xlabel('x')
    plt.ylabel('reconstruction error')
    plt.legend()
    plt.show()
        

##########################################################################################################
# hf
##########################################################################################################