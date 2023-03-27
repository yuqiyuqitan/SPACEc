#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 25 12:43:45 2023

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


# load functions 
def stacked_bar_plot(data, per_cat, grouping, cell_list, output_dir,norm=True, save_name=None,\
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


##########################################################################################################
# This function creates a box plot and swarm plot from the given data
# and returns a plot object.

def swarm_box(data, grouping, replicate, sub_col, sub_list, per_cat, output_dir, norm=True, \
              figure_sizing=(10,5), save_name=None, h_order=None, col_in=None, \
              pal_color=None, flip=False):
       
    # If norm is True, get the percentage of cell type by subsetting the data
    # and computing the unique values of a given category column. Otherwise,
    # copy the entire data.
    
    if norm==True:
        test1 = data.loc[data[sub_col].isin(sub_list)]
        immune_list = list(test1[per_cat].unique())
    else:
        test1=data.copy()
        immune_list = list(data.loc[data[sub_col].isin(sub_list)][per_cat].unique())
    
    # Cast the category column to categorical type.
    test1[per_cat] = test1[per_cat].astype('category')
    
    # Compute the percentage of each category column by group and replicate.
    test_freq = test1.groupby([grouping,replicate]).apply(lambda x: x[per_cat].value_counts(normalize = True,sort = False)*100)
    
    # Convert column names to string type and reset index.
    test_freq.columns = test_freq.columns.astype(str)
    test_freq.reset_index(inplace=True)
    
    # Add grouping and replicate to immune_list and subset the data.
    immune_list.extend([grouping,replicate])
    test_freq1 = test_freq[immune_list]

    # Melt the data frame and rename columns.
    melt_per_plot = pd.melt(test_freq1, id_vars=[grouping,replicate,])
    melt_per_plot.rename(columns={'value': 'percentage'}, inplace=True)
    
    # If col_in is not None, subset melt_per_plot to include only those values.
    if col_in:
        melt_per_plot = melt_per_plot.loc[melt_per_plot[per_cat].isin(col_in)]
    else:
        melt_per_plot = melt_per_plot
    
    # Order the data by average percentage of each category column.
    plot_order = melt_per_plot.groupby(per_cat).mean().reset_index().sort_values(by='percentage')[per_cat].to_list()

    # If h_order is None, use unique values of the grouping column as the order.
    if h_order is None:
        h_order = list(melt_per_plot[grouping].unique()) 
    
    # If pal_color is None, create a figure with box plot and swarm plot
    # for each category column or grouping column based on flip value.
    if pal_color is None:
        # Create a figure and axis object with given figure size.
        plt.figure(figsize=figure_sizing)
        
        # If flip is True, plot box plot and swarm plot for grouping column.
        if flip==True:
            plt.figure(figsize=figure_sizing)
            
            # Create a box plot with given parameters.
            ax = sns.boxplot(data = melt_per_plot, x=grouping,  y='percentage',  dodge=True,order=h_order)
                           
            # Create a swarm plot with given parameters.
            ax = sns.swarmplot(data = melt_per_plot, x=grouping, y='percentage', dodge=True,order=h_order,\
                            edgecolor='black',linewidth=1, color="white")
        
            # Set the transparency of box plot patches.
            for patch in ax.artists:
                r, g, b, a = patch.get_facecolor()
                patch.set_facecolor((r, g, b,)
                )
    
    if save_name:
        plt.savefig(output_dir+save_name+'.png', format='png',\
                    dpi=300, transparent=True, bbox_inches='tight')



##########################################################################################################

# function
# calculates diversity of cell types within a sample 
def Shan_div(data1, sub_l, group_com, per_categ, rep, sub_column, coloring, output_dir, normalize=True, save=False, \
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
    return tt, test_results, table1


##########################################################################################################
def cell_type_composition_vis(df, sample_column = "sample", cell_type_column = "Cell Type", output_dir = None):
    
    if output_dir == None:
        print("You have defined no output directory!")
    
    #plotting option1
    #pd.crosstab(df['sample'], df['final_cell_types']).plot(kind='barh', stacked=True,figsize = (10,12))
    #plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
    #plt.show()
    

    #plotting option2
    ax = pd.crosstab(df[sample_column], df[cell_type_column]).plot(kind='barh', stacked=True,figsize = (10,10))
    ax.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
    fig = ax.get_figure()
    ax.set(xlabel='count')
    plt.savefig(output_dir +'/cell_types_composition_hstack.png', bbox_inches='tight')

    #plotting option1
    #pd.crosstab(df['sample'], df['final_cell_types']).plot(kind='barh', figsize = (10,10))
    #plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
    #plt.show()

    #plotting option2
    ax = pd.crosstab(df[sample_column], df[cell_type_column]).plot(kind='barh', stacked=False,figsize = (10,10))
    ax.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
    fig = ax.get_figure()
    ax.set(xlabel='count')
    plt.savefig(output_dir +'/cell_types_composition_hUNstack.png', bbox_inches='tight')

    # Cell type percentage 
    st = pd.crosstab(df[sample_column], df[cell_type_column])
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


##########################################################################################################
def neighborhood_analysis(df, values, sum_cols, X = 'x', Y = 'y', reg = 'unique_region', cluster_col = 'Cell Type', ks = [20, 30, 35], output_dir = None, k = 35, n_neighborhoods = 30, save_to_csv = False, plot_specific_neighborhoods = None ):

    cells = df.copy()

    neighborhood_name = "neighborhood"+str(k)

    keep_cols = [X ,Y ,reg,cluster_col]

    n_neighbors = max(ks)

    cells[reg] = cells[reg].astype('str')

    #Get each region
    tissue_group = cells[[X,Y,reg]].groupby(reg)
    exps = list(cells[reg].unique())
    tissue_chunks = [(time.time(),exps.index(t),t,a) for t,indices in tissue_group.groups.items() for a in np.array_split(indices,1)] 

    tissues = [get_windows(job, n_neighbors, exps= exps, tissue_group = tissue_group, X = X, Y = Y) for job in tissue_chunks]

    #Loop over k to compute neighborhoods
    out_dict = {}
    for k in ks:
        for neighbors,job in zip(tissues,tissue_chunks):

            chunk = np.arange(len(neighbors))#indices
            tissue_name = job[2]
            indices = job[3]
            window = values[neighbors[chunk,:k].flatten()].reshape(len(chunk),k,len(sum_cols)).sum(axis = 1)
            out_dict[(tissue_name,k)] = (window.astype(np.float16),indices)
            
    windows = {}
    for k in ks:
    
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

    #modify figure size aesthetics for each neighborhood
    figs = catplot(cells,X = X,Y=Y,exp = reg,hue = 'neighborhood'+str(k),invert_y=True,size = 5,)
    if save_to_csv is True:
        cells.to_csv(output_dir + 'neighborhood.csv')
        
    else: 
        print("results will not be stored as csv file")

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
    
    return(cells)


##########################################################################################################

def xycorr(df, sample_col, y_rows, x_columns, X_pix, Y_pix):
    
    #Make a copy for xy correction
    df_XYcorr = df.copy()
    
    df_XYcorr["Xcorr"] = 0
    df_XYcorr["Ycorr"] = 0
    
    for sample in df_XYcorr[sample_col].unique():
        df_sub = df_XYcorr.loc[df_XYcorr[sample_col]==sample]
        region_num = df_sub.region.max().astype(int)

        #first value of tuple is y and second is x
        d = list(product(range(0,y_rows,1),range(0,x_columns,1)))
        e = list(range(1,region_num+1,1))
        dict_corr = {}
        dict_corr = dict(zip(e, d)) 

        #Adding the pixels with the dictionary
        for x in range(1,region_num+1,1):
            df_XYcorr["Xcorr"].loc[(df_XYcorr["region"]== x)&(df_XYcorr[sample_col]==sample)] = df_XYcorr['x'].loc[(df_XYcorr['region']==x)&(df_XYcorr[sample_col]==sample)] +dict_corr[x][1]*X_pix

        for x in range(1,region_num+1,1):
            df_XYcorr["Ycorr"].loc[(df_XYcorr["region"]== x)&(df_XYcorr[sample_col]==sample)] = df_XYcorr['y'].loc[(df_XYcorr['region']==x)&(df_XYcorr[sample_col]==sample)] +dict_corr[x][0]*Y_pix

    return df_XYcorr


##########################################################################################################

'''
data: Pandas data frame which is used as input for plotting.


group1: Categorical column in data that will be used as the x-axis in the pairplot.

per_cat: Categorical column in data that will be used to calculate the correlation between categories in group1.

sub_col (optional): Categorical column in data that is used to subset the data.

sub_list (optional): List of values that is used to select a subset of data based on the sub_col.

norm (optional): Boolean that determines if the data should be normalized or not.

group2 (optional): Categorical column in data that is used to group the data.

count (optional): Boolean that determines if the count of each category in per_cat should be used instead of the percentage.

plot_scatter (optional): Boolean that determines if the scatterplot should be plotted or not.

cor_mat: Output data frame containing the correlation matrix.

mp: Output data frame containing the pivot table of the count or percentage of each category in per_cat based on group1.


Returns:
cor_mat (pandas dataframe): Correlation matrix.
mp (pandas dataframe): Data after pivoting and grouping.
'''

def cor_plot(data, group1,per_cat, sub_col=None,sub_list=None,norm=False,\
             group2=None, count=False, plot_scatter=True):
    if group2:
        plt.rcParams["legend.markerscale"] = 1
        tf = data.groupby([group1,group2]).apply(lambda x: x[per_cat].value_counts(normalize = True,sort = False)*100).to_frame()
        tf.columns = tf.columns.astype(str)
        tf.reset_index(inplace=True)
        mp = pd.pivot_table(tf, columns = ['level_2'], index=[group1,group2], values=[per_cat])
        mp.columns = mp.columns.droplevel(0)
        mp.reset_index(inplace=True)
        mp2 = mp.fillna(0)
        cor_mat = mp2.corr()
        mask = np.triu(np.ones_like(cor_mat, dtype=bool))
        plt.figure(figsize = (len(cor_mat.index),len(cor_mat.columns)*0.8))
        sns.heatmap(cor_mat, cmap='coolwarm',center=0,vmin=-1,vmax=1,mask=mask)
        if plot_scatter:
            sns.pairplot(mp,diag_kind = 'kde',
                     plot_kws ={'alpha': 0.6, 's': 80, 'edgecolor': 'k'},
                     size = 4, hue=group2)
    else:
        if count:
                tf = data.groupby([group1,per_cat]).count()['region'].to_frame()
                tf.reset_index(inplace=True)
                mp = pd.pivot_table(tf, columns = [per_cat], index=[group1], values=['region'])
                mp.columns = mp.columns.droplevel(0)
                mp.reset_index(inplace=True)
                mp2 = mp.fillna(0)
                cor_mat = mp2.corr()
                mask = np.triu(np.ones_like(cor_mat, dtype=bool))
                plt.figure(figsize = (len(cor_mat.index),len(cor_mat.columns)*0.8))
                sns.heatmap(cor_mat, cmap='coolwarm',center=0,vmin=-1,vmax=1,mask=mask)
                if plot_scatter:
                    sns.pairplot(mp,diag_kind = 'kde',
                                 plot_kws = {'scatter_kws':{'alpha': 0.6, 's': 80, 'edgecolor': 'k'}},
                                 size = 4, kind='reg')
        else:
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
            tf = test1.groupby([group1]).apply(lambda x: x[per_cat].value_counts(normalize = True,sort = False)*100)
            tf.columns = tf.columns.astype(str)
            mp = tf[immune_list]
            mp.reset_index(inplace=True)
            cor_mat = mp.corr()
            mask = np.triu(np.ones_like(cor_mat, dtype=bool))
            plt.figure(figsize = (len(cor_mat.index),len(cor_mat.columns)*0.8))
            sns.heatmap(cor_mat, cmap='coolwarm',center=0,vmin=-1,vmax=1,mask=mask)
            if plot_scatter:
                sns.pairplot(mp,diag_kind = 'kde',
                             plot_kws = {'scatter_kws':{'alpha': 0.6, 's': 80, 'edgecolor': 'k'}},
                             size = 4, kind='reg')

        
    return cor_mat, mp


##########################################################################################################

def cor_subset(cor_mat, threshold, cell_type):
    pairs = get_top_abs_correlations(cor_mat,thresh=threshold)
    
    piar1 = pairs.loc[pairs['col1']==cell_type]
    piar2 = pairs.loc[pairs['col2']==cell_type]
    piar=pd.concat([piar1,piar2])
    
    pair_list = list(set(list(piar['col1'].unique())+list(piar['col2'].unique())))
    
    return pair_list, piar, pairs


##########################################################################################################

"""
mp: A pandas dataframe from which a subset of columns will be selected and plotted.
sub_list: A list of column names from the dataframe mp that will be selected and plotted.
save_name (optional): A string that specifies the file name for saving the plot. 
If save_name is not provided, the plot will not be saved.
"""
#def cor_subplot(mp, sub_list,save_name=None):
 #   sub_cor = mp[sub_list]
 #   sns.pairplot(sub_cor,diag_kind = 'kde',
  #                           plot_kws = {'scatter_kws':{'alpha': 0.6, 's': 80, 'edgecolor': 'k'}},
  #                           size = 4, kind='reg', corner=True)
 #   if save_name:
    #    plt.savefig(output_filepath+save_name+'_corrplot.png', format='png', dpi=300, transparent=True, bbox_inches='tight')
#

##########################################################################################################

"""
data: the input pandas data frame.
sub_l2: a list of subcategories to be considered.
per_categ: the categorical column in the data frame to be used.
group2: the grouping column in the data frame.
repl: the replicate column in the data frame.
sub_collumn: the subcategory column in the data frame.
cell: the cell type column in the data frame.
thres (optional): the threshold for the correlation, default is 0.9.
normed (optional): if the percentage should be normalized, default is True.
cell2 (optional): the second cell type column in the data frame.
"""
def corr_cell(data,  sub_l2, per_categ, group2, repl, sub_collumn, cell,\
              output_dir, save_name, thres = 0.9, normed=True, cell2=None):
    result = per_only1(data = data, per_cat = per_categ, grouping=group2,\
                      sub_list=sub_l2, replicate=repl, sub_col = sub_collumn, norm=normed)

    #Format for correlation function
    mp = pd.pivot_table(result, columns = [per_categ], index=[group2,repl], values=['percentage'])
    mp.columns = mp.columns.droplevel(0)
    cc = mp.reset_index()
    cmat = cc.corr()

    #Plot
    sl2, pair2, all_pairs = cor_subset(cor_mat=cmat, threshold = thres, cell_type=cell)
    
    if cell2:
        sl3 = [cell2, cell]
        cor_subplot(mp=cc, sub_list=sl3, output_dir = output_dir, save_name=cell+'_'+cell2)
    else:
        cor_subplot(mp=cc, sub_list=sl2, output_dir = output_dir, save_name=cell)
        
    if save_name:
        plt.savefig(output_dir+save_name+'.png', format='png',\
                    dpi=300, transparent=True, bbox_inches='tight')
    
    return all_pairs, pair2

##########################################################################################################

def cor_subplot(mp, sub_list, output_dir, save_name=None):
    sub_cor = mp[sub_list]
    sns.pairplot(sub_cor,diag_kind = 'kde',
                             plot_kws = {'scatter_kws':{'alpha': 0.6, 's': 80, 'edgecolor': 'k'}},
                             size = 4, kind='reg', corner=True)
    if save_name:
        plt.savefig(output_dir+save_name+'_corrplot.png', format='png', dpi=300, transparent=True, bbox_inches='tight')

##########################################################################################################
# Cell type differential enrichment 
def normalize(X):
    arr = np.array(X.fillna(0).values)
    return pd.DataFrame(np.log2(1e-3 + arr/arr.sum(axis =1, keepdims = True)), index = X.index.values, columns = X.columns).fillna(0)


def cell_types_de_helper(df, ID_component1, ID_component2, neighborhood_col, group_col, group_dict, cell_type_col):
    
    # read data 
    cells2 = df
    cells2.reset_index(inplace=True, drop=True)
    cells2
    
    # generate unique ID
    cells2['donor_tis'] = cells2[ID_component1]+'_'+cells2[ID_component2]
    
    # This code is creating a dictionary called neigh_num that maps each unique value 
    #in the Neighborhood column of a pandas DataFrame cells2 to a unique integer index 
    #starting from 0.
    neigh_num = {list(cells2[neighborhood_col].unique())[i]:i for i in range(len(cells2[neighborhood_col].unique()))}
    cells2['neigh_num'] = cells2[neighborhood_col].map(neigh_num)
    cells2['neigh_num'].unique()
    
    '''
    This Python code is performing the following data transformation operations on a pandas DataFrame named cells2:
    The first three lines of code create a dictionary called treatment_dict that maps two specific strings, 'SB' and 'CL', to the integers 0 and 1, respectively. Then, the map() method is used to create a new column called group, where each value in the tissue column is replaced with its corresponding integer value from the treatment_dict dictionary.
    The fourth to seventh lines of code create a new dictionary called pat_dict that maps each unique value in the donor_tis column of the cells2 DataFrame to a unique integer index starting from 0. The for loop loops through the range object and assigns each integer to the corresponding unique value in the donor_tis column, creating a dictionary that maps each unique value to a unique integer index.
    The last two lines of code create a new column called patients in the cells2 DataFrame, where each value in the donor_tis column is replaced with its corresponding integer index from the pat_dict dictionary. This code assigns these integer indices to each patient in the donor_tis column. The unique() method is used to return an array of unique values in the patients column to verify that each unique value in the donor_tis column has been mapped to a unique integer index in the patients column.
    Overall, the code is converting categorical data in the tissue and donor_tis columns to numerical data in the group and patients columns, respectively, which could be useful for certain types of analysis.
    '''
    # Code treatment/group with number
    cells2['group']=cells2[group_col].map(group_dict)
    cells2['group'].unique()
    
    pat_dict = {}
    for i in range(len(list(cells2['donor_tis'].unique()))):
        pat_dict[list(cells2['donor_tis'].unique())[i]] = i
    pat_dict
    
    cells2['patients']=cells2['donor_tis'].map(pat_dict)
    cells2['patients'].unique()
    
    # drop duplicates 
    pat_gp = cells2[['patients','group']].drop_duplicates()
    pat_to_gp= {a:b for a,b in pat_gp.values}
    
    # get cell type (ct) frequences per patient 
    ct_freq1 = cells2.groupby(['patients']).apply(lambda x: x[cell_type_col].value_counts(normalize = True,sort = False)*100)
    #ct_freq = ct_freq1.to_frame()
    ct_freq = ct_freq1.unstack().fillna(0)
    ct_freq.reset_index(inplace=True)
    ct_freq.rename(columns={'level_1':'cell_type', 'Cell Type':'Percentage'}, inplace=True)
    ct_freq
    
    # Get frequences for every neighborhood per patient 
    all_freqs1 = cells2.groupby(['patients','neigh_num']).apply(lambda x: x[cell_type_col].value_counts(normalize = True,sort = False)*100)
    #all_freqs = all_freqs1.to_frame()
    all_freqs = all_freqs1.unstack().fillna(0)
    all_freqs.reset_index(inplace=True)
    all_freqs.rename(columns={'level_2':'cell_type', cell_type_col:'Percentage'}, inplace=True)
    all_freqs
    
    return(cells2, ct_freq, all_freqs, pat_to_gp, neigh_num)


def cell_types_de(ct_freq, all_freqs, neighborhood_num, nbs, patients, group, cells, cells1, neigh_num, output_dir):
    
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

########################################################################################################## Cell_distance 
def get_distances(df, cell_list, cell_type_col):
    names = cell_list
    cls = {}
    for i,cname in enumerate(names):
        cls[i] = df[["x","y"]][df[cell_type_col]==cname].to_numpy()
        cls[i] = cls[i][~np.isnan(cls[i]).any(axis=1), :]

    dists = {}

    for i in range(5):
        for j in range(0,i):
            dists[(j,i)] = (cdist(cls[j], cls[i]))
            dists[(i,j)] = dists[(j,i)]
    return cls, dists    

########################################################################################################## Community analysis 



def community_analysis(df, values, sum_cols, output_dir, X = 'x', Y = 'y', reg = 'unique_region', cluster_col = 'neigh_name', ks = [100], save_path = None, k = 100, n_neighborhoods = 30, plot_specific_community = None):
    
    output_dir2 = output_dir+"community_analysis/"
    if not os.path.exists(output_dir2):
        os.makedirs(output_dir2)
    
    cells = df.copy()

    neighborhood_name = "community"+str(k)

    keep_cols = [X ,Y ,reg,cluster_col]

    n_neighbors = max(ks)

    cells[reg] = cells[reg].astype('str')

    #Get each region
    tissue_group = cells[[X,Y,reg]].groupby(reg)
    exps = list(cells[reg].unique())
    tissue_chunks = [(time.time(),exps.index(t),t,a) for t,indices in tissue_group.groups.items() for a in np.array_split(indices,1)] 

    tissues = [get_windows(job, n_neighbors, exps= exps, tissue_group = tissue_group, X = X, Y = Y) for job in tissue_chunks]

    #Loop over k to compute neighborhoods
    out_dict = {}
    for k in ks:
        for neighbors,job in zip(tissues,tissue_chunks):

            chunk = np.arange(len(neighbors))#indices
            tissue_name = job[2]
            indices = job[3]
            window = values[neighbors[chunk,:k].flatten()].reshape(len(chunk),k,len(sum_cols)).sum(axis = 1)
            out_dict[(tissue_name,k)] = (window.astype(np.float16),indices)
            
    windows = {}
    for k in ks:
    
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
    
    return(cells)

##########################################################################################################

def annotate_communities(df, community_column, annotations):
    df['community']=df['community100'].map(annotations)
    print(df['community'].unique())
    return(df)

##########################################################################################################
# Helper Functions
##########################################################################################################

def get_pathcells(query_database, query_dict_list):
    '''
    Return set of cells that match query_dict path.
    '''
    out = []
    
    if type(query_dict_list) == dict:
        query_dict_list = [query_dict_list]
    
    
    for query_dict in query_dict_list:
        qd = query_database
        for k,v in query_dict.items():
            if type(v)!=list:
                v = [v]
            qd = qd[qd[k].isin(v)]
        out+=[qd]
    if len(query_database)==1:
        return out[0]
    return out

# annotated 
'''
def get_pathcells(query_database: Union[Dict, List[Dict]], query_dict_list: List[Dict]) -> Union[Dict, List[Dict]]:
    
    #Return set of cells that match query_dict path.
    
    out: List[Dict] = []   # initialize an empty list to store results
    
    if type(query_dict_list) == dict:    # if query_dict_list is a dictionary, convert it into a list
        query_dict_list = [query_dict_list]
        
    for query_dict in query_dict_list:    # loop through each dictionary in query_dict_list
        qd = query_database   # initialize a reference to query_database
        for k,v in query_dict.items():   # loop through each key-value pair in the current dictionary
            if type(v)!=list:   # if the value is not a list, convert it into a list
                v = [v]
            qd = qd[qd[k].isin(v)]   # filter the rows of qd based on the key-value pair
        out+=[qd]   # append the resulting qd to the out list
        
    if len(query_database)==1:    # if query_database contains only one row, return the first item in out
        return out[0]
    return out    # otherwise, return the entire out list
'''

class Neighborhoods(object):
    def __init__(self, cells,ks,cluster_col,sum_cols,keep_cols,X='X:X',Y = 'Y:Y',reg = 'Exp',add_dummies = True):
        self.cells_nodumz = cells
        self.X = X
        self.Y = Y
        self.reg = reg
        self.keep_cols = keep_cols
        self.sum_cols = sum_cols
        self.ks = ks
        self.cluster_col = cluster_col
        self.n_neighbors = max(ks)
        self.exps = list(self.cells_nodumz[self.reg].unique())
        self.bool_add_dummies = add_dummies
        
    def add_dummies(self):
        
        c = self.cells_nodumz
        dumz = pd.get_dummies(c[self.cluster_col])
        keep = c[self.keep_cols]
        
        self.cells = pd.concat([keep,dumz],1)
        
        
        
    def get_tissue_chunks(self):
        self.tissue_group = self.cells[[self.X,self.Y,self.reg]].groupby(self.reg)
        
        tissue_chunks = [(time.time(),self.exps.index(t),t,a) for t,indices in self.tissue_group.groups.items() for a in np.array_split(indices,1)] 
        return tissue_chunks
    
    def make_windows(self,job):
        

        start_time,idx,tissue_name,indices = job
        job_start = time.time()

        print ("Starting:", str(idx+1)+'/'+str(len(self.exps)),': ' + self.exps[idx])

        tissue = self.tissue_group.get_group(tissue_name)
        to_fit = tissue.loc[indices][[self.X,self.Y]].values

        fit = NearestNeighbors(n_neighbors=self.n_neighbors+1).fit(tissue[[self.X,self.Y]].values)
        m = fit.kneighbors(to_fit)
        m = m[0][:,1:], m[1][:,1:]


        #sort_neighbors
        args = m[0].argsort(axis = 1)
        add = np.arange(m[1].shape[0])*m[1].shape[1]
        sorted_indices = m[1].flatten()[args+add[:,None]]

        neighbors = tissue.index.values[sorted_indices]

        end_time = time.time()

        print ("Finishing:", str(idx+1)+"/"+str(len(self.exps)),": "+ self.exps[idx],end_time-job_start,end_time-start_time)
        return neighbors.astype(np.int32)
    
    def k_windows(self):
        if self.bool_add_dummies:
            self.add_dummies()
        else:
            self.cells =self.cells_nodumz
        sum_cols = list(self.sum_cols)
        for col in sum_cols:
            if col in self.keep_cols:
                self.cells[col+'_sum'] = self.cells[col]
                self.sum_cols.remove(col)
                self.sum_cols+=[col+'_sum']

        values = self.cells[self.sum_cols].values
        tissue_chunks = self.get_tissue_chunks()
        tissues = [self.make_windows(job) for job in tissue_chunks]
        
        out_dict = {}
        for k in self.ks:
            for neighbors,job in zip(tissues,tissue_chunks):

                chunk = np.arange(len(neighbors))#indices
                tissue_name = job[2]
                indices = job[3]
                window = values[neighbors[chunk,:k].flatten()].reshape(len(chunk),k,len(self.sum_cols)).sum(axis = 1)
                out_dict[(tissue_name,k)] = (window.astype(np.float16),indices)
        
        windows = {}
        for k in self.ks:

            window = pd.concat([pd.DataFrame(out_dict[(exp,k)][0],index = out_dict[(exp,k)][1].astype(int),columns = self.sum_cols) for exp in self.exps],0)
            window = window.loc[self.cells.index.values]
            window = pd.concat([self.cells[self.keep_cols],window],1)
            windows[k] = window
        return windows
    
    
##################

# helper function 
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
    

##################

# Define a Python function named `get_windows` that takes two arguments: `job` and `n_neighbors`.
def get_windows(job,n_neighbors, exps, tissue_group, X, Y):
    
    # Unpack the tuple `job` into four variables: `start_time`, `idx`, `tissue_name`, and `indices`.
    start_time,idx,tissue_name,indices = job
    
    # Record the time at which the function starts.
    job_start = time.time()
    
    # Print a message indicating the start of the function execution, including the current job index and the corresponding experiment name.
    print ("Starting:", str(idx+1)+'/'+str(len(exps)),': ' + exps[idx])
    
    # Retrieve the subset of data that corresponds to the given `tissue_name`.
    tissue = tissue_group.get_group(tissue_name)
    
    # Select only the `X` and `Y` columns of the data corresponding to the given `indices`.
    to_fit = tissue.loc[indices][[X,Y]].values
    
    # Fit a model with the data that corresponds to the `X` and `Y` columns of the given `tissue`.
    fit = NearestNeighbors(n_neighbors=n_neighbors).fit(tissue[[X,Y]].values)
    
    # Find the indices of the `n_neighbors` nearest neighbors of the `to_fit` data points.
    # The `m` variable contains the distances and indices of these neighbors.
    m = fit.kneighbors(to_fit)
    
    # Sort the `m[1]` array along each row, and store the resulting indices in the `args` variable.
    args = m[0].argsort(axis = 1)
    
    # Create the `add` variable to offset the indices based on the number of rows in `m[1]`, and store the sorted indices in the `sorted_indices` variable.
    add = np.arange(m[1].shape[0])*m[1].shape[1]
    sorted_indices = m[1].flatten()[args+add[:,None]]
    
    # Create the `neighbors` variable by selecting the indices from the `tissue` data frame that correspond to the sorted indices.
    neighbors = tissue.index.values[sorted_indices]
    
    # Record the time at which the function ends.
    end_time = time.time()
    
    # Print a message indicating the end of the function execution, including the current job index and the corresponding experiment name, and the time it took to execute the function.
    print ("Finishing:", str(idx+1)+"/"+str(len(exps)),": "+ exps[idx],end_time-job_start,end_time-start_time)
    
    # Return the `neighbors` array as an array of 32-bit integers.
    return neighbors.astype(np.int32)


###################

def index_rank(a,axis):
    '''
    returns the index of every index in the sorted array
    haven't tested on ndarray yet
    '''
    arg =np.argsort(a,axis)
    
    return np.arange(a.shape[axis])[np.argsort(arg,axis)]

def znormalize (raw_cells,grouper,markers,clip = (-7,7),dropinf = True):
    not_inf = raw_cells[np.isinf(raw_cells[markers].values).sum(axis = 1)==0]
    if not_inf.shape[0]!=raw_cells.shape[0]:
        print ('removing cells with inf values' ,raw_cells.shape,not_inf.shape)
    not_na = not_inf[not_inf[markers].isnull().sum(axis = 1)==0]
    if not_na.shape[0]!=not_inf.shape[0]:
        print ('removing cells with nan values', not_inf.shape,not_na.shape)
    
    
    znorm = not_na.groupby(grouper).apply(lambda x: ((x[markers]-x[markers].mean(axis = 0))/x[markers].std(axis = 0)).clip(clip[0],clip[1]))
    Z = not_na.drop(markers,1).merge(znorm,left_index = True,right_index = True)
    return Z

def fast_divisive_cluster(X,num_clusters,metric = 'cosine',prints = True):
    
    #optimized divisive_cluster.  Faster because doesn't recompute distance matrix to centroids at 
    #each iteration
    centroids = np.zeros((num_clusters,X.shape[1])) # fill with cluster centroids
    dists = np.zeros((X.shape[0],num_clusters))  # fill with dist matrix

    avg_seed = X.mean(axis = 0,keepdims = True)
    d = cdist(X,avg_seed,metric = metric)
    dists[:,0] = d[:,0]
    c1 = d.argmax()
    centroids[0] = X[c1]

    for x in range(1,num_clusters):
        if x%10==0:
            print (x, 'clusters')
        d = cdist(X,centroids[x-1][None,:],metric = metric)
        dists[:,x] = d[:,0]
        allocs = dists[:,:x+1].argmin(axis = 1)
        next_centroid = dists[np.arange(len(dists)),allocs].argmax()
        centroids[x] = X[next_centroid]
    return centroids,allocs

def alloc_cells(X,centroids,metric = 'cosine'):
    dists = cdist(X,centroids,metric = metric)
    allocs = dists.argmin(axis = 1)
    return allocs
    

def conplot(df,feature,exp = 'Exp',X = 'X',Y = 'Y',invert_y = False,cmap = "RdBu",size = 5,alpha = 1 ,figsize = 10, exps = None,fig = None ,**kwargs):
    '''
    Plot continuous variable with a colormap:
    
    df:  dataframe of cells with spatial location and feature to color.  Must have columns ['X','Y','Exp',feature]
    feature:  feature in df to color points by
    cmap:  matplotlib colormap
    size:  point size
    thresh_val: only include points below this value
    '''
    if invert_y:
        y_orig = df[Y].values.copy()
        df[Y]*=-1
        
    if exps is None:
        exps = list(df[exp].unique()) #display all experiments
    elif type(exps)!= list:
        exps = [exps]

    if fig is None:
        f,ax = plt.subplots(len(exps),1,figsize = (figsize,len(exps)*figsize))
        if len(exps) == 1:
            ax = [ax]
    else:
        f,ax = fig
    
    for i,name in enumerate(exps):
        data = df[df[exp] == name]
    
        ax[i].scatter(data[X],-data[Y],c = data[feature],cmap = cmap,s = size,alpha = alpha,**kwargs)
        ax[i].set_title(name + "_" + str(feature)+"_"+str(len(data)))
        ax[i].axis('off')
                        
    
    if invert_y:
        df[Y] = y_orig
    return f,ax 

def get_sum_cols(cell_cuts,panel):
    arr = np.where(cell_cuts[:,0]==panel)[0]
    return slice(arr[0],arr[-1]+1)

def catplot(df,hue,exp = 'Exp',X = 'X',Y = 'Y',invert_y = False,size = 3,legend = True, palette="bright",figsize = 5,style = 'white',exps = None,axis = 'on',scatter_kws = {}):
    '''
    Plots cells in tissue section color coded by either cell type or node allocation.
    df:  dataframe with cell information
    size:  size of point to plot for each cell.
    hue:  color by "Clusterid" or "Node" respectively.
    legend:  to include legend in plot.
    '''
    scatter_kws_ = {'s':size,'alpha':1}
    scatter_kws_.update(scatter_kws)
    
    
    figures = []
    df = df.rename(columns = lambda x: str(x))
    
    df[hue] = df[hue].astype("category")
    if invert_y:
        y_orig = df[Y].values.copy()
        df[Y]*=-1


    style = {'axes.facecolor': style}
    sns.set_style(style)
    if exps == None:
        exps = list(df[exp].unique()) #display all experiments
    elif type(exps)!= list:
        exps = [exps]

    for name in exps:
        data = df[df[exp] == name]
        
        print (name)
        f = sns.lmplot(x = X,y = Y,data = data,hue = hue,
                   legend = legend,fit_reg = False,markers = '.',height = figsize, palette=palette,scatter = True,scatter_kws = scatter_kws_)
        
        if axis =='off':
            sns.despine(top=True, right=True, left=True, bottom=True)
            f = f.set(xticks = [],yticks=[]).set_xlabels('').set_ylabels('')
       
        
        

        plt.title(name)


        plt.show()
        figures +=[f] 
    if invert_y:
        df[Y] = y_orig
    
    return figures

##########

def prepare_neighborhood_df(cells_df, patient_ID_component1, patient_ID_component2, neighborhood_column = None):
    # Spacer for output 
    print("")
    
    # Combine two columns to form unique ID which will be stored as patients column 
    cells_df['patients'] = cells_df[patient_ID_component1]+'_'+cells_df[patient_ID_component2]
    print("You assigned following identifiers to the column 'patients':")
    print(cells_df['patients'].unique())
    
    # Spacer for output 
    print("")
    
    if neighborhood_column == True :
    # Assign numbers to neighborhoods
        neigh_num = {list(cells_df[neighborhood_column].unique())[i]:i for i in range(len(cells_df[neighborhood_column].unique()))}
        cells_df['neigh_num'] = cells_df[neighborhood_column].map(neigh_num)
        print("You assigned following numbers to the column 'neigh_num'. Each number represents one neighborhood:")
        print(cells_df['neigh_num'].unique())
        cells_df['neigh_num'] = cells_df['neigh_num'].astype('category')
    
    cells_df['patients'] = cells_df['patients'].astype('category')
    
  
    
    return(cells_df)

##########################################################################################################
# correlation analysis 
def get_redundant_pairs(df):
    '''Get diagonal and lower triangular pairs of correlation matrix'''
    pairs_to_drop = set()
    cols = df.columns
    for i in range(0, df.shape[1]):
        for j in range(0, i+1):
            pairs_to_drop.add((cols[i], cols[j]))
    return pairs_to_drop

#############

def get_top_abs_correlations(df, thresh=0.5):
    au_corr = df.corr().unstack()
    labels_to_drop = get_redundant_pairs(df)
    au_corr = au_corr.drop(labels=labels_to_drop).sort_values(ascending=False)
    cc = au_corr.to_frame()
    cc.index.rename(['col1','col2'],inplace=True)
    cc.reset_index(inplace=True)
    cc.rename(columns={0:'value'},inplace=True)
    gt_pair = cc.loc[cc['value'].abs().gt(thresh)]
    return gt_pair

##########################################################################################################
# CCA Analysis 


def Perform_CCA(cca, n_perms, nsctf, cns, subsets, group):
    stats_group1 = {}
    for cn_i in cns:
        for cn_j in cns:
            if cn_i < cn_j:
                print(cn_i, cn_j)
                #concat dfs
                combined = pd.concat([nsctf.loc[cn_i].loc[nsctf.loc[cn_i].index.isin(group)],nsctf.loc[cn_j].loc[nsctf.loc[cn_j].index.isin(group)]], axis = 1).dropna(axis = 0, how = 'any')
                if combined.shape[0]>2:
                    x = combined.iloc[:,:len(subsets)].values
                    y = combined.iloc[:,len(subsets):].values

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

            
def Visulize_CCA_results(CCA_results, save_path, save_fig = False, p_thresh = 0.1, save_name = "CCA_vis.png"):
    # Visualization of CCA 
    g1 = nx.petersen_graph()
    for cn_pair, cc in CCA_results.items():
        
        s,t = cn_pair
        obs, perms = cc
        p =np.mean(obs>perms)
        if p>p_thresh :
            g1.add_edge(s,t, weight = p)
 
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
        else:
            p = g1.get_edge_data(0, 1, default =0)['weight']
        plt.plot([pos[e0][0],pos[e1][0]],[pos[e0][1],pos[e1][1]], c= 'black',alpha = 3*p**1,linewidth = 3*p**3)

    if save_fig == True:
        plt.savefig(save_path + "/" + save_name, format='png', dpi=300, transparent=True, bbox_inches='tight')
    
              
    


##########################################################################################################
# tensor decomposition 

def evaluate_ranks(dat, num_tissue_modules = 2):
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

#######

def plot_modules_heatmap(dat, cns, cts, figsize = (20,5), num_tissue_modules = 2, num_cn_modules = 5):
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

def plot_modules_graphical(dat, cts, cns, num_tissue_modules = 2, num_cn_modules = 4, scale = 0.4, figsize = (1.5, 0.8), pal=None,save_name=None, save_path = None):
    core, factors = non_negative_tucker(dat,rank=[num_tissue_modules,num_cn_modules,num_cn_modules],random_state = 32)
    
    if pal is None:
        pal = sns.color_palette('bright',10)
    palg = sns.color_palette('Greys',10)
    
    #figsize(3.67*scale,2.00*scale)
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
        
#######

def build_tensors(df, group, cns, cts):
    
    counts = df.groupby(['patients','neigh_num','Coarse Cell']).size()
    
    #initialize the tensors
    T1 = np.zeros((len(group),len(cns),len(cts)))
    
    for i,pat in enumerate(group):
        for j,cn in enumerate(cns):
            for k,ct in enumerate(cts):
                T1[i,j,k] = counts.loc[(pat,cn,ct)]

    #normalize so we have joint distributions each slice
    dat1 =np.nan_to_num(T1/T1.sum((1,2), keepdims = True))
    
    return(dat1)
 

##########################################################################################################
# 'simple' distance analysis (Cell distance)

def get_distances(df, cell_list, celltype_column):
    names = cell_list
    cls = {}
    for i,cname in enumerate(names):
        cls[i] = df[["x","y"]][df[celltype_column]==cname].to_numpy()
        cls[i] = cls[i][~np.isnan(cls[i]).any(axis=1), :]

    dists = {}

    for i in range(5):
        for j in range(0,i):
            dists[(j,i)] = (cdist(cls[j], cls[i]))
            dists[(i,j)] = dists[(j,i)]
    return cls, dists

#cls, dists = get_distances(df_sub)


##########################################################################################################
# Spatial context analysis 
'''
this is the code that finds the minimal combination of CNs
required to make up a threshold percentage of assignments in a window
combinations are stored as a sorted tuple
'''
def get_thresh_simps(x,thresh):
    sorts = np.argsort(-x, axis = 1)
    x_sorted = -np.sort(-x, axis = 1)
    cumsums = np.cumsum(x_sorted,axis = 1)
    thresh_simps = pd.Series([tuple(sorted(sorts[i,:(1+j)])) for i,j in enumerate(np.argmax(cumsums>thresh,axis = 1))])
    return thresh_simps

#######

def get_network(ttl_per_thres, comb_per_thres, color_dic, windows, n_num, l, tissue_col = None, tissue_subset_list = None, sub_col='Tissue Unit',\
                neigh_sub=None, save_name=None, save_path = None):
    
    plt.figure(figsize=(20,10))
    #Choose the windows size to continue with
    w = windows[n_num]
    if tissue_col == True:
        w = w[w.tissue_col.isin(tissue_subset_list)]
    if neigh_sub:
        w = w[w[sub_col].isin(neigh_sub)]
    xm = w.loc[:,l].values/n_num
          
    # Get the neighborhood combinations based on the threshold
    simps = get_thresh_simps(xm,ttl_per_thres)
    simp_freqs = simps.value_counts(normalize = True)
    simp_sums = np.cumsum(simp_freqs)
          
    g = nx.DiGraph()
    thresh_cumulative = .95
    thresh_freq = comb_per_thres
    #selected_simps = simp_sums[simp_sums<=thresh_cumulative].index.values
    selected_simps = simp_freqs[simp_freqs>=thresh_freq].index.values
          
    #this builds the graph for the CN combination map
    selected_simps
    for e0 in selected_simps:
        for e1 in selected_simps:
            if (set(list(e0))<set(list(e1))) and (len(e1) == len(e0)+1):
                g.add_edge(e0,e1)   

    #this plots the CN combination map

    draw = g
    pos = nx.drawing.nx_pydot.graphviz_layout(draw, prog='dot')
    height = 8

    plt.figure(figsize=(20,10))
    for n in draw.nodes():
        col = 'black'
        if len(draw.in_edges(n))<len(n):
            col = 'black'
        plt.scatter(pos[n][0],pos[n][1]-5, s = simp_freqs[list(simp_freqs.index).index(n)]*10000, c = col, zorder = -1)
#         if n in tops:
#             plt.text(pos[n][0],pos[n][1]-7, '*', fontsize = 25, color = 'white', ha = 'center', va = 'center',zorder = 20)
        delta = 8
        #plot_sim((pos[n][0]+delta, pos[n][1]+delta),n, scale = 20,s = 200,text = True,fontsize = 15)
        plt.scatter([pos[n][0]]*len(n),[pos[n][1]+delta*(i+1) for i in range(len(n))],c = [color_dic[l[i]] for i in n] ,marker = '^', zorder = 5,s = 400)

    j = 0
    for e0,e1 in draw.edges():
        weight = 0.2
        alpha = .3
        if len(draw.in_edges(e1))<len(e1):
            color = 'black'
            lw =1
            weight = 0.4
        color='black'
        plt.plot([pos[e0][0], pos[e1][0]],[pos[e0][1], pos[e1][1]], color = color, linewidth = weight,alpha = alpha,zorder = -10)

    plt.axis('off')
    if save_name is not None:
        plt.savefig(save_path+save_name+'_spatial_contexts.pdf')#'.png', dpi=300)
    plt.show()


#######

def simp_rep(data, patient_col, subset_col, subset_list, ttl_per_thres, comb_per_thres, l, n_num, thres_num = 3):
    
    #Choose the windows size to continue with
    w2 = data.loc[data[subset_col].isin(subset_list)]
    
    simp_list = []
    for patient in list(w2[patient_col].unique()):
        w = w2.loc[w2[patient_col]==patient]
        xm = w.loc[:,l].values/n_num

        # Get the neighborhood combinations based on the threshold
        simps = get_thresh_simps(xm,ttl_per_thres)
        simp_freqs = simps.value_counts(normalize = True)
        sf = simp_freqs.to_frame()
        sf.rename(columns={0:patient},inplace=True)
        sf.reset_index(inplace=True)
        sf.rename(columns={'index':'merge'},inplace=True)
        simp_list.append(sf)
        #simp_sums = np.cumsum(simp_freqs)

        #thresh_cumulative = .95
        #selected_simps = simp_sums[simp_sums<=thresh_cumulative].index.values
       # selected_simps = simp_freqs[simp_freqs>=comb_per_thres].index.values
    
    simp_df = reduce(lambda  left,right: pd.merge(left,right,on=['merge'],
                                            how='outer'), simp_list)
    #simp_df = pd.concat(simp_list, axis=0)
    #simp_df.index = simp_df.index.to_series()
    simp_df.fillna(0,inplace=True)
    simp_df.set_index('merge', inplace=True)
    simp_out = simp_df.loc[simp_df.gt(0).sum(axis=1).ge(thres_num)]

    return simp_out

#######

def comb_num_freq(data_list):
    df_new = []
    for df in data_list:
        df.reset_index(inplace=True)
        df.rename(columns={'merge':'combination'},inplace=True)
        df['count'] = df['combination'].apply(len)
        sum_df = df.groupby('count').sum()
        
        tbt = sum_df.reset_index()
        ttt = tbt.melt(id_vars = ['count'])
        ttt.rename(columns={'variable':'unique_cond','value':'fraction'}, inplace=True)
        df_new.append(ttt)
    df_exp = pd.concat(df_new)
    
    df_exp[['donor', 'tissue']] = df_exp['unique_cond'].str.split('_',expand=True)
    
    
    #swarmplot to compare 
    plt.figure(figsize=(5,5))

    ax = sns.boxplot(data = df_exp, x='count',  y='fraction', hue = 'tissue', dodge=True, \
                     hue_order=plot_order, palette=pal_tis)
    ax = sns.swarmplot(data = df_exp, x='count', y='fraction', hue = 'tissue', dodge=True, \
                      hue_order=plot_order, edgecolor='black',linewidth=1, palette=pal_tis)
    for patch in ax.artists:
        r, g, b, a = patch.get_facecolor()
        patch.set_facecolor((r, g, b, .3))
    #ax.set_yscale(\log\)
    plt.xlabel('')
    handles, labels = ax.get_legend_handles_labels()
    plt.legend(handles[:len(df_exp['tissue'].unique())], labels[:len(df_exp['tissue'].unique())],\
               bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.,frameon=False)
    plt.xticks(rotation=90)
    sns.despine(trim=True)
    
    return df_exp

#######


def calculate_neigh_combs(w, l, n_num, threshold = 0.85, per_keep_thres = 0.85):
    w.loc[:,l]

    #need to normalize by number of neighborhoods or k chosen for the neighborhoods
    xm = w.loc[:,l].values/n_num


    # Get the neighborhood combinations based on the threshold
    simps = get_thresh_simps(xm, threshold)
    simp_freqs = simps.value_counts(normalize = True)
    simp_sums = np.cumsum(simp_freqs)


    #See the percent to keep threshold or percent of neigbhorhoods that fall above a certain threshold
    test_sums_thres =simp_sums[simp_sums < per_keep_thres]
    test_len = len(test_sums_thres)
    per_values_above = simp_sums[test_len]-simp_sums[test_len-1]
    print(test_len, per_values_above)


    w['combination'] = [tuple(l[a] for a in s) for s in simps]
    w['combination_num'] = [tuple(a for a in s) for s in simps]

    # this shows what proportion (y) of the total cells are assigned to the top x combinations
    plt.figure(figsize(20,5))
    plt.plot(simp_sums.values)
    plt.title("proportion (y) of the total cells are assigned to the top x combinations")
    plt.show()

    # this shows what proportion (y) of the total cells are assigned to the top x combinations
    plt.figure(figsize(20,5))
    plt.plot(test_sums_thres.values)
    plt.title("proportion (y) of the total cells are assigned to the top x combinations - thresholded")
    plt.show()
    #plt.xticks(range(0,350,35),range(0,350,35),rotation = 90,fontsize = 10)

    return(simps, simp_freqs, simp_sums)

#######


def build_graph_CN_comb_map(simp_freqs):
    g = nx.DiGraph()
    thresh_cumulative = .95
    thresh_freq = .001
    #selected_simps = simp_sums[simp_sums<=thresh_cumulative].index.values
    selected_simps = simp_freqs[simp_freqs>=thresh_freq].index.values
    selected_simps
    
    
    '''
    this builds the graph for the CN combination map
    '''
    for e0 in selected_simps:
        for e1 in selected_simps:
            if (set(list(e0))<set(list(e1))) and (len(e1) == len(e0)+1):
                g.add_edge(e0,e1)
                
    tops = simp_freqs[simp_freqs>=thresh_freq].sort_values(ascending = False).index.values.tolist()[:20]
    
    return(g, tops, e0, e1)

#######


def generate_CN_comb_map(graph, tops, e0, e1, simp_freqs, l, color_dic):
        
    draw = graph
    pos = nx.drawing.nx_pydot.graphviz_layout(draw, prog='dot')
    height = 8
    
    plt.figure(figsize(40,20))
    for n in draw.nodes():
        col = 'black'
        if len(draw.in_edges(n))<len(n):
            col = 'black'
        plt.scatter(pos[n][0],pos[n][1]-5, s = simp_freqs[list(simp_freqs.index).index(n)]*10000, c = col, zorder = -1)
        if n in tops:
            plt.text(pos[n][0],pos[n][1]-7, '*', fontsize = 25, color = 'white', ha = 'center', va = 'center',zorder = 20)
        delta = 8
        #plot_sim((pos[n][0]+delta, pos[n][1]+delta),n, scale = 20,s = 200,text = True,fontsize = 15)
        plt.scatter([pos[n][0]]*len(n),[pos[n][1]+delta*(i+1) for i in range(len(n))],c = [color_dic[l[i]] for i in n] ,marker = 's', zorder = 5,s = 400)
        
    #     #add profiles below node
    #     x = pos[n][0]
    #     y = pos[n][1]
    #     y = y-height*2
    #     standard_node_size =  16
    #     node_heights = [0,3,8,5,3,2,1,5]
    #     marker_colors = ['red','red','blue','blue','red','red','blue','blue']
        
    #     plt.plot([x+(18*(i-1.5)) for i in range(len(node_heights))],[(y-height*.9)+v for v in node_heights],c = 'red',zorder =3)#,s = v*2 ,c= c,edgecolors='black',lw = 1)
    #     plt.scatter([x+(18*(i-1.5)) for i in range(len(node_heights))],[(y-height*.9)+v for v in node_heights],c = marker_colors,s = standard_node_size,zorder = 4)
        
            
    j = 0
    for e0,e1 in draw.edges():
        weight = 0.2
        alpha = .3
        color='black'
        if len(draw.in_edges(e1))<len(e1):
            color = 'black'
            lw =1
            weight = 0.4
            
    #     if (e0,e1) in set(draw.out_edges(tuple(sorted([lmap['3'],lmap['1']])))):
    #         j+=1
    #         print(j)
    #         color = 'green'
    #         weight = 2
    #         alpha = 1
            
    #     if (lmap['3'] in e0) and (lmap['1'] not in e0) and (lmap['1'] in e1):
    #         color = 'green'
    #         weight = 2
    #         alpha = 1
    
        plt.plot([pos[e0][0], pos[e1][0]],[pos[e0][1], pos[e1][1]], color = color, linewidth = weight,alpha = alpha,zorder = -10)
    
    plt.axis('off')
    #plt.savefig('CNM.pdf')
    plt.show()

#def add_patient_IDs(df, ID_component1, ID_component2):
    # Spacer for output 
    
    # Combine two columns to form unique ID which will be stored as patients column 
#    df['patients'] = df[ID_component1]+'_'+df[ID_component2]
#    print("You assigned following identifiers to the column 'patients':")
#    print(df['patients'].unique())
    
#    return(df)

#######

def simp_rep(data, patient_col, tissue_column, subset_list_tissue, ttl_per_thres, comb_per_thres, thres_num = 3):
    
    #Choose the windows size to continue with
    if tissue_column != None:
        w2 = data.loc[data[tissue_column].isin(subset_list_tissue)]
        print("tissue_column true")
        
    else:
        w2 = data.copy()
        print("tissue_column false")
    
    simp_list = []
    for patient in list(w2[patient_col].unique()):
        w = w2.loc[w2[patient_col]==patient]
        xm = w.loc[:,l].values/n_num

        # Get the neighborhood combinations based on the threshold
        simps = get_thresh_simps(xm,ttl_per_thres)
        simp_freqs = simps.value_counts(normalize = True)
        sf = simp_freqs.to_frame()
        sf.rename(columns={0:patient},inplace=True)
        sf.reset_index(inplace=True)
        sf.rename(columns={'index':'merge'},inplace=True)
        simp_list.append(sf)
        #simp_sums = np.cumsum(simp_freqs)

        #thresh_cumulative = .95
        #selected_simps = simp_sums[simp_sums<=thresh_cumulative].index.values
       # selected_simps = simp_freqs[simp_freqs>=comb_per_thres].index.values
    
    simp_df = reduce(lambda  left,right: pd.merge(left,right,on=['merge'],
                                            how='outer'), simp_list)
    #simp_df = pd.concat(simp_list, axis=0)
    #simp_df.index = simp_df.index.to_series()
    simp_df.fillna(0,inplace=True)
    simp_df.set_index('merge', inplace=True)
    simp_out = simp_df.loc[simp_df.gt(0).sum(axis=1).ge(thres_num)]

    return simp_out

#######

def comb_num_freq(data_list, plot_order = None, pal_tis = None):
    df_new = []
    for df in data_list:
        df.reset_index(inplace=True)
        df.rename(columns={'merge':'combination'},inplace=True)
        df['count'] = df['combination'].apply(len)
        sum_df = df.groupby('count').sum()
        
        tbt = sum_df.reset_index()
        ttt = tbt.melt(id_vars = ['count'])
        ttt.rename(columns={'variable':'unique_cond','value':'fraction'}, inplace=True)
        df_new.append(ttt)
    df_exp = pd.concat(df_new)
    
    df_exp[['donor', 'tissue']] = df_exp['unique_cond'].str.split('_',expand=True)
    
    
    #swarmplot to compare 
    plt.figure(figsize=(5,5))

    ax = sns.boxplot(data = df_exp, x='count',  y='fraction', hue = 'tissue', dodge=True, \
                     hue_order=plot_order, palette=pal_tis)
    ax = sns.swarmplot(data = df_exp, x='count', y='fraction', hue = 'tissue', dodge=True, \
                      hue_order=plot_order, edgecolor='black',linewidth=1, palette=pal_tis)
    for patch in ax.artists:
        r, g, b, a = patch.get_facecolor()
        patch.set_facecolor((r, g, b, .3))
    #ax.set_yscale(\log\)
    plt.xlabel('')
    handles, labels = ax.get_legend_handles_labels()
    plt.legend(handles[:len(df_exp['tissue'].unique())], labels[:len(df_exp['tissue'].unique())],\
               bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.,frameon=False)
    plt.xticks(rotation=90)
    sns.despine(trim=True)
    
    return df_exp



def Create_neighborhoods(df,
                         n_num,
                         cluster_col,
                         X,
                         Y,
                         reg,
                         sum_cols = None,
                         keep_cols = None,
                         ks = [20]):
    
    if sum_cols == None:
        sum_cols=df[cluster_col].unique()
        
    if keep_cols == None:
        keep_cols = df.columns.values.tolist()
        
    Neigh = Neighborhoods(df,ks,cluster_col,sum_cols,keep_cols,X,Y,reg,add_dummies=True)
    windows = Neigh.k_windows()
    
    return(windows, sum_cols)

def Chose_window_size(windows,
                      n_num,
                      n_neighborhoods,
                      sum_cols,
                      n2_name = 'neigh_ofneigh'):
    #Choose the windows size to continue with
    w = windows[n_num]
    
    k_centroids = {}
    
    km = MiniBatchKMeans(n_clusters = n_neighborhoods,random_state=0)
    labels = km.fit_predict(w[sum_cols].values)
    k_centroids[n_num] = km.cluster_centers_
    w[n2_name] = labels
    
    return(w, k_centroids)


def Niche_heatmap(k_centroids,
                  w, n_num, sum_cols):
    #this plot shows the types of cells (ClusterIDs) in the different niches (0-9)
    k_to_plot = n_num
    niche_clusters = (k_centroids[k_to_plot])
    values = w[sum_cols].values
    tissue_avgs = values.mean(axis = 0)
    fc = np.log2(((niche_clusters+tissue_avgs)/(niche_clusters+tissue_avgs).sum(axis = 1, keepdims = True))/tissue_avgs)
    fc = pd.DataFrame(fc,columns = sum_cols)
    s=sns.clustermap(fc, cmap = 'bwr', vmax=-5)

def generate_random_colors(n):
    from random import randint
    color = []
    for i in range(n):
        color.append('#%06X' % randint(0, 0xFFFFFF))
    return (color)
        
def assign_colors(names, colors):
     
    # Printing original keys-value lists
    print("Original key list is : " + str(names))
    print("Original value list is : " + str(colors))
     
    # using naive method
    # to convert lists to dictionary
    res = {}
    for key in names:
        for value in colors:
            res[key] = value
            colors.remove(value)
            break
     
    # Printing resultant dictionary
    print("Resultant dictionary is : " + str(res))
    
    l=list(res.keys())

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
    
    sns.palplot(res.values())
    plt.xticks(range(len(res)),res.keys(),rotation = 30,ha='right')
    #plt.savefig(save_path+'color_legend.png', dpi=300)
    
    
    return(res)

def Barycentric_coordinate_projection(w, 
                                      plot_list, 
                                      threshold, 
                                      output_dir, 
                                      save_name, 
                                      col_dic, 
                                      l,
                                      n_num,
                                      cluster_col,
                                      SMALL_SIZE = 14, 
                                      MEDIUM_SIZE = 16, 
                                      BIGGER_SIZE = 18):
    
    #Settings for graph
    plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

    lmap = {j:i for i,j in enumerate(l)}
    palt=col_dic
    
    wgc  = w.loc[w.loc[:,plot_list].sum(axis=1)>threshold,:]
    idx = wgc.index.values
    xl = wgc.loc[:,plot_list]
    proj = np.array([[0,0],[np.cos(np.pi/3),np.sin(np.pi/3)], [1,0]])
    coords = np.dot(xl/n_num,proj) #####window size fraction
    
    plt.figure(figsize(14,14))
    jit = .002
    cols = [palt[a] for a in wgc[cluster_col]]
    
    plt.scatter(coords[:,0]+jit*np.random.randn(len(coords)),coords[:,1]+jit*np.random.randn(len(coords)),s = 1,alpha = .5, c = cols)
    plt.axis('off')
    plt.show()
    
    if save_name:
        plt.savefig(output_dir+save_name+'.png', format='png',\
                    dpi=300, transparent=True, bbox_inches='tight')
        










def calculate_neigh_combs(w, l, n_num, threshold = 0.85, per_keep_thres = 0.85):
    w.loc[:,l]

    #need to normalize by number of neighborhoods or k chosen for the neighborhoods
    xm = w.loc[:,l].values/n_num


    # Get the neighborhood combinations based on the threshold
    simps = get_thresh_simps(xm, threshold)
    simp_freqs = simps.value_counts(normalize = True)
    simp_sums = np.cumsum(simp_freqs)


    #See the percent to keep threshold or percent of neigbhorhoods that fall above a certain threshold
    test_sums_thres =simp_sums[simp_sums < per_keep_thres]
    test_len = len(test_sums_thres)
    per_values_above = simp_sums[test_len]-simp_sums[test_len-1]
    print(test_len, per_values_above)


    w['combination'] = [tuple(l[a] for a in s) for s in simps]
    w['combination_num'] = [tuple(a for a in s) for s in simps]

    # this shows what proportion (y) of the total cells are assigned to the top x combinations
    plt.figure(figsize(20,5))
    plt.plot(simp_sums.values)
    plt.title("proportion (y) of the total cells are assigned to the top x combinations")
    plt.show()

    # this shows what proportion (y) of the total cells are assigned to the top x combinations
    plt.figure(figsize(20,5))
    plt.plot(test_sums_thres.values)
    plt.title("proportion (y) of the total cells are assigned to the top x combinations - thresholded")
    plt.show()
    #plt.xticks(range(0,350,35),range(0,350,35),rotation = 90,fontsize = 10)

    return(simps, simp_freqs, simp_sums)

def build_graph_CN_comb_map(simp_freqs):
    g = nx.DiGraph()
    thresh_cumulative = .95
    thresh_freq = .001
    #selected_simps = simp_sums[simp_sums<=thresh_cumulative].index.values
    selected_simps = simp_freqs[simp_freqs>=thresh_freq].index.values
    selected_simps
    
    
    '''
    this builds the graph for the CN combination map
    '''
    for e0 in selected_simps:
        for e1 in selected_simps:
            if (set(list(e0))<set(list(e1))) and (len(e1) == len(e0)+1):
                g.add_edge(e0,e1)
                
    tops = simp_freqs[simp_freqs>=thresh_freq].sort_values(ascending = False).index.values.tolist()[:20]
    
    return(g, tops, e0, e1)

def generate_CN_comb_map(graph, tops, e0, e1, l, simp_freqs, color_dic):
        
    draw = graph
    pos = nx.drawing.nx_pydot.graphviz_layout(draw, prog='dot')
    height = 8
    
    plt.figure(figsize(40,20))
    for n in draw.nodes():
        col = 'black'
        if len(draw.in_edges(n))<len(n):
            col = 'black'
        plt.scatter(pos[n][0],pos[n][1]-5, s = simp_freqs[list(simp_freqs.index).index(n)]*10000, c = col, zorder = -1)
        if n in tops:
            plt.text(pos[n][0],pos[n][1]-7, '*', fontsize = 25, color = 'white', ha = 'center', va = 'center',zorder = 20)
        delta = 8
        #plot_sim((pos[n][0]+delta, pos[n][1]+delta),n, scale = 20,s = 200,text = True,fontsize = 15)
        plt.scatter([pos[n][0]]*len(n),[pos[n][1]+delta*(i+1) for i in range(len(n))],c = [color_dic[l[i]] for i in n] ,marker = 's', zorder = 5,s = 400)
        
    #     #add profiles below node
    #     x = pos[n][0]
    #     y = pos[n][1]
    #     y = y-height*2
    #     standard_node_size =  16
    #     node_heights = [0,3,8,5,3,2,1,5]
    #     marker_colors = ['red','red','blue','blue','red','red','blue','blue']
        
    #     plt.plot([x+(18*(i-1.5)) for i in range(len(node_heights))],[(y-height*.9)+v for v in node_heights],c = 'red',zorder =3)#,s = v*2 ,c= c,edgecolors='black',lw = 1)
    #     plt.scatter([x+(18*(i-1.5)) for i in range(len(node_heights))],[(y-height*.9)+v for v in node_heights],c = marker_colors,s = standard_node_size,zorder = 4)
        
            
    j = 0
    for e0,e1 in draw.edges():
        weight = 0.2
        alpha = .3
        color='black'
        if len(draw.in_edges(e1))<len(e1):
            color = 'black'
            lw =1
            weight = 0.4
            
    #     if (e0,e1) in set(draw.out_edges(tuple(sorted([lmap['3'],lmap['1']])))):
    #         j+=1
    #         print(j)
    #         color = 'green'
    #         weight = 2
    #         alpha = 1
            
    #     if (lmap['3'] in e0) and (lmap['1'] not in e0) and (lmap['1'] in e1):
    #         color = 'green'
    #         weight = 2
    #         alpha = 1
    
        plt.plot([pos[e0][0], pos[e1][0]],[pos[e0][1], pos[e1][1]], color = color, linewidth = weight,alpha = alpha,zorder = -10)
    
    plt.axis('off')
    #plt.savefig('CNM.pdf')
    plt.show()






def spatial_context_stats(n_num, 
                          patient_ID_component1, \
                          patient_ID_component2, \
                          windows, \
                          total_per_thres = 0.9, \
                          comb_per_thres = 0.005, \
                          tissue_column = 'Block type',\
                          subset_list = ["Resection"],\
                          plot_order = ['Resection','Biopsy'],\
                          pal_tis = {'Resection':'blue','Biopsy':'orange'},\
                          subset_list_tissue1 = ["Resection"],\
                          subset_list_tissue2 = ["Biopsy"]):
    
    data_compare = windows[n_num]
    
    # Prepare IDs this could for example be the combination of patient ID and tissue type. Apart from that, the function assigns a number to each name from the neighborhood column
    data_compare = prepare_neighborhood_df(data_compare, neighborhood_column = None, patient_ID_component1 = patient_ID_component1, patient_ID_component2 = patient_ID_component2) # this is a helper function 

    data_compare['donor_tis'].unique()




    simp_df_tissue1 = simp_rep(data = data_compare, patient_col='donor_tis', tissue_column = tissue_column, subset_list_tissue = subset_list_tissue1,\
                          ttl_per_thres=total_per_thres, comb_per_thres=comb_per_thres, thres_num = 1)
    print(simp_df_tissue1)
    
    simp_df_tissue2 = simp_rep(data = data_compare, patient_col='donor_tis', tissue_column = tissue_column, subset_list_tissue = subset_list_tissue2,\
                          ttl_per_thres=total_per_thres, comb_per_thres=comb_per_thres, thres_num = 1)
    print(simp_df_tissue2)


    ##### Compare the organization at high level to see if differences in combinations - more or less structured/compartmentalized
    data_simp = [simp_df_tissue1, simp_df_tissue2]
    df_num_count = comb_num_freq(data_list=data_simp)
    print(df_num_count)

    return(simp_df_tissue1, simp_df_tissue2)


def spatial_context_stats_vis(neigh_comb,
                              simp_df_tissue1,
                              simp_df_tissue2,
                              pal_tis = {'Resection': 'blue', 'Biopsy': 'orange'},
                              plot_order = ['Resection', 'Biopsy']):
    #Set Neigh and make comparison
    neigh_comb = (9,)

    df1 = simp_df_tissue1.loc[[neigh_comb]].T
    df2 = simp_df_tissue2.loc[[neigh_comb]].T
    print(stats.mannwhitneyu(df1[df1.columns[0]],df2[df2.columns[0]]))

    df1.reset_index(inplace=True)
    df1[['donor', 'tissue']] = df1['index'].str.split("_",expand=True)
    df2.reset_index(inplace=True)
    df2[['donor', 'tissue']] = df2['index'].str.split("_",expand=True)
    df_m = pd.concat([df1,df2])
    df_m['combo'] = str(neigh_comb)


    #swarmplot to compare 
    plt.figure(figsize=(5,5))

    ax = sns.boxplot(data = df_m, x='combo',  y= neigh_comb, hue = 'tissue', dodge=True, \
                     hue_order=plot_order, palette=pal_tis)
    ax = sns.swarmplot(data = df_m, x='combo', y=neigh_comb, hue = 'tissue', dodge=True, \
                      hue_order=plot_order, edgecolor='black',linewidth=1, palette=pal_tis)
    for patch in ax.artists:
        r, g, b, a = patch.get_facecolor()
        patch.set_facecolor((r, g, b, .3))
    #ax.set_yscale(\log\)
    plt.xlabel('')
    handles, labels = ax.get_legend_handles_labels()
    plt.legend(handles[:len(df_m['tissue'].unique())], labels[:len(df_m['tissue'].unique())],\
               bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.,frameon=False)
    plt.xticks(rotation=90)
    sns.despine(trim=True)

    #pt.savefig(save_path+save_name+'_swarm_boxplot.png', format='png', dpi=300, transparent=True, bbox_inches='tight')
