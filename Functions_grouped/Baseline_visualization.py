#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 22 12:15:30 2023

@author: timnoahkempchen
"""
# Librarys 

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

##########################################################################################################

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