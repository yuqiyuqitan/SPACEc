#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 22 12:19:28 2023

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
