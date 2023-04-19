#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 18 12:01:17 2023

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

# helper functions
############################################################


def hf_generate_random_colors(n):
    from random import randint
    color = []
    for i in range(n):
        color.append('#%06X' % randint(0, 0xFFFFFF))
    return (color)


#########


def hf_assign_colors(names, colors):
     
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


#########


def hf_per_only(data, grouping, replicate,sub_col, sub_list, per_cat, norm=True):
    
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


########


def hf_normalize(X):
    arr = np.array(X.fillna(0).values)
    return pd.DataFrame(np.log2(1e-3 + arr/arr.sum(axis =1, keepdims = True)), index = X.index.values, columns = X.columns).fillna(0)


########


def hf_cell_types_de_helper(df, ID_component1, ID_component2, neighborhood_col, group_col, group_dict, cell_type_col):
    
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


##########



def hf_get_pathcells(query_database, query_dict_list):
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
def hf_get_pathcells(query_database: Union[Dict, List[Dict]], query_dict_list: List[Dict]) -> Union[Dict, List[Dict]]:
    
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
    

##########



# Define a Python function named `hf_get_windows` that takes two arguments: `job` and `n_neighbors`.
def hf_get_windows(job,n_neighbors, exps, tissue_group, X, Y):
    
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

def hf_index_rank(a,axis):
    '''
    returns the index of every index in the sorted array
    haven't tested on ndarray yet
    '''
    arg =np.argsort(a,axis)
    
    return np.arange(a.shape[axis])[np.argsort(arg,axis)]

def hf_znormalize (raw_cells,grouper,markers,clip = (-7,7),dropinf = True):
    not_inf = raw_cells[np.isinf(raw_cells[markers].values).sum(axis = 1)==0]
    if not_inf.shape[0]!=raw_cells.shape[0]:
        print ('removing cells with inf values' ,raw_cells.shape,not_inf.shape)
    not_na = not_inf[not_inf[markers].isnull().sum(axis = 1)==0]
    if not_na.shape[0]!=not_inf.shape[0]:
        print ('removing cells with nan values', not_inf.shape,not_na.shape)
    
    
    znorm = not_na.groupby(grouper).apply(lambda x: ((x[markers]-x[markers].mean(axis = 0))/x[markers].std(axis = 0)).clip(clip[0],clip[1]))
    Z = not_na.drop(markers,1).merge(znorm,left_index = True,right_index = True)
    return Z

def hf_fast_divisive_cluster(X,num_clusters,metric = 'cosine',prints = True):
    
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

def hf_alloc_cells(X,centroids,metric = 'cosine'):
    dists = cdist(X,centroids,metric = metric)
    allocs = dists.argmin(axis = 1)
    return allocs

###########

def hf_get_sum_cols(cell_cuts,panel):
    arr = np.where(cell_cuts[:,0]==panel)[0]
    return slice(arr[0],arr[-1]+1)

###############
def hf_get_thresh_simps(x,thresh):
    sorts = np.argsort(-x, axis = 1)
    x_sorted = -np.sort(-x, axis = 1)
    cumsums = np.cumsum(x_sorted,axis = 1)
    thresh_simps = pd.Series([tuple(sorted(sorts[i,:(1+j)])) for i,j in enumerate(np.argmax(cumsums>thresh,axis = 1))])
    return thresh_simps


###############
def hf_prepare_neighborhood_df(cells_df, patient_ID_component1, patient_ID_component2, neighborhood_column = None):
    # Spacer for output 
    print("")
    
    # Combine two columns to form unique ID which will be stored as patients column 
    cells_df['patients'] = cells_df[patient_ID_component1]+'_'+cells_df[patient_ID_component2]
    print("You assigned following identifiers to the column 'patients':")
    print(cells_df['patients'].unique())
    
    # Spacer for output 
    print("")
    
    if neighborhood_column:
    # Assign numbers to neighborhoods
        neigh_num = {list(cells_df[neighborhood_column].unique())[i]:i for i in range(len(cells_df[neighborhood_column].unique()))}
        cells_df['neigh_num'] = cells_df[neighborhood_column].map(neigh_num)
        print("You assigned following numbers to the column 'neigh_num'. Each number represents one neighborhood:")
        print(cells_df['neigh_num'].unique())
        cells_df['neigh_num'] = cells_df['neigh_num'].astype('category')
    
    cells_df['patients'] = cells_df['patients'].astype('category')
    
  
    
    return(cells_df)


############


def hf_cor_subset(cor_mat, threshold, cell_type):
    pairs = tl_get_top_abs_correlations(cor_mat,thresh=threshold)
    
    piar1 = pairs.loc[pairs['col1']==cell_type]
    piar2 = pairs.loc[pairs['col2']==cell_type]
    piar=pd.concat([piar1,piar2])
    
    pair_list = list(set(list(piar['col1'].unique())+list(piar['col2'].unique())))
    
    return pair_list, piar, pairs


#############

# correlation analysis 
def hf_get_redundant_pairs(df):
    '''Get diagonal and lower triangular pairs of correlation matrix'''
    pairs_to_drop = set()
    cols = df.columns
    for i in range(0, df.shape[1]):
        for j in range(0, i+1):
            pairs_to_drop.add((cols[i], cols[j]))
    return pairs_to_drop

##############

# Spatial context analysis 
'''
this is the code that finds the minimal combination of CNs
required to make up a threshold percentage of assignments in a window
combinations are stored as a sorted tuple
'''

def hf_simp_rep(data, patient_col, tissue_column, subset_list_tissue, ttl_per_thres, comb_per_thres, thres_num = 3):
    
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
        simps = hf_get_thresh_simps(xm,ttl_per_thres)
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