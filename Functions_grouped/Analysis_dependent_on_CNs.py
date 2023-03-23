#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 22 12:25:00 2023

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
    
    
    ##########################################################################################################
    # CCA Analysis 

    def Perform_CCA(cca, n_perms, nsctf, cns, subsets, group):
        stats_group1 = {}
        for cn_i in cns:
            for cn_j in cns:
                if cn_i < cn_j:
        
                    #concat dfs
                    combined = pd.concat([nsctf.loc[cn_i].loc[nsctf.loc[cn_i].index.isin(group)],nsctf.loc[cn_j].loc[nsctf.loc[cn_j].index.isin(group)]], axis = 1).dropna(axis = 0, how = 'any')
                    if combined.shape[0]<2:
                        continue
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
                        
                    return(stats_group1, arr)
                
    def Visulize_CCA_results(CCA_results, save_path, save_fig = False, save_name = "CCA_vis.png"):
        # Visualization of CCA 
        g1 = nx.Graph()
        for cn_pair, cc in CCA_results.items():
            s,t = cn_pair
            obs, perms = cc
            p =np.mean(obs>perms)
            if p>0.9 :
                g1.add_edge(s,t, weight = p)
            
            
        pal = sns.color_palette('bright',50)
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
