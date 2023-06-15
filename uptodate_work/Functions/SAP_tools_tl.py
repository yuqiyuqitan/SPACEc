#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 18 12:00:44 2023

@author: timnoahkempchen
"""
# load required packages 
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import time
from sklearn.cluster import MiniBatchKMeans
from scipy import stats
import statsmodels.api as sm
import networkx as nx
from scipy.spatial.distance import cdist
from yellowbrick.cluster import KElbowVisualizer
from sklearn.cross_decomposition import CCA
from scipy.stats import pearsonr,spearmanr

from  SAP_helperfunctions_hf import *

# Tools
############################################################


'''
tl_Shan_div calculates Shannon Diversity for each subgroup in a given dataset, and then performs an ANOVA test to compare the Shannon Diversity between different groups. 
The function takes in arguments such as data, subgroup list, grouping column, category column, replicate column, and a boolean for normalization. 
The function returns the Shannon Diversity values, ANOVA test results, and the results dataframe.
'''
# calculates diversity of cell types within a sample 
def tl_Shan_div(data, sub_l, group_com, per_categ, rep, sub_column, normalize=True):
    #calculate Shannon Diversity
    tt = hf_per_only(data = data, per_cat = per_categ, grouping = group_com,\
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

'''
The tl_neighborhood_analysis_2 function performs neighborhood analysis on single-cell data. It takes in data as input along with values, sum_cols, X and Y coordinates, reg, cluster_col, k, n_neighborhoods, and calc_silhouette_score as arguments. 
The function first groups cells into neighborhoods, calculates cluster centroids, and assigns cells to neighborhoods based on the nearest centroid. 
If calc_silhouette_score is True, it also returns the silhouette score. Finally, it returns the cell data with neighborhood labels and the cluster centroids.

The windows variable is used to store the data after grouping cells into neighborhoods. Each window represents a neighborhood and contains the values of a certain number of neighboring cells.

The function creates windows by grouping cells based on their location and a user-specified k value, which determines the number of neighbors to consider. It then calculates the sum of specified columns for each window.

The windows are stored in a dictionary called out_dict, with keys consisting of tuples (tissue_name, k) and values consisting of a numpy array with the summed column values for that window and a list of indices indicating the cells that were used to create that window.

After all windows have been created, the function combines them into a single DataFrame called window by concatenating the arrays for each tissue and adding the original cell indices as row indices. 
This DataFrame is then combined with the original cell DataFrame to produce a new DataFrame that includes a neighborhood label column called neighborhood_name.

The function also uses the windows dictionary to calculate the centroids of each neighborhood using k-means clustering. The centroids are stored in a dictionary called k_centroids, with keys consisting of the same k values used to create the windows.

'''


def tl_neighborhood_analysis_2(data, values, sum_cols, X = 'x', Y = 'y', reg = 'unique_region', cluster_col = 'Cell Type', k = 35, n_neighborhoods = 30,  elbow = False, metric = "distortion"):

    cells = data.copy()

    neighborhood_name = "neighborhood"+str(k)

    keep_cols = [X ,Y ,reg,cluster_col]

    n_neighbors = k

    cells[reg] = cells[reg].astype('str')

    #Get each region
    tissue_group = cells[[X,Y,reg]].groupby(reg)
    exps = list(cells[reg].unique())
    tissue_chunks = [(time.time(),exps.index(t),t,a) for t,indices in tissue_group.groups.items() for a in np.array_split(indices,1)] 

    tissues = [hf_get_windows(job, n_neighbors, exps= exps, tissue_group = tissue_group, X = X, Y = Y) for job in tissue_chunks]

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
    
    if elbow != True:
        km = MiniBatchKMeans(n_clusters = n_neighborhoods,random_state=0)
        
        labels = km.fit_predict(windows2[sum_cols].values)
        k_centroids[k] = km.cluster_centers_
        cells[neighborhood_name] = labels
        
    else:  
        
        km = MiniBatchKMeans(random_state=0)
            
        X = windows2[sum_cols].values
            
        labels = km.fit_predict(X)
        k_centroids[k] = km.cluster_centers_
        cells[neighborhood_name] = labels
            
        visualizer = KElbowVisualizer(km, k=(n_neighborhoods), timings=False, metric = metric)
        visualizer.fit(X)        # Fit the data to the visualizer
        visualizer.show()        # Finalize and render the figure
    
   
    return(cells, k_centroids)
    
    
############
'''
The function tl_cell_types_de performs differential enrichment analysis for various cell subsets between different neighborhoods using linear regression. 
It takes in several inputs such as cell type frequencies, neighborhood numbers, and patient information. 
The function first normalizes overall cell type frequencies and then neighborhood-specific cell type frequencies. Next, a linear regression model is fitted to find the coefficients and p-values for the group coefficient. 
Finally, the function returns a dataframe with the coefficients and p-values for each cell subset. The p-values can be corrected for multiple testing after the function has been executed.
'''

def tl_cell_types_de(ct_freq, all_freqs, neighborhood_num, nbs, patients, group, cells, cells1):
    
    # data prep
    # normalized overall cell type frequencies
    X_cts = hf_normalize(ct_freq.reset_index().set_index('patients').loc[patients,cells])
    
    # normalized neighborhood specific cell type frequencies
    df_list = []
    
    for nb in nbs:
        cond_nb = all_freqs.loc[all_freqs[neighborhood_num]==nb,cells1].rename({col: col+'_'+str(nb) for col in cells}, axis = 1).set_index('patients')
        df_list.append(hf_normalize(cond_nb))
    
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


def tl_community_analysis_2(data, values, sum_cols, X = 'x', Y = 'y', reg = 'unique_region', cluster_col = 'neigh_name', k = 100, n_neighborhoods = 30, elbow = False):
    
    cells = data.copy()

    neighborhood_name = "community"+str(k)

    keep_cols = [X ,Y ,reg,cluster_col]

    n_neighbors = k

    cells[reg] = cells[reg].astype('str')

    #Get each region
    tissue_group = cells[[X,Y,reg]].groupby(reg)
    exps = list(cells[reg].unique())
    tissue_chunks = [(time.time(),exps.index(t),t,a) for t,indices in tissue_group.groups.items() for a in np.array_split(indices,1)] 

    tissues = [hf_get_windows(job, n_neighbors, exps= exps, tissue_group = tissue_group, X = X, Y = Y) for job in tissue_chunks]

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
    
    if elbow != True:
        km = MiniBatchKMeans(n_clusters = n_neighborhoods,random_state=0)
        
        labels = km.fit_predict(windows2[sum_cols].values)
        k_centroids[k] = km.cluster_centers_
        cells[neighborhood_name] = labels
        
    else:  
        
        km = MiniBatchKMeans(random_state=0)
            
        X = windows2[sum_cols].values
            
        labels = km.fit_predict(X)
        k_centroids[k] = km.cluster_centers_
        cells[neighborhood_name] = labels
            
        visualizer = KElbowVisualizer(km, k=(n_neighborhoods), timings=False)
        visualizer.fit(X)        # Fit the data to the visualizer
        visualizer.show()        # Finalize and render the figure
    
    
    
    return(cells, neighborhood_name, k_centroids)

   
#################

'''
This Python function performs CCA analysis (Canonical Correlation Analysis) using the input parameters cca, n_perms, nsctf, cns, subsets, and group. The function first initializes an empty dictionary stats_group1. 
It then iterates through all pairs of indices cn_i and cn_j in cns, and if cn_i is less than cn_j, it concatenates the corresponding data frames in nsctf and drops any rows with missing values.

If the resulting concatenated data frame has more than two rows, the function fits the CCA model to the data, computing the canonical correlation achieving components with respect to observed data, as well as permuted data. 
It then computes and saves the Pearson correlation coefficient between the first components of the canonical correlation achieving vectors for the observed data and the permuted data. 
The function returns a dictionary stats_group1 containing the Pearson correlation coefficient values for each pair of cell type indices.
'''

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
'''
This function takes in a DataFrame df and creates a tensor T1 with dimensions (len(group),len(cns),len(cts)) where group is a list of patient IDs, cns is a list of neighborhood IDs, and cts is a list of cell type IDs. 
For each sample, neighborhood, and cell type, the corresponding count is stored in T1. The function then normalizes T1 so that each slice represents a joint distribution. Finally, it returns the normalized tensor dat1.
'''

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

    
#########



def tl_Create_neighborhoods(df,
                         n_num,
                         cluster_col,
                         X,
                         Y,
                         regions,
                         sum_cols = None,
                         keep_cols = None,
                         ks = [20]):

    if sum_cols == None:
        sum_cols=df[cluster_col].unique()
        
    if keep_cols == None:
        keep_cols = df.columns.values.tolist()
        
    Neigh = Neighborhoods(df,ks,cluster_col,sum_cols,keep_cols,X,Y,regions,add_dummies=True)
    windows = Neigh.k_windows()
    
    return(windows, sum_cols)

######

def tl_Chose_window_size(windows,
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


#######


def tl_calculate_neigh_combs(w, l, n_num, threshold = 0.85, per_keep_thres = 0.85):
    w.loc[:,l]

    #need to normalize by number of neighborhoods or k chosen for the neighborhoods
    xm = w.loc[:,l].values/n_num


    # Get the neighborhood combinations based on the threshold
    simps = hf_get_thresh_simps(xm, threshold)
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
    plt.figure(figsize = (7,3))
    plt.plot(simp_sums.values)
    plt.title("proportion (y) of the total cells are assigned to the top x combinations")
    plt.show()

    # this shows what proportion (y) of the total cells are assigned to the top x combinations
    plt.figure(figsize = (7,3))
    plt.plot(test_sums_thres.values)
    plt.title("proportion (y) of the total cells are assigned to the top x combinations - thresholded")
    plt.show()
    #plt.xticks(range(0,350,35),range(0,350,35),rotation = 90,fontsize = 10)

    return(simps, simp_freqs, simp_sums)

#######


def tl_build_graph_CN_comb_map(simp_freqs):
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



def tl_spatial_context_stats(n_num, 
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




    simp_df_tissue1 = hf_simp_rep(data = data_compare, patient_col='donor_tis', tissue_column = tissue_column, subset_list_tissue = subset_list_tissue1,\
                          ttl_per_thres=total_per_thres, comb_per_thres=comb_per_thres, thres_num = 1)
    print(simp_df_tissue1)
    
    simp_df_tissue2 = hf_simp_rep(data = data_compare, patient_col='donor_tis', tissue_column = tissue_column, subset_list_tissue = subset_list_tissue2,\
                          ttl_per_thres=total_per_thres, comb_per_thres=comb_per_thres, thres_num = 1)
    print(simp_df_tissue2)


    ##### Compare the organization at high level to see if differences in combinations - more or less structured/compartmentalized
    data_simp = [simp_df_tissue1, simp_df_tissue2]
    df_num_count = pl_comb_num_freq(data_list=data_simp)
    print(df_num_count)

    return(simp_df_tissue1, simp_df_tissue2)

###########

def tl_xycorr(df, sample_col, y_rows, x_columns, X_pix, Y_pix):
    
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


###############

def tl_get_distances(df, cell_list, cell_type_col):
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

###############
# clustering
def tl_clustering(adata, clustering = 'leiden', res=1, n_neighbors = 10, reclustering = False):
    if clustering not in ['leiden','louvain']:
        print("Invalid clustering options. Please select from leiden or louvain!")
        exit()
    #Compute the neighborhood relations of single cells the range 2 to 100 and usually 10
    if reclustering:
        print("Clustering")
        if clustering == 'leiden':
            sc.tl.leiden(adata, resolution = res, key_added = "leiden_" + str(res))
        else:
            sc.tl.louvain(adata, resolution = res, key_added = "louvain" + str(res))        
    else:
        print("Computing neighbors and UMAP")
        sc.pp.neighbors(adata, n_neighbors=n_neighbors)
        #UMAP computation
        sc.tl.umap(adata)
        print("Clustering")
        #Perform leiden clustering - improved version of louvain clustering
        if clustering == 'leiden':
            sc.tl.leiden(adata, resolution = res, key_added = "leiden_" + str(res))
        else:
            sc.tl.louvain(adata, resolution = res, key_added = "louvain" + str(res))



