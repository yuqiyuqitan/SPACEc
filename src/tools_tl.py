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
from tqdm import tqdm
import skimage.io as io
import skimage.transform
import skimage.filters.rank
import skimage.color
import skimage.exposure
import skimage.morphology
import skimage
from  helperfunctions_hf import *

from tqdm import tqdm

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


def tl_community_analysis_2(data, values, sum_cols, X='x', Y='y', reg='unique_region', cluster_col='neigh_name', k=100, n_neighborhoods=30, elbow=False):
    neighborhood_name = "community" + str(k)

    keep_cols = [X, Y, reg, cluster_col]

    n_neighbors = k

    data[reg] = data[reg].astype('str')

    # Get each region
    tissue_group = data[[X, Y, reg]].groupby(reg)
    exps = list(data[reg].unique())
    tissue_chunks = [(time.time(), exps.index(t), t, a) for t, indices in tissue_group.groups.items() for a in np.array_split(indices, 1)]

    tissues = [hf_get_windows(job, n_neighbors, exps=exps, tissue_group=tissue_group, X=X, Y=Y) for job in tissue_chunks]

    # Loop over k to compute neighborhoods
    out_dict = {}
    for neighbors, job in zip(tissues, tissue_chunks):
        chunk = np.arange(len(neighbors))  # indices
        tissue_name = job[2]
        indices = job[3]
        window = values[neighbors[chunk, :k].flatten()].reshape(len(chunk), k, len(sum_cols)).sum(axis=1)
        out_dict[(tissue_name, k)] = (window.astype(np.float16), indices)

    windows = {}

    window = pd.concat([pd.DataFrame(out_dict[(exp, k)][0], index=out_dict[(exp, k)][1].astype(int), columns=sum_cols) for exp in exps], 0)
    window = window.loc[data.index.values]
    window = pd.concat([data[keep_cols], window], 1)
    windows[k] = window

    # Fill in based on above
    k_centroids = {}

    # Producing what to plot
    windows2 = windows[k]
    windows2[cluster_col] = data[cluster_col]

    if not elbow:
        km = MiniBatchKMeans(n_clusters=n_neighborhoods, random_state=0)
        labels = km.fit_predict(windows2[sum_cols].values)
        k_centroids[k] = km.cluster_centers_
        data[neighborhood_name] = labels
    else:
        km = MiniBatchKMeans(random_state=0)
        X = windows2[sum_cols].values
        labels = km.fit_predict(X)
        k_centroids[k] = km.cluster_centers_
        data[neighborhood_name] = labels
        visualizer = KElbowVisualizer(km, k=(n_neighborhoods), timings=False)
        visualizer.fit(X)  # Fit the data to the visualizer
        visualizer.show()  # Finalize and render the figure

    return data, neighborhood_name, k_centroids

   

#################

'''
This Python function performs CCA analysis (Canonical Correlation Analysis) using the input parameters cca, n_perms, nsctf, cns, subsets, and group. The function first initializes an empty dictionary stats_group1. 
It then iterates through all pairs of indices cn_i and cn_j in cns, and if cn_i is less than cn_j, it concatenates the corresponding data frames in nsctf and drops any rows with missing values.

If the resulting concatenated data frame has more than two rows, the function fits the CCA model to the data, computing the canonical correlation achieving components with respect to observed data, as well as permuted data. 
It then computes and saves the Pearson correlation coefficient between the first components of the canonical correlation achieving vectors for the observed data and the permuted data. 
The function returns a dictionary stats_group1 containing the Pearson correlation coefficient values for each pair of cell type indices.
'''

# CCA Analysis 

def tl_Perform_CCA(cca, n_perms, nsctf, cns, subsets):
    stats_group1 = {}
    total_iterations = len(cns) * (len(cns) - 1) // 2  # Total number of iterations for the progress bar

    with tqdm(total=total_iterations, desc="Performing CCA", initial=0, position=0, leave=True) as pbar:
        idx = 0  # Variable to track the progress manually
        for cn_i in cns:
            for cn_j in cns:
                if cn_i < cn_j:
                    idx += 1  # Increment the progress manually
                    pbar.update(1)  # Update the progress bar
                    #print(cn_i, cn_j)
                    # Concat dfs
                    combined = pd.concat([nsctf.loc[cn_i], nsctf.loc[cn_j]], axis=1).dropna(axis=0, how='any')
                    if combined.shape[0] > 2:
                        if subsets is not None:
                            x = combined.iloc[:, :len(subsets)].values
                            y = combined.iloc[:, len(subsets):].values
                        else:
                            x = combined.values
                            y = combined.values

                        arr = np.zeros(n_perms)
                        # Compute the canonical correlation achieving components with respect to observed data
                        ccx, ccy = cca.fit_transform(x, y)
                        stats_group1[cn_i, cn_j] = (pearsonr(ccx[:, 0], ccy[:, 0])[0], arr)
                        # Initialize array for perm values
                        for i in range(n_perms):
                            idx = np.arange(len(x))
                            np.random.shuffle(idx)
                            # Compute with permuted data
                            cc_permx, cc_permy = cca.fit_transform(x[idx], y)
                            arr[i] = pearsonr(cc_permx[:, 0], cc_permy[:, 0])[0]
    return stats_group1

        

# tensor decomposition 




#######
'''
This function takes in a DataFrame df and creates a tensor T1 with dimensions (len(group),len(cns),len(cts)) where group is a list of patient IDs, cns is a list of neighborhood IDs, and cts is a list of cell type IDs. 
For each sample, neighborhood, and cell type, the corresponding count is stored in T1. The function then normalizes T1 so that each slice represents a joint distribution. Finally, it returns the normalized tensor dat1.
'''

def old_tl_build_tensors(df, group, cns, cts, counts):
    
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

def tl_build_tensors(df, neighborhood_col, celltype_col, id_col, counts, subset_cns=None, subset_cts=None):
    uid = list(df[id_col].unique())

    if subset_cns is None:
        cns = list(df[neighborhood_col].unique())
    else:
        cns = subset_cns

    if subset_cts is None:
        cts = list(df[celltype_col].unique())
    else:
        cts = subset_cts

    T1 = np.zeros((len(uid), len(cns), len(cts)))

    for i, pat in enumerate(uid):
        for j, cn in enumerate(cns):
            for k, ct in enumerate(cts):
                T1[i, j, k] = counts.loc[(pat, cn, ct)]

    # normalize so we have joint distributions for each slice
    dat1 = np.nan_to_num(T1 / T1.sum((1, 2), keepdims=True))

    return dat1, cns, cts
    
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


def tl_build_graph_CN_comb_map(simp_freqs, thresh_freq = .001):
    g = nx.DiGraph()
    
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
def tl_clustering(adata, clustering='leiden', res=1, n_neighbors=10, reclustering=False, markers_for_clustering=None):
    if markers_for_clustering is not None:
        adata_subset = adata[:, markers_for_clustering]
    else:
        adata_subset = adata
        
    if clustering not in ['leiden', 'louvain']:
        print("Invalid clustering options. Please select from leiden or louvain!")
        exit()
    
    if reclustering:
        print("Computing neighbors")
        sc.pp.neighbors(adata_subset, n_neighbors=n_neighbors)
        print("Clustering")
        if clustering == 'leiden':
            sc.tl.leiden(adata_subset, resolution=res, key_added="leiden_" + str(res))
            adata.obs["leiden_" + str(res)] = adata_subset.obs["leiden_" + str(res)]
        else:
            sc.tl.louvain(adata_subset, resolution=res, key_added="louvain" + str(res))      
            adata.obs["louvain" + str(res)] = adata_subset.obs["louvain" + str(res)]  
    else:
        print("Computing neighbors and UMAP")
        sc.pp.neighbors(adata_subset, n_neighbors=n_neighbors)
        sc.tl.umap(adata_subset)
        print("Clustering")
        if clustering == 'leiden':
            sc.tl.leiden(adata_subset, resolution=res, key_added="leiden_" + str(res))
            adata.obs["leiden_" + str(res)] = adata_subset.obs["leiden_" + str(res)]
            adata.obsm.update(adata_subset.obsm)
            adata.uns.update(adata_subset.uns)
        else:
            sc.tl.louvain(adata_subset, resolution=res, key_added="louvain" + str(res))
            adata.obs["louvain" + str(res)] = adata_subset.obs["louvain" + str(res)]
            adata.obsm.update(adata_subset.obsm)
            adata.uns.update(adata_subset.uns)


###############
# Patch analysis


def tl_generate_voronoi_plots(df, output_path, grouping_col = 'Community',
                           tissue_col = 'tissue',
                           region_col = 'unique_region',
                           x_col = "x",
                           y_col = "y"):
    """
    Generate Voronoi plots for unique combinations of tissue and region.

    Parameters:
        df (pandas.DataFrame): Input DataFrame containing the data.
        output_path (str): Output path to save the plots.
        grouping_col (str): Column that contains group label that is used to color the voronoi diagrams
        tissue_col (str): Column that contains tissue labels
        region_col (str): Column that contains region labels
        x_col (str): Column that contains x coordinates
        y_col (str): Column that contains y coordinates

    Returns:
        None
    """

    unique_tissues = df[tissue_col].unique()
    unique_regions = df[region_col].unique()

    combinations = list(itertools.product(unique_tissues, unique_regions))

    for tissue, region in combinations:
        subset_df = df[(df[tissue_col] == tissue) & (df[region_col] == region)]
        sorted_df = subset_df.sort_values(grouping_col)
        unique_values = sorted_df[grouping_col].unique()

        specific_output = os.path.join(output_path, tissue)
        os.makedirs(specific_output, exist_ok=True)
        specific_output = os.path.join(specific_output, region)
        os.makedirs(specific_output, exist_ok=True)

        for group in unique_values:
            start = time.time()

            output_filename = group + "_plot.png"
            output_path2 = os.path.join(specific_output, output_filename)

            color_dict = {}
            for value in unique_values:
                color_dict[value] = 'black'
            color_dict[group] = 'white'

            X = sorted_df[x_col]
            Y = sorted_df[y_col]
            np.random.seed(1234)
            points = np.c_[X, Y]

            vor = Voronoi(points)
            regions, vertices = hf_voronoi_finite_polygons_2d(vor)
            groups = sorted_df[grouping_col].values

            fig, ax = plt.subplots()
            ax.set_ylim(0, max(Y))
            ax.set_xlim(0, max(X))
            ax.axis('off')

            for i, region in tqdm(enumerate(regions), total=len(regions), desc="Processing regions"):
                group = groups[i]
                color = color_dict.get(group, 'gray')
                polygon = vertices[region]
                ax.fill(*zip(*polygon), color=color)

            ax.plot(points[:, 0], points[:, 1], 'o', color='black', zorder=1, alpha=0)

            fig.set_size_inches(9.41, 9.07 * 1.02718006795017)
            fig.savefig(output_path2, bbox_inches='tight', pad_inches=0, dpi=129.0809327846365)
            plt.close(fig)

            end = time.time()
            print(end - start)
            

  



  


def tl_generate_masks_from_images(image_folder, mask_output, image_type = ".tif", filter_size = 5, threshold_value = 10):
    """
    Generate binary masks from CODEX images.

    Parameters:
        image_folder (str): Directory that contains the images that are used to generate the masks
        mask_output (str): Directory to store the generated masks
        image_type (str): File type of image. By default ".tif"
        filter_size (num): Size for filter disk during mask generation
        threshold_value (num): Threshold value for binary mask generation

    Returns:
        None
    """
    folders_list = hf_list_folders(image_folder)
    print(folders_list)
    for folder in tqdm(folders_list, desc="Processing folders"):
        direc = image_folder + "/" + folder
        print(direc)
    
        filelist = os.listdir(direc)
        filelist = [f for f in filelist if f.endswith(image_type)]
        print(filelist)
    
        output_dir = mask_output + folder 
        os.makedirs(output_dir, exist_ok=True)
    
        for f in tqdm(filelist, desc="Processing files"):
            path = os.path.join(direc, f)
            print(path)
    
            tl_generate_mask(path=path, output_dir=output_dir, filename="/" + f, filter_size=filter_size, threshold_value=threshold_value)

def tl_generate_info_dataframe(df, voronoi_output, mask_output, filter_list = None, info_cols = ['tissue', 'donor', 'unique_region', 'region', 'array']):
    """
    Generate a filtered DataFrame based on specific columns and values.

    Parameters:
        df (pandas.DataFrame): Input DataFrame.
        voronoi_output (str): Path to the Voronoi output directory.
        mask_output (str): Path to the mask output directory.
        info_cols (list): columns to extract from input df
        filter_list (list, optional): List of values to filter.

    Returns:
        pandas.DataFrame: Filtered DataFrame.
    """
    df_info = df[info_cols].drop_duplicates()
    df_info['folder_names'] = df_info['array']
    df_info['region'] = df_info['region'].astype(int)
    df_info['region_long'] = ['reg00' + str(region) for region in df_info['region']]
    df_info['voronoi_path'] = voronoi_output + df_info['tissue'] + "/" + df_info['unique_region']
    df_info['mask_path'] = mask_output + df_info['folder_names'] + "/"
    
    if filter_list != None:
        # remove unwanted folders
        df_info = df_info[~df_info['folder_names'].isin(filter_list)]
        
    else:
        print("no filter used")

    return df_info


###

def tl_process_files(voronoi_path, mask_path, region):
    """
    Process files based on the provided paths and region.

    Parameters:
        voronoi_path (str): Path to the Voronoi files.
        mask_path (str): Path to the mask files.
        region (str): Region identifier.

    Returns:
        None
    """
    png_files_list = hf_get_png_files(voronoi_path)
    tiff_file_path = hf_find_tiff_file(mask_path, region)

    if tiff_file_path:
        print(f"Matching TIFF file found: {tiff_file_path}")
    else:
        print("No matching TIFF file found.")

    for f in tqdm(png_files_list, desc="Processing files"):
        print(f)
        tl_apply_mask(f, tiff_file_path, f + "_cut.png")

###

def tl_process_data(df_info, output_dir_csv):
    """
    Process data based on the information provided in the DataFrame.

    Parameters:
        df_info (pandas.DataFrame): DataFrame containing the information.
        output_dir_csv (str): Output directory for CSV results.
        
    Returns:
        pandas.DataFrame: Concatenated DataFrame of results.
        list: List of contours.
    """
    DF_list = []
    contour_list = []

    for index, row in df_info.iterrows():
        voronoi_path = row['voronoi_path']
        mask_path = row['mask_path']
        region = row['region_long']
        donor = row['donor']
        unique_region = row['unique_region']

        png_files_list = hf_get_png_files(voronoi_path)
        png_files_list = [filename for filename in png_files_list if not filename.endswith("cut.png")]

        tiff_file_path = hf_find_tiff_file(mask_path, region)

        if tiff_file_path:
            print(f"Matching TIFF file found: {tiff_file_path}")
        else:
            print("No matching TIFF file found.")

        for f in png_files_list:
            print(f)
            g = f + "_cut" + ".png"
            print(g)
            tl_apply_mask(f, tiff_file_path, g)

            output_dir_csv_tmp = output_dir_csv + "/" + donor + "_" + unique_region
            os.makedirs(output_dir_csv_tmp, exist_ok=True)

            image_dir = output_dir_csv + "/" + donor + "_" + unique_region
            os.makedirs(image_dir, exist_ok=True)
            print(f"Path created: {image_dir}")

            image_dir = os.path.join(image_dir, os.path.basename(os.path.normpath(g)))
            path = g

            df, contour = tl_analyze_image(path, invert=False, output_dir=image_dir, )

            df["group"] = hf_extract_filename(g)
            df["unique_region"] = unique_region

            DF_list.append(df)
            contour_list.append(contour)

    results_df = pd.concat(DF_list)
    contour_list_results_df = pd.concat(DF_list)

    results_df.to_csv(os.path.join(output_dir_csv, "results.csv"))

    return results_df, contour_list
###

def tl_analyze_image(path, output_dir, invert=False, properties_list = [
    "label",
    "centroid",
    "area",
    "perimeter",
    "solidity",
    "coords",
    "axis_minor_length",
    "axis_major_length",
    "orientation",
    "slice"]):
    """
    Analyze an image by performing connected component analysis on patches and storing their information.

    The function applies image processing techniques such as Gaussian smoothing, thresholding, and connected component
    labeling to identify and analyze patches within the image. It extracts region properties of these patches,
    calculates their circularity, and stores the coordinates of their contour. The resulting information is saved
    in a DataFrame along with a visualization plot.

    Parameters:
        path (str): Path to the input image.
        output_dir (str): Directory to save the output plot.
        invert (bool, optional): Flag indicating whether to invert the image (default is False).
        properties_list: (list of str): Define properties to be measured (see SciKit Image), by default "label", "centroid", "area", "perimeter", "solidity", "coords", "axis_minor_length", "axis_major_length", "orientation", "slice"

    Returns:
        tuple: A tuple containing the DataFrame with region properties, including patch contour coordinates, and
               the list of contour coordinates for each patch.
    """
    image = skimage.io.imread(path)

    if image.ndim == 2:
        print("2D array")
    else:
        image = image[:, :, 0]

    if invert:
        print("The original background color was white. The image was inverted for further analysis.")
        # image = 255 - image
    else:
        print("no inversion")

    smooth = skimage.filters.gaussian(image, sigma=1.5)
    thresh = smooth > skimage.filters.threshold_otsu(smooth)

    blobs_labels = skimage.measure.label(thresh, background=0)

    properties = skimage.measure.regionprops(blobs_labels)

    props_table = skimage.measure.regionprops_table(
        blobs_labels,
        properties=(properties_list
        ),
    )

    prop_df = pd.DataFrame(props_table)

    prop_df["circularity"] = (4 * np.pi * prop_df["area"]) / (prop_df["perimeter"] ** 2)

    # Store the contour of each patch in the DataFrame
    contour_list = []
    for index in range(1, blobs_labels.max()):
        label_i = properties[index].label
        contour = skimage.measure.find_contours(blobs_labels == label_i, 0.5)[0]
        contour_list.append(contour)

    contour_list_df = pd.DataFrame({"contours": contour_list})

    prop_df = pd.concat([prop_df, contour_list_df], axis=1)

    plt.figure(figsize=(9, 3.5))
    plt.subplot(1, 2, 1)
    plt.imshow(thresh, cmap="gray")
    plt.axis("off")
    plt.subplot(1, 2, 2)
    plt.imshow(blobs_labels, cmap="nipy_spectral")
    plt.axis("off")

    plt.tight_layout()

    plt.savefig(output_dir)
    plt.close()

    return prop_df, contour_list

###

def tl_apply_mask(image_path, mask_path, output_path):
    """
    Apply a mask to an image and save the resulting masked image.

    Parameters:
        image_path (str): Path to the input image.
        mask_path (str): Path to the mask image.
        output_path (str): Path to save the masked image.

    Returns:
        None
    """
    # Load the image and the mask
    image = io.imread(image_path)
    mask = io.imread(mask_path, as_gray=True)
    mask = np.flip(mask, axis=0)

    width = 941
    height = 907
    image = skimage.transform.resize(image, (height, width))

    if image.ndim == 2:
        print("2D array")
    else:
        image = image[:, :, :3]

        # Convert to grayscale
        image = skimage.color.rgb2gray(image)

    # Convert to 8-bit
    image = skimage.img_as_ubyte(image)

    print("Image shape:", image.shape)
    print("Mask shape:", mask.shape)

    # Ensure the mask is binary
    mask = mask > 0

    # Apply the mask to the image
    masked_image = image.copy()
    masked_image[~mask] = 0

    # Check if the image has an alpha channel (transparency)
    if masked_image.ndim == 2:
        print("2D array")
    else:
        masked_image = masked_image[:, :, :3]

    # Save the masked image
    io.imsave(output_path, skimage.img_as_ubyte(masked_image))
   
### 

def tl_generate_mask(path, output_dir, filename="mask.png", filter_size=5, threshold_value=5):
    """
    Generate a mask from a maximum projection of an input image.

    Parameters:
        path (str): Path to the input image.
        output_dir (str): Directory to save the generated mask and quality control plot.
        filename (str, optional): Name of the generated mask file (default is "mask.png").
        filter_size (int, optional): Size of the filter disk used for image processing (default is 5).
        threshold_value (int, optional): Threshold value for binary conversion (default is 5).

    Returns:
        None
    """
    # Load the image
    image = io.imread(path)

    # Perform Z projection using Maximum Intensity
    z_projection = np.max(image, axis=0)

    # Resize the image
    width = 941
    height = 907
    resized_image = skimage.transform.resize(z_projection, (height, width, 3), preserve_range=True)
    print("Resized image shape:", resized_image.shape)

    # Remove alpha channel if present
    if resized_image.shape[-1] == 4:
        resized_image = resized_image[:, :, :3]

    # Convert to grayscale
    gray_image = skimage.color.rgb2gray(resized_image)

    # Assuming gray_image has pixel values outside the range [0, 1]
    # Normalize the pixel values to the range [0, 1]
    gray_image_normalized = (gray_image - gray_image.min()) / (gray_image.max() - gray_image.min())

    # Convert to 8-bit
    gray_image_8bit = skimage.img_as_ubyte(gray_image_normalized)

    # Apply maximum filter
    max_filtered = skimage.filters.rank.maximum(gray_image_8bit, skimage.morphology.disk(filter_size))

    # Apply minimum filter
    min_filtered = skimage.filters.rank.minimum(max_filtered, skimage.morphology.disk(filter_size))

    # Apply median filter
    median_filtered = skimage.filters.rank.median(min_filtered, skimage.morphology.disk(filter_size))

    # Manual Thresholding
    binary = median_filtered > threshold_value

    # Convert to mask
    mask = skimage.morphology.closing(binary, skimage.morphology.square(3))

    # Plotting
    fig, axes = plt.subplots(2, 3, figsize=(12, 6))

    axes[0, 0].imshow(gray_image, cmap='gray')
    axes[0, 0].set_title('Grayscale Image')

    axes[0, 1].imshow(gray_image_8bit, cmap='gray')
    axes[0, 1].set_title('8-bit Image')

    axes[0, 2].imshow(max_filtered, cmap='gray')
    axes[0, 2].set_title('Maximum Filtered')

    axes[1, 0].imshow(min_filtered, cmap='gray')
    axes[1, 0].set_title('Minimum Filtered')

    axes[1, 1].imshow(median_filtered, cmap='gray')
    axes[1, 1].set_title('Median Filtered')

    axes[1, 2].imshow(mask, cmap='gray')
    axes[1, 2].set_title('Mask')

    for ax in axes.ravel():
        ax.axis('off')

    plt.tight_layout()
    fig.savefig(output_dir + filename + '_QC_plot.png', dpi=300, format='png')

    plt.show()

    # Save the result
    io.imsave(output_dir + filename, mask)

#####

def tl_test_clustering_resolutions(adata, clustering='leiden', n_neighbors=10, resolutions=[1]):
    """
    Test different resolutions for reclustering using Louvain or Leiden algorithm.

    Parameters:
        adata (AnnData): Anndata object containing the data.
        clustering (str, optional): Clustering algorithm to use (default is 'leiden').
        n_neighbors (int, optional): Number of nearest neighbors (default is 10).
        resolutions (list, optional): List of resolutions to test (default is [1]).

    Returns:
        None
    """
    for res in tqdm(resolutions, desc="Testing resolutions"):
        if 'leiden' in clustering:
            tl_clustering(adata, clustering='leiden', n_neighbors=n_neighbors, res=res, reclustering=True)
        else:
            tl_clustering(adata, clustering='louvain', n_neighbors=n_neighbors, res=res, reclustering=True)

        sc.pl.umap(adata, color=f'{clustering}_{res}', legend_loc="on data")
        
        
###############
# clustering

def tl_clustering_ad(adata, clustering = 'leiden', marker_list = None, res=1, n_neighbors = 10, reclustering = False):
    if clustering not in ['leiden','louvain']:
        print("Invalid clustering options. Please select from leiden or louvain!")
        exit()
    #input a list of markers for clustering
    #reconstruct the anndata
    if marker_list is not None:
        if len(list(set(marker_list) - set(adata.var_names)))>0:
            print("Marker list not all in adata var_names! Using intersection instead!")
            marker_list = list(set(marker_list) & set(adata.var_names))
            print("New marker_list: " + ' '.join(marker_list))
        adata_tmp = adata
        adata = adata[:,marker_list]
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
    
    if marker_list is None:
        return adata
    else:
        if clustering == 'leiden':
            adata_tmp.obs["leiden_" + str(res)] = adata.obs["leiden_" + str(res)].values
        else:
            adata_tmp.obs["leiden_" + str(res)] = adata.obs["louvain_" + str(res)].values
        #append other data
        adata_tmp.obsm = adata.obsm
        adata_tmp.obsp = adata.obsp
        adata_tmp.uns = adata.uns
        return adata_tmp  




def tl_neighborhood_analysis_ad(adata, unique_region, cluster_col, 
    X = 'x', Y = 'y',
    k = 35, n_neighborhoods = 30,  elbow = False, metric = "distortion"):
    '''
    Compute for Cellular neighborhood
    adata:  anndata containing information
    unique_region: each region is one independent CODEX image
    cluster_col:  columns to compute CNs on, typicall 'celltype'
    X:  X
    Y: Y
    k: k neighbors to compute
    n_neighborhoods: number of neighborhoods one ends ups with
    elbow: whether to see the optimal elbow plots or not
    metric:
    '''
    df = pd.DataFrame(adata.obs[[X, Y, cluster_col, unique_region]])

    cells = pd.concat([df,pd.get_dummies(df[cluster_col])],1)
    sum_cols = cells[cluster_col].unique()
    values = cells[sum_cols].values

    neighborhood_name = "CN"+ "_k" +str(k) + "_n" + str(n_neighborhoods)
    centroids_name = "Centroid"+ "_k" +str(k) + "_n" + str(n_neighborhoods)
    
    n_neighbors = k

    cells[unique_region] = cells[unique_region].astype('str')
    cells['cellid'] = cells.index.values
    cells.reset_index(inplace=True)

    keep_cols = [X, Y , unique_region,  cluster_col]

    #Get each region
    tissue_group = cells[[X,Y,unique_region]].groupby(unique_region)
    exps = list(cells[unique_region].unique())
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
        k_centroids[str(k)] = km.cluster_centers_
        adata.obs[neighborhood_name] = labels
        adata.uns[centroids_name] = k_centroids
        
    else:  
        km = MiniBatchKMeans(random_state=0)
            
        X = windows2[sum_cols].values
            
        labels = km.fit_predict(X)
        k_centroids[str(k)] = km.cluster_centers_
        adata.obs[neighborhood_name] = labels
        adata.uns[centroids_name] = k_centroids
            
        visualizer = KElbowVisualizer(km, k=(n_neighborhoods), timings=False, metric = metric)
        visualizer.fit(X)        # Fit the data to the visualizer
        visualizer.show()        # Finalize and render the figure
    
   
    return adata 



def tl_CNmap_ad(adata, cn_col, palette, unique_reg = 'group',
             k = 75, X='x', Y='y',threshold = 0.85, per_keep_thres = 0.85):
    ks=[k]
    cells_df = pd.DataFrame(adata.obs)
    cells_df.reset_index(inplace=True)
    sum_cols=cells_df[cn_col].unique()
    keep_cols = cells_df.columns
    Neigh = Neighborhoods(cells_df, ks, cn_col,sum_cols,keep_cols,X,Y,reg=unique_reg,add_dummies=True)
    windows = Neigh.k_windows()
    w = windows[k]
    l=list(palette.keys())
    simps, simp_freqs, simp_sums = tl_calculate_neigh_combs(w, 
                                                     l, #color palette
                                                     k, 
                                                     threshold = threshold, 
                                                     per_keep_thres = per_keep_thres)
    g, tops, e0, e1 = tl_build_graph_CN_comb_map(simp_freqs)
    return g, tops, e0, e1, simp_freqs

def tl_spat_cont(data, col_neigh='Neighborhood', col_donor='donor', col_comp='Therapy', group1='Pre', \
              group2 = 'Post', col_unique='filename',permute=False,\
              neigh_map = None, examples = None, stats=False):

  neigh_num = {list(data[col_neigh].unique())[i]:i for i in range(len(data[col_neigh].unique()))}
  data['neigh_num'] = data[col_neigh].map(neigh_num)

  data['donor_gp'] = data[col_donor]+'_'+data[col_comp]
  pat_gp = data[['donor_gp',col_comp]].drop_duplicates()
  pat_to_gp= {a:b for a,b in pat_gp.values}

  spot_to_patient = {a:b for a,b in data[[col_unique,'donor_gp']].drop_duplicates().values}

  spot_group_assignment = {a:b for a,b in data[[col_unique,col_comp]].drop_duplicates().values}

  #check that our segmentation is working as we expect for a few randomly selected spot,cn pairs
  print('Checking that our segmentation is working as we expect for a few randomly selected spot,cn pairs')
  #plotting params
  #pal = sns.color_palette('bright',10)
  figsize=(10,4)

  #generate random samples
  spot_ids  = data[col_unique].unique()
  spot_cn_combs = list(itertools.product(spot_ids,range(len(data['neigh_num'].unique()))))
  np.random.seed(10)
  np.random.shuffle(spot_cn_combs)

  #set params
  num_neighbors = 10
  min_instance_size = 10

  if neigh_map is None:
    neigh_list = list(data[col_neigh].unique())
    color_list=list(pal_color_cells.values())
    neigh_map = dict(zip(neigh_list, color_list))

  dict_new = {}
  for k in neigh_num.keys():
      dict_new[neigh_num[k]]=neigh_map[k]

  pal = list(dict_new.values())


  for k in spot_cn_combs[:20]:
      (spot_id, cn) = k
      # use the segment instances script
      spot, spot_cn_cell_idxs, spot_inst_assignments = segment_instances(spot_id,num_neighbors=num_neighbors,min_instance_size=min_instance_size)


      #plot spot cells of that CN
      plt.subplot(1,2,1)
      sub_spot = spot.iloc[spot_cn_cell_idxs[cn]]
      plt.scatter(sub_spot['x'],sub_spot['y'],c = [pal[cn]],s = 2)


      plt.title('spot_id {}, cn {} cells'.format(spot_id,cn))
      plt.axis('off')

      #plot segmentation colored by instance
      plt.subplot(1,2,2)
      cell_idx,assignment = spot_inst_assignments[cn]
      for j in np.unique(assignment):

          sub_spot = spot.iloc[cell_idx[assignment ==j]]
          plt.scatter(sub_spot['x'],sub_spot['y'],c = [pal[j%10]],s = 2)
      plt.axis('off')
      plt.title('segmented')
      plt.show()

  if examples:

    #validate adj graph construction
    print('Visualizing a adjacency graphs with example')

    num_neighbors = 10
    min_instance_size = 10

    spot_id = examples[0]

    spot_data = {}

    spot, spot_cn_cell_idxs, inst_assignments,kgr  = segment_instances(spot_id,num_neighbors=num_neighbors,min_instance_size=min_instance_size,return_kgr = True)

    spot_adj_graph = nx.Graph()

    for cn1 in range(len(data['neigh_num'].unique())):
        for cn2 in range(cn1):
            e1,e2 = kgr[inst_assignments[cn1][0],:][:,inst_assignments[cn2][0]].nonzero()
            for s,t in set(list(zip(inst_assignments[cn1][1][e1],inst_assignments[cn2][1][e2]))):
                spot_adj_graph.add_edge( (spot_id,cn1,s), (spot_id,cn2,t))

    xmin = np.min(spot['x'])
    xmax = np.max(spot['x'])

    ymin = np.min(spot['y'])
    ymax = np.max(spot['y'])

    figsize=(40,40)
    edges = list(spot_adj_graph.edges())
    np.random.seed(0)
    np.random.shuffle(edges)
    for i,edge in enumerate(edges[:40]):
        plt.subplot(8,5,i+1)
        s,t = edge
        _,cn1,inst1 = s
        _,cn2,inst2 = t
        good_cn1_cells,good_cn1_cell_inst_assignments = inst_assignments[cn1]
        good_cn2_cells,good_cn2_cell_inst_assignments = inst_assignments[cn2]


        sub_spot = spot.iloc[good_cn1_cells[good_cn1_cell_inst_assignments == inst1]]
        plt.scatter(sub_spot['x'],sub_spot['y'],c = [pal[cn1]],s = 2)

        sub_spot = spot.iloc[good_cn2_cells[good_cn2_cell_inst_assignments == inst2]]
        plt.scatter(sub_spot['x'],sub_spot['y'],c = [pal[cn2]],s = 2)

        plt.xlim(xmin,xmax)
        plt.ylim(ymin,ymax)
        plt.axis('off')
        plt.title('({},{},{},{})'.format(cn1,inst1,cn2,inst2))
    plt.show()

    # construct the adj graphs
    print('Constructing the adjaceny graphs')

    num_neighbors = 10
    min_instance_size = 10

    spot_adj_graphs = {}
    spot_data = {}


    for spot_id in spot_ids:

        spot_data[spot_id] = segment_instances(spot_id,num_neighbors=num_neighbors,min_instance_size=min_instance_size,return_kgr = True)
        spot, spot_cn_cell_idxs, inst_assignments,kgr  = spot_data[spot_id]

        for cn1 in range(len(data['neigh_num'].unique())):
            for cn2 in range(cn1):
                e1,e2 = kgr[inst_assignments[cn1][0],:][:,inst_assignments[cn2][0]].nonzero()
                for s,t in set(list(zip(inst_assignments[cn1][1][e1],inst_assignments[cn2][1][e2]))):
                    spot_adj_graphs.setdefault(spot_id, nx.Graph()).add_edge( (spot_id,cn1,s), (spot_id,cn2,t))



    # visualizing a tissue graph
    print('Visualizing a tissue graph')
    for spot_id in examples:
        spot, spot_cn_cell_idxs, inst_assignments, kgr = spot_data[spot_id]

        figsize=(10,5)


        plt.subplot(1,2,1)
        plt.scatter(spot['x'],spot['y'], c = [pal[cn] for cn in spot['neigh_num']],s  = 1)
        plt.axis('off');

        plt.subplot(1,2,2)
        pos = {}
        draw = spot_adj_graphs[spot_id]

        for n in draw.nodes():
            cn = n[1]
            cn_cells, cn_assignments = inst_assignments[cn]
            sub_spot = spot.iloc[cn_cells[cn_assignments == n[2]]]
            pos[n] = np.mean(sub_spot['x']),np.mean(sub_spot['y'])
            plt.scatter(pos[n][0],pos[n][1], c = [pal[n[1]]], s =40)

        for s,t in draw.edges():
            x = pos[s][0]
            y = pos[s][1]
            dx = (pos[t][0]-x)
            dy = (pos[t][1]-y)


            alpha = 1
            col = 'black'

            plt.arrow(x,y,dx,dy,head_length =0, head_width = 0,length_includes_head= True,alpha = alpha,lw = 0.5,color = col,zorder = -1)

        plt.axis('off');
        plt.show()

  if stats:
    spot_counts = {}
    for a,v in spot_adj_graphs.items():
        spot_counts[a] = {'edge': np.mean([v.degree(n) for n in v.nodes()]), 'node': len(v.nodes)}

    print('Plotting nodes')
    data_concat = pd.concat([pd.DataFrame({'gp':spot_group_assignment}),pd.DataFrame(spot_counts).T], axis = 1)
    sns.catplot(data = data_concat,
                x = 'gp',
                y = 'node',height = 5, s = 10)

    plt.xticks([])
    plt.ylabel('# vertices',fontsize = 15)
    plt.xlabel('')

    print('Plotting edges')
    data_concat = pd.concat([pd.DataFrame({'gp':spot_group_assignment}),pd.DataFrame(spot_counts).T], axis = 1)
    sns.catplot(data = data_concat,
                x = 'gp',
                y = 'edge',height = 5, s = 10)

    plt.xticks([])
    plt.ylabel('# vertices',fontsize = 15)
    plt.xlabel('')

    dd = pd.concat([pd.DataFrame({'gp':spot_group_assignment}),pd.DataFrame({'pat':spot_to_patient}),pd.DataFrame(spot_counts).T], axis = 1)
    dd = dd.groupby(['gp','pat'])[['edge','node']].mean().reset_index()
    print(mannwhitneyu(dd['node'][dd['gp']==group1],dd['node'][dd['gp']==group2]))
    print(mannwhitneyu(dd['edge'][dd['gp']==group1],dd['edge'][dd['gp']==group2]))

  adj_graphs = {}
  adj_graph = nx.Graph()
  for k,v in spot_adj_graphs.items():
      for s,t in v.edges():
          adj_graphs.setdefault(spot_group_assignment[k],nx.Graph()).add_edge(s,t)
          adj_graph.add_edge(s,t)

  #start by defining a statistic over a graph, e.g.

  def total_edgetype_counts(graph):
      counts = {}
      for s,t in graph.edges():
          key = tuple(sorted([s[1],t[1]]))
          counts.setdefault(key,0)
          counts[key] +=1
      return counts


  #build a partition into isomorphism up to a certain level of structure
  nodes_by_isom = {}
  nodes_by_isom[group1] = {}
  nodes_by_isom[group2] = {}

  for v in adj_graph.nodes():
      spot, cn, _ = v

      # try that any instance can be mapped to any instance *within the same spot*
      isom = spot

      #edges_Tass = tuple(sorted(set([edge[1][1] for edge in adj_graph.edges(v)])))
      nodes_by_isom[spot_group_assignment[spot]].setdefault(isom,set()).add(v)

  ty_nodes = {}
  ty_edges = {}
  for n in adj_graph.nodes():
      pat = spot_to_patient[n[0]]
      cn = n[1]
      ty_nodes.setdefault(pat,{}).setdefault(cn,set()).add(n)
      for _,e1 in adj_graph.edges(n):
          cn1 = e1[1]
          pair = tuple(sorted([cn,cn1]))
          ty_edges.setdefault(pat,{}).setdefault((cn,pair),set()).add(n)

  all_edge_types = set()
  for e in ty_edges.keys():
      all_edge_types = all_edge_types.union(ty_edges[e].keys())

  all_node_types = set()
  for e in ty_nodes.keys():
      all_node_types = all_node_types.union(ty_nodes[e].keys())

  # observed distribution
  count_base = {}
  count_edges = {}
  lift_freq = {}

  for node_type in all_node_types:
      for pat,gp in pat_to_gp.items():
          to_add = len(ty_nodes[pat].setdefault(node_type,set()))
          cb = count_base.setdefault((gp,node_type), 0)
          cb = cb + to_add
          count_base[(gp,node_type)] = cb


  for edge_type in all_edge_types:
      for pat,gp in pat_to_gp.items():
          to_add = len(ty_edges[pat].setdefault(edge_type,set()))
          ce = count_edges.setdefault((gp,edge_type), 0)
          ce = ce + to_add
          count_edges[(gp,edge_type)] = ce

      freq = count_edges[(group1,edge_type)]/(1+count_base[(group1, edge_type[0])])
      if freq > 1:
          print(edge_type)
          print(count_edges[(1,edge_type)])
          print(edge_type[0],count_base[(1,edge_type[0])])
          print('aaa')
      lift_freq['act',edge_type, 1] = count_edges[(group1,edge_type)]/(1+count_base[(group1, edge_type[0])])
      lift_freq['act',edge_type, 2] = count_edges[(group2,edge_type)]/(1+count_base[(group2, edge_type[0])])

  obs_lfs = pd.DataFrame({'gp1':{j: lift_freq['act',j,1] for j in all_edge_types }, 'gp2': {j: lift_freq['act',j,2] for j in all_edge_types}})

  cn_names = {i:i for i in range(20)}

  print('Plotting common distribution of combinations')
  figsize=(20,20)
  oo = obs_lfs
  oo['not1'] = True
  #oo['not1'] = [1 not in a[1] for a in oo.index.values ]
  q = ((oo['gp2']>.7)|(oo['gp1']>.7)) & (oo['not1'])
  g =sns.clustermap(oo[q].iloc[:,:2], figsize = (1,1));
  plt.close()

  sns.heatmap(oo[q].iloc[g.dendrogram_row.reordered_ind,:-1],cmap = 'BrBG', vmin = 0.5,vmax = 1)
  j = 0
  for s, t in oo[q].iloc[g.dendrogram_row.reordered_ind,:-1].index.values:
      j+=1

      plt.scatter(-0.4,0.75 + (j-1), c = [pal[s]], s = 100)
      plt.text(-0.4,0.75 + (j-1),cn_names[s], ha = 'center', va = 'center',fontsize = 10)
      plt.scatter(-0.55,0.2 + (j-1),c = [pal[t[0]]], s = 100)
      plt.text(-0.55,0.2 + (j-1),cn_names[t[0]], ha = 'center', va = 'center',fontsize = 10)
      plt.scatter(-0.25,0.2 + (j-1),c = [pal[t[1]]], s = 100)
      plt.text(-0.25,0.2 + (j-1),cn_names[t[1]], ha = 'center', va = 'center',fontsize = 10)
      plt.arrow(-0.4,0.75+j-1, 0, -0.4,  head_width= 0.04, color = 'black',zorder = -2)
      plt.plot([-0.55,-0.25], [0.2 + (j-1),0.2 + (j-1)],zorder = -1, color = 'black')

  plt.xticks([])
  plt.xlim(-1.5,3)

  #plt.savefig(save_path+'window10_2-chain_extensionfreq10_comm.png',bbox_inches = 'tight')
  plt.show()

  # permutation distribution of extension frequencies
  for gp_perm in range(10000):
      pat_to_gp_samp = {a:b for a,b, in zip(pat_gp['donor_gp'].values,pat_gp[col_comp].sample(frac = 1,random_state = gp_perm).values)}



      count_base = {}
      for node_type in all_node_types:
          for pat,gp in pat_to_gp_samp.items():
              to_add = len(ty_nodes[pat].setdefault(node_type,set()))
              cb = count_base.setdefault((gp,node_type), 0)
              cb = cb + to_add
              count_base[(gp,node_type)] = cb

      count_edges = {}

      for edge_type in all_edge_types:
          for pat,gp in pat_to_gp_samp.items():
              to_add = len(ty_edges[pat].setdefault(edge_type,set()))
              ce = count_edges.setdefault((gp,edge_type), 0)
              ce = ce + to_add
              count_edges[(gp,edge_type)] = ce


          lift_freq[gp_perm,edge_type, 1] = count_edges[(group1,edge_type)]/(1+count_base[(group1, edge_type[0])])
          lift_freq[gp_perm,edge_type, 2] = count_edges[(group2,edge_type)]/(1+count_base[(group2, edge_type[0])])

  # bonferroni corrected permutation p-values
  perm_dist = pd.DataFrame({i:{j:lift_freq[i,j,1]- lift_freq[i,j,2] for j in all_edge_types} for i in range(10000)})

  obs = pd.DataFrame({'obs':{j: lift_freq['act',j,1] -  lift_freq['act',j,2] for j in all_edge_types }})
  pvals = np.minimum(1.0,90*(1 - np.mean((obs.values>0)*(obs.values> perm_dist.values) + (obs.values<0)*(obs.values< perm_dist.values),axis = 1)))
  sig_lfs =  obs_lfs.loc[obs[pvals< 0.05].index.values]

  for k in sig_lfs.index.values.tolist():
    x = (sig_lfs['gp1'].loc[k[0]])[k[1]]
    y = (sig_lfs['gp2'].loc[k[0]])[k[1]]
    if x>0.5 and y<0.1:
        print('Significant differences in two chain frequency')
        print(cn_names[k[0]])

  print('Plottin significant differences in two chain frequency')
  figsize=(20,20)
  #pal = sns.color_palette('bright',11)
  for k in sig_lfs.index.values.tolist():
      x = (sig_lfs['gp1'].loc[k[0]])[k[1]]
      y = (sig_lfs['gp2'].loc[k[0]])[k[1]]
      plt.scatter(x-0.02,y, c= [pal[k[0]]], s = 100,alpha = 0.5, linewidths=1, edgecolors='black', marker='^')
      new = list(set(list(k[1]))-set([k[0]]))[0]
      plt.scatter(x+0.02,y, c= [pal[new]], s= 100,alpha = 0.5, linewidths=1, edgecolors='black', marker='^')
      plt.arrow(x-0.02,y,0.02,0,zorder = -10,lw = 1)
      if x>0.5 and y<0.1:
          plt.text(x+0.03,y,cn_names[new],fontsize = 10,va = 'bottom',ha = 'left',weight = 'bold')
          plt.text(x-0.03,y,cn_names[k[0]],fontsize = 10,va = 'bottom',ha = 'left',weight = 'bold')
      if x<0.1 and y>0.5:
          plt.text(x+0.03,y,cn_names[new],fontsize = 10,va = 'bottom',ha = 'left',weight = 'bold')
          plt.text(x-0.03,y,cn_names[k[0]],fontsize = 10,va = 'bottom',ha = 'left',weight = 'bold')
      if x>0.8 or y>0.8:
          plt.text(x+0.03,y,cn_names[new],fontsize = 10,va = 'bottom',ha = 'left',weight = 'bold')
          plt.text(x-0.03,y,cn_names[k[0]],fontsize = 10,va = 'bottom',ha = 'left',weight = 'bold')
      #print(cn_names[k[0]],'-',cn_names[new])
  plt.ylim(-0.05,1)
  plt.xlim(-0.05,1)
  plt.xticks(fontsize = 30);
  plt.yticks(fontsize = 30);
  plt.plot([-0.5,1],[-0.5,1],c = 'black')
  #plt.savefig(save_path+'window10_2-chain_frequency10_comm.png',bbox_inches = 'tight')
  plt.show()


  if permute:
    # generate a random map respecting the structure as defined by the partition
    print('performing random permutation to tissue graph 10000 times for first condition')
    np.random.seed(23)

    null_dist = []
    for i in range(10000):
        if i % 100 == 0:
          print(f"Passed {i} iterations")
        permutation = {}
        for isom,equiv_Tass in nodes_by_isom[group1].items():
            block0 = list(equiv_Tass)
            block1 = block0.copy()
            np.random.shuffle(block1)
            permutation.update({a:b for a,b in zip(block0,block1)})

        # now get a new graph (group 1 for now)
        perm_graph = nx.Graph()
        for s,t in adj_graphs[group1].edges():
            perm_graph.add_edge(permutation[s], permutation[t])

        # compute the statistic
        null_dist.append(pd.DataFrame({i:total_edgetype_counts(perm_graph)}))
    edge_null_dist = pd.concat(null_dist,axis = 1).fillna(0)

    # now we see which ones are significant after bonferroni
    edge_observed = pd.DataFrame({'obs':total_edgetype_counts(adj_graphs[group1])})
    edge_null_distfill = edge_null_dist.fillna(0) # fillna 0 is OK because it's just the missing ones from each spot
    corrected_pvalues = {}
    for pair in edge_observed.index.values:
        s1 = 1-np.mean(edge_null_distfill.loc[pair].values < edge_observed.loc[pair]['obs'])
        s2 = 1-np.mean(edge_null_distfill.loc[pair].values > edge_observed.loc[pair]['obs'])
        # leave correction until later
        corrected_pvalues[pair] = min(s1,s2)#min(1,2*len(edge_observed.index.values) * min(s1,s2))

    corrected = pd.DataFrame({'p':corrected_pvalues})
    good1 = edge_observed.loc[edge_observed['obs'] >5]
    meds1 = np.median(np.log2(good1.values/(1+edge_null_distfill.loc[good1.index.values].values)),axis = 1)
    pv1 = corrected.loc[good1.index.values]


    # generate a random map respecting the structure as defined by the partition
    np.random.seed(23)
    print('performing random permutation to tissue graph 10000 times for second condition')
    null_dist = []
    for i in range(10000):
        if i % 100 == 0:
          print(f"Passed {i} iterations")
        permutation = {}
        for isom,equiv_Tass in nodes_by_isom[group2].items():
            block0 = list(equiv_Tass)
            block1 = block0.copy()
            np.random.shuffle(block1)
            permutation.update({a:b for a,b in zip(block0,block1)})

        # now get a new graph (group 2 for now)
        perm_graph = nx.Graph()
        for s,t in adj_graphs[group2].edges():
            perm_graph.add_edge(permutation[s], permutation[t])

        # compute the statistic
        null_dist.append(pd.DataFrame({i:total_edgetype_counts(perm_graph)}))
    edge_null_dist = pd.concat(null_dist,axis = 1).fillna(0)

    # now we see which ones are significant after bonferroni
    edge_observed = pd.DataFrame({'obs':total_edgetype_counts(adj_graphs[group2])})
    edge_null_distfill = edge_null_dist.fillna(0) # fillna 0 is OK because it's just the missing ones from each spot
    corrected_pvalues = {}
    for pair in edge_observed.index.values:
        s1 = 1-np.mean(edge_null_distfill.loc[pair].values < edge_observed.loc[pair]['obs'])
        s2 = 1-np.mean(edge_null_distfill.loc[pair].values > edge_observed.loc[pair]['obs'])
        # leave correction until later
        corrected_pvalues[pair] = min(s1,s2)#min(1,2*len(edge_observed.index.values) * min(s1,s2))

    corrected = pd.DataFrame({'p':corrected_pvalues})
    good2 = edge_observed.loc[edge_observed['obs'] >5]
    meds2 = np.median(np.log2(good2.values/(1+edge_null_distfill.loc[good2.index.values].values)),axis = 1)
    pv2 = corrected.loc[good2.index.values]


    pv1_2ch = pv1.copy()
    meds1_1ch = meds1.copy()
    pv2_2ch = pv2.copy()
    meds2_2ch = meds2.copy()

    print('Plotting significant 2-chain frequencies for first condition')
    Pre_dict = {}
    figsize=(6,4)
    for j,name in enumerate(pv1_2ch.index.values):
        z = 0
        u = meds1_1ch[j]
        v = -np.log2(1e-4+pv1_2ch.values[j])

        if pv1_2ch.values[j]<0.05/len(pv1_2ch):
            a,b = name
            plt.scatter(u,v-1, c=  [pal[a]],s = 600,alpha = 0.5, linewidths=1, edgecolors='black', marker='o')
            plt.scatter(u,v+1, c=  [pal[b]], s = 600,alpha = 0.5, linewidths=1, edgecolors='black', marker='o')
            #plt.text(u,v-0.25, cn_names[a], ha = 'center', va = 'center',fontsize = 15, weight = 'bold')
            #plt.text(u,v+0.25, cn_names[b], ha = 'center', va = 'center', fontsize = 15, weight = 'bold')
            plt.plot([u,u],[v-0.25,v+0.25], c = 'black',lw = .5,zorder = -10)
            print(cn_names[a],'-',cn_names[b],': ',u)
            Pre_dict[str(cn_names[a])+'_'+str(cn_names[b])]=u
        else:
            plt.scatter(u,v, c = ['black'],marker = 'x')


    plt.xlim(-1.5,2)

    plt.xticks(fontsize = 20);
    plt.yticks(fontsize = 20);
    #plt.savefig(save_path+'2-chain_10_M_comm.png',bbox_inches = 'tight', transparent=True)

    print('Plotting significant 2-chain frequencies for second condition')
    Post_dict = {}
    figsize=(6,4)
    for j,name in enumerate(pv2_2ch.index.values):
        z = 0
        u = meds2_2ch[j]
        v = -np.log2(1e-4+pv2_2ch.values[j])

        if pv2_2ch.values[j]<0.05/len(pv2_2ch):
            a,b = name
            plt.scatter(u,v-1, c=  [pal[a]],s = 600,alpha = 0.5, linewidths=1, edgecolors='black', marker='o')
            plt.scatter(u,v+1, c=  [pal[b]], s = 600,alpha = 0.5, linewidths=1, edgecolors='black', marker='o')
            #plt.text(u,v-0.25, cn_names[a], ha = 'center', va = 'center',fontsize = 15, weight = 'bold')
            #plt.text(u,v+0.25, cn_names[b], ha = 'center', va = 'center', fontsize = 15, weight = 'bold')
            plt.plot([u,u],[v-0.25,v+0.25], c = 'black',lw = .5,zorder = -10)
            print(cn_names[a],'-',cn_names[b],': ',u)
            Post_dict[str(cn_names[a])+'_'+str(cn_names[b])]=u
        else:
            plt.scatter(u,v, c = ['black'],marker = 'x')


    plt.xlim(-1.5,2);

    plt.xticks(fontsize = 20);
    plt.yticks(fontsize = 20);
    #plt.savefig(save_path+'2-chain_10_D_comm.png',bbox_inches = 'tight', transparent=True)

    shared_keys = list(set(Post_dict.keys() ) & set(Pre_dict.keys() ) )
    Pre_keys = list(set(Pre_dict.keys()) - set(Post_dict.keys()))
    Post_keys = list(set(Post_dict.keys()) - set(Pre_dict.keys()))


    shared_neg_keys = []
    shared_pos_keys = []
    shared_Prepos_keys = []
    shared_Postpos_keys = []
    for k1 in shared_keys:
        if Post_dict[k1]<0 and Pre_dict[k1]<0:
            shared_neg_keys.append(k1)
        elif Post_dict[k1]<0 and Pre_dict[k1]>0:
            shared_Prepos_keys.append(k1)
        elif Post_dict[k1]>0 and Pre_dict[k1]<0:
            shared_Postpos_keys.append(k1)
        elif Post_dict[k1]>0 and Pre_dict[k1]>0:
            shared_pos_keys.append(k1)

    Pre_neg_keys = []
    Pre_pos_keys = []
    for k1 in Pre_keys:
        if Pre_dict[k1]<0:
            Pre_neg_keys.append(k1)
        elif Pre_dict[k1]>0:
            Pre_pos_keys.append(k1)

    Post_neg_keys = []
    Post_pos_keys = []
    for k1 in Post_keys:
        if Post_dict[k1]<0:
            Post_neg_keys.append(k1)
        elif Post_dict[k1]>0:
            Post_pos_keys.append(k1)

    sublist = shared_neg_keys + shared_pos_keys

    print('Plotting for shared keys in both pos/neg for both conditions within second condition')
    comb_dict = {}
    figsize=(20,5)
    for j,name in enumerate(pv2_2ch.index.values):
        z = 0
        u = meds2_2ch[j]
        v = -np.log2(1e-4+pv2_2ch.values[j])

        if pv2_2ch.values[j]<0.05/len(pv2):
            a,b = name
            if str(cn_names[a])+'_'+str(cn_names[b]) in sublist:
                plt.scatter(u,v-0.5, c=  [pal[a]],s = 300,alpha = 0.5, linewidths=1, edgecolors='black', marker='^')
                plt.scatter(u,v+0.5, c=  [pal[b]], s = 300,alpha = 0.5, linewidths=1, edgecolors='black', marker='^')
                #plt.text(u,v-0.5, cn_names[a], ha = 'center', va = 'center',fontsize = 15, weight = 'bold')
                #plt.text(u,v+0.5, cn_names[b], ha = 'center', va = 'center', fontsize = 15, weight = 'bold')
                plt.plot([u,u],[v-0.5,v+0.5], c = 'black',lw = .5,zorder = -10)
                print(cn_names[a],'-',cn_names[b],': ',u)
                comb_dict[str(cn_names[a])+'_'+str(cn_names[b])]=u
            else:
              # plt.scatter(u,v, c = ['black'],marker = 'x')
                continue
        else:
            plt.scatter(u,v, c = ['black'],marker = 'x')


    plt.xlim(-3,3);

    plt.xticks(fontsize = 20);
    plt.yticks(fontsize = 20);
    #plt.savefig(save_path+'window10_2-chain_10_comm_common.png',bbox_inches = 'tight')


  return sig_lfs, pv1_2ch, meds1_1ch, pv2_2ch, meds2_2ch, cn_names, Pre_dict, Post_dict



