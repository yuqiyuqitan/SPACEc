# load required packages
import os
import random
import time
from functools import reduce
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import scipy as sp
import seaborn as sns
import tifffile as tiff
from scipy.spatial.distance import cdist
from scipy.stats import pearsonr
from sklearn.cross_decomposition import CCA
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm
import warnings
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from anndata import AnnData

# sns.set_style("ticks")

# helper functions
############################################################


def hf_generate_random_colors(n, rand_seed=0):
    # from random import randint
    random.seed(rand_seed)
    color = []
    for i in range(n):
        color.append("#%06X" % random.randint(0, 0xFFFFFF))
    return color


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

    l = list(res.keys())

    SMALL_SIZE = 14
    MEDIUM_SIZE = 16
    BIGGER_SIZE = 18

    # Settings for graph
    plt.rc("font", size=SMALL_SIZE)  # controls default text sizes
    plt.rc("axes", titlesize=SMALL_SIZE)  # fontsize of the axes title
    plt.rc("axes", labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
    plt.rc("xtick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc("ytick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc("legend", fontsize=SMALL_SIZE)  # legend fontsize
    plt.rc("figure", titlesize=BIGGER_SIZE)  # fontsize of the figure title

    sns.palplot(res.values())
    plt.xticks(range(len(res)), res.keys(), rotation=30, ha="right")
    # plt.savefig(save_path+'color_legend.png', dpi=300)

    return res


#########


def hf_per_only(data, grouping, replicate, sub_col, sub_list, per_cat, norm=True):
    # Find Percentage of cell type
    if norm == True:
        test1 = data.loc[
            data[sub_col].isin(sub_list)
        ]  # filters df for values by values in sub_list which are in the sub_col column
        immune_list = list(
            test1[per_cat].unique()
        )  # stores unique values for the per_cat column
    else:
        test1 = data.copy()
        immune_list = list(data.loc[data[sub_col].isin(sub_list)][per_cat].unique())

    test1[per_cat] = test1[per_cat].astype("category")
    test_freq = test1.groupby([grouping, replicate]).apply(
        lambda x: x[per_cat].value_counts(normalize=True, sort=False) * 100
    )  # group data by grouping variable and replicates, than applies the lambda function to count the frequency of each category in the per_cat column and normalizes by dividing by the total count.
    test_freq.columns = test_freq.columns.astype(str)
    test_freq.reset_index(inplace=True)
    immune_list.extend(
        [grouping, replicate]
    )  # adds grouping and replicate column to immune_list
    test_freq1 = test_freq[immune_list]  # subsets test_freq by immune_list

    melt_per_plot = pd.melt(
        test_freq1, id_vars=[grouping, replicate]
    )  # ,value_vars=immune_list) #converts columns specified in id_vars into rows
    melt_per_plot.rename(
        columns={"value": "percentage"}, inplace=True
    )  # rename value to percentage

    return melt_per_plot  # returns a df which contains the group_column followed by the replicate column and the per category column, and a column specifying the percentage
    # Example: percentage CD4+ TCs in unique region E08 assigned to community xxx


########


def hf_normalize(X):
    arr = np.array(X.fillna(0).values)
    return pd.DataFrame(
        np.log2(1e-3 + arr / arr.sum(axis=1, keepdims=True)),
        index=X.index.values,
        columns=X.columns,
    ).fillna(0)


########


def hf_cell_types_de_helper(
    df,
    ID_component1,
    ID_component2,
    neighborhood_col,
    group_col,
    group_dict,
    cell_type_col,
):
    # read data
    cells2 = df
    cells2.reset_index(inplace=True, drop=True)
    cells2

    # generate unique ID
    cells2["donor_tis"] = cells2[ID_component1] + "_" + cells2[ID_component2]

    # This code is creating a dictionary called neigh_num that maps each unique value
    # in the Neighborhood column of a pandas DataFrame cells2 to a unique integer index
    # starting from 0.
    neigh_num = {
        list(cells2[neighborhood_col].unique())[i]: i
        for i in range(len(cells2[neighborhood_col].unique()))
    }
    cells2["neigh_num"] = cells2[neighborhood_col].map(neigh_num)
    cells2["neigh_num"].unique()

    """
    This Python code is performing the following data transformation operations on a pandas DataFrame named cells2:
    The first three lines of code create a dictionary called treatment_dict that maps two specific strings, 'SB' and 'CL', to the integers 0 and 1, respectively. Then, the map() method is used to create a new column called group, where each value in the tissue column is replaced with its corresponding integer value from the treatment_dict dictionary.
    The fourth to seventh lines of code create a new dictionary called pat_dict that maps each unique value in the donor_tis column of the cells2 DataFrame to a unique integer index starting from 0. The for loop loops through the range object and assigns each integer to the corresponding unique value in the donor_tis column, creating a dictionary that maps each unique value to a unique integer index.
    The last two lines of code create a new column called patients in the cells2 DataFrame, where each value in the donor_tis column is replaced with its corresponding integer index from the pat_dict dictionary. This code assigns these integer indices to each patient in the donor_tis column. The unique() method is used to return an array of unique values in the patients column to verify that each unique value in the donor_tis column has been mapped to a unique integer index in the patients column.
    Overall, the code is converting categorical data in the tissue and donor_tis columns to numerical data in the group and patients columns, respectively, which could be useful for certain types of analysis.
    """
    # Code treatment/group with number
    cells2["group"] = cells2[group_col].map(group_dict)
    cells2["group"].unique()

    pat_dict = {}
    for i in range(len(list(cells2["donor_tis"].unique()))):
        pat_dict[list(cells2["donor_tis"].unique())[i]] = i
    pat_dict

    cells2["patients"] = cells2["donor_tis"].map(pat_dict)
    cells2["patients"].unique()

    # drop duplicates
    pat_gp = cells2[["patients", "group"]].drop_duplicates()
    pat_to_gp = {a: b for a, b in pat_gp.values}

    # get cell type (ct) frequences per patient
    ct_freq1 = cells2.groupby(["patients"]).apply(
        lambda x: x[cell_type_col].value_counts(normalize=True, sort=False) * 100
    )
    # ct_freq = ct_freq1.to_frame()
    ct_freq = ct_freq1.unstack().fillna(0)
    ct_freq.reset_index(inplace=True)
    ct_freq.rename(
        columns={"level_1": "cell_type", "Cell Type": "Percentage"}, inplace=True
    )
    ct_freq

    # Get frequences for every neighborhood per patient
    all_freqs1 = cells2.groupby(["patients", "neigh_num"]).apply(
        lambda x: x[cell_type_col].value_counts(normalize=True, sort=False) * 100
    )
    # all_freqs = all_freqs1.to_frame()
    all_freqs = all_freqs1.unstack().fillna(0)
    all_freqs.reset_index(inplace=True)
    all_freqs.rename(
        columns={"level_2": "cell_type", cell_type_col: "Percentage"}, inplace=True
    )
    all_freqs

    return (cells2, ct_freq, all_freqs, pat_to_gp, neigh_num)


##########


def hf_get_pathcells(query_database, query_dict_list):
    """
    Return set of cells that match query_dict path.
    """
    out = []

    if type(query_dict_list) == dict:
        query_dict_list = [query_dict_list]

    for query_dict in query_dict_list:
        qd = query_database
        for k, v in query_dict.items():
            if type(v) != list:
                v = [v]
            qd = qd[qd[k].isin(v)]
        out += [qd]
    if len(query_database) == 1:
        return out[0]
    return out


# annotated
"""
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
"""


class Neighborhoods(object):
    def __init__(
        self,
        cells,
        ks,
        cluster_col,
        sum_cols,
        keep_cols,
        X="X:X",
        Y="Y:Y",
        reg="Exp",
        add_dummies=True,
    ):
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

        self.cells = pd.concat([keep, dumz], axis=1)

    def get_tissue_chunks(self):
        self.tissue_group = self.cells[[self.X, self.Y, self.reg]].groupby(self.reg)

        tissue_chunks = [
            (time.time(), self.exps.index(t), t, a)
            for t, indices in self.tissue_group.groups.items()
            for a in np.array_split(indices, 1)
        ]
        return tissue_chunks

    def make_windows(self, job):
        start_time, idx, tissue_name, indices = job
        job_start = time.time()

        print(
            "Starting:", str(idx + 1) + "/" + str(len(self.exps)), ": " + self.exps[idx]
        )

        tissue = self.tissue_group.get_group(tissue_name)
        to_fit = tissue.loc[indices][[self.X, self.Y]].values

        fit = NearestNeighbors(n_neighbors=self.n_neighbors + 1).fit(
            tissue[[self.X, self.Y]].values
        )
        m = fit.kneighbors(to_fit)
        m = m[0][:, 1:], m[1][:, 1:]

        # sort_neighbors
        args = m[0].argsort(axis=1)
        add = np.arange(m[1].shape[0]) * m[1].shape[1]
        sorted_indices = m[1].flatten()[args + add[:, None]]

        neighbors = tissue.index.values[sorted_indices]

        end_time = time.time()

        print(
            "Finishing:",
            str(idx + 1) + "/" + str(len(self.exps)),
            ": " + self.exps[idx],
            end_time - job_start,
            end_time - start_time,
        )
        return neighbors.astype(np.int32)

    def k_windows(self):
        if self.bool_add_dummies:
            self.add_dummies()
        else:
            self.cells = self.cells_nodumz
        sum_cols = list(self.sum_cols)
        for col in sum_cols:
            if col in self.keep_cols:
                self.cells[col + "_sum"] = self.cells[col]
                self.sum_cols.remove(col)
                self.sum_cols += [col + "_sum"]

        values = self.cells[self.sum_cols].values
        tissue_chunks = self.get_tissue_chunks()
        tissues = [self.make_windows(job) for job in tissue_chunks]

        out_dict = {}
        for k in self.ks:
            for neighbors, job in zip(tissues, tissue_chunks):
                chunk = np.arange(len(neighbors))  # indices
                tissue_name = job[2]
                indices = job[3]
                window = (
                    values[neighbors[chunk, :k].flatten()]
                    .reshape(len(chunk), k, len(self.sum_cols))
                    .sum(axis=1)
                )
                out_dict[(tissue_name, k)] = (window.astype(np.float16), indices)

        windows = {}
        for k in self.ks:
            window = pd.concat(
                [
                    pd.DataFrame(
                        out_dict[(exp, k)][0],
                        index=out_dict[(exp, k)][1].astype(int),
                        columns=self.sum_cols,
                    )
                    for exp in self.exps
                ],
                axis=0,
            )
            window = window.loc[self.cells.index.values]
            window = pd.concat([self.cells[self.keep_cols], window], axis=1)
            windows[k] = window
        return windows


##########


# Define a Python function named `hf_get_windows` that takes two arguments: `job` and `n_neighbors`.
def hf_get_windows(job, n_neighbors, exps, tissue_group, X, Y):
    # Unpack the tuple `job` into four variables: `start_time`, `idx`, `tissue_name`, and `indices`.
    start_time, idx, tissue_name, indices = job

    # Record the time at which the function starts.
    job_start = time.time()

    # Print a message indicating the start of the function execution, including the current job index and the corresponding experiment name.
    print("Starting:", str(idx + 1) + "/" + str(len(exps)), ": " + exps[idx])

    # Retrieve the subset of data that corresponds to the given `tissue_name`.
    tissue = tissue_group.get_group(tissue_name)

    # Select only the `X` and `Y` columns of the data corresponding to the given `indices`.
    to_fit = tissue.loc[indices][[X, Y]].values

    # Fit a model with the data that corresponds to the `X` and `Y` columns of the given `tissue`.
    fit = NearestNeighbors(n_neighbors=n_neighbors).fit(tissue[[X, Y]].values)

    # Find the indices of the `n_neighbors` nearest neighbors of the `to_fit` data points.
    # The `m` variable contains the distances and indices of these neighbors.
    m = fit.kneighbors(to_fit)

    # Sort the `m[1]` array along each row, and store the resulting indices in the `args` variable.
    args = m[0].argsort(axis=1)

    # Create the `add` variable to offset the indices based on the number of rows in `m[1]`, and store the sorted indices in the `sorted_indices` variable.
    add = np.arange(m[1].shape[0]) * m[1].shape[1]
    sorted_indices = m[1].flatten()[args + add[:, None]]

    # Create the `neighbors` variable by selecting the indices from the `tissue` data frame that correspond to the sorted indices.
    neighbors = tissue.index.values[sorted_indices]

    # Record the time at which the function ends.
    end_time = time.time()

    # Print a message indicating the end of the function execution, including the current job index and the corresponding experiment name, and the time it took to execute the function.
    print(
        "Finishing:",
        str(idx + 1) + "/" + str(len(exps)),
        ": " + exps[idx],
        end_time - job_start,
        end_time - start_time,
    )

    # Return the `neighbors` array as an array of 32-bit integers.
    return neighbors.astype(np.int32)


###################


def hf_index_rank(a, axis):
    """
    returns the index of every index in the sorted array
    haven't tested on ndarray yet
    """
    arg = np.argsort(a, axis)

    return np.arange(a.shape[axis])[np.argsort(arg, axis)]


def hf_znormalize(raw_cells, grouper, markers, clip=(-7, 7), dropinf=True):
    not_inf = raw_cells[np.isinf(raw_cells[markers].values).sum(axis=1) == 0]
    if not_inf.shape[0] != raw_cells.shape[0]:
        print("removing cells with inf values", raw_cells.shape, not_inf.shape)
    not_na = not_inf[not_inf[markers].isnull().sum(axis=1) == 0]
    if not_na.shape[0] != not_inf.shape[0]:
        print("removing cells with nan values", not_inf.shape, not_na.shape)

    znorm = not_na.groupby(grouper).apply(
        lambda x: (
            (x[markers] - x[markers].mean(axis=0)) / x[markers].std(axis=0)
        ).clip(clip[0], clip[1])
    )
    Z = not_na.drop(markers, 1).merge(znorm, left_index=True, right_index=True)
    return Z


def hf_fast_divisive_cluster(X, num_clusters, metric="cosine", prints=True):
    # optimized divisive_cluster.  Faster because doesn't recompute distance matrix to centroids at
    # each iteration
    centroids = np.zeros((num_clusters, X.shape[1]))  # fill with cluster centroids
    dists = np.zeros((X.shape[0], num_clusters))  # fill with dist matrix

    avg_seed = X.mean(axis=0, keepdims=True)
    d = cdist(X, avg_seed, metric=metric)
    dists[:, 0] = d[:, 0]
    c1 = d.argmax()
    centroids[0] = X[c1]

    for x in range(1, num_clusters):
        if x % 10 == 0:
            print(x, "clusters")
        d = cdist(X, centroids[x - 1][None, :], metric=metric)
        dists[:, x] = d[:, 0]
        allocs = dists[:, : x + 1].argmin(axis=1)
        next_centroid = dists[np.arange(len(dists)), allocs].argmax()
        centroids[x] = X[next_centroid]
    return centroids, allocs


def hf_alloc_cells(X, centroids, metric="cosine"):
    dists = cdist(X, centroids, metric=metric)
    allocs = dists.argmin(axis=1)
    return allocs


###########


def hf_get_sum_cols(cell_cuts, panel):
    arr = np.where(cell_cuts[:, 0] == panel)[0]
    return slice(arr[0], arr[-1] + 1)


###############
def hf_get_thresh_simps(x, thresh):
    sorts = np.argsort(-x, axis=1)
    x_sorted = -np.sort(-x, axis=1)
    cumsums = np.cumsum(x_sorted, axis=1)
    thresh_simps = pd.Series(
        [
            tuple(sorted(sorts[i, : (1 + j)]))
            for i, j in enumerate(np.argmax(cumsums > thresh, axis=1))
        ]
    )
    return thresh_simps


###############
def hf_prepare_neighborhood_df(
    cells_df, patient_ID_component1, patient_ID_component2, neighborhood_column=None
):
    # Spacer for output
    print("")

    # Combine two columns to form unique ID which will be stored as patients column
    cells_df["patients"] = (
        cells_df[patient_ID_component1] + "_" + cells_df[patient_ID_component2]
    )
    print("You assigned following identifiers to the column 'patients':")
    print(cells_df["patients"].unique())

    # Spacer for output
    print("")

    if neighborhood_column:
        # Assign numbers to neighborhoods
        neigh_num = {
            list(cells_df[neighborhood_column].unique())[i]: i
            for i in range(len(cells_df[neighborhood_column].unique()))
        }
        cells_df["neigh_num"] = cells_df[neighborhood_column].map(neigh_num)
        print(
            "You assigned following numbers to the column 'neigh_num'. Each number represents one neighborhood:"
        )
        print(cells_df["neigh_num"].unique())
        cells_df["neigh_num"] = cells_df["neigh_num"].astype("category")

    cells_df["patients"] = cells_df["patients"].astype("category")

    return cells_df


def hf_prepare_neighborhood_df2(
    cells_df, patient_ID_component1, patient_ID_component2, neighborhood_column=None
):
    # Spacer for output
    print("")

    # Combine two columns to form unique ID which will be stored as patients column
    cells_df["unique_ID"] = (
        cells_df[patient_ID_component1] + "_" + cells_df[patient_ID_component2]
    )
    print("You assigned following identifiers to the column 'unique_ID':")
    print(cells_df["unique_ID"].unique())

    # Spacer for output
    print("")

    if neighborhood_column:
        # Assign numbers to neighborhoods
        neigh_num = {
            list(cells_df[neighborhood_column].unique())[i]: i
            for i in range(len(cells_df[neighborhood_column].unique()))
        }
        cells_df["neigh_num"] = cells_df[neighborhood_column].map(neigh_num)
        print(
            "You assigned following numbers to the column 'neigh_num'. Each number represents one neighborhood:"
        )
        print(cells_df["neigh_num"].unique())
        cells_df["neigh_num"] = cells_df["neigh_num"].astype("category")

    cells_df["unique_ID"] = cells_df["unique_ID"].astype("category")

    return cells_df


############


def hf_cor_subset(cor_mat, threshold, cell_type):
    pairs = hf_get_top_abs_correlations(cor_mat, thresh=threshold)

    piar1 = pairs.loc[pairs["col1"] == cell_type]
    piar2 = pairs.loc[pairs["col2"] == cell_type]
    piar = pd.concat([piar1, piar2])

    pair_list = list(set(list(piar["col1"].unique()) + list(piar["col2"].unique())))

    return pair_list, piar, pairs


#############


# correlation analysis
def hf_get_redundant_pairs(df):
    """Get diagonal and lower triangular pairs of correlation matrix"""
    pairs_to_drop = set()
    cols = df.columns
    for i in range(0, df.shape[1]):
        for j in range(0, i + 1):
            pairs_to_drop.add((cols[i], cols[j]))
    return pairs_to_drop


##############

# Spatial context analysis
"""
this is the code that finds the minimal combination of CNs
required to make up a threshold percentage of assignments in a window
combinations are stored as a sorted tuple
"""


def hf_simp_rep(
    data,
    patient_col,
    tissue_column,
    subset_list_tissue,
    ttl_per_thres,
    comb_per_thres,
    thres_num=3,
):
    # Choose the windows size to continue with
    if tissue_column != None:
        w2 = data.loc[data[tissue_column].isin(subset_list_tissue)]
        print("tissue_column true")

    else:
        w2 = data.copy()
        print("tissue_column false")

    simp_list = []
    for patient in list(w2[patient_col].unique()):
        w = w2.loc[w2[patient_col] == patient]
        xm = w.loc[:, l].values / n_num

        # Get the neighborhood combinations based on the threshold
        simps = hf_get_thresh_simps(xm, ttl_per_thres)
        simp_freqs = simps.value_counts(normalize=True)
        sf = simp_freqs.to_frame()
        sf.rename(columns={0: patient}, inplace=True)
        sf.reset_index(inplace=True)
        sf.rename(columns={"index": "merge"}, inplace=True)
        simp_list.append(sf)
        # simp_sums = np.cumsum(simp_freqs)

        # thresh_cumulative = .95
        # selected_simps = simp_sums[simp_sums<=thresh_cumulative].index.values
    # selected_simps = simp_freqs[simp_freqs>=comb_per_thres].index.values

    simp_df = reduce(
        lambda left, right: pd.merge(left, right, on=["merge"], how="outer"), simp_list
    )
    # simp_df = pd.concat(simp_list, axis=0)
    # simp_df.index = simp_df.index.to_series()
    simp_df.fillna(0, inplace=True)
    simp_df.set_index("merge", inplace=True)
    simp_out = simp_df.loc[simp_df.gt(0).sum(axis=1).ge(thres_num)]

    return simp_out


################


def hf_get_top_abs_correlations(df, thresh=0.5):
    au_corr = df.corr().unstack()
    labels_to_drop = hf_get_redundant_pairs(df)
    au_corr = au_corr.drop(labels=labels_to_drop).sort_values(ascending=False)
    cc = au_corr.to_frame()
    cc.index.rename(["col1", "col2"], inplace=True)
    cc.reset_index(inplace=True)
    cc.rename(columns={0: "value"}, inplace=True)
    gt_pair = cc.loc[cc["value"].abs().gt(thresh)]
    return gt_pair


################
# help to convert dataframe after denoising into anndata
def make_anndata(
    df_nn,  # data frame coming out from denoising
    col_sum,  # this is the column index that has the last protein feature
    nonFuncAb_list,  # inspect which markers work, and drop the ones that did not work from the clustering step
):
    """
    Convert a denoised DataFrame into anndata format.

    Parameters:
        df_nn (pandas.DataFrame): Denoised data frame.
        col_sum (int): Column index of the last protein feature.
        nonFuncAb_list (list): List of markers that did not work in the clustering step.

    Returns:
        AnnData: Anndata object containing the converted data.
    """
    adata = sc.AnnData(X=df_nn.iloc[:, : col_sum + 1].drop(columns=nonFuncAb_list))
    adata.obs = df_nn.iloc[:, col_sum + 1 :]
    return adata


################
def hf_split_channels(input_path, output_folder, channel_names_file=None):
    """
    Split channels of a TIFF image and save them as separate files.

    Parameters:
        input_path (str): Path to the input TIFF image.
        output_folder (str): Path to the output folder where the channels will be saved.
        channel_names_file (str, optional): Path to the file containing channel names.
            If None, numbers will be used as channel names.

    Raises:
        FileNotFoundError: If the input_path or channel_names_file (if provided) is not found.

    Returns:
        None
    """
    # Load the large TIFF image
    print("loading image file...")
    image = tiff.imread(input_path)
    print("Done")

    print("splitting channels...")
    if channel_names_file is not None:
        # Read the channel names from the text file
        with open(channel_names_file, "r") as f:
            channel_names = f.read().splitlines()
    else:
        # Generate channel names as numbers
        channel_names = [str(i) for i in range(image.shape[0])]
    print("Done")

    # Split the channels
    channels = np.split(image, image.shape[0], axis=0)

    # Save each channel as a TIFF file with the corresponding channel name
    for i, channel in tqdm(
        enumerate(channels), total=len(channels), desc="Saving channels"
    ):
        channel = np.squeeze(channel)  # Remove single-dimensional entries

        output_filename = f"{output_folder}/{channel_names[i]}.tif"
        tiff.imsave(output_filename, channel)

    print("Channels saved successfully!")


##############
# Patch analysis


def hf_voronoi_finite_polygons_2d(vor, radius=None):
    """
    Reconstruct infinite voronoi regions in a 2D diagram to finite
    regions.

    Parameters
    ----------
    vor : Voronoi
        Input diagram
    radius : float, optional
        Distance to 'points at infinity'.

    Returns
    -------
    regions : list of tuples
        Indices of vertices in each revised Voronoi regions.
    vertices : list of tuples
        Coordinates for revised Voronoi vertices. Same as coordinates
        of input vertices, with 'points at infinity' appended to the
        end.

    """

    if vor.points.shape[1] != 2:
        raise ValueError("Requires 2D input")

    new_regions = []
    new_vertices = vor.vertices.tolist()

    center = vor.points.mean(axis=0)
    if radius is None:
        radius = vor.points.ptp().max()

    # Construct a map containing all ridges for a given point
    all_ridges = {}
    for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
        all_ridges.setdefault(p1, []).append((p2, v1, v2))
        all_ridges.setdefault(p2, []).append((p1, v1, v2))

    # Reconstruct infinite regions
    for p1, region in enumerate(vor.point_region):
        vertices = vor.regions[region]

        if all(v >= 0 for v in vertices):
            # finite region
            new_regions.append(vertices)
            continue

        # reconstruct a non-finite region
        ridges = all_ridges[p1]
        new_region = [v for v in vertices if v >= 0]

        for p2, v1, v2 in ridges:
            if v2 < 0:
                v1, v2 = v2, v1
            if v1 >= 0:
                # finite ridge: already in the region
                continue

            # Compute the missing endpoint of an infinite ridge
            t = vor.points[p2] - vor.points[p1]  # tangent
            t /= np.linalg.norm(t)
            n = np.array([-t[1], t[0]])  # normal

            midpoint = vor.points[[p1, p2]].mean(axis=0)
            direction = np.sign(np.dot(midpoint - center, n)) * n
            far_point = vor.vertices[v2] + direction * radius * 10

            new_region.append(len(new_vertices))
            new_vertices.append(far_point.tolist())

        # sort region counterclockwise
        vs = np.asarray([new_vertices[v] for v in new_region])
        c = vs.mean(axis=0)
        angles = np.arctan2(vs[:, 1] - c[1], vs[:, 0] - c[0])
        new_region = np.array(new_region)[np.argsort(angles)]

        # finish
        new_regions.append(new_region.tolist())

    return new_regions, np.asarray(new_vertices)


###


def hf_list_folders(directory):
    """
    Retrieve a list of folders in a given directory.

    Parameters:
        directory (str): Path to the directory.

    Returns:
        list: List of folder names within the specified directory.
    """
    folders = []
    for item in os.listdir(directory):
        item_path = os.path.join(directory, item)
        if os.path.isdir(item_path):
            folders.append(item)
    return folders


###


def hf_process_dataframe(df):
    """
    Extract information from a pandas DataFrame containing file paths and IDs.

    Parameters:
        df (pandas.DataFrame): The input DataFrame containing file paths and IDs.

    Returns:
        tuple: A tuple containing the voronoi_path, mask_path, and region for the first row.
    """
    for index, row in df.iterrows():
        voronoi_path = row["voronoi_path"]
        mask_path = row["mask_path"]
        region = row["region_long"]
        # Process the voronoi_path and mask_path variables as needed
        # Here, we are printing them for demonstration purposes
        print(f"Voronoi Path: {voronoi_path}")
        print(f"Mask Path: {mask_path}")

        return voronoi_path, mask_path, region


###


def hf_get_png_files(directory):
    """
    Get a list of PNG files in a given directory.

    Parameters:
        directory (str): Path to the directory.

    Returns:
        list: List of PNG file paths within the specified directory.
    """
    png_files = []
    for filename in os.listdir(directory):
        if filename.endswith(".png"):
            png_files.append(os.path.join(directory, filename))
    return png_files


###


def hf_find_tiff_file(directory, prefix):
    """
    Find a TIFF file in a given directory with a specified prefix.

    Parameters:
        directory (str): Path to the directory.
        prefix (str): Prefix of the TIFF file name.

    Returns:
        str or None: Path to the found TIFF file, or None if no matching file is found.
    """
    for filename in os.listdir(directory):
        if filename.endswith(".tif") and filename.startswith(prefix):
            return os.path.join(directory, filename)
    return None


###


def hf_extract_filename(filepath):
    """
    Extract the filename from a given filepath.

    Parameters:
        filepath (str): The input filepath.

    Returns:
        str: The extracted filename without the extension.
    """
    filename = os.path.basename(filepath)  # Extracts the last element of the path
    filename = filename.replace(
        "_plot.png_cut.png", ""
    )  # Removes the ".png_cut.png" extension
    return filename


###


def hf_get_tif_filepaths(directory):
    """
    Recursively searches the specified directory and its subdirectories for TIFF files (.tif) and returns a list of their file paths.

    Args:
        directory (str): The directory path to search for TIFF files.

    Returns:
        list: A list of file paths for all TIFF files found in the directory and its subdirectories.
    """
    tif_filepaths = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".tif"):
                tif_filepaths.append(os.path.join(root, file))
    return tif_filepaths


################ CCA
def hf_prepare_cca(df, neighborhood_column, subsets=None):
    neigh_num = {
        list(df[neighborhood_column].unique())[i]: i
        for i in range(len(df[neighborhood_column].unique()))
    }
    df["neigh_num"] = df[neighborhood_column].map(neigh_num)
    df["neigh_num"] = df["neigh_num"].astype("category")

    cca = CCA(n_components=1, max_iter=5000)
    func = pearsonr

    # select which neighborhoods and functional subsets
    cns = list(df["neigh_num"].unique())
    # print(cns)

    # log (1e-3 + neighborhood specific cell type frequency) of functional subsets ('nsctf')
    if subsets is not None:
        nsctf = np.log(
            1e-3
            + df.groupby(["unique_region", "neigh_num"])[subsets]
            .mean()
            .reset_index()
            .set_index(["neigh_num", "unique_region"])
        )
        # print(nsctf)
    else:
        nsctf = np.log(
            1e-3
            + df.groupby(["unique_region", "neigh_num"])
            .mean()
            .reset_index()
            .set_index(["neigh_num", "unique_region"])
        )
        # print(nsctf)

    cca = CCA(n_components=1, max_iter=5000)
    func = pearsonr
    nsctf = nsctf.fillna(1e-3)

    return df, cns, nsctf, cca, func, neigh_num


def invert_dictionary(dictionary):
    inverted_dict = {value: key for key, value in dictionary.items()}
    return inverted_dict


def hf_replace_names(color_dict, name_dict):
    color_dict = hf_invert_dictionary(color_dict)
    for color, name in color_dict.items():
        if name in name_dict:
            color_dict[color] = name_dict[name]
    color_dict = hf_invert_dictionary(color_dict)
    return color_dict


def hf_annotate_cor_plot(x, y, **kws):
    data = kws["data"]
    r, p = sp.stats.pearsonr(data[x], data[y])
    ax = plt.gca()
    ax.text(0.5, 0.8, f"r={r:.2f}, p={p:.2g}", transform=ax.transAxes, fontsize=14)


def is_dark(color):
    """
    Determines if a color is dark based on its RGB values.

    Parameters:
    color (str): The color to check. This can be any valid color string accepted by mcolors.to_rgb().

    Returns:
    bool: True if the color is dark, False otherwise. A color is considered dark if its brightness is less than 0.5.
    """
    r, g, b = mcolors.to_rgb(color)
    brightness = (r * 299 + g * 587 + b * 114) / 1000
    return brightness < 0.5


def anndata_to_GPU(
    adata: AnnData,
    layer: str | None = None,
    convert_all: bool = False,
) -> AnnData:
    """
    Transfers matrices and arrays to the GPU

    Parameters
    ----------
    adata
        AnnData object

    layer
        Layer to use as input instead of `X`. If `None`, `X` is used.

    convert_all
        If True, move all supported arrays and matrices on the GPU

    Returns
    -------
    Returns an updated copy with data on GPU
    """

    adata_gpu = adata.copy()

    if convert_all:
        anndata_to_GPU(adata_gpu)
        if adata_gpu.layers:
            for key in adata_gpu.layers.keys():
                anndata_to_GPU(adata_gpu, layer=key)
    else:
        X = _get_obs_rep(adata_gpu, layer=layer)
        if isspmatrix_csr_cpu(X):
            X = csr_matrix_gpu(X)
        elif isspmatrix_csc_cpu(X):
            X = csc_matrix_gpu(X)
        elif isinstance(X, np.ndarray):
            # Convert to CuPy array only when necessary for GPU computations
            X_gpu = cp.asarray(X)
            X = X_gpu
        else:
            error = layer if layer else "X"
            warnings.warn(f"{error} not supported for GPU conversion", Warning)

        _set_obs_rep(adata_gpu, X, layer=layer)

    return adata_gpu


def anndata_to_CPU(
    adata: AnnData,
    layer: str | None = None,
    convert_all: bool = False,
    copy: bool = False,
) -> AnnData | None:
    """
    Transfers matrices and arrays from the GPU

    Parameters
    ----------
    adata
        AnnData object

    layer
        Layer to use as input instead of `X`. If `None`, `X` is used.

    convert_all
        If True, move all GPU based arrays and matrices to the host memory

    copy
        Whether to return a copy or update `adata`.

    Returns
    -------
    Updates `adata` inplace or returns an updated copy
    """

    if copy:
        adata = adata.copy()

    if convert_all:
        anndata_to_CPU(adata)
        if adata.layers:
            for key in adata.layers.keys():
                anndata_to_CPU(adata, layer=key)
    else:
        X = _get_obs_rep(adata, layer=layer)
        if isspmatrix_csr_gpu(X):
            X = X.get()
        elif isspmatrix_csc_gpu(X):
            X = X.get()
        elif isinstance(X, cp.ndarray):
            X = X.get()
        else:
            pass

        _set_obs_rep(adata, X, layer=layer)

    if copy:
        return adata
