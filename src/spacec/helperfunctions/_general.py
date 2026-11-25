# load required packages
import importlib.metadata
import os
import random
import subprocess
import sys
import time
import warnings
from functools import reduce
from typing import TYPE_CHECKING

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import scipy as sp
import seaborn as sns
import tifffile as tiff
from cellpose.core import use_gpu
from scipy.spatial.distance import cdist
from scipy.stats import pearsonr
from sklearn.cross_decomposition import CCA
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm

from .._shared.segmentation import load_image_dictionary

# sns.set_style("ticks")

# helper functions
############################################################


def hf_generate_random_colors(n, rand_seed=0):
    """
    Generate a list of random hex color strings.

    Parameters
    ----------
    n : int
        Number of random colors to generate.
    rand_seed : int, optional
        Seed for the random number generator to ensure reproducibility.
        Default is 0.

    Returns
    -------
    list of str
        List of ``n`` color strings in ``"#RRGGBB"`` hex format.
    """
    # from random import randint
    random.seed(rand_seed)
    color = []
    for i in range(n):
        color.append("#%06X" % random.randint(0, 0xFFFFFF))
    return color


#########


def hf_assign_colors(names, colors):
    """
    Pair a list of names with a list of colors and display the resulting palette.

    Each name is matched one-to-one with a color in order. The mapping is
    printed to stdout and visualized as a seaborn color palette with labeled
    ticks.

    Parameters
    ----------
    names : list of str
        Category or cell-type names to assign colors to.
    colors : list of str
        Color values (hex strings or any matplotlib-compatible color
        specification) to assign. Must have at least as many entries as
        ``names``; excess colors are discarded.

    Returns
    -------
    dict
        Dictionary mapping each name to its assigned color.
    """
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
    """
    Compute the percentage of each category within grouped data.

    Optionally filters the DataFrame to a subset of rows before computing
    per-category frequencies, then melts the result into long format.

    Parameters
    ----------
    data : pandas.DataFrame
        Input cell-level DataFrame.
    grouping : str
        Column name used as the primary grouping variable (e.g. condition).
    replicate : str
        Column name used as the secondary grouping variable (e.g. sample ID).
    sub_col : str
        Column name used for subsetting rows when ``norm=True``.
    sub_list : list
        Values in ``sub_col`` to retain for the subset filter.
    per_cat : str
        Column whose category frequencies are computed (e.g. cell type).
    norm : bool, optional
        If ``True`` (default), filter rows to those where ``sub_col`` is in
        ``sub_list`` before computing frequencies. If ``False``, use all rows
        but only report categories present in the filtered subset.

    Returns
    -------
    pandas.DataFrame
        Long-format DataFrame with columns for ``grouping``, ``replicate``,
        ``per_cat``, and ``"percentage"``.
    """
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
    # Example: percentage CD4+ TCs in unique region E08 assigned to community 1


########


def hf_normalize(X):
    """
    Log-normalize a DataFrame of raw counts.

    NaN values are filled with zero before normalization. Each row is divided
    by its sum (relative abundance), then a log2 transform with a small
    pseudocount is applied.

    Parameters
    ----------
    X : pandas.DataFrame
        DataFrame of raw numeric values (e.g. cell-type counts per region).
        Rows are observations; columns are features.

    Returns
    -------
    pandas.DataFrame
        Log2-normalized DataFrame with the same index and columns as ``X``,
        with any remaining NaN values replaced by zero.
    """
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
    """
    Prepare a cell DataFrame for differential expression analysis across neighborhoods.

    Generates unique donor/tissue identifiers and numeric encodings for
    neighborhoods, groups, and patients. Also computes overall cell-type
    frequency per patient and per-neighborhood cell-type frequency.

    Parameters
    ----------
    df : pandas.DataFrame
        Cell-level DataFrame containing metadata and cell-type annotations.
    ID_component1 : str
        First column used to build the unique donor-tissue identifier (e.g. donor ID).
    ID_component2 : str
        Second column used to build the unique donor-tissue identifier (e.g. tissue region).
    neighborhood_col : str
        Column containing neighborhood labels.
    group_col : str
        Column containing group/condition labels (e.g. treatment).
    group_dict : dict
        Mapping from group label strings to integer codes.
    cell_type_col : str
        Column containing cell-type annotations.

    Returns
    -------
    tuple
        A 5-element tuple ``(cells2, ct_freq, all_freqs, pat_to_gp, neigh_num)``
        where:

        cells2 : pandas.DataFrame
            Input DataFrame with added columns ``donor_tis``, ``neigh_num``,
            ``group``, and ``patients``.
        ct_freq : pandas.DataFrame
            Wide-format DataFrame of overall cell-type frequencies per patient.
        all_freqs : pandas.DataFrame
            Wide-format DataFrame of cell-type frequencies per patient per
            neighborhood.
        pat_to_gp : dict
            Mapping from patient integer code to group integer code.
        neigh_num : dict
            Mapping from neighborhood label to integer code.
    """
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
        """
        One-hot-encode the cluster column and attach it to the kept columns.

        Reads ``self.cells_nodumz``, creates binary dummy variables from
        ``self.cluster_col``, concatenates them with the columns listed in
        ``self.keep_cols``, and stores the result in ``self.cells``.

        Returns
        -------
        None
            Result is stored in ``self.cells`` in-place.
        """
        c = self.cells_nodumz
        dumz = pd.get_dummies(c[self.cluster_col])
        keep = c[self.keep_cols]

        self.cells = pd.concat([keep, dumz], axis=1)

    def get_tissue_chunks(self):
        """
        Group cells by tissue/experiment and split each tissue into processing chunks.

        Stores a GroupBy object in ``self.tissue_group`` (keyed by the region
        column) and returns a list of job tuples, one per chunk, ready to be
        passed to :meth:`make_windows`.

        Returns
        -------
        list of tuple
            Each tuple contains ``(start_time, tissue_index, tissue_name, indices)``
            where ``indices`` is an array of DataFrame row labels belonging to
            that chunk.
        """
        self.tissue_group = self.cells[[self.X, self.Y, self.reg]].groupby(self.reg)

        tissue_chunks = [
            (time.time(), self.exps.index(t), t, a)
            for t, indices in self.tissue_group.groups.items()
            for a in np.array_split(indices, 1)
        ]
        return tissue_chunks

    def make_windows(self, job):
        """
        Compute the sorted nearest-neighbor indices for one tissue chunk.

        Fits a k-nearest-neighbor model on all cells in the tissue, queries
        it with the cells in the given chunk, and returns neighbor indices
        sorted by distance (closest first, excluding the query cell itself).

        Parameters
        ----------
        job : tuple
            A 4-element tuple ``(start_time, idx, tissue_name, indices)`` as
            produced by :meth:`get_tissue_chunks`.

        Returns
        -------
        numpy.ndarray of int32
            Array of shape ``(n_chunk_cells, n_neighbors)`` containing the
            DataFrame index values of the nearest neighbors for each cell in
            the chunk, sorted by ascending distance.
        """
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
        """
        Build neighborhood composition windows for every k in ``self.ks``.

        Optionally one-hot-encodes the cluster column (via :meth:`add_dummies`),
        computes nearest neighbors for all tissue chunks, then aggregates
        cell-type (or marker) values over each window of k neighbors.

        Returns
        -------
        dict
            Dictionary mapping each k value to a pandas.DataFrame whose rows
            correspond to cells and whose columns contain the summed
            cell-type/marker values over the k-nearest-neighbor window, prepended
            by the columns listed in ``self.keep_cols``.
        """
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
    """
    Compute sorted nearest-neighbor indices for a single tissue chunk.

    Fits a k-nearest-neighbor model on all cells in the tissue, queries it
    with the subset of cells specified by ``job``, and returns their neighbor
    indices sorted by distance.

    Parameters
    ----------
    job : tuple
        A 4-element tuple ``(start_time, idx, tissue_name, indices)`` where
        ``start_time`` is a float timestamp, ``idx`` is the integer position
        of the tissue in ``exps``, ``tissue_name`` is the tissue label, and
        ``indices`` are the DataFrame row labels of cells to query.
    n_neighbors : int
        Number of nearest neighbors to find for each query cell.
    exps : list of str
        Ordered list of all tissue/experiment names (used for progress output).
    tissue_group : pandas.core.groupby.DataFrameGroupBy
        GroupBy object keyed by tissue name, containing X/Y coordinate columns.
    X : str
        Column name for the x-coordinate in the tissue group.
    Y : str
        Column name for the y-coordinate in the tissue group.

    Returns
    -------
    numpy.ndarray of int32
        Array of shape ``(n_query_cells, n_neighbors)`` containing the
        DataFrame index values of the nearest neighbors for each query cell,
        sorted by ascending distance.
    """
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
    """
    Z-score normalize marker intensities within groups and clip extreme values.

    Removes cells with infinite or NaN marker values before computing
    per-group z-scores. The z-scores are clipped to a configurable range.

    Parameters
    ----------
    raw_cells : pandas.DataFrame
        Cell-level DataFrame containing marker columns and metadata.
    grouper : str or list of str
        Column(s) to group by before computing mean and standard deviation
        (e.g. tissue region or sample ID).
    markers : list of str
        Column names of the marker intensities to normalize.
    clip : tuple of float, optional
        ``(lower, upper)`` bounds to clip z-scores to. Default is ``(-7, 7)``.
    dropinf : bool, optional
        Retained for API compatibility; infinite values are always removed.
        Default is ``True``.

    Returns
    -------
    pandas.DataFrame
        DataFrame with the same non-marker columns as ``raw_cells`` plus
        z-scored marker columns, restricted to cells that had no infinite
        or NaN marker values.
    """
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
    """
    Perform fast divisive (top-down) clustering by iterative furthest-point selection.

    An optimized divisive clustering algorithm that avoids recomputing the
    full distance matrix to all centroids at each iteration. The first
    centroid is the point furthest from the data mean; subsequent centroids
    are chosen as the point furthest from its nearest existing centroid.

    Parameters
    ----------
    X : numpy.ndarray
        Data matrix of shape ``(n_samples, n_features)``.
    num_clusters : int
        Number of clusters to produce.
    metric : str, optional
        Distance metric passed to ``scipy.spatial.distance.cdist``.
        Default is ``"cosine"``.
    prints : bool, optional
        If ``True``, print progress every 10 clusters. Default is ``True``.

    Returns
    -------
    centroids : numpy.ndarray
        Array of shape ``(num_clusters, n_features)`` containing the final
        cluster centroids.
    allocs : numpy.ndarray of int
        Array of shape ``(n_samples,)`` with the cluster index assigned to
        each sample.
    """
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
    """
    Assign each sample to its nearest centroid.

    Parameters
    ----------
    X : numpy.ndarray
        Data matrix of shape ``(n_samples, n_features)``.
    centroids : numpy.ndarray
        Centroid matrix of shape ``(n_clusters, n_features)``.
    metric : str, optional
        Distance metric passed to ``scipy.spatial.distance.cdist``.
        Default is ``"cosine"``.

    Returns
    -------
    numpy.ndarray of int
        Array of shape ``(n_samples,)`` with the index of the nearest centroid
        for each sample.
    """
    dists = cdist(X, centroids, metric=metric)
    allocs = dists.argmin(axis=1)
    return allocs


###########


def hf_get_sum_cols(cell_cuts, panel):
    """
    Return a slice covering all columns belonging to a given panel.

    Parameters
    ----------
    cell_cuts : numpy.ndarray
        2-D array whose first column contains panel identifiers and whose
        rows define column ranges.
    panel : object
        Panel identifier to search for in the first column of ``cell_cuts``.

    Returns
    -------
    slice
        A slice ``slice(first_index, last_index + 1)`` spanning all rows of
        ``cell_cuts`` that match ``panel``.
    """
    arr = np.where(cell_cuts[:, 0] == panel)[0]
    return slice(arr[0], arr[-1] + 1)


###############
def hf_get_thresh_simps(x, thresh):
    """
    Find the minimal sorted combination of columns that cumulatively exceeds a threshold.

    For each row of ``x``, columns are added in descending order of value until
    their cumulative sum exceeds ``thresh``. The resulting column indices are
    stored as a sorted tuple (a "simplex").

    Parameters
    ----------
    x : numpy.ndarray
        2-D array of shape ``(n_windows, n_neighborhoods)`` containing
        fractional neighborhood compositions (rows should sum to approximately 1).
    thresh : float
        Cumulative-sum threshold. The smallest set of top columns whose sum
        exceeds this value is returned for each row.

    Returns
    -------
    pandas.Series
        Series of length ``n_windows`` where each element is a sorted tuple
        of column indices representing the minimal neighborhood combination.
    """
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
    """
    Add a ``"patients"`` identifier column and optionally encode neighborhoods numerically.

    Combines two metadata columns into a unique patient/sample identifier and,
    if a neighborhood column is supplied, maps neighborhood labels to integer
    codes stored in a ``"neigh_num"`` column.

    Parameters
    ----------
    cells_df : pandas.DataFrame
        Cell-level DataFrame.
    patient_ID_component1 : str
        First column to concatenate for the patient identifier (e.g. donor ID).
    patient_ID_component2 : str
        Second column to concatenate for the patient identifier (e.g. tissue).
    neighborhood_column : str, optional
        Column containing neighborhood labels. When provided, a numeric mapping
        ``"neigh_num"`` is added to ``cells_df``. Default is ``None``.

    Returns
    -------
    pandas.DataFrame
        Modified ``cells_df`` with a new categorical ``"patients"`` column and,
        if requested, a categorical ``"neigh_num"`` column.
    """
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
    """
    Add a ``"unique_ID"`` identifier column and optionally encode neighborhoods numerically.

    Variant of :func:`hf_prepare_neighborhood_df` that stores the combined
    patient/sample identifier under ``"unique_ID"`` instead of ``"patients"``.

    Parameters
    ----------
    cells_df : pandas.DataFrame
        Cell-level DataFrame.
    patient_ID_component1 : str
        First column to concatenate for the unique identifier (e.g. donor ID).
    patient_ID_component2 : str
        Second column to concatenate for the unique identifier (e.g. tissue).
    neighborhood_column : str, optional
        Column containing neighborhood labels. When provided, a numeric mapping
        ``"neigh_num"`` is added to ``cells_df``. Default is ``None``.

    Returns
    -------
    pandas.DataFrame
        Modified ``cells_df`` with a new categorical ``"unique_ID"`` column and,
        if requested, a categorical ``"neigh_num"`` column.
    """
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
    """
    Extract correlation pairs involving a specific cell type above a given threshold.

    Filters the upper-triangular correlation matrix to pairs whose absolute
    correlation exceeds ``threshold`` and that include ``cell_type`` as one of
    the two variables.

    Parameters
    ----------
    cor_mat : pandas.DataFrame
        Square correlation matrix with cell-type labels as both index and columns.
    threshold : float
        Minimum absolute correlation value to retain a pair.
    cell_type : str
        Cell type of interest; only pairs containing this label are returned.

    Returns
    -------
    pair_list : list of str
        Unique cell-type names found in the filtered pairs (includes ``cell_type``
        itself and all its correlated partners).
    piar : pandas.DataFrame
        Subset of the full pairs table containing only rows where ``cell_type``
        appears in ``col1`` or ``col2``.
    pairs : pandas.DataFrame
        Full pairs table of all correlation pairs above ``threshold``, with
        columns ``col1``, ``col2``, and ``value``.
    """
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
    """
    Compute simplified neighborhood-combination representations per patient.

    For each patient, finds the minimal combinations of cellular neighborhoods
    (simplices) that together account for a threshold fraction of window
    assignments. Retains only simplices that appear in at least ``thres_num``
    patients.

    Parameters
    ----------
    data : pandas.DataFrame
        Window-level DataFrame containing neighborhood composition columns,
        a patient identifier column, and an optional tissue column.
    patient_col : str
        Column name identifying patients/samples.
    tissue_column : str or None
        Column name for tissue type filtering. Pass ``None`` to use all rows.
    subset_list_tissue : list
        Values in ``tissue_column`` to retain. Ignored when ``tissue_column``
        is ``None``.
    ttl_per_thres : float
        Cumulative-fraction threshold passed to :func:`hf_get_thresh_simps`.
        Neighborhoods are added until their combined fraction exceeds this value.
    comb_per_thres : float
        Minimum frequency threshold for a simplex within a patient (retained
        for API compatibility; filtering is controlled by ``thres_num``).
    thres_num : int, optional
        Minimum number of patients a simplex must appear in to be included in
        the output. Default is 3.

    Returns
    -------
    pandas.DataFrame
        Wide-format DataFrame indexed by simplex tuples, with one column per
        patient containing the simplex frequency within that patient. Only
        simplices present in at least ``thres_num`` patients are included.
    """
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
    """
    Return all unique variable pairs whose absolute Pearson correlation exceeds a threshold.

    Diagonal and lower-triangular pairs are dropped to avoid redundancy.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame of numeric values; correlations are computed between columns.
    thresh : float, optional
        Minimum absolute correlation value to retain. Default is 0.5.

    Returns
    -------
    pandas.DataFrame
        DataFrame with columns ``col1``, ``col2``, and ``value``, containing
        only pairs whose absolute correlation exceeds ``thresh``, sorted in
        descending order of correlation.
    """
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
    df_nn,
    col_sum=None,
    nonFuncAb_list=None,
    x_col="x",
    y_col="y",
    obs_names_col="unique_cell_id",
    marker_list=None,
):
    """
    Convert a denoised DataFrame into an AnnData object.

    If coordinate columns are available, spatial coordinates are stored in
    ``adata.obsm["spatial"]`` with columns ordered as [x, y].

    Parameters
    ----------
    df_nn : DataFrame
        The denoised data (e.g. from ``sp.pp.remove_noise``).
    col_sum : int, optional
        Deprecated alias for ``marker_list``, kept only for backwards compatibility --
        raises a ``DeprecationWarning``. Selects the markers positionally (the first
        ``col_sum + 1`` columns of ``df_nn``) rather than by name. Mutually exclusive
        with ``marker_list``. Will be removed in a future version.
    marker_list : list of str, optional
        The marker columns to use as ``adata.X``, specified directly by name. Required
        unless the deprecated ``col_sum`` is given instead.
    nonFuncAb_list : list of str, optional
        Markers (from ``marker_list``) to drop entirely rather than include in
        ``adata.X`` -- e.g. antibodies that didn't stain as expected.
    x_col, y_col : str, optional
        Coordinate columns, by default ``"x"``/``"y"``.
    obs_names_col : str, optional
        Column used to set ``adata.obs_names``, by default ``"unique_cell_id"``.
    """
    import warnings

    if marker_list is not None and col_sum is not None:
        raise ValueError(
            "'marker_list' and 'col_sum' are mutually exclusive -- pass one or the "
            "other."
        )
    if marker_list is None:
        if col_sum is None:
            raise ValueError(
                "make_anndata() requires either 'marker_list' or the deprecated "
                "'col_sum'."
            )
        warnings.warn(
            "'col_sum' is deprecated and will be removed in a future version. "
            "Use 'marker_list' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        marker_list = list(df_nn.columns[: col_sum + 1])

    columns_to_drop = [] if nonFuncAb_list is None else list(nonFuncAb_list)

    feature_df = df_nn.loc[:, marker_list].drop(
        columns=columns_to_drop,
        errors="ignore",
    )

    adata = sc.AnnData(X=feature_df.copy())
    adata.obs = df_nn.drop(columns=marker_list).copy()
    # newer anndata no longer stringifies an int index on .obs assignment
    adata.obs_names = adata.obs_names.astype(str)

    if obs_names_col is not None and obs_names_col in adata.obs.columns:
        adata.obs_names = adata.obs[obs_names_col].astype(str).to_numpy()

        if not adata.obs_names.is_unique:
            warnings.warn(
                f"adata.obs_names built from '{obs_names_col}' are not unique. "
                "Check for duplicate region identifiers or IDs in the input DataFrame.",
                stacklevel=2,
            )

    missing_coordinate_columns = [
        column for column in (x_col, y_col) if column not in adata.obs.columns
    ]

    if missing_coordinate_columns:
        warnings.warn(
            "Spatial coordinates were not added to `adata.obsm['spatial']` "
            f"because the following coordinate column(s) are missing from "
            f"`adata.obs`: {missing_coordinate_columns}. "
            f"Currently requested columns are x_col='{x_col}' and y_col='{y_col}'. "
            "Specify the correct column names using the `x_col` and `y_col` "
            "arguments.",
            UserWarning,
            stacklevel=2,
        )
    else:
        adata.obsm["spatial"] = adata.obs[[x_col, y_col]].to_numpy(dtype=float)

    return adata


def hf_sync_spatial_obsm(adata, x_col="x", y_col="y"):
    """
    Synchronise ``adata.obsm["spatial"]`` from spatial coordinate columns in ``adata.obs``.

    Call this after any operation that modifies spatial coordinates stored in
    ``adata.obs`` (e.g. registration correction, sub-setting, concatenation) to
    keep ``obsm["spatial"]`` consistent.  The coordinate columns are left
    untouched in ``adata.obs`` so that existing analysis functions continue to
    work without modification.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix.
    x_col : str, optional
        Column name for x-coordinates in ``adata.obs``, by default ``"x"``.
    y_col : str, optional
        Column name for y-coordinates in ``adata.obs``, by default ``"y"``.

    Returns
    -------
    AnnData
        The same ``adata`` object with ``obsm["spatial"]`` updated in-place.

    Raises
    ------
    KeyError
        If ``x_col`` or ``y_col`` is not found in ``adata.obs``.
    """
    missing = [c for c in (x_col, y_col) if c not in adata.obs.columns]
    if missing:
        raise KeyError(
            f"Coordinate column(s) {missing} not found in adata.obs. "
            f"Available columns: {list(adata.obs.columns)}"
        )
    adata.obsm["spatial"] = adata.obs[[x_col, y_col]].to_numpy(dtype=float)
    return adata


def hf_attach_masks(adata, masks, region_col="unique_region", uns_key="masks"):
    """
    Attach raw integer-label segmentation masks to an AnnData object for later
    per-cell pixel lookup. Storing masks under ``adata.uns["masks"]`` keyed by
    region is consumed directly by :func:`spacec.pl.plot_masks`.

    A cell's exact pixels can then be recovered from any ``adata.obs`` row via::

        mask = adata.uns["masks"][row[region_col]]
        cell_pixels = mask == int(row["label"])

    This works even for cells removed by ``size_cutoff`` during segmentation:
    their label is simply absent from ``adata.obs["label"]`` while still present
    as pixels in the attached mask.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix.
    masks : dict[str, numpy.ndarray]
        Mapping of region identifier (matching ``adata.obs[region_col]`` values)
        to 2D integer-labeled mask arrays.
    region_col : str, optional
        Column in ``adata.obs`` whose values are the expected keys of ``masks``,
        by default ``"unique_region"``.
    uns_key : str, optional
        Key under ``adata.uns`` to store the masks dict, by default ``"masks"``.

    Returns
    -------
    AnnData
        The same ``adata`` with ``adata.uns[uns_key]`` updated (merged with any
        previously attached masks, not overwritten) in-place.
    """
    if region_col in adata.obs.columns:
        known_regions = set(adata.obs[region_col].astype(str).unique())
        unknown = set(masks) - known_regions
        if unknown:
            warnings.warn(
                f"masks keys not found in adata.obs['{region_col}']: {unknown}",
                stacklevel=2,
            )
    existing = adata.uns.get(uns_key, {})
    existing.update(masks)
    adata.uns[uns_key] = existing
    return adata


def hf_load_masks(mask_paths):
    """
    Load raw label masks previously saved by
    ``cell_segmentation(..., save_mask=True)`` for use with
    :func:`hf_attach_masks`.

    Parameters
    ----------
    mask_paths : dict[str, str or pathlib.Path]
        Mapping of region identifier to the path of a mask ``.tif`` written by
        ``cell_segmentation(save_mask=True)``.

    Returns
    -------
    dict[str, numpy.ndarray]
        Mapping of region identifier to the loaded 2D integer-labeled mask array.
    """
    return {region: tiff.imread(path) for region, path in mask_paths.items()}


def hf_load_image_dict(
    file_name, channel_file, input_format="Multichannel", nuclei_channel="DAPI"
):
    """
    Load a region's channel images by path, for use with
    :func:`spacec.pp.compensate_cell_matrix`. Complements
    :func:`hf_load_masks`, letting a region's mask and images be loaded
    from disk independently rather than unpickling a saved
    ``cell_segmentation`` output dict.

    Parameters
    ----------
    file_name : str or pathlib.Path
        Path to the region's image file (for ``input_format`` in
        ``{"Multichannel", "CODEX"}``), or to the directory of per-channel
        TIFFs (for ``input_format="Channels"``).
    channel_file : str or pathlib.Path
        Path to a text file listing channel names, one per line. Used for
        ``"Multichannel"``/``"CODEX"``; ignored for ``"Channels"``.
    input_format : str, optional
        One of ``"Multichannel"``, ``"CODEX"``, or ``"Channels"``, by
        default ``"Multichannel"``.
    nuclei_channel : str, optional
        Name of the nuclei channel, must be present in the loaded
        channels, by default ``"DAPI"``.

    Returns
    -------
    dict[str, numpy.ndarray]
        Mapping of channel name to 2D image array, suitable for passing as
        ``image_dict`` to :func:`spacec.pp.compensate_cell_matrix`.

    Raises
    ------
    ValueError
        If the image could not be loaded (e.g. missing file, missing
        `nuclei_channel`, or wrong `input_format`).
    """
    _, image_dict, _ = load_image_dictionary(
        file_name, channel_file, input_format, nuclei_channel
    )
    if image_dict is None:
        raise ValueError(
            f"failed to load image dict from {file_name!r} "
            f"(input_format={input_format!r}); see the printed error above."
        )
    return image_dict


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
    """
    Prepare data structures for Canonical Correlation Analysis (CCA) across neighborhoods.

    Encodes neighborhood labels as integers, computes log-transformed mean
    marker (or subset) values per unique region per neighborhood, and
    initialises a CCA model and Pearson correlation function ready for use.

    Parameters
    ----------
    df : pandas.DataFrame
        Cell-level DataFrame containing a ``"unique_region"`` column, the
        neighborhood column, and marker/subset columns.
    neighborhood_column : str
        Column containing neighborhood labels.
    subsets : list of str, optional
        Subset of marker/feature columns to use for the CCA input matrix. If
        ``None``, all non-grouping columns are used. Default is ``None``.

    Returns
    -------
    df : pandas.DataFrame
        Input DataFrame with added ``"neigh_num"`` categorical column.
    cns : list
        Ordered list of unique neighborhood integer codes.
    nsctf : pandas.DataFrame
        Log-transformed mean feature values indexed by ``(neigh_num, unique_region)``.
    cca : sklearn.cross_decomposition.CCA
        Initialised CCA model with 1 component and max 5000 iterations.
    func : callable
        ``scipy.stats.pearsonr`` function for correlation computation.
    neigh_num : dict
        Mapping from neighborhood label strings to integer codes.
    """
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
    """
    Invert a dictionary so that values become keys and keys become values.

    Parameters
    ----------
    dictionary : dict
        Dictionary to invert. Values must be hashable.

    Returns
    -------
    dict
        New dictionary with keys and values swapped. If the original
        dictionary contains duplicate values, only the last key for each
        value is retained.
    """
    inverted_dict = {value: key for key, value in dictionary.items()}
    return inverted_dict


def hf_replace_names(color_dict, name_dict):
    """
    Rename keys in a color dictionary using a name-mapping dictionary.

    Inverts ``color_dict`` to a color-to-name mapping, applies any
    replacements found in ``name_dict``, then re-inverts back to a
    name-to-color mapping.

    Parameters
    ----------
    color_dict : dict
        Dictionary mapping names (str) to color values.
    name_dict : dict
        Dictionary mapping old names to new names. Only names present in
        both ``color_dict`` and ``name_dict`` are replaced.

    Returns
    -------
    dict
        Updated dictionary mapping (possibly renamed) names to color values.

    Notes
    -----
    This function calls ``hf_invert_dictionary`` which is not defined in this
    module. Use :func:`invert_dictionary` instead, or ensure ``hf_invert_dictionary``
    is available in the namespace before calling this function.
    """
    color_dict = hf_invert_dictionary(color_dict)
    for color, name in color_dict.items():
        if name in name_dict:
            color_dict[color] = name_dict[name]
    color_dict = hf_invert_dictionary(color_dict)
    return color_dict


def hf_annotate_cor_plot(x, y, **kws):
    """
    Annotate a scatter plot facet with the Pearson correlation coefficient and p-value.

    Intended for use as a ``map_dataframe`` function in a seaborn PairGrid or
    FacetGrid. Reads the data from ``kws["data"]``, computes the Pearson r and
    p-value for the given x/y column pair, and renders the annotation at a
    fixed position inside the current axes.

    Parameters
    ----------
    x : str
        Column name for the x-axis variable.
    y : str
        Column name for the y-axis variable.
    **kws
        Keyword arguments forwarded by seaborn; must contain ``"data"``
        (a pandas.DataFrame with columns ``x`` and ``y``).

    Returns
    -------
    None
        Annotation is added to the current matplotlib axes in-place.
    """
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


def check_for_gpu(tensorflow=False, torch=True):
    """
    Check if a GPU is available for use by Cellpose/PyTorch.

    TensorFlow support has been removed from SPACEc segmentation. The
    ``tensorflow`` argument is retained for backward compatibility but always
    reports unavailable.

    Returns
    -------
    bool
        True if PyTorch can see a GPU (and `torch=True`, the default). Always False
        when `tensorflow=True`, since TensorFlow support was removed.
    """
    gpu_tf = False
    if tensorflow:
        warnings.warn(
            "check_for_gpu(tensorflow=...) is deprecated because TensorFlow support was removed.",
            DeprecationWarning,
            stacklevel=2,
        )
        print(
            "GPU is not available to Tensorflow (TensorFlow backend has been removed)."
        )

    gpu_torch = use_gpu()
    if gpu_torch:
        print("GPU is available to Pytorch")
    else:
        print("GPU is not available to Pytorch")

    if torch and tensorflow:
        return gpu_torch and gpu_tf
    elif torch:
        return gpu_torch
    elif tensorflow:
        return gpu_tf
    else:
        return False


_CELLPOSE_VERSION_ALIASES = {
    3: 3,
    "3": 3,
    "cyto3": 3,
    4: 4,
    "4": 4,
    "sam": 4,
    "cpsam": 4,
}


def set_cellpose_version(version, force=False, dry_run=False, quiet=False):
    """
    Install or switch the cellpose version backing SPACEc segmentation.

    SPACEc supports two cellpose generations: classic cellpose 3.x (per-tissue
    model zoo, e.g. 'cyto3') and cellpose>=4.x "Cellpose-SAM" (single
    generalist 'cpsam' model). Which one `sp.tl.cell_segmentation` uses is
    determined entirely by which cellpose package is installed in the active
    Python environment — this function switches that.

    Because cellpose and torch are already imported once SPACEc has loaded,
    installing a different cellpose version has **no effect on the running
    process**. You must restart your kernel / Python process after switching
    for the change to take effect.

    Parameters
    ----------
    version : {3, 4, "cyto3", "sam", "cpsam"}
        Target cellpose generation. 3 and "cyto3" both mean classic cellpose
        3.x. 4, "sam", and "cpsam" all mean Cellpose-SAM (cellpose>=4).
    force : bool, optional
        Skip the interactive "yes" confirmation before installing. Required
        in non-interactive sessions (scripts, CI) since there is no one to
        confirm a prompt. Default False.
    dry_run : bool, optional
        Print what would happen (current version, target version, pip
        command, consequences) without installing anything. Default False.
    quiet : bool, optional
        Suppress the informational prints (consequences, progress). Default
        False.

    Returns
    -------
    str or None
        The installed cellpose version string after the call (unchanged if
        no switch was needed or `dry_run=True`), or None if cellpose is not
        installed and `dry_run=True`.

    Raises
    ------
    ValueError
        If `version` is not a recognized alias.
    RuntimeError
        If called in a non-interactive session without `force=True`, if the
        user declines the confirmation prompt, or if the pip install fails.
    """
    version_key = version.lower() if isinstance(version, str) else version
    if version_key not in _CELLPOSE_VERSION_ALIASES:
        raise ValueError(
            f"Unrecognized cellpose version {version!r}. Use 3 (or 'cyto3') for "
            "classic Cellpose, or 4 (or 'sam'/'cpsam') for Cellpose-SAM."
        )
    target_major = _CELLPOSE_VERSION_ALIASES[version_key]

    try:
        installed_version = importlib.metadata.version("cellpose")
        installed_major = int(installed_version.split(".")[0])
    except importlib.metadata.PackageNotFoundError:
        installed_version = None
        installed_major = None

    if installed_major == target_major:
        if not quiet:
            print(f"cellpose {installed_version} already installed — nothing to do.")
        return installed_version

    spec = "cellpose>=3,<4" if target_major == 3 else "cellpose>=4,<5"

    consequences = [
        "Switching does NOT take effect in this running process — cellpose "
        "and torch are already imported. You MUST restart your kernel / "
        "Python process after this completes.",
    ]
    if target_major == 4:
        consequences += [
            "Custom/fine-tuned cellpose 3.x models will no longer load "
            "(Cellpose-SAM uses a different backbone) — retrain on cpsam, "
            "or stay on cellpose 3.",
            "Segmentation results will differ from cyto3 — do not mix "
            "versions of cellpose within one study.",
            "First run downloads the cpsam weights (~GB) and needs more "
            "GPU/VRAM; consider lowering tile_size for cell_segmentation.",
        ]
    else:
        consequences += [
            "model='cpsam' calls will start failing.",
            "You lose Cellpose-SAM's diameter- and channel-invariant "
            "segmentation and go back to per-tissue models.",
        ]

    if not quiet:
        print(f"Installed cellpose: {installed_version or 'not installed'}")
        print(f'Requested: cellpose {target_major}.x  ->  pip install "{spec}"')
        print("Consequences of switching:")
        for line in consequences:
            print(f"  - {line}")

    if dry_run:
        if not quiet:
            print("(dry_run=True: no changes made)")
        return installed_version

    if not force:
        if not sys.stdin.isatty():
            raise RuntimeError(
                "Refusing to switch cellpose versions in a non-interactive "
                "session without confirmation. Re-run with force=True to proceed."
            )
        answer = input("Type 'yes' to proceed with this switch: ")
        if answer.strip().lower() != "yes":
            print("Aborted — cellpose version unchanged.")
            return installed_version

    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", spec],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f'pip install "{spec}" failed (exit {result.returncode}):\n{result.stderr}'
        )

    importlib.invalidate_caches()
    new_version = importlib.metadata.version("cellpose")
    if not quiet:
        print(f"Installed cellpose {new_version}.")
        print(
            "*** RESTART YOUR KERNEL / PYTHON PROCESS NOW for this to take effect. ***"
        )
    return new_version
