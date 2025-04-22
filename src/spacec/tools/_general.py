# load required packages
from __future__ import annotations

import os
import platform
import subprocess
import sys
import tempfile
import zipfile

import requests

if platform.system() == "Windows":
    vipsbin = r"c:\vips-dev-8.15\bin\vips-dev-8.15\bin"
    vips_file_path = os.path.join(vipsbin, "vips.exe")

    # Check if VIPS is installed
    if not os.path.exists(vips_file_path):
        # VIPS is not installed, download and extract it
        url = "https://github.com/libvips/build-win64-mxe/releases/download/v8.15.2/vips-dev-w64-all-8.15.2.zip"
        zip_file_path = "vips-dev-w64-all-8.15.2.zip"
        response = requests.get(url, stream=True)

        if response.status_code == 200:
            with open(zip_file_path, "wb") as f:
                f.write(response.raw.read())

            # Extract the zip file
            with zipfile.ZipFile(zip_file_path, "r") as zip_ref:
                zip_ref.extractall(vipsbin)
        else:
            print("Error downloading the file.")

        # Install pyvips
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pyvips"])

    # Add vipsbin to the DLL search path or PATH environment variable
    add_dll_dir = getattr(os, "add_dll_directory", None)
    os.environ["PATH"] = os.pathsep.join((vipsbin, os.environ["PATH"]))


import argparse
import pathlib
import pickle
import time
from builtins import range
from itertools import combinations
from multiprocessing import Pool
from typing import TYPE_CHECKING

import anndata
import concave_hull
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import panel as pn
import scipy.stats as st
import skimage
import skimage.color
import skimage.exposure
import skimage.filters.rank
import skimage.io as io
import skimage.morphology
import skimage.transform
import statsmodels.api as sm
import tissuumaps.jupyter as tj
import torch
from concave_hull import concave_hull_indexes
from joblib import Parallel, delayed
from pyFlowSOM import map_data_to_nodes, som
from scipy import stats
from scipy.spatial import Delaunay, KDTree, distance
from scipy.spatial.distance import cdist
from scipy.stats import pearsonr, spearmanr
from skimage.io import imsave
from skimage.segmentation import find_boundaries
from sklearn.cluster import HDBSCAN, MiniBatchKMeans
from sklearn.cross_decomposition import CCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score, pairwise_distances
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.svm import SVC
from tqdm import tqdm
from yellowbrick.cluster import KElbowVisualizer

if TYPE_CHECKING:
    from anndata import AnnData

from ..helperfunctions._general import *
from ..plotting._general import catplot

try:
    from torch_geometric.data import ClusterData, ClusterLoader, Data, InMemoryDataset
except ImportError:
    pass

try:
    import cupy as cp
    import rapids_singlecell as rsc
    from cupyx.scipy.sparse import csc_matrix as csc_matrix_gpu
    from cupyx.scipy.sparse import csr_matrix as csr_matrix_gpu
    from cupyx.scipy.sparse import isspmatrix_csc as isspmatrix_csc_gpu
    from cupyx.scipy.sparse import isspmatrix_csr as isspmatrix_csr_gpu
    from scanpy.get import _get_obs_rep, _set_obs_rep
    from scipy.sparse import isspmatrix_csc as isspmatrix_csc_cpu
    from scipy.sparse import isspmatrix_csr as isspmatrix_csr_cpu
except ImportError:
    pass

# Tools
############################################################


def tl_calculate_neigh_combs(w, l, n_num, threshold=0.85, per_keep_thres=0.85):
    """
    Calculate neighborhood combinations based on a threshold.

    Parameters
    ----------
    w : DataFrame
        DataFrame containing the data.
    l : list
        List of column names to be used.
    n_num : int
        Number of neighborhoods or k chosen for the neighborhoods.
    threshold : float, optional
        Threshold for neighborhood combinations, by default 0.85.
    per_keep_thres : float, optional
        Percent to keep threshold or percent of neighborhoods that fall above a certain threshold, by default 0.85.

    Returns
    -------
    tuple
        A tuple containing:
        - simps: Series of neighborhood combinations.
        - simp_freqs: Series of frequency counts of the combinations.
        - simp_sums: Series of cumulative sums of the frequency counts.

    """
    w.loc[:, l]

    # need to normalize by number of neighborhoods or k chosen for the neighborhoods
    xm = w.loc[:, l].values / n_num

    # Get the neighborhood combinations based on the threshold
    simps = hf_get_thresh_simps(xm, threshold)
    simp_freqs = simps.value_counts(normalize=True)
    simp_sums = np.cumsum(simp_freqs)

    # See the percent to keep threshold or percent of neigbhorhoods that fall above a certain threshold
    test_sums_thres = simp_sums[simp_sums < per_keep_thres]
    test_len = len(test_sums_thres)
    per_values_above = simp_sums[test_len] - simp_sums[test_len - 1]
    print(test_len, per_values_above)

    w["combination"] = [tuple(l[a] for a in s) for s in simps]
    w["combination_num"] = [tuple(a for a in s) for s in simps]

    # this shows what proportion (y) of the total cells are assigned to the top x combinations
    plt.figure(figsize=(7, 3))
    plt.plot(simp_sums.values)
    plt.title(
        "proportion (y) of the total cells are assigned to the top x combinations"
    )
    plt.show()

    # this shows what proportion (y) of the total cells are assigned to the top x combinations
    plt.figure(figsize=(7, 3))
    plt.plot(test_sums_thres.values)
    plt.title(
        "proportion (y) of the total cells are assigned to the top x combinations - thresholded"
    )
    plt.show()
    # plt.xticks(range(0,350,35),range(0,350,35),rotation = 90,fontsize = 10)

    return (simps, simp_freqs, simp_sums)


def tl_build_graph_CN_comb_map(simp_freqs, thresh_freq=0.001):
    """
    Build a directed graph for the CN combination map.

    Parameters
    ----------
    simp_freqs : pandas.Series
        A series containing the frequencies of simplices.
    thresh_freq : float, optional
        The threshold frequency to filter simplices, by default 0.001.

    Returns
    -------
    tuple
        A tuple containing:
        - g : networkx.DiGraph
            The directed graph with edges representing the CN combination map.
        - tops : list
            A list of the top 20 simplices sorted by frequency.
        - e0 : str
            The last simplex in the outer loop.
        - e1 : str
            The last simplex in the inner loop.
    """
    g = nx.DiGraph()

    # selected_simps = simp_sums[simp_sums<=thresh_cumulative].index.values
    selected_simps = simp_freqs[simp_freqs >= thresh_freq].index.values
    selected_simps

    """
    this builds the graph for the CN combination map
    """
    for e0 in selected_simps:
        for e1 in selected_simps:
            if (set(list(e0)) < set(list(e1))) and (len(e1) == len(e0) + 1):
                g.add_edge(e0, e1)

    tops = (
        simp_freqs[simp_freqs >= thresh_freq]
        .sort_values(ascending=False)
        .index.values.tolist()[:20]
    )

    return (g, tops, e0, e1)


def clustering(
    adata,
    clustering="leiden",
    marker_list=None,
    resolution=1,
    n_neighbors=10,
    reclustering=False,
    key_added=None,
    key_filter=None,
    subset_cluster=None,
    seed=42,
    fs_xdim=10,
    fs_ydim=10,
    fs_rlen=10,  # FlowSOM parameters
    **cluster_kwargs,
):
    """
    Perform clustering on the given annotated data matrix.

    Parameters
    ----------
    adata : AnnData
        The annotated data matrix of shape n_obs x n_vars. Rows correspond
        to cells and columns to stained markers.
    clustering : str, optional
        The clustering algorithm to use. Options are "leiden" or "louvain". Defaults to "leiden".
    marker_list : list, optional
        A list of markers for clustering. Defaults to None.
    resolution : int, optional
        The resolution for the clustering algorithm. Defaults to 1.
    n_neighbors : int, optional
        The number of neighbors to use for the neighbors graph. Defaults to 10.
    reclustering : bool, optional
        Whether to recluster the data. Defaults to False.
    key_added : str, optional
        The key name to add to the adata object. Defaults to None.
    key_filter : str, optional
        The key name to filter the adata object. Defaults to None.
    subset_cluster : list, optional
        The list of clusters to subset. Defaults to None.
    seed : int, optional
        Seed for random state. Default is 42.
    fs_xdim : int, optional
        X dimension for FlowSOM. Default is 10.
    fs_ydim : int, optional
        Y dimension for FlowSOM. Default is 10.
    fs_rlen : int, optional
        Rlen for FlowSOM. Default is 10.
    **cluster_kwargs : dict
        Additional keyword arguments for the clustering function.

    Returns
    -------
    AnnData
        The annotated data matrix with the clustering results added.
    """
    if clustering not in ["leiden", "louvain", "leiden_gpu", "flowSOM"]:
        print(
            "Invalid clustering options. Please select from leiden, louvain, leiden_gpu or flowSOM!"
        )
        print("For GPU accelerated leiden clustering, please use leiden_gpu")
        sys.exit()

    # test if rapids_singlecell is available
    if clustering == "leiden_gpu":
        try:
            import cudf
            import cuml
            import cupy
            import rapids_singlecell as rsc
        except ImportError:
            print("Please install rapids_singlecell to use leiden_gpu!")
            print("install_gpu_leiden(CUDA = your cuda version as string)")
            print("For example: sp.tl.install_gpu_leiden(CUDA = '12')")
            print("THIS FUNCTION DOES NOT WORK ON MacOS OR WINDOWS")
            print("using leiden instead of leiden_gpu")
            clustering = "leiden"

    if key_added is None:
        key_added = clustering + "_" + str(resolution)

    if key_filter is not None:
        if subset_cluster is None:
            print("Please provide subset_cluster!")
            sys.exit()
        else:
            adata_tmp = adata
            adata = adata[adata.obs[key_filter].isin(subset_cluster)]

    # input a list of markers for clustering
    # reconstruct the anndata
    if marker_list is not None:
        if len(list(set(marker_list) - set(adata.var_names))) > 0:
            print("Marker list not all in adata var_names! Using intersection instead!")
            marker_list = list(set(marker_list) & set(adata.var_names))
            print("New marker_list: " + " ".join(marker_list))
        if key_filter is None:
            adata_tmp = adata
        adata = adata[:, marker_list]

    # Compute the neighborhood relations of single cells the range 2 to 100 and usually 10
    if reclustering:
        if clustering == "leiden_gpu":
            print("Clustering on GPU")
            anndata_to_GPU(adata)  # moves `.X` to the GPU
            rsc.tl.leiden(
                adata,
                resolution=resolution,
                key_added=key_added,
                random_state=seed,
                **cluster_kwargs,
            )
            anndata_to_CPU(adata)  # moves `.X` to the CPU
        else:
            print("Clustering")
            if clustering == "leiden":
                sc.tl.leiden(
                    adata,
                    resolution=resolution,
                    key_added=key_added,
                    random_state=seed,
                    **cluster_kwargs,
                )
            else:
                if clustering == "louvain":
                    print("Louvain clustering")
                    sc.tl.louvain(
                        adata,
                        resolution=resolution,
                        key_added=key_added,
                        random_state=seed,
                        **cluster_kwargs,
                    )
                else:
                    print("FlowSOM clustering")
                    adata_df = pd.DataFrame(
                        adata.X, index=adata.obs.index, columns=adata.var.index
                    )
                    # df to numpy array
                    som_input_arr = adata_df.to_numpy()
                    # train the SOM
                    node_output = som(
                        som_input_arr,
                        xdim=fs_xdim,
                        ydim=fs_ydim,
                        rlen=fs_rlen,
                        seed=seed,
                    )
                    # use trained SOM to assign clusters to each observation in your data
                    clusters, dists = map_data_to_nodes(node_output, som_input_arr)
                    clusters = pd.Categorical(clusters)
                    # add cluster to adata
                    adata.obs[key_added] = clusters
    else:
        if clustering == "leiden_gpu":
            anndata_to_GPU(adata)  # moves `.X` to the GPU
            print("Computing neighbors and UMAP on GPU")
            rsc.pp.neighbors(adata, n_neighbors=n_neighbors)
            # UMAP computation
            rsc.tl.umap(adata)
            print("Clustering on GPU")
            # Perform leiden clustering - improved version of louvain clustering
            rsc.tl.leiden(
                adata, resolution=resolution, key_added=key_added, random_state=seed
            )
            anndata_to_CPU(adata)  # moves `.X` to the CPU

        else:

            print("Computing neighbors and UMAP")
            print("- neighbors")
            sc.pp.neighbors(adata, n_neighbors=n_neighbors)
            # UMAP computation
            print("- UMAP")
            sc.tl.umap(adata)
            print("Clustering")
            # Perform leiden clustering - improved version of louvain clustering
            if clustering == "leiden":
                print("Leiden clustering")
                sc.tl.leiden(
                    adata,
                    resolution=resolution,
                    key_added=key_added,
                    random_state=seed,
                    **cluster_kwargs,
                )
            else:
                if clustering == "louvain":
                    print("Louvain clustering")
                    sc.tl.louvain(
                        adata,
                        resolution=resolution,
                        key_added=key_added,
                        random_state=seed,
                        **cluster_kwargs,
                    )
                else:
                    print("FlowSOM clustering")
                    adata_df = pd.DataFrame(
                        adata.X, index=adata.obs.index, columns=adata.var.index
                    )
                    # df to numpy array
                    som_input_arr = adata_df.to_numpy()
                    # train the SOM
                    node_output = som(
                        som_input_arr,
                        xdim=fs_xdim,
                        ydim=fs_ydim,
                        rlen=fs_rlen,
                        seed=seed,
                    )
                    # use trained SOM to assign clusters to each observation in your data
                    clusters, dists = map_data_to_nodes(node_output, som_input_arr)

                    # make clusters a string
                    clusters = clusters.astype(str)

                    clusters = pd.Categorical(clusters)
                    # add cluster to adata
                    adata.obs[key_added] = clusters

    if key_filter is None:
        if marker_list is None:
            return adata
        else:
            adata_tmp.obs[key_added] = adata.obs[key_added].values
            # append other data
            adata_tmp.obsm = adata.obsm
            adata_tmp.obsp = adata.obsp
            adata_tmp.uns = adata.uns

    if key_filter is not None:
        original_df = adata_tmp.obs
        donor_df = adata.obs

        donor_df_cols = donor_df.loc[:, donor_df.columns != key_added].columns.tolist()
        # Perform the merge operation
        merged_df = pd.merge(
            original_df,
            donor_df,
            left_on=donor_df_cols,
            right_on=donor_df_cols,
            how="left",
        )

        # Fill NA/NaN values in 'key_added' using the values from 'key_filter'
        merged_df[key_filter] = merged_df[key_filter].astype(str)
        merged_df[key_added] = merged_df[key_added].astype(str)

        merged_df.replace("nan", np.nan, inplace=True)

        merged_df[key_added].fillna(merged_df[key_filter], inplace=True)

        merged_df[key_filter] = merged_df[key_filter].astype("category")
        merged_df[key_added] = merged_df[key_added].astype("category")

        merged_df.index = merged_df.index.astype(str)
        # assign df as obs for adata_tmp
        adata_tmp.obs = merged_df

    return adata_tmp


def neighborhood_analysis(
    adata,
    unique_region,
    cluster_col,
    X="x",
    Y="y",
    k=35,
    n_neighborhoods=30,
    elbow=False,
    metric="distortion",
):
    """
    Compute for Cellular neighborhoods (CNs).

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix.
    unique_region : str
        Each region is one independent CODEX image.
    cluster_col : str
        Columns to compute CNs on, typically 'celltype'.
    X : str, optional
        X coordinate column name, by default "x".
    Y : str, optional
        Y coordinate column name, by default "y".
    k : int, optional
        Number of neighbors to compute, by default 35.
    n_neighborhoods : int, optional
        Number of neighborhoods one ends up with, by default 30.
    elbow : bool, optional
        Whether to test for optimal number of clusters and visualize as elbow plot or not, by default False. If set to True, the function will test 1 to n_neighborhoods and plot the distortion score in an elbow plot to assist the user in finding the optimal number of clusters.
    metric : str, optional
        The metric to use when calculating distance between instances in a feature array, by default "distortion". Other options include "silhouette" and "calinski_harabasz".

    Returns
    -------
    AnnData
        Annotated data matrix with updated neighborhood information.

    Notes
    -----
    The function performs the following steps:
    1. Extracts relevant columns from the input AnnData object.
    2. Computes dummy variables for the cluster column.
    3. Groups data by the unique region and computes neighborhoods.
    4. Optionally performs k-means clustering and visualizes the elbow plot if `elbow` is set to True.
    5. Updates the input AnnData object with neighborhood labels and centroids.

    """
    df = pd.DataFrame(adata.obs[[X, Y, cluster_col, unique_region]])

    cells = pd.concat([df, pd.get_dummies(df[cluster_col])], axis=1)
    sum_cols = cells[cluster_col].unique()
    values = cells[sum_cols].values

    neighborhood_name = "CN" + "_k" + str(k) + "_n" + str(n_neighborhoods)
    centroids_name = "Centroid" + "_k" + str(k) + "_n" + str(n_neighborhoods)

    n_neighbors = k

    cells[unique_region] = cells[unique_region].astype("str")
    cells["cellid"] = cells.index.values
    cells.reset_index(inplace=True)

    keep_cols = [X, Y, unique_region, cluster_col]

    # Get each region
    tissue_group = cells[[X, Y, unique_region]].groupby(unique_region)
    exps = list(cells[unique_region].unique())
    tissue_chunks = [
        (time.time(), exps.index(t), t, a)
        for t, indices in tissue_group.groups.items()
        for a in np.array_split(indices, 1)
    ]

    tissues = [
        hf_get_windows(job, n_neighbors, exps=exps, tissue_group=tissue_group, X=X, Y=Y)
        for job in tissue_chunks
    ]

    # Loop over k to compute neighborhoods
    out_dict = {}

    for neighbors, job in zip(tissues, tissue_chunks):
        chunk = np.arange(len(neighbors))  # indices
        tissue_name = job[2]
        indices = job[3]
        window = (
            values[neighbors[chunk, :k].flatten()]
            .reshape(len(chunk), k, len(sum_cols))
            .sum(axis=1)
        )
        out_dict[(tissue_name, k)] = (window.astype(np.float16), indices)

    windows = {}

    window = pd.concat(
        [
            pd.DataFrame(
                out_dict[(exp, k)][0],
                index=out_dict[(exp, k)][1].astype(int),
                columns=sum_cols,
            )
            for exp in exps
        ],
        axis=0,
    )
    window = window.loc[cells.index.values]
    window = pd.concat([cells[keep_cols], window], axis=1)
    windows[k] = window

    # Fill in based on above
    k_centroids = {}

    # producing what to plot
    windows2 = windows[k]
    windows2[cluster_col] = cells[cluster_col]

    if elbow != True:
        km = MiniBatchKMeans(n_clusters=n_neighborhoods, random_state=0)

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

        visualizer = KElbowVisualizer(
            km, k=(n_neighborhoods), timings=False, metric=metric
        )
        visualizer.fit(X)  # Fit the data to the visualizer
        visualizer.show()  # Finalize and render the figure

    return adata


def build_cn_map(
    adata,
    cn_col,
    unique_region,
    palette=None,
    k=75,
    X="x",
    Y="y",
    threshold=0.85,
    per_keep_thres=0.85,
    sub_list=None,
    sub_col=None,
    rand_seed=1,
):
    """
    Generate a cellular neighborhood (CN) map.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix.
    cn_col : str
        Column name for cellular neighborhood.
    unique_region : str
        Unique region identifier.
    palette : dict, optional
        Color palette for the CN map, by default None.
    k : int, optional
        Number of neighbors to compute, by default 75.
    X : str, optional
        X coordinate column name, by default "x".
    Y : str, optional
        Y coordinate column name, by default "y".
    threshold : float, optional
        Threshold for neighborhood computation, by default 0.85.
    per_keep_thres : float, optional
        Threshold for keeping percentage, by default 0.85.
    sub_list : list, optional
        List of sub regions, by default None.
    sub_col : str, optional
        Column name for sub regions, by default None.
    rand_seed : int, optional
        Random seed for color generation, by default 1.

    Returns
    -------
    dict
        Dictionary containing the graph, top nodes, edges and simplicial frequencies.
    """
    ks = [k]
    cells_df = pd.DataFrame(adata.obs)
    cells_df = cells_df[[X, Y, unique_region, cn_col]]
    cells_df.reset_index(inplace=True)
    sum_cols = cells_df[cn_col].unique()
    keep_cols = cells_df.columns

    cn_colors = hf_generate_random_colors(
        len(adata.obs[cn_col].unique()), rand_seed=rand_seed
    )

    if palette is None:
        if cn_col + "_colors" not in adata.uns.keys():
            palette = dict(zip(np.sort(adata.obs[cn_col].unique()), cn_colors))
            adata.uns[cn_col + "_colors"] = cn_colors
        else:
            palette = dict(
                zip(np.sort(adata.obs[cn_col].unique()), adata.uns[cn_col + "_colors"])
            )

    Neigh = Neighborhoods(
        cells_df,
        ks,
        cn_col,
        sum_cols,
        keep_cols,
        X,
        Y,
        reg=unique_region,
        add_dummies=True,
    )
    windows = Neigh.k_windows()
    w = windows[k]
    if sub_list:
        # convert sub_list to list if only str is provided
        if isinstance(sub_list, str):
            sub_list = [sub_list]
        w = w[w[sub_col].isin(sub_list)]
    l = list(palette.keys())
    simps, simp_freqs, simp_sums = tl_calculate_neigh_combs(
        w, l, k, threshold=threshold, per_keep_thres=per_keep_thres  # color palette
    )
    g, tops, e0, e1 = tl_build_graph_CN_comb_map(simp_freqs)
    return {
        "g": g,
        "tops": tops,
        "e0": e0,
        "e1": e1,
        "simp_freqs": simp_freqs,
        "w": w,
        "l": l,
        "k": k,
        "threshold": threshold,
    }


def tl_format_for_squidpy(adata, x_col, y_col):
    """
    Format an AnnData object for use with Squidpy.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix.
    x_col : str
        Column name for x spatial coordinates.
    y_col : str
        Column name for y spatial coordinates.

    Returns
    -------
    AnnData
        Annotated data matrix formatted for Squidpy, with spatial data in the 'obsm' attribute.
    """
    # Validate input types
    if not isinstance(adata, ad.AnnData):
        raise TypeError("adata must be an AnnData object")
    if not isinstance(x_col, str) or not isinstance(y_col, str):
        raise TypeError("x_col and y_col must be strings")

    # Check if the columns exist in the 'obs' metadata
    if x_col not in adata.obs.columns or y_col not in adata.obs.columns:
        raise ValueError(f"Columns {x_col} and/or {y_col} not found in adata.obs")

    # Extract the count data from your original AnnData object
    counts = adata.X

    # Extract the spatial coordinates from the 'obs' metadata
    spatial_coordinates = adata.obs[[x_col, y_col]].values

    # Ensure spatial coordinates are numeric
    if not np.issubdtype(spatial_coordinates.dtype, np.number):
        raise ValueError("Spatial coordinates must be numeric")

    # Create a new AnnData object with the expected format
    new_adata = ad.AnnData(counts, obsm={"spatial": spatial_coordinates})

    return new_adata


def calculate_triangulation_distances(df_input, id, x_pos, y_pos, cell_type, region):
    """
    Calculate distances between cells using Delaunay triangulation.

    Parameters
    ----------
    df_input : pandas.DataFrame
        Input dataframe containing cell information.
    id : str
        Column name for cell id.
    x_pos : str
        Column name for x position of cells.
    y_pos : str
        Column name for y position of cells.
    cell_type : str
        Column name for cell type annotations.
    region : str
        Column name for region.

    Returns
    -------
    pandas.DataFrame
        Annotated result dataframe with calculated distances and additional information.
    """
    # Perform Delaunay triangulation
    points = df_input[[x_pos, y_pos]].values
    tri = Delaunay(points)
    indices = tri.simplices

    # Get interactions going both directions
    edges = set()
    for simplex in indices:
        for i in range(3):
            for j in range(i + 1, 3):
                edges.add(tuple(sorted([simplex[i], simplex[j]])))
    edges = np.array(list(edges))

    # Create dataframe from edges
    rdelaun_result = pd.DataFrame(edges, columns=["ind1", "ind2"])
    rdelaun_result[["x1", "y1"]] = df_input.iloc[rdelaun_result["ind1"]][
        [x_pos, y_pos]
    ].values
    rdelaun_result[["x2", "y2"]] = df_input.iloc[rdelaun_result["ind2"]][
        [x_pos, y_pos]
    ].values

    # Annotate results with cell type and region information
    df_input["XYcellID"] = (
        df_input[x_pos].astype(str) + "_" + df_input[y_pos].astype(str)
    )
    rdelaun_result["cell1ID"] = (
        rdelaun_result["x1"].astype(str) + "_" + rdelaun_result["y1"].astype(str)
    )
    rdelaun_result["cell2ID"] = (
        rdelaun_result["x2"].astype(str) + "_" + rdelaun_result["y2"].astype(str)
    )

    annotated_result = pd.merge(
        rdelaun_result, df_input, left_on="cell1ID", right_on="XYcellID"
    )
    annotated_result = annotated_result.rename(
        columns={cell_type: "celltype1", id: "celltype1_index"}
    )
    annotated_result = annotated_result.drop(columns=[x_pos, y_pos, region, "XYcellID"])

    annotated_result = pd.merge(
        annotated_result,
        df_input,
        left_on="cell2ID",
        right_on="XYcellID",
        suffixes=(".x", ".y"),
    )
    annotated_result = annotated_result.rename(
        columns={cell_type: "celltype2", id: "celltype2_index"}
    )
    annotated_result = annotated_result.drop(columns=[x_pos, y_pos, "XYcellID"])

    # Calculate distance
    annotated_result["distance"] = np.sqrt(
        (annotated_result["x2"] - annotated_result["x1"]) ** 2
        + (annotated_result["y2"] - annotated_result["y1"]) ** 2
    )

    # Ensure symmetry by adding reversed pairs
    reversed_pairs = annotated_result.copy()
    reversed_pairs = reversed_pairs.rename(
        columns={
            "celltype1_index": "celltype2_index",
            "celltype1": "celltype2",
            "celltype1_X": "celltype2_X",
            "celltype1_Y": "celltype2_Y",
            "celltype2_index": "celltype1_index",
            "celltype2": "celltype1",
            "celltype2_X": "celltype1_X",
            "celltype2_Y": "celltype1_Y",
        }
    )
    annotated_result = pd.concat([annotated_result, reversed_pairs])

    # Reorder columns
    annotated_result = annotated_result[
        [
            region,
            "celltype1_index",
            "celltype1",
            "x1",
            "y1",
            "celltype2_index",
            "celltype2",
            "x2",
            "y2",
            "distance",
        ]
    ]
    annotated_result.columns = [
        region,
        "celltype1_index",
        "celltype1",
        "celltype1_X",
        "celltype1_Y",
        "celltype2_index",
        "celltype2",
        "celltype2_X",
        "celltype2_Y",
        "distance",
    ]

    return annotated_result


# Define the process_region function at the top level
def process_region(df, unique_region, id, x_pos, y_pos, cell_type, region):
    """
    Process a specific region of a dataframe, calculating triangulation distances.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe containing cell information.
    unique_region : str
        Unique region identifier.
    id : str
        Column name for cell id.
    x_pos : str
        Column name for x position of cells.
    y_pos : str
        Column name for y position of cells.
    cell_type : str
        Column name for cell type.
    region : str
        Column name for region.

    Returns
    -------
    pandas.DataFrame
        Result dataframe with calculated distances and additional information for the specified region.
    """
    subset = df[df[region] == unique_region].copy()
    subset["uniqueID"] = (
        subset[id].astype(str)
        + "-"
        + subset[x_pos].astype(str)
        + "-"
        + subset[y_pos].astype(str)
    )
    subset["XYcellID"] = subset[x_pos].astype(str) + "_" + subset[y_pos].astype(str)
    result = calculate_triangulation_distances(
        df_input=subset,
        id=id,
        x_pos=x_pos,
        y_pos=y_pos,
        cell_type=cell_type,
        region=region,
    )
    return result


def get_triangulation_distances(
    df_input, id, x_pos, y_pos, cell_type, region, num_cores=None, correct_dtype=True
):
    """
    Calculate triangulation distances for each unique region in the input dataframe.

    Parameters
    ----------
    df_input : pandas.DataFrame
        Input dataframe containing cell information.
    id : str
        Column name for cell id.
    x_pos : str
        Column name for x position of cells.
    y_pos : str
        Column name for y position of cells.
    cell_type : str
        Column name for cell type.
    region : str
        Column name for region.
    num_cores : int, optional
        Number of cores to use for parallel processing. If None, defaults to half of available cores.
    correct_dtype : bool, optional
        If True, corrects the data type of the cell_type and region columns to string.

    Returns
    -------
    pandas.DataFrame
        Result dataframe with calculated distances and additional information for each unique region.
    """
    if correct_dtype == True:
        # change columns to pandas string
        df_input[cell_type] = df_input[cell_type].astype(str)
        df_input[region] = df_input[region].astype(str)

    # Check if x_pos and y_pos are integers, and if not, convert them
    if not issubclass(df_input[x_pos].dtype.type, np.integer):
        print("This function expects integer values for xy coordinates.")
        print(
            x_pos
            + " and "
            + y_pos
            + " will be changed to integer. Please check the generated output!"
        )
        df_input[x_pos] = df_input[x_pos].astype(int).values
        df_input[y_pos] = df_input[y_pos].astype(int).values

    # Get unique regions
    unique_regions = df_input[region].unique()

    # Select only necessary columns
    df_input = df_input.loc[:, [id, x_pos, y_pos, cell_type, region]]

    # Set up parallelization
    if num_cores is None:
        num_cores = os.cpu_count() // 2  # default to using half of available cores

    # Parallel processing using joblib
    results = Parallel(n_jobs=num_cores)(
        delayed(process_region)(df_input, reg, id, x_pos, y_pos, cell_type, region)
        for reg in unique_regions
    )

    triangulation_distances = pd.concat(results)

    return triangulation_distances


def shuffle_annotations(df_input, cell_type, region, permutation):
    """
    Shuffle annotations within each unique region in the input dataframe.

    Parameters
    ----------
    df_input : pandas.DataFrame
        Input dataframe containing cell information.
    cell_type : str
        Column name for cell type annotations.
    region : str
        Column name for region.
    permutation : int
        Seed for the random number generator.

    Returns
    -------
    pandas.DataFrame
        Result dataframe with shuffled annotations for each unique region.
    """
    # Set the seed for reproducibility
    np.random.seed(permutation + 1234)

    # Create a copy to avoid modifying the original dataframe
    df_shuffled = df_input.copy()

    # Shuffle annotations within each region
    for region_name in df_shuffled[region].unique():
        region_mask = df_shuffled[region] == region_name
        shuffled_values = df_shuffled.loc[region_mask, cell_type].sample(frac=1).values
        df_shuffled.loc[region_mask, "random_annotations"] = shuffled_values

    return df_shuffled


def tl_iterate_tri_distances(
    df_input, id, x_pos, y_pos, cell_type, region, num_cores=None, num_iterations=1000
):
    """
    Iterate over triangulation distances for each unique region in the input dataframe.

    Parameters
    ----------
    df_input : pandas.DataFrame
        Input dataframe containing cell information.
    id : str
        Column name for cell id.
    x_pos : str
        Column name for x position of cells.
    y_pos : str
        Column name for y position of cells.
    cell_type : str
        Column name for cell type.
    region : str
        Column name for region.
    num_cores : int, optional
        Number of cores to use for parallel processing. If None, defaults to half of available cores.
    num_iterations : int, optional
        Number of iterations to perform. Defaults to 1000.

    Returns
    -------
    pandas.DataFrame
        Result dataframe with iterative triangulation distances for each unique region.
    """
    unique_regions = df_input[region].unique()
    # Use only the necessary columns
    df_input = df_input[[id, x_pos, y_pos, cell_type, region]]

    if num_cores is None:
        num_cores = os.cpu_count() // 2  # Default to using half of available cores

    # Define a helper function to process each region and iteration
    def process_iteration(region_name, iteration):
        # Filter by region
        subset = df_input[df_input[region] == region_name].copy()
        # Create unique IDs
        subset.loc[:, "uniqueID"] = (
            subset[id].astype(str)
            + "-"
            + subset[x_pos].astype(str)
            + "-"
            + subset[y_pos].astype(str)
        )
        subset.loc[:, "XYcellID"] = (
            subset[x_pos].astype(str) + "_" + subset[y_pos].astype(str)
        )

        # Shuffle annotations
        shuffled = shuffle_annotations(subset, cell_type, region, iteration)

        # Get triangulation distances
        results = get_triangulation_distances(
            df_input=shuffled,
            id=id,
            x_pos=x_pos,
            y_pos=y_pos,
            cell_type="random_annotations",
            region=region,
            num_cores=num_cores,
            correct_dtype=False,
        )

        # Summarize results
        per_cell_summary = (
            results.groupby(["celltype1_index", "celltype1", "celltype2"])
            .distance.mean()
            .reset_index(name="per_cell_mean_dist")
        )

        per_celltype_summary = (
            per_cell_summary.groupby(["celltype1", "celltype2"])
            .per_cell_mean_dist.mean()
            .reset_index(name="mean_dist")
        )
        per_celltype_summary[region] = region_name
        per_celltype_summary["iteration"] = iteration

        return per_celltype_summary

    # TODO: remove nans valid here a good idea (attempt to fix windows unpickle issue)?
    unique_regions = [r for r in unique_regions if r != np.nan]

    # Parallel processing for each region and iteration
    results = Parallel(n_jobs=num_cores)(
        delayed(process_iteration)(region_name, iteration)
        for region_name in unique_regions
        for iteration in range(1, num_iterations + 1)
    )

    # Combine all results
    iterative_triangulation_distances = pd.concat(results, ignore_index=True)
    # iterative_triangulation_distances = iterative_triangulation_distances.dropna()
    return iterative_triangulation_distances


# def tl_iterate_tri_distances_ad(
#    adata,
#    id,
#    x_pos,
#    y_pos,
#    cell_type,
#    region,
#    num_cores=None,
#    num_iterations=1000,
#    key_name=None,
#    correct_dtype=True,
# ):
#    """
#    Iterate over triangulation distances for each unique region in the input AnnData object

#    Parameters
#    ----------
#    adata : anndata.AnnData
#        Annotated data matrix.
#    id : str
#        Column name for cell id.
#    x_pos : str
#        Column name for x position of cells.
#    y_pos : str
#        Column name for y position of cells.
#    cell_type : str
#        Column name for cell type.
#    region : str
#        Column name for region.
#    num_cores : int, optional
#        Number of cores to use for parallel processing. If None, defaults to half of available cores.
#    num_iterations : int, optional
#        Number of iterations to perform. Defaults to 1000.
#    key_name : str, optional
#        Key name to use when saving the result to the AnnData object. If None, defaults to "iTriDist_" + str(num_iterations).
#    correct_dtype : bool, optional
#        If True, corrects the data type of the cell type and region columns to string. Defaults to True.

#    Returns
#    -------
#    pandas.DataFrame
#        Result dataframe with iterative triangulation distances for each unique region.
#    """
#    df_input = pd.DataFrame(adata.obs)
#    df_input[id] = df_input.index

#    if correct_dtype == True:
#        # change columns to pandas string
#        df_input[cell_type] = df_input[cell_type].astype(str)
#        df_input[region] = df_input[region].astype(str)

#    # Check if x_pos and y_pos are integers, and if not, convert them
#    if not issubclass(df_input[x_pos].dtype.type, np.integer):
#        print("This function expects integer values for xy coordinates.")
#        print("Class will be changed to integer. Please check the generated output!")
#        df_input[x_pos] = df_input[x_pos].astype(int).values
#        df_input[y_pos] = df_input[y_pos].astype(int).values

#    unique_regions = df_input[region].unique()
#    # Use only the necessary columns
#    df_input = df_input.loc[:, [id, x_pos, y_pos, cell_type, region]]

#    if num_cores is None:
#        num_cores = os.cpu_count() // 2  # Default to using half of available cores

# Define a helper function to process each region and iteration
#    def process_iteration(region_name, iteration):
#        # Filter by region
#        subset = df_input.loc[df_input[region] == region_name, :].copy()
#        subset.loc[:, "uniqueID"] = (
#            subset[id].astype(str)
#            + "-"
#            + subset[x_pos].astype(str)
#            + "-"
#            + subset[y_pos].astype(str)
#        )
#        subset.loc[:, "XYcellID"] = (
#            subset[x_pos].astype(str) + "_" + subset[y_pos].astype(str)
#        )

# Shuffle annotations
#        shuffled = shuffle_annotations(subset, cell_type, region, iteration)

# Get triangulation distances
#        results = get_triangulation_distances(
#            df_input=shuffled,
#            id=id,
#            x_pos=x_pos,
#            y_pos=y_pos,
#            cell_type="random_annotations",
#            region=region,
#            num_cores=num_cores,
#            correct_dtype=False,
#        )

#       # Summarize results
#        per_cell_summary = (
#            results.groupby(["celltype1_index", "celltype1", "celltype2"])
#            .distance.mean()
#            .reset_index(name="per_cell_mean_dist")
#        )

#        # Ensure symmetry by aggregating distances in both directions
#        per_cell_summary_reversed = per_cell_summary.rename(
#            columns={"celltype1": "celltype2", "celltype2": "celltype1"}
#        )
#        per_cell_summary_combined = pd.concat([per_cell_summary, per_cell_summary_reversed])#

#        per_celltype_summary = (
#            per_cell_summary_combined.groupby(["celltype1", "celltype2"])
#            .per_cell_mean_dist.mean()
#            .reset_index(name="mean_dist")
#        )
#        per_celltype_summary[region] = region_name
#        per_celltype_summary["iteration"] = iteration#

#       return per_celltype_summary

# Parallel processing for each region and iteration
#    results = Parallel(n_jobs=num_cores)(
#        delayed(process_iteration)(region_name, iteration)
#        for region_name in unique_regions
#        for iteration in range(1, num_iterations + 1)
#    )

# Combine all results
#    iterative_triangulation_distances = pd.concat(results, ignore_index=True)

# append result to adata
#    if key_name is None:
#        key_name = "iTriDist_" + str(num_iterations)
#    adata.uns[key_name] = iterative_triangulation_distances
#    print("Save iterative triangulation distance output to anndata.uns " + key_name)

#    return iterative_triangulation_distances


def add_missing_columns(
    triangulation_distances, metadata, shared_column="unique_region"
):
    """
    Add missing columns from metadata to triangulation_distances dataframe.

    Parameters
    ----------
    triangulation_distances : pandas.DataFrame
        DataFrame containing triangulation distances.
    metadata : pandas.DataFrame
        DataFrame containing metadata.
    shared_column : str, optional
        Column name that is shared between the two dataframes. Defaults to "unique_region".

    Returns
    -------
    pandas.DataFrame
        Updated triangulation_distances dataframe with missing columns added.
    """
    # Find the difference in columns
    missing_columns = set(metadata.columns) - set(triangulation_distances.columns)
    # Add missing columns to triangulation_distances with NaN values
    for column in missing_columns:
        triangulation_distances[column] = pd.NA
        # Create a mapping from unique_region to tissue in metadata
        region_to_tissue = pd.Series(
            metadata[column].values, index=metadata[shared_column]
        ).to_dict()

        # Apply this mapping to the triangulation_distances dataframe to create/update the tissue column
        triangulation_distances[column] = triangulation_distances[shared_column].map(
            region_to_tissue
        )

        # Handle regions with no corresponding tissue in the metadata by filling in a default value
        triangulation_distances[column].fillna("Unknown", inplace=True)
    return triangulation_distances


# Calculate p-values and log fold differences
def calculate_pvalue(row):
    """
    Calculate the p-value using the Mann-Whitney U test.

    Parameters
    ----------
    row : pandas.Series
        A row of data containing 'expected' and 'observed' values.

    Returns
    -------
    float
        The calculated p-value. Returns np.nan if there is insufficient data to perform the test.
    """
    # function body here
    try:
        return st.mannwhitneyu(
            row["expected"], row["observed"], alternative="two-sided"
        ).pvalue
    except ValueError:  # This handles cases with insufficient data
        return np.nan


def identify_interactions(
    adata,
    cellid,
    x_pos,
    y_pos,
    cell_type,
    region,
    comparison,
    triDist_keyname=None,
    min_observed=10,
    distance_threshold=128,
    num_cores=None,
    num_iterations=1000,
    correct_dtype=False,
    aggregate_per_cell=True,
):
    """
    Identify interactions between cell types based on their spatial distances.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix.
    id : str
        Identifier for cells.
    x_pos : str
        Column name for x position of cells.
    y_pos : str
        Column name for y position of cells.
    cell_type : str
        Column name for cell type.
    region : str
        Column name for region.
    comparison : str
        Column name for comparison.
    triDist_keyname : str, optional
        Key name for triangulation distances, by default None
    min_observed : int, optional
        Minimum number of observed distances, by default 10
    distance_threshold : int, optional
        Threshold for distance, by default 128
    num_cores : int, optional
        Number of cores to use for computation, by default None
    num_iterations : int, optional
        Number of iterations for computation, by default 1000
    correct_dtype : bool, optional
        Whether to correct data type or not, by default False

    Returns
    -------
    dict
        Dictionary with the following keys and values:
          "distance_pvals": DataFrame with p-values and logfold changes for interactions.
          "triangulation_distances_observed": DataFrame with the observed cell-cell distances computed from the triangulation
          "triangulation_distances_iterated": DataFrame with the cell-cell distances computed from the triangulation following random iteration
    """
    df_input = pd.DataFrame(adata.obs)
    if cellid in df_input.columns:
        df_input.index = df_input[cellid]
    else:
        print(cellid + " is not in the adata.obs, use index as cellid instead!")
        df_input[cellid] = df_input.index

    # change columns to pandas string
    df_input[cell_type] = df_input[cell_type].astype(str)
    df_input[region] = df_input[region].astype(str)

    print("Computing for observed distances between cell types!")
    triangulation_distances = get_triangulation_distances(
        df_input=df_input,
        id=cellid,
        x_pos=x_pos,
        y_pos=y_pos,
        cell_type=cell_type,
        region=region,
        num_cores=num_cores,
        correct_dtype=correct_dtype,
    )
    if triDist_keyname is None:
        triDist_keyname = "triDist"
    adata.uns[triDist_keyname] = triangulation_distances
    print("Save triangulation distances output to anndata.uns " + triDist_keyname)

    print("Permuting data labels to obtain the randomly distributed distances!")
    print("this step can take awhile")
    iterative_triangulation_distances = tl_iterate_tri_distances(
        df_input=df_input,
        id=cellid,
        x_pos=x_pos,
        y_pos=y_pos,
        cell_type=cell_type,
        region=region,
        num_cores=num_cores,
        num_iterations=num_iterations,
    )

    metadata = df_input.loc[:, [region, comparison]].copy()
    # Reformat observed dataset
    triangulation_distances_long = add_missing_columns(
        triangulation_distances, metadata, shared_column=region
    )
    if aggregate_per_cell == True:
        observed_distances = (
            triangulation_distances_long.query("distance <= @distance_threshold")
            .groupby(["celltype1_index", "celltype1", "celltype2", comparison, region])
            .agg(mean_per_cell=("distance", "mean"))
            .reset_index()
            .groupby(["celltype1", "celltype2", comparison])
            .agg(
                observed=("mean_per_cell", list),
                observed_mean=("mean_per_cell", "mean"),
            )
            .reset_index()
        )
    else:
        observed_distances = (
            triangulation_distances_long.query("distance <= @distance_threshold")
            .groupby(
                [
                    "celltype1_index",
                    "celltype2_index",
                    "celltype1",
                    "celltype2",
                    comparison,
                    region,
                ]
            )
            .agg(mean_per_cell=("distance", "mean"))
            .reset_index()
            .groupby(["celltype1", "celltype2", comparison])
            .agg(
                observed=("mean_per_cell", list),
                observed_mean=("mean_per_cell", "mean"),
            )
            .reset_index()
        )

    # Reformat expected dataset
    iterated_triangulation_distances_long = add_missing_columns(
        iterative_triangulation_distances, metadata, shared_column=region
    )

    expected_distances = (
        iterated_triangulation_distances_long.query("mean_dist <= @distance_threshold")
        .groupby(["celltype1", "celltype2", comparison])
        .agg(expected=("mean_dist", list), expected_mean=("mean_dist", "mean"))
        .reset_index()
    )

    # Drop comparisons with low numbers of observations
    observed_distances["keep"] = observed_distances["observed"].apply(
        lambda x: len(x) > min_observed
    )
    observed_distances = observed_distances[observed_distances["keep"]]

    expected_distances["keep"] = expected_distances["expected"].apply(
        lambda x: len(x) > min_observed
    )
    expected_distances = expected_distances[expected_distances["keep"]]

    # concatenate observed and expected distances
    distance_pvals = expected_distances.merge(
        observed_distances, on=["celltype1", "celltype2", comparison], how="left"
    )

    distance_pvals = expected_distances.merge(
        observed_distances, on=["celltype1", "celltype2", comparison], how="left"
    )
    distance_pvals["pvalue"] = distance_pvals.apply(calculate_pvalue, axis=1)
    distance_pvals["logfold_group"] = np.log2(
        distance_pvals["observed_mean"] / distance_pvals["expected_mean"]
    )
    distance_pvals["interaction"] = (
        distance_pvals["celltype1"] + " --> " + distance_pvals["celltype2"]
    )

    # drop na from distance_pvals
    # distance_pvals = distance_pvals.dropna()

    # append result to adata

    # create dictionary for the results
    triangulation_distances_dict = {
        "distance_pvals": distance_pvals,
        "triangulation_distances_observed": triangulation_distances_long,
        "triangulation_distances_iterated": iterated_triangulation_distances_long,
    }

    return triangulation_distances_dict


def adata_cell_percentages(adata, column_percentage="cell_type"):
    """
    Calculate the percentage of each cell type in an AnnData object.

    Parameters:
    adata (AnnData): An AnnData object containing single-cell data.
    column_percentage (str): The column name in adata.obs that contains cell type information. Default is 'cell_type'.

    Returns:
    DataFrame: A pandas DataFrame with two columns: the specified column name and 'percentage', representing the percentage of each cell type.
    """
    # Assuming 'adata' is an AnnData object and 'cell_type' is the column with cell type information
    cell_type_counts = adata.obs[column_percentage].value_counts()
    total_cells = len(adata)
    cell_type_percentages = (cell_type_counts / total_cells) * 100

    # Convert to DataFrame for better readability
    cell_type_percentages_df = pd.DataFrame(
        {
            column_percentage: cell_type_counts.index,
            "percentage": cell_type_percentages.values,
        }
    )

    return cell_type_percentages_df


def filter_interactions(
    distance_pvals, pvalue=0.05, logfold_group_abs=0.1, comparison="condition"
):
    """
    Filters interactions based on p-value, logfold change, and other conditions.

    Parameters
    ----------
    distance_pvals : pandas.DataFrame
        DataFrame containing p-values, logfold changes, and interactions for each comparison.
    pvalue : float, optional
        The maximum p-value to consider for significance. Defaults to 0.05.
    logfold_group_abs : float, optional
        The minimum absolute logfold change to consider for significance. Defaults to 0.1.
    comparison : str, optional
        The comparison condition to filter by. Defaults to "condition".

    Returns
    -------
    dist_table : pandas.DataFrame
        DataFrame containing logfold changes sorted into two columns by the comparison condition.
    distance_pvals_sig_sub : pandas.DataFrame
        Subset of the original DataFrame containing only significant interactions based on the specified conditions.
    """
    # calculate absolute logfold difference
    distance_pvals["logfold_group_abs"] = distance_pvals["logfold_group"].abs()

    # Creating pairs
    distance_pvals["pairs"] = (
        distance_pvals["celltype1"] + "_" + distance_pvals["celltype2"]
    )

    # Filter significant p-values and other specified conditions
    distance_pvals_sig = distance_pvals[
        (distance_pvals["pvalue"] < pvalue)
        & (distance_pvals["celltype1"] != distance_pvals["celltype2"])
        & (~distance_pvals["observed_mean"].isna())
        & (distance_pvals["logfold_group_abs"] > logfold_group_abs)
    ]

    # Assuming distance_pvals_interesting2 is a pandas DataFrame with the same structure as the R dataframe.
    # pair_to = distance_pvals_sig["interaction"].unique()
    pairs = distance_pvals_sig["pairs"].unique()

    # Filtering data
    data = distance_pvals[~distance_pvals["interaction"].isna()]

    # Subsetting data
    distance_pvals_sig_sub = data[data["pairs"].isin(pairs)]
    distance_pvals_sig_sub_reduced = distance_pvals_sig_sub.loc[
        :, [comparison, "logfold_group", "pairs"]
    ].copy()

    # set pairs as index
    distance_pvals_sig_sub_reduced = distance_pvals_sig_sub_reduced.set_index("pairs")

    # sort logfold_group into two columns by tissue
    dist_table = distance_pvals_sig_sub_reduced.pivot(
        columns=comparison, values="logfold_group"
    )
    dist_table.dropna(inplace=True)

    return dist_table, distance_pvals_sig_sub


def remove_rare_cell_types(
    adata, distance_pvals, cell_type_column="cell_type", min_cell_type_percentage=1
):
    """
    Remove cell types with a percentage lower than the specified threshold from the distance_pvals DataFrame.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix.
    distance_pvals : DataFrame
        DataFrame containing distance p-values with columns 'celltype1' and 'celltype2'.
    cell_type_column : str, optional
        Column name in adata containing cell type information, by default "cell_type".
    min_cell_type_percentage : float, optional
        Minimum percentage threshold for cell types to be retained, by default 1.

    Returns
    -------
    DataFrame
        Filtered distance_pvals DataFrame with rare cell types removed.
    """
    cell_type_percentages_df = adata_cell_percentages(
        adata, column_percentage=cell_type_column
    )

    # Identify cell types with less than the specified percentage of the total cells
    rare_cell_types = cell_type_percentages_df[
        cell_type_percentages_df["percentage"] < min_cell_type_percentage
    ][cell_type_column].values

    # Print the names of the cell types with less than the specified percentage of the total cells
    print(
        "Cell types that belong to less than "
        + str(min_cell_type_percentage)
        + "% of total cells:"
    )
    print(rare_cell_types)

    # Remove rows from distance_pvals that contain rare cell types in column celltype1 or celltype2
    distance_pvals = distance_pvals[
        ~distance_pvals["celltype1"].isin(rare_cell_types)
        & ~distance_pvals["celltype2"].isin(rare_cell_types)
    ]

    return distance_pvals


# Function for patch identification
## Adjust clustering parameter to get the desired number of clusters
def apply_dbscan_clustering(df, min_cluster_size=10):
    """
    Apply DBSCAN clustering to a dataframe and update the cluster labels in the original dataframe.
    Parameters
    ----------
    df : pandas.DataFrame
        The dataframe to be clustered.
    min_cluster_size : int, optional
        The number of samples in a neighborhood for a point to be considered as a core point, by default 10
    Returns
    -------
    None
    """
    # Initialize a new column for cluster labels
    df["cluster"] = -1
    # Apply DBSCAN clustering
    hdbscan = HDBSCAN(
        min_samples=None,
        min_cluster_size=min_cluster_size,
        cluster_selection_epsilon=0.0,
        max_cluster_size=None,
        metric="euclidean",
        alpha=1.0,
        cluster_selection_method="eom",
        allow_single_cluster=True,
    )
    labels = hdbscan.fit_predict(df[["x", "y"]])
    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)
    print("Estimated number of clusters: %d" % n_clusters_)
    print("Estimated number of noise points: %d" % n_noise_)
    # Update the cluster labels in the original dataframe
    df.loc[df.index, "cluster"] = labels


def identify_points_in_proximity(
    df,
    full_df,
    identification_column,
    cluster_column="cluster",
    x_column="x",
    y_column="y",
    radius=200,
    edge_neighbours=3,
    plot=True,
    concave_hull_length_threshold=50,
):
    """
    Identify points in proximity within clusters and generate result and outline DataFrames.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing the points to be processed.
    full_df : pandas.DataFrame
        Full DataFrame containing all points.
    identification_column : str
        Column name used for identification.
    cluster_column : str, optional
        Column name for cluster labels, by default "cluster".
    x_column : str, optional
        Column name for x-coordinates, by default "x".
    y_column : str, optional
        Column name for y-coordinates, by default "y".
    radius : int, optional
        Radius for proximity search, by default 200.
    edge_neighbours : int, optional
        Number of edge neighbours, by default 3.
    plot : bool, optional
        Whether to plot the results, by default True.
    concave_hull_length_threshold : int, optional
        Threshold for concave hull length, by default 50.

    Returns
    -------
    result : pandas.DataFrame
        DataFrame containing the result points.
    outlines : pandas.DataFrame
        DataFrame containing the outline points.
    """

    nbrs, unique_clusters = precompute(
        df, x_column, y_column, full_df, identification_column, edge_neighbours
    )
    num_processes = max(
        1, os.cpu_count() - 1
    )  # Use all available CPUs minus 2, but at least 1
    with Pool(processes=num_processes) as pool:
        results = pool.starmap(
            process_cluster,
            [
                (
                    (
                        df,
                        cluster,
                        cluster_column,
                        x_column,
                        y_column,
                        concave_hull_length_threshold,
                        edge_neighbours,
                        full_df,
                        radius,
                        plot,
                        identification_column,
                    ),
                    nbrs,
                    unique_clusters,
                )
                for cluster in set(df[cluster_column]) - {-1}
            ],
        )
    # Unpack the results
    result_list, outline_list = zip(*results)
    # Concatenate the list of DataFrames into a single result DataFrame
    if len(result_list) > 0:
        result = pd.concat(result_list)
    else:
        result = pd.DataFrame(columns=["x", "y", "patch_id", identification_column])
    if len(outline_list) > 0:
        outlines = pd.concat(outline_list)
    else:
        outlines = pd.DataFrame(columns=["x", "y", "patch_id", identification_column])
    return result, outlines


# Precompute nearest neighbors model and unique clusters
def precompute(df, x_column, y_column, full_df, identification_column, edge_neighbours):
    """
    Precompute nearest neighbors and unique clusters.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing the points to be processed.
    x_column : str
        Column name for x-coordinates.
    y_column : str
        Column name for y-coordinates.
    full_df : pandas.DataFrame
        Full DataFrame containing all points.
    identification_column : str
        Column name used for identification.
    edge_neighbours : int
        Number of edge neighbours.

    Returns
    -------
    nbrs : sklearn.neighbors.NearestNeighbors
        Fitted NearestNeighbors model.
    unique_clusters : numpy.ndarray
        Array of unique cluster identifiers.
    """
    nbrs = NearestNeighbors(n_neighbors=edge_neighbours).fit(df[[x_column, y_column]])
    unique_clusters = full_df[identification_column].unique()
    return nbrs, unique_clusters


def process_cluster(args, nbrs, unique_clusters):
    (
        df,
        cluster,
        cluster_column,
        x_column,
        y_column,
        concave_hull_length_threshold,
        edge_neighbours,
        full_df,
        radius,
        plot,
        identification_column,
    ) = args

    """
    Process a single cluster to identify points in proximity and generate hull points.

    Parameters
    ----------
    args : tuple
        Tuple containing the following elements:
        - df : pandas.DataFrame
            DataFrame containing the points to be processed.
        - cluster : int
            Cluster identifier.
        - cluster_column : str
            Column name for cluster labels.
        - x_column : str
            Column name for x-coordinates.
        - y_column : str
            Column name for y-coordinates.
        - concave_hull_length_threshold : int
            Threshold for concave hull length.
        - edge_neighbours : int
            Number of edge neighbours.
        - full_df : pandas.DataFrame
            Full DataFrame containing all points.
        - radius : int
            Radius for proximity search.
        - plot : bool
            Whether to plot the results.
        - identification_column : str
            Column name used for identification.
    nbrs : sklearn.neighbors.NearestNeighbors
        Fitted NearestNeighbors model.
    unique_clusters : numpy.ndarray
        Array of unique cluster identifiers.

    Returns
    -------
    prox_points : pandas.DataFrame
        DataFrame containing points within the proximity of the cluster.
    hull_nearest_neighbors : pandas.DataFrame
        DataFrame containing the nearest neighbors of the hull points.
    """

    # Filter DataFrame for the current cluster
    subset = df.loc[df[cluster_column] == cluster]
    points = subset[[x_column, y_column]].values
    # Compute concave hull indexes
    idxes = concave_hull_indexes(
        points[:, :2],
        length_threshold=concave_hull_length_threshold,
    )
    # Get hull points from the DataFrame
    hull_points = pd.DataFrame(points[idxes], columns=["x", "y"])
    # Find nearest neighbors of hull points in the original DataFrame
    distances, indices = nbrs.kneighbors(hull_points[["x", "y"]])
    hull_nearest_neighbors = df.iloc[indices.flatten()]
    # DataFrame to store points within the circle but from a different cluster
    all_in_circle_diff_cluster = []
    # Extract hull points coordinates
    hull_coords = hull_nearest_neighbors[["x", "y"]].values
    # Calculate distances from all points in full_df to all hull points
    distances = cdist(full_df[["x", "y"]].values, hull_coords)
    # Identify points within the circle for each hull point
    in_circle = distances <= radius
    # Identify points from a different cluster for each hull point
    diff_cluster = (
        full_df[identification_column].values[:, np.newaxis]
        != hull_nearest_neighbors[identification_column].values
    )
    # Combine the conditions
    in_circle_diff_cluster = in_circle & diff_cluster
    # Collect all points within the circle but from a different cluster
    all_in_circle_diff_cluster = full_df[np.any(in_circle_diff_cluster, axis=1)]
    # Plot the points with a different shape if plot is True
    if plot:
        plt.scatter(
            all_in_circle_diff_cluster["x"],
            all_in_circle_diff_cluster["y"],
            facecolors="none",
            edgecolors="#DC0000B2",
            marker="*",
            s=100,
            zorder=5,
            label="Cell within proximity",
        )
    # Remove duplicates from the final DataFrame
    all_in_circle_diff_cluster = all_in_circle_diff_cluster.drop_duplicates()
    # Plot selected points in yellow and draw circles around them if plot is True
    if plot:
        plt.scatter(
            hull_nearest_neighbors["x"],
            hull_nearest_neighbors["y"],
            color="#3C5488B2",
            label="Boarder cells",
            s=100,
            edgecolor="black",
            zorder=6,
        )
        for _, row in hull_nearest_neighbors.iterrows():
            circle = plt.Circle(
                (row["x"], row["y"]),
                radius,
                color="#3C5488B2",
                fill=False,
                linestyle="--",
                alpha=0.5,
            )
            plt.gca().add_patch(circle)
        # Set plot labels and title
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.title(f"Cells within {radius} radius")
        plt.grid(False)
        plt.axis("equal")
        # Place the legend outside the plot
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(
            by_label.values(),
            by_label.keys(),
            loc="center left",
            bbox_to_anchor=(1, 0.5),
        )
        plt.tight_layout()
        # set figure size
        plt.gcf().set_size_inches(15, 5)
        plt.show()
    # Remove duplicates from the final DataFrame
    prox_points = all_in_circle_diff_cluster.drop_duplicates()
    # Add a 'patch_id' column to identify the cluster
    prox_points["patch_id"] = cluster
    return prox_points, hull_nearest_neighbors


# This function analyzes what is in proximity of a selected group (CN, Celltype, etc...).
def patch_proximity_analysis(
    adata,
    region_column,
    patch_column,
    group,
    min_cluster_size=80,
    x_column="x",
    y_column="y",
    radius=128,
    edge_neighbours=3,
    plot=True,
    savefig=False,
    output_dir="./",
    output_fname="",
    key_name="ppa_result",
    plot_color="#6a3d9a",
):
    """
    Performs a proximity analysis on patches of a given group within each region of a dataset.

    Parameters:
    adata (AnnData): The annotated data matrix of shape n_obs x n_vars. Rows correspond to cells and columns to genes.
    region_column (str): The name of the column in the DataFrame that contains the region information.
    patch_column (str): The name of the column in the DataFrame that contains the patch information.
    group (str): The group to perform the proximity analysis on.
    min_cluster_size (int, optional): The minimum number of samples required to form a dense region. Default is 80.
    x_column (str, optional): The name of the column in the DataFrame that contains the x-coordinate. Default is 'x'.
    y_column (str, optional): The name of the column in the DataFrame that contains the y-coordinate. Default is 'y'.
    radius (int, optional): The radius within which to identify points in proximity. Default is 128.
    edge_neighbours (int, optional): The number of edge neighbours to consider. Default is 3.
    plot (bool, optional): Whether to plot the patches. Default is True.
    savefig (bool, optional): Whether to save the figure. Default is False.
    output_dir (str, optional): The directory to save the figure in. Default is "./".
    output_fname (str, optional): The filename to save the figure as. Default is "".
    key_name (str, optional): The key name to store the results in the AnnData object. Default is 'ppa_result'.

    Returns:
    final_results (DataFrame): A DataFrame containing the results of the proximity analysis.
    outlines_results (DataFrame): A DataFrame containing the outlines of the patches.
    """

    df = adata.obs

    for col in df.select_dtypes(["category"]).columns:
        df[col] = df[col].astype(str)

    # list to store results for each region
    region_results = []
    outlines = []

    for region in df[region_column].unique():
        df_region = df[df[region_column] == region].copy()

        df_community = df_region[df_region[patch_column] == group].copy()

        if df_community.shape[0] < min_cluster_size:
            print(f"No {group} in {region}")
            continue

        else:
            apply_dbscan_clustering(df_community, min_cluster_size=min_cluster_size)

            # plot patches
            if plot:
                df_filtered = df_community[df_community["cluster"] != -1]
                fig, ax = plt.subplots(figsize=(10, 10))
                ax.scatter(
                    df_filtered[x_column],
                    df_filtered[y_column],
                    c=plot_color,
                    alpha=0.5,
                )
                ax.set_title(f"HDBSCAN Clusters for {region}_{group}")
                ax.set_xlabel(x_column)
                ax.set_ylabel(y_column)
                ax.grid(False)
                ax.axis("equal")

                if savefig:
                    fig.savefig(
                        output_dir
                        + output_fname
                        + "_"
                        + str(region)
                        + "_patch_proximity.pdf",
                        bbox_inches="tight",
                    )
                else:
                    plt.show()

            results, hull_nearest_neighbors = identify_points_in_proximity(
                df=df_community,
                full_df=df_region,
                cluster_column="cluster",
                identification_column=patch_column,
                x_column=x_column,
                y_column=y_column,
                radius=radius,
                edge_neighbours=edge_neighbours,
                plot=plot,
            )

            # add hull_nearest_neighbors to list
            outlines.append(hull_nearest_neighbors)

            print(f"Finished {region}_{group}")

            # append to region_results
            region_results.append(results)

    # Concatenate all results into a single DataFrame
    final_results = pd.concat(region_results)

    outlines_results = pd.concat(outlines)

    # generate new column named unique_patch_ID that combines the region, group and patch ID
    final_results["unique_patch_ID"] = (
        final_results[region_column]
        + "_"
        + final_results[patch_column].astype(str)
        + "_"
        + "patch_no_"
        + final_results["patch_id"].astype(str)
    )

    adata.uns[key_name] = final_results

    return final_results, outlines_results


def stellar_get_edge_index(
    pos, distance_thres, max_memory_usage=1.6e10, chunk_size=1000
):
    """
    Constructs edge indexes in one region based on pairwise distances and a distance threshold.

    Parameters:
    pos (array-like): An array-like object of shape (n_samples, n_features) representing the positions.
    distance_thres (float): The distance threshold. Pairs of positions with distances less than this threshold will be considered as edges.
    max_memory_usage (float): The maximum memory usage in bytes before switching to chunk processing.
    chunk_size (int): The size of the chunks to process at a time.

    Returns:
    edge_list (list): A list of lists where each inner list contains two indices representing an edge.
    """
    n_samples = pos.shape[0]
    estimated_memory_usage = (
        n_samples * n_samples * 8
    )  # Estimate memory usage for the distance matrix (float64)

    if estimated_memory_usage > max_memory_usage:
        print("Processing will be done in chunks to save memory.")
        edge_list = []
        for i in tqdm(range(0, n_samples, chunk_size), desc="Processing chunks"):
            pos_chunk = pos[i : i + chunk_size]
            dists_chunk = pairwise_distances(pos_chunk, pos)
            dists_mask_chunk = dists_chunk < distance_thres
            np.fill_diagonal(dists_mask_chunk[:, i : i + chunk_size], 0)
            chunk_edge_list = np.transpose(np.nonzero(dists_mask_chunk)).tolist()
            chunk_edge_list = [[i + edge[0], edge[1]] for edge in chunk_edge_list]
            edge_list.extend(chunk_edge_list)
    else:
        dists = pairwise_distances(pos)
        dists_mask = dists < distance_thres
        np.fill_diagonal(dists_mask, 0)
        edge_list = np.transpose(np.nonzero(dists_mask)).tolist()

    return edge_list


def adata_stellar(
    adata_train,
    adata_unannotated,
    celltype_col="coarse_anno3",
    x_col="x",
    y_col="y",
    sample_rate=0.5,
    distance_thres=50,
    epochs=50,
    key_added="stellar_pred",
    STELLAR_path="",
):
    """
    Applies the STELLAR algorithm to the given annotated and unannotated data.

    Parameters:
    adata_train (AnnData): The annotated data.
    adata_unannotated (AnnData): The unannotated data.
    celltype_col (str, optional): The column name for cell types in the annotated data. Defaults to 'coarse_anno3'.
    x_col (str, optional): The column name for x coordinates in the data. Defaults to 'x'.
    y_col (str, optional): The column name for y coordinates in the data. Defaults to 'y'.
    sample_rate (float, optional): The rate at which to sample the training data. Defaults to 0.5.
    distance_thres (int, optional): The distance threshold for constructing edge indexes. Defaults to 50.
    key_added (str, optional): The key to be added to the unannotated data's obs dataframe for the predicted results. Defaults to 'stellar_pred'.

    Returns:
    adata (AnnData): The unannotated data with the added key for the predicted results.
    """

    print(
        "Please consider to cite the following paper when using STELLAR: Brbić, M., Cao, K., Hickey, J.W. et al. Annotation of spatially resolved single-cell data with STELLAR. Nat Methods 19, 1411–1418 (2022). https://doi.org/10.1038/s41592-022-01651-8"
    )

    sys.path.append(str(STELLAR_path))
    from datasets import GraphDataset
    from STELLAR import STELLAR
    from utils import prepare_save_dir

    parser = argparse.ArgumentParser(description="STELLAR")
    parser.add_argument(
        "--seed", type=int, default=1, metavar="S", help="random seed (default: 1)"
    )
    parser.add_argument("--name", type=str, default="STELLAR")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--wd", type=float, default=5e-2)
    parser.add_argument("--input-dim", type=int, default=26)
    parser.add_argument("--num-heads", type=int, default=13)
    parser.add_argument("--num-seed-class", type=int, default=3)
    parser.add_argument("--sample-rate", type=float, default=0.5)
    parser.add_argument(
        "-b", "--batch-size", default=1, type=int, metavar="N", help="mini-batch size"
    )
    parser.add_argument("--distance_thres", default=50, type=int)
    parser.add_argument("--savedir", type=str, default="./")
    args = parser.parse_args(args=[])
    args.cuda = torch.cuda.is_available()
    args.device = torch.device("cuda" if args.cuda else "cpu")
    args.epochs = 50

    # prepare input data
    print("Preparing input data")
    train_df = adata_train.to_df()

    # add to train_df
    positions_celltype = adata_train.obs[[x_col, y_col, celltype_col]]

    train_df = pd.concat([train_df, positions_celltype], axis=1)

    train_df = train_df.sample(n=round(sample_rate * len(train_df)), random_state=1)

    train_X = train_df.iloc[:, 0:-3].values
    test_X = adata_unannotated.to_df().values

    train_y = train_df[celltype_col].str.lower()
    train_y

    labeled_pos = train_df.iloc[
        :, -3:-1
    ].values  # x,y coordinates, indexes depend on specific datasets
    unlabeled_pos = adata_unannotated.obs[[x_col, y_col]].values

    cell_types = np.sort(list(set(train_y))).tolist()
    cell_types

    cell_type_dict = {}
    inverse_dict = {}

    for i, cell_type in enumerate(cell_types):
        cell_type_dict[cell_type] = i
        inverse_dict[i] = cell_type

    train_y = np.array([cell_type_dict[x] for x in train_y])

    labeled_edges = stellar_get_edge_index(labeled_pos, distance_thres=distance_thres)
    unlabeled_edges = stellar_get_edge_index(
        unlabeled_pos, distance_thres=distance_thres
    )

    # build dataset
    print("Building dataset")
    dataset = GraphDataset(train_X, train_y, test_X, labeled_edges, unlabeled_edges)

    # run stellar
    print("Running STELLAR")
    stellar = STELLAR(args, dataset)
    stellar.train()
    _, results = stellar.pred()

    results = results.astype("object")
    for i in range(len(results)):
        if results[i] in inverse_dict.keys():
            results[i] = inverse_dict[results[i]]
    adata_unannotated.obs[key_added] = pd.Categorical(results)

    # make stellar_pred a string
    adata_unannotated.obs["stellar_pred"] = adata_unannotated.obs[
        "stellar_pred"
    ].astype(str)

    return adata_unannotated


def ml_train(
    adata_train,
    label,
    test_size=0.33,
    random_state=0,
    model="svm",
    nan_policy_y="raise",
    showfig=True,
    figsize=(10, 8),
):
    """
    Train a svm model on the provided data.

    Parameters
    ----------
    adata_train : AnnData
        The training data as an AnnData object.
    label : str
        The label to predict.
    test_size : float, optional
        The proportion of the dataset to include in the test split, by default 0.33.
    random_state : int, optional
        The seed used by the random number generator, by default 0.
    model : str, optional
        The type of model to train, by default "svm".
    nan_policy_y : str, optional
        How to handle NaNs in the label, by default "raise". Can be either 'omit' or 'raise'.
    showfig : bool, optional
        Whether to show the confusion matrix as a heatmap, by default True.

    Returns
    -------
    SVC
        The trained Support Vector Classifier model.

    Raises
    ------
    ValueError
        If `nan_policy_y` is not 'omit' or 'raise'.
    """
    X = pd.DataFrame(adata_train.X)
    y = adata_train.obs[label].values

    if nan_policy_y == "omit":
        y_msk = ~y.isna()
        X = X[y_msk]
        y = y[y_msk]
    elif nan_policy_y == "raise":
        pass
    else:
        raise ValueError("nan_policy_y must be either 'omit' or 'raise'")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    print(y.unique().sort_values())

    print("Training now!")
    svc = SVC(kernel="linear", probability=True)
    svc.fit(X_train, y_train)
    pred = []
    y_prob = svc.predict_proba(X_test)
    y_prob = pd.DataFrame(y_prob)
    y_prob.columns = svc.classes_

    svm_label = y_prob.idxmax(axis=1, skipna=True)
    target_names = svc.classes_
    print("Evaluating now!")
    svm_eval = classification_report(
        y_true=y_test, y_pred=svm_label, target_names=target_names, output_dict=True
    )
    if showfig:
        plt.figure(figsize=figsize)
        sns.heatmap(pd.DataFrame(svm_eval).iloc[:-1, :].T, annot=True)
        plt.show()

    return svc


def ml_predict(adata_val, svc, save_name="svm_pred", return_prob_mat=False):
    """
    Predict labels for a given dataset using a trained Support Vector Classifier (SVC) model.

    Parameters
    ----------
    adata_val : AnnData
        The validation data as an AnnData object.
    svc : SVC
        The trained Support Vector Classifier model.
    save_name : str, optional
        The name under which the predictions will be saved in the AnnData object, by default "svm_pred".
    return_prob_mat : bool, optional
        Whether to return the probability matrix, by default False.

    Returns
    -------
    DataFrame or None
        If `return_prob_mat` is True, returns a DataFrame with the probability matrix. Otherwise, returns None.

    """
    print("Classifying!")
    X_val = pd.DataFrame(adata_val.X)
    y_prob_val = svc.predict_proba(X_val)
    y_prob_val = pd.DataFrame(y_prob_val)
    y_prob_val.columns = svc.classes_
    svm_label_val = y_prob_val.idxmax(axis=1, skipna=True)
    svm_label_val.index = X_val.index
    print("Saving cell type labels to adata!")
    adata_val.obs[save_name] = svm_label_val.values
    if return_prob_mat:
        print("Returning probability matrix!")
        y_prob_val.columns = svc.classes_
        svm_label_val = y_prob_val.idxmax(axis=1, skipna=True)
        return svm_label_val


def masks_to_outlines_scikit_image(masks):
    """get outlines of masks as a 0-1 array

    Parameters
    ----------------

    masks: int, 2D or 3D array
        size [Ly x Lx] or [Lz x Ly x Lx], 0=NO masks; 1,2,...=mask labels

    Returns
    ----------------

    outlines: 2D or 3D array
        size [Ly x Lx] or [Lz x Ly x Lx], True pixels are outlines

    """
    if masks.ndim > 3 or masks.ndim < 2:
        raise ValueError(
            "masks_to_outlines takes 2D or 3D array, not %dD array" % masks.ndim
        )

    if masks.ndim == 3:
        outlines = np.zeros(masks.shape, bool)
        for i in range(masks.shape[0]):
            outlines[i] = find_boundaries(masks[i], mode="inner")
        return outlines
    else:
        return find_boundaries(masks, mode="inner")


def download_file_tm(url, save_path):
    """
    Download a file from a given URL and save it to a specified path.

    Parameters
    ----------
    url : str
        The URL of the file to download.
    save_path : str
        The local path where the downloaded file will be saved.

    Raises
    ------
    requests.exceptions.HTTPError
        If the HTTP request returned an unsuccessful status code.
    """
    response = requests.get(url)
    response.raise_for_status()  # Check if the request was successful

    with open(save_path, "wb") as file:
        file.write(response.content)


def check_download_tm_plugins():
    """
    Check and download the TissUUmaps plugins if they are not already present.

    This function checks if the required TissUUmaps plugins are present in the
    appropriate directory within the active Conda environment. If any plugins
    are missing, they are downloaded from the specified URLs.

    Raises
    ------
    EnvironmentError
        If the Conda environment is not activated.
    """
    urls = [
        "https://tissuumaps.github.io/TissUUmaps/plugins/latest/ClassQC.js",
        "https://tissuumaps.github.io/TissUUmaps/plugins/latest/Plot_Histogram.js",
        "https://tissuumaps.github.io/TissUUmaps/plugins/latest/Points2Regions.js",
        "https://tissuumaps.github.io/TissUUmaps/plugins/latest/Spot_Inspector.js",
        "https://tissuumaps.github.io/TissUUmaps/plugins/latest/Feature_Space.js",
    ]

    conda_env_path = os.getenv("CONDA_PREFIX")
    if not conda_env_path:
        raise EnvironmentError("Conda environment is not activated.")

    python_version = f"python{sys.version_info.major}.{sys.version_info.minor}"
    save_directory = os.path.join(
        conda_env_path, "lib", python_version, "site-packages", "tissuumaps", "plugins"
    )

    if not os.path.exists(save_directory):
        save_directory_option = os.path.join(
            conda_env_path, "lib", "site-packages", "tissuumaps", "plugins"
        )
        for url in urls:
            file_name = os.path.basename(url)
            save_path = os.path.join(save_directory_option, file_name)
            if not os.path.exists(save_path):
                download_file_tm(url, save_path)
                print(f"Plug-in downloaded and saved to {save_path}")

    else:
        for url in urls:
            file_name = os.path.basename(url)
            save_path = os.path.join(save_directory, file_name)
            if not os.path.exists(save_path):
                download_file_tm(url, save_path)
                print(f"Plug-in downloaded and saved to {save_path}")


def tm_viewer(
    adata,
    images_pickle_path,
    directory=None,
    region_column="unique_region",
    region="",
    xSelector="x",
    ySelector="y",
    color_by="cell_type",
    keep_list=None,
    include_masks=True,
    open_viewer=True,
    add_UMAP=True,
    use_jpg_compression=False,
):
    """
    Prepare and visualize spatial transcriptomics data using TissUUmaps.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix.
    images_pickle_path : str
        Path to the pickle file containing images and masks.
    directory : str, optional
        Directory to save the output files. If None, a temporary directory will be created.
    region_column : str, optional
        Column name in `adata.obs` that specifies the region, by default "unique_region".
    region : str, optional
        Specific region to process, by default "".
    xSelector : str, optional
        Column name for x coordinates, by default "x".
    ySelector : str, optional
        Column name for y coordinates, by default "y".
    color_by : str, optional
        Column name for coloring the points, by default "celltype_fine".
    keep_list : list, optional
        List of columns to keep from `adata.obs`, by default None.
    include_masks : bool, optional
        Whether to include masks in the output, by default True.
    open_viewer : bool, optional
        Whether to open the TissUUmaps viewer, by default True.
    add_UMAP : bool, optional
        Whether to add UMAP coordinates to the output, by default True.
    use_jpg_compression : bool, optional
        Whether to use JPEG compression for saving images, by default False.

    Returns
    -------
    list
        List of paths to the saved image files.
    list
        List of paths to the saved CSV files.
    """

    print(
        "Please consider to cite the following paper when using TissUUmaps: TissUUmaps 3: Improvements in interactive visualization, exploration, and quality assessment of large-scale spatial omics data - Pielawski, Nicolas et al. 2023 - Heliyon, Volume 9, Issue 5, e15306"
    )

    check_download_tm_plugins()

    segmented_matrix = adata.obs

    with open(images_pickle_path, "rb") as f:
        seg_output = pickle.load(f)

    image_dict = seg_output["image_dict"]
    masks = seg_output["masks"]

    if keep_list is None:
        keep_list = [region_column, xSelector, ySelector, color_by]

    print("Preparing TissUUmaps input...")

    if directory is None:
        directory = tempfile.mkdtemp()

    cache_dir = pathlib.Path(directory) / region
    cache_dir.mkdir(parents=True, exist_ok=True)

    # only keep columns in keep_list
    segmented_matrix = segmented_matrix[keep_list]

    if add_UMAP:
        # add UMAP coordinates to segmented_matrix
        segmented_matrix["UMAP_1"] = adata.obsm["X_umap"][:, 0]
        segmented_matrix["UMAP_2"] = adata.obsm["X_umap"][:, 1]

    csv_paths = []
    # separate matrix by region and save every region as single csv file
    region_matrix = segmented_matrix.loc[segmented_matrix[region_column] == region]

    region_matrix.to_csv(cache_dir / (region + ".csv"))
    csv_paths.append(cache_dir / (region + ".csv"))

    # generate subdirectory for images
    image_dir = cache_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    image_list = []
    # save every image as tif file in image directory from image_dict. name by key in image_dict
    if use_jpg_compression == True:
        print("Using jpg compression")
    for key, image in image_dict.items():
        if use_jpg_compression == True:
            file_path = os.path.join(image_dir, f"{key}.jpg")
            imsave(file_path, image, quality=100)
        else:
            file_path = os.path.join(image_dir, f"{key}.tif")
            imsave(file_path, image, check_contrast=False)
        image_list.append(file_path)

    if include_masks:
        # select first item from image_dict as reference image
        reference_image = list(image_dict.values())[0]

        # make reference image black by setting all values to 0
        reference_image = np.zeros_like(reference_image)

        # make the reference image rgb. Add empty channels
        if len(reference_image.shape) == 2:
            reference_image = np.expand_dims(reference_image, axis=-1)
            reference_image = np.repeat(reference_image, 3, axis=-1)

        # remove last dimension from masks
        masks_3d = np.squeeze(masks)
        outlines = masks_to_outlines_scikit_image(masks_3d)

        reference_image[outlines] = [255, 0, 0]

        file_path = os.path.join(image_dir, "masks.jpg")

        # save black pixel as transparent
        reference_image = reference_image.astype(np.uint8)

        imsave(file_path, reference_image)
        image_list.append(file_path)

    if open_viewer:
        print("Opening TissUUmaps viewer...")
        tj.loaddata(
            images=image_list,
            csvFiles=[str(p) for p in csv_paths],
            xSelector=xSelector,
            ySelector=ySelector,
            keySelector=color_by,
            nameSelector=color_by,
            colorSelector=color_by,
            piechartSelector=None,
            shapeSelector=None,
            scaleSelector=None,
            fixedShape=None,
            scaleFactor=1,
            colormap=None,
            compositeMode="source-over",
            boundingBox=None,
            port=5100,
            host="localhost",
            height=900,
            tmapFilename=region + "_project",
            plugins=[
                "Plot_Histogram",
                "Points2Regions",
                "Spot_Inspector",
                "Feature_Space",
                "ClassQC",
            ],
        )

    return image_list, csv_paths


def tm_viewer_catplot(
    adata,
    directory=None,
    region_column="unique_region",
    x="x",
    y="y",
    color_by="cell_type",
    open_viewer=True,
    add_UMAP=False,
    keep_list=None,
):
    """
    Generate and visualize categorical plots using TissUUmaps.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix.
    directory : str, optional
        Directory to save the output CSV files. If None, a temporary directory is created.
    region_column : str, optional
        Column name in `adata.obs` that contains region information. Default is "unique_region".
    x : str, optional
        Column name in `adata.obs` to be used for x-axis. Default is "x".
    y : str, optional
        Column name in `adata.obs` to be used for y-axis. Default is "y".
    color_by : str, optional
        Column name in `adata.obs` to be used for coloring the points. Default is "cell_type".
    open_viewer : bool, optional
        Whether to open the TissUUmaps viewer after generating the CSV files. Default is True.
    add_UMAP : bool, optional
        Whether to add UMAP coordinates to the output data. Default is False.
    keep_list : list of str, optional
        List of columns to keep from `adata.obs`. If None, defaults to [region_column, x, y, color_by].

    Returns
    -------
    list of str
        List of paths to the generated CSV files.
    """

    check_download_tm_plugins()
    segmented_matrix = adata.obs

    if keep_list is None:
        keep_list = [region_column, x, y, color_by]

    print("Preparing TissUUmaps input...")

    if directory is None:
        print(
            "Creating temporary directory... If you want to save the files, please specify a directory."
        )
        directory = tempfile.mkdtemp()

    if not os.path.exists(directory):
        os.makedirs(directory)

    # only keep columns in keep_list
    segmented_matrix = segmented_matrix[keep_list]

    if add_UMAP:
        # add UMAP coordinates to segmented_matrix
        segmented_matrix["UMAP_1"] = adata.obsm["X_umap"][:, 0]
        segmented_matrix["UMAP_2"] = adata.obsm["X_umap"][:, 1]

    csv_paths = []
    # separate matrix by region and save every region as single csv file
    unique_regions = segmented_matrix[region_column].unique()
    for region in unique_regions:
        region_matrix = segmented_matrix.loc[segmented_matrix[region_column] == region]
        region_csv_path = os.path.join(directory, region + ".csv")
        region_matrix.to_csv(region_csv_path)
        csv_paths.append(region_csv_path)

    if open_viewer:
        print("Opening TissUUmaps viewer...")
        tj.loaddata(
            images=[],
            csvFiles=[str(p) for p in csv_paths],
            xSelector=x,
            ySelector=y,
            keySelector=color_by,
            nameSelector=color_by,
            colorSelector=color_by,
            piechartSelector=None,
            shapeSelector=None,
            scaleSelector=None,
            fixedShape=None,
            scaleFactor=1,
            colormap=None,
            compositeMode="source-over",
            boundingBox=None,
            port=5100,
            host="localhost",
            height=900,
            tmapFilename="project",
            plugins=[
                "Plot_Histogram",
                "Points2Regions",
                "Spot_Inspector",
                "Feature_Space",
                "ClassQC",
            ],
        )

    return csv_paths


def install_gpu_leiden(CUDA="12"):
    """
    Install the necessary packages for GPU-accelerated Leiden clustering.

    Parameters
    ----------
    CUDA : str, optional
        The version of CUDA to use for the installation. Options are '11' and '12'. Default is '12'.

    Returns
    -------
    None

    Notes
    -----
    This function runs a series of pip install commands to install the necessary packages. The specific packages and versions installed depend on the CUDA
    version. The function prints the output and any errors from each command.
    """
    if platform.system() != "Linux":
        print("This feature is currently only supported on Linux.")

    else:
        print("installing rapids_singlecell")
        # Define the commands to run
        if CUDA == "11":
            commands = [
                "pip install rapids-singlecell==0.9.5",
                "pip install --extra-index-url=https://pypi.nvidia.com cudf-cu11==24.2.* dask-cudf-cu11==24.2.* cuml-cu11==24.2.* cugraph-cu11==24.2.* cuspatial-cu11==24.2.* cuproj-cu11==24.2.* cuxfilter-cu11==24.2.* cucim-cu11==24.2.* pylibraft-cu11==24.2.* raft-dask-cu11==24.2.*",
                "pip install protobuf==3.20",
            ]
        else:
            commands = [
                "pip install rapids-singlecell==0.9.5",
                "pip install --extra-index-url=https://pypi.nvidia.com cudf-cu12==24.2.* dask-cudf-cu12==24.2.* cuml-cu12==24.2.* cugraph-cu12==24.2.* cuspatial-cu12==24.2.* cuproj-cu12==24.2.* cuxfilter-cu12==24.2.* cucim-cu12==24.2.* pylibraft-cu12==24.2.* raft-dask-cu12==24.2.*",
                "pip install protobuf==3.20",
            ]

        # Run each command
        for command in commands:
            process = subprocess.Popen(
                command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )
            stdout, stderr = process.communicate()

            # Print the output and error, if any
            if stdout:
                print(f"Output:\n{stdout.decode()}")
            if stderr:
                print(f"Error:\n{stderr.decode()}")


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


def install_stellar(CUDA=12):
    if CUDA == 12:
        subprocess.run(["pip", "install", "torch"], check=True)
        subprocess.run(["pip", "install", "torch_geometric"], check=True)
        subprocess.run(
            [
                "pip",
                "install",
                "pyg_lib",
                "torch_scatter",
                "torch_sparse",
                "torch_cluster",
                "torch_spline_conv",
                "-f",
                "https://data.pyg.org/whl/torch-2.3.0+cu121.html",
            ],
            check=True,
        )
    elif CUDA == 11.8:
        subprocess.run(
            [
                "pip3",
                "install",
                "torch",
                "--index-url",
                "https://download.pytorch.org/whl/cu118",
            ],
            check=True,
        )
        subprocess.run(["pip", "install", "torch_geometric"], check=True)
        subprocess.run(
            [
                "pip",
                "install",
                "pyg_lib",
                "torch_scatter",
                "torch_sparse",
                "torch_cluster",
                "torch_spline_conv",
                "-f",
                "https://data.pyg.org/whl/torch-2.3.0+cu118.html",
            ],
            check=True,
        )
    else:
        print("Please choose between CUDA 12 or 11.8")
        print(
            "If neither is working for you check the installation guide at: https://pytorch.org/get-started/locally/ and https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html"
        )


def launch_interactive_clustering(adata=None, output_dir=None):
    """
    Launch an interactive clustering application for single-cell data analysis.

    Parameters
    ----------
    adata : AnnData, optional
        An AnnData object containing single-cell data. If provided, the data will be loaded automatically.
    output_dir : str, optional
        The directory where the annotated AnnData object will be saved. Required if `adata` is provided.

    Returns
    -------
    main_layout : panel.layout.Row
        The main layout of the interactive clustering application.

    Raises
    ------
    ValueError
        If `adata` is provided but `output_dir` is not specified, or if `output_dir` is not a string.
    """
    warnings.filterwarnings("ignore")
    pn.extension("deckgl", design="bootstrap", theme="default", template="bootstrap")
    pn.state.template.config.raw_css.append(
        """
    #main {
    padding: 0;
    }"""
    )

    # check if output_dir is provided if adata is provided
    if adata is not None and not output_dir:
        raise ValueError(
            "Please provide an output directory to save the annotated AnnData object."
        )
        # exit the function if output_dir is not provided
        return

    else:
        # check if output_dir is a string
        if output_dir and not isinstance(output_dir, str):
            raise ValueError("output_dir must be a string.")

        # check if output directory exists and create if not:
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)

    # Define the app
    def create_clustering_app():

        # Callback to load data
        def load_data(event=None):
            if adata is not None:
                adata_container["adata"] = adata
                marker_list_input.options = list(adata.var_names)
                output_area.object = "**AnnData object loaded successfully.**"
                return
            if not input_path.value or not os.path.isfile(input_path.value):
                output_area.object = "**Please enter a valid AnnData file path.**"
                return
            loaded_adata = sc.read_h5ad(input_path.value)
            adata_container["adata"] = loaded_adata
            marker_list_input.options = list(loaded_adata.var_names)
            output_area.object = "**AnnData file loaded successfully.**"

        # Callback to run clustering
        def run_clustering(event):
            adata = adata_container.get("adata", None)
            if adata is None:
                output_area.object = "**Please load an AnnData file first.**"
                return
            marker_list = (
                list(marker_list_input.value) if marker_list_input.value else None
            )
            key_added = (
                key_added_input.value
                if key_added_input.value
                else clustering_method.value + "_" + str(resolution.value)
            )
            # Start loading indicator
            loading_indicator.active = True
            output_area.object = "**Clustering in progress...**"
            # Run clustering
            try:
                adata = clustering(
                    adata,
                    clustering=clustering_method.value,
                    marker_list=marker_list,
                    resolution=resolution.value,
                    n_neighbors=n_neighbors.value,
                    reclustering=reclustering.value,
                    key_added=key_added,
                    key_filter=None,
                    subset_cluster=None,
                    seed=42,
                    fs_xdim=fs_xdim.value,
                    fs_ydim=fs_ydim.value,
                    fs_rlen=fs_rlen.value,
                )

                adata_container["adata"] = adata
                output_area.object = "**Clustering completed.**"
                # Automatically generate visualization
                key_to_visualize = key_added
                tabs = []
                sc.pl.umap(adata, color=[key_to_visualize], show=False)
                umap_fig = plt.gcf()
                plt.close()
                tabs.append(("UMAP", pn.pane.Matplotlib(umap_fig, dpi=100)))
                if marker_list:
                    sc.pl.dotplot(
                        adata,
                        marker_list,
                        groupby=key_to_visualize,
                        dendrogram=True,
                        show=False,
                    )
                    dotplot_fig = plt.gcf()
                    plt.close()
                    tabs.append(("Dotplot", pn.pane.Matplotlib(dotplot_fig, dpi=100)))
                # Generate histogram plot
                cluster_counts = adata.obs[key_to_visualize].value_counts()
                cluster_counts.sort_index(inplace=True)
                cluster_counts.plot(kind="bar")
                plt.xlabel("Cluster")
                plt.ylabel("Number of Cells")
                plt.title(f"Cluster Counts for {key_to_visualize}")
                hist_fig = plt.gcf()
                plt.close()
                tabs.append(("Histogram", pn.pane.Matplotlib(hist_fig, dpi=100)))
                # Add new tabs to visualization area
                for name, pane in tabs:
                    visualization_area.append((name, pane))
                # Update cluster annotations
                clusters = adata.obs[key_to_visualize].unique().astype(str)
                annotations_df = pd.DataFrame(
                    {"Cluster": clusters, "Annotation": [""] * len(clusters)}
                )
                cluster_annotation.value = annotations_df
            except Exception as e:
                output_area.object = f"**Error during clustering: {e}**"
            finally:
                # Stop loading indicator
                loading_indicator.active = False

        # Callback to run subclustering
        def run_subclustering(event):
            adata = adata_container.get("adata", None)
            if adata is None:
                output_area.object = "**Please run clustering first.**"
                return
            if not subcluster_key.value or not subcluster_values.value:
                output_area.object = "**Please provide subcluster key and values.**"
                return
            clusters = [c.strip() for c in subcluster_values.value.split(",")]
            key_added = subcluster_key.value + "_subcluster"
            # Start loading indicator for subclustering
            loading_indicator_subcluster.active = True
            output_area.object = "**Subclustering in progress...**"
            try:
                sc.tl.leiden(
                    adata,
                    seed=seed.value,
                    restrict_to=(subcluster_key.value, clusters),
                    resolution=subcluster_resolution.value,
                    key_added=key_added,
                )
                adata_container["adata"] = adata
                output_area.object = "**Subclustering completed.**"
                # Update visualization
                tabs = []
                sc.pl.umap(adata, color=[key_added], show=False)
                umap_fig = plt.gcf()
                plt.close()
                tabs.append(("UMAP_Sub", pn.pane.Matplotlib(umap_fig, dpi=100)))
                marker_list = (
                    list(marker_list_input.value) if marker_list_input.value else None
                )
                if marker_list:
                    sc.pl.dotplot(
                        adata,
                        marker_list,
                        groupby=key_added,
                        dendrogram=True,
                        show=False,
                    )
                    dotplot_fig = plt.gcf()
                    plt.close()
                    tabs.append(
                        ("Dotplot_Sub", pn.pane.Matplotlib(dotplot_fig, dpi=100))
                    )
                # Generate histogram plot
                cluster_counts = adata.obs[key_added].value_counts()
                cluster_counts.sort_index(inplace=True)
                cluster_counts.plot(kind="bar")
                plt.xlabel("Subcluster")
                plt.ylabel("Number of Cells")
                plt.title(f"Subcluster Counts for {key_added}")
                hist_fig = plt.gcf()
                plt.close()
                tabs.append(("Histogram_Sub", pn.pane.Matplotlib(hist_fig, dpi=100)))
                # Add new tabs to visualization area
                for name, pane in tabs:
                    visualization_area.append((name, pane))
                # Update cluster annotations
                clusters = adata.obs[key_added].unique().astype(str)
                annotations_df = pd.DataFrame(
                    {"Cluster": clusters, "Annotation": [""] * len(clusters)}
                )
                cluster_annotation.value = annotations_df
            except Exception as e:
                output_area.object = f"**Error during subclustering: {e}**"
            finally:
                # Stop loading indicator for subclustering
                loading_indicator_subcluster.active = False

        # Callback to save annotations
        def save_annotations(event):
            adata = adata_container.get("adata", None)
            if adata is None:
                output_area.object = "**No AnnData object to annotate.**"
                return
            annotation_dict = dict(
                zip(
                    cluster_annotation.value["Cluster"],
                    cluster_annotation.value["Annotation"],
                )
            )
            key_to_annotate = (
                key_added_input.value
                if key_added_input.value
                else clustering_method.value + "_" + str(resolution.value)
            )
            adata.obs["cell_type"] = (
                adata.obs[key_to_annotate]
                .astype(str)
                .map(annotation_dict)
                .astype("category")
            )
            output_area.object = "**Annotations saved to AnnData object.**"

        def save_adata(event):
            adata = adata_container.get("adata", None)
            if adata is None:
                output_area.object = "**No AnnData object to save.**"
                return
            if not output_dir_widget.value:
                output_area.object = "**Please specify an output directory.**"
                return
            os.makedirs(output_dir_widget.value, exist_ok=True)
            output_filepath = os.path.join(
                output_dir_widget.value, "adata_annotated.h5ad"
            )
            adata.write(output_filepath)
            output_area.object = f"**AnnData saved to {output_filepath}.**"

        # Callback to run spatial visualization
        def run_spatial_visualization(event):
            adata = adata_container.get("adata", None)
            if adata is None:
                output_area.object = "**Please load an AnnData file first.**"
                return
            try:
                catplot(
                    adata,
                    color=spatial_color.value,
                    unique_region=spatial_unique_region.value,
                    X=spatial_x.value,
                    Y=spatial_y.value,
                    n_columns=spatial_n_columns.value,
                    palette=spatial_palette.value,
                    savefig=spatial_savefig.value,
                    output_fname=spatial_output_fname.value,
                    output_dir=output_dir_widget.value,
                    figsize=spatial_figsize.value,
                    size=spatial_size.value,
                )
                spatial_fig = plt.gcf()
                plt.close()
                # Add new tab to visualization area
                visualization_area.append(
                    ("Spatial Visualization", pn.pane.Matplotlib(spatial_fig, dpi=100))
                )
                output_area.object = "**Spatial visualization completed.**"
            except Exception as e:
                output_area.object = f"**Error during spatial visualization: {e}**"

        # File paths
        input_path = pn.widgets.TextInput(
            name="AnnData File Path", placeholder="Enter path to .h5ad file"
        )
        output_dir_widget = pn.widgets.TextInput(
            name="Output Directory",
            placeholder="Enter output directory path",
            value=output_dir if output_dir else "",
        )
        load_data_button = pn.widgets.Button(name="Load Data", button_type="primary")

        # Clustering parameters
        clustering_method = pn.widgets.Select(
            name="Clustering Method",
            options=["leiden", "louvain", "flowSOM", "leiden_gpu"],
        )
        resolution = pn.widgets.FloatInput(name="Resolution", value=1.0)
        n_neighbors = pn.widgets.IntInput(name="Number of Neighbors", value=10)
        reclustering = pn.widgets.Checkbox(name="Reclustering", value=False)
        seed = pn.widgets.IntInput(name="Random Seed", value=42)
        key_added_input = pn.widgets.TextInput(
            name="Key Added", placeholder="Enter key to add to AnnData.obs", value=""
        )
        marker_list_input = pn.widgets.MultiChoice(
            name="Marker List", options=[], width=950
        )

        # Subclustering parameters
        subcluster_key = pn.widgets.TextInput(
            name="Subcluster Key",
            placeholder='Enter key to filter on (e.g., "leiden_1")',
        )
        subcluster_values = pn.widgets.TextInput(
            name="Subcluster Values",
            placeholder="Enter clusters to subset (comma-separated)",
        )
        subcluster_resolution = pn.widgets.FloatInput(
            name="Subcluster Resolution", value=0.3
        )
        subcluster_button = pn.widgets.Button(
            name="Run Subclustering", button_type="primary"
        )

        # Cluster annotation
        cluster_annotation = pn.widgets.DataFrame(
            pd.DataFrame(columns=["Cluster", "Annotation"]),
            name="Cluster Annotations",
            autosize_mode="fit_columns",
        )
        save_annotations_button = pn.widgets.Button(
            name="Save Annotations", button_type="success"
        )

        fs_xdim = pn.widgets.IntInput(name="FlowSOM xdim", value=10)
        fs_ydim = pn.widgets.IntInput(name="FlowSOM ydim", value=10)
        fs_rlen = pn.widgets.IntInput(name="FlowSOM rlen", value=10)

        # Buttons
        run_clustering_button = pn.widgets.Button(
            name="Run Clustering", button_type="primary"
        )
        save_adata_button = pn.widgets.Button(
            name="Save AnnData", button_type="success"
        )

        # Loading indicators
        loading_indicator = pn.widgets.Progress(
            name="Clustering Progress", active=False, bar_color="primary"
        )
        loading_indicator_subcluster = pn.widgets.Progress(
            name="Subclustering Progress", active=False, bar_color="primary"
        )

        # Output areas
        output_area = pn.pane.Markdown()
        visualization_area = pn.Tabs()  # Changed to pn.Tabs to hold multiple plots

        # Global variable to hold the AnnData object
        adata_container = {}

        # Spatial visualization parameters
        spatial_color = pn.widgets.TextInput(
            name="Color By Column",
            placeholder="Enter group column name (e.g., cell_type_coarse)",
        )
        spatial_unique_region = pn.widgets.TextInput(
            name="Unique Region Column", value="unique_region"
        )
        spatial_x = pn.widgets.TextInput(name="X Coordinate Column", value="x")
        spatial_y = pn.widgets.TextInput(name="Y Coordinate Column", value="y")
        spatial_n_columns = pn.widgets.IntInput(name="Number of Columns", value=2)
        spatial_palette = pn.widgets.TextInput(name="Color Palette", value="tab20")
        spatial_figsize = pn.widgets.FloatInput(name="Figure Size", value=17)
        spatial_size = pn.widgets.FloatInput(name="Point Size", value=20)
        spatial_savefig = pn.widgets.Checkbox(name="Save Figure", value=False)
        spatial_output_fname = pn.widgets.TextInput(
            name="Output Filename", placeholder="Enter output filename"
        )
        run_spatial_visualization_button = pn.widgets.Button(
            name="Run Spatial Visualization", button_type="primary"
        )

        # Link callbacks
        load_data_button.on_click(load_data)
        run_clustering_button.on_click(run_clustering)
        subcluster_button.on_click(run_subclustering)
        save_annotations_button.on_click(save_annotations)
        save_adata_button.on_click(save_adata)
        run_spatial_visualization_button.on_click(run_spatial_visualization)

        # Clustering Tab Layout
        clustering_tab = pn.Column(
            pn.pane.Markdown("### Load Data"),
            (
                pn.Row(input_path, output_dir_widget, load_data_button)
                if adata is None
                else pn.pane.Markdown("AnnData object loaded.")
            ),
            pn.layout.Divider(),
            pn.pane.Markdown("### Clustering Parameters"),
            pn.Row(clustering_method, resolution, n_neighbors),
            pn.Row(seed, reclustering),
            pn.Row(fs_xdim, fs_ydim, fs_rlen),
            key_added_input,
            marker_list_input,
            pn.layout.Divider(),
            pn.Row(run_clustering_button, loading_indicator),
            output_area,
        )

        # Subclustering Tab Layout
        subclustering_tab = pn.Column(
            pn.pane.Markdown("### Subclustering Parameters"),
            pn.Row(subcluster_key, subcluster_values, subcluster_resolution),
            pn.layout.Divider(),
            pn.Row(subcluster_button, loading_indicator_subcluster),
            output_area,
        )

        # Annotation Tab Layout
        annotation_tab = pn.Column(
            pn.pane.Markdown("### Cluster Annotation"),
            cluster_annotation,
            pn.layout.Divider(),
            save_annotations_button,
            output_area,
        )

        # Save Tab Layout
        save_tab = pn.Column(
            pn.pane.Markdown("### Save Data"), save_adata_button, output_area
        )

        # Spatial Visualization Tab Layout
        spatial_visualization_tab = pn.Column(
            pn.pane.Markdown("### Spatial Visualization Parameters"),
            pn.Row(spatial_color, spatial_palette),
            pn.Row(spatial_unique_region, spatial_n_columns),
            pn.Row(spatial_x, spatial_y),
            pn.Row(spatial_figsize, spatial_size),
            pn.layout.Divider(),
            pn.Row(spatial_savefig, spatial_output_fname),
            pn.layout.Divider(),
            pn.Row(run_spatial_visualization_button),
            output_area,
        )

        # Assemble Tabs
        tabs = pn.Tabs(
            ("Clustering", clustering_tab),
            ("Subclustering", subclustering_tab),
            ("Annotation", annotation_tab),
            ("Spatial Visualization", spatial_visualization_tab),
            ("Save", save_tab),
        )

        # Main Layout with Visualization Area
        main_layout = pn.Row(tabs, visualization_area, sizing_mode="stretch_both")

        # Automatically load data if adata is provided
        if adata is not None:
            load_data()

        return main_layout

    # Run the app
    main_layout = create_clustering_app()

    main_layout.servable(title="SPACEc Clustering App")

    return main_layout
