# load required packages
from __future__ import annotations

# Filter out specific FutureWarnings from anndata and tm
import warnings

warnings.filterwarnings("ignore", category=FutureWarning, module="anndata.utils")
import logging

logging.disable(
    logging.CRITICAL
)  # disables all logging messages at and below CRITICAL level

import os
import platform
import subprocess
import sys
import tempfile
import zipfile

import requests

# TODO: Remove this!
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
import json
import pathlib
import pickle
import re
import time
from builtins import range
from multiprocessing import Pool
from typing import TYPE_CHECKING

import geopandas as gpd
import matplotlib.patheffects as PathEffects
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import panel as pn
import scipy.stats as st
import skimage.filters.rank
import tissuumaps.jupyter as tj
import torch
from concave_hull import concave_hull_indexes
from joblib import Parallel, delayed
from matplotlib.patches import Patch
from pyFlowSOM import map_data_to_nodes, som
from scipy import stats
from scipy.spatial import Delaunay
from scipy.spatial.distance import cdist
from shapely.geometry import MultiPolygon
from skimage.io import imsave
from skimage.segmentation import find_boundaries
from sklearn.calibration import CalibratedClassifierCV
from sklearn.cluster import HDBSCAN, MiniBatchKMeans
from sklearn.cross_decomposition import CCA
from sklearn.metrics import classification_report, f1_score, pairwise_distances
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.svm import SVC, LinearSVC
from tqdm import tqdm
from yellowbrick.cluster import KElbowVisualizer

if TYPE_CHECKING:
    from anndata import AnnData

from ..helperfunctions._general import *
from ..plotting._general import catplot

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
        If set to True, the function will skip the calculation of neighbors and UMAP. This can be used to speed up the process when just reclustering or running flowSOM.
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


def compute_triangulation_edges(df_input, x_pos, y_pos):
    """
    Compute unique Delaunay triangulation edges from input coordinates.

    This function computes the Delaunay triangulation for the set of points defined by the
    x and y positions contained in a DataFrame. It then extracts all unique edges from the
    triangulation, calculates their Euclidean distances, and returns these as a new DataFrame.

    Parameters
    ----------
    df_input : pandas.DataFrame
        DataFrame containing the coordinate data.
    x_pos : str
        The column name in df_input for the x-coordinate.
    y_pos : str
        The column name in df_input for the y-coordinate.

    Returns
    -------
    pandas.DataFrame
        A DataFrame with columns:
            - ind1: Index of the first point in each edge.
            - ind2: Index of the second point in each edge.
            - x1: x-coordinate of the first point.
            - y1: y-coordinate of the first point.
            - x2: x-coordinate of the second point.
            - y2: y-coordinate of the second point.
            - distance: Euclidean distance between the two points.
    """
    points = df_input[[x_pos, y_pos]].values
    tri = Delaunay(points)

    # Generate edges from triangles and remove duplicates
    edges = np.vstack(
        [tri.simplices[:, [0, 1]], tri.simplices[:, [1, 2]], tri.simplices[:, [2, 0]]]
    )
    # Sort each edge so that [i, j] and [j, i] are considered the same
    edges = np.sort(edges, axis=1)
    # Remove duplicate edges
    edges = np.unique(edges, axis=0)

    # Vectorized distance computation
    x_coords = points[:, 0]
    y_coords = points[:, 1]

    ind1, ind2 = edges[:, 0], edges[:, 1]
    x1_arr, y1_arr = x_coords[ind1], y_coords[ind1]
    x2_arr, y2_arr = x_coords[ind2], y_coords[ind2]

    dist_arr = np.sqrt((x2_arr - x1_arr) ** 2 + (y2_arr - y1_arr) ** 2)

    edges_df = pd.DataFrame(
        {
            "ind1": ind1,
            "ind2": ind2,
            "x1": x1_arr,
            "y1": y1_arr,
            "x2": x2_arr,
            "y2": y2_arr,
            "distance": dist_arr,
        }
    )
    return edges_df


def annotate_triangulation_vectorized(
    edges_df, df_input, id_col, x_pos, y_pos, cell_type_col, region
):
    """
    Annotate edges with cell metadata in a vectorized manner.

    This function takes the computed edges from the triangulation and annotates them with
    additional information retrieved from the input DataFrame. It creates both the forward
    and reverse (symmetrical) edges with cell identifiers, cell types, positions, and region info.

    Parameters
    ----------
    edges_df : pandas.DataFrame
        DataFrame containing the triangulation edges and their distances.
    df_input : pandas.DataFrame
        DataFrame containing cell metadata.
    id_col : str
        The column name in df_input that serves as the cell identifier.
    x_pos : str
        The column name in df_input for the x-coordinate.
    y_pos : str
        The column name in df_input for the y-coordinate.
    cell_type_col : str
        The column name in df_input for cell type annotation.
    region : str
        The column name in df_input for region information.

    Returns
    -------
    pandas.DataFrame
        A DataFrame containing annotated edges with the following columns:
            - region: The region identifier.
            - celltype1_index, celltype1, celltype1_X, celltype1_Y:
                Information for the first cell.
            - celltype2_index, celltype2, celltype2_X, celltype2_Y:
                Information for the second cell.
            - distance: The Euclidean distance between the two cells.
    """
    if len(df_input[region].unique()) == 1:
        region_val = df_input[region].iloc[0]
    else:
        # In case of multiple regions, use the first region as annotation.
        region_val = df_input[region].iloc[0]

    # Convert needed columns to arrays for fast indexing
    id_array = df_input[id_col].values
    ct_array = df_input[cell_type_col].values
    x_array = df_input[x_pos].values
    y_array = df_input[y_pos].values

    # Build references from edges DataFrame
    ind1 = edges_df["ind1"].values
    ind2 = edges_df["ind2"].values
    x1_arr = edges_df["x1"].values
    y1_arr = edges_df["y1"].values
    x2_arr = edges_df["x2"].values
    y2_arr = edges_df["y2"].values
    dist_arr = edges_df["distance"].values

    # Create direct "forward" annotated DataFrame
    data_forward = pd.DataFrame(
        {
            region: [region_val] * len(ind1),
            "celltype1_index": id_array[ind1],
            "celltype1": ct_array[ind1],
            "celltype1_X": x1_arr,
            "celltype1_Y": y1_arr,
            "celltype2_index": id_array[ind2],
            "celltype2": ct_array[ind2],
            "celltype2_X": x2_arr,
            "celltype2_Y": y2_arr,
            "distance": dist_arr,
        }
    )

    # Create symmetrical (reverse) annotated DataFrame
    data_reverse = pd.DataFrame(
        {
            region: [region_val] * len(ind1),
            "celltype1_index": id_array[ind2],
            "celltype1": ct_array[ind2],
            "celltype1_X": x2_arr,
            "celltype1_Y": y2_arr,
            "celltype2_index": id_array[ind1],
            "celltype2": ct_array[ind1],
            "celltype2_X": x1_arr,
            "celltype2_Y": y1_arr,
            "distance": dist_arr,
        }
    )

    # Concatenate forward and reverse dataframes
    annotated_result = pd.concat([data_forward, data_reverse], ignore_index=True)
    annotated_result = annotated_result[
        [
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
    ]
    return annotated_result


def calculate_triangulation_distances(df_input, id, x_pos, y_pos, cell_type, region):
    """
    Calculate and annotate triangulation distances for cells.

    This function computes the triangulation edges for input cell data and then annotates
    them with cell metadata. It serves as a wrapper combining both steps into one process.

    Parameters
    ----------
    df_input : pandas.DataFrame
        DataFrame containing the cell data.
    id : str
        Column name for cell identifiers.
    x_pos : str
        Column name for the x-coordinate.
    y_pos : str
        Column name for the y-coordinate.
    cell_type : str
        Column name for cell type information.
    region : str
        Column name for region information.

    Returns
    -------
    pandas.DataFrame
        Annotated DataFrame with triangulation edges and metadata.
    """
    edges_df = compute_triangulation_edges(df_input, x_pos, y_pos)
    annotated_result = annotate_triangulation_vectorized(
        edges_df, df_input, id, x_pos, y_pos, cell_type, region
    )
    return annotated_result


def process_region(df, unique_region, id, x_pos, y_pos, cell_type, region):
    """
    Process triangulation distances for a specific region.

    This function subsets the dataframe to one specific region, adds unique identifier
    columns, and calculates the triangulation distances for that region.

    Parameters
    ----------
    df : pandas.DataFrame
        The full dataset containing cell information.
    unique_region : str
        The specific region to process.
    id : str
        Column name for cell identifiers.
    x_pos : str
        Column name for x-coordinate.
    y_pos : str
        Column name for y-coordinate.
    cell_type : str
        Column name for cell type information.
    region : str
        Column name for region information.

    Returns
    -------
    pandas.DataFrame
        Annotated DataFrame with triangulation distances for the specified region.
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
    Compute triangulation distances for each unique region with parallel processing.

    This function processes the input DataFrame by first ensuring datatype consistency
    (optionally converting coordinate values to integers), and then computes triangulation
    distances per region in parallel using half of the available CPU cores (by default).

    Parameters
    ----------
    df_input : pandas.DataFrame
        DataFrame containing cell data including coordinates, cell types, and region info.
    id : str
        Column name for cell identifiers.
    x_pos : str
        Column name for the x-coordinate.
    y_pos : str
        Column name for the y-coordinate.
    cell_type : str
        Column name for cell type information.
    region : str
        Column name for region information.
    num_cores : int, optional
        Number of CPU cores to use for parallel processing. If None, defaults to half of os.cpu_count().
    correct_dtype : bool, optional
        Flag to convert columns to proper data types. Defaults to True.

    Returns
    -------
    pandas.DataFrame
        A concatenated DataFrame with triangulation distances computed for all regions.
    """
    if correct_dtype:
        df_input[cell_type] = df_input[cell_type].astype(str)
        df_input[region] = df_input[region].astype(str)

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

    unique_regions = df_input[region].unique()
    df_input = df_input.loc[:, [id, x_pos, y_pos, cell_type, region]]

    if num_cores is None:
        num_cores = os.cpu_count() // 2

    # Parallelize region processing
    results = Parallel(n_jobs=num_cores)(
        delayed(process_region)(df_input, reg, id, x_pos, y_pos, cell_type, region)
        for reg in unique_regions
    )

    triangulation_distances = pd.concat(results)
    return triangulation_distances


def shuffle_annotations(df_input, cell_type, region, permutation):
    """
    Shuffle cell type annotations within each region.

    This function randomizes the cell type annotations of the input DataFrame on a per-region basis
    using a pseudo-random permutation seed.

    Parameters
    ----------
    df_input : pandas.DataFrame
        DataFrame containing cell data.
    cell_type : str
        Column name for cell type information.
    region : str
        Column name for region information.
    permutation : int
        An integer used to seed the random number generator for reproducible shuffling.

    Returns
    -------
    pandas.DataFrame
        A copy of df_input with an added column "random_annotations" representing the shuffled cell types.
    """
    np.random.seed(permutation + 1234)
    df_shuffled = df_input.copy()

    for region_name in df_shuffled[region].unique():
        region_mask = df_shuffled[region] == region_name
        shuffled_values = df_shuffled.loc[region_mask, cell_type].sample(frac=1).values
        df_shuffled.loc[region_mask, "random_annotations"] = shuffled_values

    return df_shuffled


def _process_region_iterations(
    subset,
    edges_df,
    id_col,
    x_col,
    y_col,
    cell_type_col,
    region_col,
    region_val,
    num_iterations,
):
    """
    Process multiple iterations of permutation for a given region.

    This helper function takes a subset of the data and precomputed triangulation edges
    and performs a series of iterations where cell type annotations are shuffled and the
    mean distances are computed for each permutation.

    Parameters
    ----------
    subset : pandas.DataFrame
        DataFrame containing a subset of cell data for a single region.
    edges_df : pandas.DataFrame
        Precomputed triangulation edges for the subset.
    id_col : str
        Column name for cell identifiers.
    x_col : str
        Column name for the x-coordinate.
    y_col : str
        Column name for the y-coordinate.
    cell_type_col : str
        Column name for cell type or annotation to be shuffled.
    region_col : str
        Column name for region information.
    region_val : str
        The specific region value being processed.
    num_iterations : int
        Number of permutation iterations to perform.

    Returns
    -------
    pandas.DataFrame
        A DataFrame concatenating the mean distance summaries for each iteration.
    """
    results_list = []
    for iteration in range(1, num_iterations + 1):
        shuffled = shuffle_annotations(subset, cell_type_col, region_col, iteration)
        annotated_df = annotate_triangulation_vectorized(
            edges_df, shuffled, id_col, x_col, y_col, "random_annotations", region_col
        )
        per_cell_summary = (
            annotated_df.groupby(["celltype1_index", "celltype1", "celltype2"])
            .distance.mean()
            .reset_index(name="per_cell_mean_dist")
        )
        per_celltype_summary = (
            per_cell_summary.groupby(["celltype1", "celltype2"])
            .per_cell_mean_dist.mean()
            .reset_index(name="mean_dist")
        )
        per_celltype_summary[region_col] = region_val
        per_celltype_summary["iteration"] = iteration
        results_list.append(per_celltype_summary)
    return pd.concat(results_list, ignore_index=True)


def tl_iterate_tri_distances(
    df_input, id, x_pos, y_pos, cell_type, region, num_cores=None, num_iterations=1000
):
    """
    Perform iterative permutation analysis for triangulation distances.

    This function iterates over each unique region to calculate permutation-based triangulation
    distance summaries using precomputed edges. It applies parallel processing to perform
    multiple iterations efficiently.

    Parameters
    ----------
    df_input : pandas.DataFrame
        DataFrame containing the cell information.
    id : str
        Column name for cell identifiers.
    x_pos : str
        Column name for the x-coordinate.
    y_pos : str
        Column name for the y-coordinate.
    cell_type : str
        Column name for cell type information.
    region : str
        Column name for region information.
    num_cores : int, optional
        Number of CPU cores to use for parallelization. Defaults to half of os.cpu_count() if None.
    num_iterations : int, optional
        Number of permutation iterations to perform. Defaults to 1000.

    Returns
    -------
    pandas.DataFrame
        A concatenated DataFrame with permutation-based mean distances for each region.
    """
    unique_regions = df_input[region].unique()
    df_input = df_input[[id, x_pos, y_pos, cell_type, region]]

    # Precompute triangulation edges for each region
    region2df = {}
    region2edges_df = {}
    for reg_name in unique_regions:
        subset = df_input[df_input[region] == reg_name].copy()
        subset["uniqueID"] = (
            subset[id].astype(str)
            + "-"
            + subset[x_pos].astype(str)
            + "-"
            + subset[y_pos].astype(str)
        )
        subset["XYcellID"] = subset[x_pos].astype(str) + "_" + subset[y_pos].astype(str)
        edges_df = compute_triangulation_edges(subset, x_pos, y_pos)
        region2df[reg_name] = subset
        region2edges_df[reg_name] = edges_df

    def process_one_region(r):
        """TODO: Add description.
        
        Parameters
        ----------
        r : Any
            TODO: Describe this parameter.
        
        Returns
        -------
        Any
            TODO: Describe return value.
        """
        subset = region2df[r]
        edges_df = region2edges_df[r]
        return _process_region_iterations(
            subset, edges_df, id, x_pos, y_pos, cell_type, region, r, num_iterations
        )

    results_per_region = Parallel(
        n_jobs=num_cores if num_cores is not None else os.cpu_count() // 2
    )(delayed(process_one_region)(r) for r in unique_regions)
    iterative_triangulation_distances = pd.concat(results_per_region, ignore_index=True)
    return iterative_triangulation_distances


def add_missing_columns(
    triangulation_distances, metadata, shared_column="unique_region"
):
    """
    Add missing metadata columns to the triangulation distances DataFrame.

    This function compares the metadata DataFrame with the triangulation distances DataFrame
    and adds any columns from the metadata that are not present. It uses the shared_column
    to map values and fills any missing values with "Unknown".

    Parameters
    ----------
    triangulation_distances : pandas.DataFrame
        DataFrame containing triangulation distances and possibly missing metadata columns.
    metadata : pandas.DataFrame
        DataFrame containing additional metadata including the shared column.
    shared_column : str, optional
        Column name that is common to both DataFrames, by default "unique_region".

    Returns
    -------
    pandas.DataFrame
        The updated triangulation distances DataFrame with added metadata columns.
    """
    missing_columns = set(metadata.columns) - set(triangulation_distances.columns)
    for column in missing_columns:
        triangulation_distances[column] = pd.NA
        region_to_tissue = pd.Series(
            metadata[column].values, index=metadata["unique_region"]
        ).to_dict()
        triangulation_distances[column] = triangulation_distances["unique_region"].map(
            region_to_tissue
        )
        triangulation_distances[column].fillna("Unknown", inplace=True)
    return triangulation_distances


def calculate_pvalue(row):
    """
    Calculate the p-value using the Mann-Whitney U test.

    For a given row containing expected and observed lists of distances, this function
    computes the p-value from the Mann-Whitney U test comparing the two distributions.
    If the test fails, a NaN is returned.

    Parameters
    ----------
    row : pandas.Series
        A row containing "expected" and "observed" distance lists.

    Returns
    -------
    float
        The p-value computed from the Mann-Whitney U test, or NaN if computation fails.
    """
    try:
        return st.mannwhitneyu(
            row["expected"], row["observed"], alternative="two-sided"
        ).pvalue
    except ValueError:
        return np.nan


def identify_interactions(
    adata,
    cellid,
    x_pos,
    y_pos,
    cell_type,
    region,
    comparison,
    min_observed=10,
    distance_threshold=128,
    num_cores=None,
    num_iterations=1000,
    key_name=None,
    correct_dtype=False,
    aggregate_per_cell=True,
):
    """
    Identify significant cell-cell interactions based on spatial distances.

    This function processes the input annotated data (adata) to compute observed triangulation
    distances and perform permutation testing to generate expected distances. It then compares
    the observed with expected mean distances using the Mann-Whitney U test to compute a p-value
    and a log-fold change for each pair of cell types. The results are stored back in the adata
    object and returned.

    Parameters
    ----------
    adata : AnnData
        Annotated data object that holds cell observation data (adata.obs).
    cellid : str
        Column name to be used as the unique cell identifier.
    x_pos : str
        Column name for the x-coordinate.
    y_pos : str
        Column name for the y-coordinate.
    cell_type : str
        Column name for cell type information.
    region : str
        Column name for region information.
    comparison : str
        Column name used to compare different conditions.
    min_observed : int, optional
        Minimum number of observed distance measurements required to consider a significant interaction (default: 10).
    distance_threshold : int, optional
        Maximum distance to consider when grouping cell interactions (default: 128).
    num_cores : int, optional
        Number of CPU cores to use for parallel processing. Defaults to half of available cores if None.
    num_iterations : int, optional
        The number of permutation iterations for generating expected distances (default: 1000).
    key_name : str, optional
        Key under which the triangulation distances will be stored in adata.uns. If None, defaults to "triDist".
    correct_dtype : bool, optional
        Flag to convert coordinate and region columns to string types (default: False).
    aggregate_per_cell : bool, optional
        Whether to aggregate distances initially at a per-cell basis (default: True).

    Returns
    -------
    tuple
        A tuple containing:
            - distance_pvals (pandas.DataFrame): DataFrame with p-values and log-fold changes for each pair of cell types.
            - triangulation_distances_dict (dict): Dictionary containing observed and iterated triangulation distance DataFrames.
    """
    df_input = pd.DataFrame(adata.obs)
    if cellid in df_input.columns:
        df_input.index = df_input[cellid]
    else:
        print(cellid + " is not in the adata.obs, use index as cellid instead!")
        df_input[cellid] = df_input.index

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
    if key_name is None:
        triDist_keyname = "triDist"
    else:
        triDist_keyname = key_name
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

    metadata = df_input.loc[:, ["unique_region", comparison]].copy()
    # Reformat observed dataset
    triangulation_distances_long = add_missing_columns(
        triangulation_distances, metadata, shared_column=region
    )
    if aggregate_per_cell:
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

    observed_distances["keep"] = observed_distances["observed"].apply(
        lambda x: len(x) > min_observed
    )
    observed_distances = observed_distances[observed_distances["keep"]]

    expected_distances["keep"] = expected_distances["expected"].apply(
        lambda x: len(x) > min_observed
    )
    expected_distances = expected_distances[expected_distances["keep"]]

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

    # Collect final results
    triangulation_distances_dict = {
        "distance_pvals": distance_pvals,
        "triangulation_distances_observed": iterated_triangulation_distances_long,
        "triangulation_distances_iterated": triangulation_distances_long,
    }

    return distance_pvals, triangulation_distances_dict


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


def stellar_get_edge_index(
    pos,
    distance_thres,
    max_memory_usage=1.6e10,
    chunk_size=5000,
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
    estimated_memory_usage = n_samples * n_samples * 8  # ~float64 for distance matrix

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
    celltype_col="cell_type",
    region_column=None,
    x_col="x",
    y_col="y",
    sample_rate=0.5,
    distance_thres=50,
    epochs=50,
    num_seed_class=3,
    key_added="stellar_pred",
    STELLAR_path="",
    max_memory_usage=1.6e10,
    chunk_size=5000,
    wd=5e-2,
    lr=1e-3,
    seed=1,
    batch_size=1,
):
    """
    Apply the STELLAR algorithm to annotated and unannotated spatial single-cell data.

    This function processes the input AnnData objects by preparing the training data,
    constructing graph edges based on spatial coordinates, and then running the STELLAR
    algorithm for label prediction. When a region column is provided, the edge computations
    are performed for each region separately and the resulting edges are concatenated.

    Parameters
    ----------
    adata_train : AnnData
        The annotated single-cell data used for training.
    adata_unannotated : AnnData
        The unannotated single-cell data for which predictions are desired.
    celltype_col : str, optional
        Column name in adata_train.obs that contains the cell type labels, by default "cell_type".
    region_column : str or None, optional
        Column name to partition data into regions. If not None, edges are computed independently
        per region, by default None.
    x_col : str, optional
        Column name in the AnnData objects denoting the x-coordinate, by default "x".
    y_col : str, optional
        Column name in the AnnData objects denoting the y-coordinate, by default "y".
    sample_rate : float, optional
        The rate at which to sample the training data (between 0 and 1), by default 0.5.
    distance_thres : int, optional
        Distance threshold (in the same unit as the spatial coordinates) used to determine whether
        a pair of cells is connected, by default 50.
    epochs : int, optional
        Number of training epochs for the STELLAR model, by default 50.
    num_seed_class : int, optional
        Number of seed classes, which are appended to the number of unique cell types, by default 3.
    key_added : str, optional
        Key under which the predicted labels will be stored in adata_unannotated.obs, by default "stellar_pred".
    STELLAR_path : str, optional
        Filesystem path to the STELLAR repository. This path is added to sys.path, by default "".
    max_memory_usage : float, optional
        Maximum allowable memory usage in bytes when computing pairwise distances; if exceeded,
        the computation will be done in chunks, by default 1.6e10.
    chunk_size : int, optional
        The size of chunks to use for edge computation when memory usage is high, by default 5000.
    wd : float, optional
        Weight decay parameter for model optimization, by default 5e-2.
    lr : float, optional
        Learning rate for model training, by default 1e-3.
    seed : int, optional
        Seed used for reproducibility, by default 1.
    batch_size : int, optional
        Batch size for model training, by default 1.

    Returns
    -------
    AnnData
        The unannotated AnnData object with an additional observation column (key_added) containing
        the predicted cell type labels.

    Notes
    -----
    The function performs the following steps:
      1. Prints a citation reminder for the STELLAR algorithm.
      2. Sets up the model arguments by parsing command-line-like arguments.
      3. Prepares the training data by concatenating coordinate information and cell types, and
         builds a mapping between original and sampled indices.
      4. Computes graph edges either globally or per region (if region_column is provided) using the
         provided spatial coordinates and distance threshold.
      5. Constructs a GraphDataset and runs the STELLAR algorithm on it.
      6. Returns the modified adata_unannotated with predictions stored in obs[key_added].

    The function assumes that helper functions (e.g., stellar_get_edge_index) and necessary modules,
    including torch, argparse, and dataset utility modules, are available in the environment.
    """
    print(
        "Please consider to cite the following paper when using STELLAR: "
        "Brbić, M., Cao, K., Hickey, J.W. et al. Annotation of spatially resolved single-cell data with STELLAR. "
        "Nat Methods 19, 1411–1418 (2022). https://doi.org/10.1038/s41592-022-01651-8"
    )

    sys.path.append(str(STELLAR_path))
    from datasets import GraphDataset
    from STELLAR import STELLAR
    from utils import prepare_save_dir

    parser = argparse.ArgumentParser(description="STELLAR")
    parser.add_argument("--seed", type=int, default=seed, metavar="S")
    parser.add_argument("--name", type=str, default="STELLAR")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=lr)
    parser.add_argument("--wd", type=float, default=wd)
    # Don't set input_dim here - let STELLAR set it from the dataset
    parser.add_argument("--num-heads", type=int, default=13)
    parser.add_argument("--num-seed-class", type=int, default=num_seed_class)
    parser.add_argument("--sample-rate", type=float, default=sample_rate)
    parser.add_argument("-b", "--batch-size", default=batch_size, type=int, metavar="N")
    parser.add_argument("--distance_thres", default=50, type=int)
    parser.add_argument("--savedir", type=str, default="./")

    args = parser.parse_args(args=[])
    args.cuda = torch.cuda.is_available()
    args.device = torch.device("cuda" if args.cuda else "cpu")
    args.epochs = epochs
    args.distance_thres = distance_thres
    # Set the number of heads based on the number of cell types (IMPORTANT)
    args.num_heads = len(adata_train.obs[celltype_col].unique()) + num_seed_class

    print("Preparing input data")
    # Create a mapping between original cell indices and sampled indices
    train_df = adata_train.to_df()
    positions_celltype = adata_train.obs[[x_col, y_col, celltype_col]]
    train_df = pd.concat([train_df, positions_celltype], axis=1)

    # Create a key to track the original indices before sampling
    train_df["original_idx"] = np.arange(len(train_df))
    # train_df = train_df.sample(n=round(sample_rate * len(train_df)), random_state=1)

    # Get the mapping from sampled indices to original indices
    sampled_to_original = dict(enumerate(train_df["original_idx"].values))
    original_to_sampled = {v: k for k, v in sampled_to_original.items()}

    # Remove the helper column
    train_df = train_df.drop(columns=["original_idx"])

    train_X = train_df.iloc[:, :-3].values
    test_X = adata_unannotated.to_df().values

    # DO NOT convert to lowercase - keep original cell type labels
    train_y = train_df[celltype_col].values

    labeled_pos = train_df.iloc[:, -3:-1].values
    unlabeled_pos = adata_unannotated.obs[[x_col, y_col]].values

    # Print information about the data to help with debugging
    print(f"Number of unique cell types: {len(np.unique(train_y))}")
    print(f"Cell types: {np.unique(train_y)[:10]}")

    cell_types = np.sort(list(set(train_y))).tolist()
    cell_type_dict = {}
    inverse_dict = {}
    for i, t in enumerate(cell_types):
        cell_type_dict[t] = i
        inverse_dict[i] = t

    # Print dictionaries to verify proper mapping
    print(f"Cell type mapping size: {len(cell_type_dict)}")
    print(f"First 3 mappings: {list(cell_type_dict.items())[:3]}")

    train_y = np.array([cell_type_dict[x] for x in train_y])

    # If region_column is provided, compute edges per region and concatenate
    if region_column is not None:
        labeled_edges_list = []
        unlabeled_edges_list = []
        train_regions_map = {}  # To map region cells to train_df indices

        # Create a mapping between original cells and their position in the sampled data
        sampled_cells = set(train_df.index)

        # Get unique regions
        unique_regions_train = adata_train.obs[region_column].unique()
        unique_regions_unannot = adata_unannotated.obs[region_column].unique()

        # For each region in annotated data
        print("Processing labeled data")
        for region in unique_regions_train:
            print(f"Processing region: {region}")
            # Get cells in this region that are also in the sampled training data
            region_mask_full = adata_train.obs[region_column] == region
            # Extract positions for just those cells that are in the training data
            region_cells = adata_train[region_mask_full].obs_names
            region_sampled_cells = [
                cell for cell in region_cells if cell in sampled_cells
            ]

            if len(region_sampled_cells) < 2:
                print(
                    f"  Skipping region {region}: too few sampled cells ({len(region_sampled_cells)})"
                )
                continue

            # Get positions of sampled cells in this region
            region_pos = train_df.loc[region_sampled_cells, [x_col, y_col]].values

            # Create mapping from local indices to global training indices
            local_to_global = {
                i: original_to_sampled.get(adata_train.obs_names.get_loc(cell), None)
                for i, cell in enumerate(region_sampled_cells)
            }
            local_to_global = {
                k: v for k, v in local_to_global.items() if v is not None
            }

            if len(local_to_global) < 2:
                print(
                    f"  Skipping region {region}: too few cells with valid mapping ({len(local_to_global)})"
                )
                continue

            # Calculate edges using local indices
            region_edges = stellar_get_edge_index(
                region_pos,
                distance_thres=distance_thres,
                max_memory_usage=max_memory_usage,
                chunk_size=chunk_size,
            )

            # Convert to global indices for the training set
            valid_edges = []
            for edge in region_edges:
                src, dst = edge
                if src in local_to_global and dst in local_to_global:
                    valid_edges.append([local_to_global[src], local_to_global[dst]])

            if valid_edges:
                labeled_edges_list.extend(valid_edges)

        # For each region in unannotated data
        print("Processing unlabeled data")
        for region in unique_regions_unannot:
            print(f"Processing region: {region}")
            region_mask = adata_unannotated.obs[region_column] == region
            region_indices = np.where(region_mask)[0]

            if len(region_indices) < 2:
                print(
                    f"  Skipping region {region}: too few cells ({len(region_indices)})"
                )
                continue

            # Get positions
            region_pos = adata_unannotated.obs.loc[region_mask, [x_col, y_col]].values

            # Calculate edges
            region_edges = stellar_get_edge_index(
                region_pos,
                distance_thres=distance_thres,
                max_memory_usage=max_memory_usage,
                chunk_size=chunk_size,
            )

            # Map local indices to global indices
            valid_edges = []
            for edge in region_edges:
                src_local, dst_local = edge
                if src_local < len(region_indices) and dst_local < len(region_indices):
                    src_global = region_indices[src_local]
                    dst_global = region_indices[dst_local]
                    if src_global < len(test_X) and dst_global < len(test_X):
                        valid_edges.append([src_global, dst_global])

            if valid_edges:
                unlabeled_edges_list.extend(valid_edges)

        # Convert edge lists to arrays
        if labeled_edges_list:
            labeled_edges = np.array(labeled_edges_list)
            # Final sanity check for labeled edges
            max_allowed_idx = len(train_X) - 1
            valid_mask = (labeled_edges[:, 0] <= max_allowed_idx) & (
                labeled_edges[:, 1] <= max_allowed_idx
            )
            labeled_edges = labeled_edges[valid_mask]
        else:
            labeled_edges = np.array([]).reshape(0, 2)

        if unlabeled_edges_list:
            unlabeled_edges = np.array(unlabeled_edges_list)
            # Final sanity check for unlabeled edges
            max_allowed_idx = len(test_X) - 1
            valid_mask = (unlabeled_edges[:, 0] <= max_allowed_idx) & (
                unlabeled_edges[:, 1] <= max_allowed_idx
            )
            unlabeled_edges = unlabeled_edges[valid_mask]
        else:
            unlabeled_edges = np.array([]).reshape(0, 2)

        print(f"Final labeled edges: {labeled_edges.shape[0]}")
        print(f"Final unlabeled edges: {unlabeled_edges.shape[0]}")
    else:
        # Standard approach with global edge indices
        labeled_edges = stellar_get_edge_index(
            labeled_pos,
            distance_thres=distance_thres,
            max_memory_usage=max_memory_usage,
            chunk_size=chunk_size,
        )
        unlabeled_edges = stellar_get_edge_index(
            unlabeled_pos,
            distance_thres=distance_thres,
            max_memory_usage=max_memory_usage,
            chunk_size=chunk_size,
        )

        # Convert to numpy arrays
        labeled_edges = np.array(labeled_edges)
        unlabeled_edges = np.array(unlabeled_edges)

        # Final sanity checks
        max_train_idx = len(train_X) - 1
        max_test_idx = len(test_X) - 1

        valid_mask = (labeled_edges[:, 0] <= max_train_idx) & (
            labeled_edges[:, 1] <= max_train_idx
        )
        labeled_edges = labeled_edges[valid_mask]

        valid_mask = (unlabeled_edges[:, 0] <= max_test_idx) & (
            unlabeled_edges[:, 1] <= max_test_idx
        )
        unlabeled_edges = unlabeled_edges[valid_mask]

    print("Building dataset")
    # Debug info to verify edge indices are within bounds
    print(f"Training data shape: {train_X.shape}, max label index: {len(train_X)-1}")
    print(f"Testing data shape: {test_X.shape}, max label index: {len(test_X)-1}")
    if len(labeled_edges) > 0:
        print(f"Max labeled edge index: {np.max(labeled_edges)}")
    if len(unlabeled_edges) > 0:
        print(f"Max unlabeled edge index: {np.max(unlabeled_edges)}")

    # Make sure all edge indices are within bounds
    if len(labeled_edges) > 0:
        max_idx = len(train_X) - 1
        valid_mask = np.all(labeled_edges <= max_idx, axis=1)
        if not np.all(valid_mask):
            print(f"Removing {(~valid_mask).sum()} out-of-bounds labeled edges")
            labeled_edges = labeled_edges[valid_mask]

    if len(unlabeled_edges) > 0:
        max_idx = len(test_X) - 1
        valid_mask = np.all(unlabeled_edges <= max_idx, axis=1)
        if not np.all(valid_mask):
            print(f"Removing {(~valid_mask).sum()} out-of-bounds unlabeled edges")
            unlabeled_edges = unlabeled_edges[valid_mask]

    # Ensure we have at least some edges - STELLAR will fail without edges
    if len(labeled_edges) == 0 or len(unlabeled_edges) == 0:
        print(
            "WARNING: No edges found for labeled or unlabeled data - STELLAR requires edges for both datasets"
        )
        if len(labeled_edges) == 0:
            print("Adding token edge to labeled data")
            labeled_edges = np.array([[0, min(1, len(train_X) - 1)]])
        if len(unlabeled_edges) == 0:
            print("Adding token edge to unlabeled data")
            unlabeled_edges = np.array([[0, min(1, len(test_X) - 1)]])

    # Build dataset
    dataset = GraphDataset(train_X, train_y, test_X, labeled_edges, unlabeled_edges)

    # IMPORTANT: Set input dimension correctly from dataset
    args.input_dim = train_X.shape[1]
    print(f"Setting input dimension to: {args.input_dim}")

    print("Running STELLAR")
    stellar = STELLAR(args, dataset)

    # Additional diagnostic info
    model_params = sum(p.numel() for p in stellar.model.parameters())
    print(f"Model has {model_params} parameters")

    # Execute training
    stellar.train()

    # Get predictions
    uncertainty, results = stellar.pred()
    print(f"Mean prediction uncertainty: {uncertainty:.4f}")

    # Count predictions per class to check for uniformity
    unique_preds, counts = np.unique(results, return_counts=True)
    print("Prediction distribution:")
    for pred, count in zip(unique_preds, counts):
        print(f"Class {pred}: {count} cells ({count/len(results)*100:.2f}%)")

    # Convert numerical predictions back to string labels
    results = results.astype("object")
    for i in range(len(results)):
        if results[i] in inverse_dict.keys():
            results[i] = inverse_dict[results[i]]
        else:
            # Handle new/unseen classes
            results[i] = f"New_Class_{results[i]}"

    adata_unannotated.obs[key_added] = pd.Categorical(results)
    adata_unannotated.obs[key_added] = adata_unannotated.obs[key_added].astype(str)

    return adata_unannotated


def ml_train(
    adata_train,
    label,
    test_size=0.2,
    random_state=0,
    nan_policy_y="omit",
    mode="accurate_SVC",  # 'accurate_SVC' or 'fast_SVC' or 'knn'
    showfig=True,
    figsize=(8, 6),
    n_neighbors=5,  # Number of neighbors for KNN
):
    """
    Train a classifier (SVC, LinearSVC, or KNN) on the data.

    Parameters
    ----------
    adata_train : anndata.AnnData
        The AnnData object containing the training data. The input features are expected in adata_train.X,
        and the target labels in adata_train.obs[label].
    label : str
        The column name in adata_train.obs to use as the target variable.
    test_size : float, optional
        The proportion of the dataset to include in the test split. Default is 0.2.
    random_state : int, optional
        Seed for the random number generator. Default is 0.
    nan_policy_y : {'omit', 'raise'}, optional
        Policy for handling NaNs in the target variable. 'omit' removes NaNs, 'raise' raises an error.
        Default is 'omit'.
    mode : {'accurate_SVC', 'fast_SVC', 'knn'}, optional
        The type of classifier to use.
        - 'accurate_SVC': Uses SVC with probability=True (slower, provides predict_proba).
        - 'fast_SVC': Uses LinearSVC with CalibratedClassifierCV (faster, optional predict_proba).
        - 'knn': Uses KNeighborsClassifier with specified n_neighbors.
        Default is 'accurate_SVC'.
    showfig : bool, optional
        Whether to display a heatmap of the classification report. Default is True.
    figsize : tuple, optional
        Size of the figure if showfig is True. Default is (8, 6).
    n_neighbors : int, optional
        Number of neighbors for KNN classifier. Default is 5.

    Returns
    -------
    svc : sklearn.base.BaseEstimator
        The trained classifier model. For 'accurate_SVC' and 'fast_SVC', the model supports predict_proba.
        For 'knn', only predict is available.

    Raises
    ------
    ValueError
        If `mode` is not one of the allowed options, or if `nan_policy_y` is not 'omit' or 'raise'.

    Notes
    -----
    - The function handles NaNs in the target variable based on `nan_policy_y`.
    - For 'accurate_SVC', the classifier provides `predict_proba` for probability estimates.
    - For 'fast_SVC', the classifier uses LinearSVC with calibration for `predict_proba`.
    - For 'knn', the classifier uses KNeighborsClassifier with the specified `n_neighbors`.
    - The classification report is displayed as a heatmap if `showfig` is True.
    - The function prints progress messages (e.g., "Preparing training data!", "Training now!", etc.).
    """
    print("Preparing training data!")
    X = adata_train.X
    if not isinstance(X, np.ndarray):
        X = X.toarray()
    y = adata_train.obs[label].values
    if nan_policy_y == "omit":
        y_msk = ~pd.isna(y)  # <-- pd.isna works fine with mixed types
        X = X[y_msk]
        y = y[y_msk]
    elif nan_policy_y != "raise":
        raise ValueError("nan_policy_y must be either 'omit' or 'raise'")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    print("Unique labels:", np.unique(y))  # Adjusted for clarity
    print("Training now!")
    if mode == "accurate_SVC":
        svc = SVC(kernel="linear", probability=True)  # SVC with probability
    elif mode == "knn":
        print("Using KNN classifier with n_neighbors = ", n_neighbors)
        svc = KNeighborsClassifier(n_neighbors=n_neighbors)
    elif mode == "fast_SVC":
        base_svc = LinearSVC()
        svc = CalibratedClassifierCV(
            base_svc
        )  # CalibratedClassifierCV to add predict_proba
    else:
        raise ValueError("mode must be 'accurate_SVC', 'fast_SVC', or 'knn'")
    svc.fit(X_train, y_train)
    print("Evaluating now!")
    # Predict probability only for 'accurate' mode
    if mode == "accurate_SVC":
        y_prob = svc.predict_proba(X_test)
        y_prob = pd.DataFrame(y_prob, columns=svc.classes_)
        svm_label = y_prob.idxmax(axis=1)
    elif mode == "knn":
        svm_label = svc.predict(X_test)  # No predict_proba for LinearSVC
    elif mode == "fast_SVC":
        svm_label = svc.predict(X_test)  # No predict_proba for LinearSVC
    # Generate classification report
    target_names = svc.classes_
    svm_eval = classification_report(
        y_true=y_test, y_pred=svm_label, target_names=target_names, output_dict=True
    )
    if showfig:
        plt.figure(figsize=figsize)
        sns.heatmap(pd.DataFrame(svm_eval).iloc[:-1, :].T, annot=True)
        plt.title(f"Classification Report ({mode})")
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
    """TODO: Add description.
    
    Parameters
    ----------
    CUDA : Any
        TODO: Describe this parameter.
    
    Returns
    -------
    Any
        TODO: Describe return value.
    """
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
        """TODO: Add description.
        
        Returns
        -------
        Any
            TODO: Describe return value.
        """
        def load_data(event=None):
            """TODO: Add description.
            
            Parameters
            ----------
            event : Any
                TODO: Describe this parameter.
            
            Returns
            -------
            Any
                TODO: Describe return value.
            """
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
            """TODO: Add description.
            
            Parameters
            ----------
            event : Any
                TODO: Describe this parameter.
            
            Returns
            -------
            Any
                TODO: Describe return value.
            """
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
            """TODO: Add description.
            
            Parameters
            ----------
            event : Any
                TODO: Describe this parameter.
            
            Returns
            -------
            Any
                TODO: Describe return value.
            """
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
            """TODO: Add description.
            
            Parameters
            ----------
            event : Any
                TODO: Describe this parameter.
            
            Returns
            -------
            Any
                TODO: Describe return value.
            """
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
            """TODO: Add description.
            
            Parameters
            ----------
            event : Any
                TODO: Describe this parameter.
            
            Returns
            -------
            Any
                TODO: Describe return value.
            """
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
            """TODO: Add description.
            
            Parameters
            ----------
            event : Any
                TODO: Describe this parameter.
            
            Returns
            -------
            Any
                TODO: Describe return value.
            """
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


# Functions for PPA
## Adjust clustering parameter to get the desired number of clusters
def apply_dbscan_clustering(
    df, min_samples=10, x_col="x", y_col="y", allow_single_cluster=True
):
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
        min_samples=min_samples,
        min_cluster_size=5,
        cluster_selection_epsilon=0.0,
        metric="euclidean",
        cluster_selection_method="eom",
        allow_single_cluster=allow_single_cluster,
    )
    coords = df[[x_col, y_col]].values
    labels = hdbscan.fit_predict(coords)
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
    concavity=2,
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
                        concavity,
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
        result = pd.DataFrame(
            columns=[x_column, y_column, "patch_id", identification_column]
        )
    if len(outline_list) > 0:
        outlines = pd.concat(outline_list)
    else:
        outlines = pd.DataFrame(
            columns=[x_column, y_column, "patch_id", identification_column]
        )
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
    """TODO: Add description.
    
    Parameters
    ----------
    args : Any
        TODO: Describe this parameter.
    nbrs : Any
        TODO: Describe this parameter.
    unique_clusters : Any
        TODO: Describe this parameter.
    
    Returns
    -------
    Any
        TODO: Describe return value.
    """
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
        concavity,
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
        - concavity : int
            Concavity parameter for hull generation.
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
        concavity=concavity,
    )
    # Get hull points from the DataFrame
    hull_points = pd.DataFrame(points[idxes], columns=[x_column, y_column])
    # Find nearest neighbors of hull points in the original DataFrame
    distances, indices = nbrs.kneighbors(hull_points[[x_column, y_column]])
    hull_nearest_neighbors = df.iloc[indices.flatten()]

    # Convert radius to a list if it's a single value
    if not isinstance(radius, (list, tuple, np.ndarray)):
        radius_list = [radius]
    else:
        radius_list = radius

    # Extract hull points coordinates
    hull_coords = hull_nearest_neighbors[[x_column, y_column]].values
    # Calculate distances from all points in full_df to all hull points
    distances = cdist(full_df[[x_column, y_column]].values, hull_coords)

    # Process each radius
    all_prox_points = []
    for r in radius_list:
        # Identify points within the circle for each hull point
        in_circle = distances <= r
        # Identify points from a different cluster for each hull point
        diff_cluster = (
            full_df[identification_column].values[:, np.newaxis]
            != hull_nearest_neighbors[identification_column].values
        )
        # Combine the conditions
        in_circle_diff_cluster = in_circle & diff_cluster
        # Collect all points within the circle but from a different cluster
        r_in_circle_diff_cluster = full_df[np.any(in_circle_diff_cluster, axis=1)]
        # Remove duplicates
        r_prox_points = r_in_circle_diff_cluster.drop_duplicates()
        # Add patch_id and distance_from_patch columns
        r_prox_points["patch_id"] = cluster
        r_prox_points["distance_from_patch"] = r

        all_prox_points.append(r_prox_points)

    # Combine results from all radii
    if all_prox_points:
        prox_points = pd.concat(all_prox_points, ignore_index=True)
        # If multiple radii were used, keep only the smallest distance for each point
        if len(radius_list) > 1:
            prox_points = prox_points.sort_values(
                "distance_from_patch"
            ).drop_duplicates(
                subset=[
                    col for col in prox_points.columns if col != "distance_from_patch"
                ]
            )
    else:
        # Create empty DataFrame with appropriate columns
        prox_points = pd.DataFrame(
            columns=full_df.columns.tolist() + ["patch_id", "distance_from_patch"]
        )

    return prox_points, hull_nearest_neighbors


def identify_hull_points(
    df,
    cluster_column="cluster",
    x_col="x",
    y_col="y",
    concave_hull_length_threshold=50,
    concavity=2,
):
    """
    Identify hull points with improved performance.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing spatial points and cluster labels.
    cluster_column : str, optional
        Column name for clusters, by default "cluster".
    x_col : str, optional
        Column name for the x-coordinate, by default "x".
    y_col : str, optional
        Column name for the y-coordinate, by default "y".
    concave_hull_length_threshold : int, optional
        Threshold for concave hull length, by default 50.
    concavity : int, optional
        Concavity parameter, by default 2.

    Returns
    -------
    pandas.DataFrame
        DataFrame of hull points sorted by patch_id and order.
    """
    clusters = sorted(set(df[cluster_column].unique()) - {-1})
    if not clusters:
        return pd.DataFrame(columns=[x_col, y_col, "patch_id"])

    hullpoints_list = []
    for cluster in clusters:
        mask = df[cluster_column] == cluster
        subset = df[mask]
        points = subset[[x_col, y_col]].values
        if len(points) < 3:
            continue
        idxes = concave_hull_indexes(
            points, concavity=concavity, length_threshold=concave_hull_length_threshold
        )
        hull_points = subset.iloc[idxes].reset_index(drop=True)
        hull_points["order"] = range(len(hull_points))
        hull_points["patch_id"] = cluster
        hullpoints_list.append(hull_points)

    if not hullpoints_list:
        return pd.DataFrame(columns=[x_col, y_col, "patch_id"])

    hull = pd.concat(hullpoints_list, ignore_index=True, sort=False)
    return hull.sort_values(by=["patch_id", "order"]).drop(columns="order")


def convert_dataframe_to_geojson(
    df,
    output_dir,
    region_name=None,
    x="x",
    y="y",
    sample_col=None,
    region_col="unique_region",
    patch_col="patch_id",
    geojson_prefix="hull_coordinates",
    save_geojson=True,
):
    """
    Convert a DataFrame into GeoJSON format with optional saving to file.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame with spatial coordinates.
    output_dir : str
        Directory in which GeoJSON files will be saved.
    region_name : str, optional
        Optional region name to create a subfolder, by default None.
    x : str, optional
        Column name for the x-coordinate, by default "x".
    y : str, optional
        Column name for the y-coordinate, by default "y".
    sample_col : str, optional
        Column name to separate by samples, by default None.
    region_col : str, optional
        Column name for the region, by default "unique_region".
    patch_col : str, optional
        Column name for the patch, by default "patch_id".
    geojson_prefix : str, optional
        Prefix for the GeoJSON filename, by default "hull_coordinates".
    save_geojson : bool, optional
        Whether to save the GeoJSON to disk, by default True.

    Returns
    -------
    list of dict
        Each dictionary contains a filename and the GeoJSON feature collection.
    """
    required_columns = [region_col, patch_col, x, y]
    if sample_col is not None:
        required_columns.append(sample_col)
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        raise KeyError(f"Missing required columns in dataframe: {missing_cols}")

    if save_geojson:
        if region_name is not None:
            region_dir = os.path.join(output_dir, f"region_{region_name}")
            os.makedirs(region_dir, exist_ok=True)
        else:
            region_dir = output_dir
            os.makedirs(region_dir, exist_ok=True)

    geojson_results = []
    sample_values = df[sample_col].unique() if sample_col is not None else [None]
    for sample in sample_values:
        if sample is None:
            sample_df = df
            sample_label = "all"
        else:
            sample_df = df[df[sample_col] == sample]
            sample_search = re.search(r"\d+", str(sample))
            sample_label = sample_search.group() if sample_search else str(sample)

        for region in sample_df[region_col].unique():
            region_df = sample_df[sample_df[region_col] == region]
            features, skipped, region_label = process_geojson_region(
                region_df, region, region_col, patch_col, x, y, sample_label
            )
            all_features = features
            if sample is not None:
                filename = f"{geojson_prefix}_sample-{sample_label}_region-{region_label}_separate_coordinates.geojson"
            else:
                filename = f"{geojson_prefix}_region-{region_label}_separate_coordinates.geojson"
            geojson_dict = {
                "type": "FeatureCollection",
                "features": all_features,
                "name": filename,
            }
            if save_geojson:
                geojson_path = os.path.join(region_dir, filename)
                with open(geojson_path, "w") as f:
                    json.dump(geojson_dict, f)
            geojson_results.append({"filename": filename, "geojson": geojson_dict})
        if skipped:
            print(f"Skipped {len(skipped)} clusters with insufficient points")
    return geojson_results


def process_geojson_region(
    region_df, region, region_col, patch_col, x, y, sample_label="all"
):
    """
    Process a single region to generate GeoJSON features.

    Parameters
    ----------
    region_df : pandas.DataFrame
        Subset DataFrame for the region.
    region : any
        Region identifier used to extract a label.
    region_col : str
        Column name indicating region information.
    patch_col : str
        Column name for patch identifiers.
    x : str
        Column name for x-coordinate.
    y : str
        Column name for y-coordinate.
    sample_label : str, optional
        Label for sample grouping, by default "all".

    Returns
    -------
    tuple
        A tuple containing:
        - features (list): List of GeoJSON feature dictionaries.
        - skipped_clusters (list): List of clusters skipped due to insufficient points.
        - region_label (str): Extracted region label.
    """
    region_search = re.search(r"\d+", str(region))
    region_label = region_search.group() if region_search else str(region)
    features = []
    skipped_clusters = []
    for cluster, cluster_df in region_df.groupby(patch_col):
        coordinates = cluster_df[[x, y]].values.tolist()
        if len(coordinates) >= 3:
            geom_type = "Polygon"
            coords_format = [coordinates]
        elif len(coordinates) == 2:
            geom_type = "LineString"
            coords_format = coordinates
        else:
            skipped_clusters.append((sample_label, region_label, cluster))
            continue
        feature = {
            "type": "Feature",
            "properties": {
                "sample": (
                    int(sample_label) if str(sample_label).isdigit() else sample_label
                ),
                "region": (
                    int(region_label) if str(region_label).isdigit() else region_label
                ),
                "cluster": int(cluster) if str(cluster).isdigit() else cluster,
            },
            "geometry": {
                "type": geom_type,
                "coordinates": coords_format,
            },
        }
        features.append(feature)
    return features, skipped_clusters, region_label


def extract_region_number(unique_region_value):
    """
    Extract the numeric part of a region identifier.

    Parameters
    ----------
    unique_region_value : int, float, or str
        The region value from which to extract digits.

    Returns
    -------
    str
        The numeric region value as a string.
    """
    try:
        if isinstance(unique_region_value, (int, float)):
            return str(int(unique_region_value))
        digits = "".join(filter(str.isdigit, str(unique_region_value)))
        return str(int(digits)) if digits else str(unique_region_value)
    except ValueError:
        return str(unique_region_value)


def analyze_peripheral_cells(
    patches_gdf, codex_gdf, buffer_distances, original_unit_scale, tolerance_distance
):
    """
    Analyze peripheral cells with parallel processing.

    Parameters
    ----------
    patches_gdf : geopandas.GeoDataFrame
        GeoDataFrame with patch geometries.
    codex_gdf : geopandas.GeoDataFrame
        GeoDataFrame with codex point geometries.
    buffer_distances : list of int
        List of distances to buffer.
    original_unit_scale : float
        Scale factor for the units.
    tolerance_distance : float
        Tolerance for determining peripheral regions.

    Returns
    -------
    tuple
        A tuple containing:
        - results (dict): Dictionary with keys as distances and values as DataFrames of peripheral cells.
        - buffer_geometries (dict): Dictionary with buffer geometries for visualization.
    """
    region_tasks = []
    for region_name, region_patches in patches_gdf.groupby("region_numeric"):
        region_codex_cells = codex_gdf[
            codex_gdf["unique_region_numeric"] == region_name
        ]
        if len(region_codex_cells) > 0:
            region_tasks.append(
                (
                    region_name,
                    region_patches,
                    region_codex_cells,
                    buffer_distances,
                    original_unit_scale,
                    tolerance_distance,
                )
            )
    results = {dist: [] for dist in buffer_distances}
    buffer_geometries = {dist: [] for dist in buffer_distances}
    max_workers = min(os.cpu_count(), len(region_tasks))
    if max_workers > 1:
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=max_workers
        ) as executor:
            for region_name, region_results, region_buffers in executor.map(
                process_region_peripheral_cells, region_tasks
            ):
                for dist, df in region_results.items():
                    if not df.empty:
                        results[dist].append(df)
                for dist, buffers in region_buffers.items():
                    buffer_geometries[dist].extend(buffers)
    else:
        for task in region_tasks:
            region_name, region_results, region_buffers = (
                process_region_peripheral_cells(task)
            )
            for dist, df in region_results.items():
                if not df.empty:
                    results[dist].append(df)
            for dist, buffers in region_buffers.items():
                buffer_geometries[dist].extend(buffers)
    for dist in buffer_distances:
        if results[dist]:
            results[dist] = pd.concat(results[dist], ignore_index=True)
        else:
            results[dist] = pd.DataFrame()
    return results, buffer_geometries


def save_peripheral_cells(results, unit_name, region_name, output_dir, save_csv=True):
    """
    Save peripheral cells for each buffer distance to CSV files.

    Parameters
    ----------
    results : dict
        Dictionary with keys as distances and values as DataFrames of peripheral cells.
    unit_name : str
        Name of the unit to include in filenames.
    region_name : str
        Region identifier used in filenames.
    output_dir : str
        Directory to save CSV files.
    save_csv : bool, optional
        Whether to save CSV files, by default True.

    Returns
    -------
    pandas.DataFrame
        Combined DataFrame of peripheral cells from all distances.
    """
    all_frames = []
    if save_csv:
        region_dir = os.path.join(output_dir, f"region_{region_name}")
        os.makedirs(region_dir, exist_ok=True)
    for dist, data in results.items():
        if data.empty:
            continue
        data["dist"] = dist
        all_frames.append(data)
        if save_csv:
            peripheral_path = os.path.join(
                region_dir,
                f"{unit_name}_region_{region_name}_peripheral_cells_{dist}um.csv",
            )
            data.to_csv(peripheral_path, index=False)
    combined_df = (
        pd.concat(all_frames, ignore_index=True) if all_frames else pd.DataFrame()
    )
    if save_csv and not combined_df.empty:
        combined_path = os.path.join(
            region_dir,
            f"{unit_name}_region_{region_name}_peripheral_cells_combined.csv",
        )
        combined_df.to_csv(combined_path, index=False)
    return combined_df


def process_region_peripheral_cells(args):
    """
    Process peripheral cells for a given region (for parallel processing).

    Parameters
    ----------
    args : tuple
        A tuple containing:
        - region_name : any
          Region identifier.
        - region_patches : geopandas.GeoDataFrame
          GeoDataFrame of patches in the region.
        - region_codex_cells : geopandas.GeoDataFrame
          GeoDataFrame of codex cells in the region.
        - buffer_distances : list of int
          List of distances to buffer.
        - original_unit_scale : float
          Original unit scale for distance conversion.
        - tolerance_distance : float
          Tolerance for buffering.

    Returns
    -------
    tuple
        A tuple containing:
        - region_name : any
        - results : dict
          Dictionary with peripheral cell DataFrames for each distance.
        - buffer_geometries : dict
          Dictionary with buffer geometry information.
    """
    (
        region_name,
        region_patches,
        region_codex_cells,
        buffer_distances,
        original_unit_scale,
        tolerance_distance,
    ) = args
    region_codex_cells_sindex = region_codex_cells.sindex
    results = {dist: [] for dist in buffer_distances}
    buffer_geometries = {dist: [] for dist in buffer_distances}
    for _, patch in region_patches.iterrows():
        patch_polygon = patch["geometry"]
        cluster_label = patch["cluster"]
        patch_id = patch.name
        for dist in buffer_distances:
            scaled_dist = dist / original_unit_scale
            expanded_patch = patch_polygon.buffer(scaled_dist)
            peripheral_region = expanded_patch.difference(patch_polygon).buffer(
                tolerance_distance
            )
            buffer_geometries[dist].append(
                {
                    "patch_id": patch_id,
                    "cluster": cluster_label,
                    "original": patch_polygon,
                    "expanded": expanded_patch,
                    "peripheral": peripheral_region,
                }
            )
            possible_matches_idx = list(
                region_codex_cells_sindex.intersection(peripheral_region.bounds)
            )
            possible_matches = region_codex_cells.iloc[possible_matches_idx]
            mask = possible_matches.geometry.within(peripheral_region)
            peripheral_cells = possible_matches[mask].copy()
            peripheral_cells["cluster"] = cluster_label
            peripheral_cells["buffer_distance"] = dist
            peripheral_cells["patch_id"] = patch_id
            results[dist].append(peripheral_cells)
    for dist in buffer_distances:
        if results[dist]:
            results[dist] = pd.concat(results[dist], ignore_index=True)
        else:
            results[dist] = pd.DataFrame()
    return region_name, results, buffer_geometries


def extract_unit_name(geojson):
    """
    Extract a unit name from a GeoJSON object.

    Parameters
    ----------
    geojson : dict or object with attribute 'name'
        The GeoJSON object or an object having a 'name' property.

    Returns
    -------
    str
        The extracted unit name.

    Raises
    ------
    ValueError
        If the GeoJSON does not have a 'name' property.
    """
    if hasattr(geojson, "name"):
        file_name = geojson.name
    elif isinstance(geojson, dict) and "name" in geojson:
        file_name = geojson["name"]
    else:
        raise ValueError("GeoJSON object does not have a 'name' property")
    file_name_no_ext = os.path.splitext(file_name)[0]
    parts = file_name_no_ext.split("_")
    if "ppa" in parts:
        return "_".join(parts[: parts.index("ppa")])
    return file_name_no_ext


def patch_proximity_analysis(
    adata,
    region_column,
    patch_column,
    group,
    min_cluster_size=80,
    x_column="x",
    y_column="y",
    radius=128,
    edge_neighbours=1,
    plot=True,
    savefig=False,
    output_dir="./",
    output_fname="",
    save_geojson=True,
    allow_single_cluster=True,
    method="border_cell_radius",
    concave_hull_length_threshold=50,
    concavity=2,
    original_unit_scale=1,
    tolerance_distance=0.001,
    key_name=None,
):
    """
    Performs a proximity analysis on patches of a given group within each region of a dataset.

    This function processes an AnnData object by extracting its cell observations and performing
    proximity analysis on a specified cell group (e.g. a cell type or neighborhood) within each
    region. Depending on the chosen method ("border_cell_radius" or "hull_expansion"), the analysis
    applies DBSCAN clustering, identifies concave hull boundaries, and then either determines nearby
    cells based on a fixed search radius or uses a peripheral buffering approach. Optionally, the
    function can plot visualization of the analysis and save outputs (figures, CSV files, and GeoJSON).

    Parameters
    ----------
    adata : AnnData
        The annotated data matrix of shape (n_obs x n_vars). Rows correspond to individual cells
        and columns to gene expression or other features.
    region_column : str
        The name of the column in adata.obs that contains region information.
    patch_column : str
        The name of the column in adata.obs that contains patch (or group) information.
    group : str
        The specific group (e.g. cell type or patch identifier) on which the proximity analysis
        is to be performed.
    min_cluster_size : int, optional
        The minimum number of cells required in a region to perform the analysis. Regions with fewer
        cells than this value will be skipped. Default is 80.
    x_column : str, optional
        The column name in adata.obs corresponding to the x-coordinate of each cell. Default is "x".
    y_column : str, optional
        The column name in adata.obs corresponding to the y-coordinate of each cell. Default is "y".
    radius : int, optional
        The distance (in spatial units) within which points are considered to be in proximity.
        This value is multiplied by original_unit_scale. Default is 128.
    edge_neighbours : int, optional
        The number of neighbouring edge points to consider when identifying proximity relationships.
        Default is 1.
    plot : bool, optional
        Whether to generate and display visualizations of the proximity analysis. Default is True.
    savefig : bool, optional
        Whether to save the generated figure to disk. Default is False.
    output_dir : str, optional
        The directory in which to save output files (figures, CSVs, or GeoJSON files). Default is "./".
    output_fname : str, optional
        The filename prefix to use when saving figures. Default is an empty string.
    save_geojson : bool, optional
        Whether to convert certain results to GeoJSON format and save them. Default is True.
    allow_single_cluster : bool, optional
        If True, allows DBSCAN to assign all cells to a single cluster even if no separate clusters
        exist. Default is True.
    method : str, optional
        The analysis method to use. Options are "border_cell_radius" (default) or "hull_expansion".
        Each method applies a different strategy for proximity detection.
    concave_hull_length_threshold : int, optional
        Threshold value used for generating the concave hull boundary. Default is 50.
    concavity : int, optional
        Parameter specifying the degree of concavity when calculating the hull boundary. Default is 2.
    original_unit_scale : int or float, optional
        A scaling factor to convert the radius from its given unit to the coordinate system unit.
        Default is 1.
    tolerance_distance : float, optional
        Tolerance value for buffering in the peripheral analysis (used when method is "hull_expansion").
        Default is 0.001.
    key_name : str, optional
        The key under which the final proximity analysis results are stored in adata.uns. If not
        provided, defaults to "ppa_result".

    Returns
    -------
    final_results : pandas.DataFrame
        A DataFrame containing the combined proximity analysis results from all processed regions.
        It includes, among other information, a newly generated "unique_patch_ID" column that
        concatenates the region, group, and patch identifier.
    outlines_results : pandas.DataFrame
        A DataFrame containing the outline (or hull) points corresponding to the patches; useful for
        visualization or further spatial analysis.
    """
    # multiply radius by original_unit_scale
    if isinstance(radius, (list, tuple)):
        radius = [r * original_unit_scale for r in radius]
    else:
        radius = radius * original_unit_scale

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    distance_from_patch = radius
    # make list
    if isinstance(distance_from_patch, int):
        distance_from_patch = [distance_from_patch]

    # Get data from adata
    df = adata.obs

    # Check if the required columns are present in the DataFrame
    if region_column not in df.columns:
        raise ValueError(f"Column '{region_column}' not found in adata.obs")
    if patch_column not in df.columns:
        raise ValueError(f"Column '{patch_column}' not found in adata.obs")
    if group not in df[patch_column].unique():
        raise ValueError(f"Group '{group}' not found in column '{patch_column}'")

    # Convert categorical columns to string once
    for col in df.select_dtypes(["category"]).columns:
        df[col] = df[col].astype(str)

    # list to store results for each region
    region_results = []
    outlines = []

    for region in df[region_column].unique():
        df_region = df[df[region_column] == region].copy()

        df_community = df_region[df_region[patch_column] == group].copy()

        # Check if region is large enough
        if df_community.shape[0] < min_cluster_size:
            print(f"No {group} in {region}")
            continue
        else:
            print(f"Processing {region}_{group}")
            # Create region directory
            if save_geojson or (plot and savefig):
                region_dir = os.path.join(output_dir, f"region_{region}")
                os.makedirs(region_dir, exist_ok=True)

            apply_dbscan_clustering(
                df_community,
                min_samples=min_cluster_size,
                x_col=x_column,
                y_col=y_column,
                allow_single_cluster=allow_single_cluster,
            )

            # Identify hull points
            hull = identify_hull_points(
                df_community,
                cluster_column="cluster",
                x_col=x_column,
                y_col=y_column,
                concave_hull_length_threshold=concave_hull_length_threshold,
                concavity=concavity,
            )
            # Skip if no clusters were found
            if hull.empty:
                print(f"No clusters found for region {region}.")
                continue

            if method == "border_cell_radius":
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
                    concave_hull_length_threshold=concave_hull_length_threshold,
                    concavity=concavity,
                )

                # add hull_nearest_neighbors to list
                outlines.append(hull_nearest_neighbors)

                # Convert to GeoJSON
                if save_geojson:
                    geojson_results = convert_dataframe_to_geojson(
                        df=hull,
                        output_dir=output_dir,
                        region_name=region,
                        x=x_column,
                        y=y_column,
                        region_col="unique_region",
                        patch_col="patch_id",
                        save_geojson=save_geojson,
                    )

                # Create visualizations for border_cell_radius method
                if plot:
                    try:
                        fig = create_visualization_border_cell_radius(
                            region_name=region,
                            group_name=group,
                            df_community=df_community,
                            df_full=df_region,
                            cluster_column="cluster",
                            identification_column=patch_column,
                            x_column=x_column,
                            y_column=y_column,
                            radius=radius,
                            hull_points=hull,
                            proximity_results=results,
                            hull_neighbors=hull_nearest_neighbors,
                        )

                        if savefig:
                            plot_path = os.path.join(
                                region_dir if region_dir else output_dir,
                                f"{output_fname}_patch_proximity_analysis_region_{region}.pdf",
                            )
                            # Save with higher quality and proper bounds
                            fig.savefig(
                                plot_path, bbox_inches="tight", dpi=300, format="pdf"
                            )
                            print(f"Saved visualization to: {plot_path}")
                        else:
                            plt.show()

                        plt.close(fig)  # Ensure figure is closed to free memory
                    except Exception as e:
                        print(
                            f"Warning: Failed to create visualization for region {region}: {str(e)}"
                        )

                print(f"Finished {region}_{group}")

                # append to region_results
                region_results.append(results)

            elif method == "hull_expansion":
                geojson_results = convert_dataframe_to_geojson(
                    df=hull,
                    output_dir=output_dir,
                    region_name=region,
                    x=x_column,
                    y=y_column,
                    region_col="unique_region",
                    patch_col="patch_id",
                    save_geojson=save_geojson,
                )

                # Create GeoDataFrames once per region
                patches_gdf = gpd.GeoDataFrame.from_features(
                    geojson_results[0]["geojson"]["features"]
                )
                codex_points = gpd.points_from_xy(
                    df_region[x_column], df_region[y_column]
                )
                codex_gdf = gpd.GeoDataFrame(df_region, geometry=codex_points)
                codex_gdf.set_crs(patches_gdf.crs, inplace=True)

                # Extract region numbers once
                patches_gdf["region_numeric"] = patches_gdf["region"].apply(
                    extract_region_number
                )
                codex_gdf["unique_region_numeric"] = df_region[region_column].apply(
                    extract_region_number
                )

                # Run peripheral analysis with buffer geometries
                buffer_results, buffer_geometries = analyze_peripheral_cells(
                    patches_gdf=patches_gdf,
                    codex_gdf=codex_gdf,
                    buffer_distances=distance_from_patch,
                    original_unit_scale=original_unit_scale,
                    tolerance_distance=tolerance_distance,
                )

                # Save results
                unit_name = extract_unit_name(geojson_results[0]["geojson"])
                combined_df = save_peripheral_cells(
                    results=buffer_results,
                    unit_name=unit_name,
                    region_name=region,
                    output_dir=output_dir,
                    save_csv=False,
                )

                # rename dist to distance_from_patch
                combined_df.rename(
                    columns={"dist": "distance_from_patch"}, inplace=True
                )

                # remove column unique_region_numeric and buffer_distance
                combined_df.drop(
                    columns=["unique_region_numeric", "buffer_distance"],
                    inplace=True,
                    errors="ignore",
                )

                # remove duplicates
                combined_df = combined_df.drop_duplicates(
                    subset=[x_column, y_column, "patch_id", "distance_from_patch"]
                )

                # append to region_results
                region_results.append(combined_df)
                outlines.append(hull)

                if plot:
                    try:
                        fig = create_visualization_hull_expansion(
                            region=region,
                            group=group,
                            df_community=df_community,
                            hull=hull,
                            patches_gdf=patches_gdf,
                            df_region=df_region,
                            buffer_geometries=buffer_geometries,
                            peripheral_results=buffer_results,
                            x_column=x_column,
                            y_column=y_column,
                            buffer_distances=distance_from_patch,
                            original_unit_scale=original_unit_scale,
                        )

                        if savefig:
                            plot_path = os.path.join(
                                region_dir if region_dir else output_dir,
                                f"{output_fname}_patch_proximity_analysis_region_{region}.pdf",
                            )
                            # Save with higher quality and proper bounds
                            fig.savefig(
                                plot_path, bbox_inches="tight", dpi=300, format="pdf"
                            )
                            print(f"Saved visualization to: {plot_path}")
                        else:
                            plt.show()

                        plt.close(fig)  # Ensure figure is closed to free memory
                    except Exception as e:
                        print(
                            f"Warning: Failed to create visualization for region {region}: {str(e)}"
                        )

            else:
                raise ValueError(
                    f"Unknown method: {method}. Please choose either 'border_cell_radius' or 'hull_expansion'."
                )

    # Concatenate all results into a single DataFrame
    final_results = pd.concat(region_results)

    outlines_results = pd.concat(outlines)

    # add as key to adata.uns
    if key_name is None:
        key_name = "ppa_result"
    if key_name in adata.uns:
        adata.uns[key_name] = pd.concat([adata.uns[key_name], final_results])
    else:
        adata.uns[key_name] = final_results

    # generate new column named unique_patch_ID that combines the region, group and patch ID
    final_results["unique_patch_ID"] = (
        final_results[region_column]
        + "_"
        + final_results[patch_column]
        + "_"
        + "patch_no_"
        + final_results["patch_id"].astype(str)
    )

    return final_results, outlines_results


def create_visualization_hull_expansion(
    region,
    group,
    df_community,
    hull,
    patches_gdf,
    df_region,
    buffer_geometries,
    peripheral_results,
    x_column,
    y_column,
    buffer_distances,
    original_unit_scale,
    figsize=(20, 16),
):
    """
    Create comprehensive visualization of the patch proximity analysis.

    This function creates a multi-panel figure that visualizes the complete workflow of the analysis,
    including original clustering of cells, identification of concave hull boundaries, visualization
    of expanded buffer zones, and detection of peripheral cells near the patch boundaries.

    Parameters
    ----------
    region : str
        Name of the region to be analyzed.
    group : str
        Name of the cell group (or category) under investigation.
    df_community : pandas.DataFrame
        DataFrame containing the subset of cells (community) used for clustering and further analysis.
    hull : pandas.DataFrame
        DataFrame with hull points that form the concave boundaries of clusters.
    patches_gdf : geopandas.GeoDataFrame
        GeoDataFrame containing the patch geometries for visualization.
    df_region : pandas.DataFrame
        DataFrame containing all cells in the region for contextual plotting.
    buffer_geometries : dict
        Dictionary mapping each buffer distance to a list of buffer geometry objects (expanded polygons).
    peripheral_results : dict
        Dictionary mapping each buffer distance to a DataFrame of peripheral cells detected within the buffer zones.
    x_column : str
        Column name for the x-coordinate in the DataFrames.
    y_column : str
        Column name for the y-coordinate in the DataFrames.
    buffer_distances : list of int
        List of radii (in spatial units) used for creating buffer zones.
    original_unit_scale : float
        Scale factor representing the unit conversion (e.g., 1 unit = N µm).
    figsize : tuple, optional
        Size of the generated figure (width, height in inches). Default is (20, 16).

    Returns
    -------
    matplotlib.figure.Figure
        A Matplotlib Figure object containing the multi-panel visualization.

    Notes
    -----
    - The function uses a colorblind-friendly palette (Matplotlib's tab20) for the clustering.
    - Search radii are visualized using dashed circles drawn around hull neighbor points.
    - Legends, annotations, and titles are added to convey clustering metrics and analysis steps.
    """
    # Filter to show only clustered points
    df_filtered = df_community[df_community["cluster"] != -1]

    # Create a colormap for clusters
    unique_clusters = sorted(df_filtered["cluster"].unique())
    n_clusters = len(unique_clusters)

    # Create a colorblind-friendly color palette for clusters
    cluster_colors = plt.cm.tab20(np.linspace(0, 1, max(20, n_clusters)))
    cluster_cmap = mcolors.ListedColormap(cluster_colors[:n_clusters])

    # Create buffer distance colors (using a different color scheme)
    buffer_colors = {
        dist: plt.cm.plasma(i / len(buffer_distances))
        for i, dist in enumerate(buffer_distances)
    }

    # Calculate data bounds to maintain consistent view across all plots
    x_min = df_region[x_column].min()
    x_max = df_region[x_column].max()
    y_min = df_region[y_column].min()
    y_max = df_region[y_column].max()

    # Add some padding to the bounds (5% on each side)
    x_padding = 0.05 * (x_max - x_min)
    y_padding = 0.05 * (y_max - y_min)
    x_bounds = [x_min - x_padding, x_max + x_padding]
    y_bounds = [y_min - y_padding, y_max + y_padding]

    # Create figure with adequate spacing between subplots
    fig, axes = plt.subplots(2, 2, figsize=figsize, constrained_layout=True)
    axes = axes.flatten()

    # PANEL 1: Original clustering
    ax1 = axes[0]
    scatter1 = ax1.scatter(
        df_filtered[x_column],
        df_filtered[y_column],
        c=df_filtered["cluster"],
        cmap=cluster_cmap,
        alpha=0.7,
        s=30,
        edgecolor="none",
    )

    # Add background points (all cells in region)
    ax1.scatter(
        df_region[x_column],
        df_region[y_column],
        color="lightgray",
        alpha=0.3,
        s=10,
        label="All cells",
    )

    ax1.set_title(f"Clustering of {group} cells in Region {region}", fontsize=14)
    ax1.set_xlabel(x_column, fontsize=12)
    ax1.set_ylabel(y_column, fontsize=12)
    ax1.grid(alpha=0.3)
    ax1.set_aspect("equal")  # Maintain aspect ratio
    ax1.set_xlim(x_bounds)
    ax1.set_ylim(y_bounds)

    # Add legend for clusters in a good position
    if n_clusters <= 10:  # Only show legend for reasonable number of clusters
        legend1 = ax1.legend(
            handles=[
                Patch(color=cluster_cmap(i), label=f"Cluster {cluster}")
                for i, cluster in enumerate(unique_clusters)
            ],
            title="Clusters",
            loc="best",
            frameon=True,
            bbox_to_anchor=(1.02, 1),
            fontsize=10,
        )
        ax1.add_artist(legend1)
    else:
        # Just add a colorbar
        cbar = fig.colorbar(scatter1, ax=ax1, pad=0.01, shrink=0.8)
        cbar.set_label("Cluster ID")

    # Annotate with number of clusters using relative positioning
    # Position in the top-left with padding from the axes
    ax1.annotate(
        f"Number of clusters: {n_clusters}",
        xy=(0.02, 0.98),
        xycoords="axes fraction",
        fontsize=11,
        ha="left",
        va="top",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8),
    )

    # PANEL 2: Hull points and polygons
    ax2 = axes[1]

    # Plot original points with lower alpha
    ax2.scatter(
        df_filtered[x_column],
        df_filtered[y_column],
        c=df_filtered["cluster"],
        cmap=cluster_cmap,
        alpha=0.3,
        s=20,
    )

    # Plot hull points
    if not hull.empty:
        ax2.scatter(
            hull[x_column],
            hull[y_column],
            color="red",
            s=50,
            label="Hull Points",
            edgecolor="black",
            linewidth=1,
            alpha=0.8,
        )

    # Plot the polygons from patches_gdf
    for idx, patch in patches_gdf.iterrows():
        cluster_idx = (
            unique_clusters.index(patch["cluster"])
            if patch["cluster"] in unique_clusters
            else 0
        )
        color = cluster_cmap(cluster_idx)

        # Plot the polygon boundary
        x, y = patch.geometry.exterior.xy
        ax2.plot(x, y, color=color, linewidth=2, alpha=0.9)

        # Add label in center of polygon, with smart positioning
        # Use path effects to ensure visibility against any background
        centroid = patch.geometry.centroid
        txt = ax2.text(
            centroid.x,
            centroid.y,
            f"C{patch['cluster']}",
            fontsize=10,
            ha="center",
            va="center",
            fontweight="bold",
            color="black",
        )
        txt.set_path_effects([PathEffects.withStroke(linewidth=3, foreground="white")])

    ax2.set_title("Hull Points and Resulting Polygons", fontsize=14)
    ax2.set_xlabel(x_column, fontsize=12)
    ax2.set_ylabel(y_column, fontsize=12)
    ax2.grid(alpha=0.3)
    ax2.set_aspect("equal")  # Maintain aspect ratio
    ax2.set_xlim(x_bounds)
    ax2.set_ylim(y_bounds)

    # Position the legend in a less crowded area
    ax2.legend(loc="best", bbox_to_anchor=(1.02, 1))

    # PANEL 3: Buffer regions
    ax3 = axes[2]

    # First plot all original polygons with light colors
    for idx, patch in patches_gdf.iterrows():
        cluster_idx = (
            unique_clusters.index(patch["cluster"])
            if patch["cluster"] in unique_clusters
            else 0
        )
        color = cluster_cmap(cluster_idx)

        # Plot the original polygon with a solid line
        x, y = patch.geometry.exterior.xy
        ax3.plot(x, y, color=color, linewidth=2, alpha=0.7)

        # For each buffer distance, draw expanded polygons
        for dist_idx, dist in enumerate(buffer_distances):
            buffer_color = buffer_colors[dist]

            # Find the corresponding buffer geometry
            for buffer_geom in buffer_geometries[dist]:
                if buffer_geom["patch_id"] == idx:
                    # Draw the expanded polygon with a dashed line
                    try:
                        x, y = buffer_geom["expanded"].exterior.xy
                        ax3.plot(
                            x,
                            y,
                            color=buffer_color,
                            linewidth=1.5,
                            linestyle="--",
                            alpha=0.7,
                            label=(
                                f"{dist} unit buffer"
                                if idx == list(patches_gdf.index)[0] and dist_idx == 0
                                else ""
                            ),
                        )
                    except:
                        # Handle MultiPolygons or other complex geometries
                        if isinstance(buffer_geom["expanded"], MultiPolygon):
                            for geom in buffer_geom["expanded"].geoms:
                                x, y = geom.exterior.xy
                                ax3.plot(
                                    x,
                                    y,
                                    color=buffer_color,
                                    linewidth=1.5,
                                    linestyle="--",
                                    alpha=0.7,
                                )

    ax3.set_title("Buffer Zones Around Polygons", fontsize=14)
    ax3.set_xlabel(x_column, fontsize=12)
    ax3.set_ylabel(y_column, fontsize=12)
    ax3.grid(alpha=0.3)
    ax3.set_aspect("equal")  # Maintain aspect ratio
    ax3.set_xlim(x_bounds)
    ax3.set_ylim(y_bounds)

    # Create legend for buffer distances with custom positioning
    buffer_legend_handles = [
        Patch(color=buffer_colors[dist], alpha=0.7, label=f"{dist} unit buffer")
        for dist in buffer_distances
    ]
    ax3.legend(handles=buffer_legend_handles, loc="best", bbox_to_anchor=(1.02, 1))

    # PANEL 4: Peripheral cells
    ax4 = axes[3]

    # Plot original polygons
    for idx, patch in patches_gdf.iterrows():
        cluster_idx = (
            unique_clusters.index(patch["cluster"])
            if patch["cluster"] in unique_clusters
            else 0
        )
        color = cluster_cmap(cluster_idx)

        # Plot the polygon outline
        x, y = patch.geometry.exterior.xy
        ax4.plot(x, y, color=color, linewidth=2, alpha=0.7)

    # Plot peripheral cells for each buffer distance
    for dist in buffer_distances:
        peripheral_cells = peripheral_results[dist]
        if peripheral_cells.empty:
            continue

        # Plot cells with distinct markers for each buffer distance
        markers = ["o", "s", "^", "d", "*"]  # circle, square, triangle, diamond, star
        marker = markers[buffer_distances.index(dist) % len(markers)]

        ax4.scatter(
            peripheral_cells[x_column],
            peripheral_cells[y_column],
            color=buffer_colors[dist],
            marker=marker,
            s=40,
            alpha=0.7,
            edgecolor="black",
            linewidth=0.5,
            label=f"Peripheral cells ({dist} unit buffer)",
        )

    ax4.set_title("Detected Peripheral Cells by Buffer Distance", fontsize=14)
    ax4.set_xlabel(x_column, fontsize=12)
    ax4.set_ylabel(y_column, fontsize=12)
    ax4.grid(alpha=0.3)
    ax4.set_aspect("equal")  # Maintain aspect ratio
    ax4.set_xlim(x_bounds)
    ax4.set_ylim(y_bounds)

    # Position the legend outside the plot area if it might overlap with data
    ax4.legend(loc="best", bbox_to_anchor=(1.02, 1))

    # Add overall title with enough space
    fig.suptitle(
        f"Patch Proximity Analysis for {group} in Region {region}", fontsize=18, y=0.98
    )

    # Add explanatory text at the bottom with enough padding
    # Position it well below the plots to avoid overlap
    explanation_text = (
        f"This analysis identifies clusters of {group} cells, creates boundary polygons, and detects "
        f"nearby cells within {', '.join(map(str, buffer_distances))} unit buffer zones. "
    )

    fig.text(
        0.5,
        0.01,
        explanation_text,
        ha="center",
        va="bottom",
        fontsize=12,
        bbox=dict(boxstyle="round,pad=0.5", fc="lightyellow", ec="orange", alpha=0.8),
    )

    # Make sure layout adapts to the content
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    return fig


def create_visualization_border_cell_radius(
    region_name,
    group_name,
    df_community,
    df_full,
    cluster_column="cluster",
    identification_column=None,
    x_column="x",
    y_column="y",
    radius=200,
    hull_points=None,
    proximity_results=None,
    hull_neighbors=None,
    figsize=(20, 16),
):
    """
    Create a multi-panel visualization for the border cell radius proximity analysis method.

    This function generates a figure that illustrates the workflow of the analysis by
    plotting four panels: (1) the original clustering with noise indicated, (2) the concave hull
    boundary detection, (3) the radius search visualization from all hull points, and (4) the
    proximity results showing points from different categories near the cluster boundaries.
    Each panel is carefully formatted to maintain a consistent view across the plots.

    Parameters
    ----------
    region_name : str
        Name of the region being analyzed.
    group_name : str
        Name of the cell group (or category) under investigation.
    df_community : pandas.DataFrame
        DataFrame containing the subset of cells (community) used for clustering and further analysis.
    df_full : pandas.DataFrame
        DataFrame containing all the cells in the region for context in plots.
    cluster_column : str, optional
        Column name for cluster labels in df_community. Default is "cluster".
    identification_column : str, optional
        Column name used to identify cell categories when plotting proximity results. Default is None.
    x_column : str, optional
        Column name for the x-coordinate in the DataFrames. Default is "x".
    y_column : str, optional
        Column name for the y-coordinate in the DataFrames. Default is "y".
    radius : int or list of int, optional
        The radius or list of radii (in the same units as the coordinates) used for the proximity search.
        Default is 200.
    hull_points : pandas.DataFrame, optional
        DataFrame containing the points that form the concave hull boundaries of clusters.
        If provided, these points are used for visualizing the hull boundaries. Default is None.
    proximity_results : pandas.DataFrame, optional
        DataFrame containing the results from the proximity search (cells near the hull points)
        with additional information (e.g. distance from patch). Default is None.
    hull_neighbors : pandas.DataFrame, optional
        DataFrame containing the hull neighbor points where the search circles (radii) are drawn.
        Default is None.
    figsize : tuple, optional
        Size of the generated figure in inches (width, height). Default is (20, 16).

    Returns
    -------
    matplotlib.figure.Figure
        A Matplotlib Figure object that contains the generated multi-panel visualization.

    Notes
    -----
    - The function uses the matplotlib patches (Circle) to draw search radii around each hull point.
    - A colorblind-friendly color palette (tab20 from Matplotlib) is used for representing cluster identities.
    - If a single radius is provided, it is internally converted to a list to allow uniform processing.
    - Legends and annotations are added to provide additional context on the clustering and proximity metrics.
    """
    from matplotlib.patches import Circle

    # Filter to show only clustered points
    df_filtered = df_community[df_community[cluster_column] != -1]

    # Create a colormap for clusters
    unique_clusters = sorted(df_filtered[cluster_column].unique())
    n_clusters = len(unique_clusters)

    # Create a colorblind-friendly color palette for clusters
    cluster_colors = plt.cm.tab20(np.linspace(0, 1, max(20, n_clusters)))
    cluster_cmap = mcolors.ListedColormap(cluster_colors[:n_clusters])

    # Convert radius to list if it's a single value
    if not isinstance(radius, (list, tuple, np.ndarray)):
        radius_list = [radius]
    else:
        radius_list = radius

    # Create a color mapping for different radii
    radius_colors = plt.cm.plasma(np.linspace(0, 0.8, len(radius_list)))
    radius_color_map = {r: radius_colors[i] for i, r in enumerate(radius_list)}

    # Calculate data bounds to maintain consistent view across all plots
    x_min = df_full[x_column].min()
    x_max = df_full[x_column].max()
    y_min = df_full[y_column].min()
    y_max = df_full[y_column].max()

    # Add some padding to the bounds (5% on each side)
    x_padding = 0.05 * (x_max - x_min)
    y_padding = 0.05 * (y_max - y_min)
    x_bounds = [x_min - x_padding, x_max + x_padding]
    y_bounds = [y_min - y_padding, y_max + y_padding]

    # Create figure with adequate spacing between subplots
    fig, axes = plt.subplots(2, 2, figsize=figsize, constrained_layout=True)
    axes = axes.flatten()

    # PANEL 1: Original clustering

    ax1 = axes[0]

    # Plot all points in the region with low opacity
    ax1.scatter(
        df_full[x_column],
        df_full[y_column],
        color="lightgray",
        alpha=0.3,
        s=10,
        label="All cells",
    )

    # Plot clustered points with colors by cluster
    scatter1 = ax1.scatter(
        df_filtered[x_column],
        df_filtered[y_column],
        c=df_filtered[cluster_column],
        cmap=cluster_cmap,
        alpha=0.7,
        s=30,
        edgecolor="none",
    )

    # Mark noise points with 'x'
    noise_points = df_community[df_community[cluster_column] == -1]
    if len(noise_points) > 0:
        ax1.scatter(
            noise_points[x_column],
            noise_points[y_column],
            color="gray",
            marker="x",
            s=20,
            alpha=0.5,
            label="Noise points",
        )

    ax1.set_title(
        f"HDBSCAN Clustering of {group_name} in Region {region_name}", fontsize=14
    )
    ax1.set_xlabel(x_column, fontsize=12)
    ax1.set_ylabel(y_column, fontsize=12)
    ax1.grid(alpha=0.3)
    ax1.set_aspect("equal")
    ax1.set_xlim(x_bounds)
    ax1.set_ylim(y_bounds)

    # Add legend for clusters
    if n_clusters <= 10:  # Only show legend for reasonable number of clusters
        legend1 = ax1.legend(
            handles=[
                Patch(color=cluster_cmap(i), label=f"Cluster {cluster}")
                for i, cluster in enumerate(unique_clusters)
            ],
            title="Clusters",
            loc="best",
            frameon=True,
            bbox_to_anchor=(1.02, 1),
            fontsize=10,
        )
        ax1.add_artist(legend1)
    else:
        # Add colorbar instead
        cbar = fig.colorbar(scatter1, ax=ax1, pad=0.01, shrink=0.8)
        cbar.set_label("Cluster ID")

    # Annotate with number of clusters and noise points
    ax1.annotate(
        f"Number of clusters: {n_clusters}\n" f"Noise points: {len(noise_points)}",
        xy=(0.02, 0.98),
        xycoords="axes fraction",
        fontsize=11,
        ha="left",
        va="top",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8),
    )

    # PANEL 2: Concave Hull Identification

    ax2 = axes[1]

    # Plot clustered points with reduced opacity
    ax2.scatter(
        df_filtered[x_column],
        df_filtered[y_column],
        c=df_filtered[cluster_column],
        cmap=cluster_cmap,
        alpha=0.3,
        s=20,
    )

    # Plot hull points if provided
    if hull_points is not None and len(hull_points) > 0:
        # Group by cluster if multiple clusters
        for cluster in unique_clusters:
            cluster_hull = (
                hull_points[hull_points["patch_id"] == cluster]
                if "patch_id" in hull_points.columns
                else hull_points
            )

            if len(cluster_hull) > 0:
                # Plot hull points
                ax2.scatter(
                    cluster_hull[x_column],
                    cluster_hull[y_column],
                    color="red",
                    s=50,
                    edgecolor="black",
                    linewidth=1,
                    alpha=0.8,
                    label="Hull Points" if cluster == unique_clusters[0] else "",
                )

                # Connect hull points to show the boundary
                if len(cluster_hull) > 2:
                    hull_x = cluster_hull[x_column].values
                    hull_y = cluster_hull[y_column].values

                    # If ordered by 'order' column, use that ordering
                    if "order" in cluster_hull.columns:
                        ordered_hull = cluster_hull.sort_values("order")
                        hull_x = ordered_hull[x_column].values
                        hull_y = ordered_hull[y_column].values

                    # Close the loop by adding the first point again
                    hull_x = np.append(hull_x, hull_x[0])
                    hull_y = np.append(hull_y, hull_y[0])

                    cluster_idx = unique_clusters.index(cluster)
                    color = cluster_cmap(cluster_idx)
                    ax2.plot(
                        hull_x,
                        hull_y,
                        color=color,
                        linestyle="-",
                        linewidth=2,
                        alpha=0.7,
                        label=(
                            f"Hull Boundary (C{cluster})"
                            if cluster == unique_clusters[0]
                            else ""
                        ),
                    )

    ax2.set_title("Concave Hull Boundary Detection", fontsize=14)
    ax2.set_xlabel(x_column, fontsize=12)
    ax2.set_ylabel(y_column, fontsize=12)
    ax2.grid(alpha=0.3)
    ax2.set_aspect("equal")
    ax2.set_xlim(x_bounds)
    ax2.set_ylim(y_bounds)

    # Add explanation of concave hull
    ax2.annotate(
        "Concave hull forms the\nouter boundary of each cluster",
        xy=(0.02, 0.02),
        xycoords="axes fraction",
        fontsize=11,
        ha="left",
        va="bottom",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8),
    )

    ax2.legend(loc="best", bbox_to_anchor=(1.02, 1))

    # PANEL 3: Radius Search from Hull Points

    ax3 = axes[2]

    # Plot background with even lower alpha to make circles more visible
    ax3.scatter(
        df_full[x_column], df_full[y_column], color="lightgray", alpha=0.1, s=10
    )

    # Plot clustered points with lower alpha to improve circle visibility
    ax3.scatter(
        df_filtered[x_column],
        df_filtered[y_column],
        c=df_filtered[cluster_column],
        cmap=cluster_cmap,
        alpha=0.3,
        s=20,
    )

    # Show search radius from ALL hull points with improved visibility
    if hull_neighbors is not None and len(hull_neighbors) > 0:
        # Better alpha calculation - minimum 0.2 alpha to ensure visibility
        num_circles = len(hull_neighbors) * len(radius_list)
        min_alpha = 0.2  # Minimum alpha value for visibility

        # If many circles, use a more aggressive scale-down but never below min_alpha
        if num_circles > 30:
            circle_alpha = max(min_alpha, 0.6 - (num_circles - 30) * 0.005)
        else:
            circle_alpha = max(min_alpha, 0.6 - num_circles * 0.005)

        # Create a list to hold circle objects for the legend
        legend_circles = []

        # Draw circles for all hull points with different colors for each radius
        for idx, hull_point in hull_neighbors.iterrows():
            for r_idx, r in enumerate(radius_list):
                # Get color for this radius
                circle_color = radius_color_map[r]
                circle_width = 1.0  # Slightly thicker lines

                # Draw search circle with improved visibility
                circle = Circle(
                    (hull_point[x_column], hull_point[y_column]),
                    r,
                    color=circle_color,
                    fill=False,
                    linestyle="--",
                    linewidth=circle_width,
                    alpha=circle_alpha,
                )
                ax3.add_patch(circle)

                # Create a single circle for legend (only once per radius)
                if idx == 0:
                    legend_circle = Circle(
                        (0, 0),
                        1,
                        color=circle_color,
                        fill=False,
                        linestyle="--",
                        linewidth=circle_width,
                        alpha=0.8,
                    )
                    ax3.add_patch(legend_circle)
                    legend_circle.set_visible(False)  # Hide it, just for legend
                    legend_circles.append((legend_circle, f"Search radius ({r} units)"))

        # Hull points are drawn on top of circles with high visibility
        ax3.scatter(
            hull_neighbors[x_column],
            hull_neighbors[y_column],
            color="red",
            s=40,
            edgecolor="black",
            linewidth=0.8,
            alpha=0.8,
            label="Hull Points",
        )

    # Format radii for title
    radii_str = (
        ", ".join(map(str, radius_list))
        if len(radius_list) > 1
        else str(radius_list[0])
    )
    ax3.set_title(f"Search Radii ({radii_str} units) from Hull Points", fontsize=14)
    ax3.set_xlabel(x_column, fontsize=12)
    ax3.set_ylabel(y_column, fontsize=12)
    ax3.grid(alpha=0.3)
    ax3.set_aspect("equal")
    ax3.set_xlim(x_bounds)
    ax3.set_ylim(y_bounds)

    # Add explanation of search radius
    if hull_neighbors is not None:
        radius_text = (
            ", ".join(map(str, radius_list))
            if len(radius_list) > 1
            else str(radius_list[0])
        )
        ax3.annotate(
            f"All {len(hull_neighbors)} hull points search for\n"
            f"neighboring cells within {radius_text} units",
            xy=(0.02, 0.02),
            xycoords="axes fraction",
            fontsize=11,
            ha="left",
            va="bottom",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8),
        )

    # Legend for radius circles
    if hull_neighbors is not None and len(hull_neighbors) > 0 and legend_circles:
        circles, labels = zip(*legend_circles)
        ax3.legend(circles, labels, loc="upper right")

    # PANEL 4: Proximity Results - Show all points

    ax4 = axes[3]

    # Start with background - all points in the dataset with low opacity
    ax4.scatter(
        df_full[x_column],
        df_full[y_column],
        color="lightgray",
        alpha=0.15,
        s=10,
        label="All cells",
    )

    # Plot clustered points with colors by cluster to highlight the community
    ax4.scatter(
        df_filtered[x_column],
        df_filtered[y_column],
        c=df_filtered[cluster_column],
        cmap=cluster_cmap,
        alpha=0.4,
        s=20,
    )

    # Plot hull points with increased visibility
    if hull_points is not None and len(hull_points) > 0:
        ax4.scatter(
            hull_points[x_column],
            hull_points[y_column],
            color="red",
            s=30,
            alpha=0.7,
            edgecolor="black",
            linewidth=0.5,
            label="Hull Points",
        )

    # Plot proximity results (points from other categories near hull)
    if proximity_results is not None and len(proximity_results) > 0:
        # Check if we need to visualize different radii
        if "distance_from_patch" in proximity_results.columns and len(radius_list) > 1:
            # For each radius, plot points with a different marker or color
            for r_idx, r in enumerate(radius_list):
                r_points = proximity_results[
                    proximity_results["distance_from_patch"] == r
                ]

                if len(r_points) > 0:
                    # If identification column is provided, use different colors for categories
                    if (
                        identification_column is not None
                        and identification_column in r_points.columns
                    ):
                        # Get unique categories in proximity results
                        prox_categories = r_points[identification_column].unique()

                        # Create color mapping for categories
                        category_colors = plt.cm.Set2(
                            np.linspace(0, 1, len(prox_categories))
                        )

                        # Markers for different radii (cycle through a few options)
                        markers = [
                            "o",
                            "s",
                            "^",
                            "d",
                            "*",
                        ]  # circle, square, triangle, diamond, star
                        marker = markers[r_idx % len(markers)]

                        # Plot each category with different color but same marker for this radius
                        for i, category in enumerate(prox_categories):
                            category_points = r_points[
                                r_points[identification_column] == category
                            ]
                            ax4.scatter(
                                category_points[x_column],
                                category_points[y_column],
                                color=category_colors[i],
                                marker=marker,
                                s=80,
                                alpha=0.8,
                                edgecolor="black",
                                linewidth=0.5,
                                label=f"{category} ({r} units)",
                            )
                    else:
                        # Use same color scheme as for radius circles with different markers
                        color = radius_color_map[r]
                        markers = ["o", "s", "^", "d", "*"]
                        marker = markers[r_idx % len(markers)]

                        ax4.scatter(
                            r_points[x_column],
                            r_points[y_column],
                            color=color,
                            marker=marker,
                            s=80,
                            alpha=0.8,
                            edgecolor="black",
                            linewidth=0.5,
                            label=f"Proximity Points ({r} units)",
                        )
        else:
            # Original behavior for single radius
            if (
                identification_column is not None
                and identification_column in proximity_results.columns
            ):
                # Get unique categories in proximity results
                prox_categories = proximity_results[identification_column].unique()

                # Create color mapping for categories
                category_colors = plt.cm.Set2(np.linspace(0, 1, len(prox_categories)))

                # Plot each category with different color
                for i, category in enumerate(prox_categories):
                    category_points = proximity_results[
                        proximity_results[identification_column] == category
                    ]
                    ax4.scatter(
                        category_points[x_column],
                        category_points[y_column],
                        color=category_colors[i],
                        marker="o",
                        s=80,
                        alpha=0.8,
                        edgecolor="black",
                        linewidth=0.5,
                        label=f"{category}",
                    )
            else:
                # Plot all proximity points with same color
                ax4.scatter(
                    proximity_results[x_column],
                    proximity_results[y_column],
                    color="gold",
                    marker="o",
                    s=80,
                    alpha=0.8,
                    edgecolor="black",
                    linewidth=0.5,
                    label="Proximity Points",
                )

    ax4.set_title("Detected Points in Proximity to Clusters", fontsize=14)
    ax4.set_xlabel(x_column, fontsize=12)
    ax4.set_ylabel(y_column, fontsize=12)
    ax4.grid(alpha=0.3)
    ax4.set_aspect("equal")
    ax4.set_xlim(x_bounds)
    ax4.set_ylim(y_bounds)

    # Add explanation text
    prox_count = len(proximity_results) if proximity_results is not None else 0
    ax4.annotate(
        f"Found {prox_count} cells from different\ncategories near cluster boundaries",
        xy=(0.02, 0.02),
        xycoords="axes fraction",
        fontsize=11,
        ha="left",
        va="bottom",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8),
    )

    # Position the legend
    ax4.legend(loc="best", bbox_to_anchor=(1.02, 1))

    # Add overall title and explanation

    fig.suptitle(
        f"Border Cell Radius Proximity Analysis for {group_name} in Region {region_name}",
        fontsize=18,
        y=0.98,
    )

    # Add explanatory text at the bottom with radius list
    radii_text = (
        ", ".join(map(str, radius_list))
        if len(radius_list) > 1
        else str(radius_list[0])
    )
    explanation_text = (
        "This analysis: (1) Clusters cells using HDBSCAN algorithm, (2) Identifies the concave hull boundary of each cluster, "
        f"(3) For each hull point, searches for cells within {radii_text} units, and "
        "(4) Identifies cells from different categories that are in proximity to cluster boundaries."
    )

    fig.text(
        0.5,
        0.01,
        explanation_text,
        ha="center",
        va="bottom",
        fontsize=12,
        bbox=dict(boxstyle="round,pad=0.5", fc="lightyellow", ec="orange", alpha=0.8),
    )

    # Make sure layout adapts to the content
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    return fig
