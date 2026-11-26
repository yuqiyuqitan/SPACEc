# load required packages
from __future__ import annotations

# Filter out specific FutureWarnings from anndata and tm
import warnings

warnings.filterwarnings("ignore", category=FutureWarning, module="anndata.utils")
import logging
import math

# TissUUmaps sets the *root* logger to INFO on import, which turns on INFO output
# from every library. Remember the caller's level and restore it after the imports
# (instead of disabling logging process-wide, which also silenced user code).
_root_log_level = logging.getLogger().level

import argparse
import json
import os
import pathlib
import pickle
import re
import subprocess
import sys
import tempfile
import time
from builtins import range
from datetime import datetime, timezone
from multiprocessing import Pool
from typing import TYPE_CHECKING

import anndata as ad
import geopandas as gpd
import matplotlib.patheffects as PathEffects
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import requests
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
from skimage.measure import regionprops_table
from skimage.segmentation import find_boundaries
from sklearn.calibration import CalibratedClassifierCV
from sklearn.cluster import HDBSCAN, MiniBatchKMeans
from sklearn.cross_decomposition import CCA
from sklearn.metrics import (
    calinski_harabasz_score,
    classification_report,
    f1_score,
    pairwise_distances,
    silhouette_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.svm import SVC, LinearSVC
from tqdm import tqdm

logging.getLogger().setLevel(_root_log_level)

# libvips no longer needs to be provisioned here: SPACEc depends on
# `pyvips[binary]`, whose pyvips-binary wheels bundle libvips on every platform
# including Windows. This used to download and unpack vips-dev-w64 at import
# time.


if TYPE_CHECKING:
    from anndata import AnnData

from ..helperfunctions._general import *
from ..plotting._general import _select_regions_for_cluster, catplot, patch_QC
from ..plotting._style import _apply_spines, _apply_style
from ._clustering_tree import _ensure_roots, _record_annotation, _record_split

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


def tl_calculate_neigh_combs(
    w,
    l,
    n_num,
    threshold=0.85,
    per_keep_thres=0.85,
    thresh_freq=0.001,
    figsize=(9, 3.5),
):
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
        Diagnostic only: sets the line in the plot, nothing is filtered.
    thresh_freq : float, optional
        Minimum combination frequency kept on the CN map, by default 0.001. Used for plot
        shading only; should match ``tl_build_graph_CN_comb_map``.
    figsize : tuple, optional
        Width, height (inches) of the coverage plot, by default (9, 3.5). The legend is placed
        below the axes, so the plot stays readable down to about 3 x 2.8 inches.

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

    # How many top combinations stay below per_keep_thres (diagnostic only, nothing is filtered)
    n = len(simp_freqs)
    test_len = min(int((simp_sums < per_keep_thres).sum()), n - 1)
    per_values_above = simp_freqs.iloc[
        test_len
    ]  # share of the combination crossing the line
    on_map = simp_freqs.values >= thresh_freq
    n_on_map = int(on_map.sum())

    w["combination"] = [tuple(l[a] for a in s) for s in simps]
    w["combination_num"] = [tuple(a for a in s) for s in simps]

    from matplotlib.lines import Line2D

    # rank 1 = most frequent combination. Thousands of combinations on a linear axis shrink the
    # bars to sub-pixel width, so ranks are drawn on a log axis as step areas (always visible)
    ranks = np.arange(1, n + 1)
    freqs = simp_freqs.values
    fig, ax = plt.subplots(figsize=figsize)
    cut = min(
        test_len + 1, n
    )  # blue part includes the combination that crosses the threshold line
    ax.fill_between(ranks[:cut], freqs[:cut], step="mid", color="#378ADD", linewidth=0)
    if cut < n:
        ax.fill_between(
            ranks[cut - 1 :], freqs[cut - 1 :], step="mid", color="#D3D1C7", linewidth=0
        )
    if (~on_map).any():
        ax.fill_between(
            ranks[~on_map],
            freqs[~on_map],
            step="mid",
            facecolor="none",
            edgecolor="#888780",
            hatch="////",
            linewidth=0,
        )
    ax.set_xscale("log")
    ax.set_xlabel("combination rank (log scale)")
    ax.set_ylabel("fraction of windows")
    ax.set_xlim(0.8, n * 1.15)
    ax.set_ylim(bottom=0)
    ax.xaxis.set_major_formatter(plt.ScalarFormatter())

    ax2 = ax.twinx()
    ax2.plot(ranks, simp_sums.values, color="black", linewidth=1.5)
    ax2.axhline(per_keep_thres, color="#E24B4A", linestyle="--", linewidth=1.2)
    ax.axvline(test_len + 0.5, color="#E24B4A", linestyle=":", linewidth=1)
    ax2.set_ylim(0, 1.02)
    ax2.set_ylabel("cumulative fraction")

    ax.set_title(
        f"{test_len} combinations cover <{per_keep_thres:.0%} of windows "
        f"(next adds {per_values_above:.1%})\n"
        f"{n_on_map} of {n} go to the CN map (freq \u2265 {thresh_freq})",
        fontsize=plt.rcParams["axes.titlesize"],
        pad=12,
    )
    ax.legend(
        handles=[
            Patch(color="#378ADD", label="counted below per_keep_thres"),
            Patch(color="#D3D1C7", label="rest"),
            Patch(
                facecolor="none",
                edgecolor="#888780",
                hatch="////",
                label=f"dropped from CN map (freq < {thresh_freq})",
            ),
            Line2D([], [], color="black", label="cumulative fraction"),
            Line2D([], [], color="#E24B4A", linestyle="--", label="per_keep_thres"),
        ],
        loc="upper center",
        bbox_to_anchor=(
            0.5,
            -0.28,
        ),  # below the axes: never covers bars, ticks or labels
        ncol=2,
        fontsize=plt.rcParams["legend.fontsize"],
        frameon=False,
    )
    fig.tight_layout()
    plt.show()

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


def _rapids_install_hint():
    """Return the pip command(s) that install rapids-singlecell for the local CUDA.

    rapids-singlecell publishes one distribution per CUDA major version
    (``rapids-singlecell-cu12`` / ``-cu13``), so the right command depends on
    the CUDA build torch was compiled against. When that can't be determined
    (torch missing, CPU-only build, unknown major) both variants are returned
    so the user can pick.
    """
    template = (
        "    pip install 'rapids-singlecell-cu{major}[rapids]'"
        " --extra-index-url=https://pypi.nvidia.com"
    )

    cuda_version = None
    try:
        import torch

        cuda_version = torch.version.cuda
    except Exception:  # torch missing or broken -- fall through to both hints
        pass

    major = str(cuda_version).split(".")[0] if cuda_version else None
    if major in ("12", "13"):
        return template.format(major=major)

    return "\n".join(
        [
            "  CUDA 13:",
            template.format(major="13"),
            "  CUDA 12:",
            template.format(major="12"),
        ]
    )


def _resolve_clustering_method(method):
    """Resolve the requested clustering algorithm name, falling back from
    ``leiden_gpu`` to ``leiden`` when rapids_singlecell isn't available.

    Shared by :func:`clustering` and :func:`subclustering` so the
    availability check and fallback message only need to live in one place.
    """
    if method == "leiden_gpu":
        try:
            import cudf  # noqa: F401
            import cuml  # noqa: F401
            import cupy  # noqa: F401
            import rapids_singlecell  # noqa: F401
        except ImportError:
            print("Please install rapids_singlecell to use leiden_gpu!")
            print(_rapids_install_hint())
            print("THIS FUNCTION DOES NOT WORK ON MacOS OR WINDOWS")
            print("using leiden instead of leiden_gpu")
            method = "leiden"
    return method


def _flavor_kwargs(method, flavor):
    """``sc.tl.leiden``/``sc.tl.louvain`` kwargs for the requested ``flavor``.

    scanpy's igraph backend needs ``n_iterations=2, directed=False`` on
    leiden to match legacy behavior (see its own FutureWarning); louvain
    needs no extra kwargs. "legacy" restores each algorithm's pre-igraph
    default flavor name (``leidenalg`` for leiden, ``vtraag`` for louvain).
    Shared by :func:`clustering` and :func:`subclustering`.
    """
    if flavor == "igraph":
        kwargs = {"flavor": "igraph"}
        if method == "leiden":
            kwargs["n_iterations"] = 2
            kwargs["directed"] = False
        return kwargs
    if method == "louvain":
        # scanpy's own hint (`pip install louvain`) resolves to the sdist-only
        # louvain 0.7.1 next to igraph>=1.0; the `legacy` extra pins a working set.
        try:
            import louvain  # noqa: F401
        except Exception as e:
            raise ImportError(
                'flavor="legacy" louvain clustering needs the deprecated `louvain` '
                "package. Install it with: pip install 'spacec[legacy]' "
                '(or use flavor="igraph", or clustering="leiden").'
            ) from e
    return {"flavor": "leidenalg" if method == "leiden" else "vtraag"}


def _append_clustering_report(adata, entry):
    """Append one step to ``adata.uns["clustering_report"]``.

    Stored as a JSON string (rather than a raw list of dicts) because
    AnnData's h5ad writer cannot serialize a list of dicts, or a DataFrame
    with list-valued cells, directly -- a JSON string round-trips cleanly.
    Never raises: a report-logging failure must not break the clustering
    call it's attached to.
    """
    try:
        raw = adata.uns.get("clustering_report")
        if isinstance(raw, str) and raw:
            steps = json.loads(raw)
        elif raw:
            steps = list(raw)
        else:
            steps = []
        entry = dict(entry)
        entry["step"] = len(steps) + 1
        entry["timestamp"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        steps.append(entry)
        adata.uns["clustering_report"] = json.dumps(steps, default=str)
    except Exception as e:
        print(f"Warning: failed to record clustering report step: {e}")


def _format_clustering_report_step(entry):
    """Render one clustering_report entry as a human-readable sentence."""
    step = entry.get("step")
    ts = entry.get("timestamp", "")
    action = entry.get("action")
    method = entry.get("method")
    resolution = entry.get("resolution")
    seed = entry.get("seed")
    key_added = entry.get("key_added")
    notes = entry.get("notes")

    if action == "clustering":
        markers = entry.get("marker_list")
        marker_txt = f"using {len(markers)} markers" if markers else "using all markers"
        head = (
            f"Step {step} ({ts}) -- clustering: ran {method} "
            f"(resolution={resolution}, n_neighbors={entry.get('n_neighbors')}, seed={seed}) "
            f"on {entry.get('n_cells')} cells {marker_txt} -> {entry.get('n_clusters')} "
            f"clusters stored in adata.obs['{key_added}']."
        )
    elif action == "subclustering":
        per_cluster = entry.get("per_cluster_new_labels") or {}
        breakdown = ", ".join(f"{k}→{len(v)}" for k, v in per_cluster.items())
        graph_txt = (
            "reusing existing neighbor graph"
            if entry.get("graph_source") == "reused_global"
            else "on a freshly computed local neighbor graph"
        )
        head = (
            f"Step {step} ({ts}) -- subclustering: split cluster(s) "
            f"{entry.get('clusters_split')} of adata.obs['{entry.get('cluster_key')}'] "
            f"using {method} (resolution={resolution}, seed={seed}, {graph_txt}) -> "
            f"{breakdown} new subclusters, stored in adata.obs['{key_added}']."
        )
    elif action == "combine_clusters":
        head = (
            f"Step {step} ({ts}) -- combine_clusters: merged clusters "
            f"{entry.get('clusters_merged')} of adata.obs['{entry.get('cluster_key')}'] "
            f"into '{entry.get('new_label')}' ({entry.get('n_cells')} cells)."
        )
    elif action == "annotation":
        head = (
            f"Step {step} ({ts}) -- annotation: mapped "
            f"{entry.get('n_labels_mapped')} of {entry.get('n_labels_total')} labels "
            f"of adata.obs['{entry.get('cluster_key')}'] to {entry.get('n_targets')} "
            f"cell types in adata.obs['{entry.get('key_added')}']."
        )
    else:
        head = f"Step {step} ({ts}) -- {action}."

    if notes:
        head += f" Reason: {notes}"
    return head


def clustering_report(adata, as_dataframe=False, print_report=True):
    """
    Reconstruct and explain the clustering strategy recorded on ``adata``.

    Every call to :func:`clustering`, :func:`subclustering`, and
    :func:`combine_clusters` appends a step to
    ``adata.uns["clustering_report"]`` (stored as JSON, so it survives
    ``.h5ad`` save/reload). This function turns that history back into a
    human-readable narrative, or a :class:`pandas.DataFrame` for
    programmatic use -- so the clustering strategy can be explained or
    audited later, even after reopening a saved file.

    Parameters
    ----------
    adata : AnnData
    as_dataframe : bool, optional
        If True, return a `pandas.DataFrame` of the recorded steps instead
        of the narrative string. Defaults to False.
    print_report : bool, optional
        If True, print the narrative. Defaults to True.

    Returns
    -------
    str, pandas.DataFrame, or None
        The narrative string, or a DataFrame if ``as_dataframe=True``. If no
        steps have been recorded, returns ``None`` (or an empty DataFrame).
    """
    raw = adata.uns.get("clustering_report")
    steps = json.loads(raw) if raw else []

    if not steps:
        if print_report:
            print("No clustering steps recorded yet.")
        return pd.DataFrame() if as_dataframe else None

    if as_dataframe:
        return pd.DataFrame(steps)

    lines = [f"Clustering strategy ({len(steps)} step(s) recorded):", ""]
    lines.extend(_format_clustering_report_step(entry) for entry in steps)
    narrative = "\n".join(lines)
    if print_report:
        print(narrative)
    return narrative


def clustering(
    adata,
    clustering="leiden",
    marker_list=None,
    resolution=1,
    n_neighbors=10,
    reclustering=False,
    compute_umap=True,
    key_added=None,
    key_filter=None,
    subset_cluster=None,
    seed=42,
    fs_xdim=10,
    fs_ydim=10,
    fs_rlen=10,  # FlowSOM parameters
    flavor="igraph",
    *,
    notes=None,
    _record_report=True,
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
    compute_umap : bool, optional
        If set to False, skips the UMAP embedding step (only the neighbor graph and cluster labels are computed). Ignored when reclustering=True, which already skips both. Defaults to True.
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
    flavor : str, optional
        Backend for leiden/louvain: ``"igraph"`` (default, scanpy's current
        recommended flavor) or ``"legacy"`` (leiden's old ``leidenalg``
        backend / louvain's old ``vtraag`` backend, for reproducing older
        results). Ignored for ``leiden_gpu`` and ``flowSOM``. Note: with
        ``clustering="louvain"``, scanpy's igraph backend ignores
        ``resolution`` entirely (a scanpy limitation, not tunable) -- use
        ``flavor="legacy"`` (requires ``pip install 'spacec[legacy]'``)
        or switch to ``clustering="leiden"`` if you need resolution control
        with louvain.
    notes : str, optional
        Free-text note describing why this clustering step was run --
        recorded in ``adata.uns["clustering_report"]`` alongside the actual
        parameters used. See :func:`clustering_report`. Defaults to None.
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
    if flavor not in ["igraph", "legacy"]:
        print('Invalid flavor. Please select from "igraph" or "legacy"!')
        sys.exit()

    clustering = _resolve_clustering_method(clustering)
    if clustering == "leiden_gpu":
        import rapids_singlecell as rsc

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
                    **{**_flavor_kwargs("leiden", flavor), **cluster_kwargs},
                )
            else:
                if clustering == "louvain":
                    print("Louvain clustering")
                    sc.tl.louvain(
                        adata,
                        resolution=resolution,
                        key_added=key_added,
                        random_state=seed,
                        **{**_flavor_kwargs("louvain", flavor), **cluster_kwargs},
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
            print("Computing neighbors on GPU")
            rsc.pp.neighbors(adata, n_neighbors=n_neighbors, random_state=seed)
            if compute_umap:
                print("Computing UMAP on GPU")
                rsc.tl.umap(adata, random_state=seed)
            print("Clustering on GPU")
            # Perform leiden clustering - improved version of louvain clustering
            rsc.tl.leiden(
                adata, resolution=resolution, key_added=key_added, random_state=seed
            )
            anndata_to_CPU(adata)  # moves `.X` to the CPU

        else:
            # FlowSOM clusters directly on the marker matrix and doesn't need
            # the neighbor graph unless UMAP (which depends on it) is requested.
            needs_neighbors = clustering in ("leiden", "louvain") or compute_umap
            if needs_neighbors:
                print("- neighbors")
                sc.pp.neighbors(adata, n_neighbors=n_neighbors, random_state=seed)
            if compute_umap:
                print("- UMAP")
                sc.tl.umap(adata, random_state=seed)
            print("Clustering")
            # Perform leiden clustering - improved version of louvain clustering
            if clustering == "leiden":
                print("Leiden clustering")
                sc.tl.leiden(
                    adata,
                    resolution=resolution,
                    key_added=key_added,
                    random_state=seed,
                    **{**_flavor_kwargs("leiden", flavor), **cluster_kwargs},
                )
            else:
                if clustering == "louvain":
                    print("Louvain clustering")
                    sc.tl.louvain(
                        adata,
                        resolution=resolution,
                        key_added=key_added,
                        random_state=seed,
                        **{**_flavor_kwargs("louvain", flavor), **cluster_kwargs},
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
            result = adata
        else:
            adata_tmp.obs[key_added] = adata.obs[key_added].values
            # append other data
            adata_tmp.obsm = adata.obsm
            adata_tmp.obsp = adata.obsp
            adata_tmp.uns = adata.uns
            result = adata_tmp
    else:
        # Cells outside subset_cluster keep their key_filter value; cells that
        # were (re)clustered get their new key_added label. Joining on the
        # shared obs index (rather than merging on every obs column) keeps
        # adata_tmp's original obs_names intact and avoids an expensive
        # multi-column hash join.
        adata_tmp.obs[key_added] = adata_tmp.obs[key_filter].astype(str)
        adata_tmp.obs.loc[adata.obs.index, key_added] = (
            adata.obs[key_added].astype(str).values
        )

        adata_tmp.obs[key_filter] = adata_tmp.obs[key_filter].astype("category")
        adata_tmp.obs[key_added] = adata_tmp.obs[key_added].astype("category")
        result = adata_tmp

    if _record_report:
        try:
            _append_clustering_report(
                result,
                {
                    "action": "clustering",
                    "method": clustering,
                    "flavor": flavor,
                    "marker_list": marker_list,
                    "resolution": resolution,
                    "n_neighbors": n_neighbors,
                    "seed": seed,
                    "key_added": key_added,
                    "n_cells": int(result.n_obs),
                    "n_clusters": int(result.obs[key_added].nunique()),
                    "notes": notes,
                },
            )
            _ensure_roots(result, key_added)
        except Exception as e:
            print(f"Warning: failed to record clustering report step: {e}")

    return result


def _numeric_suffix_sort_key(label, prefix):
    """Sort ``f"{prefix}{i}"``-style subcluster labels numerically on ``i``
    when possible, else lexicographically -- keeps printed/report ordering
    stable regardless of how many subclusters a round produced."""
    suffix = label[len(prefix) :] if label.startswith(prefix) else label
    try:
        return (0, float(suffix))
    except ValueError:
        return (1, suffix)


def subclustering(
    adata,
    cluster_key,
    clusters,
    resolution=0.3,
    key_added=None,
    method="leiden",
    seed=42,
    recompute_graph=False,
    n_neighbors=10,
    marker_list=None,
    flavor="igraph",
    *,
    notes=None,
    _record_report=True,
    **cluster_kwargs,
):
    """
    Split one or more existing clusters into finer subclusters.

    Replaces a hand-written chain of ``sc.tl.leiden(restrict_to=...)`` calls
    (one per cluster) with a single call. Clusters are split **sequentially,
    one at a time**: the first cluster is split from ``cluster_key``, and
    every later cluster is split from ``key_added`` (which already carries
    the previous round's new labels) -- this is what makes chaining several
    splits into the same ``key_added`` column work correctly instead of
    silently overwriting earlier splits or pooling all requested clusters
    into one joint split.

    Parameters
    ----------
    adata : AnnData
    cluster_key : str
        Column in ``adata.obs`` holding the cluster labels to split.
    clusters : str, list of str, or dict
        Cluster label(s) to split. A ``dict`` maps each cluster label to a
        per-cluster resolution override, e.g. ``{"0": 0.3, "3": 0.5}``.
    resolution : float, optional
        Resolution used for every cluster in ``clusters`` that isn't given
        its own override via a ``clusters`` dict. Defaults to 0.3.
    key_added : str, optional
        Column added to ``adata.obs`` with the new labels. Defaults to
        ``f"{cluster_key}_subcluster"``.
    method : str, optional
        Clustering algorithm: ``"leiden"``, ``"louvain"``, or
        ``"leiden_gpu"``. ``"flowSOM"`` is not supported (it has no
        ``restrict_to`` equivalent). Defaults to ``"leiden"``.
    seed : int, optional
        Random seed, passed through as ``random_state``. Defaults to 42.
    recompute_graph : bool, optional
        If False (default), reuse the existing global neighbor graph for
        each split -- the same thing ``restrict_to`` does today, and what
        reproduces the current tutorial output. If True, subset to each
        cluster's cells and build a fresh local neighbor graph
        (``n_neighbors``) before clustering it.
    n_neighbors : int, optional
        Neighbors used when ``recompute_graph=True``. Ignored otherwise.
        Defaults to 10.
    marker_list : list, optional
        Markers to restrict to when ``recompute_graph=True``. Ignored
        otherwise. Defaults to None (use all markers).
    flavor : str, optional
        Backend for leiden/louvain: ``"igraph"`` (default, scanpy's current
        recommended flavor) or ``"legacy"`` (leiden's old ``leidenalg``
        backend / louvain's old ``vtraag`` backend, for reproducing older
        results). Ignored for ``method="leiden_gpu"``. Note: with
        ``method="louvain"``, scanpy's igraph backend ignores ``resolution``
        entirely (a scanpy limitation, not tunable) -- use
        ``flavor="legacy"`` (requires ``pip install 'spacec[legacy]'``)
        or ``method="leiden"`` if you need resolution control with louvain.
    notes : str, optional
        Free-text note describing why this split was run -- recorded in
        ``adata.uns["clustering_report"]``. See :func:`clustering_report`.
        Defaults to None.
    **cluster_kwargs : dict
        Additional keyword arguments forwarded to the underlying
        ``sc.tl.leiden`` / ``sc.tl.louvain`` call.

    Returns
    -------
    AnnData
        ``adata`` with the new subcluster labels added to
        ``adata.obs[key_added]``.

    Raises
    ------
    ValueError
        If ``cluster_key`` is not in ``adata.obs``, if any requested cluster
        is not a value of ``adata.obs[cluster_key]``, or if
        ``method="flowSOM"``.

    Notes
    -----
    The wrapper standardises on ``random_state=seed`` (matching
    :func:`clustering`), whereas some existing notebooks pass ``seed=`` to
    ``sc.tl.leiden`` directly, which is forwarded to leidenalg instead --
    subcluster label *assignment* may therefore differ slightly from
    previously committed notebook output even though the method and
    resolution are identical.
    """
    if cluster_key not in adata.obs:
        raise ValueError(f"Column '{cluster_key}' not found in adata.obs")
    if method == "flowSOM":
        raise ValueError(
            "method='flowSOM' is not supported by subclustering() -- flowSOM "
            "has no restrict_to equivalent. Use 'leiden', 'louvain', or "
            "'leiden_gpu' instead."
        )
    if flavor not in ["igraph", "legacy"]:
        raise ValueError('flavor must be "igraph" or "legacy"')

    if isinstance(clusters, dict):
        cluster_list = list(clusters.keys())
        res_for = {c: clusters[c] for c in cluster_list}
    elif isinstance(clusters, str):
        cluster_list = [clusters]
        res_for = {clusters: resolution}
    else:
        cluster_list = list(clusters)
        res_for = {c: resolution for c in cluster_list}

    available_labels = adata.obs[cluster_key].astype(str).unique().tolist()
    missing_clusters = [c for c in cluster_list if c not in available_labels]
    if missing_clusters:
        raise ValueError(
            f"Cluster(s) {missing_clusters} not found in adata.obs['{cluster_key}']. "
            f"Available clusters: {sorted(available_labels)}"
        )

    method = _resolve_clustering_method(method)
    if key_added is None:
        key_added = f"{cluster_key}_subcluster"

    per_cluster_new_labels = {}

    if not recompute_graph:
        # Reuse the existing global neighbor graph -- exactly what
        # restrict_to does today. First cluster restricts on cluster_key;
        # every later cluster restricts on key_added, which now carries the
        # previous round's new labels (and every unsplit label carried
        # forward unchanged) -- this sequential chaining is the whole point
        # of the wrapper.
        for i, c in enumerate(cluster_list):
            source_key = cluster_key if i == 0 else key_added
            res = res_for[c]
            n_before = int((adata.obs[source_key].astype(str) == c).sum())

            if method == "leiden_gpu":
                import rapids_singlecell as rsc

                anndata_to_GPU(adata)
                rsc.tl.leiden(
                    adata,
                    restrict_to=(source_key, [c]),
                    resolution=res,
                    key_added=key_added,
                    random_state=seed,
                    **cluster_kwargs,
                )
                anndata_to_CPU(adata)
            elif method == "louvain":
                sc.tl.louvain(
                    adata,
                    restrict_to=(source_key, [c]),
                    resolution=res,
                    key_added=key_added,
                    random_state=seed,
                    **{**_flavor_kwargs("louvain", flavor), **cluster_kwargs},
                )
            else:
                sc.tl.leiden(
                    adata,
                    restrict_to=(source_key, [c]),
                    resolution=res,
                    key_added=key_added,
                    random_state=seed,
                    **{**_flavor_kwargs("leiden", flavor), **cluster_kwargs},
                )

            prefix = f"{c},"
            categories = list(adata.obs[key_added].astype("category").cat.categories)
            new_labels = [lbl for lbl in categories if lbl.startswith(prefix)]
            per_cluster_new_labels[c] = new_labels
            print(f"- '{c}': {n_before} cells -> {len(new_labels)} subclusters")
    else:
        # Local graph per cluster: subset to the cluster's cells (optionally
        # restricted to marker_list), recompute neighbors, cluster, then
        # write the new "{c},{j}" labels back by index -- same
        # astype(str) -> .loc[] -> astype("category") reintegration pattern
        # used by clustering()'s key_filter path.
        if key_added not in adata.obs:
            adata.obs[key_added] = adata.obs[cluster_key].astype(str).values
        adata.obs[key_added] = adata.obs[key_added].astype("category")

        for c in cluster_list:
            res = res_for[c]
            mask = (adata.obs[key_added].astype(str) == c).values
            n_before = int(mask.sum())

            sub = adata[mask, marker_list].copy() if marker_list else adata[mask].copy()

            if method == "leiden_gpu":
                import rapids_singlecell as rsc

                anndata_to_GPU(sub)
                rsc.pp.neighbors(sub, n_neighbors=n_neighbors, random_state=seed)
                rsc.tl.leiden(
                    sub,
                    resolution=res,
                    key_added="_spacec_local_sub",
                    random_state=seed,
                    **cluster_kwargs,
                )
                anndata_to_CPU(sub)
            else:
                sc.pp.neighbors(sub, n_neighbors=n_neighbors, random_state=seed)
                if method == "louvain":
                    sc.tl.louvain(
                        sub,
                        resolution=res,
                        key_added="_spacec_local_sub",
                        random_state=seed,
                        **{**_flavor_kwargs("louvain", flavor), **cluster_kwargs},
                    )
                else:
                    sc.tl.leiden(
                        sub,
                        resolution=res,
                        key_added="_spacec_local_sub",
                        random_state=seed,
                        **{**_flavor_kwargs("leiden", flavor), **cluster_kwargs},
                    )

            local_labels = sub.obs["_spacec_local_sub"].astype(str)
            order = sorted(
                local_labels.unique(),
                key=lambda v: _numeric_suffix_sort_key(v, ""),
            )
            prefix = f"{c},"
            remap = {orig: f"{prefix}{j}" for j, orig in enumerate(order)}
            new_vals = local_labels.map(remap)

            adata.obs[key_added] = adata.obs[key_added].astype(str)
            adata.obs.loc[sub.obs.index, key_added] = new_vals.values
            adata.obs[key_added] = adata.obs[key_added].astype("category")

            new_labels = [remap[o] for o in order]
            per_cluster_new_labels[c] = new_labels
            print(f"- '{c}': {n_before} cells -> {len(new_labels)} subclusters")

    if _record_report:
        entry = {
            "clusters_split": cluster_list,
            "cluster_key": cluster_key,
            "per_cluster_new_labels": per_cluster_new_labels,
            "graph_source": "local" if recompute_graph else "reused_global",
            "method": method,
            "flavor": flavor,
            "resolution": resolution,
            "seed": seed,
            "key_added": key_added,
            "notes": notes,
        }
        if isinstance(clusters, dict):
            entry["per_cluster_resolution"] = res_for
        try:
            _append_clustering_report(adata, {"action": "subclustering", **entry})
        except Exception as e:
            print(f"Warning: failed to record clustering report step: {e}")
        _record_split(
            adata, cluster_key, key_added, per_cluster_new_labels, params=entry
        )

    return adata


def annotate_clusters(
    adata,
    cluster_key,
    mapping,
    key_added="cell_type",
    missing="keep",
    *,
    notes=None,
    _record_report=True,
):
    """
    Map cluster labels to cell type annotations.

    A safer drop-in for ``adata.obs[key_added] = adata.obs[cluster_key].map(mapping)``:
    by default, cluster labels absent from ``mapping`` keep their original
    label instead of silently becoming ``NaN``, so callers no longer need to
    re-list every already-annotated label as an identity mapping just to
    avoid losing cells.

    Parameters
    ----------
    adata : AnnData
    cluster_key : str
        Column in ``adata.obs`` holding the cluster labels to annotate.
    mapping : dict
        ``{cluster_label: cell_type}``. Labels not present as keys are
        handled according to ``missing``. Keys are matched against
        ``adata.obs[cluster_key]`` as strings (the column is internally
        ``.astype(str)``'d before matching), so int-typed cluster ids (e.g.
        ``0``, ``1``, ...) work equally well as ``int`` or ``str`` keys.
    key_added : str, optional
        Column added to ``adata.obs`` with the resolved annotations.
        Defaults to ``"cell_type"``.
    missing : {"keep", "nan", "raise"}, optional
        How to handle a label present in ``adata.obs[cluster_key]`` but
        absent from ``mapping``:

        - ``"keep"`` (default): keep the original label unchanged.
        - ``"nan"``: set it to ``NaN`` (plain ``.map()`` behaviour).
        - ``"raise"``: raise :class:`ValueError` listing the unmapped labels.
    notes : str, optional
        Free-text note describing why this annotation was applied --
        recorded in ``adata.uns["clustering_report"]``. See
        :func:`clustering_report`. Defaults to None.

    Returns
    -------
    AnnData
        ``adata`` with the resolved annotation added to
        ``adata.obs[key_added]`` (dtype ``category``).

    Raises
    ------
    ValueError
        If ``cluster_key`` is not in ``adata.obs``, if ``missing`` is not one
        of ``"keep"``/``"nan"``/``"raise"``, or if ``missing="raise"`` and
        some label in ``adata.obs[cluster_key]`` has no entry in ``mapping``.
    """
    if cluster_key not in adata.obs:
        raise ValueError(f"Column '{cluster_key}' not found in adata.obs")
    if missing not in ("keep", "nan", "raise"):
        raise ValueError(
            f"Invalid missing='{missing}'. Choose from 'keep', 'nan', 'raise'."
        )

    series = adata.obs[cluster_key].astype(str)
    labels = series.unique().tolist()

    # `series` is matched against `mapping` as strings (see astype(str) above), so a
    # mapping keyed by the column's original dtype (e.g. numpy.int64 cluster ids) would
    # otherwise match nothing at all -- str-cast keys here so int/str mapping keys both
    # work the way a caller would expect.
    mapping = {str(k): v for k, v in mapping.items()}

    unknown_keys = sorted(set(mapping.keys()) - set(labels))
    if unknown_keys:
        print(
            f"Warning: mapping key(s) {unknown_keys} not found in "
            f"adata.obs['{cluster_key}']; ignored."
        )

    if mapping and not (set(mapping.keys()) & set(labels)):
        print(
            f"Warning: none of the {len(mapping)} mapping key(s) matched any label in "
            f"adata.obs['{cluster_key}'] -- every cell will keep its original label. "
            "Check that mapping keys match the column's values (as strings)."
        )

    unmapped_labels = sorted(l for l in labels if l not in mapping)
    if missing == "raise" and unmapped_labels:
        raise ValueError(
            f"Unmapped label(s) {unmapped_labels} in adata.obs['{cluster_key}']; "
            "provide a mapping entry for each, or use missing='keep'/'nan'."
        )

    if missing == "keep":
        new_col = series.map(lambda v: mapping.get(v, v))
    else:  # "nan" or "raise" (already validated complete for "raise")
        new_col = series.map(mapping)

    adata.obs[key_added] = new_col.astype("category")

    # resolved_mapping: the source label -> target label mapping actually
    # applied above, after `missing` has been resolved. Labels that ended up
    # NaN (missing="nan") are excluded -- they got no coherent annotation.
    resolved_mapping = {}
    for label in labels:
        if label in mapping:
            resolved_mapping[label] = str(mapping[label])
        elif missing == "keep":
            resolved_mapping[label] = label

    n_labels_total = len(labels)
    n_labels_mapped = n_labels_total - len(unmapped_labels)
    n_targets = len(set(resolved_mapping.values()))
    n_total_cells = int(len(series))
    n_mapped_cells = int(series.isin(mapping.keys()).sum())

    print(f"\nAnnotation summary ('{cluster_key}' -> '{key_added}')")
    print("─" * 46)
    print(f"  Labels mapped        : {n_labels_mapped:>6,} / {n_labels_total:>6,}")
    print(
        f"  Cells mapped         : {n_mapped_cells:>6,} / {n_total_cells:>6,}"
        f"  ({n_mapped_cells / n_total_cells * 100:.1f} %)"
    )
    if missing == "keep":
        n_kept = n_total_cells - n_mapped_cells
        if n_kept:
            print(
                f"  Cells kept as-is     : {n_kept:>6,}"
                f"  ({n_kept / n_total_cells * 100:.1f} %)"
            )
    elif missing == "nan":
        n_nan = int(new_col.isna().sum())
        if n_nan:
            print(
                f"  Cells set to NaN     : {n_nan:>6,}"
                f"  ({n_nan / n_total_cells * 100:.1f} %)"
            )
    print("─" * 46)

    if _record_report:
        entry = {
            "cluster_key": cluster_key,
            "key_added": key_added,
            "n_labels_mapped": n_labels_mapped,
            "n_labels_total": n_labels_total,
            "n_targets": n_targets,
            "missing": missing,
            "notes": notes,
        }
        try:
            _append_clustering_report(adata, {"action": "annotation", **entry})
        except Exception as e:
            print(f"Warning: failed to record clustering report step: {e}")
        _record_annotation(
            adata, cluster_key, key_added, resolved_mapping, params=entry
        )

    return adata


class _StopAnnotationReview(Exception):
    """Raised by `_review_cluster_annotation` when the reviewer asks to stop early
    (`mode="manual"`'s "q"/"quit" option) -- caught by `suggest_annotation`'s loop, which then
    returns whatever was decided so far instead of continuing or crashing."""


def suggest_annotation(
    adata,
    cluster_key,
    marker_dict,
    mode="auto",
    min_score=0.1,
    min_margin=0.25,
    return_scores=False,
    masks=None,
    images=None,
    region_col="unique_region",
    x_col="x",
    y_col="y",
    n_patch_markers=4,
    resume_dict=None,
    style=None,
    legends=True,
):
    """
    Propose a cluster -> cell type mapping from a marker panel.

    Each marker is z-scored across clusters (so "high" means high *relative to
    the other clusters*, not high in absolute intensity), and every cell type
    is scored as the mean z-score of its markers. The best-scoring cell type
    wins, unless it is too weak (``min_score``) or too close to the runner-up
    (``min_margin``), in which case the cluster is proposed as ``"Unknown"``
    or ``"Mixed"``, respectively.

    Nothing is written to ``adata``: the mapping is returned so it can be
    inspected, edited, and then applied with :func:`annotate_clusters`, which
    records the step in ``adata.uns["clustering_report"]`` and
    ``adata.uns["clustering_tree"]``::

        mapping = suggest_annotation(adata, "leiden", marker_dict)
        annotate_clusters(adata, "leiden", mapping, notes="marker-based")

    Parameters
    ----------
    adata : AnnData
    cluster_key : str
        Column in ``adata.obs`` holding the cluster labels to annotate.
    marker_dict : dict
        ``{cell_type: [marker, ...]}``. Markers absent from
        ``adata.var_names`` are dropped with a warning; a cell type that loses
        all of its markers is dropped too.
    mode : {"auto", "manual"}, optional
        ``"auto"`` (default) returns the proposal as-is. ``"manual"`` walks
        through the clusters one at a time in a notebook, showing a per-marker
        z-score barplot and prompting for confirmation, so the proposal can be
        accepted or overridden cluster by cluster.
    min_score : float, optional
        Minimum score the top cell type must reach; below it the cluster is
        proposed as ``"Unknown"``. Defaults to 0.1.
    min_margin : float, optional
        Minimum gap between the best and second-best score; below it the
        cluster is proposed as ``"Mixed"``. Defaults to 0.25.
    return_scores : bool, optional
        If True, also return the clusters x cell types score table.
        Defaults to False.
    masks, images : optional
        Pre-loaded segmentation masks/images, as in :func:`spacec.plotting.patch_QC`. Only used
        by ``mode="manual"``: when both are given, its prompt gains a ``p`` option that opens
        `patch_QC` for the current cluster (region auto-picked the same way
        :func:`spacec.plotting.run_patch_QC` does), so an uncertain call can be checked visually
        before accepting/overriding it, without leaving the review loop. Omitted (default) or
        only one given, the ``p`` option is left off the prompt entirely -- it's opt-in, not a
        new required setup step for `mode="manual"`.
    region_col, x_col, y_col : str, optional
        Passed through to `patch_QC` (and to region auto-selection) when the ``p`` option is
        used. Defaults ``"unique_region"``, ``"x"``, ``"y"``.
    n_patch_markers : int, optional
        How many of the cluster's own top (most z-scored) markers `patch_QC` shows when the
        ``p`` option is used -- not `marker_dict`'s per-cell-type panels, the cluster's own
        highest-scoring markers overall, regardless of which candidate cell type they came from.
        By default 4.
    resume_dict : dict, optional
        A previous (possibly partial) ``{cluster_label: cell_type}`` result -- e.g. one returned
        after ``mode="manual"``'s ``q``/``quit`` option stopped review early. Every key must already be
        one of this call's actual cluster labels (``ValueError`` otherwise -- the usual cause is
        resuming with a mapping from a different `cluster_key`/round). Clusters already in
        `resume_dict` are carried over as-is and never re-reviewed; only the remaining clusters
        are scored/reviewed this call. The scores table (``return_scores=True``) doesn't depend
        on `resume_dict` at all -- it's the same per-cluster z-score computation every call makes
        regardless, so it's always complete, never partial.
    style : SPACEcStyle, str, or pathlib.Path, optional
        Style for `mode="manual"`'s barplot (spine removal via the active style's
        `remove_top_spine`/`remove_right_spine`). Defaults to the active style.
    legends : bool, optional
        If False, hides both the per-cell-type color legend below `mode="manual"`'s barplot
        and the on-figure "Options" reference panel next to it -- the same options are still
        printed in the text prompt either way, so nothing about what you can type changes.
        Defaults to True. If True, prints an options box and a legend on cell types associated
        with markers, according to marker_dict. Markers present in multiple cell type
        definitions get colored by the last cell type in the `marker_dict`.

    Returns
    -------
    dict
        ``{cluster_label: cell_type}``, keyed by cluster label as ``str`` (the
        form :func:`annotate_clusters` matches against).
    pandas.DataFrame
        Only if ``return_scores=True``: the per-cluster score for every cell
        type, indexed by cluster label.

    Raises
    ------
    ValueError
        If ``cluster_key`` is not in ``adata.obs``, if ``mode`` is not
        ``"auto"``/``"manual"``, if no marker in ``marker_dict`` is present in
        ``adata.var_names``, if ``adata.obs[cluster_key]`` holds fewer than
        two distinct labels (scoring is relative, so a single cluster cannot
        be scored against anything), or if ``resume_dict`` has a key that
        isn't one of this call's actual cluster labels.

    See Also
    --------
    annotate_clusters : Apply the returned mapping and record it.
    spacec.plotting.marker_dotplot : Check the result across annotated types.
    """
    if cluster_key not in adata.obs:
        raise ValueError(f"Column '{cluster_key}' not found in adata.obs")
    if mode not in ("auto", "manual"):
        raise ValueError(f"Invalid mode='{mode}'. Choose from 'auto', 'manual'.")

    # Keep only markers that actually exist, and report what was dropped -- a
    # panel dict is usually copy-pasted between datasets, so misses are normal.
    var_names = set(adata.var_names)
    valid_markers = {}
    dropped = []
    for cell_type, markers in marker_dict.items():
        present = [m for m in markers if m in var_names]
        dropped.extend(m for m in markers if m not in var_names)
        if present:
            valid_markers[cell_type] = present
        else:
            print(
                f"Warning: cell type '{cell_type}' has no markers in adata.var_names; "
                "skipped."
            )
    if dropped:
        print(
            f"Warning: marker(s) not in adata.var_names, ignored: {sorted(set(dropped))}"
        )
    if not valid_markers:
        raise ValueError(
            "None of the markers in marker_dict were found in adata.var_names."
        )

    unique_markers = list(dict.fromkeys(m for ms in valid_markers.values() for m in ms))

    clusters = adata.obs[cluster_key].astype(str)
    labels = sorted(clusters.dropna().unique())
    if len(labels) < 2:
        raise ValueError(
            f"adata.obs['{cluster_key}'] has {len(labels)} distinct label(s); "
            "marker scores are computed relative to the other clusters, so at "
            "least 2 are required."
        )

    if resume_dict:
        foreign = set(resume_dict) - set(labels)
        if foreign:
            raise ValueError(
                f"resume_dict has label(s) not in adata.obs['{cluster_key}']: "
                f"{sorted(foreign)} -- resuming a mapping from a different cluster_key "
                "or round?"
            )

    X = adata[:, unique_markers].X
    if not isinstance(X, np.ndarray):
        X = X.toarray()
    expr = pd.DataFrame(X, index=adata.obs_names, columns=unique_markers)
    means = expr.groupby(clusters.values, observed=True).mean()
    # replace(0, 1) rather than adding an epsilon: a marker that is constant
    # across clusters would otherwise blow up to a ~1/eps z-score and win every
    # cluster. A zero-variance marker should contribute 0, not dominate.
    zscores = (means - means.mean()) / means.std().replace(0, 1)

    scores = pd.DataFrame(
        {ct: zscores[ms].mean(axis=1) for ct, ms in valid_markers.items()}
    )

    mapping = dict(resume_dict) if resume_dict else {}
    to_review = [l for l in labels if l not in mapping]
    already_done = len(mapping)  # captured before the loop -- mapping grows as it runs
    if resume_dict:
        print(
            f"Resuming: {already_done} cluster(s) already annotated, {len(to_review)} remaining."
        )
    stopped_early = False

    _style = _apply_style(style) if mode == "manual" else None

    for j, label in enumerate(to_review, start=1):
        i = (
            already_done + j
        )  # overall position across the full round, not just this session
        row = scores.loc[label].sort_values(ascending=False)
        best, best_score = row.index[0], row.iloc[0]
        second_score = row.iloc[1] if len(row) > 1 else -np.inf

        if best_score < min_score:
            proposal, reason = "Unknown", f"top score {best_score:.2f} < {min_score}"
        elif (best_score - second_score) < min_margin:
            proposal, reason = (
                "Mixed",
                f"{best} {best_score:.2f} vs {row.index[1]} {second_score:.2f}, "
                f"gap < {min_margin}",
            )
        else:
            proposal, reason = best, f"score {best_score:.2f}"

        if mode == "auto":
            print(f"  Cluster {label:>6} -> {proposal}  ({reason})")
            mapping[label] = proposal
        else:
            try:
                mapping[label] = _review_cluster_annotation(
                    label,
                    proposal,
                    reason,
                    zscores,
                    valid_markers,
                    i,
                    len(labels),
                    adata=adata,
                    cluster_key=cluster_key,
                    masks=masks,
                    images=images,
                    region_col=region_col,
                    x_col=x_col,
                    y_col=y_col,
                    n_patch_markers=n_patch_markers,
                    style=_style,
                    legends=legends,
                )
            except _StopAnnotationReview:
                stopped_early = True
                break

    if mode == "manual":
        from IPython.display import clear_output

        clear_output(wait=True)
        if stopped_early:
            print(
                f"\nStopped early -- {len(mapping)}/{len(labels)} cluster(s) annotated so far "
                "('q'/'quit'). Resume later with resume_dict=<this mapping>."
            )
        print(f"\nProposed annotation ('{cluster_key}')")
        print("─" * 46)
        for label, ct in mapping.items():
            print(f"  Cluster {label:>6} -> {ct}")
        print("─" * 46)
    print(f"\nApply with: annotate_clusters(adata, '{cluster_key}', mapping)")

    if return_scores:
        return mapping, scores
    return mapping


def _review_cluster_annotation(
    label,
    proposal,
    reason,
    zscores,
    valid_markers,
    i=1,
    n_total=1,
    adata=None,
    cluster_key=None,
    masks=None,
    images=None,
    region_col="unique_region",
    x_col="x",
    y_col="y",
    n_patch_markers=4,
    style=None,
    legends=True,
):
    """Show one cluster's marker profile and prompt for its annotation.

    Interactive helper for ``suggest_annotation(mode="manual")``; returns the
    label the user chose (or ``proposal`` if they just accept it). ``i``/``n_total``
    are only used for the progress line in the printed header.

    ``masks``/``images`` (both required together -- see ``suggest_annotation``) enable an
    opt-in ``p`` prompt option: opens ``patch_QC`` for this cluster's own top
    ``n_patch_markers`` markers, in a region auto-picked the same way ``run_patch_QC`` does
    (``_select_regions_for_cluster``, first pick only -- a quick spot-check here, not the
    multi-region sweep ``run_patch_QC`` does). Choosing ``p`` re-prompts for the *same*
    cluster afterward rather than advancing -- it's a look, not a decision.

    ``q``/``quit`` (typed, not a literal keypress -- see ``suggest_annotation``'s docstring)
    raises `_StopAnnotationReview`, letting the caller's loop stop and keep whatever was
    already decided instead of continuing or crashing.

    ``legends=False`` hides both the per-cell-type color legend and the on-figure "Options"
    panel; the printed text prompt below the figure lists the same options regardless.

    Markers present in multiple cell type definitions get colored by the last cell type in
    the `marker_dict`.
    """
    patches_available = adata is not None and masks is not None and images is not None
    _style = style if style is not None else _apply_style(None)
    import seaborn as sns
    from IPython.display import clear_output

    # Single source of truth for both the prompt text and the options side-panel, so the two
    # can't drift out of sync. "cell type" has no fixed trigger set -- it's whatever's left
    # over once nothing else matches -- so it's display-only, not matched against `triggers`.
    option_specs = [("accept", "[Enter]/y/yes", ("", "y", "yes"))]
    option_specs.append(("unknown", "u/unknown", ("u", "unknown", "unk")))
    option_specs.append(("cell type", "[any other string]", None))
    if patches_available:
        option_specs.append(("show patches", "p/patch", ("p", "patch")))
    option_specs.append(("stop, keep progress", "q/quit", ("q", "quit")))

    clear_output(wait=True)
    print("=" * 60)
    print(f"CLUSTER {label}   [{i}/{n_total}]")
    bar_width = 30
    filled = round(bar_width * i / n_total) if n_total else 0
    print("[" + "█" * filled + "░" * (bar_width - filled) + "]")
    print(f"Proposed: {proposal}  ({reason})")
    print("=" * 60)

    df_plot = pd.DataFrame(
        [
            {"Cell type": ct, "Marker": m, "Z-score": float(zscores.loc[label, m])}
            for ct, markers in valid_markers.items()
            for m in markers
        ]
    )
    fig, ax = plt.subplots(figsize=(12, 4))
    # hue="Cell type" groups/colors adjacent bars by candidate cell type (df_plot is already
    # built cell-type-by-cell-type, so same-colored bars are already adjacent). The legend is
    # the only way to read that color->cell-type mapping back off: the "Proposed: ..." line
    # above only names the winning candidate, not the others also shown in this chart.
    sns.barplot(
        data=df_plot,
        x="Marker",
        y="Z-score",
        hue="Cell type",
        dodge=False,
        ax=ax,
    )
    ax.axhline(0, color="black", linewidth=1)
    ax.set_title(f"Cluster {label} marker profile")
    plt.setp(ax.get_xticklabels(), rotation=90)

    # seaborn's own default-positioned legend is never what we want (wrong place, or
    # not wanted at all) -- grab its handles/labels, then remove it; passing
    # legend=False to barplot() above instead of doing this would also drop the
    # per-artist labels those handles come from, so the replacement below couldn't
    # be built at all.
    handles, legend_labels = ax.get_legend_handles_labels()
    if ax.get_legend() is not None:
        ax.get_legend().remove()

    # How much vertical room the rotated tick labels + "Marker" axis title actually
    # need varies with the active style (tick-label font size ranges from 6pt to 14pt
    # across `_style.py`'s styles) and with marker-name length, so it's measured from
    # the real render rather than assumed -- a fixed margin that looks right for one
    # style/dataset overlaps the legend for another.
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    xaxis_gap = (
        ax.get_window_extent(renderer).y0 - ax.xaxis.get_tightbbox(renderer).y0
    ) / fig.bbox.height
    pad, bottom_pad = 0.02, 0.01

    bottom = xaxis_gap + pad  # overridden below when the legend is shown
    if legends:
        # A legend to the side (Tim's original placement) works for a handful of cell
        # types, but `marker_dict` can realistically hold dozens -- a tall side legend
        # then either overflows the figure or forces tight_layout() to crush the axes
        # down to a sliver. A horizontal, multi-column legend below the axes scales
        # much better: wrap into at most 6 columns, and size its own height to the
        # number of rows that produces, so a 2-3-cell-type panel doesn't reserve the
        # same empty space a 30-cell-type one needs.
        n_cell_types = len(valid_markers)
        ncol = min(n_cell_types, 6)
        # bbox_transform=fig.transFigure (not the axes-fraction default) so the anchor
        # lines up directly with the figure-fraction gap measured above, with no extra
        # conversion for the axes' own (also dynamic) height. Still ax.legend(), not
        # fig.legend(): that keeps the legend attached to the axes (ax.get_legend()),
        # matching what the side panel and the existing tests expect.
        legend = ax.legend(
            handles,
            legend_labels,
            title="Associated cell type",
            bbox_to_anchor=(0.5, bottom_pad),
            bbox_transform=fig.transFigure,
            loc="lower center",
            ncol=ncol,
            fontsize=8,
            frameon=True,
        )
        # The legend's own height isn't just row count * a guessed row height --
        # title size, row spacing, and padding all scale with the active style's
        # rcParams too (e.g. `presentation()`'s larger legend.title_fontsize), so it's
        # measured the same way as the x-axis gap above: its bottom edge is already
        # pinned to bottom_pad (loc="lower center" above), so drawing once more and
        # reading its real rendered height gives the axes margin that leaves it
        # exactly enough room, with no change to its anchor.
        fig.canvas.draw()
        legend_h = legend.get_window_extent(renderer).height / fig.bbox.height
        bottom = xaxis_gap + pad + bottom_pad + legend_h

        # Small reference panel for the prompt's options, back in its original spot to the
        # right of the axes -- that space is free again now that the color legend moved
        # below the plot instead of competing for the same area. Built from the same
        # option_specs as the prompt below, so the two can't drift out of sync.
        legend_text = "Options:\n" + "\n".join(
            f"  • {opt} ({trigger})" for opt, trigger, _ in option_specs
        )
        ax.text(
            1.02,
            0.5,
            legend_text,
            transform=ax.transAxes,
            va="center",
            ha="left",
            fontsize=9,
            bbox=dict(boxstyle="round", facecolor="white", edgecolor="#999999"),
        )
    _apply_spines(ax, _style)
    # subplots_adjust, not tight_layout() -- tight_layout() doesn't account for a
    # legend placed outside the axes via bbox_to_anchor and will mis-size the plot.
    # The Options panel's own width doesn't depend on how many cell types there are
    # (always the same handful of prompt lines), so it needs a fixed, roomier right
    # margin -- 0.98 is fine with no side panel, but clips the Options box's text.
    right = 0.8 if legends else 0.98
    fig.subplots_adjust(bottom=bottom, right=right, top=0.9, left=0.06)
    plt.show()

    prompt = (
        f"\nAccept '{proposal}'?\n  "
        + ", ".join(f"{label} ({trigger})" for label, trigger, _ in option_specs)
        + ": "
    )

    while True:
        try:
            response = input(prompt).strip()
        except EOFError:
            # No interactive stdin (script, CI, piped input): don't hang or crash
            # halfway through the loop -- take the proposal and say so.
            print(f"Warning: no interactive input available; accepted '{proposal}'.")
            return proposal

        resp = response.lower()
        triggers = {label: opts for label, _, opts in option_specs if opts is not None}

        if resp in triggers["stop, keep progress"]:
            # "q"/"quit" are typed words, not a literal keypress -- input() is line-buffered
            # (waits for Enter) and Jupyter's own input() widget is a plain text box, not a
            # real terminal, so a single raw keystroke can't be captured either way.
            raise _StopAnnotationReview()

        if patches_available and resp in triggers["show patches"]:
            top_markers = (
                zscores.loc[label]
                .sort_values(ascending=False)
                .index[:n_patch_markers]
                .tolist()
            )
            region = _select_regions_for_cluster(adata, cluster_key, label, region_col)[
                0
            ]
            fig = patch_QC(
                adata,
                cluster_key,
                label,
                top_markers,
                masks,
                images,
                region=region,
                region_col=region_col,
                x_col=x_col,
                y_col=y_col,
                style=_style,
            )
            plt.show()
            plt.close(fig)
            continue  # re-prompt for this same cluster -- looking isn't deciding

        if resp in triggers["accept"]:
            return proposal
        if resp in triggers["unknown"]:
            return "Unknown"
        return response  # "cell type" option: whatever's left over is the custom label


def _plot_kelbow(X, k_max, metric="distortion", **kmeans_kwargs):
    """Fit MiniBatchKMeans for k in range(2, k_max) and plot the score curve."""
    ks = list(range(2, k_max))
    scores = []
    for k in ks:
        km = MiniBatchKMeans(n_clusters=k, **{"random_state": 0, **kmeans_kwargs})
        labels = km.fit_predict(X)
        if metric == "distortion":
            scores.append(km.inertia_)
        elif metric == "silhouette":
            scores.append(silhouette_score(X, labels))
        elif metric == "calinski_harabasz":
            scores.append(calinski_harabasz_score(X, labels))
        else:
            raise ValueError(f"Unsupported metric: {metric}")

    fig, ax = plt.subplots()
    ax.plot(ks, scores, marker="o")
    ax.set_xlabel("k")
    ax.set_ylabel(metric)
    ax.set_title(f"{metric.capitalize()} elbow for KMeans clustering")
    plt.show()


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
    kmeans_kwargs=None,
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
        Whether to test for optimal number of clusters and visualize as elbow plot or not, by default False. If set to True, the function will test k = 2 to n_neighborhoods - 1 and plot the distortion score in an elbow plot to assist the user in finding the optimal number of clusters.
    metric : str, optional
        The metric to use when calculating distance between instances in a feature array, by default "distortion". Other options include "silhouette" and "calinski_harabasz".
    kmeans_kwargs : dict, optional
        Extra keyword arguments passed to ``sklearn.cluster.MiniBatchKMeans``, by default None.
        ``random_state`` defaults to 0; ``n_clusters`` is set by ``n_neighborhoods`` and must
        not be passed here. E.g. ``{"batch_size": 100, "n_init": 3}`` restores scikit-learn's
        pre-1.0 defaults (as used by nolanlab/NeighborhoodCoordination). Labels still won't be
        bit-identical to scikit-learn 0.22, since the algorithm itself was reworked in 1.0.

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

    km_params = {"random_state": 0, **(kmeans_kwargs or {})}

    if elbow != True:
        km = MiniBatchKMeans(n_clusters=n_neighborhoods, **km_params)

        labels = km.fit_predict(windows2[sum_cols].values)
        k_centroids[str(k)] = km.cluster_centers_
        adata.obs[neighborhood_name] = labels
        adata.uns[centroids_name] = k_centroids

    else:
        km = MiniBatchKMeans(**km_params)

        X = windows2[sum_cols].values

        labels = km.fit_predict(X)
        k_centroids[str(k)] = km.cluster_centers_
        adata.obs[neighborhood_name] = labels
        adata.uns[centroids_name] = k_centroids

        _plot_kelbow(X, n_neighborhoods, metric=metric, **km_params)

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
    figsize=(9, 3.5),
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
        Cumulative-fraction cutoff that defines each window's CN combination, by
        default 0.85. CNs are added in descending order of their fraction in the
        window until the cumulative fraction exceeds ``threshold``; that smallest
        set of top CNs is the window's combination (see
        ``hf_get_thresh_simps``).
    per_keep_thres : float, optional
        Diagnostic only, by default 0.85. Drawn as the dashed line in the
        cumulative-frequency plot (with a count of how many top combinations
        cover less than this fraction of windows); nothing is filtered by it.
        Combinations occurring in fewer than 0.001 (0.1%) of windows are dropped
        from the CN map regardless (``thresh_freq`` in
        ``tl_build_graph_CN_comb_map``).
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
    keep = [X, Y, unique_region, cn_col] + ([sub_col] if sub_col else [])
    cells_df = cells_df[list(dict.fromkeys(keep))]
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
    # Palette may name CNs absent from this data; give them all-zero columns.
    w = w.reindex(
        columns=[*w.columns, *(c for c in l if c not in w.columns)], fill_value=0
    )
    simps, simp_freqs, simp_sums = tl_calculate_neigh_combs(
        w,
        l,
        k,
        threshold=threshold,
        per_keep_thres=per_keep_thres,
        figsize=figsize,  # color palette
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


def tl_format_for_squidpy(adata, x_col="x", y_col="y"):
    """
    Format an AnnData object for use with Squidpy.

    Creates a new AnnData whose ``obsm["spatial"]`` contains the cell coordinates
    expected by squidpy.  If the input already has ``obsm["spatial"]`` populated
    (e.g. via :func:`make_anndata` or :func:`hf_sync_spatial_obsm`) it is used
    directly; otherwise the coordinates are read from ``adata.obs[x_col]`` and
    ``adata.obs[y_col]``.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix.
    x_col : str, optional
        Column name for x spatial coordinates in ``adata.obs``, by default ``"x"``.
        Ignored when ``obsm["spatial"]`` is already present.
    y_col : str, optional
        Column name for y spatial coordinates in ``adata.obs``, by default ``"y"``.
        Ignored when ``obsm["spatial"]`` is already present.

    Returns
    -------
    AnnData
        New annotated data matrix with ``obsm["spatial"]`` (shape n_cells × 2,
        columns [x, y]) as expected by squidpy.
    """
    if not isinstance(adata, ad.AnnData):
        raise TypeError("adata must be an AnnData object")

    # Prefer obsm["spatial"] if already populated (e.g. by make_anndata).
    if "spatial" in adata.obsm:
        spatial_coordinates = adata.obsm["spatial"]
    else:
        if not isinstance(x_col, str) or not isinstance(y_col, str):
            raise TypeError("x_col and y_col must be strings")
        if x_col not in adata.obs.columns or y_col not in adata.obs.columns:
            raise ValueError(f"Columns {x_col} and/or {y_col} not found in adata.obs")
        spatial_coordinates = adata.obs[[x_col, y_col]].to_numpy(dtype=float)

    if not np.issubdtype(spatial_coordinates.dtype, np.number):
        raise ValueError("Spatial coordinates must be numeric")

    new_adata = ad.AnnData(adata.X, obsm={"spatial": spatial_coordinates})
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
    (optionally casting cell type and region to str), and then computes triangulation
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
            metadata[column].values, index=metadata[shared_column]
        ).to_dict()
        triangulation_distances[column] = triangulation_distances[shared_column].map(
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
        Column name used to compare different conditions. Must have exactly 2
        unique values -- downstream ``filter_interactions`` pivots to one
        column per unique value and requires simultaneous significance
        across all of them, so more than 2 groups (e.g. an 8-category
        location column) silently collapses its result toward empty.
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

    Notes
    -----
    ``distance_pvals``'s ``expected``/``observed`` columns hold a list of
    per-iteration/per-cell raw distances for each row, not a scalar -- storing
    this DataFrame directly in ``adata.uns`` (e.g. ``adata.uns["x"] =
    distance_pvals``) will fail on ``adata.write_h5ad`` with ``TypeError:
    Can't implicitly convert non-string objects to strings``, since h5ad
    cannot serialize object-dtype columns holding lists. Persist the raw
    result externally instead (e.g. ``pickle.dump``), and only store
    ``filter_interactions``'s scalar-valued ``dist_table`` output in
    ``adata.uns`` if you want something saved with the AnnData.
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

    metadata = df_input.loc[:, [region, comparison]].copy()
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
        "triangulation_distances_observed": triangulation_distances_long,
        "triangulation_distances_iterated": iterated_triangulation_distances_long,
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

    A row is significant if its ``pvalue`` column is below the ``pvalue`` argument,
    ``|logfold_group| > logfold_group_abs``, ``celltype1 != celltype2`` and
    ``observed_mean`` is not NaN. A cell-type pair is kept if it is significant in at
    least one of the two groups; ``dist_table`` further requires a ``logfold_group``
    value in both groups (not significance in both). P-values are used as given, with
    no multiple-testing correction.

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

    Raises
    ------
    ValueError
        If ``distance_pvals[comparison]`` does not have exactly 2 unique values.
        The final pivot produces one column per unique value and keeps only
        rows with a value in every column -- with more than 2 groups this
        silently requires simultaneous significance across all of them,
        collapsing the result toward empty with no indication why.
    """
    n_conditions = distance_pvals[comparison].nunique()
    if n_conditions != 2:
        raise ValueError(
            f"filter_interactions expects exactly 2 unique values in the "
            f"'{comparison}' column (found {n_conditions}: "
            f"{sorted(distance_pvals[comparison].unique())}). The final pivot keeps "
            "only pairs significant in every resulting column simultaneously, so "
            "more than 2 groups silently collapses the result toward empty instead "
            "of raising -- pass a binary grouping column (e.g. group A vs. "
            "everything else) instead."
        )

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
    STELLAR_path=None,
    max_memory_usage=1.6e10,
    chunk_size=5000,
    wd=5e-2,
    lr=1e-3,
    seed=1,
    batch_size=1,
    device="auto",
    louvain_flavor="auto",
    num_parts=100,
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
    STELLAR_path : str or None, optional
        Optional, legacy. Filesystem path to a source checkout of the STELLAR fork (its root
        contains a `stellar/` package); prepended to sys.path if given. Not needed when STELLAR
        is installed as a package (see `install_stellar`), by default None.
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
    device : str, optional
        Device to train on: "auto", "cuda", "mps", or "cpu". "auto" picks the best available
        device, by default "auto".
    louvain_flavor : str, optional
        Louvain community detection backend passed through to STELLAR, by default "auto".
        "auto" uses ``"vtraag"`` if the ``louvain`` package is installed
        (``pip install 'spacec[legacy]'``) and otherwise falls back to ``"igraph"``
        with a warning. The backends give different partitions, which affects
        the result when ``num_seed_class > 0``.
    num_parts : int, optional
        Number of METIS graph partitions passed through to STELLAR, by default 100.
        Clamped so that each partition holds at least 10 cells on small inputs.

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

    The function assumes that helper functions (e.g., stellar_get_edge_index) are available in
    the environment.
    """
    print(
        "Please consider to cite the following paper when using STELLAR: "
        "Brbić, M., Cao, K., Hickey, J.W. et al. Annotation of spatially resolved single-cell data with STELLAR. "
        "Nat Methods 19, 1411–1418 (2022). https://doi.org/10.1038/s41592-022-01651-8"
    )

    if STELLAR_path:  # legacy: source checkout of the fork, its root contains stellar/
        sys.path.insert(0, str(STELLAR_path))
    try:
        from stellar import STELLAR, GraphDataset
        from stellar.utils import select_device, set_seed
    except ImportError as e:
        raise ImportError(
            "STELLAR not installed. Run sp.tl.install_stellar() or see README "
            "(pip install git+https://github.com/TKempchen/stellar-py3.12)."
        ) from e

    set_seed(seed)
    args = argparse.Namespace(
        seed=seed,
        epochs=epochs,
        lr=lr,
        wd=wd,
        batch_size=batch_size,
        num_seed_class=num_seed_class,
        sample_rate=sample_rate,
        distance_thres=distance_thres,
        # ponytail: ClusterLoader crashes on partitions with no internal edges, so keep
        # >=10 nodes per METIS part on small inputs; pass a larger num_parts explicitly if needed.
        num_parts=max(
            1,
            min(
                num_parts,
                round(sample_rate * adata_train.n_obs) // 10,
                adata_unannotated.n_obs // 10,
            ),
        ),
        louvain_flavor=louvain_flavor,
        device=select_device(device),
    )
    args.cuda = args.device.type == "cuda"

    print("Preparing input data")
    # Create a mapping between original cell indices and sampled indices
    train_df = adata_train.to_df()
    positions_celltype = adata_train.obs[[x_col, y_col, celltype_col]]
    train_df = pd.concat([train_df, positions_celltype], axis=1)

    # Create a key to track the original indices before sampling
    assert 0 < sample_rate <= 1, "sample_rate must be in (0, 1]"
    train_df["original_idx"] = np.arange(len(train_df))
    train_df = train_df.sample(frac=sample_rate, random_state=seed)

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

    # Set the number of heads based on the number of cell types actually present
    # after subsampling (IMPORTANT: a rare class can vanish during sampling)
    args.num_heads = len(cell_types) + num_seed_class

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
        Default is 'accurate_SVC'. Note: 'accurate_SVC' (``sklearn.svm.SVC`` with
        ``probability=True``) scales roughly O(n^2)-O(n^3) with the number of training
        cells and becomes impractically slow above roughly 20,000-50,000 cells: for
        larger datasets, use 'fast_SVC' or 'knn', or subsample ``adata_train`` first.
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
    # Stratify by label so the held-out test fold keeps at least one example of every
    # class present in y -- an unstratified split can drop the rarest classes from the
    # test fold entirely, and the classification_report(target_names=svc.classes_) call
    # a few lines below then raises ValueError: "Number of classes, N, does not match
    # size of target_names, M" (every class trained on but not all seen in the test
    # fold). stratify=y requires every class to have at least 2 members; fall back to
    # an unstratified split (with a warning) for the rare case that isn't true.
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
    except ValueError as e:
        print(
            f"Warning: stratified split failed ({e}); falling back to an unstratified "
            "split. Classes with very few samples may be under- or unrepresented in "
            "the test fold."
        )
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
    # Generate classification report. Passing `labels=target_names` explicitly (in
    # addition to `target_names`) tells classification_report which full set of
    # classes to report on regardless of which ones actually occur in y_test/svm_label
    # -- without it, a class trained on but absent from the (possibly very small) test
    # fold raises ValueError: "Number of classes, N, does not match size of
    # target_names, M" instead of just reporting zero support for that class.
    target_names = svc.classes_
    svm_eval = classification_report(
        y_true=y_test,
        y_pred=svm_label,
        labels=target_names,
        target_names=target_names,
        output_dict=True,
        zero_division=0,
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


def _count_contacts(mask):
    """Number of distinct touching neighbor labels per cell in a 2D label image (4-connected:
    pixels sharing an edge, not just a corner)."""
    contacts = {}
    for a_sl, b_sl in [(np.s_[:, :-1], np.s_[:, 1:]), (np.s_[:-1, :], np.s_[1:, :])]:
        a, b = mask[a_sl], mask[b_sl]
        border = (a != b) & (a > 0) & (b > 0)
        if not border.any():
            continue
        pairs = np.unique(
            np.column_stack([a[border].ravel(), b[border].ravel()]), axis=0
        )
        for l1, l2 in pairs:
            contacts.setdefault(int(l1), set()).add(int(l2))
            contacts.setdefault(int(l2), set()).add(int(l1))
    return {label: len(neighbors) for label, neighbors in contacts.items()}


def add_morphology_features(
    adata,
    masks=None,
    region_col="unique_region",
    feature_cols=("circularity", "axis_ratio", "extent", "solidity", "n_contacts"),
):
    """
    Add per-cell shape and contact descriptors to `adata.obs`.

    `circularity`, `axis_ratio`, and `solidity` are derived directly from morphology columns
    `cell_segmentation`/`extract_features` already write to `adata.obs` (`area`, `perimeter`,
    `axis_major_length`, `axis_minor_length`, `convex_area`) -- no mask needed for these three.
    `extent` and `n_contacts` need the raw segmentation mask (bounding-box area and neighbor
    counts aren't in `adata.obs`), matched to `adata.obs` rows by the exact `label` column
    `cell_segmentation` assigns -- the same label-matching convention `hf_attach_masks` documents
    -- rather than a pixel-centroid lookup.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix. Must already have `label` and `region_col` in `.obs`, plus
        whichever of `area`/`perimeter`/`axis_major_length`/`axis_minor_length`/`convex_area`
        the requested `feature_cols` need (all written by `cell_segmentation` by default).
    masks : dict[str, numpy.ndarray], optional
        Mapping of `region_col` value to that region's 2D integer-labeled mask array (e.g. from
        `sp.hf.hf_load_masks`). Only required if `feature_cols` includes `"extent"` or
        `"n_contacts"`.
    region_col : str, optional
        obs column identifying which mask each cell belongs to, by default "unique_region".
    feature_cols : tuple of str, optional
        Which features to add. Available: "circularity" (4*pi*area/perimeter^2, 1.0 = perfect
        circle -- uses the plain regionprops perimeter already in `adata.obs`, not the Crofton
        perimeter estimator), "axis_ratio" (minor/major ellipse axis length), "solidity"
        (area / convex-hull area), "extent" (area / bounding-box area, needs `masks`),
        "n_contacts" (number of distinct neighboring cells directly touching this one,
        4-connected, needs `masks`).

    Returns
    -------
    AnnData
        `adata` with the requested columns added to `.obs`.
    """
    obs_requirements = {
        "circularity": ("area", "perimeter"),
        "axis_ratio": ("axis_minor_length", "axis_major_length"),
        "solidity": ("area", "convex_area"),
    }
    mask_features = {"extent", "n_contacts"}

    unknown = set(feature_cols) - set(obs_requirements) - mask_features
    if unknown:
        raise ValueError(f"Unknown feature_cols: {sorted(unknown)}")

    needs_mask = [f for f in feature_cols if f in mask_features]
    if needs_mask and masks is None:
        raise ValueError(
            f"feature_cols {needs_mask} require `masks` (e.g. sp.hf.hf_load_masks(...))"
        )

    missing_obs_cols = sorted(
        {
            col
            for f in feature_cols
            if f in obs_requirements
            for col in obs_requirements[f]
        }
        - set(adata.obs.columns)
    )
    if missing_obs_cols:
        raise ValueError(
            f"adata.obs is missing columns required for {feature_cols}: {missing_obs_cols}"
        )

    obs = adata.obs
    if "circularity" in feature_cols:
        obs["circularity"] = 4 * np.pi * obs["area"] / obs["perimeter"] ** 2
    if "axis_ratio" in feature_cols:
        obs["axis_ratio"] = obs["axis_minor_length"] / obs["axis_major_length"]
    if "solidity" in feature_cols:
        obs["solidity"] = obs["area"] / obs["convex_area"]

    if needs_mask:
        for col in needs_mask:
            obs[col] = np.nan
        for region, mask in masks.items():
            in_region = (obs[region_col] == region).to_numpy()
            if not in_region.any():
                continue
            props = pd.DataFrame(
                regionprops_table(mask, properties=["label", "area", "area_bbox"])
            ).set_index("label")
            props["extent"] = props["area"] / props["area_bbox"]
            props["n_contacts"] = (
                pd.Series(_count_contacts(mask), dtype=float)
                .reindex(props.index)
                .fillna(0)
                .astype(int)
            )
            labels = obs.loc[in_region, "label"].to_numpy()
            cell_feats = props.reindex(labels)[needs_mask].to_numpy()
            obs.loc[in_region, needs_mask] = cell_feats

    return adata


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

    Plugins live in the ``plugins`` folder of the installed ``tissuumaps``
    package, so this works in any environment (venv, conda, system Python).
    Missing plugins are downloaded from the TissUUmaps website.
    """
    urls = [
        "https://tissuumaps.github.io/TissUUmaps/plugins/latest/ClassQC.js",
        "https://tissuumaps.github.io/TissUUmaps/plugins/latest/Plot_Histogram.js",
        "https://tissuumaps.github.io/TissUUmaps/plugins/latest/Points2Regions.js",
        "https://tissuumaps.github.io/TissUUmaps/plugins/latest/Spot_Inspector.js",
        "https://tissuumaps.github.io/TissUUmaps/plugins/latest/Feature_Space.js",
    ]

    save_directory = os.path.join(os.path.dirname(tj.__file__), "plugins")
    os.makedirs(save_directory, exist_ok=True)

    for url in urls:
        save_path = os.path.join(save_directory, os.path.basename(url))
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


def install_stellar(cuda=None):
    """
    Install PyTorch Geometric and the Python 3.12 STELLAR fork for the installed torch.

    Parameters
    ----------
    cuda : str or None, optional
        CUDA wheel tag to install torch-scatter/torch-sparse for, e.g. "cu124". If None,
        it is derived from `torch.version.cuda` ("cpu" if torch has no CUDA build), by
        default None.

    Returns
    -------
    None
        Packages (torch-geometric, torch-scatter, torch-sparse, and the
        `stellar-gnn` package from https://github.com/TKempchen/stellar-py3.12) are
        installed via pip as a side effect.
    """
    v = torch.__version__.split("+")[0]
    if cuda is None:
        cuda = (
            "cpu"
            if torch.version.cuda is None
            else "cu" + torch.version.cuda.replace(".", "")
        )
    pip = [sys.executable, "-m", "pip", "install"]
    subprocess.run(pip + ["torch-geometric>=2.4"], check=True)
    subprocess.run(
        pip
        + [
            "torch-scatter",
            "torch-sparse",
            "-f",
            f"https://data.pyg.org/whl/torch-{v}+{cuda}.html",
        ],
        check=True,
    )
    subprocess.run(
        pip + ["git+https://github.com/TKempchen/stellar-py3.12"], check=True
    )


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
        - radius : int or list
            Radius (or list of radii) for proximity search.
        - plot : bool
            Whether to plot the results.
        - identification_column : str
            Column name used for identification.
        - concavity : int
            Concavity parameter for hull generation.
    nbrs : sklearn.neighbors.NearestNeighbors
        Fitted NearestNeighbors model used to find hull-point neighbors.
    unique_clusters : numpy.ndarray
        Array of unique cluster identifiers present in the data.

    Returns
    -------
    prox_points : pandas.DataFrame
        DataFrame containing points within the proximity of the cluster hull
        that belong to a different cluster.
    hull_nearest_neighbors : pandas.DataFrame
        DataFrame containing the nearest neighbors of the concave hull points.
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
        r_prox_points = r_in_circle_diff_cluster.drop_duplicates().copy()
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
        Serves two roles, default 80: (1) regions with fewer cells of ``group`` than this value are
        skipped; (2) it is passed as HDBSCAN ``min_samples`` when clustering those cells into patches
        (see ``apply_dbscan_clustering``), so larger values give fewer, denser patches and more cells
        labeled as noise.
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
        A scaling factor to convert the radius from its given unit to the coordinate system unit
        (``radius`` is multiplied by this value internally). Default is 1. Example: if your
        coordinates are in pixels at ``S`` micrometers/pixel and you want to pass ``radius`` in
        micrometers, use ``original_unit_scale = 1 / S`` (pixels per micrometer) -- not ``S``
        itself, which is the easy way to get this backwards.
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

                # remove column unique_region_numeric and buffer_distance, and the per-cell
                # Point `geometry` inherited from codex_gdf -- it only duplicates x_column/
                # y_column (already present as plain floats) and has no registered anndata
                # writer, so leaving it in would break adata.write() once this DataFrame is
                # stored in adata.uns below.
                combined_df.drop(
                    columns=["unique_region_numeric", "buffer_distance", "geometry"],
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

    # generate new column named unique_patch_ID that combines the region, group and patch ID --
    # must happen before storing into adata.uns below. Storing used to come first, and this
    # column only ended up in adata.uns[key_name] by accident the first time a given key_name
    # was used, because `adata.uns[key_name] = final_results` shares the same DataFrame object
    # that this in-place column assignment then mutated. On a second call reusing the same
    # key_name, `pd.concat([adata.uns[key_name], final_results])` makes a new object before
    # this column existed on `final_results`, leaving it NaN for the new rows -- a NaN/str
    # mixed column that h5py's vlen-string writer can't serialize on adata.write().
    final_results["unique_patch_ID"] = (
        final_results[region_column]
        + "_"
        + final_results[patch_column]
        + "_"
        + "patch_no_"
        + final_results["patch_id"].astype(str)
    )

    # add as key to adata.uns
    if key_name is None:
        key_name = "ppa_result"
    if key_name in adata.uns:
        adata.uns[key_name] = pd.concat([adata.uns[key_name], final_results])
    else:
        adata.uns[key_name] = final_results

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
