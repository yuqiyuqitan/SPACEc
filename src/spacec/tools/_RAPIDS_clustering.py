import numpy as np
import pandas as pd
import os

import rapids_singlecell as rsc

from pyFlowSOM import map_data_to_nodes, som
from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import cupy as cp

from cupyx.scipy.sparse import csc_matrix as csc_matrix_gpu
from cupyx.scipy.sparse import csr_matrix as csr_matrix_gpu
from cupyx.scipy.sparse import isspmatrix_csc as isspmatrix_csc_gpu
from cupyx.scipy.sparse import isspmatrix_csr as isspmatrix_csr_gpu
from scanpy.get import _get_obs_rep, _set_obs_rep
from scipy.sparse import isspmatrix_csc as isspmatrix_csc_cpu
from scipy.sparse import isspmatrix_csr as isspmatrix_csr_cpu

if TYPE_CHECKING:
    from anndata import AnnData
    
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
