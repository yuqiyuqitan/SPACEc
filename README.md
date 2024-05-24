#SPatial Analysis for CodEX data (SPACEc)

## Installation notes

**Note**: Due to some dependencies, we currently only support Python up to `3.10`.

We generally recommend to use a `conda` environment. It makes installing requirements like `graphviz` a lot easier.

### Install

```bash
# setup `conda` repository
conda create -n spacec python==3.10
conda activate spacec

# install `graphviz`
conda install graphviz

# install 'libvips'
conda install -c conda-forge libvips pyvips openslide-python

# install `SPACEc` from pypi
pip install spacec

# install `SPACEc` from cloned repo
#pip install -e .

# on Apple M1/M2
# conda install tensorflow=2.10.0
# and always import spacec first before importing other packages
```

### Install additional features 
#### GPU accelerated clustering
NOTE: This module is based on Nvidia `RAPIDS` that is currently only available on linux! If you run SPACEc on a Windows machine you need to run SPACEc in WSL to take advantage of this module. For further information read the offical RAPIDS documentation:
- https://t1p.de/hxo3c

To use RAPIDS you need a Linux-based system (we tested under Ubuntu 22) and an Nvidia RTX 20 Series GPU or better.

```bash
# before installing GPU related features check your installed CUDA version
nvcc --version

# make sure to use the right CUDA version! Here is an example for CUDA 12

pip install rapids-singlecell==0.9.5

pip install --extra-index-url=https://pypi.nvidia.com cudf-cu12==24.2.* dask-cudf-cu12==24.2.* cuml-cu12==24.2.* cugraph-cu12==24.2.* cuspatial-cu12==24.2.* cuproj-cu12==24.2.* cuxfilter-cu12==24.2.* cucim-cu12==24.2.* pylibraft-cu12==24.2.* raft-dask-cu12==24.2.*

pip install protobuf==3.20
```

#### STELLAR machine learning-based cell annotation
Further install information for `PyTorch` and `PyTorch Geometric` can be found here:
- https://pytorch.org/get-started/locally/
- https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html

```bash
# before installing GPU related features check your installed CUDA version
nvcc --version

# install 'PyTorch' and 'PyTorch Geometric' (only needed if STELLAR is used)
# make sure to use the right CUDA version! Here is an example for CUDA 12 and PyTorch 2.3

pip install torch

pip install torch_geometric

pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.3.0+cu121.html
```

### Run tests.

```bash
pip install pytest pytest-cov
pytest
```


```bash
# conda create -n sap python==3.8.0
# pip install deepcell cellpose

# conda install glob2 matplotlib numpy pandas scanpy seaborn scipy networkx tensorly statsmodels scikit-learn yellowbrick joblib tifffile tensorflow
# conda install anaconda::graphviz
# conda install -c conda-forge scikit-image
# pip install leidenalg concave-hull==0.0.6
```

## General outline of SPACEc analysis

![SPACEc](https://github.com/yuqiyuqitan/SAP/tree/master/docs/overview.png?raw=true "")


### Tissue extraction
	Step 1: Set up the environment
	Step 2: Downscale the whole tissue image
	Step 3: Rename tissue number (optional)
	Step 4: Extract individual labeled tissues into separate tiff stack

### Cell segmentation & visualization
	Step 5: Visualize segmentation channel (optional)
	Step 6: Run cell segmentation
	Step 7: Visually inspect segmentation results

### Data preprocessing & normalization
	Step 8: Load the segmented results
	Step 9: Initial filtering of artifacts/debris by DAPI intensity and cell size
	Step 10: Normalize data
	Step 11: Second filtering for noisy cell

### Cell type annotation
	Step 12: Cell type annotation via clustering
	Step 13: Visualize & annotate clustering results
	Step 14: Compute basic statistics of the cell type composition
	Step 15: Cell type annotation via machine-learning classification

### Interactive data inspection and exploration
	Step 16: Prepare data for an interactive session
	Step 17: Additional interaction data exploration (optional)

### Spatial analysis
	Step 18: Compute cellular neighborhoods
	Step 19: Visualize & annotate the cellular neighborhoods analysis results
	Step 20: Generate spatial context maps
	Step 21: Create cellular neighborhood interface analysis via barycentric coordinate plot
	Step 22: Compute for patch proximity analysis
	Step 23: Calculate cell-cell interaction
