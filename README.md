# SPatial Analysis for CodEX data (SPACEc)

[![Documentation Status](https://readthedocs.org/projects/spacec/badge/?version=latest)](https://spacec.readthedocs.io/en/latest/?badge=latest)
![example workflow](https://github.com/yuqiyuqitan/SPACEc/actions/workflows/ci.yml/badge.svg)

[Preprint](https://doi.org/10.1101/2024.06.29.601349): more detailed explanation on each steps in Supplementary Notes 2 (p13-24).
[Tutorial](https://spacec.readthedocs.io/en/latest/?badge=latest)

## Installation notes

**Note**: We currently support Python==`3.9` and `3.10`.

We generally recommend to use a `conda` environment. It makes installing requirements like `graphviz` a lot easier.

### Install

<details><summary>Linux</summary>

```bash
# setup `conda` repository
conda create -n spacec
conda activate spacec

# install Python
conda install python==3.10

# install `graphviz`
conda install graphviz

# install 'libvips'; Mac and Linux specific
conda install -c conda-forge libvips pyvips openslide-python

# install `SPACEc` from pypi
pip install spacec
```

* ⚠️ **IMPORTANT**: always import `spacec` first before importing any other packages
* **Example tonsil data** on [dryad](https://datadryad.org/stash/share/OXTHu8fAybiINGD1S3tIVUIcUiG4nOsjjeWmrvJV-dQ)

</details>


<details><summary>Apple M1/M2</summary>

```bash
# setup `conda` repository
conda create -n spacec
conda activate spacec

# set environment; Apple specific
conda config --env --set subdir osx-64

# install Python
conda install python==3.9

# install `graphviz`
conda install graphviz

# install 'libvips'; Mac and Linux specific
conda install -c conda-forge libvips pyvips openslide-python

# requirements not automatically installed otherwise; Apple specific
pip install numpy==1.26.4 werkzeug==2.3.8

# install `SPACEc` from pypi
pip install spacec

# reinstall tensorflow; Apple specific
conda install tensorflow=2.10.0
```

* ⚠️ **IMPORTANT**: always import `spacec` first before importing any other packages
* **Example tonsil data** on [dryad](https://datadryad.org/stash/share/OXTHu8fAybiINGD1S3tIVUIcUiG4nOsjjeWmrvJV-dQ)

</details>

<details><summary>Windows</summary>

Although SPACEc can run directly on Windows systems, we highly recommend running it in WSL. If you are unfamiliar with WSL, you can find more information on how to use and install it here: https://learn.microsoft.com/de-de/windows/wsl/install

If you decide to use WSL, follow the Linux instructions.

```bash
# setup `conda` repository
conda create -n spacec
conda activate spacec

# install Python
conda install python==3.9

# install `graphviz`
conda install graphviz

# install `SPACEc` from pypi
pip install spacec
```

* ⚠️ **IMPORTANT**: always import `spacec` first before importing any other packages
* **Example tonsil data** on [dryad](https://datadryad.org/stash/share/OXTHu8fAybiINGD1S3tIVUIcUiG4nOsjjeWmrvJV-dQ)

</details>


### Docker
If you run into an installation issue or want to run SPACEc in a containerized environment, we have created a Docker image for you to use SPACEc so that you don't have to install manually. You can find the SPACEc Docker image here: https://hub.docker.com/r/tkempchen/spacec

```bash
#Run CPU version:
docker pull tkempchen/spacec:cpu
docker run -p 8888:8888 -p 5100:5100 spacec:cpu

#Or run GPU version:
docker pull tkempchen/spacec:gpu
docker run --gpus all -p 8888:8888 -p 5100:5100 spacec:gpu
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

# Note: before you run `pytest` you might have to deactivate and activate the conda environment first
# conda deactivate; conda activate spacec

pytest
```

## General outline of SPACEc analysis

![SPACEc](https://raw.githubusercontent.com/yuqiyuqitan/SPACEc/master/docs/overview.png)
