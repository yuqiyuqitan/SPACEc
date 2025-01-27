# SPatial Analysis for CodEX data (SPACEc)

[![Stars](https://img.shields.io/github/stars/yuqiyuqitan/SPACEc?style=flat&logo=GitHub&color=yellow)](https://github.com/yuqiyuqitan/SPACEc/stargazers)
[![Downloads](https://pepy.tech/badge/spacec)](https://pepy.tech/project/spacec)
[![PyPI-Server](https://img.shields.io/pypi/v/spacec?logo=PyPI)](https://pypi.org/project/spacec/)
[![Documentation Status](https://readthedocs.org/projects/spacec/badge/?version=latest)](https://spacec.readthedocs.io/en/latest/?badge=latest)
[![Test Workflow](https://github.com/yuqiyuqitan/SPACEc/actions/workflows/ci.yml/badge.svg)](https://github.com/yuqiyuqitan/SPACEc/actions/workflows/ci.yml)
[![DOI:10.1101/2024.06.29.601349](https://zenodo.org/badge/doi/10.5281/zenodo.4018965.svg)](https://doi.org/10.1101/2024.06.29.601349)

**Multiplexed imaging** technologies offer valuable insights into intricate tissue structures, yet they pose significant computational hurdles. These include cumbersome data handoffs, inefficiencies in processing large images (often ranging from 8 to 40 gigabytes per image), and limited spatial analysis capabilities. We created **SPACEc, an all-in-one, scalable Python platform that advances both analytical capabilities and computational efficiency.** Through careful engineering optimization, it streamlines the entire process from image extraction and cell segmentation to data preprocessing, while introducing novel approaches such as Patch Proximity Analysis for mapping cellular microenvironments to fill in the current analytic gaps. The platform significantly improves the performance of existing tools through parallelization and GPU acceleration, including enhanced cell-cell interaction analysis and simplified deep-learning annotation workflows, while its intuitive user-friendly design makes these advanced spatial analyses accessible to a wider scientific audience.

## General outline of SPACEc analysis

![SPACEc](https://raw.githubusercontent.com/yuqiyuqitan/SPACEc/master/docs/overview.png)

## Installation notes

**Note**: We currently support Python==`3.9` and `3.10`.
* The **example tonsil data** is available on [dryad](https://datadryad.org/stash/share/OXTHu8fAybiINGD1S3tIVUIcUiG4nOsjjeWmrvJV-dQ)

### Install

<details><summary>Linux</summary>

SPACEc CPU

```bash
    # Create conda environment
    conda create -n spacec python==3.10
    conda activate spacec

    # Install dependencies via conda.
    conda install -c conda-forge graphviz libvips openslide

    # Install spacec
    pip install spacec
```

#### Continue the following steps only if you have GPU(s)

SPACEc GPU

```bash
    # Set environment variables
    conda install conda-forge::cudatoolkit=11.2.2 cudnn=8.1.0.77 -y
    # Set environment variables for the conda environment
    mkdir -p $CONDA_PREFIX/etc/conda/activate.d && \
    echo -e 'export PATH=$CONDA_PREFIX/bin:$PATH\nexport LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH' > $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh && \
    chmod +x $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh

    # Ensure package compatibility
    # Note: Ignore dependency issues for now (seaborn)!
    pip install protobuf==3.19.6 tensorflow-gpu==2.8.0 # numpy==1.24

    # If Pytorch does not find the GPU try:
    # pip install torch==1.12.0+cu113 torchvision==0.13.0+cu113 torchaudio==0.12.0 --extra-index-url https://download.pytorch.org/whl/cu113

    # restart the environment to reload PATH variable
    conda deactivate
    conda activate spacec
```

1. For GPU-accelerated clustering via RAPIDS, note that only RTX20XX or better GPUs are supported (optional).
```bash
    conda install -c rapidsai -c conda-forge -c nvidia rapids=24.02
    pip install rapids-singlecell==0.9.5 pandas==1.5
    pip install --ignore-installed networkx==3.2
```

2. To run STELLAR (optional).
```bash
    # more information please refer to https://pytorch-geometric.readthedocs.io/en/2.1.0/notes/installation.html
    pip install torch==1.12.0+cu113 torchvision==0.13.0+cu113 torchaudio==0.12.0 --extra-index-url https://download.pytorch.org/whl/cu113
    pip install torch-scatter==2.1.0 torch-sparse==0.6.16 torch-cluster==1.6.0 torch-spline-conv==1.2.1 torch-geometric==2.2.0 -f https://data.pyg.org/whl/torch-1.12.0+cu113.html
    # conda install pyvips  # if you get the error "OSError: cannot load library 'libvips.so.42'"
```

<!-- Martin: I don't think this is required. -->
3. Reinstall SPACEc to be compatible with the GPU setting
```bash
    # Install spacec
    pip install spacec
```

4. Test if SPACEc loads and if your GPU is visible if you installed the GPU version. In Python:
    ```python
    import spacec as sp
    sp.hf.check_for_gpu()
    ```

* ⚠️ **IMPORTANT**: always import `spacec` first before importing any other packages

</details>


<details><summary>Apple M1/M2/M3/M4</summary>


SPACEc CPU:

```bash
    conda create -n spacec
    conda activate spacec

    # Install Python via conda
    conda install python==3.10

    # Install dependencies via conda.
    conda install -c conda-forge graphviz libvips openslide
    # conda install -c conda-forge pyvips  # only required for Python 3.9

    # Install spacec
    pip install spacec

    # Install remaining requirements for deepcell
    # NOTE: Ignore the error about pip's dependency resolver
    pip install -r https://raw.githubusercontent.com/nolanlab/SPACEc/master/requirements-deepcell-mac-arm64_tf210-metal.txt
    pip install deepcell --no-deps
```
SPACEc GPU: Mac GPU support is currently only supported for Tensorflow based methods not PyTorch, we recommend you use Linux system for full GPU acceleration.

* ⚠️ **IMPORTANT**: always import `spacec` first before importing any other packages
</details>

<details><summary>Windows</summary>

Although SPACEc can run directly on Windows systems, we highly recommend running it in WSL. If you are unfamiliar with WSL, you can find more information on how to use and install it here: https://learn.microsoft.com/en-us/windows/wsl/install If you decide to use WSL, follow the Linux instructions.

If you plan to continue with the native Windows environment
1. One of the segmentation tools within SPACEc neeeds a C++ compiler. If your environment doesn't have it already, the easiest way is to:
    1. Download the community version of Visual Studio from the official Microsoft website: [https://visualstudio.microsoft.com](https://visualstudio.microsoft.com/). After installing the software on your system, select the following options to install the components needed for C++ development (see screenshots)

        ![image](https://github.com/user-attachments/assets/ca35fe30-8deb-448f-bac7-688774b770aa)

        ![image 1](https://github.com/user-attachments/assets/f4344363-5a31-4695-b75c-5ed8c416b7c2)

    2. In the meantime, you can already install libvips ([https://www.libvips.org/](https://www.libvips.org/)) by downloading the pre-compiled Windows binaries from this repository: https://github.com/libvips/build-win64-mxe/releases/tag/v8.16.0 and adding them to your PATH. If you are unsure about which version to choose, [vips-dev-w64-all-8.16.0.zip](https://github.com/libvips/build-win64-mxe/releases/download/v8.16.0/vips-dev-w64-all-8.16.0.zip) should work for you.
    3. Unpack the zip file and add the directory to your PATH environment. If you don’t know how to do that, consider watching this tutorial video that explains the process: [https://www.youtube.com/watch?v=O5iBsdAd1_w](https://www.youtube.com/watch?v=O5iBsdAd1_w)

SPACEc CPU:

```bash
    # Create conda environment
    conda create -n spacec python==3.10
    conda activate spacec

    # Install dependencies via conda.
    conda install -c conda-forge graphviz

    # Install spacec
    pip install spacec
```

SPACEc GPU:
```bash
    conda install conda-forge::cudatoolkit=11.2.2 cudnn=8.1.0.77 -y

    mkdir %CONDA_PREFIX%\etc\conda\activate.d && (
    echo @echo off > %CONDA_PREFIX%\etc\conda\activate.d\env_vars.bat
    echo set PATH=%CONDA_PREFIX%\bin;%PATH% >> %CONDA_PREFIX%\etc\conda\activate.d\env_vars.bat
    echo set LD_LIBRARY_PATH=%CONDA_PREFIX%\lib;%LD_LIBRARY_PATH% >> %CONDA_PREFIX%\etc\conda\activate.d\env_vars.bat
    )

    # If Pytorch does not find the GPU try:
    # pip install torch==1.12.0+cu113 torchvision==0.13.0+cu113 torchaudio==0.12.0 --extra-index-url https://download.pytorch.org/whl/cu113
```

Reinstall SPACEc to be compatible with the GPU setting
```bash
    # Install spacec
    pip install spacec
```

Test if SPACEc loads and if your GPU is visible if you installed the GPU version.
```python
    import spacec as sp
    sp.hf.check_for_gpu()
```

* ⚠️ **IMPORTANT**: always import `spacec` first before importing any other packages
</details>


<details><summary>Docker</summary>
If you encounter installation issues or prefer a containerized setup, use the SPACEc Docker image. You can build or modify it using the repository's Dockerfiles.

```bash
# Run CPU version:
docker build -f ../Docker/spacec_cpu_build.dockerfile -t spacec:cpu .
docker run -p 8888:8888 -p 5100:5100 tkempchen/spacec:cpu

# If running an amd64 image on apple silicon, use the following command:
docker run --platform linux/amd64 -p 8888:8888 -p 5100:5100 tkempchen/spacec:cpu

# Or run GPU version:
docker build -f ../Docker/spacec_gpu_build.dockerfile -t spacec:gpu .
docker run --gpus all -p 8888:8888 -p 5100:5100 tkempchen/spacec:gpu
```
</details>
