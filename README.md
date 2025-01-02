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

SPACEc should work on most Linux distributions. However, we only tested it on Ubuntu.

1. Download & install Conda package manager.
    1. We recommend the Miniforge distribution of Mamba (can be found here: https://github.com/conda-forge/miniforge) because it is fast. However, every other version of conda will work as well. The following installation guide will use conda commands.
2. Go to your terminal and create a conda environemt for SPACEc.

    ```bash
    conda create -n spacec
    ```

    After creating the environment, activate it. Only if the environment is activated, you install SPACEc into the correct virtual environment. Remember to activate the environment every time you use SPACEc!

    ```bash
    conda activate spacec
    ```

3. Install dependencies via conda.

    ```bash
    # install Python
    conda install python==3.10

    # ignore if you have no Nvidia GPU installed
    conda install conda-forge::cudatoolkit=11.2.2 -y
    conda install conda-forge::cudnn=8.1.0.77 -y

    conda install -c conda-forge graphviz libvips pyvips openslide-python
    ```

4. If you want to use your GPU run these additional commands. Otherwise, only install SPACEc.

    ```bash
    # Set environment variables
    mkdir -p $CONDA_PREFIX/etc/conda/activate.d
    echo 'export PATH=$CONDA_PREFIX/bin:$PATH' > $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
    echo 'export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH' >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
    chmod +x $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
    ```

    ```bash
    pip install spacec
    ```

    ```bash
    pip install protobuf==3.20.0
    pip install numpy==1.24.*
    pip install tensorflow-gpu==2.8.0
    ```

5. If you want to install RAPIDS for GPU-accelerated clustering, note that only RTX20XX or better GPUs are supported.

    ```bash
    conda install -c rapidsai -c conda-forge -c nvidia rapids=24.02
    pip install rapids-singlecell==0.9.5
    pip install pandas==1.5.*
    ```

6. In case you want to run STELLAR from within SPACEc you also need to install PyTorch Geometric. The right version can be found by checking for your PyTorch version from within Python and then installing the correct version of PyTorch Geometric on top of SPACEc: [https://pytorch-geometric.readthedocs.io/en/2.5.2/notes/installation.html](https://pytorch-geometric.readthedocs.io/en/2.5.2/notes/installation.html)
7. Now you can start your analysis with SPACEc! Consider downloading and stepping through the provided notebooks to learn how SPACEc can be used. SPACEc can be used from your IDE of choice e.g. Jupyter Lab:

    ```bash
    jupyter-lab
    ```

8. Test if SPACEc loads and if your GPU is visible if you installed the GPU version.

    ```python
    import spacec as sp
    sp.hf.check_for_gpu()
    ```

* ⚠️ **IMPORTANT**: always import `spacec` first before importing any other packages
* **Example tonsil data** on [dryad](https://datadryad.org/stash/share/OXTHu8fAybiINGD1S3tIVUIcUiG4nOsjjeWmrvJV-dQ)

</details>


<details><summary>Apple M1/M2</summary>

If you run SPACEc on an Apple M chip, you should consider the additional step 3 to avoid compatibility issues. At the moment, Apple GPUs are not officially supported by SPACEc.

1. Download & install Conda package manager.
    1. We recommend the Miniforge distribution of Mamba (which can be found here: https://github.com/conda-forge/miniforge) because it is fast. However, every other version of conda will work as well. The following installation guide will use conda commands.
2. Go to your terminal and create a conda environemt for SPACEc.

    ```bash
    conda create -n spacec
    ```

    After creating the environment, activate it. Only if the environment is activated you install SPACEc into the correct virtual environment. Remember to activate the environment every time you use SPACEc!

    ```bash
    conda activate spacec
    ```

3. If you experience problems with running SPACEc on an Apple M chip, please force the environment to run as an Intel environment.

    ```bash
    # set environment; Apple specific
    conda config --env --set subdir osx-64
    ```

4. Install dependencies via conda.

    ```bash
    # install Python
    conda install python==3.10

    conda install install -c conda-forge graphviz libvips pyvips openslide-python
    ```

5. Install SPACEc in your conda environment. Note: We provide SPACEc via PyPi; therefore, you need to use pip.

    ```bash
    pip install spacec
    ```

6. We observed that on some Apple machines, the correct versions of some dependencies are not always automatically downloaded. Fix the environment for Apple machines by running the following code after installing SPACEc.

    ```bash
    conda install tensorflow=2.10.0
    pip uninstall werkzeug -y
    pip install numpy==1.26.4 werkzeug==2.3.8
    ```

7. Now you can start your analysis with SPACEc! Consider downloading and stepping through the provided notebooks to learn how SPACEc can be used. SPACEc can be used from your IDE of choice e.g. Jupyter Lab:

    ```bash
    jupyter-lab
    ```

8. Test if SPACEc loads. In theory, PyTorch supports MPS now and allows Apple M users to use their GPU. However, not all dependencies support this yet without issues. If you experience MPS-related issues, switch to a CPU-only version of PyTorch.

    ```python
    import spacec as sp
    sp.hf.check_for_gpu()
    ```

* ⚠️ **IMPORTANT**: always import `spacec` first before importing any other packages
* **Example tonsil data** on [dryad](https://datadryad.org/stash/share/OXTHu8fAybiINGD1S3tIVUIcUiG4nOsjjeWmrvJV-dQ)

</details>

<details><summary>Windows</summary>

Although SPACEc can run directly on Windows systems, we highly recommend running it in WSL. If you are unfamiliar with WSL, you can find more information on how to use and install it here: https://learn.microsoft.com/en-us/windows/wsl/install

If you decide to use WSL, follow the Linux instructions.

1. Download & install Conda package manager.
    1. We recommend the Miniforge distribution of Mamba (which can be found here: https://github.com/conda-forge/miniforge) because it is fast. However, every other version of conda will work as well. The following installation guide will use conda commands.
2. To run SPACEc you will need to install some additional software on windows.
    1. Download the community version of Visual Studio from the official Microsoft website: [https://visualstudio.microsoft.com](https://visualstudio.microsoft.com/)
    2. After installing the software on your system, you will be presented with a launcher that allows you to select the components to be installed on your system.
    3. Install the components needed for C++ development (see screenshots) - The download will be a few GB in size, so prepare to wait depending on your internet connection.

        ![image](https://github.com/user-attachments/assets/ca35fe30-8deb-448f-bac7-688774b770aa)

        ![image 1](https://github.com/user-attachments/assets/f4344363-5a31-4695-b75c-5ed8c416b7c2)

    5. In the meantime, you can already install libvips ([https://www.libvips.org/](https://www.libvips.org/)) by downloading the pre-compiled Windows binaries from this repository: https://github.com/libvips/build-win64-mxe/releases/tag/v8.16.0 and adding them to your PATH. If you are unsure about which version to choose, [vips-dev-w64-all-8.16.0.zip](https://github.com/libvips/build-win64-mxe/releases/download/v8.16.0/vips-dev-w64-all-8.16.0.zip) should work for you.
    6. Unpack the zip file and add the directory to your PATH environment. If you don’t know how to do that, consider watching this tutorial video that explains the process: [https://www.youtube.com/watch?v=O5iBsdAd1_w](https://www.youtube.com/watch?v=O5iBsdAd1_w)
3. Open conda command line and create an environment for SPACEc

    ```bash
    conda create -n spacec
    ```

    After creating the environment, activate it. Only if the environment is activated you install SPACEc into the correct virtual environment. Remember to activate the environment every time you use SPACEc!

    ```bash
    conda activate spacec
    ```

4. Install Python 3.10, git and graphviz in your conda environment using the following command:

    ```bash
    conda install python==3.10
    conda install git graphviz
    ```

5. If you want to use Deepcell Mesmer with GPU acceleration (Nvidia GPU), you need to install the correct version of cudatoolkit and cudnn in your environment. For that execute the following code:

    ```bash
    conda install conda-forge::cudatoolkit=11.2.2 -y
    conda install conda-forge::cudnn=8.1.0.77 -y

    mkdir %CONDA_PREFIX%\etc\conda\activate.d
    echo @echo off > %CONDA_PREFIX%\etc\conda\activate.d\env_vars.bat
    echo set PATH=%CONDA_PREFIX%\bin;%PATH% >> %CONDA_PREFIX%\etc\conda\activate.d\env_vars.bat
    echo set LD_LIBRARY_PATH=%CONDA_PREFIX%\lib;%LD_LIBRARY_PATH% >> %CONDA_PREFIX%\etc\conda\activate.d\env_vars.bat
    ```

6. Install SPACEc in your conda environment. Note: We provide SPACEc via PyPi; therefore, you need to use pip.

    ```bash
    pip install spacec
    ```

7. Now you can start your analysis with SPACEc! Consider downloading and stepping through the provided notebooks to learn how SPACEc can be used. SPACEc can be used from your IDE of choice e.g. Jupyter Lab:

    ```bash
    jupyter-lab
    ```

8. Test if SPACEc loads and if your GPU is visible if you installed the GPU version.

    ```python
    import spacec as sp
    sp.hf.check_for_gpu()
    ```

* ⚠️ **IMPORTANT**: always import `spacec` first before importing any other packages
* **Example tonsil data** on [dryad](https://datadryad.org/stash/share/OXTHu8fAybiINGD1S3tIVUIcUiG4nOsjjeWmrvJV-dQ)

</details>


### Docker
If you run into an installation issue or want to run SPACEc in a containerized environment, we have created a Docker image for you to use SPACEc so that you don't have to install manually. You can find the SPACEc Docker image here: https://hub.docker.com/r/tkempchen/spacec
You can also build and modify the Docker image yourself by using the Dockerfiles in the repository.

```bash
#Run CPU version:
docker pull tkempchen/spacec:cpu
docker run -p 8888:8888 -p 5100:5100 spacec:cpu

#Or run GPU version:
docker pull tkempchen/spacec:gpu
docker run --gpus all -p 8888:8888 -p 5100:5100 spacec:gpu
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
