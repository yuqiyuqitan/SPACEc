# Copyright (c) Jupyter Development Team.
# Distributed under the terms of the Modified BSD License.
ARG OWNER=jupyter
ARG BASE_CONTAINER=$OWNER/r-notebook:python-3.9
FROM $BASE_CONTAINER

LABEL maintainer="Jupyter Project <jupyter@googlegroups.com>"

# Fix: https://github.com/hadolint/hadolint/wiki/DL4006
# Fix: https://github.com/koalaman/shellcheck/wiki/SC3014
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

USER root

RUN apt-get update --yes && \
    apt-get install --yes --no-install-recommends \
    # for cython: https://cython.readthedocs.io/en/latest/src/quickstart/install.html
    build-essential \
    # for latex labels
    cm-super \
    dvipng \
    # for matplotlib anim
    ffmpeg && \
    apt-get clean && rm -rf /var/lib/apt/lists/* 

USER ${NB_UID}

# Install Python 3 packages
RUN mamba install --yes \
    'altair' \
    'beautifulsoup4' \
    'bokeh' \
    'bottleneck' \
    'cloudpickle' \
    'conda-forge::blas=*=openblas' \
    'cython' \
    'dask' \
    'dill' \
    'h5py' \
    'ipympl'\
    'ipywidgets' \
    # Temporary fix for: https://github.com/jupyter/docker-stacks/issues/1851
    'jupyter_server>=2.0.0' \
    'jupyterlab-git' \
    'matplotlib-base=3.6.3' \
    'numba=0.55.2' \
    'numexpr' \
    'numpy=1.22.3' \
    'openpyxl' \
    'pandas=1.5.3' \
    'patsy' \
    'protobuf' \
    'pytables' \
	'tensorly' \
	'scanpy' \
    'scikit-image' \
    'scikit-learn' \
    'scipy' \
    'seaborn=0.12.2' \
    'sqlalchemy' \
    'statsmodels=0.13.5' \
    'sympy' \
    'widgetsnbextension'\
    'yellowbrick=1.5' \
    "graphviz" \
    "pydot" \
    "pygraphviz" \
    "conda-forge::libvips" \
    "leidenalg" \
    'xlrd' && \
    mamba clean --all -f -y && \
    fix-permissions "${CONDA_DIR}" && \
    fix-permissions "/home/${NB_USER}"

RUN conda install -c conda-forge rpy2
RUN pip install "TissUUmaps[full]"

# Install facets which does not have a pip or conda package at the moment
WORKDIR /tmp
RUN git clone https://github.com/PAIR-code/facets.git && \
    jupyter nbextension install facets/facets-dist/ --sys-prefix && \
    rm -rf /tmp/facets && \
    fix-permissions "${CONDA_DIR}" && \
    fix-permissions "/home/${NB_USER}"

# Import matplotlib the first time to build the font cache.
ENV XDG_CACHE_HOME="/home/${NB_USER}/.cache/"

RUN MPLBACKEND=Agg python -c "import matplotlib.pyplot" && \
    fix-permissions "/home/${NB_USER}"



WORKDIR "${HOME}"


COPY SAP_helperfunctions_hf.py .
COPY SAP_preprocessing_pp.py .
COPY SAP_tools_tl.py .
COPY SAP_plot_pl.py .
COPY triangulation_distances.R .