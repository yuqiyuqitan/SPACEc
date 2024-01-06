ARG BASE_IMAGE=ubuntu:22.04
FROM ${BASE_IMAGE}

# install packages 
RUN apt-get -y update --allow-releaseinfo-change\
    && apt-get -y autoremove \
    && apt-get clean \
    && apt-get install -y \
    fonts-liberation \
    locales \
    wget curl \
    unzip \    
    # - bzip2 is necessary to extract the micromamba executable.
    bzip2 \
    # - pandoc is used to convert notebooks to html files
    #   it's not present in aarch64 ubuntu image, so we install it here
    pandoc \
    git \
    vim \
    zsh \
    tmux \
    build-essential \
    gcc \
    # clean up
    && rm -fr /var/lib/apt/lists/* /tmp/* /var/tmp/* \
    && echo "en_US.UTF-8 UTF-8" > /etc/locale.gen \
    && locale-gen

# TODO: not really sure why I need this
ENV LC_ALL=en_US.UTF-8 \
    LANG=en_US.UTF-8 \
    LANGUAGE=en_US.UTF-8

# install and setup micromamba
ENV MAMBA_ROOT_PREFIX=/opt/mamba
RUN curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xvj bin/micromamba \
   && /bin/micromamba shell init -s zsh -p ${MAMBA_ROOT_PREFIX} \
   && echo "alias mamba=micromamba" >> ~/.zshrc
ENV MAMBA_EXEC=/bin/micromamba

ARG ENV_NAME="SPACEc"
RUN ${MAMBA_EXEC} env create -n ${ENV_NAME} -c conda-forge python=3.10 jupyterlab

# install python packages
ARG GITHUB_USER
ARG GITHUB_TOKEN
RUN ${MAMBA_EXEC} run -n ${ENV_NAME} pip install --no-cache-dir \
    git+https://${GITHUB_USER}:${GITHUB_TOKEN}@github.com/yuqiyuqitan/SAP.git@preppip


# download tutorial files and run jupyterlab on startup

ENV ENV_NAME=${ENV_NAME}
EXPOSE 8888

# attempt to fix situation by running with -u $UID
# but then new files end up with weird permissions: drwxr-Sr--+
# probably could be done easier with some MAMBA path change
#RUN \
#  mkdir -p /.cache/mamba; chmod -R 777 /.cache; \
#  mkdir /.local; chmod 777 /.local; \
#  mkdir /.jupyter; chmod 777 /.jupyter;

WORKDIR /workspace
COPY tutorial_init.sh /
CMD ["bash", "/tutorial_init.sh"]
