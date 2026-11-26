# Build with (from the repo root):
#   $ docker build -f Docker/spacec_gpu_build.dockerfile -t spacec:gpu .
# Run with:
#   $ docker run --gpus all -p 8888:8888 -p 5100:5100 spacec:gpu
ARG PYTHON_VERSION=3.12
FROM python:${PYTHON_VERSION}-slim-bookworm

# git: GitPython (an install_requires dep) clones STELLAR from Git at runtime.
# build-essential/pkg-config: pyvips ships sdist-only, so pip compiles it here;
#   do NOT also install a system libvips, or pyvips builds against the wrong
#   headers and fails at runtime with "cannot load library libvips.42".
# libglib2.0-0: needed by cv2 (pulled in transitively via cellpose).
# fontconfig/fonts-*: needed by matplotlib for text rendering.
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    pkg-config \
    git \
    libglib2.0-0 \
    fontconfig \
    fonts-dejavu-core \
    fonts-droid-fallback \
    fonts-liberation \
    fonts-noto-mono \
    ttf-bitstream-vera && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir spacec

# The PyTorch wheels ship their own CUDA runtime and cuDNN, so no system
# cudatoolkit/cudnn install is needed. Reinstall torch against a specific CUDA
# build; bump TORCH_CUDA to track a newer CUDA release. torch<2.6 matches the
# ceiling pinned in setup.cfg's install_requires.
ARG TORCH_CUDA=cu124
RUN pip install --no-cache-dir --force-reinstall \
    "torch<2.6" torchvision --index-url https://download.pytorch.org/whl/${TORCH_CUDA}

# RAPIDS for GPU-accelerated clustering (sp.tl.clustering(..., clustering="leiden_gpu")).
# Uses the rapids12 extra already declared in setup.cfg; pick rapids13 instead
# if the host's CUDA major version is 13.
RUN pip install --no-cache-dir "spacec[rapids12]" --extra-index-url=https://pypi.nvidia.com

# STELLAR is not installed here: setup.cfg's [stellar] extra still pins
# torch==1.12.0+cu113, which conflicts with cellpose>=3 and the torch<2.6
# floor above. Call sp.tl.install_stellar(CUDA=12) at runtime instead, which
# installs a compatible modern torch/PyG stack.

# Copy notebooks into the image. YOU CAN CHANGE THIS TO COPY YOUR OWN NOTEBOOKS
COPY notebooks /notebooks
WORKDIR /notebooks

# Expose relevant ports
EXPOSE 8888
EXPOSE 5100

# Default command. No auth token/password: intended for localhost port-forwarding only.
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--ServerApp.token=", "--ServerApp.password="]
