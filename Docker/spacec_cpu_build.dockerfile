# Build with (from the repo root):
#   $ docker build -f Docker/spacec_cpu_build.dockerfile -t spacec:cpu .
# Run with:
#   $ docker run -p 8888:8888 -p 5100:5100 spacec:cpu
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

# Copy notebooks into the image. YOU CAN CHANGE THIS TO COPY YOUR OWN NOTEBOOKS
COPY notebooks /notebooks
WORKDIR /notebooks

# Expose relevant ports
EXPOSE 8888
EXPOSE 5100

# Default command. No auth token/password: intended for localhost port-forwarding only.
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--ServerApp.token=", "--ServerApp.password="]
