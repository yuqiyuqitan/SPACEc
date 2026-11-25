# Docker container for efficient iterative testing
# - build with (from the repo root):
#   $ docker build -f Docker/spacec_test_cpu.dockerfile -t spacec_test_cpu .
# - run with:
#   $ docker run spacec_test_cpu
#   or if iteratively updating tests:
#   $ docker run -v .:/app/spacec spacec_test_cpu
ARG PYTHON_VERSION=3.12
FROM python:${PYTHON_VERSION}-slim-bookworm

# git: GitPython (an install_requires dep) clones STELLAR from Git at runtime;
#   `git init` below also gives setuptools_scm a repo to read the version from.
# build-essential/pkg-config: pyvips ships sdist-only, so pip compiles it here;
#   do NOT also install a system libvips, or pyvips builds against the wrong
#   headers and fails at runtime with "cannot load library libvips.42".
# libglib2.0-0: needed by cv2 (pulled in transitively via cellpose).
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    pkg-config \
    git \
    libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

# Copy SPACEc development code and install with test extras
COPY . /app/spacec
WORKDIR /app/spacec
RUN git init && \
    pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -e ".[testing]"

# Default command
CMD ["pytest"]
