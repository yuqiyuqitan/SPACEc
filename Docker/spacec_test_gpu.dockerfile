# Docker container for efficient iterative testing
# - build the CPU test image first (Docker/spacec_test_cpu.dockerfile), then:
#   $ docker build -f Docker/spacec_test_gpu.dockerfile -t spacec_test_gpu .
# - run with:
#   $ docker run --gpus all spacec_test_gpu
#   or if iteratively updating tests:
#   $ docker run --gpus all -v .:/app/spacec spacec_test_gpu
FROM spacec_test_cpu

# The PyTorch wheels ship their own CUDA runtime and cuDNN, so no system
# cudatoolkit/cudnn install is needed. Reinstall torch against a specific CUDA
# build; bump TORCH_CUDA to track a newer CUDA release. torch<2.6 matches the
# ceiling pinned in setup.cfg's install_requires.
ARG TORCH_CUDA=cu124
RUN pip install --no-cache-dir --force-reinstall \
    "torch<2.6" torchvision --index-url https://download.pytorch.org/whl/${TORCH_CUDA}

# RAPIDS for GPU-accelerated clustering. Uses the rapids12 extra already
# declared in setup.cfg; pick rapids13 instead if the host's CUDA major
# version is 13.
RUN pip install --no-cache-dir -e ".[rapids12]" --extra-index-url=https://pypi.nvidia.com

# STELLAR is not installed here: setup.cfg's [stellar] extra still pins
# torch==1.12.0+cu113, which conflicts with cellpose>=3 and the torch<2.6
# floor above, so `pip install -e ".[stellar]"` is unsatisfiable today. Call
# sp.tl.install_stellar(CUDA=12) at runtime instead, which installs a
# compatible modern torch/PyG stack.

# Default command
# Note: We are running all tests because some require results from previous tests (I think)
CMD ["pytest", "-m", "gpu or (not skip and not slow)"]
