# Docker container for efficient iterative testing
# - build with
#   $ docker buildx build -t spacec_test_gpu -f Docker/spacec_test_gpu.dockerfile .
# - run with:
#   $ docker run --gpus all spacec_test_gpu
#   or if iteratively updating tests:
#   $ docker run --gpus all -v .:/app/spacec spacec_test_gpu
FROM spacec_test_cpu

# Install CUDA
RUN mamba install -n spacec -c conda-forge cudatoolkit=11.2.2 cudnn=8.1.0.77 -y

# Setup environment for GPU libraries
ENV CONDA_PREFIX=/miniforge/envs/spacec
RUN mkdir -p $CONDA_PREFIX/etc/conda/activate.d && \
    echo 'export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH' > $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh

# Install RAPIDS (single cell)
RUN mamba run -n spacec \
    pip install -e ".[rapids]" --extra-index-url=https://pypi.nvidia.com

# Install requirements for STELLAR
RUN mamba run -n spacec \
    pip install -e ".[stellar]" \
    --extra-index-url https://download.pytorch.org/whl/cu113 \
    -f https://data.pyg.org/whl/torch-1.12.0+cu113.html

# Clean up
RUN mamba clean --all -f -y && rm -rf /root/.cache/pip

# Default command
# Note: We are running all tests because some require results from previous tests (I think)
CMD ["bash", "-c", "source /miniforge/etc/profile.d/conda.sh && conda activate spacec && pytest -m 'gpu or (not skip and not slow)'"]
