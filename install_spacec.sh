#!/bin/bash

# Function to prompt user for package manager choice
choose_package_manager() {
    read -p "Choose package manager (mamba/conda) [conda]: " pkg_manager
    pkg_manager=${pkg_manager:-conda}
    if [[ "$pkg_manager" != "mamba" && "$pkg_manager" != "conda" ]]; then
        echo "Invalid choice. Defaulting to conda."
        pkg_manager="conda"
    fi
    echo "Using $pkg_manager"
}

# Choose package manager
choose_package_manager

# Create and activate conda environment
$pkg_manager create -n spacec python=3.10 -y
source activate spacec

if [[ "$(uname -s)" == *"MINGW"* || "$(uname -s)" == *"CYGWIN"* ]]; then
    mkdir -p $CONDA_PREFIX/etc/conda/activate.d
    # Windows specific environment variable setup
    echo "Running on Windows"
    vipsbin="c:/vips-dev-8.15/bin"
    vips_file_path="$vipsbin/vips.exe"

    if [ ! -f "$vips_file_path" ]; then
        # VIPS is not installed, download and extract it
        echo "Downloading VIPS..."
        url="https://github.com/libvips/build-win64-mxe/releases/download/v8.15.2/vips-dev-w64-all-8.15.2.zip"
        zip_file_path="vips-dev-w64-all-8.15.2.zip"
        curl -L -o "$zip_file_path" "$url"
        unzip "$zip_file_path" -d "c:/vips-dev-8.15"
    fi

    echo 'export PATH=$CONDA_PREFIX/bin:$PATH' > $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
    echo 'export PATH=c:/vips-dev-8.15/bin:$PATH' >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
    chmod +x $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
fi

read -p "Do you want to install the GPU version of SPACEc? (y/n): " confirm
if [ "$confirm" = "y" ]; then
    nvidia-smi
    # Install packages
    $pkg_manager install conda-forge::cudatoolkit=11.2.2 -y
    $pkg_manager install conda-forge::cudnn=8.1.0.77 -y
    if [[ "$(uname -s)" == *"MINGW"* || "$(uname -s)" == *"CYGWIN"* ]]; then
        $pkg_manager install -c conda-forge graphviz -y

        # Set environment variables
        mkdir -p $CONDA_PREFIX/etc/conda/activate.d
        echo 'export PATH=$CONDA_PREFIX/bin:$PATH' > $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
        echo 'export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH' >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
        chmod +x $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh

        # Install Python packages
        pip install --upgrade pip
        pip install git+https://github.com/yuqiyuqitan/SPACEc.git@combined_updates_tim
        pip install pandas==1.*
        pip install protobuf==3.20.0
        pip install numpy==1.24.*
        pip install tensorflow-gpu==2.8.0

        echo "Installation complete. Please restart the terminal to apply the changes."

    else
        $pkg_manager install -c conda-forge graphviz libvips pyvips openslide-python -y

        # Set environment variables
        mkdir -p $CONDA_PREFIX/etc/conda/activate.d
        echo 'export PATH=$CONDA_PREFIX/bin:$PATH' > $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
        echo 'export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH' >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
        chmod +x $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh

        # Install Python packages
        pip install --upgrade pip
        pip install git+https://github.com/yuqiyuqitan/SPACEc.git@combined_updates_tim
        pip install pandas==1.*
        pip install protobuf==3.20.0
        pip install numpy==1.24.*
        pip install tensorflow-gpu==2.8.0

        # Install RAPIDS
        read -p "Do you want to install the RAPIDS? NVIDIA RTX20XX or better required! (y/n): " confirm
        if [ "$confirm" = "y" ]; then
            $pkg_manager install -c rapidsai -c conda-forge -c nvidia rapids=24.02 python=3.10 cuda-version=11.2 -y
            pip install rapids-singlecell==0.9.5
            $pkg_manager install pandas -y
        else
            echo "RAPIDS installation aborted by the user."
        fi

        echo "Installation complete. Please restart the terminal to apply the changes."
    fi
else
    if [[ "$(uname)" == "Darwin" ]]; then
        echo "Configure SPACEc for macOS"
        conda config --env --set subdir osx-64
        $pkg_manager install -c conda-forge graphviz libvips pyvips openslide-python -y
        # Install Python packages
        pip install --upgrade pip
        pip install git+https://github.com/yuqiyuqitan/SPACEc.git@combined_updates_tim
        $pkg_manager install tensorflow=2.10.0
        pip uninstall werkzeug -y
        pip install numpy==1.26.4 werkzeug==2.3.8
    else
        # Check if running on Windows
        if [[ "$(uname -s)" == *"MINGW"* || "$(uname -s)" == *"CYGWIN"* ]]; then
            $pkg_manager install -c conda-forge graphviz -y
        else
            echo "Running on Linux"
            $pkg_manager install -c conda-forge graphviz libvips pyvips openslide-python -y
            pip install --upgrade pip
            pip install git+https://github.com/yuqiyuqitan/SPACEc.git@combined_updates_tim
            pip install pandas==1.*
            pip install protobuf==3.20.0
            pip install numpy==1.24.*
        fi
    fi

    echo "Installation complete. Please restart the terminal to apply the changes."
fi