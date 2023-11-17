#!/bin/bash

# Initialize conda for bash shell
eval "$(conda shell.bash hook)"

function has_gpu {
    command -v nvidia-smi > /dev/null && nvidia-smi > /dev/null 2>&1
}

function get_cuda_version {
    if ! type "nvidia-smi" > /dev/null; then
        # CUDA is not installed; return an empty string
        echo ""
    else
        # Extract the CUDA version from the nvidia-smi output
        local cuda_version
        cuda_version=$(nvidia-smi | grep -oP '(?<=CUDA Version: )\d+\.\d+')
        echo "$cuda_version"
    fi
}

ENV_NAME="worm-graph"

# Create a new conda environment with Python 3.9
echo "Creating $ENV_NAME environment."
echo ""
conda create -y -n $ENV_NAME python=3.9 pip
echo ""
conda activate $ENV_NAME

# Update pip in the new environment
python -m pip install --upgrade pip

# Install PyTorch and Pytorch Geometric
echo "Installing PyTorch and PyTorch Geometric."
echo ""
# Detect the Operating System and Check for GPU
case "$(uname -s)" in
    Darwin)
        echo "Mac OS Detected"
        # Macs typically do not have Nvidia GPUs
        pip install torch torchvision torchaudio
        # conda install --name $ENV_NAME -y pytorch::pytorch torchvision torchaudio -c pytorch
        # conda install --name $ENV_NAME -y pyg -c pyg
        pip install torch_geometric
    ;;

    Linux)
        echo "Linux OS Detected"
        if has_gpu; then
            echo "Nvidia GPU Detected"
            CUDA_VER=$(get_cuda_version)
            if [ "$CUDA_VER" != "" ]; then
                echo "CUDA Version $CUDA_VER Detected"
                pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
                # conda install --name $ENV_NAME -y pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.8 -c pytorch -c nvidia
                # conda install --name $ENV_NAME -y pytorch torchvision torchaudio pytorch-cuda=$CUDA_VER -c pytorch -nvidia
                # conda install --name $ENV_NAME -y pyg -c pyg # Add this line for PyG with GPU support
                pip install torch_geometric
            else
                pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
                # conda install --name $ENV_NAME -y pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 cpuonly -c pytorch
                # conda install --name $ENV_NAME -y pytorch torchvision torchaudio -c pytorch
                # conda install --name $ENV_NAME -y pyg -c pyg # Add this line for PyG with CPU support
                pip install torch_geometric
            fi
        else
            pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
            # conda install --name $ENV_NAME -y pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 cpuonly -c pytorch
            # conda install --name $ENV_NAME -y pytorch torchvision torchaudio cpuonly -c pytorch
            # conda install --name $ENV_NAME -y pyg -c pyg # Add this line for PyG with CPU support
            pip install torch_geometric
        fi
    ;;

    CYGWIN*|MINGW32*|MSYS*|MINGW*)
        echo "Windows OS Detected"
        if has_gpu; then
            echo "Nvidia GPU Detected"
            pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
            # conda install --name $ENV_NAME -y pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.8 -c pytorch -c nvidia
            # conda install --name $ENV_NAME -y pytorch torchvision torchaudio pytorch-cuda -c pytorch -c nvidia
            # conda install --name $ENV_NAME -y pyg -c pyg # Add this line for PyG with GPU support
            pip install torch_geometric
        else
            pip install torch torchvision torchaudio
            # conda install --name $ENV_NAME -y pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 cpuonly -c pytorch
            # conda install --name $ENV_NAME -y pytorch torchvision torchaudio cpuonly -c pytorch
            # conda install --name $ENV_NAME -y pyg -c pyg # Add this line for PyG with CPU support
            pip install torch_geometric
        fi
    ;;

    *)
        echo "unknown OS"
    ;;
esac

# Install packages from conda-forge
echo "Installing packages from conda-forge."
conda install --name $ENV_NAME -c defaults -c conda-forge -y \
    numpy \
    matplotlib \
    scikit-learn \
    scipy \
    pandas \
    seaborn \
    dtaidistance \
    fvcore \
    ipykernel \
    jupyter \
    jupyterlab \
    notebook \
    h5py \
    networkx \
    ipython \
    jinja2 \
    spectrum \
    typing \
    typing_extensions \
    pickleshare \
    pytest \
    requests \
    tqdm \
    yaml \
    conda-build \
    black \
    hydra-core \
    hydra-submitit-launcher \
    mat73 \
    omegaconf \
    statsmodels 

# Install any packages that are not available on conda-forge using pip
echo ""
echo "Run conda activate $ENV_NAME to activate the environment."
echo "Run pip install --upgrade -r requirements.txt to install other package dependencies if any."
echo "Run conda develop . to install the package in development mode."
echo ""

echo "Installing any remaining packages using pip."
conda activate $ENV_NAME
pip install --upgrade -r requirements.txt
cd ..
conda develop .

