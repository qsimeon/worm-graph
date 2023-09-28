#!/bin/bash

# Initialize conda for bash shell
eval "$(conda shell.bash hook)"

function has_gpu {
    command -v nvidia-smi > /dev/null && nvidia-smi > /dev/null 2>&1
}

ENV_NAME="worm-graph"

# Create a new conda environment with Python 3.9
echo "Creating $ENV_NAME environment."
echo ""
conda create -y -n $ENV_NAME python=3.9
echo ""
conda activate $ENV_NAME

python -m pip install --upgrade pip

# Install PyTorch
echo "Installing PyTorch."
echo ""
# Detect the Operating System and Check for GPU
case "$(uname -s)" in
    Darwin)
        echo "Mac OS Detected"
        # Macs typically do not have Nvidia GPUs, but you could still
        # use the has_gpu function here if you wanted to.
        conda install --name $ENV_NAME -y pytorch::pytorch torchvision torchaudio -c pytorch
    ;;

    Linux)
        echo "Linux OS Detected"
        if has_gpu; then
            echo "Nvidia GPU Detected"
            conda install --name $ENV_NAME -y pytorch torchvision torchaudio -c pytorch
        else
            conda install --name $ENV_NAME -y pytorch torchvision torchaudio cpuonly -c pytorch
        fi
    ;;

    CYGWIN*|MINGW32*|MSYS*|MINGW*)
        echo "Windows OS Detected"
        if has_gpu; then
            echo "Nvidia GPU Detected"
            conda install --name $ENV_NAME -y pytorch torchvision torchaudio -c pytorch
        else
            conda install --name $ENV_NAME -y pytorch torchvision torchaudio cpuonly -c pytorch
        fi
    ;;

    *)
        echo "unknown OS"
    ;;
esac

# Install large, complex dependencies
echo "Installing large, complex dependencies."
echo ""
conda install --name $ENV_NAME -y numpy matplotlib scikit-learn scipy 
conda install --name $ENV_NAME -y pandas seaborn dtaidistance -c conda-forge
cond install --name $ENV_NAME -y jupyter jupyterlab notebook 

# Install dependencies with moderate complexity
echo "Installing dependencies with moderate complexity."
echo ""
conda install --name $ENV_NAME -y h5py networkx ipython jinja2 spectrum typing typing_extensions

# Install small, simple dependencies
echo "Installing small, simple dependencies."
echo ""
conda install --name $ENV_NAME -y pickleshare requests tqdm yaml conda-build 

# Install code formatting and linting tools
echo "Installing code formatting and linting tools."
echo ""
conda install --name $ENV_NAME -y black black[jupyter]

echo ""
echo "Run conda activate $ENV_NAME to activate the environment."
echo "Run pip install --upgrade -r requirements.txt to install other package dependencies."
echo "Run conda develop . to install the package in development mode."
echo ""