#!/bin/bash

# Initialize conda for bash shell
eval "$(conda shell.bash hook)"

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
# Detect the Operating System
case "$(uname -s)" in
    Darwin)
        echo "Mac OS Detected"
        # Command for Mac OS
        conda install --name $ENV_NAME -y pytorch::pytorch torchvision torchaudio -c pytorch
    ;;

    Linux)
        echo "Linux OS Detected"
        # Command for Linux
        conda install --name $ENV_NAME -y pytorch torchvision torchaudio cpuonly -c pytorch
    ;;

    CYGWIN*|MINGW32*|MSYS*|MINGW*)
        echo "Windows OS Detected"
        # Command for Windows
        conda install --name $ENV_NAME -y pytorch torchvision torchaudio cpuonly -c pytorch
    ;;

    *)
        echo "unknown OS"
        # Handle AmigaOS, CPM, and others if required.
    ;;
esac

# Install large, complex dependencies
echo "Installing large, complex dependencies."
echo ""
conda install --name $ENV_NAME -y numpy scipy pandas matplotlib scikit-learn seaborn
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
conda install --name $ENV_NAME -y black "black[jupyter]"

echo ""
echo "Run conda activate $ENV_NAME to activate the environment."
echo "Run pip install --upgrade -r requirements.txt to install other package dependencies."
echo "Run conda develop . to install the package in development mode."
echo ""