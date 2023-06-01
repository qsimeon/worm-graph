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

# Install PyTorch with CUDA support
echo "Installing PyTorch with CUDA."
echo ""
### uncomment line below for GPU: 
conda install --name $ENV_NAME -y pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
### uncomment line below if no GPU:
# conda install --name $ENV_NAME -y pytorch torchvision torchaudio cpuonly -c pytorch

# Install PyTorch Geometric
echo "Installing PyTorch Geometric."
echo ""
conda install --name $ENV_NAME -y pyg -c pyg

# Install large, complex dependencies
echo "Installing large, complex dependencies."
echo ""
conda install --name $ENV_NAME -y numpy scipy pandas matplotlib jupyter jupyterlab notebook scikit-learn seaborn

# Install dependencies with moderate complexity
echo "Installing dependencies with moderate complexity."
echo ""
conda install --name $ENV_NAME -y h5py networkx ipython jinja2 typing typing_extensions

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
echo "Run pip install -r requirements.txt to install other package dependencies."
echo "Run conda develop . to install the package in development mode."
echo ""