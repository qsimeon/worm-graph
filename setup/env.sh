
#!/bin/bash

ENV_NAME="worm-graph"

# Create a new conda environment with Python 3.9
echo "Creating $ENV_NAME environment."
echo ""
conda create -y -n $ENV_NAME python=3.9

# Install PyTorch with CUDA support
echo "Installing PyTorch with CUDA."
echo ""
#### uncomment line below for GPU: 
# conda install --name $ENV_NAME -y pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
### uncomment line below if no GPU:
conda install --name $ENV_NAME -y pytorch torchvision torchaudio -c pytorch -c nvidia

# Install PyTorch Geometric
echo "Installing PyTorch Geometric."
echo ""
conda install --name $ENV_NAME -y pyg -c pyg

# Install additional dependencies from a separate .yml file
echo "Installing other dependencies."
echo ""
conda install --name $ENV_NAME -y h5py ipython jinja2 networkx matplotlib numpy
conda install --name $ENV_NAME -y pandas pickleshare requests seaborn
conda install --name $ENV_NAME -y scipy tqdm yaml conda-build typing typing_extensions
conda install --name $ENV_NAME -y jupyter
conda install --name $ENV_NAME -y black scikit-learn scikit-learn-intelex

echo ""
echo "Run conda activate $ENV_NAME to activate the environment."
echo "Run pip install -r requirements.txt to install other package dependencies."
echo "Run conda develop . to install the package in development mode."
echo ""
