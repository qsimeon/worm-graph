#!/bin/bash

# Initialize conda for bash shell
eval "$(conda shell.bash hook)"

function has_gpu {
    command -v nvidia-smi > /dev/null && nvidia-smi > /dev/null 2>&1
}

ENV_NAME="worm-graph"

# Create a new conda environment with Python 3.11
echo ""
echo "Creating $ENV_NAME environment."
echo ""
conda clean -y --packages
conda create -y -n $ENV_NAME python=3.11 conda-build pip
echo ""
conda activate $ENV_NAME

# Update pip in the new environment
python -m pip install --upgrade pip

# Detect the Operating System and Check for GPU
case "$(uname -s)" in
    Darwin)
        echo ""
        echo "Mac OS Detected"
        # If you prefer to use conda:
        conda install -y -n $ENV_NAME pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 -c pytorch
        conda install -y -n $ENV_NAME pyg -c pyg
        # # If you prefer to use pip:
        # pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2
        # pip install torch_geometric
    ;;

    Linux)
        echo "Linux OS Detected"
        if has_gpu; then
            echo ""
            echo "Nvidia GPU Detected"
            # If you prefer to use conda:
            conda install -y -n $ENV_NAME pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.8 -c pytorch -c nvidia
            conda install -y -n $ENV_NAME pyg -c pyg
            # # If you prefer to use pip:
            # pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2
            # pip install torch_geometric
        else
            # If you prefer to use conda:
            conda install -y -n $ENV_NAME pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 cpuonly -c pytorch
            conda install -y -n $ENV_NAME pyg -c pyg
            # # If you prefer to use pip:
            # pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cpu
            # pip install torch_geometric
        fi
    ;;

    CYGWIN*|MINGW32*|MSYS*|MINGW*)
        echo ""
        echo "Windows OS Detected"
        if has_gpu; then
            echo "Nvidia GPU Detected"
            # If you prefer to use conda:
            conda install -y -n $ENV_NAME pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.8 -c pytorch -c nvidia
            conda install -y -n $ENV_NAME pyg -c pyg
            # # If you prefer to use pip:
            # pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
            # pip install torch_geometric
        else
            # If you prefer to use conda:
            conda install -y -n $ENV_NAME pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 cpuonly -c pytorch
            conda install -y -n $ENV_NAME pyg -c pyg
            # # If you prefer to use pip:
            # pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cpu
            # pip install torch_geometric
        fi
    ;;

    *)
        echo ""
        echo "Unknown OS"
    ;;
esac

# Split requirements into conda and pip specific files
echo ""
echo "Splitting requirements.txt file for conda and pip."
awk '/# uses pip/{print $1 > "requirements_pip.txt"; next} {print > "requirements_conda.txt"}' requirements.txt

# Install common packages using conda
echo ""
echo "Installing common packages using conda."
conda install -y -n $ENV_NAME -c conda-forge --file requirements_conda.txt

# Install packages that require pip (after conda to avoid conflicts)
echo ""
echo "Installing pip-specific packages."
pip install -r requirements_pip.txt

echo ""
echo "Run conda activate $ENV_NAME to activate the environment."

# Set the repository environment to development mode
cd ..
conda develop .

echo ""
echo "Environment setup complete."