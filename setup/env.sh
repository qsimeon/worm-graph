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

# Create a new conda environment with Python 3.10
echo ""
echo "Creating $ENV_NAME environment."
echo ""
conda clean -y --packages
conda create -y -n $ENV_NAME python=3.10 conda-build pip
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
        conda install -y -n $ENV_NAME -c pytorch pytorch torchvision torchaudio
        conda install -y -n $ENV_NAME pyg -c pyg
        # # If you prefer to use pip:
        # pip install torch torchvision torchaudio
        # pip install torch_geometric
    ;;

    Linux)
        echo "Linux OS Detected"
        if has_gpu; then
            echo ""
            echo "Nvidia GPU Detected"
            CUDA_VER=$(get_cuda_version)
            if [ "$CUDA_VER" != "" ]; then
                echo "CUDA Version $CUDA_VER Detected"
                # If you prefer to use conda:
                conda install -y -n $ENV_NAME pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
                conda install -y -n $ENV_NAME pyg -c pyg
                # # If you prefer to use pip:
                # pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
                # pip install torch_geometric
            else
                # If you prefer to use conda:
                conda install -y -n $ENV_NAME pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
                conda install -y -n $ENV_NAME pyg -c pyg
                # # If you prefer to use pip:
                # pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
                # pip install torch_geometric
            fi
        else
            # If you prefer to use conda:
            conda install -y -n $ENV_NAME pytorch torchvision torchaudio cpuonly -c pytorch
            conda install -y -n $ENV_NAME pyg -c pyg
            # # If you prefer to use pip:
            # pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
            # pip install torch_geometric
        fi
    ;;

    CYGWIN*|MINGW32*|MSYS*|MINGW*)
        echo ""
        echo "Windows OS Detected"
        if has_gpu; then
            echo "Nvidia GPU Detected"
            # If you prefer to use conda:
            conda install -y -n $ENV_NAME pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
            conda install -y -n $ENV_NAME pyg -c pyg
            # # If you prefer to use pip:
            # pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
            # pip install torch_geometric
        else
            # If you prefer to use conda:
            conda install -y -n $ENV_NAME pytorch torchvision torchaudio cpuonly -c pytorch
            conda install -y -n $ENV_NAME pyg -c pyg
            # # If you prefer to use pip:
            # pip install torch torchvision torchaudio
            # pip install torch_geometric
        fi
    ;;

    *)
        echo ""
        echo "Unknown OS"
    ;;
esac

# Install cudf Pandas accelerator and related packages
echo ""
echo "Installing cudf Pandas accelerator and related packages."
if has_gpu && [ "$(get_cuda_version)" != "" ] && [ "$(uname -s)" == "Linux" ]; then
    # If you prefer to use conda:
    conda install -y -n $ENV_NAME --solver=libmamba -c rapidsai -c nvidia -c conda-forge cudf=23.10 python=3.10 cuda-version=11.8
    # # If you prefer to use pip:
    # pip install --extra-index-url=https://pypi.nvidia.com cudf-cu11
else
    echo ""
    echo "Skipping cudf installation as no compatible GPU/CUDA version found."
fi

# Split requirements into conda and pip specific files
echo ""
echo "Splitting requirements.txt file for conda and pip."
awk '/# uses pip/{print $1 > "requirements_pip.txt"; next} {print > "requirements_conda.txt"}' requirements.txt

# Install common packages using conda
echo ""
echo "Installing common packages using conda."
conda install -y -n $ENV_NAME -c conda-forge --file requirements_conda.txt

# Install packages that require pip
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