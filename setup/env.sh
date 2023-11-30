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

# Create a new conda environment with Python 3.11
echo "Creating $ENV_NAME environment."
echo ""
conda create -y -n $ENV_NAME python=3.10 pip
echo ""
conda activate $ENV_NAME

# Update pip in the new environment
python -m pip install --upgrade pip

# Detect the Operating System and Check for GPU
case "$(uname -s)" in
    Darwin)
        echo "Mac OS Detected"
        pip install torch torchvision torchaudio
        pip install torch_geometric
    ;;

    Linux)
        echo "Linux OS Detected"
        if has_gpu; then
            echo "Nvidia GPU Detected"
            CUDA_VER=$(get_cuda_version)
            if [ "$CUDA_VER" != "" ]; then
                echo "CUDA Version $CUDA_VER Detected"
                pip install torch torchvision torchaudio #--index-url https://download.pytorch.org/whl/cu118
                pip install torch_geometric
            else
                pip install torch torchvision torchaudio #--index-url https://download.pytorch.org/whl/cu118
                pip install torch_geometric
            fi
        else
            pip install torch torchvision torchaudio #--index-url https://download.pytorch.org/whl/cpu
            pip install torch_geometric
        fi
    ;;

    CYGWIN*|MINGW32*|MSYS*|MINGW*)
        echo "Windows OS Detected"
        if has_gpu; then
            echo "Nvidia GPU Detected"
            pip install torch torchvision torchaudio #--index-url https://download.pytorch.org/whl/cu118
            pip install torch_geometric
        else
            pip install torch torchvision torchaudio
            pip install torch_geometric
        fi
    ;;

    *)
        echo "unknown OS"
    ;;
esac

# # Install cudf Pandas accelerator and related packages
# echo "Installing cudf Pandas accelerator and related packages."
# if has_gpu && [ "$(get_cuda_version)" != "" ] && [ "$(uname -s)" == "Linux" ]; then
#     pip install --extra-index-url=https://pypi.nvidia.com cudf-cu12
# else
#     echo "Skipping cudf installation as no compatible GPU/CUDA version found."
# fi

# Install common packages using pip
echo "Installing common packages using pip."
pip install --force-reinstall --upgrade -r requirements.txt

echo ""
echo "Run conda activate $ENV_NAME to activate the environment."        
echo "Environment setup complete."
echo ""

# Set the repository environment to development mode
cd ..
conda develop .