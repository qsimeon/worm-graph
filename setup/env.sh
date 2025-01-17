#!/bin/bash

# Get the absolute path of the script's directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
echo ""
echo "env.sh called from $SCRIPT_DIR"

# Initialize conda for bash shell
eval "$(conda shell.bash hook)"

# Function to detect if running on a SLURM cluster
function is_slurm_cluster {
    if command -v sinfo >/dev/null 2>&1 || [ -n "$SLURM_JOB_ID" ]; then
        return 0  # True (Slurm cluster)
    else
        return 1  # False (not Slurm)
    fi
}

# Debug: Display if on a SLURM cluster
if is_slurm_cluster; then
    echo ""
    echo "Slurm cluster detected."
else
    echo ""
    echo "Not on a Slurm cluster."
fi

# Debug: Display the detected OS
echo ""
echo "Detected OS: $OSTYPE"

# Function to check for Nvidia GPU
function has_gpu {
    if command -v nvidia-smi > /dev/null && nvidia-smi > /dev/null 2>&1; then
        cuda_version=$(nvidia-smi | grep -oP '(?<=CUDA Version: )[\d\.]+')  # Extract CUDA version
        return 0  # GPU available
    else
        return 1  # No GPU available
    fi
}

# Create a new conda environment with Python 3.12
ENV_NAME="worm-graph"
echo ""
echo "Creating $ENV_NAME environment."
conda create -y -n $ENV_NAME python=3.12 conda-build

# # Activate the environment
# if ! conda activate $ENV_NAME; then
#     echo ""
#     echo "Failed to activate $ENV_NAME environment. Exiting."
#     exit 1
# fi

# # Update Python and pip in the new environment
# conda update -y python
# python -m pip install --upgrade pip

# # Adjust configs/pipeline.yaml based on SLURM cluster detection
# CONFIG_FILE="$SCRIPT_DIR/configs/pipeline.yaml"
# if ! is_slurm_cluster; then
#     echo ""
#     echo "Not on a Slurm cluster. Adjusting configs/pipeline.yaml for local execution."
#     if [[ "$OSTYPE" == "darwin"* ]]; then
#         sed -i '' 's/submitit_slurm/submitit_local/g' "$CONFIG_FILE"
#     else
#         sed -i 's/submitit_slurm/submitit_local/g' "$CONFIG_FILE"
#     fi
# else
#     echo ""
#     echo "On a SLURM computing cluster. Keeping configs/pipeline.yaml for remote execution."
# fi

# # OS Detection and Environment Setup
# echo ""
# echo "Starting environment setup ..."

# case "$(uname -s)" in
#     Darwin)
#         echo "Mac OS Detected"
#         conda install -y -n $ENV_NAME pytorch torchvision torchaudio cpuonly -c pytorch -c conda-forge
#         ;;
#     Linux)
#         echo "Linux OS Detected"
#         if has_gpu; then
#             echo "Nvidia GPU Detected with CUDA version $cuda_version"
#             conda install -y -n $ENV_NAME pytorch torchvision torchaudio pytorch-cuda=$cuda_version -c pytorch -c nvidia
#         else
#             echo "No GPU detected. Installing CPU-only PyTorch."
#             conda install -y -n $ENV_NAME pytorch torchvision torchaudio cpuonly -c pytorch
#         fi
#         ;;
#     CYGWIN*|MINGW32*|MSYS*|MINGW*)
#         echo "Windows OS Detected"
#         conda install -y -n $ENV_NAME pytorch torchvision torchaudio cpuonly -c pytorch
#         ;;
#     *)
#         echo "Unknown OS Detected. Exiting."
#         exit 1
#         ;;
# esac

# Splitting requirements.txt for conda and pip installations
REQUIREMENTS_FILE="$SCRIPT_DIR/setup/requirements.txt"
if [ -f "$REQUIREMENTS_FILE" ]; then
    awk '/# uses pip/{print > "'"$SCRIPT_DIR"'/setup/requirements_pip.txt"; next} {print > "'"$SCRIPT_DIR"'/setup/requirements_conda.txt"}' "$REQUIREMENTS_FILE"
else
    echo "requirements.txt not found in $SCRIPT_DIR. Exiting."
    exit 1
fi

# # Install common packages using conda
# echo ""
# echo "Installing packages with conda."
# conda install -y -n $ENV_NAME -c conda-forge --file "$SCRIPT_DIR/requirements_conda.txt"

# # Install pip-specific packages
# echo ""
# echo "Installing pip-specific packages."
# python -m pip install -r "$SCRIPT_DIR/requirements_pip.txt"

# echo ""
# echo "Environment setup complete. Run 'conda activate $ENV_NAME' to use the environment."
