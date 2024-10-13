#!/bin/bash

# Get the absolute path of the script's directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
echo ""
echo "env.sh called from $SCRIPT_DIR"

# Initialize conda for bash shell
eval "$(conda shell.bash hook)"

# Function to detect if running on a Slurm cluster
function is_slurm_cluster {
    if command -v sinfo >/dev/null 2>&1 || [ -n "$SLURM_JOB_ID" ]; then
        return 0  # True (Slurm cluster)
    else
        return 1  # False (not Slurm)
    fi
}

# Debug: Display if on a Slurm cluster
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
        cuda_version=$(nvidia-smi | grep -oP '(?<=CUDA Version: )[\d\.]+')  # Now accessible globally
        return 0 # Indicates that a GPU is available (exit status 0 means success)
    else
        return 1 # Indicates that no GPU is available (exit status 1 means failure)
    fi
}

# Create a new conda environment with Python 3.12
ENV_NAME="worm-graph"
echo ""
echo "Creating $ENV_NAME environment."
# conda clean -y --packages
conda create -y -n $ENV_NAME python=3.12 conda-build


# Check if the environment was successfully created and activated
if ! conda activate $ENV_NAME; then
    if ! source activate $ENV_NAME; then
        echo ""
        echo "Failed to activate $ENV_NAME environment. Exiting."
        exit 1
    fi
fi

# Update to the latest version of Python
conda update -y -n $ENV_NAME python

# Update pip in the new environment
python -m pip install --upgrade pip

# Check if running on a Slurm cluster and adjust configs/pipeline.yaml accordingly
if ! is_slurm_cluster; then
    echo ""
    echo "Not on a Slurm cluster. Adjusting configs/pipeline.yaml for local execution."
    
    if [[ "$OSTYPE" == "darwin"* ]]; then
        echo "Modifying pipeline.yaml for MacOS"
        sed -i '' 's/submitit_slurm/submitit_local/g' "$SCRIPT_DIR/configs/pipeline.yaml"
    else
        echo "Modifying pipeline.yaml for Linux"
        sed -i 's/submitit_slurm/submitit_local/g' "$SCRIPT_DIR/configs/pipeline.yaml"
    fi
else
    echo ""
    echo "On a SLURM computing cluster. Keeping configs/pipeline.yaml for remote execution."
fi

# Operating System Detection and Environment Setup
echo ""
echo "Starting environment setup ..."

# Detect the Operating System and Check for GPU
case "$(uname -s)" in
    Darwin)
        echo ""
        echo "Mac OS Detected"
        conda install pytorch::pytorch torchvision torchaudio -c pytorch
        # conda install -y -n $ENV_NAME pyg -c pyg
        ;;

    Linux)
        echo ""
        echo "Linux OS Detected"
        if has_gpu; then
            echo ""
            echo "Nvidia GPU Detected with CUDA version $cuda_version"
            conda install -y -n $ENV_NAME pytorch torchvision torchaudio pytorch-cuda=$cuda_version -c pytorch -c nvidia
            # conda install -y -n $ENV_NAME pyg -c pyg
        else
            echo ""
            conda install -y -n $ENV_NAME pytorch torchvision torchaudio cpuonly -c pytorch
            # conda install -y -n $ENV_NAME pyg -c pyg
        fi
        ;;

    CYGWIN*|MINGW32*|MSYS*|MINGW*)
        echo ""
        echo "Windows OS Detected"
        if has_gpu; then
            echo ""
            echo "Nvidia GPU Detected with CUDA version $cuda_version"
            conda install -y -n $ENV_NAME pytorch torchvision torchaudio pytorch-cuda=$cuda_version -c pytorch -c nvidia
            # conda install -y -n $ENV_NAME pyg -c pyg
        else
            echo ""
            conda install -y -n $ENV_NAME pytorch torchvision torchaudio cpuonly -c pytorch
            # conda install -y -n $ENV_NAME pyg -c pyg
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
REQUIREMENTS_FILE="$SCRIPT_DIR/requirements.txt"

# Check if requirements.txt exists
if [ -f "$REQUIREMENTS_FILE" ]; then
    awk '/# uses pip/{print $1 > "'"$SCRIPT_DIR"'/requirements_pip.txt"; next} {print > "'"$SCRIPT_DIR"'/requirements_conda.txt"}' "$REQUIREMENTS_FILE"
else
    echo ""
    echo "requirements.txt not found in $SCRIPT_DIR. Exiting."
    exit 1
fi

# Install common packages using conda
echo ""
echo "Installing common packages using conda."
conda install -y -n $ENV_NAME -c conda-forge --file "$SCRIPT_DIR/requirements_conda.txt"

# Install packages that require pip (after conda to avoid conflicts)
echo ""
echo "Installing pip-specific packages."
python -m pip install -r "$SCRIPT_DIR/requirements_pip.txt"

echo ""
echo "Run conda activate $ENV_NAME to activate the environment."

# Set the repository environment to development mode
cd "$SCRIPT_DIR"/..
conda develop .

echo ""
echo "Environment setup complete."

# Navigate to the root of the repo
cd "$SCRIPT_DIR"/..
