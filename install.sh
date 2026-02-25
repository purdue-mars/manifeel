#!/bin/bash

set -e  # Exit on any error

echo "=========================================="
echo "ManiFeel Installation Script"
echo "=========================================="

# Check if conda or mamba is available
if command -v mamba &> /dev/null; then
    CONDA_CMD="mamba"
    echo "✓ Found mamba"
elif command -v conda &> /dev/null; then
    CONDA_CMD="conda"
    echo "✓ Found conda"
else
    echo "⚠ conda/mamba not found. Installing Miniforge3..."
    
    # Download Miniforge
    wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
    
    # Install Miniforge
    bash Miniforge3-Linux-x86_64.sh -b -p $HOME/miniforge3
    
    # Initialize conda
    $HOME/miniforge3/bin/conda init bash
    
    # Source bashrc to make conda available
    source ~/.bashrc
    
    CONDA_CMD="$HOME/miniforge3/bin/conda"
    
    echo "✓ Miniforge3 installed successfully"
    echo "Please restart your terminal or run: source ~/.bashrc"
fi

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PARENT_DIR="$(dirname "$SCRIPT_DIR")"

# Get conda base path early so we can use it as the default env path
CONDA_BASE="$("$CONDA_CMD" info --base 2>/dev/null | tail -n 1 | awk '{print $NF}')"

# Function to prompt the user for the Miniforge/conda home directory.
# The environment is always named "manifeel" and will be created at
# <miniforge_home>/envs/manifeel. Press Enter to accept the detected default.
# In CI mode (CI=true), the default is used automatically without prompting.
get_conda_env_path() {
    local default_home="$CONDA_BASE"

    echo ""
    echo "=========================================="
    echo "Conda Environment Path"
    echo "=========================================="
    echo "Detected Miniforge/conda home: $default_home"

    if [ "${CI:-false}" = "true" ]; then
        MINIFORGE_HOME="$default_home"
        echo "CI mode: using default Miniforge home"
    else
        read -rp "Enter Miniforge home directory (press Enter to use default): " user_home
        if [ -z "$user_home" ]; then
            MINIFORGE_HOME="$default_home"
        else
            MINIFORGE_HOME="$user_home"
        fi
    fi

    CONDA_ENV_PATH="$MINIFORGE_HOME/envs/manifeel"
    echo "✓ Environment will be created at: $CONDA_ENV_PATH"
}

echo ""
echo "=========================================="
echo "Setting up ManiFeel environment"
echo "=========================================="

get_conda_env_path

# Create conda environment at the chosen path
echo "Creating Python 3.8 environment at '$CONDA_ENV_PATH'..."
$CONDA_CMD create --prefix "$CONDA_ENV_PATH" python=3.8 -y

source "$CONDA_BASE/etc/profile.d/conda.sh"

# Activate environment
conda activate "$CONDA_ENV_PATH"

echo ""
echo "=========================================="
echo "Installing IsaacGym TacSL"
echo "=========================================="

if [ "${CI:-false}" = "true" ]; then
    echo "⏭ Skipping IsaacGym install in CI mode"
else
    # Check if IsaacGym is already downloaded
    if [ ! -d "$PARENT_DIR/IsaacGym_Preview_TacSL_Package" ]; then
        echo "⚠ IsaacGym_Preview_TacSL_Package not found in parent directory."
        echo "Please download it from: https://drive.google.com/file/d/1FHs1tf3QajvYb11UkLaLcDD9THL-C0G5/view"
        echo "Extract it to: $PARENT_DIR"
        echo "Then run this script again."
        exit 1
    else
        echo "✓ Found IsaacGym_Preview_TacSL_Package"
        pip install -e "$PARENT_DIR/IsaacGym_Preview_TacSL_Package/isaacgym/python/"
    fi
fi

echo ""
echo "=========================================="
echo "Cloning repositories"
echo "=========================================="

# Clone IsaacGymEnvs
if [ "${CI:-false}" = "true" ]; then
    echo "⏭ Skipping manifeel-isaacgymenvs clone in CI mode"
elif [ ! -d "$PARENT_DIR/manifeel-isaacgymenvs" ]; then
    echo "Cloning manifeel-isaacgymenvs..."
    cd "$PARENT_DIR"
    git clone https://github.com/purdue-mars/manifeel-isaacgymenvs.git
    cd manifeel-isaacgymenvs
    pip install -e .
else
    echo "✓ manifeel-isaacgymenvs already exists"
    cd "$PARENT_DIR/manifeel-isaacgymenvs"
    pip install -e .
fi

# Clone Diffusion Policy
if [ ! -d "$PARENT_DIR/diffusion_policy" ]; then
    echo "Cloning diffusion_policy..."
    cd "$PARENT_DIR"
    git clone https://github.com/real-stanford/diffusion_policy.git
    cd diffusion_policy
    pip install -e .
else
    echo "✓ diffusion_policy already exists"
    cd "$PARENT_DIR/diffusion_policy"
    pip install -e .
fi

echo ""
echo "=========================================="
echo "Installing ManiFeel"
echo "=========================================="

# Install ManiFeel
cd "$SCRIPT_DIR"
pip install -e .

# Install additional dependencies
echo "Installing additional dependencies..."
pip install wandb==0.12.21 dill==0.3.9 tqdm==4.67.1 av==12.3.0 numpy==1.23.3 \
opencv-python==4.10.0.84 zarr==2.16.1 einops==0.4.1 huggingface-hub==0.25.0 \
diffusers==0.11.1 pandas==2.0.3 numba==0.56.4 rtree==1.3.0

echo ""
echo "=========================================="
echo "Installation Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Download IsaacGym_Preview_TacSL_Package (if not already done)"
echo "   from: https://drive.google.com/file/d/1FHs1tf3QajvYb11UkLaLcDD9THL-C0G5/view"
echo "2. Download ManiFeel dataset from:"
echo "   https://purdue0-my.sharepoint.com/:f:/g/personal/luu15_purdue_edu/IgClDSeuVGAKR4nlaok2yv2QAaOTl1FiHtebNThmTxuWi5U?e=s6z0jX"
echo "   and place it in manifeel/data/"
echo ""
echo "To activate the environment:"
echo "  conda activate $CONDA_ENV_PATH"
echo "  export LD_LIBRARY_PATH=\${CONDA_PREFIX}/lib:\${LD_LIBRARY_PATH}"
echo ""
