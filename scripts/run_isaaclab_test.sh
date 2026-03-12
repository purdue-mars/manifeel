#!/bin/bash
# Test script for Isaac Lab environment

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Activate isaaclab conda environment
eval "$(conda shell.bash hook)"
conda activate /scratch/gilbreth/halchaer/conda/envs/isaaclab

# Isaac Lab source path: the git-cloned Isaac Lab repo source packages
# (the pip isaaclab wheel is a launcher stub, actual framework lives in the git repo)
export ISAACLAB_SOURCE=/scratch/gilbreth/halchaer/git/IsaacLab/source/isaaclab

cd "$PROJECT_DIR"

echo "Running Isaac Lab environment tests..."
python test/test_isaaclab_env.py --headless "$@"
