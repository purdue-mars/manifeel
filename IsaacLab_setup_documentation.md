## Isaac Lab Setup

Isaac Lab backend (`IsaacLabRunner`) can be used as a drop-in replacement for Isaac Gym. It uses the same `BaseImageRunner` interface so existing training code works unchanged.

### Isaac Lab GitHub Clone Required

The PyPI `isaaclab` package is **only a launcher stub**. It does not ship the actual framework source code (`isaaclab.envs`, `isaaclab.managers`, etc.). You must also clone the GitHub repo:

```bash
git clone https://github.com/isaac-sim/IsaacLab.git --branch v2.1.0 --depth 1 \
    /scratch/<user>/git/IsaacLab
```

### Isaac Lab Requires Python >=3.10, so you need to make a new environment (you cannot use 3.8):

```bash
conda create --prefix /scratch/<user>/conda/envs/isaaclab python=3.10 -y
conda activate /scratch/<user>/conda/envs/isaaclab

# Fix setuptools for flatdict build
pip install "setuptools<71"

# PyTorch 2.5.1 (required by isaaclab 2.1.0)
pip install torch==2.5.1 torchvision --index-url https://download.pytorch.org/whl/cu124

# Isaac Sim 4.5.0.0 — meta-package + extensions from NVIDIA PyPI
pip install isaacsim==4.5.0.0 --extra-index-url https://pypi.nvidia.com
pip install \
  isaacsim-rl==4.5.0.0 \
  isaacsim-replicator==4.5.0.0 \
  isaacsim-extscache-physics==4.5.0.0 \
  isaacsim-extscache-kit-sdk==4.5.0.0 \
  isaacsim-extscache-kit==4.5.0.0 \
  --extra-index-url https://pypi.nvidia.com

# Isaac Lab 2.1.0
pip install isaaclab==2.1.0

# ManiFeel + Diffusion Policy
# Note: diffusion_policy uses legacy setup.py, needs compat editable mode
pip install -e /path/to/manifeel
cd /path/to/diffusion_policy && pip install -e . --config-settings editable_mode=compat && cd -

# Remaining deps (pin numpy<2 and opencv for numpy 1.x compat)
pip install wandb dill tqdm av "numpy<2" "opencv-python==4.9.0.80" zarr diffusers numba

# gym 0.26.2 needed by MultiStepWrapper and VideoRecordingWrapper (which use 'import gym')
pip install "gym==0.26.2"

# Accept NVIDIA Omniverse EULA (one-time)
echo "Yes" | python -c "import isaacsim"
```

> **Note on versions**: `isaaclab` 2.0.x/2.1.0 require Python 3.10 and `numpy<2`. Versions 2.2.0+ require Python 3.11 and `isaacsim>=5.0`. Always use `isaacsim` and `isaaclab` at matching major versions.

### Install Isaac Lab Source Sub-Packages

The GitHub source contains the actual framework. Install each sub-package from it:

```bash
ISAACLAB_GIT=/scratch/<user>/git/IsaacLab

# Install the main framework (isaaclab.envs, isaaclab.managers, etc.)
cd $ISAACLAB_GIT/source/isaaclab
pip install -e . --config-settings editable_mode=compat --no-deps

# Install task definitions (Isaac-Lift-Cube-Franka-v0, etc.)
cd $ISAACLAB_GIT/source/isaaclab_tasks
pip install -e . --config-settings editable_mode=compat --no-deps

# Install robot assets
cd $ISAACLAB_GIT/source/isaaclab_assets
pip install -e . --config-settings editable_mode=compat --no-deps
```

> **Note**: Set `ISAACLAB_SOURCE` env var to point to the cloned source.
> `IsaacLabEnvWrapper` reads this to ensure the git source takes priority over the pip launcher stub:
> ```bash
> export ISAACLAB_SOURCE=/scratch/<user>/git/IsaacLab/source/isaaclab
> ```

### Accept the NVIDIA EULA and make sure isaacsim import works:

```bash
python -c "import isaacsim"
```

### Run test using the test-script (should all pass)

```bash
bash scripts/run_isaaclab_test.sh
# Or run individual test levels:
python test/test_isaaclab_env.py --headless --env_id Isaac-Lift-Cube-Franka-v0 --test A
```

### Run training with Isaac Lab (not fully tested yet)

```bash
python train.py \
    --config-name=train_diffusion_workspace.yaml \
    task=isaaclab_lift_cube \
    exp_name=isaaclab_test \
    dataset_path=data/<your_dataset>
```
