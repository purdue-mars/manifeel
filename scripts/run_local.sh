#!/bin/bash
set -euo pipefail

# -------------------------
# User-editable parameters
# -------------------------
SEED=${SEED:-44}
NUM_DEMOS=${NUM_DEMOS:-50}
NUM_EPOCH=${NUM_EPOCH:-1000}

DATASET_PATH=${DATASET_PATH:-data/usb_quan_Aug05}
ISAACGYM_CONFIG=${ISAACGYM_CONFIG:-isaacgym_config_usb.yaml}

TASK_NAME=${TASK_NAME:-vistac_pih_multiple_vision_onecam}
INPUT_TYPE=${INPUT_TYPE:-vision}   # vision | vistac | tacff
ENV_TAG=${ENV_TAG:-usb_wrist_0805}
LOG_NAME=${LOG_NAME:-dp_usb_tacff}

# Optional Hydra overrides
IMAGENET_NORM=${IMAGENET_NORM:-false}   # set true for tacff runs if needed
ACTION_SHAPE=${ACTION_SHAPE:-}          # set to 7 for gripper tasks, leave empty otherwise

# Container file (default assumes you run from repo root and built it there)
CONTAINER_FILE=${CONTAINER_FILE:-manifeel.sif}

# -------------------------
# Derived names
# -------------------------
EXP_NAME="${INPUT_TYPE}_${ENV_TAG}_${NUM_DEMOS}"
OUT_DIR="data/outputs/${EXP_NAME}/${SEED}"

# Repo root resolution (script works from any directory)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# -------------------------
# Build Hydra args
# -------------------------
HYDRA_ARGS=(
  "--config-name=train_diffusion_workspace.yaml"
  "task=${TASK_NAME}"
  "exp_name=${EXP_NAME}"
  "dataset_path=${DATASET_PATH}"
  "isaacgym_cfg_name=${ISAACGYM_CONFIG}"
  "training.seed=${SEED}"
  "training.num_epochs=${NUM_EPOCH}"
  "task.dataset.max_train_episodes=${NUM_DEMOS}"
  "hydra.run.dir=${OUT_DIR}"
  "logging.project=${LOG_NAME}"
)

if [[ "${IMAGENET_NORM}" == "true" ]]; then
  HYDRA_ARGS+=("policy.obs_encoder.imagenet_norm=True")
fi

if [[ -n "${ACTION_SHAPE}" ]]; then
  HYDRA_ARGS+=("task.shape_meta.action.shape=[${ACTION_SHAPE}]")
fi

# -------------------------
# Run inside Apptainer
# -------------------------
apptainer exec --nv --cleanenv --env LD_PRELOAD= "${REPO_ROOT}/${CONTAINER_FILE}" bash -ic "
  set -e
  conda activate manifeel
  export LD_LIBRARY_PATH=\${CONDA_PREFIX}/lib:\${LD_LIBRARY_PATH}
  cd '${REPO_ROOT}'
  python train.py ${HYDRA_ARGS[*]}
"