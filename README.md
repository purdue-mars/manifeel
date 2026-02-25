# ManiFeel

ManiFeel is a benchmarking and learning platform for supervised visuotactile policy learning. It provides a comprehensive collection of visuotactile manipulation tasks and modular learning pipelines that include sensing modality configurations, tactile encoders, and policy heads. Built on IsaacGym/TacSL, a simulator for GelSight tactile sensors, the platform supports systematic studies and fair comparisons of supervised policies for contact-rich and visually-degraded manipulation tasks with the integration of visual and tactile sensing.

---

## 1. Installation

ManiFeel provides an automated installation script that handles all setup steps.

### Prerequisites

• **Download the TacSL specific Isaac Gym binary** from [here](https://drive.google.com/file/d/13dFRF9EXpzIWaJF2Z6f7BsuPUGQkPE8v/view?usp=sharing) and extract it to the parent directory of the manifeel repository:

```bash
tar -xvzf IsaacGym_Preview_TacSL_Package.tar.gz
```

The directory structure should look like:
```
parent_directory/
├── IsaacGym_Preview_TacSL_Package/
└── manifeel/
```

### Automated Installation

Clone the ManiFeel repository and run the installation script:

```bash
git clone https://github.com/purdue-mars/manifeel.git
cd manifeel
bash install.sh
```

The installation script will:
- Check for conda/mamba, and install Miniforge3 if not found
- Create a Python 3.8 environment named `manifeel`
- Install IsaacGym TacSL
- Clone and install manifeel-isaacgymenvs (TacSL fork)
- Clone and install Diffusion Policy
- Install ManiFeel and all dependencies
---

## 2. Download ManiFeel dataset

Download and unzip the ManiFeel dataset for your target task from [here](https://huggingface.co/datasets/purdue-mars/manifeel/tree/main/data) and place it inside the `manifeel/data` directory of the `manifeel` repository. If the `data` directory does not exist, please create it.

---

## 3. Setup Apptainer for Training

To ensure a consistent and reproducible environment across clusters, workstations, and local PCs, we provide an Apptainer-based setup for ManiFeel. System configurations and dependency versions may vary across machines, which can lead to compatibility issues.

Apptainer allows ManiFeel to run inside a controlled Ubuntu-based container with all required dependencies pre-defined, simplifying setup and improving portability.

Please follow the steps below to configure the containerized training environment.

---

The repository includes Apptainer definition file `manifeel.def`. From the root directory of the repository, build the Apptainer image (`manifeel.sif`):

```bash
apptainer build manifeel.sif manifeel.def
```

You can also download the prebuilt `manifeel.sif` Apptainer image from the following link: [Download manifeel.sif](https://drive.google.com/file/d/1MGMfEA7EnhgfbnHF9moOPPLHyEZowxfC/view?usp=sharing)

You can then try running the container with:

```bash
apptainer exec --nv manifeel.sif bash
```

This will drop you into a bash shell inside the ManiFeel compatible Ubuntu-based Apptainer environment.

Then, run the following commands inside the Apptainer environment to verify that everything is working correctly:

```bash
source ~/.bashrc
conda activate manifeel
export LD_LIBRARY_PATH=${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH}
python -c "from isaacgym import gymtorch"
```
If the `gymtorch` library builds and imports correctly (that is, no errors appear), you can exit the Apptainer environment:

```bash
exit
```

---

## 4. ManiFeel Run on Cluster with Slurm

Once the ManiFeel environment and Apptainer container have been correctly set up, you can run training for any ManiFeel task.  
As an example, this section shows how to train a **vision-only Diffusion Policy** for the **USB insertion** task, i.e., its dataset `usb_quan_Aug05` could be found under [this HuggingFace link](https://huggingface.co/datasets/purdue-mars/manifeel/tree/main/data). Make sure that the ManiFeel demo dataset for USB insertion has already been downloaded and placed in `manifeel/data/usb_quan_Aug05`:

---

### 4.1 Creating the Slurm Submission Script

To run ManiFeel training on the cluster, you need a Slurm job script.  
Create a file named `job_submit.sh`:

```bash
touch job_submit.sh
```

Paste the following script into it:

> **Important:**  
> Before using the job script below, update the following fields:
> 
> • Searh for `[user]` in the script file and replace `[user]` with your own cluster username.  
> • Ensure that `CONTAINER_FILE` correctly points to where you stored your `manifeel.sif` file
>   ```
>   CONTAINER_FILE=/path/to/cluster/[user]/manifeel.sif
>   ```  
> • Confirm that the `cd` command correctly points to your `manifeel` repository path, matching the actual location of your `manifeel` repo on the cluster.
>   ```
>   cd /path/to/cluster/[user]/Projects/manifeel
>   ```  


```bash
#!/bin/bash

SEED=44      
                     
NUM_DEMOS=50
NUM_EPOCH=1000
DATASET_PATH=data/usb_quan_Aug05
ISAACGYM_CONFIG="isaacgym_config_usb.yaml"
ENV="usb_wrist_0805"
LOG_NAME="dp_usb_tacff"
TASK_NAME=vision_wrist
INPUT_TYPE="vision"
EXP_NAME="${INPUT_TYPE}_${ENV}_${NUM_DEMOS}"  

JOB_NAME="${EXP_NAME}_${SEED}" # The name of the Slurm job to monitor 

CONTAINER_FILE=/path/to/cluster/[user]/manifeel.sif

cat <<EOT > job_script_${JOB_NAME}.sh
#!/bin/bash
#SBATCH --job-name=${JOB_NAME}
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --account=shey
#SBATCH --gres=gpu:1
#SBATCH --partition=a30
#SBATCH --mem=24G
#SBATCH --qos=normal
#SBATCH --cpus-per-task=8
#SBATCH --time=8:00:00

# Run the commands inside the Apptainer container
apptainer exec --nv ${CONTAINER_FILE} bash -c "
    source ~/.bashrc
    conda activate manifeel
    export LD_LIBRARY_PATH=\${CONDA_PREFIX}/lib:\${LD_LIBRARY_PATH}
    cd /path/to/cluster/[user]/Projects/manifeel
    python train.py \
        --config-name=train_diffusion_workspace.yaml \
        task=${TASK_NAME} \
        exp_name=${EXP_NAME} \
        dataset_path=${DATASET_PATH} \
        isaacgym_cfg_name=${ISAACGYM_CONFIG} \
        training.seed=${SEED} \
        training.num_epochs=${NUM_EPOCH} \
        task.dataset.max_train_episodes=${NUM_DEMOS} \
        hydra.run.dir=data/outputs/${EXP_NAME}/${SEED} \
        logging.project=${LOG_NAME} \
"
EOT

# Infinite loop to monitor and resubmit the job
while true; do
    # Check if the job is currently running
    JOB_ID=$(squeue --name=$JOB_NAME --noheader --format=%A)

    if [ -z "$JOB_ID" ]; then
        # If no job with the specified name is running, resubmit the job
        echo "Job $JOB_NAME is not running. Resubmitting..."
        # Submit the dynamically created script
        sbatch job_script_${JOB_NAME}.sh

        # Wait a few seconds to avoid rapid resubmission
        sleep 10
    else
        # Output a message indicating the job is still running
        echo "Job $JOB_NAME is still running (Job ID: $JOB_ID)."
    fi

    # Wait for a specified interval before checking the job status again
    sleep 30
done
```

---

### 4.2 Submitting the Training Job

Once the script is ready, grant the run permission

```bash
chmod +x job_submit.sh
```

then, submit it using:

```bash
./job_submit.sh
```

Slurm will schedule your job, and logs will appear in the `logs/` directory.

If everything runs correctly, you will see the success rate and selected simulation rollouts logged to your W&B account.

---

### 4.3 Running Vision + TacRGB Policy

To run the vision+tacRGB policy of the USB insertion policy, create a new copy of the bash script file `job_submit.sh` and/or modify the following two fields in your `job_submit.sh` script:

```bash
TASK_NAME=vistac_wrist
INPUT_TYPE="vistac"
```

After updating, submit the script file:

```bash
./job_submit.sh
```

---

### 4.4 Running Vision + TacFF Policy

To run the vision+tacFF (tactile force-field) policy of the USB insertion policy, create a new copy of the bash script file `job_submit.sh` and/or modify the following two fields in your `job_submit.sh` script:

```bash
TASK_NAME=visff_wrist
INPUT_TYPE="tacff"
```

Also update the Hydra config by adding `policy.obs_encoder.imagenet_norm=True` to the `train` Python command in your `job_submit.sh` script, as shown below:

```bash
python train.py \
    --config-name=train_diffusion_workspace.yaml \
    task=${TASK_NAME} \
    exp_name=${EXP_NAME} \
    dataset_path=${DATASET_PATH} \
    isaacgym_cfg_name=${ISAACGYM_CONFIG} \
    policy.obs_encoder.imagenet_norm=True \
    training.seed=${SEED} \
    training.num_epochs=${NUM_EPOCH} \
    task.dataset.max_train_episodes=${NUM_DEMOS} \
    hydra.run.dir=data/outputs/${EXP_NAME}/${SEED} \
    logging.project=${LOG_NAME} \
```

After updating, submit the script file:

```bash
./job_submit.sh
```

### 4.5 Run Other ManiFeel Tasks

You can run any ManiFeel task, such as **Ball Sorting**, by preparing the dataset and updating your `job_submit.sh` script.  

First, download and unzip the demo dataset `sorting_quan_Aug8` under this [HuggingFace link](https://huggingface.co/datasets/purdue-mars/manifeel/tree/main/data), then place the extracted folder inside the `manifeel/data` directory.

Next, create a new copy of `job_submit.sh` or modify your existing one by updating the following fields:

```bash
DATASET_PATH=data/sorting_quan_Aug8
ISAACGYM_CONFIG="isaacgym_config_ball_sorting.yaml"
ENV="sorting_0923"
LOG_NAME="dp_sorting_tacff"
TASK_NAME=vision_front
INPUT_TYPE="vision"
```

For example, your training command may look like:

```bash
python train.py \
    --config-name=train_diffusion_workspace.yaml \
    task=${TASK_NAME} \
    exp_name=${EXP_NAME} \
    dataset_path=${DATASET_PATH} \
    isaacgym_cfg_name=${ISAACGYM_CONFIG} \
    training.seed=${SEED} \
    training.num_epochs=${NUM_EPOCH} \
    task.shape_meta.action.shape="[7]" \
    task.dataset.max_train_episodes=${NUM_DEMOS} \
    hydra.run.dir=data/outputs/${EXP_NAME}/${SEED} \
    logging.project=${LOG_NAME}
```

> **Note:**  
> You can modify `TASK_NAME` and `INPUT_TYPE` to match the sensing configuration you want to test  
> (vision-only, vision+tacRGB, or vision+tacFF).  
> For example, in the Ball Sorting task, which uses the **front camera** instead of a wrist camera, the valid task names are:  
> - `TASK_NAME=vision_front` for vision-only  
> - `TASK_NAME=vistac_front` for vision+tacRGB  
> - `TASK_NAME=visff_front` for vision+tacFF  
>
> Tasks such as **Ball Sorting**, **Object Search**, **Bulb Installation**, and **Nut-Bolt Threading** require gripper control and therefore use a **7 dimensional action space**. In these cases, ensure that  
> ```
> task.shape_meta.action.shape="[7]"
> ```  
> is included in your `python train.py` command.

After updating your script, start the run:

```bash
./job_submit.sh
```

> **Important:**  
> Among the parameters in `job_submit.sh`, the most critical ones to update when switching tasks or sensing modalities are:  
> `DATASET_PATH`, `ISAACGYM_CONFIG`, and `TASK_NAME`.  
> Other fields primarily affect file naming and experiment logging.  
>  
> You can freely adjust `SEED`, `NUM_DEMOS`, and `NUM_EPOCH` to control the randomness seed, number of demonstrations used for training, and total training epochs.

---

## 5. Run ManiFeel Locally (PC or Workstation)

This section mirrors the Cluster workflow but runs training directly on a local machine without Slurm. It assumes:

* `manifeel.sif` has already been built
* The `manifeel` Conda environment
* `scripts/run_local.sh` is available

---

### 5.1 Prepare the Local Script

Grant execution permission to the local script:

```bash
chmod +x scripts/run_local.sh
```

You can now launch training directly from your workstation. Logs and checkpoints will be saved under `data/outputs/${EXP_NAME}/${SEED}`. If everything runs correctly, you will see success rate metrics and rollout videos logged to your W&B account.

### 5.2 Running Vision-Only Policy
To run the vision-only **USB insertion** policy, override the following variables at launch time:

```bash
TASK_NAME=vision_wrist \
INPUT_TYPE=vision \
bash scripts/run_local.sh
```

You do not need to edit the script itself; the environment variables passed before the command override the default values inside `run_local.sh`.

### 5.3 Running Vision + TacRGB Policy
To run the vision + TacRGB policy, which enables RGB tactile images together with vision input, override:

```bash
TASK_NAME=vistac_wrist \
INPUT_TYPE=vistac \
bash scripts/run_local.sh
```

### 5.4 Running Vision + TacFF Policy
To run the vision + TacFF (tactile force-field) policy, override:

```bash
TASK_NAME=visff_wrist \
INPUT_TYPE=tacff \
bash scripts/run_local.sh
```

### 5.5 Running Other ManiFeel Tasks Locally
To run other tasks such as **Ball Sorting**, first prepare the dataset inside `manifeel/data/`, then override the required fields when launching:

```bash
DATASET_PATH=data/sorting_quan_Aug8 \
ISAACGYM_CONFIG=isaacgym_config_ball_sorting.yaml \
ENV=sorting_0923 \
TASK_NAME=vision_front \
INPUT_TYPE=vision \
bash scripts/run_local.sh
```

For front-camera tasks, valid task names include:
  * **Vision-only**: `TASK_NAME=vision_front`
  * **Vision + TacRGB**: `TASK_NAME=vistac_front`
  * **Vision + TacFF**: `TASK_NAME=visff_front`
    
### 5.6 Important Parameters
When switching tasks or sensing modalities, the most critical variables are: `DATASET_PATH`, `ISAACGYM_CONFIG`, `TASK_NAME`, `INPUT_TYPE`.

You can also adjust training hyperparameters: `SEED`, `NUM_DEMOS`, `NUM_EPOCH`.

Example:
```bash
SEED=44 \
NUM_DEMOS=50 \
NUM_EPOCH=1000 \
TASK_NAME=visff_wrist \
INPUT_TYPE=tacff \
bash scripts/run_local.sh
```    

### 5.7 Summary
The local workflow is identical to the Cluster setup, except:
  * No Slurm submission or `job_submit.sh`
  * Direct execution via bash `scripts/run_local.sh`
  * All sensing configurations are controlled by overriding environment variables at launch time
  
## 6. Pretrained Tactile Representation Checkpoints

Pretrained tactile representation checkpoints are required to run benchmarking across different tactile models. Download the pretrained checkpoints for UniT, T3, and AnyTouch from [here](https://drive.google.com/file/d/1MYDMc61vYzL8BTRPCbqNKIBgy-Rp66Sl/view?usp=sharing), then extract the `representation_models` folder into the top-level directory of the `manifeel` repository.

The directory structure should look like:
```
parent_directory/
├── IsaacGym_Preview_TacSL_Package/
└── manifeel/
    ├── data/
    ├── representation_models/
    └── manifeel/
```