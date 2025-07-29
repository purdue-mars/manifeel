"""
Usage:
Visualize first 3 episodes
EPISODES="0,1,2" python data_vis.py --config-name data_vis_peginhole.yaml
"""

import hydra
import os
import pathlib
from omegaconf import OmegaConf
from diffusion_policy.dataset.base_dataset import BaseImageDataset
import sys
import rerun as rr
from tqdm import tqdm
import numpy as np

from manifeel.utils.shear_tactile_viz_utils import visualize_tactile_shear_image, visualize_penetration_depth

EPISODES = os.getenv("EPISODES")

# JOINT_NAMES = [
#     "waist",
#     "shoulder",
#     "elbow",
#     "forearm_roll",
#     "wrist_angle",
#     "wrist_rotate",
#     "gripper",
# ]

AXES = ['x', 'y', 'z', 'roll', 'pitch', 'yaw']

sys.stdout = open(sys.stdout.fileno(), mode="w", buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode="w", buffering=1)
OmegaConf.register_new_resolver("eval", eval, replace=True)

@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.joinpath("./manifeel", "config")),
)
def main(cfg: OmegaConf):
    OmegaConf.resolve(cfg)

    rr.init("vistac_demonstration_data_vis", spawn=False)
    save_path = pathlib.Path(cfg.dataset_path) / "debug.rrd"
    if save_path.exists():
        save_path.unlink()
    rr.save(str(save_path))

    dataset: BaseImageDataset
    dataset = hydra.utils.instantiate(cfg.task.dataset)

    if EPISODES is not None:
        episode_idxs = [int(EPISODE.strip()) for EPISODE in EPISODES.split(",")]
    else:
        episode_idxs = list(range(dataset.replay_buffer.n_episodes))

    for episode_idx in tqdm(episode_idxs, "Loading episodes"):
        vis_episode = dataset.replay_buffer.get_episode(episode_idx)
        action_buffer = vis_episode["action"]
        # qpos_buffer = vis_episode["qpos"]
        size = action_buffer.shape[0]

        action_dim = action_buffer.shape[1]
        for i in range(size):
            for j in range(action_dim):
                    name = AXES[j]
                    rr.log(f"action/{name}", rr.Scalar(action_buffer[i, j]))
                    # rr.log(f"qpos/{name}", rr.Scalar(qpos_buffer[i, j]))

            for name in [
                "front",
                "side",
                "wrist",
                "wrist_2",
                "right_tactile_camera_taxim",
                "left_tactile_camera_taxim",
                "tactile_force_field_left", 
                "tactile_force_field_right",
                "tactile_depth_left", 
                "tactile_depth_right",
            ]:
                if name not in vis_episode:
                    continue
                if name == "tactile_force_field_left" or name == "tactile_force_field_right":
                    raw_force_field = vis_episode[name][i]
                    img = visualize_tactile_shear_image(
                        raw_force_field[..., 0],
                        raw_force_field[..., 1:],
                        normal_force_threshold=0.004,
                        shear_force_threshold=0.0010, #0.0005
                        resolution=25)
                    # print(f"✅img.dtype: {img.dtype}")
                    img = np.moveaxis(img[:, :, ::-1], 1, 0)
                    # print(f"✅raw_force_field.shape: {raw_force_field.shape}")
                    # print(f"🏁tactile_image.shape: {image.shape}")
                else:
                    img = vis_episode[name][i]
                    # print(f"💾img.dtype: {img.dtype}")
                    # print(f"💾img.shape: {img.shape}")
                
                rr.log(f"image/{name}", rr.Image(img))
            
            rr.log("episode", rr.Scalar(episode_idx))

    rr.disconnect()
    print(f"Saved at {save_path}!")

# %%
if __name__ == "__main__":
    main()