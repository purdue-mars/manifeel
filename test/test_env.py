import numpy as np
import cv2
import hydra
from envs.env_wrapper import IsaacEnvWrapper

from omegaconf import DictConfig


@hydra.main(version_base="1.1", 
            config_path="../manipulation/IsaacGymEnvs/isaacgymenvs/cfg", 
            config_name="config")
def main(cfg: DictConfig):
    
    # Pass the config explicitly
    env = IsaacEnvWrapper(cfg)

    obs = env.reset()
    for i in range(256):
        actions = 2.0 * np.random.rand(6) - 1.0
        # step env and get observations
        obs, reward, reset, info = env.step(actions)

        left_tactile_rgb_image = obs['left_tactile_camera_taxim'][0].cpu().numpy()
        right_tactile_rgb_image = obs['right_tactile_camera_taxim'][0].cpu().numpy()
        wrist_rgb_image = obs['wrist'][0].cpu().numpy()
        wrist2_rgb_image = obs['wrist_2'][0].cpu().numpy()

        image = np.random.rand(256, 256, 3)
        print(image.dtype)
        # cv2.imshow('image', image)

        cv2.waitKey(1)

main()
