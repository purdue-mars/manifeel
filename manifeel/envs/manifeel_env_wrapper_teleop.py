from gym import spaces

import isaacgym
import isaacgymenvs
from isaacgymenvs.utils.utils import set_seed
from isaacgymenvs.tasks.tacsl.tacsl_task_insertion import TacSLTaskInsertion
from isaacgymenvs.tasks.tacsl.tacsl_task_gear import TacSLTaskGear
from isaacgymenvs.tasks.tacsl.tacsl_task_USB import TacSLTaskUSB
from isaacgymenvs.tasks.tacsl.tacsl_task_power import TacSLTaskPowerInsertion
from isaacgymenvs.tasks.tacsl.tacsl_task_bolt_nut import TacSLTaskBoltNut
from isaacgymenvs.tasks.tacsl.tacsl_task_pick_in_box import TacSLTaskPickInBox
from isaacgymenvs.tasks.tacsl.tacsl_task_search_in_box import TacSLTaskSearchInBox
from isaacgymenvs.tasks.tacsl.tacsl_task_class_ball import TacSLTaskClassBall
from isaacgymenvs.utils.reformat import omegaconf_to_dict

import torch
import numpy as np
import hydra
from omegaconf import DictConfig, OmegaConf

# Mappings from strings to environments
isaacgym_task_map = {
    "TacSLTaskInsertion": TacSLTaskInsertion,
    "TacSLTaskUSB": TacSLTaskUSB,
    "TacSLTaskGear": TacSLTaskGear,
    "TacSLTaskPowerInsertion": TacSLTaskPowerInsertion,
    "TacSLTaskPickInBox": TacSLTaskPickInBox,
    "TacSLTaskBoltNut": TacSLTaskBoltNut,
    "TacSLTaskSearchInBox": TacSLTaskSearchInBox,
    "TacSLTaskClassBall": TacSLTaskClassBall,
}

class ManiFeelTeleopEnvWrapper():
    def __init__(self, cfg):
        self.cfg = cfg
        self._start_task(self.cfg)

    @property
    def observation_space(self):
        return self.envs.observation_space
    
    @property
    def action_space(self):
        return self.envs.action_space

    @property
    def num_acts(self) -> int:
        """Get the number of actions in the environment."""
        return self.envs.num_actions

    @property
    def num_obs(self) -> int:
        """Get the number of observations in the environment."""
        return self.envs.num_observations

    @property
    def num_envs(self) -> int:
        """Get the number of environments."""
        return self.envs.num_environments

    def _start_task(self, cfg):
        rl_device = cfg.rl_device
        sim_device = cfg.sim_device
        graphics_device_id = cfg.graphics_device_id
        headless = cfg.headless
        virtual_screen_capture = cfg.capture_video
        force_render = cfg.force_render

        cfg_task = cfg['task']
        cfg_dict = omegaconf_to_dict(cfg_task)

        task_name = cfg_dict['name'] #e.g., TacSLTaskUSB

        self.envs = isaacgym_task_map[task_name](cfg=cfg_dict,
                                       rl_device=rl_device,
                                       sim_device=sim_device,
                                       graphics_device_id=graphics_device_id,
                                       headless=headless,
                                       virtual_screen_capture=virtual_screen_capture,
                                       force_render=force_render,
                                       )

    def seed(self, seed=None):
        if seed is None:
            seed = np.random.randint(0,25536)
        # global rank of the GPU
        # global_rank = int(os.getenv("RANK", "0"))
        global_rank = 0
        # sets seed. if seed is -1 will pick a random one
        seed = set_seed(seed, torch_deterministic=self.cfg.torch_deterministic, rank=global_rank)
        # seed = set_seed(seed)
        self._seed = seed
        
    def step(self, actions):
        actions = torch.from_numpy(actions).to(dtype=torch.float32).unsqueeze(0)
        obs, reward, reset, info = self.envs.step(actions)
        obs = obs['obs']
        return obs, reward, reset, info

    def reset(self):
        self.envs.reset_idx(torch.arange(self.num_envs))
        self.envs.compute_observations()
        obs = self.envs.reset()
        obs = obs['obs']
        return obs
    
    def render(self, mode="rgb_array"):
        if self.envs.virtual_display and mode == "rgb_array":
            img = self.envs.virtual_display.grab()
            return np.array(img)

    def close(self):
        pass


if __name__ == '__main__':
    @hydra.main(version_base="1.1", 
                config_path="../config", 
                config_name="isaacgym_config")
    def main(cfg: DictConfig):
        # Pass the config explicitly
        wrapped_env = ManiFeelTeleopEnvWrapper(cfg)
        print("Observation space is", wrapped_env.observation_space)
        print("Action space is", wrapped_env.action_space)
        wrapped_env.seed(0)
        obs = wrapped_env.reset()
        for _ in range(5):
            random_actions = 2.0 * np.random.rand(wrapped_env.action_space.shape[0]) - 1.0
            obs, _, _, _ = wrapped_env.step(random_actions)
            print(f"🚀obs_dict[tactile_force_field_left].shape: {obs['tactile_force_field_left'].shape}")
            print(f"🚀obs_dict[tactile_depth_left].shape: {obs['tactile_depth_left'].shape}")

            # print(reward, reset)
            # tactile_rgb_image = observations['left_tactile_camera_taxim'][0]
            # wrist_rgb_image = observations['wrist_2'][0]

    main()