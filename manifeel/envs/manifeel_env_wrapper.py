from gym import spaces

import isaacgym
# import isaacgymenvs
# from isaacgymenvs.utils.utils import set_seed
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
import cv2
import random
import os
import hydra
from omegaconf import DictConfig

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

class ManiFeelEnvWrapper():

    metadata = {"render.modes": ["rgb_array"], "video.frames_per_second": 10}

    def __init__(self, cfg):
        self.cfg = cfg
        self.render_cache = []
        self._start_task(self.cfg)
        self.light_factor = self.cfg.light_factor

        # obtain task observation keys from the shape_meta
        shape_meta = self.cfg['shape_meta']
        rgb_keys = list()
        lowdim_keys = list()
        obs_shape_meta = shape_meta['obs']
        for key, attr in obs_shape_meta.items():
            type = attr.get('type', 'low_dim')
            if type == 'rgb':
                rgb_keys.append(key)
            elif type == 'low_dim':
                lowdim_keys.append(key)
        
        self.task_obs_keys = rgb_keys + lowdim_keys 
        # Exclude tactile keys from rgb_keys
        tactile_rgb_obs_keys = [
                                'tactile_force_field_left', \
                                'tactile_force_field_right', \
                                'tactile_depth_left', \
                                'tactile_depth_right', \
                                ]
        self.vision_obs_keys = [key for key in rgb_keys if key not in tactile_rgb_obs_keys]
        # add side view for better visualization
        if 'side' not in self.vision_obs_keys:
            self.vision_obs_keys.append('side')
        if 'wrist' not in self.vision_obs_keys:
            self.vision_obs_keys.append('wrist')
        # this case handles tactile-only setting
        if not self.vision_obs_keys:
            self.vision_obs_keys = ['front', 'side']

    @property
    def observation_space(self):
        obs_space = self.envs.observation_space
        obs_space = self._update_observation_space(obs_space)
        return obs_space
    
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
        global_rank = 0
        # sets seed. if seed is -1 will pick a random one
        seed = set_seed(seed, torch_deterministic=self.cfg.torch_deterministic, rank=global_rank)
        # seed = set_seed(seed)
        self._seed = seed
        
    def step(self, actions):
        actions = torch.from_numpy(actions).to(dtype=torch.float32)
        obs, reward, reset, info = self.envs.step(actions)
        
        obs = self._transform_obs_data(obs['obs'])
        self.render_cache = obs
        obs = self._apply_obs_by_keys(obs)

        reset = reset.cpu().numpy()
        # reward = reward.cpu().numpy()
        reward = np.copy(reset)
        info = {key: value.cpu().numpy() for key, value in info.items() if isinstance(value, torch.Tensor)}

        return obs, reward, reset, info

    def reset(self):
        self.envs.reset_idx(torch.arange(self.num_envs))
        self.envs.compute_observations()
        obs = self.envs.reset()
        obs = self._transform_obs_data(obs['obs'])
        self.render_cache = obs
        obs = self._apply_obs_by_keys(obs)
        return obs
    
    def render(self, mode="rgb_array"):
        if self.render_cache is None:
            raise RuntimeError('Must run reset or step before render.')
        img_list = []

        # If 'side' is available, compute the target size (height, width)
        target_shape = None
        if 'side' in self.vision_obs_keys:
            target_shape = self.render_cache['side'].shape[2:4]  # (H, W)

        for obs_view in self.vision_obs_keys:
            imgs = self.render_cache[obs_view]  # (num_envs, 3, H, W)
            imgs = np.moveaxis(imgs, 1, -1)      # (num_envs, H, W, 3)
            imgs = (imgs * 255).astype(np.uint8)

            # For tactile camera views, if target_shape is defined, resize each image in the batch
            if obs_view in ['left_tactile_camera_taxim', 'right_tactile_camera_taxim'] and target_shape is not None:
                resized_imgs = []
                for img in imgs:
                    # cv2.resize expects size as (width, height)
                    resized_img = cv2.resize(img, (target_shape[1], target_shape[0]))
                    resized_imgs.append(resized_img)
                imgs = np.stack(resized_imgs, axis=0)

            img_list.append(imgs)

        return np.concatenate(img_list, axis=2)


    def _transform_obs_data(self, obs):
        tf_obs = dict()
        for key, value in obs.items():
            # Squeeze singleton dimensions
            tf_obs[key] = value.cpu().numpy()

            # Check if the last dimension has size 3
            # check if it is the image data
            # (num_envs, H, W, 3) -> (num_envs, 3, H, W)
            if len(tf_obs[key].shape) >= 3 and tf_obs[key].shape[-1] == 3:
                tf_obs[key] = np.moveaxis(tf_obs[key], -1, 1)
                if key in ["front", "side", "wrist", "wrist_2"]:
                    tf_obs[key] = self.light_condition(tf_obs[key], factor=self.light_factor)

        assert all(key in tf_obs for key in ('ee_pos', 'ee_quat')), \
                    "Keys 'ee_pos' and 'ee_quat' are missing in tf_obs"
        
        ee_pos = tf_obs['ee_pos']
        ee_quat = tf_obs['ee_quat']
        state = np.concatenate([ee_pos, ee_quat], axis=1)  # Shape: [2, 6]
        tf_obs['state'] = state

        return tf_obs

    def _apply_obs_by_keys(self, obs):
        task_obs = {key: obs[key] for key in self.task_obs_keys if key in obs}
        return task_obs

    def _update_observation_space(self, observation_space):
        updated_observation_space = {}
        for key, space in observation_space.spaces.items():
            if key in self.task_obs_keys: 
                if isinstance(space, spaces.Box):
                    # Check if the shape ends with (W, H, 3) and modify to (3, W, H)
                    if len(space.shape) == 3 and space.shape[-1] == 3:
                        updated_observation_space[key] = spaces.Box(
                            low=np.float32(-np.inf), high=np.float32(np.inf), 
                            shape=(3, space.shape[0], space.shape[1]), 
                            dtype=space.dtype
                        )
                    else:
                        # Keep the original space for non-image data
                        updated_observation_space[key] = space
                else:
                    # Keep the original space for non-Box types
                    updated_observation_space[key] = space

        if 'state' in self.task_obs_keys:
            updated_observation_space['state'] = spaces.Box(
                            low=np.float32(-np.inf), high=np.float32(np.inf), 
                            shape=(7,), 
                            dtype=np.float32
                        )

        # Return the updated Dict observation space
        return spaces.Dict(updated_observation_space)

    def light_condition(self, image, factor=0.5):
        if image.dtype in [np.float32, np.float64]:
            darkened = np.clip(image * factor, 0, 1)
        else:        
            darkened = np.clip(image.astype(np.float32) * factor, 0, 255).astype(np.uint8)
        return darkened

    def close(self):
        pass
        # return self.envs.close()

def set_seed(seed, torch_deterministic=False, rank=0):
    """ set seed across modules """
    if seed == -1 and torch_deterministic:
        seed = 42 + rank
    elif seed == -1:
        seed = np.random.randint(0, 10000)
    else:
        seed = seed + rank

    print("Setting seed: {}".format(seed))

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    return seed

if __name__ == '__main__':
    @hydra.main(version_base="1.1", 
                config_path="../config", 
                config_name="isaacgym_config")
    def main(cfg: DictConfig):
        wrapped_env = ManiFeelEnvWrapper(cfg)
        print("Observation space is", wrapped_env.observation_space)
        print("Action space is", wrapped_env.action_space)
        num_envs = wrapped_env.num_envs
        
        wrapped_env.seed(0)
        _ = wrapped_env.reset()
        
        for _ in range(5):
            random_actions = 2.0 * np.random.rand(num_envs, wrapped_env.action_space.shape[0]) - 1.0
            obs, reward, reset, _ = wrapped_env.step(random_actions)
            tactile_rgb_image = obs['left_tactile_camera_taxim']
            state = obs['state']
            print(obs.keys())
            print(tactile_rgb_image.shape, state.shape)

    main()