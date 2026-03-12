from gym import spaces

import torch
import numpy as np
import cv2
import random
import os
import sys

# Should point to Isaac Lab source path, must come before site-packages so the git source package
# takes priority over the pip launcher.
_ISAACLAB_SOURCE = os.environ.get(
    "ISAACLAB_SOURCE", # set env var to point to path like "/scratch/gilbreth/.../git/IsaacLab/source/isaaclab"
)
if _ISAACLAB_SOURCE not in sys.path:
    sys.path.insert(0, _ISAACLAB_SOURCE)

class IsaacLabEnvWrapper():

    metadata = {"render.modes": ["rgb_array"], "video.frames_per_second": 10}

    def __init__(self, cfg):
        self.cfg = cfg
        self.render_cache = []

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
        self.vision_obs_keys = list(rgb_keys)
        # this case handles no-vision setting
        if not self.vision_obs_keys:
            self.vision_obs_keys = ['front']

        self._start_task(self.cfg)

    def _start_task(self, cfg):
        env_id = cfg['env_id']
        num_envs = cfg.get('num_envs', 1)
        headless = cfg.get('headless', True)
        device = cfg.get('device', 'cuda:0')

        # bootstrap isaacsim kernel (initializes omni.kit path)
        import isaacsim  # noqa: F401

        # Start AppLauncher to initialize full Omniverse runtime
        # (must be done before importing any isaaclab.* modules)
        # Only start if not already running
        if not getattr(IsaacLabEnvWrapper, '_app_launcher', None):
            from isaaclab.app import AppLauncher
            launcher_cfg = {"headless": headless}
            IsaacLabEnvWrapper._app_launcher = AppLauncher(launcher_cfg)
            IsaacLabEnvWrapper._sim_app = IsaacLabEnvWrapper._app_launcher.app

        # Register all Isaac Lab task environments
        import isaaclab_tasks  # noqa: F401

        # Create env using Isaac Lab's parse_env_cfg + ManagerBasedRLEnv
        # (gymnasium.make() does not properly handle the env_cfg_entry_point kwarg)
        from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
        from isaaclab.envs import ManagerBasedRLEnv

        env_cfg = parse_env_cfg(env_id, device=device, num_envs=num_envs)
        self.envs = ManagerBasedRLEnv(cfg=env_cfg)
        self._num_envs = num_envs

    @property
    def observation_space(self):
        obs_space = self._build_observation_space()
        return obs_space

    @property
    def action_space(self):
        # Convert gymnasium.spaces to gym.spaces
        # Isaac Lab action_space.shape = (num_envs, action_dim); strip num_envs
        gym_action_space = self.envs.action_space
        if hasattr(gym_action_space, 'shape'):
            action_shape = gym_action_space.shape[1:] if len(gym_action_space.shape) > 1 else gym_action_space.shape
            return spaces.Box(
                low=np.float32(-1.0), high=np.float32(1.0),
                shape=action_shape,
                dtype=np.float32
            )
        return gym_action_space

    @property
    def num_envs(self) -> int:
        """Get the number of environments."""
        return self._num_envs

    def seed(self, seed=None):
        if seed is None:
            seed = np.random.randint(0, 25536)
        self._seed = seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        return seed

    def step(self, actions):
        actions = torch.from_numpy(actions).to(dtype=torch.float32)
        # Isaac Lab returns (obs, reward, terminated, truncated, info)
        obs, reward, terminated, truncated, info = self.envs.step(actions)

        obs = self._transform_obs_data(obs)
        self.render_cache = obs
        obs = self._apply_obs_by_keys(obs)

        # Merge terminated | truncated -> done (gym API compatibility)
        done = (terminated | truncated)
        if isinstance(done, torch.Tensor):
            done = done.cpu().numpy().astype(np.int64)
        else:
            done = np.array(done, dtype=np.int64)

        if isinstance(reward, torch.Tensor):
            reward = reward.cpu().numpy()
        else:
            reward = np.array(reward)

        # Convert info tensors to numpy
        np_info = {}
        for key, value in info.items():
            if isinstance(value, torch.Tensor):
                np_info[key] = value.cpu().numpy()
            else:
                np_info[key] = value

        return obs, reward, done, np_info

    def reset(self):
        # Isaac Lab returns (obs, info)
        obs, info = self.envs.reset()

        obs = self._transform_obs_data(obs)
        self.render_cache = obs
        obs = self._apply_obs_by_keys(obs)
        return obs

    def render(self, mode="rgb_array"):
        if self.render_cache is None:
            raise RuntimeError('Must run reset or step before render.')
        img_list = []

        for obs_view in self.vision_obs_keys:
            if obs_view in self.render_cache:
                imgs = self.render_cache[obs_view]  # (num_envs, 3, H, W)
                imgs = np.moveaxis(imgs, 1, -1)      # (num_envs, H, W, 3)
                imgs = (imgs * 255).astype(np.uint8) if imgs.max() <= 1.0 else imgs.astype(np.uint8)
                img_list.append(imgs)

        if img_list:
            return np.concatenate(img_list, axis=2)
        else:
            # render placeholder frame per env when no vision obs cached
            frames = []
            for i in range(self.num_envs):
                frame = np.zeros((128, 128, 3), dtype=np.uint8)
                frames.append(frame)
            return np.stack(frames, axis=0)

    def _transform_obs_data(self, obs):
        tf_obs = dict()

        if isinstance(obs, dict):
            for key, value in obs.items():
                if isinstance(value, torch.Tensor):
                    tf_obs[key] = value.cpu().numpy()
                else:
                    tf_obs[key] = np.array(value)

                # Check if the last dimension has size 3
                # check if it is the image data
                # (num_envs, H, W, 3) -> (num_envs, 3, H, W)
                if len(tf_obs[key].shape) >= 3 and tf_obs[key].shape[-1] == 3:
                    tf_obs[key] = np.moveaxis(tf_obs[key], -1, 1)
        else:
            # If obs is a flat tensor (e.g. state-only env)
            if isinstance(obs, torch.Tensor):
                obs_np = obs.cpu().numpy()
            else:
                obs_np = np.array(obs)
            tf_obs['flat_obs'] = obs_np

        # Construct state from policy observation if available
        if 'state' not in tf_obs:
            if 'policy' in tf_obs:
                # Isaac Lab often puts state obs under 'policy' key
                policy_obs = tf_obs.pop('policy')
                # Extract EE pose (first 7 dims: pos[3] + quat[4])
                state = policy_obs[:, :7]
                tf_obs['state'] = state
            elif 'flat_obs' in tf_obs:
                state = tf_obs['flat_obs'][:, :7]
                tf_obs['state'] = state

        return tf_obs

    def _apply_obs_by_keys(self, obs):
        task_obs = {key: obs[key] for key in self.task_obs_keys if key in obs}
        return task_obs

    def _build_observation_space(self):
        updated_observation_space = {}

        for key in self.task_obs_keys:
            shape_meta = self.cfg['shape_meta']['obs']
            if key in shape_meta:
                shape = tuple(shape_meta[key]['shape'])
                obs_type = shape_meta[key].get('type', 'low_dim')
                if obs_type == 'rgb':
                    updated_observation_space[key] = spaces.Box(
                        low=np.float32(-np.inf), high=np.float32(np.inf),
                        shape=shape,
                        dtype=np.float32
                    )
                else:
                    updated_observation_space[key] = spaces.Box(
                        low=np.float32(-np.inf), high=np.float32(np.inf),
                        shape=shape,
                        dtype=np.float32
                    )

        return spaces.Dict(updated_observation_space)

    def close(self):
        self.envs.close()
