import gym
from gym import spaces
import numpy as np
from collections import defaultdict, deque
import dill
import tqdm


def stack_repeated(x, n):
    return np.repeat(np.expand_dims(x,axis=0),n,axis=0)

def repeated_box(box_space, n):
    return spaces.Box(
        low=stack_repeated(box_space.low, n),
        high=stack_repeated(box_space.high, n),
        shape=(n,) + box_space.shape,
        dtype=box_space.dtype
    )

def repeated_space(space, n):
    if isinstance(space, spaces.Box):
        return repeated_box(space, n)
    elif isinstance(space, spaces.Dict):
        result_space = spaces.Dict()
        for key, value in space.items():
            result_space[key] = repeated_space(value, n)
        return result_space
    else:
        raise RuntimeError(f'Unsupported space type {type(space)}')

def take_last_n(x, n):
    # add transpose to swap 
    # time and env dimension
    # return shape (env, T)
    x = list(x)
    n = min(len(x), n)
    return np.array(x[-n:]).T

def dict_take_last_n(x, n):
    result = dict()
    for key, value in x.items():
        result[key] = take_last_n(value, n)
    return result

def aggregate(data, method='max'):
    # compute aggregate per environment
    # data: (T, num_envs, shape)
    if method == 'max':
        # equivalent to any
        return np.max(data, axis=0)
    elif method == 'min':
        # equivalent to all
        return np.min(data, axis=0)
    elif method == 'mean':
        return np.mean(data, axis=0)
    elif method == 'sum':
        return np.sum(data, axis=0)
    else:
        raise NotImplementedError()

def stack_last_n_obs(all_obs, n_steps):
    assert(len(all_obs) > 0)
    all_obs = list(all_obs)
    result = np.zeros((n_steps,) + all_obs[-1].shape, 
        dtype=all_obs[-1].dtype)
    start_idx = -min(n_steps, len(all_obs))
    result[start_idx:] = np.array(all_obs[start_idx:])
    if n_steps > len(all_obs):
        # pad
        result[:start_idx] = result[start_idx]
    return np.moveaxis(result, 1, 0)


class MultiStepWrapper(gym.Wrapper):
    def __init__(self, 
            env, 
            n_obs_steps, 
            n_action_steps, 
            max_episode_steps=None,
            reward_agg_method='max'
        ):
        super().__init__(env)
        self._action_space = repeated_space(env.action_space, n_action_steps)
        self._observation_space = repeated_space(env.observation_space, n_obs_steps)
        self.max_episode_steps = max_episode_steps
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.reward_agg_method = reward_agg_method
        self.n_obs_steps = n_obs_steps

        self.obs = deque(maxlen=n_obs_steps+1)
        self.reward = list()
        self.done = list()
        self.info = defaultdict(lambda : deque(maxlen=n_obs_steps+1))
    
    def reset(self):
        """Resets the environment using kwargs."""
        obs = super().reset()

        self.obs = deque([obs], maxlen=self.n_obs_steps+1)
        self.reward = list()
        self.done = list()
        self.info = defaultdict(lambda : deque(maxlen=self.n_obs_steps+1))

        obs = self._get_obs(self.n_obs_steps)
        return obs

    def step(self, action):
        """
        actions: (num_evns, n_action_steps,) + action_shape
        """

        # reverse num_envs and n_action_steps
        actions = np.moveaxis(action, 1, 0)

        for act in actions:
            if len(self.done) > 0 and np.all(self.done[-1]):
                # termination
                break
            observation, reward, done, info = super().step(act)

            self.obs.append(observation)
            self.reward.append(reward)
            if (self.max_episode_steps is not None) \
                and (len(self.reward) >= self.max_episode_steps):
                # truncation
                done = np.ones(self.num_envs, dtype=np.int64)
            self.done.append(done)
            self._add_info(info)

        observation = self._get_obs(self.n_obs_steps)
        reward = aggregate(self.reward, self.reward_agg_method)
        done = aggregate(self.done, 'max')
        info = dict_take_last_n(self.info, self.n_obs_steps)
        return observation, reward, done, info

    def _get_obs(self, n_steps=1):
        """
        Output (n_steps,) + obs_shape
        """
        assert(len(self.obs) > 0)
        if isinstance(self.observation_space, spaces.Box):
            return stack_last_n_obs(self.obs, n_steps)
        elif isinstance(self.observation_space, spaces.Dict):
            result = dict()
            for key in self.observation_space.keys():
                result[key] = stack_last_n_obs(
                    [obs[key] for obs in self.obs],
                    n_steps
                )
            return result
        else:
            raise RuntimeError('Unsupported space type')

    def _add_info(self, info):
        for key, value in info.items():
            self.info[key].append(value)
    
    def get_rewards(self):
        # add transpose to swap 
        # time and env dimension
        # return shape (env, T)
        return np.array(self.reward).T
    
    def get_attr(self, name):
        return getattr(self, name)

    def run_dill_function(self, dill_fn):
        fn = dill.loads(dill_fn)
        return fn(self)
    
    def get_infos(self):
        # add transpose to swap 
        # time and env dimension
        # return shape (env, T)
        result = dict()
        for k, v in self.info.items():
            result[k] = np.array(v).T
        return result


if __name__=='__main__':
    import wandb.sdk.data_types.video as wv
    import wandb
    import pathlib
    import hydra
    import collections
    from omegaconf import DictConfig
    from gym_util.video_recording_wrapper import VideoRecordingWrapper
    from envs.vistac_isaacgym_multiple_env_wrapper import MultipleIsaacEnvWrapper

    @hydra.main(version_base="1.1", 
                config_path="../config", 
                config_name="isaacgym_config")
    def main(cfg: DictConfig):
        # # obtain config for isaacgym environment
        # config_path = 'isaacgym_config.yaml'  # relative to Gym's Hydra search path (cfg dir)
        # isaacgym_cfg = hydra.compose(config_name=config_path)
        
        # Initialize W&B run
        wandb.init(project="video-logging-example")

        output_dir='./'
        max_steps = 200
        n_obs_steps = 3
        n_action_steps = 5
        fps = 10
        crf = 22
        n_test_vis = 3
        steps_per_render = max(10 // fps, 1)
        n_envs = cfg.num_envs
        file_paths = [None] * n_envs
        print("before: ", file_paths)
        for i in range(n_envs):
            enable_render = i < n_test_vis
            if enable_render:
                filename = pathlib.Path(output_dir).joinpath(
                        'media', wv.util.generate_id() + ".mp4")
                filename.parent.mkdir(parents=False, exist_ok=True)
                filename = str(filename)
                file_paths[i] = filename
        print("after: ", file_paths)

        env = MultiStepWrapper(
                VideoRecordingWrapper(
                        MultipleIsaacEnvWrapper(cfg),
                        output_dir=output_dir,
                        n_records=n_test_vis,
                        fps=fps,
                        crf=crf,
                        file_paths=None,
                        steps_per_render=steps_per_render
                ),
                n_obs_steps=n_obs_steps,
                n_action_steps=n_action_steps,
                max_episode_steps=max_steps
        )

        print("Obs space: ", env.observation_space)
        print("Actions space: ", env.action_space)
        print("Num envs: ", env.num_envs)

        obs = env.reset()
        # print(obs.keys())
        print(obs['left_tactile_camera_taxim'].shape, obs['state'].shape)


        pbar = tqdm.tqdm(total=max_steps, desc=f"Eval IsaacgymRunner", 
            leave=False, mininterval=2.0)
        done = False
        while not done:
            random_actions = 2.0 * np.random.rand(n_envs, n_action_steps, env.action_space.shape[1]) - 1.0
            obs, reward, done, info = env.step(random_actions)

            done = np.all(done)  
        
            # update pbar
            pbar.update(random_actions.shape[1])
        pbar.close()

        all_rewards = env.get_rewards()
        all_video_paths = env.render()

        # clear out video buffer
        _ = env.reset()

        # log
        max_rewards = collections.defaultdict(list)
        log_data = dict()
        # results reported in the paper are generated using the commented out line below
        # which will only report and average metrics from first n_envs initial condition and seeds
        # fortunately this won't invalidate our conclusion since
        # 1. This bug only affects the variance of metrics, not their mean
        # 2. All baseline methods are evaluated using the same code
        # to completely reproduce reported numbers, uncomment this line:
        # for i in range(len(self.env_fns)):
        # and comment out this line
        for i in range(n_envs):
            prefix = 'test/'
            video_path = all_video_paths[i]
            max_reward = np.max(all_rewards[i])
            max_rewards[prefix].append(max_reward)
            log_data[prefix+f'sim_max_reward_{i}'] = max_reward

            # visualize sim
            if video_path is not None:
                
                sim_video = wandb.Video(video_path)
                log_data[prefix + f'sim_video_{i}'] = sim_video

        # log aggregate metrics
        for prefix, value in max_rewards.items():
            name = prefix+'mean_score'
            value = np.mean(value)
            log_data[name] = value

        wandb.log(log_data)
        wandb.finish()

        print("Finished")
    
    main()