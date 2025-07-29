#  Tactile Sensing and Learning -> TacSL
#  Vision-Tactile Peg-in-Hole Task
#  VT_PiH task env_runner
#  based on pusht_image_runner.py

from manifeel.gym_util.multistep_wrapper import MultiStepWrapper
from manifeel.gym_util.video_recording_wrapper import VideoRecordingWrapper
from manifeel.envs.manifeel_env_wrapper import ManiFeelEnvWrapper

from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.env_runner.base_image_runner import BaseImageRunner
import cv2
import hydra
from omegaconf import OmegaConf
import wandb
import numpy as np
import torch
import collections
import tqdm

class ManifeelRunner(BaseImageRunner):
    def __init__(self,
            output_dir: str,
            shape_meta: dict,
            isaacgym_cfg_name: str,
            n_test=22,
            n_test_vis=6,
            test_start_seed=10000,
            max_steps=200,
            n_obs_steps=8,
            n_action_steps=8,
            fps=10,
            crf=22,
            past_action=False,
            tqdm_interval_sec=5.0,
            tactile_size=[256, 256],
            zero_action_dims=None
        ):
        super().__init__(output_dir)

        # obtain config for isaacgym environment
        # path relative to Gym's Hydra search path (cfg dir)
        isaacgym_cfg = hydra.compose(config_name=isaacgym_cfg_name)
        isaacgym_cfg['shape_meta'] = OmegaConf.create(shape_meta)

        if n_test is not None:
            # override number of evaluated environments
            isaacgym_cfg.num_envs = n_test
        
        steps_per_render = max(10 // fps, 1)
        env = MultiStepWrapper(
                VideoRecordingWrapper(
                        ManiFeelEnvWrapper(isaacgym_cfg),
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

        self.env = env
        self.test_seed = test_start_seed
        self.fps = fps
        self.crf = crf
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.past_action = past_action
        self.max_steps = max_steps
        self.tqdm_interval_sec = tqdm_interval_sec
        self.tactile_size = tactile_size
        self.zero_action_dims = zero_action_dims
    
    def run(self, policy: BaseImagePolicy):
        device = policy.device
        dtype = policy.dtype
        env = self.env

        # plan for rollout
        n_envs = env.num_envs

        print("Number of envs runner: ", n_envs)

        # allocate data
        all_video_paths = [None] * n_envs
        all_rewards = [None] * n_envs

        # start rollout
        env.seed(self.test_seed)
        obs = env.reset()
        past_action = None
        policy.reset()

        pbar = tqdm.tqdm(total=self.max_steps, desc=f"Eval IsaacgymRunner", 
            leave=False, mininterval=self.tqdm_interval_sec)
        done = False

        while not done:
            # create obs dict
            np_obs_dict = dict(obs)

            for key in ['left_tactile_camera_taxim', 'right_tactile_camera_taxim']:
                if key in np_obs_dict:
                    # print(f"{key} shape before resizing: ", np_obs_dict[key].shape)
                    
                    # Shape should be (batch, T, C, H, W), range [0, 1]
                    data_temp = np_obs_dict[key]
                    B, T, C, H, W = data_temp.shape

                    # Initialize resized data array
                    resized_data = np.zeros((B, T, C, self.tactile_size[0], self.tactile_size[1]), dtype=data_temp.dtype)

                    for b in range(B):
                        for t in range(T):
                            for c in range(C):
                                resized_data[b, t, c] = cv2.resize(data_temp[b, t, c], (self.tactile_size[1], self.tactile_size[0]), interpolation=cv2.INTER_LINEAR)

                    np_obs_dict[key] = resized_data
                    # print(f"{key} shape after resizing: ", np_obs_dict[key].shape)


            # device transfer
            obs_dict = dict_apply(np_obs_dict, 
                lambda x: torch.from_numpy(x).to(
                    device=device))

            # run policy
            with torch.no_grad():
                action_dict = policy.predict_action(obs_dict)

            # device_transfer
            np_action_dict = dict_apply(action_dict,
                lambda x: x.detach().to('cpu').numpy())

            action = np_action_dict['action']
            if self.zero_action_dims is not None:
                action[..., self.zero_action_dims] = 0.

            # step env
            obs, reward, done, info = env.step(action)
            done = np.all(done)
            # past_action = action

            # update pbar
            pbar.update(action.shape[1])
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
            max_reward = np.max(all_rewards[i])
            max_rewards[prefix].append(max_reward)
            # log_data[prefix+f'sim_max_reward_{i}'] = max_reward

            # visualize sim
            video_path = all_video_paths[i]
            if video_path is not None:
                sim_video = wandb.Video(video_path)
                log_data[prefix+f'sim_video_{i}'] = sim_video
                log_data[prefix+f'sim_video_{i}'] = sim_video

        # log aggregate metrics
        for prefix, value in max_rewards.items():
            name = prefix+'mean_score'
            value = np.mean(value)
            log_data[name] = value

        return log_data

def test():
    import hydra
    from omegaconf import OmegaConf
    OmegaConf.register_new_resolver("eval", eval, replace=True)

    # Initialize W&B run
    wandb.init(project="video-logging-example")

    with hydra.initialize('../config'):
        cfg = hydra.compose('train_diffusion_vision_tactile_workspace')
        OmegaConf.resolve(cfg)
        runner = hydra.utils.instantiate(cfg.task.env_runner,
                                    output_dir='./')

    envs = runner.env
    # runner.run(policy=None)
    observations = envs.reset()
    print("obs space: ", envs.observation_space)
    print(observations.keys())

    # plan for rollout
    n_envs = envs.num_envs

    # allocate data
    all_video_paths = [None] * n_envs

    for i in range(10):
        actions = 2.0 * np.random.rand(n_envs, envs.action_space.shape[0], envs.action_space.shape[1]) - 1.0
        # actions = envs.action_space.sample()
        print(actions.shape, type(actions), actions.dtype)
        observations, _, _, _ = envs.step(actions)

    all_video_paths = envs.render()

    # visualize sim
    for i in range(n_envs):
        # print(observations['socket_pos'][i][0])
        video_path = all_video_paths[i]
        if video_path is not None:
            sim_video = wandb.Video(video_path)
            # Log a video
            video_name = f'sim_video_{i}'
            wandb.log({video_name: sim_video})

    # Finish the W&B run
    wandb.finish()

    print("Finished")

if __name__ == '__main__':
    test()