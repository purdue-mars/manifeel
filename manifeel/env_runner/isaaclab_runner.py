#  Isaac Lab env_runner
#  based on vistac_pih_runner.py

from manifeel.gym_util.multistep_wrapper import MultiStepWrapper
from manifeel.gym_util.video_recording_wrapper import VideoRecordingWrapper
from manifeel.envs.isaaclab_env_wrapper import IsaacLabEnvWrapper

from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.env_runner.base_image_runner import BaseImageRunner
import cv2
import wandb
import numpy as np
import torch
import collections
import tqdm
import time

class IsaacLabRunner(BaseImageRunner):
    def __init__(self,
            output_dir: str,
            shape_meta: dict,
            env_id: str,
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
            headless=True,
            device='cuda:0',
        ):
        super().__init__(output_dir)

        # Build Isaac Lab env config
        isaaclab_cfg = {
            'env_id': env_id,
            'num_envs': n_test if n_test is not None else 10,
            'headless': headless,
            'device': device,
            'shape_meta': shape_meta,
        }

        steps_per_render = max(10 // fps, 1)
        env = MultiStepWrapper(
                VideoRecordingWrapper(
                        IsaacLabEnvWrapper(isaaclab_cfg),
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
        time.sleep(2)
        # past_action = None
        policy.reset()

        pbar = tqdm.tqdm(total=self.max_steps, desc=f"Eval IsaacLabRunner",
            leave=False, mininterval=self.tqdm_interval_sec)
        done = False

        while not done:
            # create obs dict
            np_obs_dict = dict(obs)

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

            # step env
            obs, reward, done, info = env.step(action)
            done = np.all(done)

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
        for i in range(n_envs):
            prefix = 'test/'
            max_reward = np.max(all_rewards[i])
            max_rewards[prefix].append(max_reward)

            # visualize sim
            video_path = all_video_paths[i]
            if video_path is not None:
                sim_video = wandb.Video(video_path)
                log_data[prefix+f'sim_video_{i}'] = sim_video

        # log aggregate metrics
        for prefix, value in max_rewards.items():
            name = prefix+'mean_score'
            value = np.mean(value)
            log_data[name] = value

        return log_data
