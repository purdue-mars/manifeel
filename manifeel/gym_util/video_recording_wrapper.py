import gym
import numpy as np
import pathlib
from diffusion_policy.real_world.video_recorder import VideoRecorder
import wandb.sdk.data_types.video as wv

class VideoRecordingWrapper(gym.Wrapper):
    def __init__(self, 
            env, 
            output_dir,
            n_records=6,
            fps=10,
            crf=22,
            mode='rgb_array',
            file_paths=list(),
            steps_per_render=1,
            **kwargs
        ):
        """
        When file_path is None, don't record.
        """
        super().__init__(env)
        
        self.mode = mode
        self.output_dir = output_dir
        self.render_kwargs = kwargs
        self.steps_per_render = steps_per_render
        self.file_paths = file_paths
        
        assert (n_records <= self.num_envs)
        # Generate record mask
        self.record_mask = [i < n_records for i in range(self.num_envs)]

        self.video_recorders = [
            VideoRecorder.create_h264(
                fps=fps,
                codec='h264',
                input_pix_fmt='rgb24',
                crf=crf,
                thread_type='FRAME',
                thread_count=1
            ) 
            if use_recorder else None
            for use_recorder in self.record_mask
        ]
        self.step_count = 0

    def reset(self, **kwargs):
        obs = super().reset(**kwargs)

        # create filepaths
        self.file_paths = [
            pathlib.Path(self.output_dir).joinpath(
                'media', wv.util.generate_id() + ".mp4")
            if use_recorder else None
            for use_recorder in self.record_mask
        ]
        self.file_paths[0].parent.mkdir(parents=False, exist_ok=True)
        self.file_paths = [str(path) if path else None for path in self.file_paths]

        self.step_count = 1
        for video_recoder in self.video_recorders:
            if video_recoder:
                video_recoder.stop()

        return obs
    
    def step(self, action):
        result = super().step(action)
        self.step_count += 1
        if (self.step_count % self.steps_per_render) == 0:
            frames = self.env.render(
                mode=self.mode, **self.render_kwargs)
            for recorder, frame, file_path in zip(self.video_recorders, frames, self.file_paths):
                if file_path:  # Skip None file paths
                    process_video_recorder(recorder, frame, file_path)
        return result
    
    def render(self, mode='rgb_array', **kwargs):
        for video_recoder in self.video_recorders:
            if video_recoder and video_recoder.is_ready():
                video_recoder.stop()
        return self.file_paths

def process_video_recorder(recorder, frame, file_path):
    """
    Handles starting the recorder if not ready and writing a frame to it.
    """
    if not recorder.is_ready():
        recorder.start(file_path)
    assert frame.dtype == np.uint8, "Frame must be of type uint8"
    recorder.write_frame(frame)

if __name__=='__main__':
    import wandb.sdk.data_types.video as wv
    import wandb
    import pathlib
    import hydra
    from omegaconf import DictConfig
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
        fps = 10
        crf = 22
        n_test_vis = 3
        steps_per_render = max(10 // fps, 1)
        n_envs = cfg.num_envs

        env = VideoRecordingWrapper(
                        MultipleIsaacEnvWrapper(cfg),
                        output_dir=output_dir,
                        n_records=n_test_vis,
                        fps=fps,
                        crf=crf,
                        file_paths=None,
                        steps_per_render=steps_per_render
                    )
        
        env.reset()

        for _ in range(50):
            random_actions = 2.0 * np.random.rand(n_envs, env.action_space.shape[0]) - 1.0
            obs, reward, reset, _ = env.step(random_actions)
            tactile_rgb_image = obs['left_tactile_camera_taxim']
            state = obs['state']
            # print(obs.keys())
            print(tactile_rgb_image.shape, state.shape)

        video_paths = env.render()
        log_data = dict()
        for i in range(n_envs):
            # visualize sim
            video_path = video_paths[i]
            if video_path is not None:
                prefix = 'test/'
                sim_video = wandb.Video(video_path)
                log_data[prefix + f'sim_video_{i}'] = sim_video
        wandb.log(log_data)
        wandb.finish()
        print("Finished")
    
    main()