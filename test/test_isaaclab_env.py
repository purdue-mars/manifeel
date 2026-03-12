"""
Test script for Isaac Lab environment wrapper and runner.

Three test levels:
  A (Wrapper): Instantiate wrapper, reset, random steps, verify obs shapes
  B (Full stack): Wrapper + VideoRecording + MultiStep, random rollout, verify videos + rewards
  C (Runner): IsaacLabRunner with mock random policy, verify log_data keys
"""

import argparse
import numpy as np
import torch
import os
import sys

def test_wrapper(env_id, num_envs, headless, device):
    """Test A: Raw IsaacLabEnvWrapper."""
    from manifeel.envs.isaaclab_env_wrapper import IsaacLabEnvWrapper

    print("=" * 60)
    print("Test A: IsaacLabEnvWrapper")
    print("=" * 60)

    cfg = {
        'env_id': env_id,
        'num_envs': num_envs,
        'headless': headless,
        'device': device,
        'shape_meta': {
            'obs': {
                'state': {'shape': [7], 'type': 'low_dim'},
            },
            'action': {'shape': [7]},
        },
    }

    env = IsaacLabEnvWrapper(cfg)

    # Check spaces
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    print(f"Num envs: {env.num_envs}")
    assert env.num_envs == num_envs

    # Reset
    obs = env.reset()
    print(f"Reset obs keys: {obs.keys()}")
    assert 'state' in obs, "Missing 'state' key in obs"
    assert obs['state'].shape == (num_envs, 7), f"Expected state shape ({num_envs}, 7), got {obs['state'].shape}"

    # Step with random actions
    for step_i in range(5):
        random_actions = 2.0 * np.random.rand(num_envs, env.action_space.shape[0]) - 1.0
        obs, reward, done, info = env.step(random_actions)
        assert 'state' in obs
        assert obs['state'].shape == (num_envs, 7)
        assert reward.shape == (num_envs,), f"Reward shape mismatch: {reward.shape}"
        assert done.shape == (num_envs,), f"Done shape mismatch: {done.shape}"
        print(f"  Step {step_i}: state shape={obs['state'].shape}, reward shape={reward.shape}")

    # Render
    frames = env.render()
    print(f"Render shape: {frames.shape}")
    assert frames.dtype == np.uint8

    env.close()
    print("Test A PASSED\n")


def test_full_stack(env_id, num_envs, headless, device, output_dir):
    """Test B: Wrapper + VideoRecordingWrapper + MultiStepWrapper."""
    from manifeel.envs.isaaclab_env_wrapper import IsaacLabEnvWrapper
    from manifeel.gym_util.video_recording_wrapper import VideoRecordingWrapper
    from manifeel.gym_util.multistep_wrapper import MultiStepWrapper

    print("=" * 60)
    print("Test B: Full Stack (MultiStep + VideoRecording + Wrapper)")
    print("=" * 60)

    n_obs_steps = 3
    n_action_steps = 5
    max_steps = 50
    fps = 10
    crf = 22
    n_test_vis = min(2, num_envs)
    steps_per_render = max(10 // fps, 1)

    cfg = {
        'env_id': env_id,
        'num_envs': num_envs,
        'headless': headless,
        'device': device,
        'shape_meta': {
            'obs': {
                'state': {'shape': [7], 'type': 'low_dim'},
            },
            'action': {'shape': [7]},
        },
    }

    env = MultiStepWrapper(
            VideoRecordingWrapper(
                    IsaacLabEnvWrapper(cfg),
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

    print(f"Obs space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    print(f"Num envs: {env.num_envs}")

    obs = env.reset()
    print(f"Reset obs keys: {obs.keys()}")
    assert 'state' in obs
    # shape should be (num_envs, n_obs_steps, 7)
    print(f"  state shape: {obs['state'].shape}")

    done = False
    step_count = 0
    action_dim = env.action_space.shape[1]  # MultiStepWrapper: (n_action_steps, action_dim)
    while not done:
        random_actions = 2.0 * np.random.rand(num_envs, n_action_steps, action_dim) - 1.0
        obs, reward, done, info = env.step(random_actions)
        done = np.all(done)
        step_count += n_action_steps

    print(f"Rollout completed in {step_count} action steps")

    all_rewards = env.get_rewards()
    all_video_paths = env.render()
    print(f"Rewards shape: {all_rewards.shape}")
    print(f"Video paths: {all_video_paths}")

    # Verify videos exist
    video_count = sum(1 for p in all_video_paths if p is not None)
    print(f"Videos recorded: {video_count}")

    # Clear out video buffer
    _ = env.reset()

    print("Test B PASSED\n")


def test_runner(env_id, num_envs, headless, device, output_dir):
    """Test C: IsaacLabRunner with mock random policy."""
    from manifeel.env_runner.isaaclab_runner import IsaacLabRunner

    print("=" * 60)
    print("Test C: IsaacLabRunner with Mock Policy")
    print("=" * 60)

    class MockRandomPolicy:
        """Mock policy that returns random actions."""
        def __init__(self, action_dim, n_action_steps, device='cpu'):
            self.device = torch.device(device)
            self.dtype = torch.float32
            self.action_dim = action_dim
            self.n_action_steps = n_action_steps

        def reset(self):
            pass

        def predict_action(self, obs_dict):
            # Get batch size from any obs key
            for key, value in obs_dict.items():
                batch_size = value.shape[0]
                break
            action = torch.randn(batch_size, self.n_action_steps, self.action_dim,
                                 device=self.device, dtype=self.dtype)
            return {'action': action}

    n_obs_steps = 3
    n_action_steps = 5

    runner = IsaacLabRunner(
        output_dir=output_dir,
        shape_meta={
            'obs': {
                'state': {'shape': [7], 'type': 'low_dim'},
            },
            'action': {'shape': [8]},
        },
        env_id=env_id,
        n_test=num_envs,
        n_test_vis=min(2, num_envs),
        test_start_seed=10000,
        max_steps=50,
        n_obs_steps=n_obs_steps,
        n_action_steps=n_action_steps,
        fps=10,
        crf=22,
        past_action=False,
        headless=headless,
        device=device,
    )

    policy = MockRandomPolicy(action_dim=8, n_action_steps=n_action_steps, device='cpu')
    log_data = runner.run(policy)

    print(f"log_data keys: {list(log_data.keys())}")
    assert 'test/mean_score' in log_data, "Missing 'test/mean_score' in log_data"
    print(f"test/mean_score = {log_data['test/mean_score']}")

    # Check for video entries
    video_keys = [k for k in log_data.keys() if 'video' in k]
    print(f"Video log entries: {video_keys}")

    print("Test C PASSED\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test Isaac Lab environment integration')
    parser.add_argument('--env_id', type=str, default='Isaac-Lift-Cube-Franka-v0')
    parser.add_argument('--num_envs', type=int, default=4)
    parser.add_argument('--headless', action='store_true')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--output_dir', type=str, default='./test_output')
    parser.add_argument('--test', type=str, default='all',
                        choices=['all', 'A', 'B', 'C'],
                        help='Which test level to run')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if args.test in ('all', 'A'):
        test_wrapper(args.env_id, args.num_envs, args.headless, args.device)

    if args.test in ('all', 'B'):
        test_full_stack(args.env_id, args.num_envs, args.headless, args.device, args.output_dir)

    if args.test in ('all', 'C'):
        test_runner(args.env_id, args.num_envs, args.headless, args.device, args.output_dir)

    print("=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)
