from manifeel.envs.manifeel_env_wrapper_teleop import ManiFeelTeleopEnvWrapper
from diffusion_policy.common.replay_buffer import ReplayBuffer

from manifeel.utils.teleop_utils.spacemouse import SpaceMouse
from manifeel.utils.input_utils import input2action

import numpy as np
import cv2
import hydra
import threading
from omegaconf import DictConfig

import os
os.environ["HYDRA_FULL_ERROR"] = "1"

def get_action(space_mouse):
    d_action, _ = input2action(space_mouse)
    d_pos = d_action[0:3]
    d_rpy = d_action[3:6]
    action = np.concatenate([d_pos, d_rpy])
    return action

# Initialize flags
done = False
retry = False
# Define a function to listen for keyboard input
def listen_for_input():
    global done
    global retry
    while True:
        user_input = input()  # Wait for User's input
        if user_input.lower() == 'd':
            done = True
            print("Done the current demo")
        elif user_input.lower() == 'r':
            print("Retry the current demo")
            retry = True

def display_demo_controls():
        """
        Method to pretty print controls.
        """

        def print_command(char, info):
            char += " " * (30 - len(char))
            print("{}\t{}".format(char, info))

        print("")
        print_command("Press", "Command")
        print_command("d", "finish one demo/epoch per task")
        print_command("r", "retry the current demo/epoch")
        print_command("p", "pause the current demo")
        print_command("ESC", "quit")
        print("")

@hydra.main(version_base="1.1", 
            config_path="../../manipulation/IsaacGymEnvs/isaacgymenvs/cfg", 
            config_name="config")
def main(cfg: DictConfig):
    
    # Pass the config explicitly
    env = ManiFeelTeleopEnvWrapper(cfg)

    print("Observation space is", env.observation_space)
    print("Action space is", env.action_space)

    # create replay buffer in read-write mode
    output = '../data/test_data'
    replay_buffer = ReplayBuffer.create_from_path(output, mode='a')

    # Start control of tele-device
    space_mouse = SpaceMouse(vendor_id=9583, product_id=50741, pos_sensitivity=1.0, rot_sensitivity=1.0)
    space_mouse.start_control()

    display_demo_controls()

    # Create and start a new thread to listen for keyboard input
    input_thread = threading.Thread(target=listen_for_input)
    input_thread.daemon = True  # Set the thread as a daemon thread
    input_thread.start()
    
    global retry
    global done

    # episode-level while loop
    for _ in range(2):
        episode = list()
        # record in seed order, starting with 0
        seed = replay_buffer.n_episodes
        print(f'Starting seed {seed}')

        # set seed for env
        env.seed(seed)

        # reset env and get observations (including info and render for recording)
        obs = env.reset()

        # loop state
        retry = False
        done = False
        # start = time.time()
        while not done:

            # handle control flow
            if retry:
                break

            left_tactile_rgb_image = obs['left_tactile_camera_taxim'][0].cpu().numpy()
            right_tactile_rgb_image = obs['right_tactile_camera_taxim'][0].cpu().numpy()
            wrist_rgb_image = obs['wrist'][0].cpu().numpy()
            wrist2_rgb_image = obs['wrist_2'][0].cpu().numpy()
            front_rgb_image = obs['front'][0].cpu().numpy()
            side_rgb_image = obs['side'][0].cpu().numpy()
            ee_pos = obs['ee_pos'][0].cpu().numpy()
            ee_quat = obs['ee_quat'][0].cpu().numpy()
            state = np.concatenate([ee_pos, ee_quat])

            actions = get_action(space_mouse=space_mouse)
            # print(actions)
            # actions = 2.0 * np.random.rand(6) - 1.0
            
            data = {
                    'tactile_img_left': left_tactile_rgb_image,
                    'tactile_img_right': right_tactile_rgb_image,
                    'wrist_img': wrist_rgb_image,
                    'wrist_img_2': wrist2_rgb_image,
                    'front_img': front_rgb_image,
                    'side_img': side_rgb_image,
                    'state': state,
                    'action': np.float32(actions),
                }
            episode.append(data)

            # step env and get observations
            obs, reward, reset, info = env.step(actions)


            print(f"🖼Reward: {reward[0].cpu().numpy()}")

            # print(obs['ee_pos'][0].cpu().numpy(), 
            #       obs['ee_quat'].cpu().numpy())

            # print(reset.cpu().numpy(), info['time_outs'].cpu().numpy())
            # print(tactile_rgb_image.shape, wrist_rgb_image.shape)

            # cv2.imshow('Front Images', np.moveaxis(front_rgb_image, 0, -1))
            # print(side_rgb_image.shape, type(side_rgb_image))

            # cv2.imshow('Side Images', side_rgb_image[:,:,::-1])

            # cv2.imshow('Left Tactile Images', np.moveaxis(left_tactile_rgb_image, 0, -1))
            # cv2.imshow('Right Tactile Images', np.moveaxis(right_tactile_rgb_image, 0, -1))
            # # cv2.imshow('Wrist Images', np.moveaxis(wrist_rgb_image, 0, -1))
            # cv2.imshow('Wrist2 Images', np.moveaxis(wrist2_rgb_image, 0, -1))

            # print(f'🖼left_tactile_rgb_image.shape, {left_tactile_rgb_image.shape} left_tactile_rgb_image type {type(left_tactile_rgb_image)}')

            cv2.waitKey(1)

        # end = time.time()
        # print("estimated dt: {}".format((end-start)/100))

        if not retry:
            # save episode buffer to replay buffer (on disk)
            data_dict = dict()
            for key in episode[0].keys():
                data_dict[key] = np.stack(
                    [x[key] for x in episode])
            replay_buffer.add_episode(data_dict, compressors='disk')
            print(f'saved seed {seed}')
        else:
            print(f'retry seed {seed}')

if __name__ == '__main__':
    main()