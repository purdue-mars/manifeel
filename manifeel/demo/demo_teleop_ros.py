from manifeel.envs.manifeel_env_wrapper_teleop import ManiFeelTeleopEnvWrapper
from manifeel.utils.shear_tactile_viz_utils import visualize_tactile_shear_image, visualize_penetration_depth
from diffusion_policy.common.replay_buffer import ReplayBuffer

from manifeel.utils.teleop_utils.spacemouse import SpaceMouse
from manifeel.utils.input_utils import input2action

import numpy as np
import cv2
import hydra
import threading
from omegaconf import DictConfig

from pynput import keyboard
import time
import matplotlib.pyplot as plt

plt.ion()  
fig, ax = plt.subplots()
reward_history = []
reward_boundary_max = 0.1 
reward_boundary_min = -0.6
check_lower = -0
check_upper = -0.
# gear task
# reward_boundary_max = -0.655 
# reward_boundary_min = -0.675
# check_lower = -0.6668
# check_upper = -0.6665

print_time_step = 1000
print_time_step = round(print_time_step)
x_data = list(range(print_time_step))
line, = ax.plot(x_data, [0]*print_time_step, 'bo-', markersize=5)
ax.set_xlim(0, print_time_step - 1)
ax.set_ylim(reward_boundary_min, reward_boundary_max) 
ax.set_xlabel("Time step")
ax.set_ylabel("Reward")

def update_plot(reward):
    reward_value = reward[0].cpu().numpy().item()
    reward_value = np.clip(reward_value, reward_boundary_min, reward_boundary_max)  
    reward_value = round(reward_value, 6)

    if len(reward_history) >= print_time_step:
        reward_history.pop(0)  
    reward_history.append(reward_value)

    colors = ['ro-' if check_upper <= r <= check_lower else 'bo-' for r in reward_history]
    ax.clear()
    ax.set_xlim(0, print_time_step - 1)
    ax.set_ylim(reward_boundary_min, reward_boundary_max)
    ax.set_xlabel("Time step")
    ax.set_ylabel("Reward")

    for i in range(len(reward_history)):
        color = 'r' if check_lower <= reward_history[i] <= check_upper else 'b'
        ax.plot(i, reward_history[i], f'{color}o', markersize=5) 
    plt.draw()
    plt.pause(0.1)  

gripper_action = 0.5  
gripper_min = 0.0 
gripper_max = 0.8
gripper_step = 0.01
left_pressed = False
right_pressed = False
def on_press(key):
    global left_pressed, right_pressed
    try:
        if key == keyboard.Key.left:
            left_pressed = True  # left pressed
        elif key == keyboard.Key.right:
            right_pressed = True  # right pressed
    except AttributeError:
        pass
def on_release(key):
    global left_pressed, right_pressed
    if key == keyboard.Key.left:
        left_pressed = False  # left release
    elif key == keyboard.Key.right:
        right_pressed = False  # right release
def start_keyboard_listener():
    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.daemon = True  
    listener.start()
def gripper_control_loop():
    global gripper_action
    while True:
        if left_pressed:
            gripper_action = max(gripper_min, gripper_action - gripper_step)
            # print(f"🔒 gripper close: {gripper_action:.3f}")
        if right_pressed:
            gripper_action = min(gripper_max, gripper_action + gripper_step)
            # print(f"🔨 gripper open: {gripper_action:.3f}")
        time.sleep(0.1)
gripper_thread = threading.Thread(target=gripper_control_loop, daemon=True)
gripper_thread.start()
# gripper_action = 0.5
# def on_press(key):
#     gripper_open = 0.5  
#     gripper_closed = 0.005
#     # open_act
#     global gripper_action
#     try:
#         if key == keyboard.Key.left:     # left -> close panda gripper
#             gripper_action = gripper_closed
#         elif key == keyboard.Key.right:  # right -> open gripper
#             gripper_action = gripper_open
#     except AttributeError:
#         pass
# def on_release(key):
#     pass
# def start_keyboard_listener():
#     listener = keyboard.Listener(on_press=on_press, on_release=on_release)
#     listener.daemon = True  
#     listener.start()
####################################
# using ros topic to load img #
####################################

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

rospy.init_node('image_publisher', anonymous=True)
#image_pub = rospy.Publisher('/camera/side_image', Image, queue_size=10)
image_pub_side = rospy.Publisher('/camera/side_image', Image, queue_size=10)
image_pub_front = rospy.Publisher('/camera/front_image', Image, queue_size=10)
image_pub_wrist = rospy.Publisher('/camera/wrist_image', Image, queue_size=10)
image_pub_wrist2 = rospy.Publisher('/camera/wrist2_image', Image, queue_size=10)
image_pub_tactile_left = rospy.Publisher('/camera/tactile_left_image', Image, queue_size=10)
image_pub_tactile_right = rospy.Publisher('/camera/tactile_right_image', Image, queue_size=10)
image_pub_tactile_shear_left = rospy.Publisher('/camera/tactile_shear_left_image', Image, queue_size=10)
image_pub_tactile_shear_right = rospy.Publisher('/camera/tactile_shear_right_image', Image, queue_size=10)

bridge = CvBridge()

####################################
# using ros topic to load img #
####################################


def get_action(space_mouse):
    d_action, _ = input2action(space_mouse)
    d_pos = d_action[0:3]
    d_rpy = d_action[3:6]
    # 🤖 🤖 🤖
    action = np.concatenate([d_pos, d_rpy])
    # action = np.concatenate([d_pos, d_rpy, [gripper_action]])
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
            config_path="../config", 
            config_name="isaacgym_config_gui")
def main(cfg: DictConfig):
    
    # Pass the config explicitly
    env = ManiFeelTeleopEnvWrapper(cfg)

    print("Observation space is", env.observation_space)
    print("Action space is", env.action_space)

    # create replay buffer in read-write mode
    ################  saving path #################
    output = '../../data/test1'

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
    for _ in range(256):

        print_count = 0
        print(f"⛳ print_count {print_count}" )
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

            # tactile_force_field_left = obs['tactile_force_field_left'][0].cpu().numpy()
            tactile_force_field_right = obs['tactile_force_field_right'][0].cpu().numpy()
            # tactile_depth_left = obs['tactile_depth_left'][0].cpu().numpy()
            tactile_depth_right = obs['tactile_depth_right'][0].cpu().numpy()

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
            
            data = {
                    # 'tactile_force_field_left': tactile_force_field_left,
                    'tactile_force_field_right': tactile_force_field_right,
                    # 'tactile_depth_left': tactile_depth_left,
                    'tactile_depth_right': tactile_depth_right,
                    'left_tactile_camera_taxim': left_tactile_rgb_image,
                    'right_tactile_camera_taxim': right_tactile_rgb_image,
                    'wrist': wrist_rgb_image,
                    'wrist_2': wrist2_rgb_image,
                    'front': front_rgb_image,
                    'side': side_rgb_image,
                    'state': state,
                    'action': np.float32(actions),
                }
            episode.append(data)

            # step env and get observations
            obs, reward, reset, info = env.step(actions)

            # for USB, reward should be (-0.30 to -0.35) 
            # for power plug, reward should be (-0.035 to -0.05)

            # power_2prongs_shape1 (-0.009 to -0.004)
            # power_2prongs_shape2 (-0.03 to -0.021)
            # power_2prongs_shape3 (-0.021 to -0.018)
            # USB_plug (-0.027, -0.03)
            # USB_plug2 (-0.024, -0.036)
            # USB_plug3 (-0.0297, -0.032)
            # gear with handle (-0.6668, -0.6658)

            ranges = {
                'USB': {'lower': -0.35, 'upper': -0.30},
                'power': {'lower': -0.051, 'upper': -0.048}
            }

            choice = 'USB'

            lower = ranges[choice]['lower']
            upper = ranges[choice]['upper']
            in_range = (reward >= lower) & (reward <= upper)
            in_range_val = ((reward >= lower) & (reward <= upper)).float()
            print(f"🐝Reward: {reward[0].cpu().numpy():.6f}")     ######  -0.005
            update_plot(reward)
            # print(f"in_range_val  {in_range_val}")
            # print(f"🚀Reward in range: {in_range.item()}")
            # print(f"✅Done signal: {reset[0].cpu().numpy()}")
            # print(f"print count {print_count}")
            
            # if in_range.item() == 1:
            #     consecutive_count += 1
            #     if consecutive_count == 3 and print_count < 3:
            #         print_count += 1
            #         if print_count ==2:
            #             #print(f"✅in_range_val {in_range_val}")
            #             # print(f"✅Done signal: {reset[0].cpu().numpy()}")
                    
            #         consecutive_count = 0  
            # else:
            #     consecutive_count = 0  

            # if print_count >= 3:
            #     pass  
            
            # ensure img uint8 (0-255)
            side_rgb_image = (side_rgb_image * 255).astype(np.uint8)
            front_rgb_image = (front_rgb_image * 255).astype(np.uint8)
            wrist_rgb_image = (wrist_rgb_image * 255).astype(np.uint8)
            wrist2_rgb_image = (wrist2_rgb_image * 255).astype(np.uint8)
            left_tactile_rgb_image = (left_tactile_rgb_image * 255).astype(np.uint8)
            right_tactile_rgb_image = (right_tactile_rgb_image * 255).astype(np.uint8)
            
            # left_shear_img = visualize_tactile_shear_image(
            #             tactile_force_field_left[..., 0],
            #             tactile_force_field_left[..., 1:],
            #             normal_force_threshold=0.004,
            #             shear_force_threshold=0.0010, #0.0005
            #             resolution=25)
            # left_shear_img = visualize_tactile_shear_image(
            #             tactile_force_field_left[..., 0],
            #             tactile_force_field_left[..., 1:])

            # left_shear_img = np.moveaxis(left_shear_img[:, :, ::-1], 1, 0)

            # left_shear_img = (left_shear_img * 255).astype(np.uint8)
            # ros_shear_left = bridge.cv2_to_imgmsg(left_shear_img, encoding="bgr8")

            # right_shear_img = visualize_tactile_shear_image(
            #             tactile_force_field_right[..., 0],
            #             tactile_force_field_right[..., 1:],
            #             normal_force_threshold=0.004,
            #             shear_force_threshold=0.0010, 
            #             #0.0005
            #             resolution=25)
            right_shear_img = visualize_tactile_shear_image(
                        tactile_force_field_right[..., 0],
                        tactile_force_field_right[..., 1:])

            # right_shear_img = np.moveaxis(right_shear_img[:, :, ::-1], 1, 0)
            right_shear_img = (right_shear_img * 255).astype(np.uint8)
            ros_shear_right = bridge.cv2_to_imgmsg(right_shear_img, encoding="bgr8")


            ros_side = bridge.cv2_to_imgmsg(side_rgb_image, encoding="bgr8")
            ros_front = bridge.cv2_to_imgmsg(front_rgb_image, encoding="bgr8")
            ros_wrist = bridge.cv2_to_imgmsg(wrist_rgb_image, encoding="bgr8")
            ros_wrist2 = bridge.cv2_to_imgmsg(wrist2_rgb_image, encoding="bgr8")
            ros_tactile_left = bridge.cv2_to_imgmsg(left_tactile_rgb_image, encoding="bgr8")
            ros_tactile_right = bridge.cv2_to_imgmsg(right_tactile_rgb_image, encoding="bgr8")
            

            image_pub_side.publish(ros_side)
            image_pub_front.publish(ros_front)
            image_pub_wrist.publish(ros_wrist)
            image_pub_wrist2.publish(ros_wrist2)
            image_pub_tactile_left.publish(ros_tactile_left)
            image_pub_tactile_right.publish(ros_tactile_right)

            # image_pub_tactile_shear_left.publish(ros_shear_left)
            image_pub_tactile_shear_right.publish(ros_shear_right)

            #rospy.sleep(0.1)  # ctrl rostopic fre

            '''
            test_image = cv2.imread("/home/quan/Downloads/UR_robot_test.jpeg")
            #
            ros_image = bridge.cv2_to_imgmsg(test_image, encoding="bgr8")
            image_pub.publish(ros_image)
            '''
            # rospy.sleep(0.1)
            # cv2.waitKey(1)
            
            # cv2.imshow('Left Tactile Images', np.moveaxis(left_tactile_rgb_image, 0, -1))
            # cv2.imshow('Right Tactile Images', np.moveaxis(right_tactile_rgb_image, 0, -1))
            # # cv2.imshow('Wrist Images', np.moveaxis(wrist_rgb_image, 0, -1))
            # cv2.imshow('Wrist2 Images', np.moveaxis(wrist2_rgb_image, 0, -1))

            #print(f'🖼left_tactile_rgb_image.shape, {left_tactile_rgb_image.shape} left_tactile_rgb_image type {type(left_tactile_rgb_image)}')

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
    start_keyboard_listener()
    main()