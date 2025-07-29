from spacemouse import SpaceMouse
from utils.input_utils import input2action
# from tf.transformations import quaternion_from_euler
import threading
import numpy as np

def get_action(space_mouse):
    d_action, _ = input2action(space_mouse)
    d_pos = d_action[0:3]
    d_rpy = d_action[3:6]
    gripper_action = float(space_mouse.single_click_and_hold)
    action = np.concatenate([d_pos, d_rpy, [gripper_action]]) 
    # d_quat = quaternion_from_euler(*d_rpy)  # [qx, qy, qz, qw]

    # action = np.concatenate([d_pos, d_quat])
    # action = d_pos
    
    return action

# Initialize flags
terminated = False
# Define a function to listen for keyboard input
def listen_for_input():
    global terminated
    global start
    while True:
        user_input = input()  # Wait for User's input
        if user_input.lower() == 'q':
            terminated = True
            print("Terminated set to True.")

# Start control of tele-device
space_mouse = SpaceMouse(vendor_id=9583, product_id=50741, pos_sensitivity=0.1, rot_sensitivity=1.0)
space_mouse.start_control()

# Create and start a new thread to listen for keyboard input
input_thread = threading.Thread(target=listen_for_input)
input_thread.daemon = True  # Set the thread as a daemon thread
input_thread.start()

while not terminated:
    action = get_action(space_mouse=space_mouse)
    print(f"action {action.shape},action {action}")
