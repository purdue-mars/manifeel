import numpy as np
from scipy.spatial.transform import Rotation as R, Slerp
from manifeel.utils.space_utils import (
    ortho6d_to_rotation_matrix,
    pose_3d_9d_to_homo_matrix_batch,
    homo_matrix_to_pose_9d_batch
)
from manifeel.utils.real_world_transforms import RealWorldTransforms

def interpolate_actions_with_ratio(actions: np.ndarray, N: int):
    """
    Perform linear interpolation between frames with a specified ratio N.

    Args:
        actions: numpy array with shape (T, D) where T is number of timesteps
                and D is the dimension of actions
        N: integer, the multiplication factor for number of frames
           (N=2 doubles the frames, N=3 triples, etc.)

    Returns:
        interpolated_actions: numpy array with shape (N*T, D)
    """
    T, D = actions.shape

    # Create empty array for result
    interpolated_actions = np.zeros((N * T, D), dtype=actions.dtype)

    # Fill in original frames
    interpolated_actions[::N] = actions

    if D == 4: # (x, y, z, gripper_width)
        cartesian_dim = np.arange(4)
        rotation_dim = None
    elif D == 10: # (x, y, z, rx1, rx2, rx3, ry1, ry2, ry3)
        cartesian_dim = np.concatenate([np.arange(3), np.arange(9, 10)])
        rotation_dim = np.arange(3, 9)
    else:
        raise NotImplementedError

    # For each pair of consecutive original frames
    for i in range(T - 1):
        # Generate N-1 interpolated frames between each pair
        for j in range(1, N):
            # Calculate interpolation ratio
            ratio = j / N
            # Linear interpolation: start*(1-ratio) + end*ratio
            interpolated_actions[i * N + j, cartesian_dim] = (1 - ratio) * actions[i, cartesian_dim] + ratio * actions[i + 1, cartesian_dim]
            # Spherical Linear Interpolation for rotation
            if rotation_dim is not None:
                assert len(rotation_dim) == 6, "Only support 6D rotation now"
                start_rotation = ortho6d_to_rotation_matrix(actions[i : i + 1, rotation_dim])[0]
                end_rotation = ortho6d_to_rotation_matrix(actions[i + 1 : i + 2, rotation_dim])[0]
                start_quaternion = R.from_matrix(start_rotation).as_quat()
                end_quaternion = R.from_matrix(end_rotation).as_quat()
                slerp = Slerp([0, 1], R.from_quat([start_quaternion, end_quaternion]))
                interpolated_quaternion = slerp(ratio)
                interpolated_rotation = interpolated_quaternion.as_matrix()
                interpolated_actions[i * N + j, rotation_dim] = interpolated_rotation[:3, :2].T.flatten()

    # Fill the remaining frames at the end by repeating the last frame
    interpolated_actions[(T - 1) * N + 1:] = actions[-1]

    return interpolated_actions

def absolute_actions_to_relative_actions(actions: np.ndarray, base_absolute_action=None):
    actions = actions.copy()
    T, D = actions.shape

    if D == 3 or D == 4:  # (x, y, z(, gripper_width))
        tcp_dim_list = [np.arange(3)]
    elif D == 6 or D == 8:  # (x_l, y_l, z_l, x_r, y_r, z_r(, gripper_width_l, gripper_width_r))
        tcp_dim_list = [np.arange(3), np.arange(3, 6)]
    elif D == 9 or D == 10:  # (x, y, z, rx1, rx2, rx3, ry1, ry2, ry3(, gripper_width))
        tcp_dim_list = [np.arange(9)]
    elif D == 18 or D == 20:  # (x_l, y_l, z_l, rotation_l, x_r, y_r, z_r, rotation_r(, gripper_width_l, gripper_width_r))
        tcp_dim_list = [np.arange(9), np.arange(9, 18)]
    else:
        raise NotImplementedError

    if base_absolute_action is None:
        base_absolute_action = actions[0].copy()
    for tcp_dim in tcp_dim_list:
        assert len(tcp_dim) == 3 or len(tcp_dim) == 9, "Only support 3D or 9D tcp pose now"
        base_tcp_pose_mat = pose_3d_9d_to_homo_matrix_batch(base_absolute_action[None, tcp_dim])
        actions[:, tcp_dim] = homo_matrix_to_pose_9d_batch(np.linalg.inv(base_tcp_pose_mat) @ pose_3d_9d_to_homo_matrix_batch(
            actions[:, tcp_dim]))[:, :len(tcp_dim)]

    return actions

def relative_actions_to_absolute_actions(actions: np.ndarray, base_absolute_action: np.ndarray):
    actions = actions.copy()
    T, D = actions.shape

    if D == 3 or D == 4:  # (x, y, z(, gripper_width))
        tcp_dim_list = [np.arange(3)]
    elif D == 6 or D == 8:  # (x_l, y_l, z_l, x_r, y_r, z_r(, gripper_width_l, gripper_width_r))
        tcp_dim_list = [np.arange(3), np.arange(3, 6)]
    elif D == 9 or D == 10:  # (x, y, z, rx1, rx2, rx3, ry1, ry2, ry3(, gripper_width))
        tcp_dim_list = [np.arange(9)]
    elif D == 18 or D == 20:  # (x_l, y_l, z_l, rotation_l, x_r, y_r, z_r, rotation_r(, gripper_width_l, gripper_width_r))
        tcp_dim_list = [np.arange(9), np.arange(9, 18)]
    else:
        raise NotImplementedError

    for tcp_dim in tcp_dim_list:
        assert len(tcp_dim) == 3 or len(tcp_dim) == 9, "Only support 3D or 9D tcp pose now"
        base_tcp_pose_mat = pose_3d_9d_to_homo_matrix_batch(base_absolute_action[None, tcp_dim])
        actions[:, tcp_dim] = homo_matrix_to_pose_9d_batch(base_tcp_pose_mat @ pose_3d_9d_to_homo_matrix_batch(
            actions[:, tcp_dim]))[:, :len(tcp_dim)]

    return actions

def get_inter_gripper_actions(obs_dict, lowdim_keys: dict, transforms: RealWorldTransforms):
    extra_obs_dict = dict()
    if 'left_robot_wrt_right_robot_tcp_pose' in lowdim_keys:
        base_absolute_action_in_world = homo_matrix_to_pose_9d_batch(
            transforms.right_robot_base_to_world_transform @ pose_3d_9d_to_homo_matrix_batch(
                obs_dict['right_robot_tcp_pose'][-1:])
        )[0]
        left_robot_tcp_pose_in_world = homo_matrix_to_pose_9d_batch(
            transforms.left_robot_base_to_world_transform @ pose_3d_9d_to_homo_matrix_batch(
                obs_dict['left_robot_tcp_pose'])
        )
        extra_obs_dict['left_robot_wrt_right_robot_tcp_pose'] = absolute_actions_to_relative_actions(
            left_robot_tcp_pose_in_world, base_absolute_action=base_absolute_action_in_world)
    if 'right_robot_wrt_left_robot_tcp_pose' in lowdim_keys:
        base_absolute_action_in_world = homo_matrix_to_pose_9d_batch(
            transforms.left_robot_base_to_world_transform @ pose_3d_9d_to_homo_matrix_batch(
                obs_dict['left_robot_tcp_pose'][-1:])
        )[0]
        right_robot_tcp_pose_in_world = homo_matrix_to_pose_9d_batch(
            transforms.right_robot_base_to_world_transform @ pose_3d_9d_to_homo_matrix_batch(
                obs_dict['right_robot_tcp_pose'])
        )
        extra_obs_dict['right_robot_wrt_left_robot_tcp_pose'] = absolute_actions_to_relative_actions(
            right_robot_tcp_pose_in_world, base_absolute_action=base_absolute_action_in_world)

    return extra_obs_dict

# Example usage
if __name__ == "__main__":
    # Create sample data: 4 timesteps, 4 dimensions
    sample_actions = np.array([
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 10, 11, 12],
        [13, 14, 15, 16]
    ], dtype=float)

    # Test with different ratios
    for N in [2, 3, 4]:
        result = interpolate_actions_with_ratio(sample_actions, N)
        print(f"\nRatio N={N}:")
        print("Original shape:", sample_actions.shape)
        print("Interpolated shape:", result.shape)
        print("Interpolated actions:")
        print(result)

    # Create sample data: 4 timesteps, 10 dimensions
    new_sample_actions = np.zeros((4, 10), dtype=float)
    for i in range(4):
        new_sample_actions[i, :3] = sample_actions[i, :3]
        new_sample_actions[i, 9:] = sample_actions[i, 3:]
        new_sample_actions[i, 3:9] = R.from_rotvec(np.array([1, 0, 0]) * np.pi / 4 * i).as_matrix()[:3, :2].T.flatten()
    sample_actions = new_sample_actions

    # Test with different ratios
    for N in [2, 3, 4]:
        result = interpolate_actions_with_ratio(sample_actions, N)
        print(f"\nRatio N={N}:")
        print("Original shape:", sample_actions.shape)
        print("Interpolated shape:", result.shape)
        print("Interpolated actions:")
        print(result)
