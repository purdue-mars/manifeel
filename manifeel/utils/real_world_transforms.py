from dataclasses import dataclass, field
from omegaconf import DictConfig
from manifeel.utils.space_utils import pose_7d_to_4x4matrix
from loguru import logger
import transforms3d as t3d
import numpy as np
import os.path as osp
import json

@dataclass
class RealWorldTransforms:
    option: DictConfig = field(default=None)
    external_camera_to_left_robot_base_transform: np.ndarray = field(default=np.eye(4))
    external_camera_to_right_robot_base_transform: np.ndarray = field(default=np.eye(4))
    world_to_external_camera_transform: np.ndarray = field(default=np.eye(4))
    external_camera_to_world_transform: np.ndarray = field(default=np.eye(4))
    left_wrist_camera_to_left_robot_tcp_transform: np.ndarray = field(default=np.eye(4))
    right_wrist_camera_to_right_robot_tcp_transform: np.ndarray = field(default=np.eye(4))
    world_to_left_robot_base_transform: np.ndarray = field(default=np.eye(4))
    world_to_right_robot_base_transform: np.ndarray = field(default=np.eye(4))
    left_robot_base_to_world_transform: np.ndarray = field(default=np.eye(4))
    right_robot_base_to_world_transform: np.ndarray = field(default=np.eye(4))
    left_robot_base_pos_in_world: np.ndarray = field(default=np.array([0.0, 0.0, 0.0]))
    right_robot_base_pos_in_world: np.ndarray = field(default=np.array([0.0, 0.0, 0.0]))
    unity_to_world_fit_matrix: np.ndarray = field(default=np.array(
        [[0., 0., 1.],
         [-1, 0., 0.],
         [0., -1, 0.]]))
    unity_to_world_transform: np.ndarray = field(default=np.array(
        [[0., 0., 1., 0.],
         [-1, 0., 0., 0.],
         [0., 1, 0., 0.],
         [0., 0., 0., 1.]]))

    def __post_init__(self):
        if self.option is not None:
            with open(osp.join(self.option.calibration_path, 'external_camera_to_left_robot_base_transform.json'), 'r') as f:
                self.external_camera_to_left_robot_base_transform = np.array(json.load(f))
            with open(osp.join(self.option.calibration_path, 'external_camera_to_right_robot_base_transform.json'), 'r') as f:
                self.external_camera_to_right_robot_base_transform = np.array(json.load(f))
            with open(osp.join(self.option.calibration_path, 'left_wrist_camera_to_left_robot_tcp_transform.json'), 'r') as f:
                self.left_wrist_camera_to_left_robot_tcp_transform = np.array(json.load(f))
            # with open(osp.join(self.option.calibration_path, 'right_wrist_camera_to_right_robot_tcp_transform.json'), 'r') as f:
            #     self.right_wrist_camera_to_right_robot_tcp_transform = np.array(json.load(f))
            with open(osp.join(self.option.calibration_path, 'world_to_left_robot_base_transform.json'), 'r') as f:
                self.world_to_left_robot_base_transform = np.array(json.load(f))
            with open(osp.join(self.option.calibration_path, 'world_to_right_robot_base_transform.json'), 'r') as f:
                self.world_to_right_robot_base_transform = np.array(json.load(f))

            self.world_to_external_camera_transform = (np.linalg.inv(self.external_camera_to_left_robot_base_transform)
                                                       @ self.world_to_left_robot_base_transform)
            self.external_camera_to_world_transform = np.linalg.inv(self.world_to_external_camera_transform)

            self.left_robot_base_to_world_transform = np.linalg.inv(self.world_to_left_robot_base_transform)
            self.right_robot_base_to_world_transform = np.linalg.inv(self.world_to_right_robot_base_transform)

            self.left_robot_base_pos_in_world = (self.left_robot_base_to_world_transform @ np.array([[0., 0., 0., 1.]]).T)[:3, 0]
            self.right_robot_base_pos_in_world = (self.right_robot_base_to_world_transform @ np.array([[0., 0., 0., 1.]]).T)[:3, 0]

            # TODO: optimize name and code here
            # self.unity_to_world_fit_matrix = t3d.euler.axangle2mat([0, 1, 0], np.pi / 2)
            # self.unity_to_world_fit_matrix = self.unity_to_world_fit_matrix @ t3d.euler.axangle2mat([0, 0, 1], -np.pi / 2)
            self.unity_fit_mat_left = np.linalg.inv(self.left_robot_base_to_world_transform[:3, :3]) @ self.unity_to_world_fit_matrix
            self.unity_fit_mat_right = np.linalg.inv(self.right_robot_base_to_world_transform[:3, :3]) @ self.unity_to_world_fit_matrix

        else:
            pass

    def unity2robot_frame(self, pos_quat: np.ndarray, left: bool) -> np.ndarray:
        """
        Convert Unity frame to Z-up right-hand frame, then convert it to robot base frame (left or right)
        :param pos_quat: (x, y, z, qw, qx, qy, qz)
        :param left: whether it is left robot
        :return: target: (x, y, z, qw, qx, qy, qz)
        """
        # TODO: optimize name and code here
        pos_quat *= np.array([1, -1, 1, 1, -1, 1, -1])
        rot_mat = t3d.quaternions.quat2mat(pos_quat[3:])
        pos_vec = pos_quat[:3]
        T = np.eye(4)
        T[:3, :3] = rot_mat
        T[:3, 3] = pos_vec

        fit_mat = self.unity_fit_mat_left if left else self.unity_fit_mat_right

        target_rot_mat = fit_mat @ rot_mat
        target_pos_vec = fit_mat @ pos_vec
        target = np.array(target_pos_vec.tolist() + t3d.quaternions.mat2quat(target_rot_mat).tolist())
        return target

    def robot_frame2unity(self, pos_quat: np.ndarray, left: bool) -> np.ndarray:
        """
        Convert robot base frame (left or right) to Z-up right-hand frame, then convert it to Unity frame
        :param pos_quat: (x, y, z, qw, qx, qy, qz)
        :param left: whether it is left robot
        :return: target: (x, y, z, qw, qx, qy, qz)
        """
        pose_in_robot_frame = pose_7d_to_4x4matrix(pos_quat)
        if left:
            pose_in_world_frame = self.left_robot_base_to_world_transform @ pose_in_robot_frame
        else:
            pose_in_world_frame = self.right_robot_base_to_world_transform @ pose_in_robot_frame
        pose_in_unity_frame = np.linalg.inv(self.unity_to_world_transform) @ pose_in_world_frame
        target_pos_vec = pose_in_unity_frame[:3, 3]

        # TODO: figure out why we need another way to calculate rotation
        rot_mat = pose_in_world_frame[:3, :3]
        target_rot_mat = np.linalg.inv(self.unity_to_world_fit_matrix) @ rot_mat
        target_rot_quat = t3d.quaternions.mat2quat(target_rot_mat)
        target_rot_quat *= np.array([1, -1, 1, -1])
        target = np.array(target_pos_vec.tolist() + target_rot_quat.tolist())
        return target

