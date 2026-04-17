from manifeel.utils.visualization_utils import visualize_rgb_image, visualize_tactile_marker
import numpy as np
import open3d as o3d
import cv2
import copy
from typing import Tuple


def marker_normalization(marker_loc: np.ndarray, marker_offset: np.ndarray,
                         dimension: int, width: int, height: int):
    # normalization
    marker_loc[:, 0] /= width
    marker_loc[:, 1] /= height
    marker_offset[:, 0] /= width
    marker_offset[:, 1] /= height

    return marker_loc, marker_offset

def process_marker_array(array, width, height, dimension) -> Tuple[np.ndarray, np.ndarray]:
    '''
    Decoding  and de-normalizing pointclouds of tactile camera marker information
    '''
    # Decode points array into marker and offsets
    marker_locations = copy.deepcopy(array[:, :2])
    marker_offsets = copy.deepcopy(array[:, 2:4])

    # De-normalization
    marker_locations[:, 0] *= width
    marker_locations[:, 1] *= height
    marker_offsets[:, 0] *= width
    marker_offsets[:, 1] *= height

    return marker_locations, marker_offsets

def marker_track_visualization(rgb_image, marker, marker_offset, dimension, vis=None):
    '''
    Visualizing the image with trackers
    dimension: the dimension of marker motions
    '''
    if rgb_image is None:
        return

    img_show = rgb_image.copy()
    show_scale = 5
    marker_img = display_motion(img_show, marker, marker_offset, show_scale, dimension)

    if dimension == 2:
        visualize_rgb_image(marker_img, "Marker Image")
    elif dimension == 3:
        visualize_tactile_marker(marker_img, vis)


def display_motion(image, cur_marker, marker_motion, show_scale, dimension):
    '''
    Combine the marker motion and the original rgb image into a single picture
    '''
    if dimension == 2:
        marker_center = np.around(cur_marker).astype(np.int16)
        for i in range(cur_marker.shape[0]):
            if marker_motion[i, 0] != 0 or marker_motion[i, 1] != 0:
                end_point = (
                    int(cur_marker[i, 0] + marker_motion[i, 0] * show_scale),
                    int(cur_marker[i, 1] + marker_motion[i, 1] * show_scale)
                )
                end_point = (
                    np.clip(end_point[0], 0, image.shape[1] - 1),
                    np.clip(end_point[1], 0, image.shape[0] - 1)
                )
                cv2.arrowedLine(image, (marker_center[i, 0], marker_center[i, 1]), end_point, (0, 255, 255), 2)

        return image
    elif dimension == 3:
        """
        Visualizes the initial and moved marker positions with spheres.
        Returns:
            geometries: The visualizer object with added components.
        """
        geometries = []
        arrow_radius = 1
        arrow_length_ratio = 0.8

        # Copy initial marker positions
        initial_positions = copy.deepcopy(cur_marker)
        initial_positions[:, 2] = 0

        for point, vector in zip(initial_positions, marker_motion):
            vertical_offset = copy.deepcopy(vector)
            vertical_offset[0:2] = 0
            sphere = o3d.geometry.TriangleMesh.create_sphere(radius=1, resolution=3)
            sphere.translate(point)
            sphere.paint_uniform_color([0, 1, 0])
            geometries.append(sphere)

            length = np.linalg.norm(vector)
            if length > 0:
                arrow = o3d.geometry.TriangleMesh.create_arrow(
                    cylinder_radius=arrow_radius, cone_radius=arrow_radius * 2,
                    cylinder_height=length * arrow_length_ratio, cone_height=length * (1 - arrow_length_ratio)
                )
                arrow_direction = vector / length
                default_arrow_direction = np.array([0, 0, 1])
                rotation_matrix = o3d.geometry.get_rotation_matrix_from_xyz(
                    np.cross(default_arrow_direction, arrow_direction))

                arrow.rotate(rotation_matrix)
                arrow.translate(point)
                arrow.paint_uniform_color([1, 0, 0])

                geometries.append(arrow)

        return geometries


