import cv2
import numpy as np
import open3d as o3d
from typing import List


def visualize_rgb_image(image: np.ndarray, name: str = 'Image'):
    cv2.imshow(name, image)
    cv2.waitKey(0)
    # cv2.destroyAllWindows()


def visualize_pcd_from_numpy(xyz_rgb: np.ndarray):
    """
    Visualize point cloud data
    :param xyz_rgb: numpy array of shape (N, 6) (x, y, z, r, g, b)
    """
    raw_pcd = o3d.geometry.PointCloud()
    raw_pcd.points = o3d.utility.Vector3dVector(xyz_rgb[:, :3])
    raw_pcd.colors = o3d.utility.Vector3dVector(xyz_rgb[:, 3:6])  # [0., 1.]
    coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
    o3d.visualization.draw_geometries([raw_pcd, coord])


def visualize_tactile_marker(geometries: List[o3d.geometry.TriangleMesh], vis):
    '''
    Visualizes tactile markers from the given geometries.
    :param geometries: List of Open3D TriangleMesh objects representing markers and arrows.
    :param vis: o3d.visualization.Visualizer()
    '''
    vis.clear_geometries()
    for geom in geometries:
        vis.add_geometry(geom)

    view_ctl = vis.get_view_control()
    view_ctl.set_lookat([100, 100, 0])
    view_ctl.set_up([0, 0, 1])
    view_ctl.set_front([1, 1, 1])

    vis.poll_events()
    vis.update_renderer()