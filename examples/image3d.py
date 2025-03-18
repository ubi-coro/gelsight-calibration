import numpy as np
import open3d as o3d


def depth_map_to_point_cloud(depth_map: np.ndarray, vis=False):
    """
    Converts a depth map into a point cloud and visualizes it using Open3D.

    Parameters:
    - depth_map (np.ndarray): Depth map of shape (H, W, 1) or (H, W).
    Returns:
    - o3d.geometry.PointCloud
    """

    # Ensure depth map is 2D
    if len(depth_map.shape) == 3:
        depth_map = depth_map.squeeze()  # Convert shape (H, W, 1) to (H, W)

    # Ensure depth map is in C-contiguous memory layout
    depth_map = np.ascontiguousarray(depth_map, dtype=np.float32)

    # Get image dimensions
    height, width = depth_map.shape

    # Generate (x, y) coordinates
    xx, yy = np.meshgrid(np.arange(width), np.arange(height))  # Create grid

    # Flatten arrays to create a point cloud
    x = xx.flatten()
    y = yy.flatten()
    z = depth_map.flatten()

    # Stack into (N, 3) shape
    points = np.vstack((x, y, z)).T

    # Convert to Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd = pcd.voxel_down_sample(voxel_size=0.9)

    # Visualize
    if vis:
        o3d.visualization.draw_geometries([pcd])
    return points

# Example usage:
# Load depth map from a NumPy file or any other source
# depth_map = np.load("your_depth_map.npy")
# display_depth_map_as_point_cloud(depth_map)
