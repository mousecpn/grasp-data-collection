"""Miscellaneous utilities."""

import cv2
# from transforms3d import euler
import numpy as np
import pybullet as p
import open3d as o3d


# -----------------------------------------------------------------------------
# HEIGHTMAP UTILS
# -----------------------------------------------------------------------------


def get_heightmap(points, colors, bounds, pixel_size):
    """Get top-down (z-axis) orthographic heightmap image from 3D pointcloud.

    Args:
      points: HxWx3 float array of 3D points in world coordinates.
      colors: HxWx3 uint8 array of values in range 0-255 aligned with points.
      bounds: 3x2 float array of values (rows: X,Y,Z; columns: min,max) defining
        region in 3D space to generate heightmap in world coordinates.
      pixel_size: float defining size of each pixel in meters.

    Returns:
      heightmap: HxW float array of height (from lower z-bound) in meters.
      colormap: HxWx3 uint8 array of backprojected color aligned with heightmap.
    """
    width = int(np.round((bounds[1, 1] - bounds[1, 0]) / pixel_size))
    height = int(np.round((bounds[0, 1] - bounds[0, 0]) / pixel_size))
    # height, width = pixel_size
    heightmap = np.zeros((height, width), dtype=np.float32)
    colormap = np.zeros((height, width, colors.shape[-1]), dtype=np.uint8)

    # Filter out 3D points that are outside of the predefined bounds.
    ix = (points[Ellipsis, 0] >= bounds[0, 0]) & (points[Ellipsis, 0] < bounds[0, 1])
    iy = (points[Ellipsis, 1] >= bounds[1, 0]) & (points[Ellipsis, 1] < bounds[1, 1])
    iz = (points[Ellipsis, 2] >= bounds[2, 0]) & (points[Ellipsis, 2] < bounds[2, 1])
    valid = ix & iy & iz

    points = points[valid]
    colors = colors[valid]

    # Sort 3D points by z-value, which works with array assignment to simulate
    # z-buffering for rendering the heightmap image.
    iz = np.argsort(points[:, -1])
    points, colors = points[iz], colors[iz]
    px = np.int32(np.floor((points[:, 0] - bounds[0, 0]) / pixel_size))
    py = np.int32(np.floor((points[:, 1] - bounds[1, 0]) / pixel_size))
    px = np.clip(px, 0, height - 1)
    py = np.clip(py, 0, width - 1)
    heightmap[px, py] = points[:, 2] - bounds[2, 0]
    for c in range(colors.shape[-1]):
        colormap[px, py, c] = colors[:, c]
    return heightmap, colormap


def get_pointcloud(depth, intrinsics):
    """Get 3D pointcloud from perspective depth image.

    Args:
      depth: HxW float array of perspective depth in meters.
      intrinsics: 3x3 float array of camera intrinsics matrix.

    Returns:
      points: HxWx3 float array of 3D points in camera coordinates.
    """
    height, width = depth.shape
    xlin = np.linspace(0, width - 1, width)
    ylin = np.linspace(0, height - 1, height)
    px, py = np.meshgrid(xlin, ylin)
    px = (px - intrinsics[0, 2]) * (depth / intrinsics[0, 0])
    py = (py - intrinsics[1, 2]) * (depth / intrinsics[1, 1])
    points = np.float32([px, py, depth]).transpose(1, 2, 0)
    return points


def transform_pointcloud(points, transform):
    """Apply rigid transformation to 3D pointcloud.

    Args:
      points: HxWx3 float array of 3D points in camera coordinates.
      transform: 4x4 float array representing a rigid transformation matrix.

    Returns:
      points: HxWx3 float array of transformed 3D points.
    """
    padding = ((0, 0), (0, 0), (0, 1))
    homogen_points = np.pad(points.copy(), padding,
                            'constant', constant_values=1)
    for i in range(3):
        points[Ellipsis, i] = np.sum(transform[i, :] * homogen_points, axis=-1)
    return points


def reconstruct_inbound_imgs_and_pcd(colors, depths, configs, bounds):
    """Return in_bounds heightmaps. colormaps and pcd."""
    pcds_xyz, pcds_rgb, inbound_colormaps, inbound_depthmaps = list(), list(), list(), list()
    if len(colors) == 480:
        colors = colors[None]
        depths = depths[None]
        configs = [configs]
    for color, depth, config in zip(colors, depths, configs):
        intrinsics = np.array(config['intrinsics']).reshape(3, 3)
        xyz = get_pointcloud(depth, intrinsics)
        position = np.array(config['position']).reshape(3, 1)
        rotation = p.getMatrixFromQuaternion(config['rotation'])
        rotation = np.array(rotation).reshape(3, 3)
        transform = np.eye(4)
        transform[:3, :] = np.hstack((rotation, position))
        points = transform_pointcloud(xyz, transform)

        ix = (points[Ellipsis, 0] >= bounds[0, 0]) & (points[Ellipsis, 0] < bounds[0, 1])
        iy = (points[Ellipsis, 1] >= bounds[1, 0]) & (points[Ellipsis, 1] < bounds[1, 1])
        iz = (points[Ellipsis, 2] >= bounds[2, 0]) & (points[Ellipsis, 2] < bounds[2, 1])
        valid = ix & iy & iz

        points_xyz = (points * valid[..., None]).reshape(-1, 3)
        color = color * valid[..., None]
        depth = depth[..., None] * valid[..., None]
        pcd_rgb = color.reshape(-1, 3) / 255

        pcds_xyz.append(points_xyz)
        pcds_rgb.append(pcd_rgb)
        inbound_colormaps.append(color)
        inbound_depthmaps.append(depth)
    return pcds_xyz, pcds_rgb, inbound_colormaps, inbound_depthmaps


def reconstruct_heightmaps(colors, depths, configs, bounds, pixel_size):
    """Reconstruct top-down heightmap views from multiple 3D pointclouds."""
    heightmaps, colormaps = [], []
    for color, depth, config in zip(colors, depths, configs):
        intrinsics = np.array(config['intrinsics']).reshape(3, 3)
        xyz = get_pointcloud(depth, intrinsics)
        position = np.array(config['position']).reshape(3, 1)
        rotation = p.getMatrixFromQuaternion(config['rotation'])
        rotation = np.array(rotation).reshape(3, 3)
        transform = np.eye(4)
        transform[:3, :] = np.hstack((rotation, position))
        xyz = transform_pointcloud(xyz, transform)
        heightmap, colormap = get_heightmap(xyz, color, bounds, pixel_size)
        heightmaps.append(heightmap)
        colormaps.append(colormap)
    return heightmaps, colormaps


def get_inbound_imgs_and_pcd(color, height, configs, bounds):
    """Reconstruct orthographic heightmaps with segmentation masks."""
    pcds_xyz, pcds_rgb, inbound_colormaps, inbound_depthmaps = reconstruct_inbound_imgs_and_pcd(color, height, configs,
                                                                                                bounds)
    pcds_xyz = np.concatenate(pcds_xyz, axis=0)
    pcds_rgb = np.concatenate(pcds_rgb, axis=0)
    inbound_colormaps = np.asarray(inbound_colormaps, dtype=np.uint8)
    inbound_depthmaps = np.asarray(inbound_depthmaps, dtype=np.float32)

    return pcds_xyz, pcds_rgb, inbound_colormaps, inbound_depthmaps


# -----------------------------------------------------------------------------
# SAMPLER UTILS
# -----------------------------------------------------------------------------

class FarthestSampler:
    def __init__(self):
        pass

    def _filter_points_by_z(self, pts, z_threshold):
        """Filter points based on their Z coordinate being above a certain threshold."""
        valid_indices = np.where(pts[:, 2] > z_threshold)[0]
        return pts[valid_indices, :], valid_indices

    def _calc_distances(self, p0, points):
        return ((p0 - points) ** 2).sum(axis=1)

    def __call__(self, pts, k, z_threshold):
        filtered_pts, valid_indices = self._filter_points_by_z(pts, z_threshold)
        if filtered_pts.shape[0] <= k:
            return filtered_pts, valid_indices

        index_list = []
        farthest_pts = np.zeros((k, 3), dtype=np.float32)
        index = np.random.randint(len(filtered_pts))
        farthest_pts[0] = filtered_pts[index]
        index_list.append(valid_indices[index])
        distances = self._calc_distances(farthest_pts[0], filtered_pts)
        for i in range(1, k):
            index = np.argmax(distances)
            farthest_pts[i] = filtered_pts[index]
            index_list.append(valid_indices[index])
            distances = np.minimum(
                distances, self._calc_distances(farthest_pts[i], filtered_pts))
        return farthest_pts, index_list


class AngleBasedSampler:
    """Sample k points based on their angle with respect to the Z axis, considering a Z threshold.
    The larger the angle, the higher the probability of being selected.
    We would use the normal vector of the point cloud to calculate the angle.
    This process is reasonable because the successful grasp usually happeds with the large angle w.r.t the Z axis."""

    def __init__(self):
        pass

    def _filter_points_by_z(self, pts, z_threshold):
        """Filter points based on their Z coordinate being above a certain threshold."""
        valid_indices = np.where(pts[:, 2] > z_threshold)[0]
        return pts[valid_indices, :], valid_indices

    def _calc_angle_weights(self, normals):
        """Calculate weights based on the angle between point normals and the Z axis."""
        z_axis = np.array([0, 0, 1])
        cos_theta = np.dot(normals, z_axis) / np.linalg.norm(normals, axis=1)
        # Use 1 minus the absolute value of the cosine of the angle to make 90 degrees have the highest weight
        weights = 1 - np.abs(cos_theta)  # Maximize weights at 90 degrees
        weights += 0.03  # Add a small constant to ensure no weight is exactly zero
        return weights / np.sum(weights)  # Normalize weights

    def __call__(self, pts, normals, k, z_threshold):
        filtered_pts, valid_indices = self._filter_points_by_z(pts, z_threshold)

        if filtered_pts.shape[0] <= k:
            return filtered_pts, valid_indices

        weights = self._calc_angle_weights(normals[valid_indices])

        # Use weighted random sampling to select k points
        chosen_indices = np.random.choice(len(filtered_pts), size=k, replace=False, p=weights)
        sampled_pts = filtered_pts[chosen_indices]
        original_indices = valid_indices[chosen_indices]  # Map back to original indices from the full list

        return sampled_pts, original_indices
