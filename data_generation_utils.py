import copy
import open3d as o3d
import numpy as np
from collections import Counter
from sklearn.cluster import KMeans
import matplotlib.patches as patches


def show_pcds_gt_segs(pcd, gt_segs, random_color=True):
    """
    Visualize an uncolored point cloud based on different ground truth segmentations,
    randomly generating a color for each segmentation.

    :param point_cloud: The original point cloud to be colored, of type open3d.geometry.PointCloud.
    :param gt_segs: A ndarray containing multiple segmentations with the same length as the number of points in the point cloud.
    """
    # Ensure the point cloud has a color attribute
    pcd_c = copy.deepcopy(pcd)
    pcd_c.colors = o3d.utility.Vector3dVector(np.zeros((len(pcd.points), 3)))

    # Retrieve the color array of the point cloud
    colors = np.asarray(pcd_c.colors)
    eles = np.unique(gt_segs)
    # Iterate over all segmentations
    for ele in eles:
        id = np.where(gt_segs == ele)[0]

        # Randomly generate a color for each segmentation
        if random_color:
            color = np.random.rand(3)
        else:
            color = np.array([1, 0, 0])

        # Update colors
        colors[id, :] = color  # Color the points corresponding to True in the gt_segs

    # Update the color of the point cloud
    pcd_c.colors = o3d.utility.Vector3dVector(colors)
    o3d.visualization.draw_geometries([pcd_c])


def show_pcds_masks(pcd, masks_list, random_color=True):
    """
        Visualize an uncolored point cloud based on different predicted segmentations,


    :param point_cloud: The original point cloud to be colored, of type open3d.geometry.PointCloud.
    :param masks_list: A list containing multiple masks, each mask is a boolean array with the same length as the number of points in the point cloud.
    """
    # Ensure the point cloud has a color attribute
    pcd_c = copy.deepcopy(pcd)
    default_color = [0.5, 0.5, 0.5]
    pcd_c.colors = o3d.utility.Vector3dVector(np.full((len(pcd.points), 3), 0.5))

    # Retrieve the color array of the point cloud
    colors = np.asarray(pcd_c.colors)

    # Iterate over all masks
    for mask in masks_list:
        # Randomly generate a color for each mask
        if random_color:
            color = np.random.rand(3)
        else:
            color = np.array([1, 0, 0])

        # # Update colors based on the mask values
        # colors[mask, :] = color  # Color the points corresponding to True in the mask
        mask_indices = np.where(mask)[0]
        default_indices = np.all(colors[mask_indices] == default_color, axis=1)
        actual_indices = mask_indices[default_indices]

        # Update colors
        colors[actual_indices] = color

    # Update the color of the point cloud
    pcd_c.colors = o3d.utility.Vector3dVector(colors)
    o3d.visualization.draw_geometries([pcd_c])


def show_grasp_indices(pcd, grasp_indices, random_color=True):
    """
    Visualize the grasp points in the point cloud, coloring the points with red.

    :param point_cloud: The original point cloud of type open3d.geometry.PointCloud.
    :param grasp_indices: Grasp indices of the point cloud.
    """
    # Ensure the point cloud has a color attribute
    pcd_c = copy.deepcopy(pcd)
    pcd_c.colors = o3d.utility.Vector3dVector(np.full((len(pcd.points), 3), 0.8))

    # Retrieve the color array of the point cloud
    colors = np.asarray(pcd_c.colors)

    # Iterate over all points
    for point in grasp_indices:
        # Randomly generate a color for each point
        if random_color:
            color = np.random.rand(3)
        else:
            color = np.array([1, 0, 0])

        # Update colors
        colors[point, :] = color  # Color the points

    # Update the color of the point cloud
    pcd_c.colors = o3d.utility.Vector3dVector(colors)
    o3d.visualization.draw_geometries([pcd_c])


def show_anns(anns, ax, draw_bbox=False):
    if len(anns) == 0:
        return

    ax.set_autoscale_on(False)
    rng = np.random.RandomState()

    img = np.ones((anns[0]['segmentation'].shape[0],
                   anns[0]['segmentation'].shape[1], 4))
    img[:, :, 3] = 0

    for ann in anns:
        m = ann['segmentation']
        rgb_color = rng.random(3).astype(np.float32)
        alpha = np.asarray([0.35]).astype(np.float32)
        color_mask = np.concatenate([rgb_color, alpha])
        img[m] = color_mask
        if draw_bbox:
            x, y, w, h = ann['bbox']

            bbox = patches.Rectangle(
                (x, y), w, h, linewidth=3, edgecolor=rgb_color, facecolor='None')
            ax.add_patch(bbox)
    ax.imshow(img)


def voxel_down_sample_with_multiple_masks(original_pcd, voxel_size, masks_arr, gt_seg_arr=None, cluster_num=2):
    # Due to the point cloud has the masks from the pretrained model, we need to downsample the point cloud as well as the masks.

    down_pcd, _, traced_indices = original_pcd.voxel_down_sample_and_trace(voxel_size=voxel_size,
                                                                           min_bound=original_pcd.get_min_bound(),
                                                                           max_bound=original_pcd.get_max_bound())

    new_normals = np.asarray(down_pcd.normals)
    norms = np.linalg.norm(new_normals, axis=1, keepdims=True)
    norms[norms == 0] = 1
    new_normals = new_normals / norms
    # down_pcd.normals = o3d.utility.Vector3dVector(new_normals)
    for i in range(len(new_normals)):
        point_z = down_pcd.points[i][2]
        if point_z > 0.065 or point_z < 0.051:
            continue

        original_normals = np.asarray(original_pcd.normals)[traced_indices[i]]

        if original_normals.size > 0:
            closest_z_normal = original_normals[np.abs(original_normals[:, 2]).argmin(), 2]

            #############################################################################################
            ## Fix the normal direction of the point cloud for the points close to the table,          ##
            ## as the normals are not consistent and noisy.                                            ##
            ## (e.g., the z-component of the normal should be near 0 but now is negative or positive). ##
            ## This is only used during the data collection process b/c if we use the original normals,##
            ## some objects cannot be grasped due to the incorrect estimation of the normal.           ##
            ## IMPORTANT: This is not used in the testing process.                                    ##
            #############################################################################################

            if -0.2 < closest_z_normal < -0.1:
                closest_z_normal += 0.1
            elif 0.1 < closest_z_normal < 0.2:
                closest_z_normal -= 0.1

            x, y, _ = new_normals[i]
            xy_squared = x ** 2 + y ** 2
            if xy_squared > 1e-8:
                alpha = np.sqrt((1 - closest_z_normal ** 2) / xy_squared)
                down_pcd_normal_xy = np.array([x * alpha, y * alpha])
            else:
                down_pcd_normal_xy = np.array([0, 0])
            new_normal = np.append(down_pcd_normal_xy, closest_z_normal)
            new_normals[i] = new_normal

    # Update normals in the downsampled point cloud
    down_pcd.normals = o3d.utility.Vector3dVector(new_normals)

    down_pcd_masks_list = []
    if gt_seg_arr is not None:
        down_pcd_gts = np.zeros(len(down_pcd.points), dtype=np.uint8)
        for i, indices in enumerate(traced_indices):
            temp = gt_seg_arr[indices]
            counter = Counter(temp)
            main_seg_idx, _ = counter.most_common(1)[0]

            down_pcd_gts[i] = main_seg_idx

    for masks in masks_arr:
        down_pcd_masks = np.zeros(len(down_pcd.points), dtype=bool)
        for i, indices in enumerate(traced_indices):
            if any(masks[j] for j in indices):
                down_pcd_masks[i] = True
        if down_pcd_masks.sum() <= 15:
            continue
        # down_pcd_masks_list.append(down_pcd_masks)
        mask_pcd_coord = np.asarray(down_pcd.points)[down_pcd_masks]
        if mask_pcd_coord.shape[0] <= 350:
            down_pcd_masks_list.append(down_pcd_masks)
        else:
            n_clusters = mask_pcd_coord.shape[0] // 300 + 1
            cluster_masks, _ = kmeans_on_masked_points(down_pcd, down_pcd_masks, n_clusters)
            down_pcd_masks_list.extend(cluster_masks[:cluster_num])
    down_pcd_masks_list_denoise = pcd_mask_denoise(down_pcd, down_pcd_masks_list)

    if gt_seg_arr is not None:
        return down_pcd, down_pcd_masks_list_denoise, down_pcd_gts
    return down_pcd, down_pcd_masks_list_denoise


def kmeans_on_masked_points(pcd, mask, n_clusters):
    mask = np.array(mask, dtype=bool)

    selected_points = np.asarray(pcd.points)[mask]

    kmeans = KMeans(n_clusters=n_clusters, n_init=10)
    labels = kmeans.fit_predict(selected_points)

    cluster_masks = []
    for i in range(n_clusters):
        cluster_mask = np.zeros(len(pcd.points), dtype=bool)
        cluster_mask[mask] = (labels == i)
        cluster_masks.append(cluster_mask)

    return cluster_masks, labels


def pcd_mask_denoise(downsampled_pcd, masks_list, eps=0.01, min_points=5, min_cluster_size=10):
    updated_masks_list = []

    for mask in masks_list:

        masked_points = np.asarray(downsampled_pcd.points)[mask]
        if masked_points.shape[0] < min_points:
            continue
        masked_point_cloud = o3d.geometry.PointCloud()
        masked_point_cloud.points = o3d.utility.Vector3dVector(masked_points)

        labels = np.array(masked_point_cloud.cluster_dbscan(eps=eps, min_points=min_points, print_progress=False))
        updated_mask = np.full(shape=len(downsampled_pcd.points), fill_value=False, dtype=bool)

        for k in range(labels.max() + 1):
            cluster_mask = labels == k
            if k == -1 or np.sum(cluster_mask) < min_cluster_size:
                continue
            cluster_indices = np.where(mask)[0][cluster_mask]
            updated_mask[cluster_indices] = True

        updated_masks_list.append(updated_mask)

    return updated_masks_list


def voxel_down_sample(original_pcd, voxel_size, gt_seg_arr=None):
    down_pcd, _, traced_indices = original_pcd.voxel_down_sample_and_trace(voxel_size=voxel_size,
                                                                           min_bound=original_pcd.get_min_bound(),
                                                                           max_bound=original_pcd.get_max_bound())

    new_normals = np.asarray(down_pcd.normals)
    norms = np.linalg.norm(new_normals, axis=1, keepdims=True)
    norms[norms == 0] = 1
    new_normals = new_normals / norms
    for i in range(len(new_normals)):
        point_z = down_pcd.points[i][2]
        if point_z > 0.065 or point_z < 0.051:
            continue

        original_normals = np.asarray(original_pcd.normals)[traced_indices[i]]

        if original_normals.size > 0:
            closest_z_normal = original_normals[np.abs(original_normals[:, 2]).argmin(), 2]

            #############################################################################################
            ## Fix the normal direction of the point cloud for the points close to the table,          ##
            ## as the normals are not consistent and noisy.                                            ##
            ## (e.g., the z-component of the normal should be near 0 but now is negative or positive). ##
            ## This is only used during the data collection process b/c if we use the original normals,##
            ## some objects cannot be grasped due to the incorrect estimation of the normal.           ##
            ## IMPORTANT: This is not used in the training process.                                    ##
            #############################################################################################
            if -0.2 < closest_z_normal < -0.1:
                closest_z_normal += 0.1
            elif 0.1 < closest_z_normal < 0.2:
                closest_z_normal -= 0.1

            x, y, _ = new_normals[i]
            xy_squared = x ** 2 + y ** 2
            if xy_squared > 1e-8:
                alpha = np.sqrt((1 - closest_z_normal ** 2) / xy_squared)
                down_pcd_normal_xy = np.array([x * alpha, y * alpha])
            else:
                down_pcd_normal_xy = np.array([0, 0])
            new_normal = np.append(down_pcd_normal_xy, closest_z_normal)
            new_normals[i] = new_normal

    # Update normals in the downsampled point cloud
    down_pcd.normals = o3d.utility.Vector3dVector(new_normals)

    if gt_seg_arr is not None:
        down_pcd_gts = np.zeros(len(down_pcd.points), dtype=np.uint8)
        for i, indices in enumerate(traced_indices):
            temp = gt_seg_arr[indices]
            counter = Counter(temp)
            main_seg_idx, _ = counter.most_common(1)[0]

            down_pcd_gts[i] = main_seg_idx

    if gt_seg_arr is not None:
        return down_pcd, down_pcd_gts
    return down_pcd


def check_normal_availability(normal, threshold=10):
    cos_theta = normal[2] / np.linalg.norm(normal)
    cos_thresh = np.cos(np.radians(threshold))
    return cos_theta < cos_thresh


def get_gripper_points(trans):
    gripper_points_sim = np.array([
        [0.012, 0.09, -0.04],
        [0.012, -0.09, -0.04],
        [-0.012, 0.09, -0.04],
        [-0.012, -0.09, -0.04],

        [0.018, 0.09, -0.11],
        [0.018, -0.09, -0.11],
        [-0.018, 0.09, -0.11],
        [-0.018, -0.09, -0.11]])

    gripper_points_sim = trans[:3, :3] @ gripper_points_sim.transpose()
    gripper_points_sim = gripper_points_sim.transpose()
    # print(gripper_points_sim.size())
    gripper_points_sim = gripper_points_sim + trans[:3, -1]
    return gripper_points_sim


def get_gripper_angle_mask(rot, threshold=75):
    """
    Determines if the angle between the gripper z-axis and the world z-axis is greater than the threshold.

    Parameters:
    - rot: Rotation from the Quaternion representation.

    Returns:
    - Boolean: True if the angle is greater than the threshold, False otherwise.
    """
    global_z_direction = rot.apply([0, 0, 1])
    z_axis = np.array([0, 0, 1])
    cos_angle = np.dot(global_z_direction, z_axis)
    cos_angle = np.clip(cos_angle, -1, 1)
    angle_rad = np.arccos(cos_angle)
    angle_deg = np.degrees(angle_rad)

    return angle_deg > threshold


def data_save(filename, mode, pcd_points, pcd_normals, sample_index, grasping_pose, success, object_num):
    assert mode in ['ab', 'wb']

    local_vars = locals()
    data_to_save = {k: local_vars[k] for k in
                    ['pcd_points', 'pcd_normals', 'sample_index', 'grasping_pose', 'success', 'object_num']}

    with open(filename, mode) as f:
        pickle.dump(data_to_save, f)  # type: ignore
