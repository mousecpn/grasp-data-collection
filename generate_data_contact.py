import numpy as np
import sys
from pathlib import Path
import os
from math import sin, cos

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), './')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

import open3d as o3d
import torch
from torch_geometric.nn import radius
from tqdm import trange
import argparse
from experiment.simulation_clutter_bandit_single_camera import ClutterRemovalSim
from utils.transform import Transform, Rotation
from utils.utils_3d import AngleBasedSampler, FarthestSampler
from data_generation_utils import show_pcds_gt_segs, show_grasp_indices, get_gripper_angle_mask, \
    voxel_down_sample
import pickle
from utils.io import *
from utils.implicit import get_mesh_pose_list_from_world

def data_save(filename, mode, pcd_points, pcd_normals, sample_index, grasping_pose, success, object_num):
    assert mode in ['ab', 'wb']

    local_vars = locals()
    data_to_save = {k: local_vars[k] for k in
                    ['pcd_points', 'pcd_normals', 'sample_index', 'grasping_pose', 'success', 'object_num']}

    with open(filename, mode) as f:
        pickle.dump(data_to_save, f)  # type: ignore


def run(args, scene, object_set, iteration_num, savePath, start=0, GUI=True, n_intervals=36):
    sim = ClutterRemovalSim(scene, object_set, gui=GUI, rand=True, seed=start, n_intervals=n_intervals)

    total_grasp = 0
    total_success = 0

    # Because we don't have segmentation masks, we need to generate the sub-point clouds by FPS
    point_sampler = FarthestSampler()
    sample_strategy = AngleBasedSampler()

    object_index = None
    position = None
    color = None

    for itr in trange(iteration_num, desc='Iteration'):

        objs_num = sim.rng.poisson(4) + 1
        # random the camera setting
        ## orbitgrasp camera ###
        r = np.random.uniform(1.5, 2) * sim.size
        theta = np.random.uniform(np.pi / 4, np.pi / 2.4)
        phi = np.random.uniform(0.0, np.pi)
        origin = Transform(Rotation.identity(), np.r_[sim.size / 2, sim.size / 2, 0.0 + 0.15])
        ## orbitgrasp camera ###

        ## ICGNet train camera ###
        # r = np.random.uniform(1.6, 2.4) * sim.size
        # theta = np.random.uniform(0, np.pi / 4.0)
        # phi = np.random.uniform(0, 2 * np.pi)
        # origin = Transform(Rotation.identity(), np.r_[sim.size / 2, sim.size / 2, sim.size / 3])
        ## ICGNet train camera ###
        
        ## ICGNet test camera ###
        # r = np.random.uniform(2, 2.5) * sim.size
        # theta = np.random.uniform(np.pi / 4, np.pi / 3)
        # phi = np.random.uniform(0.0, np.pi)
        # origin = Transform(
        #     Rotation.identity(),
        #     np.r_[sim.size / 2, sim.size / 2, sim.size / 3],
        # )
        ## ICGNet test camera ###
        
        eye = np.r_[
            r * sin(theta) * cos(phi),
            r * sin(theta) * sin(phi),
            r * cos(theta),
        ]
        eye = eye + origin.translation

        a = Transform.look_at(eye - origin.translation, np.array([0, 0, 0]), [0, 0, 1]) * origin.inverse()
        a = a.inverse().to_list()
        camera_config = {
            'image_size': (480, 640),
            'intrinsics': (450., 0, 320., 0, 450., 240., 0, 0, 1),
            'position': eye,
            'rotation': a[:4],
            'zrange': (0.01, 2.),
            'noise': False,
            'name': 'random'
        }

        

        #######################################################################################
        # original camera setting of edge grasp.
        # ours is slightly different with their setting.
        # Meanwhile, the camera intrinsic is different, so if you need to generate grasp pose
        # on the original setting, please use the following setting and camera intrinsic.

        # r = np.random.uniform(2, 2.5) * sim.size
        # theta = np.random.uniform(np.pi / 4, np.pi / 3)
        # phi = np.random.uniform(0.0, np.pi)
        # origin = Transform(Rotation.identity(), np.r_[sim.size / 2, sim.size / 2, 0.0 + 0.25])
        # eye = np.r_[
        #     r * sin(theta) * cos(phi),
        #     r * sin(theta) * sin(phi),
        #     r * cos(theta),
        # ]
        # eye = eye + origin.translation
        #
        # a = Transform.look_at(eye - origin.translation, np.array([0, 0, 0]), [0, 0, 1]) * origin.inverse()
        # a = a.inverse().to_list()
        # config = {
        #     'image_size': (480, 640),
        #     'intrinsics': (540., 0, 320., 0, 540., 240., 0, 0, 1),   ### original camera intrinsic
        #     'position': eye,
        #     'rotation': a[:4],
        #     'zrange': (0.01, 2.),
        #     'noise': False,
        #     'name': 'random'
        # }
        #######################################################################################

        sim.reset(objs_num, index=object_index, pose_=position, color_=color, from_save=False) ##### offset=0.022 
        if not os.path.exists(args.root / "scenes"):
            (args.root / "scenes").mkdir(parents=True, exist_ok=True)
            write_setup(
                args.root,
                sim.size,
                CameraIntrinsic(640, 480, 450., 450., 320., 240.),
                sim.gripper.max_opening_width,
                sim.gripper.finger_depth,
            )
            if args.save_scene:
                (args.root / "mesh_pose_list").mkdir(parents=True, exist_ok=True)
        # Generate the point cloud
        sim.getRandomViewObservation(camera_config)

        extrinsic = camera_on_sphere(origin, r, theta, phi)
        scene_id = write_sensor_data(args.root, sim.heightmaps[:,:,:,0], [extrinsic.to_list()])
        if args.save_scene:
            mesh_pose_list = get_mesh_pose_list_from_world(sim.world, "/".join([scene, object_set]))
            write_mesh(args.root, scene_id, mesh_pose_list, name="mesh_pose_list")

        seg_gts = sim.segmaps
        if sim.num_objects == 0:
            continue
        sim.save_state()

        pcds_all = list()
        index_list = list()
        pcd_xyz = sim.pcds_xyz[sim.non_zero_indices]
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pcd_xyz)

        if args.add_noise:
            vertices = np.asarray(pcd.points)
            # add gaussian noise 95% confident interval (-1.96,1.96)
            vertices = vertices + np.random.normal(loc=0.0, scale=0.0005, size=(len(vertices), 3))
            pcd.points = o3d.utility.Vector3dVector(vertices)

        pcd, ind_1 = pcd.remove_statistical_outlier(nb_neighbors=30, std_ratio=3)
        pcd, ind_2 = pcd.remove_radius_outlier(nb_points=30, radius=0.03)
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.04, max_nn=30))
        pcd.orient_normals_consistent_tangent_plane(30)
        pcd.orient_normals_towards_camera_location(camera_location=camera_config['position'])
        pcds_all.append(pcd)
        index_list.append([ind_1, ind_2])

        pcds_gt_segs = seg_gts.reshape(-1, 1)[sim.non_zero_indices]
        pcds_gt_segs = pcds_gt_segs[index_list[0][0]]
        pcds_gt_segs = pcds_gt_segs[index_list[0][1]].squeeze(-1)

        all_points = np.asarray(pcd.points)
        all_normals = np.asarray(pcd.normals)

        ori_pcd = o3d.geometry.PointCloud()
        ori_pcd.points = o3d.utility.Vector3dVector(all_points)
        ori_pcd.normals = o3d.utility.Vector3dVector(all_normals)

        # Downsample the point cloud
        down_pcd, down_gt_segs = voxel_down_sample(ori_pcd, voxel_size=0.0055,
                                                   gt_seg_arr=pcds_gt_segs)

        action_all_masks = list()
        success_list_all = list()
        down_pcd_points = np.asarray(down_pcd.points)
        down_pcd_normals = np.asarray(down_pcd.normals)

        grasp_indices = list()

        # Sample the center points
        sample_center_points, sample_center_index = point_sampler(down_pcd_points, int(sim.num_objects * 1.5), 0.055)
        down_pcd_points_tensor = torch.tensor(down_pcd_points, dtype=torch.float32, requires_grad=False)

        # Generate the grasping poses
        for order, center_point in enumerate(sample_center_points):
            center_point = torch.tensor(center_point, dtype=torch.float32, requires_grad=False).reshape(1, 3)
            edge_index = radius(down_pcd_points_tensor, center_point, r=0.04, max_num_neighbors=900)
            points_down_mask = down_pcd_points[edge_index[1]]
            normals_down_mask = down_pcd_normals[edge_index[1]]

            local_pcd = o3d.geometry.PointCloud()
            local_pcd.points = o3d.utility.Vector3dVector(points_down_mask)
            local_pcd.normals = o3d.utility.Vector3dVector(normals_down_mask)

            actions_list, points_list, normals_list, indices_list = sim.mask_random_actions(local_pcd, sample_strategy,
                                                                                            m=sample_num,
                                                                                            n_intervals=n_intervals,
                                                                                            z_thresh=0.052)

            global_indices = edge_index[1][indices_list]
            grasp_indices.append(global_indices)

            action_info = {'actions': actions_list, 'points': points_list, 'normals': normals_list,
                           'indices': indices_list, 'grasp_indices': global_indices, 'order': order}
            action_all_masks.append(action_info)

        if SHOW_GRASP_POINTS:
            show_grasp_indices(down_pcd, grasp_indices, random_color=False)

        succ_flag = 0
        total_flag = 0

        pbar = trange(len(action_all_masks), desc='Actions', leave=False)
        # Execute the grasping actions and evaluate the success
        # Some geometry priors are used to filter out the invalid grasping poses
        for p_index in pbar:
            actions = action_all_masks[p_index]['actions']
            normals = action_all_masks[p_index]['normals']
            grasp_points = grasp_indices[p_index]
            success_list = list()
            for i, normal in enumerate(normals):
                grasp_point = grasp_points[i]

                for action in actions[i]:
                    total_grasp += 1
                    pos = action[:3]
                    rot = action[-4:]
                    _, real_pos, real_rot = sim._decodeAction(action=[pos, rot, normal], z_thresh=0.052)

                    gripper_angle_mask = get_gripper_angle_mask(Rotation.from_quat(rot), threshold=75)
                    if gripper_angle_mask:
                        result, grasped_object_ind, graspable = sim.execute_grasp(action=[pos, rot, normal],
                                                                       waitTime=0, remove=False)
                        width = result[1]
                        label = 0
                        if graspable == 0:
                            success_list.append(-1)
                            label = 0
                        else:
                            success, object_id = sim.eval_single_grasp(index=grasp_point, segmentation_gt=down_gt_segs,
                                                                       grasp_obj_id=grasped_object_ind)
                            if success:
                                succ_flag += 1
                                total_success += 1
                                success_list.append(1)
                                label = 1
                            else:
                                success_list.append(0)
                                label = 0
                        # print('label:', label)
                        grasp_pose = Transform(Rotation.from_quat(real_rot), real_pos)
                        T_grasp_shift = Transform(Rotation.identity(), [0.0, 0.0, -0.0405])
                        grasp_pose = grasp_pose * T_grasp_shift
                        grasp_pose = Grasp(grasp_pose,  width)
                        write_grasp(args.root, scene_id, grasp_pose, label)
                    else:
                        success_list.append(-1)
                        label = 0
                        width = 0.08
                    
                    total_flag += 1
                    pbar.set_postfix({"sratio:": f'{succ_flag}/ {total_flag}',
                                      "total sratio:": f'{total_success}/ {total_grasp}'})

                    sim.restore_state()
            success_list = np.asarray(success_list).reshape(len(normals), n_intervals, 1)
            success_list_all.append(success_list)

    sim.close()


if __name__ == "__main__":
    sys.setrecursionlimit(150000)
    np.set_printoptions(threshold=np.inf)


    def strToBool(value):
        if value.lower() in {'false', 'f', '0', 'no', 'n'}:
            return False
        elif value.lower() in {'true', 't', '1', 'yes', 'y'}:
            return True
        raise ValueError(f'{value} is not a valid boolean value')


    parser = argparse.ArgumentParser()
    parser.add_argument('--scene', type=str, default="pile")
    parser.add_argument('--root', type=Path, default=Path("data/data_pile_train_raw"))
    # Visualization
    parser.add_argument('--GUI', type=strToBool, default=True)
    parser.add_argument('--show_grasp_points', type=strToBool, default=False)
    # Saving
    parser.add_argument('--save_data', type=strToBool, default=True)
    parser.add_argument('--save_path', type=str,
                        default=Path(__file__).resolve().parent / "collected_data")
    # Data generation configurations
    parser.add_argument('--n_intervals', type=int, default=12)
    parser.add_argument('--start_scene', type=int, default=0)
    parser.add_argument('--iteration_num', type=int, default=50)
    parser.add_argument('--FilterSame', type=strToBool, default=True)
    parser.add_argument('--sample_num', type=int, default=20)
    parser.add_argument('--add_noise', type=strToBool, default=False)
    parser.add_argument('--object_set', type=str, default='train')
    parser.add_argument('--save_scene', type=bool, default=True)

    args = parser.parse_args()
    SAVING = args.save_data
    GUI = args.GUI
    sample_num = args.sample_num
    n_intervals = args.n_intervals
    SHOW_GRASP_POINTS = args.show_grasp_points
    PATH = args.save_path
    START = args.start_scene
    ITERATIONNUM = args.iteration_num
    FilterSame = args.FilterSame

    scene = args.scene
    object_set = args.object_set

    run(args, scene, object_set, ITERATIONNUM, PATH, start=START, GUI=GUI, n_intervals=n_intervals)
