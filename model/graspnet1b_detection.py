import numpy as np
import sys

import os
import sys
import numpy as np
import scipy.io as scio
from PIL import Image
import open3d as o3d
import torch
from graspnetAPI import GraspGroup
import time
import trimesh
from utils.transform import Transform, Rotation
from utils.grasp import Grasp
from utils import visual
import configargparse

ROOT_DIR = "~/graspnet-baseline" # root of your graspnet-baseline folder
sys.path.append(ROOT_DIR)

sys.path.append(os.path.join(ROOT_DIR, 'models'))
sys.path.append(os.path.join(ROOT_DIR, 'dataset'))
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
# sys.path.append("/home/pinhao/graspnet-baseline")

from graspnet import GraspNet, pred_decode
from graspnet_dataset import GraspNetDataset
from collision_detector import ModelFreeCollisionDetector
from data_utils import CameraInfo, create_point_cloud_from_depth_image

# parser = argparse.ArgumentParser()
# parser.add_argument('--checkpoint_path', default="/home/pinhao/graspnet-baseline/logs/log_rs/checkpoint-rs.tar", help='Model checkpoint path')
# parser.add_argument('--num_point', type=int, default=20000, help='Point Number [default: 20000]')
# parser.add_argument('--num_view', type=int, default=300, help='View Number [default: 300]')
# parser.add_argument('--collision_thresh', type=float, default=0.01, help='Collision Threshold in collision detection [default: 0.01]')
# parser.add_argument('--voxel_size', type=float, default=0.01, help='Voxel Size to process point clouds before collision detection [default: 0.01]')
# cfgs = parser.parse_args()

def make_parser():
    args = configargparse.ArgumentParser()
    args.checkpoint_path = ROOT_DIR+"/logs/log_rs/checkpoint-rs.tar"
    args.num_point = 20000
    args.num_view = 300
    args.collision_thresh = 0.01
    args.voxel_size = 0.01
    return args

cfgs = make_parser()

def get_net():
    # Init the model
    net = GraspNet(input_feature_dim=0, num_view=cfgs.num_view, num_angle=12, num_depth=4,
            cylinder_radius=0.05, hmin=-0.02, hmax_list=[0.01,0.02,0.03,0.04], is_training=False)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.to(device)
    # Load checkpoint
    checkpoint = torch.load(cfgs.checkpoint_path)
    net.load_state_dict(checkpoint['model_state_dict'])
    start_epoch = checkpoint['epoch']
    print("-> loaded checkpoint %s (epoch: %d)"%(cfgs.checkpoint_path, start_epoch))
    # set model to eval mode
    net.eval()
    return net

def get_and_process_data(data_dir):
    # load data
    color = np.array(Image.open(os.path.join(data_dir, 'color.png')), dtype=np.float32) / 255.0
    depth = np.array(Image.open(os.path.join(data_dir, 'depth.png')))
    workspace_mask = np.array(Image.open(os.path.join(data_dir, 'workspace_mask.png')))
    meta = scio.loadmat(os.path.join(data_dir, 'meta.mat'))
    intrinsic = meta['intrinsic_matrix']
    factor_depth = meta['factor_depth']

    # generate cloud
    camera = CameraInfo(1280.0, 720.0, intrinsic[0][0], intrinsic[1][1], intrinsic[0][2], intrinsic[1][2], factor_depth)
    cloud = create_point_cloud_from_depth_image(depth, camera, organized=True)

    # get valid points
    mask = (workspace_mask & (depth > 0))
    cloud_masked = cloud[mask]
    color_masked = color[mask]

    # sample points
    if len(cloud_masked) >= cfgs.num_point:
        idxs = np.random.choice(len(cloud_masked), cfgs.num_point, replace=False)
    else:
        idxs1 = np.arange(len(cloud_masked))
        idxs2 = np.random.choice(len(cloud_masked), cfgs.num_point-len(cloud_masked), replace=True)
        idxs = np.concatenate([idxs1, idxs2], axis=0)
    cloud_sampled = cloud_masked[idxs]
    color_sampled = color_masked[idxs]

    # convert data
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(cloud_masked.astype(np.float32))
    cloud.colors = o3d.utility.Vector3dVector(color_masked.astype(np.float32))
    end_points = dict()
    cloud_sampled = torch.from_numpy(cloud_sampled[np.newaxis].astype(np.float32))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cloud_sampled = cloud_sampled.to(device)
    end_points['point_clouds'] = cloud_sampled
    end_points['cloud_colors'] = color_sampled

    return end_points, cloud

def get_grasps(net, end_points):
    # Forward pass
    with torch.no_grad():
        end_points = net(end_points)
        grasp_preds = pred_decode(end_points)
    gg_array = grasp_preds[0].detach().cpu().numpy()
    gg = GraspGroup(gg_array)
    return gg

def collision_detection(gg, cloud):
    mfcdetector = ModelFreeCollisionDetector(cloud, voxel_size=cfgs.voxel_size)
    collision_mask = mfcdetector.detect(gg, approach_dist=0.05, collision_thresh=cfgs.collision_thresh)
    gg = gg[~collision_mask]
    return gg

def vis_grasps(gg, cloud):
    # gg.nms()
    gg.sort_by_score()
    gg = gg[:100]
    grippers = gg.to_open3d_geometry_list()
    o3d.visualization.draw_geometries([cloud, *grippers])

def farthest_point_sampling(point_cloud, num_samples):
    sampled_indices = []
    distances = np.ones(point_cloud.shape[0])

    # 选择第一个点
    current_index = np.random.choice(point_cloud.shape[0])
    sampled_indices.append(current_index)

    for _ in range(1, num_samples):
        current_point = np.asarray(point_cloud[current_index])
        # 计算当前点与所有其他点的距离
        distances = np.minimum(distances, np.linalg.norm(point_cloud - current_point, axis=1))
        # 选择距离最远的点
        current_index = np.argmax(distances)
        sampled_indices.append(current_index)

    return sampled_indices


class GraspNet1b(object):
    def __init__(self, best=False, force_detection=False, visualize=False, **kwargs):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.visualize = visualize
        self.n_grasps = 800

        self.net = get_net()

        self.extrinsics = np.array([[ 1.00000000e+00,  6.12323400e-17, -1.92592994e-34,
        -1.50000000e-01],
       [ 6.12021255e-17, -9.99506560e-01, -3.14107591e-02,
         1.53067060e-01],
       [-1.92335428e-18,  3.14107591e-02, -9.99506560e-01,
         3.95239042e-01],
       [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
         1.00000000e+00]])
        
        self.extrinsics = torch.tensor(self.extrinsics, dtype=torch.float32, device=self.device)

        
    def __call__(self, state, scene_mesh=None, aff_kwargs={}):
        tic = time.time()

        pc = np.asarray(state.pc.points)

        if len(pc) >= 20000:
            idxs = np.random.choice(len(pc), 20000, replace=False)
        else:
            idxs1 = np.arange(len(pc))
            idxs2 = np.random.choice(len(pc), 20000-len(pc), replace=True)
            idxs = np.concatenate([idxs1, idxs2], axis=0)
        P = pc[idxs]

        # pc_viz = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(P))
        # p_cloud_tri = trimesh.points.PointCloud(np.asarray(pc_viz.points))
        # scene = trimesh.Scene([p_cloud_tri,])

        end_points = dict()
        # cloud_sampled = torch.from_numpy(cloud_sampled[np.newaxis].astype(np.float32))
        # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # cloud_sampled = cloud_sampled.to(device)
        
        P = torch.from_numpy(P).float().to(self.device)

        
        P = self.extrinsics @ torch.cat((P,torch.ones_like(P[:,:1])), dim=1).permute(1,0)
        P = P.permute(1,0)[:,:3]

        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(P.cpu().numpy().astype(np.float32))

        end_points['point_clouds'] = P[None]
        end_points['cloud_colors'] = None

        
        gg = get_grasps(self.net, end_points)
        if cfgs.collision_thresh > 0:
            gg = collision_detection(gg, np.array(cloud.points))
        
        gg.sort_by_score()

        # gg.nms()
        
        gg = gg[gg.scores > 0.1]
        # scene.show()
        
        
        # grippers = gg.to_open3d_geometry_list()
        # mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        #         size=0.06, origin=(0,0,0))
        # o3d.visualization.draw_geometries([cloud, mesh_frame, *grippers])

        # pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(P))
        # o3d.visualization.draw_geometries([pcd], window_name="scene", width=800,height=600, left=50, top=50, point_show_normal=False, mesh_show_wireframe=True, mesh_show_back_face=True)


        T_camera_base = Transform.from_matrix(self.extrinsics.cpu().numpy())
        # T_g1_g2 = Transform(Rotation.from_rotvec(np.pi / 2 * np.r_[0.0, -1.0, 0.0]), [0.0, 0.0, +2.1e-02])

        toc = time.time() - tic
        grasps = []
        scores = []
        viz_grasps = []
        new_gg = []
        for i in range(len(gg)):
            pos = gg[i].translation
            ori = Rotation.from_matrix(gg[i].rotation_matrix)
            T_g1_g2 = Transform(Rotation.from_rotvec(np.pi / 2 * np.r_[0.0, -1.0, 0.0]), [0.0, 0.0, 2.1e-02 - (gg[i].depth-0.029) ]) # 0.042-gg[i].depth
            grasp_pose= Grasp(T_camera_base.inverse()*Transform(ori, pos)*T_g1_g2.inverse(), gg[i].width) # T_camera_base.inverse()* X 
            pos = grasp_pose.pose.translation
            
            if bound(pos):
                grasps.append(grasp_pose) #*T_g1_g2.inverse()
                viz_grasps.append(grasp_pose)
                scores.append(gg[i].score)
        
        # vertices = np.asarray(grippers[0].vertices)
        # faces = np.asarray(grippers[0].triangles)
        # gripper_mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        # gripper_mesh2 = visual.grasp2mesh(grasps[0], scores[0])
        # combine_scene = trimesh.Scene(gripper_mesh)
        # combine_scene.add_geometry(gripper_mesh2, node_name='our_grasp')
        # combine_scene.show()
        
        
        grasp_mesh_list = []
        # for j in range(min(20, len(sorted_grasps))):
        #     grasp_mesh_list.append(visual.grasp2mesh(grasps[j], scores[j]))
        # p_cloud_tri = trimesh.points.PointCloud(to_numpy(self.P))
        # scene = trimesh.Scene([p_cloud_tri,]+ grasp_mesh_list)
        # scene.show()

        if self.visualize:
            # grasp_mesh_list = [visual.grasp2mesh(g, s) for g, s in zip(grasps, scores)]
            composed_scene = trimesh.Scene(scene_mesh)
            # gripper_mesh = gg.to_open3d_geometry_list()
            for i in range(len(grasps)):
                
                # gripper_mesh.apply_transform(viz_grasps[i].pose.as_matrix())
                # vertices = np.asarray(gripper_mesh[i].vertices)
                # faces = np.asarray(gripper_mesh[i].triangles)
                # trimesh_mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
                # scene.add_geometry(trimesh_mesh, node_name=f'grasp{i}')
                composed_scene.add_geometry(visual.grasp2mesh(grasps[i], scores[i]), node_name=f'grasp{i}')
            # for i, g_mesh in enumerate(grasp_mesh_list):
            #     composed_scene.add_geometry(g_mesh, node_name=f'grasp_{i}')
            #     break
            composed_scene.show()
            return grasps, scores, toc, composed_scene
        else:
            return grasps, scores, toc

def bound(pred_pos, limit=[0.02, 0.02, 0.055]):
    if pred_pos[0]<limit[0] or pred_pos[0]>0.3 - limit[0] or pred_pos[1]<limit[1] or pred_pos[1]>0.3 - limit[1] or pred_pos[2]<limit[2] or pred_pos[2]>0.3 - limit[2]:
        return False
    return True