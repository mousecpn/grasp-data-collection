import argparse
from pathlib import Path

import numpy as np
import open3d as o3d
import scipy.signal as signal
from tqdm import tqdm
import multiprocessing as mp

from utils.grasp import Grasp, Label
from utils.io import *
from utils.perception import *
from utils.visual import grasp2mesh
from experiment.simulation import ClutterRemovalSim
from utils.transform import Rotation, Transform
from utils.implicit import get_mesh_pose_list_from_world
from model.graspnet1b_detection import GraspNet1b
import os
from utils.implicit import get_scene_from_mesh_pose_list
import collections
State = collections.namedtuple("State", ["tsdf", "pc"])

OBJECT_COUNT_LAMBDA = 4
MAX_VIEWPOINT_COUNT = 6
MAX_ROTATION = 6

def trimesh_to_open3d(trimesh_mesh):
    """
    Converts a Trimesh object to an Open3D object.

    Args:
        trimesh_mesh (trimesh.Trimesh): A Trimesh object.

    Returns:
        open3d.geometry.TriangleMesh: The corresponding Open3D mesh.
    """
    # Extract vertices and faces from Trimesh
    vertices = trimesh_mesh.vertices
    faces = trimesh_mesh.faces

    # Create Open3D TriangleMesh
    o3d_mesh = o3d.geometry.TriangleMesh()
    o3d_mesh.vertices = o3d.utility.Vector3dVector(vertices)
    o3d_mesh.triangles = o3d.utility.Vector3iVector(faces)

    # Optionally compute normals
    o3d_mesh.compute_vertex_normals()

    return o3d_mesh 

def main(args, rank):
    GRASPS_PER_SCENE = args.grasps_per_scene
    np.random.seed()
    seed = np.random.randint(0, 1000) + rank
    np.random.seed(seed)
    sim = ClutterRemovalSim(args.scene, args.object_set, gui=args.sim_gui)
    finger_depth = sim.gripper.finger_depth
    grasps_per_worker = args.num_grasps // args.num_proc
    pbar = tqdm(total=grasps_per_worker, disable=rank != 0)

    plan_grasps = GraspNet1b()

    if rank == 0:
        if not os.path.exists(args.root / "scenes"):
            (args.root / "scenes").mkdir(parents=True)
            write_setup(
                args.root,
                sim.size,
                sim.camera.intrinsic,
                sim.gripper.max_opening_width,
                sim.gripper.finger_depth,
            )
            if args.save_scene:
                (args.root / "mesh_pose_list").mkdir(parents=True)

    for _ in range(grasps_per_worker // GRASPS_PER_SCENE):
        # generate heap
        object_count = np.random.poisson(OBJECT_COUNT_LAMBDA) + 1
        sim.reset(object_count)
        sim.save_state()

        # render synthetic depth images
        n = MAX_VIEWPOINT_COUNT
        depth_imgs, extrinsics = render_images(sim, n)
        depth_imgs_side, extrinsics_side = render_side_images(sim, 1, args.random)

        # reconstrct point cloud using a subset of the images
        tsdf = create_tsdf(sim.size, 120, depth_imgs, sim.camera.intrinsic, extrinsics)
        pc = tsdf.get_cloud()

        # crop surface and borders from point cloud
        # bounding_box = o3d.geometry.AxisAlignedBoundingBox(sim.lower, sim.upper)
        # pc = pc.crop(bounding_box)
        # o3d.visualization.draw_geometries([pc])

        if pc.is_empty():
            print("Point cloud empty, skipping scene")
            continue

        # store the raw data
        if not args.debug:
            scene_id = write_sensor_data(args.root, depth_imgs_side, extrinsics_side)
            if args.save_scene:
                mesh_pose_list = get_mesh_pose_list_from_world(sim.world, args.object_set)
                write_mesh(args.root, scene_id, mesh_pose_list, name="mesh_pose_list")
        else:
            mesh_pose_list = get_mesh_pose_list_from_world(sim.world, args.object_set)
        
        # use GPD to sample grasps
        state = State(tsdf, pc)
        grasp_mesh_list = []
        candidates, scores, planning_time = plan_grasps(state)
        # for i in range(GRASPS_PER_SCENE):
        for i in range(len(candidates)):
            grasp, label = evaluate_grasp_giga(sim, candidates[i])
            if not args.debug:
                if label == 1:
                    write_grasp(args.root, scene_id, grasp, label)  # scores[i]
                else:
                    write_grasp(args.root, scene_id, grasp, label)
            else:
                if label == 1:
                    grasp_mesh_list.append(trimesh_to_open3d(grasp2mesh(grasp, label)))
            pbar.update()
        if args.debug:
            scene = trimesh_to_open3d(get_scene_from_mesh_pose_list(mesh_pose_list, return_list=False))
            o3d.visualization.draw_geometries([scene, ]+ grasp_mesh_list,
                                                window_name="Point Cloud with Normals",
                                                point_show_normal=True)
        

    pbar.close()
    print('Process %d finished!' % rank)


def render_images(sim, n):
    height, width = sim.camera.intrinsic.height, sim.camera.intrinsic.width
    origin = Transform(Rotation.identity(), np.r_[sim.size / 2, sim.size / 2, 0.0])

    extrinsics = np.empty((n, 7), np.float32)
    depth_imgs = np.empty((n, height, width), np.float32)

    for i in range(n):
        r = np.random.uniform(1.6, 2.4) * sim.size
        theta = np.random.uniform(0.0, np.pi / 4.0)
        phi = np.random.uniform(0.0, 2.0 * np.pi)

        extrinsic = camera_on_sphere(origin, r, theta, phi)
        depth_img = sim.camera.render(extrinsic)[1]

        extrinsics[i] = extrinsic.to_list()
        depth_imgs[i] = depth_img

    return depth_imgs, extrinsics

def render_side_images(sim, n=1, random=False):
    height, width = sim.camera.intrinsic.height, sim.camera.intrinsic.width
    origin = Transform(Rotation.identity(), np.r_[sim.size / 2, sim.size / 2, sim.size / 3])

    extrinsics = np.empty((n, 7), np.float32)
    depth_imgs = np.empty((n, height, width), np.float32)

    for i in range(n):
        if random:
            # r = np.random.uniform(1.6, 2.4) * sim.size
            # theta = np.random.uniform(np.pi / 4.0, 5.0 * np.pi / 12.0)
            # phi = np.random.uniform(- 5.0 * np.pi / 5, - 3.0 * np.pi / 8.0)
            r = np.random.uniform(1.5, 2) * sim.size
            theta = np.random.uniform(np.pi / 4, np.pi / 2.4)
            phi = np.random.uniform(0.0, np.pi)
            origin = Transform(Rotation.identity(), np.r_[sim.size / 2, sim.size / 2, 0.0 + 0.15])
        else:
            r = 2 * sim.size
            theta = np.pi / 3.0
            phi = - np.pi / 2.0

        extrinsic = camera_on_sphere(origin, r, theta, phi)
        depth_img = sim.camera.render(extrinsic)[1]

        extrinsics[i] = extrinsic.to_list()
        depth_imgs[i] = depth_img

    return depth_imgs, extrinsics


def evaluate_grasp_giga(sim, candidate):
    # outcomes, widths = [], []
    # grasps = []
    # for candidate in candidates:
    sim.restore_state()
    outcome, width = sim.execute_grasp(candidate, remove=False)

    return Grasp(candidate.pose, width), int(np.max(outcome))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path,default=Path("data/data_pile_train_raw"))
    parser.add_argument("--scene", type=str, choices=["pile", "packed"], default="pile")
    parser.add_argument("--object-set", type=str, default="pile")
    parser.add_argument("--num-grasps", type=int, default=10000)
    parser.add_argument("--grasps-per-scene", type=int, default=120)
    parser.add_argument("--num-proc", type=int, default=1)
    parser.add_argument("--save-scene", action="store_true")
    parser.add_argument("--random", action="store_true", help="Add distrubation to camera pose")
    parser.add_argument("--sim-gui", action="store_true")
    parser.add_argument("--debug", action="store_true",default=True)
    args = parser.parse_args()
    mp.set_start_method('spawn')
    args.save_scene = True
    if args.num_proc > 1:
        pool = mp.Pool(processes=args.num_proc)
        for i in range(args.num_proc):
            pool.apply_async(func=main, args=(args, i))
        pool.close()
        pool.join()
    else:
        main(args, 0)
