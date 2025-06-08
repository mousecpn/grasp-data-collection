import os
import sys
import time

file_dir = os.path.dirname(__file__)
sys.path.append(file_dir)
sys.path.append('..')
import numpy as np
from pathlib import Path
import torch
import pybullet
from utils.grasp import Label
import btsim
from experiment import btsim2 as btsim, workspace_lines
from utils.transform import Rotation, Transform
from scipy.spatial.transform import Slerp
# from segment_anything import SamAutomaticMaskGenerator, build_sam
from experiment import camera
import utils.utils_3d as utils_3d


class ClutterRemovalSim(object):
    def __init__(self, scene, object_set, gui=True, seed=None, rand=False, load_sam=False, n_intervals=36):
        assert scene in ["pile", "packed", "obj", "egad"]
        # self.urdf_root = Path(f"{os.path.abspath(os.path.dirname(__file__))}/data_robot/urdfs/")
        self.urdf_root = Path("data/urdfs")
        self.scene = scene
        self.object_set = f'{scene}/{object_set}'
        # get the list of urdf files or obj files
        self.discover_objects()
        self.rand = rand
        self.global_scaling = {
            "blocks": 1.67,
            "google": 0.7,
            'google_pile': 0.7,
            'google_packed': 0.7,
        }.get(object_set, 1.0)

        self.random_rotations_limit = np.pi * 2.0
        self.n_intervals = n_intervals
        self.mask_generator = None
        self.colormaps, self.heightmaps, self.segmaps, self.maskmaps, self.maskrgbmaps = None, None, None, None, None
        self.pcds_xyz, self.pcds_rgb, self.pcds_masks = None, None, None

        self.gui = gui
        self.rng = np.random.RandomState(seed) if seed else np.random
        self.world = btsim.BtWorld(self.gui)

        self.size = 0.30
        self.camera = self.world.add_camera()
        self.cams_config = camera.RealSenseD415.CONFIG
        X_center = 0.15
        Y_center = 0.15
        Z = 0.05
        self.workspace = np.asarray([[X_center - self.size / 2, X_center + self.size / 2],
                                     [Y_center - self.size / 2, Y_center + self.size / 2],
                                     [Z, Z + self.size]])

        if load_sam:
            self.loadSAM()
        self.bound_radio = 2 if scene == "pile" else 1.5

    @property
    def num_objects(self):
        return max(0, self.world.p.getNumBodies() - 2)  # remove table and the gripper from body count


    def discover_objects(self):
        root = self.urdf_root / self.object_set
        self.object_urdfs = [f for f in root.iterdir() if f.suffix == ".urdf"]
        # self.object_urdfs = self.object_urdfs[:200]
        # print(self.object_urdfs)

    def save_state(self):
        self._snapshot_id = self.world.save_state()

    def restore_state(self):
        self.world.restore_state(self._snapshot_id)

    def reset(self, object_count, index=None, color_=None, pose_=None, from_save=False, offset=0.0625):
        self.heightmaps, self.colormaps, self.segmaps = None, None, None
        self.maskmaps, self.maskrgbmaps = None, None
        self.pcds_xyz, self.pcds_rgb, self.pcds_masks = None, None, None
        self.world.reset()
        self.world.set_gravity([0.0, 0.0, -9.81])
        self.draw_workspace()
        # self.world.p.configureDebugVisualizer(self.world.p.COV_ENABLE_GUI, 0)
        if self.gui:
            self.world.p.resetDebugVisualizerCamera(
                cameraDistance=1.0,
                cameraYaw=0.0,
                cameraPitch=-45,
                cameraTargetPosition=[0.15, 0.50, -0.3],
            )

        table_height = 0.05
        self.place_table(table_height)
        self.gripper = Gripper(self.world, offset=offset)

        indexes, pose_list, color_list = list(), list(), list()
        if self.scene == "pile":
            if from_save:
                self.generate_pile_scene_from_save(index=index, pose_=pose_, color_=color_, table_height=table_height)
            else:
                indexes, pose_list, color_list = self.generate_pile_scene(object_count, table_height)
        elif self.scene == "packed":
            if from_save:
                self.generate_packed_scene_from_save(index=index, pose_=pose_, color_=color_)
            else:
                indexes, pose_list, color_list = self.generate_packed_scene(object_count, table_height)

        else:
            raise ValueError("Invalid scene argument")

        # check validity
        self.remove_and_wait(timeout=3)
        # self.getObservation()
        if index is None:
            return indexes, pose_list, color_list

    def draw_workspace(self):
        points = workspace_lines(self.size)
        color = [0.5, 0.5, 0.5]
        for i in range(0, len(points), 2):
            self.world.p.addUserDebugLine(
                lineFromXYZ=points[i], lineToXYZ=points[i + 1], lineColorRGB=color
            )

    def place_table(self, height):
        urdf = self.urdf_root / "setup" / "plane.urdf"
        pose = Transform(Rotation.identity(), [0.15, 0.15, height])
        self.world.load_urdf(urdf, pose, scale=0.6, env_obj=True)

    def generate_pile_scene(self, object_count, table_height, index=None):
        # place box
        urdf = self.urdf_root / "setup" / "box.urdf"
        pose = Transform(Rotation.identity(), np.r_[0.02, 0.02, table_height])
        box = self.world.load_urdf(urdf, pose, scale=1.3, env_obj=True)
        # drop objects
        if index is not None:
            urdfs = [self.object_urdfs[index]]
        else:
            index = self.rng.choice(len(self.object_urdfs), size=object_count)
            urdfs = [self.object_urdfs[idx] for idx in index]

        pose_list = list()
        color_list = list()

        for i, urdf in enumerate(urdfs):
            if self.rand:
                rotation = Rotation.random(random_state=self.rng)
            else:
                rotation = Rotation.from_matrix(np.array([[1, 0, 0],
                                                          [0, 1, 0],
                                                          [0, 0, 1]]))

            xy = self.rng.uniform(1.0 / 3.0 * self.size, 2.0 / 3.0 * self.size, 2)
            scale = self.rng.uniform(0.8, 1.0)
            pose = Transform(rotation, np.r_[xy, table_height + 0.2], scale=scale)

            _, color = self.world.load_urdf(urdf, pose, scale=self.global_scaling * scale, return_color=True)

            color_list.append(color)
            pose_list.append(pose)

            self.wait_for_objects_to_rest(timeout=2.0)
        # remove box
        self.world.p.removeBody(box.uid)
        self.remove_and_wait()
        return index, pose_list, color_list

    def generate_pile_scene_from_save(self, index=None, pose_=None, color_=None, table_height=0.3):
        """
        Generate the scene from the saved state
        """
        urdf = self.urdf_root / "setup" / "box.urdf"
        pose = Transform(Rotation.identity(), np.r_[0.02, 0.02, table_height])
        box = self.world.load_urdf(urdf, pose, scale=1.3, env_obj=True)

        urdfs = [self.object_urdfs[idx] for idx in index]
        for i, urdf in enumerate(urdfs):
            scale = pose_[i].scale
            self.world.load_urdf(urdf, pose_[i], scale=self.global_scaling * scale, color=color_[i])

            self.wait_for_objects_to_rest(timeout=2.0)
        self.world.p.removeBody(box.uid)
        self.remove_and_wait()

    def generate_packed_scene(self, object_count, table_height):
        attempts = 0
        max_attempts = 12
        pose_list = list()
        color_list = list()
        index_list = list()
        while self.num_objects < object_count and attempts < max_attempts:
            self.save_state()
            index = self.rng.choice(len(self.object_urdfs))
            urdf = self.object_urdfs[index]

            x = self.rng.uniform(0.08, 0.22)
            y = self.rng.uniform(0.08, 0.22)
            z = 1.0
            angle = self.rng.uniform(0.0, 2.0 * np.pi)
            rotation = Rotation.from_rotvec(angle * np.r_[0.0, 0.0, 1.0])
            pose = Transform(rotation, np.r_[x, y, z])
            scale = self.rng.uniform(0.7, 0.8)
            body, color = self.world.load_urdf(urdf, pose, scale=self.global_scaling * scale, return_color=True)
            lower, upper = self.world.p.getAABB(body.uid)
            z = table_height + 0.5 * (upper[2] - lower[2]) + 0.002
            final_pose = Transform(rotation, np.r_[x, y, z], scale=scale)
            body.set_pose(pose=final_pose)
            self.world.step()

            if self.world.p.getContactPoints(body.uid):
                self.world.remove_body(body)
                self.restore_state()
            else:
                self.remove_and_wait()

            pose_list.append(final_pose)
            color_list.append(color)
            index_list.append(index)
            attempts += 1
        return index_list, pose_list, color_list

    def generate_packed_scene_from_save(self, index=None, pose_=None, color_=None):
        for i, idx in enumerate(index):
            self.save_state()
            urdf = self.object_urdfs[idx]

            pose = pose_[i]
            color = color_[i]
            scale = pose.scale
            trans = pose.translation.copy()
            trans[2] = 1.0
            new_pose = Transform(pose.rotation, trans)

            body = self.world.load_urdf(urdf, new_pose, scale=self.global_scaling * scale, color=color)

            body.set_pose(pose=pose)
            self.world.step()

            if self.world.p.getContactPoints(body.uid):
                self.world.remove_body(body)
                self.restore_state()
            else:
                self.remove_and_wait()

    def advance_sim(self, frames):
        for _ in range(frames):
            self.world.step()

    def _random_rotations(self):
        """
        Divide rotation radius into n intervals and randomly pick one from each interval.
        Args:
            n_intervals: Number of intervals to divide into

        Returns:
        A list containing a randomly chosen degree from each interval.

        """
        interval_size = self.random_rotations_limit / self.n_intervals
        intervals = [(i * interval_size, (i + 1) * interval_size) for i in range(self.n_intervals)]

        rotation_list = [np.random.uniform(start, end) for start, end in intervals]

        return np.asarray(rotation_list)

    def rotation_matrix_from_vectors(self, vec1, vec2):
        """ Find the rotation matrix that aligns vec1 to vec2
        :param vec1: A 3d "source" vector
        :param vec2: A 3d "destination" vector
        :return mat: A transform matrix (3x3) which when applied to vec1, aligns it with vec2.
        """
        cross_product = lambda x, y: np.cross(x, y)
        a, b = vec1 / np.linalg.norm(vec1), vec2 / np.linalg.norm(vec2)
        v = cross_product(a, b)
        c = np.dot(a, b)
        s = np.linalg.norm(v)
        if s == 0:  # special case where vectors are parallel or anti-parallel
            if c == 1:
                return Rotation.from_matrix(np.eye(3))  # No rotation needed
            else:
                # 180 degree rotation around any axis perpendicular to vec1
                # Here we choose an axis perpendicular to the maximum component of vec1 for stability
                if abs(a[0]) > abs(a[1]):
                    # x component is larger, choose y axis for rotation to be more stable
                    axis = np.array([0, 1, 0])
                else:
                    # y component is larger or equal, choose x axis for rotation
                    axis = np.array([1, 0, 0])
                rotation_matrix = np.array([
                    [-1, 0, 0],
                    [0, -1, 0],
                    [0, 0, 1]
                ]) if np.allclose(axis, a) else np.array([
                    [-1, 0, 0],
                    [0, 1, 0],
                    [0, 0, -1]
                ])
                return Rotation.from_matrix(rotation_matrix)
        kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
        return Rotation.from_matrix(rotation_matrix)

    def sample_rotations_around_normal(self, normal, gripper_normal=(0, 1, 0)):
        if isinstance(normal, torch.Tensor):
            normal = normal.cpu().numpy()

        # Align the gripper direction to the normal
        rotation_align_gripper_to_normal = self.rotation_matrix_from_vectors(gripper_normal, normal)

        # Generate a series of rotations around the Z axis
        angles_degrees = self._random_rotations()

        rotations = [Rotation.from_rotvec(
            normal * angle) * rotation_align_gripper_to_normal for angle in
                     angles_degrees]

        return rotations, angles_degrees

    def mask_random_actions(self, pcd, sample_strategy, m=20, n_intervals=None, z_thresh=0.051):
        if n_intervals is not None:
            self.n_intervals = n_intervals

        pos = np.asarray(pcd.points)
        normals = np.asarray(pcd.normals)

        actions_list = list()
        points_list = list()
        normals_list = list()

        farthest_pts, index_list = sample_strategy(pos, normals, m, z_thresh)

        for index in index_list:

            point = pos[index]
            normal = normals[index]

            points_list.append(point)
            normals_list.append(normal)

            # rotation_list, planar_rotation_list = self.sample_rotations_around_normal_with_reference(normal)
            rotation_list, planar_rotation_list = self.sample_rotations_around_normal(normal)

            actions_for_point = list()
            for rotation, planar_rotation in zip(rotation_list, planar_rotation_list):
                action = np.concatenate([point, [planar_rotation], rotation.as_quat()])
                actions_for_point.append(action)

            actions_list.append(np.array(actions_for_point))
        # (m, n_intervals, 8), (m,3), (m,3)
        return np.array(actions_list), np.array(points_list), np.array(normals_list), np.array(index_list)

    def mask_all_actions(self, pcd_points, pcd_normals, n_intervals=None, z_thresh=0.052):
        if n_intervals is not None:
            self.n_intervals = n_intervals

        device = pcd_points.device
        actions_list = list()

        indices = torch.where(pcd_points[:, 2] > z_thresh)[0]

        if len(indices) == 0:
            return None, None, None, None

        norms = torch.norm(pcd_normals[indices], dim=1, keepdim=True)
        norms = torch.where(norms == 0, torch.ones_like(norms, device=device), norms)
        normals_normalized = pcd_normals[indices] / norms

        for i, index in enumerate(indices):
            point = pcd_points[index]
            normal = normals_normalized[i]

            rotation_list, rotation_degrees = self.sample_rotations_around_normal(normal)

            actions_for_point = [torch.cat([point, torch.tensor([planar_rotation], device=device, requires_grad=False),
                                            torch.tensor(rotation.as_quat(), device=device, requires_grad=False)])
                                 for rotation, planar_rotation in zip(rotation_list, rotation_degrees)]

            actions_list.append(torch.stack(actions_for_point))
        # (m, n_intervals, 8), (m,3), (m,3)
        return torch.stack(actions_list).to(device), pcd_points[indices].to(device), pcd_normals[indices].to(
            device), indices

    def mask_all_actions_batch(self, pcd_points, pcd_normals, n_intervals=None, z_thresh=0.052):
        if n_intervals is not None:
            self.n_intervals = n_intervals

        device = pcd_points.device
        actions_list = list()

        indices = torch.where(pcd_points[:, 2] > z_thresh)[0]

        if len(indices) == 0:
            return None, None, None, None

        norms = torch.norm(pcd_normals[indices], dim=1, keepdim=True)
        norms = torch.where(norms == 0, torch.ones_like(norms, device=device), norms)
        normals_normalized = pcd_normals[indices] / norms

        # Generate all rotations and planar rotations in a batch
        rotations, planar_rotations = zip(
            *[self.sample_rotations_around_normal_batch(normal) for normal in normals_normalized])

        # Concatenate rotations and planar rotations for batch processing
        rotations = [torch.tensor([rotation.as_quat() for rotation in rotation_set], device=device) for rotation_set in
                     rotations]
        planar_rotations = torch.tensor(planar_rotations, device=device)

        # Create actions for all points in batch
        points_expanded = pcd_points[indices].unsqueeze(1).expand(-1, self.n_intervals, -1)
        planar_rotations_expanded = planar_rotations.unsqueeze(2).expand(-1, -1, 1)
        rotations_expanded = torch.stack(rotations).to(device)

        actions_for_points = torch.cat([points_expanded, planar_rotations_expanded, rotations_expanded], dim=2)

        return actions_for_points, pcd_points[indices].to(device), pcd_normals[indices].to(device), indices

    def sample_rotations_around_normal_batch(self, normal, gripper_normal=(0, 1, 0)):
        if isinstance(normal, torch.Tensor):
            normal = normal.cpu().numpy()

        # Align the gripper direction to the normal
        rotation_align_gripper_to_normal = self.rotation_matrix_from_vectors(gripper_normal, normal)

        # Generate a series of rotations around the Z axis
        angles_degrees = self._random_rotations()

        rotations = [Rotation.from_rotvec(normal * angle) * rotation_align_gripper_to_normal for angle in
                     angles_degrees]

        return rotations, angles_degrees

    def get_segmentation_fast(self):
        if self.mask_generator is None:
            self.loadSAM()
        self.maskmaps = []
        for i in range(len(self.colormaps)):
            masks_origin = self.mask_generator.generate(self.colormaps[i])
            if len(masks_origin) != 0:
                self.maskmaps.append(masks_origin)
            else:
                self.maskmaps.append([])

    def remove_and_wait(self, timeout=2.0):
        # wait for objects to rest while removing bodies that fell outside the workspace
        removed_object = True
        while removed_object:
            self.wait_for_objects_to_rest(timeout=timeout)
            removed_object = self.remove_objects_outside_workspace()

    def wait_for_objects_to_rest(self, timeout=2.0, tol=0.01):
        timeout = self.world.sim_time + timeout
        objects_resting = False
        while not objects_resting and self.world.sim_time < timeout:
            # simulate a quarter of a second
            for _ in range(60):
                self.world.step()
            # check whether all objects are resting
            objects_resting = True
            for _, body in self.world.bodies.items():
                if np.linalg.norm(body.get_velocity()) > tol:
                    objects_resting = False
                    break

    def remove_objects_outside_workspace(self):
        removed_object = False
        for body in list(self.world.bodies.values()):
            xyz = body.get_pose().translation
            if np.any(xyz < 0.0) or np.any(xyz > self.size):
                self.world.remove_body(body)
                removed_object = True
        return removed_object

    def check_success(self, gripper):
        # check that the fingers are in contact with some object and not fully closed
        contacts = self.world.p.getContactPoints(gripper.body.uid)
        res = len(contacts) > 0 and gripper.read() > 0.1 * gripper.max_opening_width
        return res

    def calculate_rectangle_vertices(self, length, width, height, tcp, quat):

        rotation_matrix = Rotation.from_quat(quat).as_matrix()

        vectors_local = [
            np.array([+width / 2, +length / 2, height]),
            np.array([-width / 2, +length / 2, height]),
            np.array([-width / 2, -length / 2, height]),
            np.array([+width / 2, -length / 2, height]),
        ]

        vertices = []
        for v in vectors_local:
            v_rotated = rotation_matrix.dot(v) + tcp
            vertices.append(v_rotated)

        return np.asarray(vertices)

    def calculate_rectangle_vertices_batch(self, length, width, height, tcp_batch, quat_batch):
        # Convert quaternion batch to rotation matrices
        rotation_matrices = Rotation.from_quat(quat_batch).as_matrix()

        # Define the local vectors for the rectangle vertices
        vectors_local = np.array([
            [+width / 2, +length / 2, height],
            [-width / 2, +length / 2, height],
            [-width / 2, -length / 2, height],
            [+width / 2, -length / 2, height]
        ])

        batch_size = tcp_batch.shape[0]

        # Repeat vectors_local for each item in the batch
        vectors_local_batch = np.tile(vectors_local, (batch_size, 1, 1))

        # Perform batch rotation and translation
        rotated_vectors = np.einsum('bij,bvj->bvi', rotation_matrices, vectors_local_batch)
        vertices_batch = rotated_vectors + tcp_batch[:, np.newaxis, :]

        return vertices_batch

    def decode_action_batch(self, actions, translation_offset=0.008, z_thresh=0.051, vertical_offset=0.007):
        orig_positions = actions[:, :3]
        rots = actions[:, 3:7]
        normal_vectors = actions[:, 7:]

        unit_normal_vectors = normal_vectors / np.linalg.norm(normal_vectors, axis=1, keepdims=True)

        translation_distance = np.full((actions.shape[0], 1), self.gripper.max_opening_width / 2 - translation_offset)
        translation_vectors = -unit_normal_vectors * translation_distance
        positions_after_translation = orig_positions + translation_vectors

        finger_positions = self.calculate_rectangle_vertices_batch(length=0.09, width=0.018, height=0.009,
                                                                   tcp_batch=positions_after_translation,
                                                                   quat_batch=rots)

        graspable_mask = np.all(finger_positions[:, :, 2] > z_thresh, axis=1)

        graspable_results = np.where(graspable_mask, 1, 0)

        # for i, graspable in enumerate(graspable_mask):
        #     if not graspable:
        #         rotation = Rotation.from_quat(rots[i])
        #         global_z_direction = rotation.apply([0, 0, 1])
        #
        #         cur_vertical_offset = -5e-4
        #         while cur_vertical_offset >= -vertical_offset:
        #             z_translation_vector = global_z_direction * cur_vertical_offset
        #             final_position = positions_after_translation[i] + z_translation_vector
        #
        #             final_finger_positions = finger_positions[i] + z_translation_vector
        #
        #             final_z = np.min(final_finger_positions[:, 2])
        #
        #             if final_z >= z_thresh:
        #                 graspable_results[i] = 1
        #                 positions_after_translation[i] = final_position
        #                 break
        #             cur_vertical_offset -= 5e-4

        return graspable_results, positions_after_translation, rots

    def _decodeAction(self, action, translation_offset=0.008, vertical_offset=0.007, z_thresh=0.051):

        orig_position, rot, normal_vector = action

        unit_normal_vector = normal_vector / np.linalg.norm(normal_vector)

        translation_distance = self.gripper.max_opening_width / 2 - translation_offset
        translation_vector = -unit_normal_vector * translation_distance
        position_after_translation = orig_position + translation_vector

        finger_positions = self.calculate_rectangle_vertices(length=0.09, width=0.018, height=0.009,
                                                             tcp=position_after_translation,
                                                             quat=rot)

        if np.all(finger_positions[:, 2] > z_thresh):
            return (1, position_after_translation, rot)

        rotation = Rotation.from_quat(rot)
        global_z_direction = rotation.apply([0, 0, 1])

        cur_vertical_offset = -5e-4
        while cur_vertical_offset >= -vertical_offset:
            z_translation_vector = global_z_direction * cur_vertical_offset
            final_position = position_after_translation + z_translation_vector

            final_finger_positions = finger_positions + z_translation_vector

            final_z = np.min(final_finger_positions[:, 2])

            if final_z >= z_thresh:
                return (1, final_position, rot)
            cur_vertical_offset -= 5e-4

        return (0, position_after_translation, rot)

    def execute_grasp(self, action, remove=False, allow_contact=True, waitTime=0, z_thresh=0.052):
        graspable, pos, rot_q = self._decodeAction(action, z_thresh=z_thresh)
        grasped_object_ind = None

        if graspable == 0:
            self.gripper.holding_obj = None
            self.gripper.move_home()
            result = Label.FAILURE, self.gripper.max_opening_width, 'pregrasp'
            return result, grasped_object_ind, graspable

        grasp_pose = Transform(Rotation.from_quat(rot_q), pos)

        # # Take action specified by motion primitive
        # self.panda.pick(pick_info, objects=self.world.bodies.values())
        T_world_grasp = grasp_pose
        T_grasp_pregrasp = Transform(Rotation.identity(), [0.0, 0.0, -0.05])
        T_world_pregrasp = T_world_grasp * T_grasp_pregrasp
        approach = T_world_grasp.rotation.as_matrix()[:, 2]
        angle = np.arccos(np.dot(approach, np.r_[0.0, 0.0, -1.0]))

        if angle > np.pi / 3.0:
            # side grasp, lift the object after establishing a grasp
            T_grasp_pregrasp_world = Transform(Rotation.identity(), [0.0, 0.0, 0.1])
            T_world_retreat = T_grasp_pregrasp_world * T_world_grasp
        else:
            T_grasp_retreat = Transform(Rotation.identity(), [0.0, 0.0, -0.1])
            T_world_retreat = T_world_grasp * T_grasp_retreat

        self.gripper.reset()
        self.gripper.set_tcp(T_world_pregrasp)
        self.world.step()
        if self.gripper.detect_contact():
            result = Label.FAILURE, self.gripper.max_opening_width, 'pregrasp'
            self.gripper.holding_obj = None
            self.gripper.move_home()
        else:
            self.gripper.move_tcp_xyz(T_world_grasp, abort_on_contact=False)
            if self.gripper.detect_contact() and not allow_contact:
                result = Label.FAILURE, self.gripper.max_opening_width, 'grasp'
                self.gripper.holding_obj = None
                self.gripper.move_home()
            else:
                self.gripper.move(0.0)
                self.gripper.move_tcp_xyz(T_world_retreat, abort_on_contact=False)
                self.gripper.holding_obj = self.gripper.getPickedObj()
                if self.gripper.holding_obj:
                    result = Label.SUCCESS, self.gripper.read(), 'success'
                    # if self.check_success(self.gripper):
                    dis_from_hand = self.gripper.get_distance_from_hand()
                    shake_label = self.gripper.shake_hand(dis_from_hand)
                        #print('finish shaking')
                    if self.check_success(self.gripper) and shake_label:
                        result = Label.SUCCESS, self.gripper.read(),'success'
                        if remove:
                            contacts = self.world.get_contacts(self.gripper.body)
                            self.world.remove_body(contacts[0].bodyB)
                    else:
                        result =  Label.FAILURE, self.gripper.max_opening_width, 'after'
                    # if remove:
                    #     contacts = self.world.get_contacts(self.gripper.body)
                    #     self.world.remove_body(contacts[0].bodyB)
                else:
                    result = Label.FAILURE, self.gripper.max_opening_width, 'after'
                self.gripper.move_home()
                self.world.step()

        if self.gripper.holding_obj is not None:
            grasped_object_ind = self.gripper.holding_obj.uid

        self.wait_for_objects_to_rest(waitTime)
        if remove:
            self.remove_and_wait()
        return result, grasped_object_ind, graspable

    def execute_grasp_fast(self, action, remove=True, allow_contact=True, waitTime=2.0):
        pos, rot_q = action

        grasp_pose = Transform(Rotation.from_quat(rot_q), pos)
        T_world_grasp = grasp_pose
        T_grasp_pregrasp = Transform(Rotation.identity(), [0.0, 0.0, -0.05])
        T_world_pregrasp = T_world_grasp * T_grasp_pregrasp

        approach = T_world_grasp.rotation.as_matrix()[:, 2]
        angle = np.arccos(np.dot(approach, np.r_[0.0, 0.0, -1.0]))
        if angle > np.pi / 3.0:
            # side grasp, lift the object after establishing a grasp
            T_grasp_pregrasp_world = Transform(Rotation.identity(), [0.0, 0.0, 0.1])
            T_world_retreat = T_grasp_pregrasp_world * T_world_grasp
        else:
            T_grasp_retreat = Transform(Rotation.identity(), [0.0, 0.0, -0.1])
            T_world_retreat = T_world_grasp * T_grasp_retreat

        self.gripper.reset()
        self.gripper.set_tcp(T_world_pregrasp)
        self.world.step()
        if self.gripper.detect_contact():
            result = Label.FAILURE, self.gripper.max_opening_width, 'pregrasp'
            self.gripper.holding_obj = None
            self.gripper.move_home()
        else:
            self.gripper.move_tcp_xyz(T_world_grasp, abort_on_contact=False)
            if self.gripper.detect_contact() and not allow_contact:
                result = Label.FAILURE, self.gripper.max_opening_width, 'grasp'
                self.gripper.holding_obj = None
                self.gripper.move_home()
            else:
                self.gripper.move(0.0)
                self.gripper.move_tcp_xyz(T_world_retreat, abort_on_contact=False)
                if self.check_success(self.gripper):
                    result = Label.SUCCESS, self.gripper.read(), 'success'
                    if remove:
                        contacts = self.world.get_contacts(self.gripper.body)
                        self.world.remove_body(contacts[0].bodyB)
                else:
                    result = Label.FAILURE, self.gripper.max_opening_width, 'after'
                self.gripper.move_home()
                self.world.step()
        self.wait_for_objects_to_rest(waitTime)
        if remove:
            self.remove_and_wait()
        return result

    def eval_single_grasp(self, index, segmentation_gt, grasp_obj_id):
        """
        validate the grasp whether success or not
        Args:
            index: the index of the grasp point in the point cloud
            segmentation_gt: the ground truth of the segmentation from the pybullet.
                             For the given pixel (x, y), get its corresponding object ID.
            grasp_obj_id: the uid of the grasping object

        Returns: whether success or not
        """
        if grasp_obj_id is None:
            return False, None
        target_object_id = segmentation_gt[index]
        if target_object_id == grasp_obj_id:
            return True, grasp_obj_id

        return False, grasp_obj_id

    def getSingleViewObservation(self, n):
        '''get raw inbound pcds and imgs,
           the pcds have lots zero values, if you need filter later,
           please use self.pcds_rgb = self.pcds_rgb[self.non_zero_indices]'''
        self.colormaps, self.heightmaps, self.segmaps = self.camera.render_camera(self.cams_config[n])
        self.pcds_xyz, self.pcds_rgb, self.colormaps, self.heightmaps = utils_3d.get_inbound_imgs_and_pcd(
            self.colormaps,
            self.heightmaps,
            self.cams_config[n],
            self.workspace
        )
        self.non_zero_indices = ~np.all(self.pcds_xyz == 0, axis=1)

    def getRandomViewObservation(self, config):
        '''get raw inbound pcds and imgs,
           the pcds have lots zero values, if you need filter later,
           please use self.pcds_rgb = self.pcds_rgb[self.non_zero_indices]'''
        self.colormaps, self.heightmaps, self.segmaps = self.camera.render_camera(config)
        self.pcds_xyz, self.pcds_rgb, self.colormaps, self.heightmaps = utils_3d.get_inbound_imgs_and_pcd(
            self.colormaps,
            self.heightmaps,
            config,
            self.workspace
        )
        self.non_zero_indices = ~np.all(self.pcds_xyz == 0, axis=1)

    def getRandomViewObservation_fast(self, config):
        '''get raw inbound pcds,
           the pcds have lots zero values, if you need filter later,
           please use self.pcds_rgb = self.pcds_rgb[self.non_zero_indices]'''
        _, self.heightmaps, __ = self.camera.render_camera(config)
        self.pcds_xyz, self.pcds_rgb, self.colormaps, self.heightmaps = utils_3d.get_inbound_imgs_and_pcd(
            self.colormaps,
            self.heightmaps,
            config,
            self.workspace
        )
        self.non_zero_indices = ~np.all(self.pcds_xyz == 0, axis=1)


    def close(self):
        self.world.close()


class Gripper(object):
    """Simulated Panda hand."""

    def __init__(self, world, offset=0.0625):
        self.world = world
        # self.urdf_path = Path(f"{os.path.abspath(os.path.dirname(__file__))}/data_robot/urdfs/panda/hand.urdf")
        self.urdf_path = Path("data/urdfs/panda/hand.urdf")

        self.max_opening_width = 0.08
        self.finger_depth = 0.05

        self.holding_obj = None
        self.position = np.array((0.15, 0.15, 0.6))

        self.T_body_tcp = Transform(Rotation.identity(), [0.0, 0.0, offset])
        self.T_tcp_body = self.T_body_tcp.inverse()
        self.home_pose = Transform(Rotation.from_euler('x', 180, degrees=True), self.position)
        self.body = self.world.load_urdf(self.urdf_path, self.home_pose * self.T_tcp_body, scale=1.0, env_obj=True)
        self.world.p.changeDynamics(self.body.uid, 0, lateralFriction=0.75, spinningFriction=0.05)
        self.world.p.changeDynamics(self.body.uid, 1, lateralFriction=0.75, spinningFriction=0.05)

        # sets the position of the COM, not URDF link
        self.constraint = self.world.add_constraint(
            self.body,
            None,
            None,
            None,
            pybullet.JOINT_FIXED,
            [0.0, 0.0, 0.0],
            Transform.identity(),
            self.home_pose * self.T_tcp_body
        )

        # constraint to keep fingers centered
        self.world.add_constraint(
            self.body,
            self.body.links["panda_leftfinger"],
            self.body,
            self.body.links["panda_rightfinger"],
            pybullet.JOINT_GEAR,
            [1.0, 0.0, 0.0],
            Transform.identity(),
            Transform.identity(),
        ).change(gearRatio=-1, erp=0.1, maxForce=30)

        self.reset()

    def reset(self, opening_width=0.08):

        self.holding_obj = None
        self.joint1 = self.body.joints["panda_finger_joint1"]
        self.joint1.set_position(0.5 * opening_width, kinematics=True)
        self.joint2 = self.body.joints["panda_finger_joint2"]
        self.joint2.set_position(0.5 * opening_width, kinematics=True)

    def update_tcp_constraint(self, T_world_tcp):
        T_world_body = T_world_tcp * self.T_tcp_body
        self.constraint.change(
            jointChildPivot=T_world_body.translation,
            jointChildFrameOrientation=T_world_body.rotation.as_quat(),
            maxForce=300,
        )

    def set_tcp(self, T_world_tcp):
        T_word_body = T_world_tcp * self.T_tcp_body
        self.body.set_pose(T_word_body)
        self.update_tcp_constraint(T_world_tcp)

    def move_home(self):
        self.set_tcp(self.home_pose)

    def move_tcp_xyz(self, target, eef_step=0.002, vel=0.10, abort_on_contact=True):
        T_world_body = self.body.get_pose()
        T_world_tcp = T_world_body * self.T_body_tcp

        diff = target.translation - T_world_tcp.translation
        n_steps = int(np.linalg.norm(diff) / eef_step)
        dist_step = diff / n_steps
        dur_step = np.linalg.norm(dist_step) / vel

        for _ in range(n_steps):
            T_world_tcp.translation += dist_step
            self.update_tcp_constraint(T_world_tcp)
            for _ in range(int(dur_step / self.world.dt)):
                self.world.step()
            if abort_on_contact and self.detect_contact():
                return

    def detect_contact(self):
        # time.sleep(1)
        if self.world.p.getContactPoints(self.body.uid):
            return True
        else:
            return False

    def grasp_object_id(self):
        contacts = self.world.get_contacts(self.body)
        for contact in contacts:
            # contact = contacts[0]
            # get rid body
            grased_id = contact.bodyB
            if grased_id.uid != self.body.uid:
                return grased_id.uid
        return None

    def getPickedObj(self):

        for body in self.world.bodies.values():
            if body.uid == self.body.uid:
                continue
            contact_points_left = self.world.p.getContactPoints(self.body.uid, body.uid,
                                                                self.body.links["panda_leftfinger"].link_index)
            contact_points_right = self.world.p.getContactPoints(self.body.uid, body.uid,
                                                                 self.body.links["panda_rightfinger"].link_index)
            if contact_points_left and contact_points_right:
                return body

        return None

    def move(self, width):
        self.joint1.set_position(0.5 * width)
        self.joint2.set_position(0.5 * width)
        for _ in range(int(0.5 / self.world.dt)):
            self.world.step()

    def read(self):
        width = self.joint1.get_position() + self.joint2.get_position()
        return width

    def move_tcp_pose(self, target, eef_step1=0.002, vel1=0.10, abs=False):
        T_world_body = self.body.get_pose()
        T_world_tcp = T_world_body * self.T_body_tcp
        pos_diff = target.translation - T_world_tcp.translation
        n_steps = max(int(np.linalg.norm(pos_diff) / eef_step1), 10)
        dist_step = pos_diff / n_steps
        dur_step = np.linalg.norm(dist_step) / vel1
        key_rots = np.stack((T_world_body.rotation.as_quat(), target.rotation.as_quat()), axis=0)
        key_rots = Rotation.from_quat(key_rots)
        slerp = Slerp([0.0, 1.0], key_rots)
        times = np.linspace(0, 1, n_steps)
        orientations = slerp(times).as_quat()
        for ii in range(n_steps):
            T_world_tcp.translation += dist_step
            T_world_tcp.rotation = Rotation.from_quat(orientations[ii])
            if abs is True:
                # todo by haojie add the relation transformation later
                self.constraint.change(
                    jointChildPivot=T_world_tcp.translation,
                    jointChildFrameOrientation=T_world_tcp.rotation.as_quat(),
                    maxForce=300,
                )
            else:
                self.update_tcp_constraint(T_world_tcp)
            for _ in range(int(dur_step / self.world.dt)):
                self.world.step()

    def move_gripper_top_down(self):
        current_pose = self.body.get_pose()
        pos = current_pose.translation + 0.1
        flip = Rotation.from_euler('y', np.pi)
        target_ori = Rotation.identity() * flip
        self.move_tcp_pose(Transform(rotation=target_ori, translation=pos), abs=True)

    def shake_hand(self,pre_dist):
        grasp_id = self.grasp_object_id()
        current_pose = self.body.get_pose()
        x,y,z = current_pose.translation[0],current_pose.translation[1],current_pose.translation[2]
        default_position = [x, y, z]
        shake_position = [x, y, z+0.05]
        hand_orientation2 = pybullet.getQuaternionFromEuler([np.pi, 0, -np.pi/2])
        shake_orientation1 = pybullet.getQuaternionFromEuler([np.pi, -np.pi / 12, -np.pi/2])
        shake_orientation2 = pybullet.getQuaternionFromEuler([np.pi, np.pi / 12, -np.pi/2])
        new_trans = current_pose.translation + np.array([0.,0.,0.05])
        self.move_tcp_pose(target=Transform(rotation=Rotation.from_quat(hand_orientation2),translation=new_trans))
        #check drop
        if self.is_dropped(grasp_id,pre_dist):
            return False
        self.move_tcp_pose(target=Transform(rotation=Rotation.from_quat(hand_orientation2), translation=default_position))
        #check drop
        if self.is_dropped(grasp_id,pre_dist):
            return False
        self.move_tcp_pose(target=Transform(rotation=Rotation.from_quat(hand_orientation2), translation=shake_position))
        # check drop
        if self.is_dropped(grasp_id,pre_dist):
            return False
        self.move_tcp_pose(target=Transform(rotation=Rotation.from_quat(hand_orientation2), translation=default_position))
        # check drop
        if self.is_dropped(grasp_id,pre_dist):
            return False
        self.move_tcp_pose(target=Transform(rotation=Rotation.from_quat(shake_orientation1), translation=default_position))
        # check drop
        if self.is_dropped(grasp_id,pre_dist):
            return False
        self.move_tcp_pose(target=Transform(rotation=Rotation.from_quat(shake_orientation2), translation=default_position))
        # check drop
        if self.is_dropped(grasp_id,pre_dist):
            return False
        else:
            return True
        
    def is_dropped(self,object_id,prev_dist):
        pos,_ = pybullet.getBasePositionAndOrientation(object_id)
        dist_from_hand = np.linalg.norm(np.array(pos) - np.array(self.body.get_pose().translation))
        if np.isclose(prev_dist,dist_from_hand,atol=0.1):
            return False
        else:
            return True
    
    def get_distance_from_hand(self,):
        object_id = self.grasp_object_id()
        pos, _ = pybullet.getBasePositionAndOrientation(object_id)
        dist_from_hand = np.linalg.norm(np.array(pos) - np.array(self.body.get_pose().translation))
        return dist_from_hand