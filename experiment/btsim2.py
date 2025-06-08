from pybullet_utils import bullet_client
import time
import numpy as np
import pybullet
from utils.transform import Transform, Rotation

assert pybullet.isNumpyEnabled(), "Pybullet needs to be built with NumPy"

"""
NOTE:
We modified the original env from GIGA (EdgeGrasp) by adding the color parameter to the load_urdf function.
Therefore, multiple functions in the original code have been modified to accommodate this change.
We added this parameter to keep the performance of pretrained vision model (SAM2) consistent.

Besides, we also modified the camera settings in the BtWorld class to make it easier to use.
e.g., we defined a new Camera2 class to render RGB-D images with specified camera configuration.
 In the BtWorld class, we added a new function add_camera2 to create a Camera2 object.
"""


class BtWorld(object):
    """Interface to a PyBullet physics server.
    Attributes:
        dt: Time step of the physics simulation.
        rtf: Real time factor. If negative, the simulation is run as fast as possible.
        sim_time: Virtual time elpased since the last simulation reset.
    """

    def __init__(self, gui=True):
        connection_mode = pybullet.GUI if gui else pybullet.DIRECT
        self.p = bullet_client.BulletClient(connection_mode)
        self.p.configureDebugVisualizer(flag=self.p.COV_ENABLE_SHADOWS, enable=1)

        self.gui = gui
        self.dt = 1.0 / 240.0
        self.sleep = 1.0 / 80.0
        self.solver_iterations = 100
        self.reset()

    def set_gravity(self, gravity):
        self.p.setGravity(*gravity)

    def load_urdf(self, urdf_path, pose, scale=1.0, env_obj=False, color=None, return_color=False):
        # the plane don't have mass
        body, color = Body.from_urdf(self.p, urdf_path, pose, scale, table=env_obj, color_=color)
        if not env_obj:
            self.bodies[body.uid] = body
        if return_color:
            return body, color
        return body

    def load_obj(self, urdf_path, pose, scale=1.0, color=None, return_color=False):
        # the plane don't have mass
        body, color = Body.from_obj(self.p, urdf_path, pose, scale, color_=color)
        self.bodies[body.uid] = body
        if return_color:
            return body, color
        return body

    def remove_body(self, body):
        self.p.removeBody(body.uid)
        del self.bodies[body.uid]

    def add_constraint(self, *argv, **kwargs):
        """See `Constraint` below."""
        constraint = Constraint(self.p, *argv, **kwargs)
        return constraint

    def add_camera(self):
        camera = Camera2(self.p)
        return camera

    def get_contacts(self, bodyA):
        points = self.p.getContactPoints(bodyA.uid)
        contacts = []
        for point in points:
            contact = Contact(
                bodyA=bodyA,
                bodyB=self.bodies[point[2]],
                point=point[5],
                normal=point[7],
                depth=point[8],
                force=point[9],
            )
            contacts.append(contact)
        return contacts

    def reset(self):
        self.p.resetSimulation()
        self.p.setPhysicsEngineParameter(
            numSubSteps=0,
            fixedTimeStep=self.dt,
            numSolverIterations=self.solver_iterations,
            solverResidualThreshold=1e-7,
        )
        self.bodies = {}
        self.sim_time = 0.0

    def step(self):
        self.p.stepSimulation()
        self.sim_time += self.dt
        if self.gui:
            time.sleep(self.sleep)
            pass

    def save_state(self):
        return self.p.saveState()

    def restore_state(self, state_uid):
        self.p.restoreState(stateId=state_uid)

    def close(self):
        self.p.disconnect()


class Body(object):
    """Interface to a multibody simulated in PyBullet.
    Attributes:
        uid: The unique id of the body within the physics server.
        name: The name of the body.
        joints: A dict mapping joint names to Joint objects.
        links: A dict mapping link names to Link objects.
    """

    def __init__(self, physics_client, body_uid, scale):
        self.p = physics_client
        self.uid = body_uid
        self.scale = scale
        self.name = self.p.getBodyInfo(self.uid)[1].decode("utf-8")
        self.joints, self.links = {}, {}

        for i in range(self.p.getNumJoints(self.uid)):
            joint_info = self.p.getJointInfo(self.uid, i)
            joint_name = joint_info[1].decode("utf8")
            self.joints[joint_name] = Joint(self.p, self.uid, i)
            link_name = joint_info[12].decode("utf8")
            self.links[link_name] = Link(self.p, self.uid, i)

    @classmethod
    def from_urdf(cls, physics_client, urdf_path, pose, scale, table=False, color_=None):
        body_uid = physics_client.loadURDF(
            str(urdf_path),
            pose.translation,
            pose.rotation.as_quat(),
            globalScaling=scale,
        )
        # print('dynamics')
        # print(physics_client.getDynamicsInfo(body_uid,-1))
        if not table:
            class_rng = np.random.RandomState()
            if color_ is not None:
                color = color_
                color[-1] = 1
            else:
                color = class_rng.uniform(0.6, 1, (4,))
                color[-1] = 1
            physics_client.changeDynamics(body_uid, -1, mass=0.5,
                                          lateralFriction=1.)
            physics_client.changeVisualShape(body_uid, -1, rgbaColor=color)
            return cls(physics_client, body_uid, scale), color
        return cls(physics_client, body_uid, scale), None

    @classmethod
    def from_obj(cls, pb, obj_filepath, pose, scale, color_=None):
        class_rng = np.random.RandomState()
        if color_ is not None:
            color = color_
            color[-1] = 1
        else:
            color = class_rng.uniform(0.6, 1, (4,))
            color[-1] = 1

        obj_edge_max = 0.15 * scale  # the maximum edge size of an obj before scaling
        obj_edge_min = 0.014 * scale  # the minimum edge size of an obj before scaling
        obj_volume_max = 0.0006 * (scale ** 3)  # the maximum volume of an obj before scaling
        obj_scale = scale
        # print(str(obj_filepath))
        while True:
            obj_visual = pb.createVisualShape(pb.GEOM_MESH,
                                              fileName=str(obj_filepath),
                                              rgbaColor=color,
                                              meshScale=[obj_scale, obj_scale, obj_scale])

            obj_collision = pb.createCollisionShape(pb.GEOM_MESH,
                                                    fileName=str(obj_filepath),
                                                    meshScale=[obj_scale, obj_scale, obj_scale])

            object_id = pb.createMultiBody(baseMass=0.15,
                                           baseCollisionShapeIndex=obj_collision,
                                           baseVisualShapeIndex=obj_visual,
                                           basePosition=pose.translation,
                                           baseOrientation=pose.rotation.as_quat())

            aabb = pb.getAABB(object_id)
            aabb = np.asarray(aabb)
            size = aabb[1] - aabb[0]

            if np.partition(size, -2)[-2] > obj_edge_max:
                obj_scale *= 0.8
                pb.removeBody(object_id)
            elif size[0] * size[1] * size[2] > obj_volume_max:
                obj_scale *= 0.85
                pb.removeBody(object_id)
            elif size.min() < obj_edge_min:
                obj_scale /= 0.95
                pb.removeBody(object_id)
            else:
                break

        pb.changeDynamics(object_id, -1, lateralFriction=0.75, spinningFriction=0.001, rollingFriction=0.001,
                          linearDamping=0.0)
        return cls(pb, object_id), color

    def get_pose(self):
        pos, ori = self.p.getBasePositionAndOrientation(self.uid)
        return Transform(Rotation.from_quat(ori), np.asarray(pos))

    def set_pose(self, pose):
        self.p.resetBasePositionAndOrientation(
            self.uid, pose.translation, pose.rotation.as_quat()
        )

    def get_velocity(self):
        linear, angular = self.p.getBaseVelocity(self.uid)
        return linear, angular


class Link(object):
    """Interface to a link simulated in Pybullet.
    Attributes:
        link_index: The index of the joint.
    """

    def __init__(self, physics_client, body_uid, link_index):
        self.p = physics_client
        self.body_uid = body_uid
        self.link_index = link_index

    def get_pose(self):
        link_state = self.p.getLinkState(self.body_uid, self.link_index)
        pos, ori = link_state[0], link_state[1]
        return Transform(Rotation.from_quat(ori), pos)

    def get_position(self):
        link_state = self.p.getLinkState(self.body_uid, self.link_index)
        pos = link_state[0]
        return pos


class Joint(object):
    """Interface to a joint simulated in PyBullet.
    Attributes:
        joint_index: The index of the joint.
        lower_limit: Lower position limit of the joint.
        upper_limit: Upper position limit of the joint.
        effort: The maximum joint effort.
    """

    def __init__(self, physics_client, body_uid, joint_index):
        self.p = physics_client
        self.body_uid = body_uid
        self.joint_index = joint_index

        joint_info = self.p.getJointInfo(body_uid, joint_index)
        self.lower_limit = joint_info[8]
        self.upper_limit = joint_info[9]
        self.effort = joint_info[10]

    def get_position(self):
        joint_state = self.p.getJointState(self.body_uid, self.joint_index)
        return joint_state[0]

    def set_position(self, position, kinematics=False):
        if kinematics:
            self.p.resetJointState(self.body_uid, self.joint_index, position)

        self.p.setJointMotorControl2(
            bodyUniqueId=self.body_uid,
            jointIndex=self.joint_index,
            controlMode=pybullet.POSITION_CONTROL,
            targetPosition=position,
            force=self.effort
        )


class Constraint(object):
    """Interface to a constraint in PyBullet.
    Attributes:
        uid: The unique id of the constraint within the physics server.
    """

    def __init__(
            self,
            physics_client,
            parent,
            parent_link,
            child,
            child_link,
            joint_type,
            joint_axis,
            parent_frame,
            child_frame,
    ):
        """
        Create a new constraint between links of bodies.
        Args:
            parent:
            parent_link: None for the base.
            child: None for a fixed frame in world coordinates.
        """
        self.p = physics_client
        parent_body_uid = parent.uid
        parent_link_index = parent_link.link_index if parent_link else -1
        child_body_uid = child.uid if child else -1
        child_link_index = child_link.link_index if child_link else -1

        self.uid = self.p.createConstraint(
            parentBodyUniqueId=parent_body_uid,
            parentLinkIndex=parent_link_index,
            childBodyUniqueId=child_body_uid,
            childLinkIndex=child_link_index,
            jointType=joint_type,
            jointAxis=joint_axis,
            parentFramePosition=parent_frame.translation,
            parentFrameOrientation=parent_frame.rotation.as_quat(),
            childFramePosition=child_frame.translation,
            childFrameOrientation=child_frame.rotation.as_quat(),
        )

    def change(self, **kwargs):
        self.p.changeConstraint(self.uid, **kwargs)


class Contact(object):
    """Contact point between two multibodies.
    Attributes:
        point: Contact point.
        normal: Normal vector from ... to ...
        depth: Penetration depth
        force: Contact force acting on body ...
    """

    def __init__(self, bodyA, bodyB, point, normal, depth, force):
        self.bodyA = bodyA
        self.bodyB = bodyB
        self.point = point
        self.normal = normal
        self.depth = depth
        self.force = force


# class Camera(object):
#     """Virtual RGB-D camera based on the PyBullet camera interface.
#     Attributes:
#         intrinsic: The camera intrinsic parameters.
#     """
#
#     def __init__(self, physics_client, intrinsic, near, far):
#         self.intrinsic = intrinsic
#         self.near = near
#         self.far = far
#         self.proj_matrix = _build_projection_matrix(intrinsic, near, far)
#         self.p = physics_client
#
#     def render(self, extrinsic, return_seg=False):
#         """Render synthetic RGB and depth images.
#         Args:
#             extrinsic: Extrinsic parameters, T_cam_ref.
#         """
#         # Construct OpenGL compatible view and projection matrices.
#         gl_view_matrix = extrinsic.as_matrix()
#         gl_view_matrix[2, :] *= -1  # flip the Z axis
#         gl_view_matrix = gl_view_matrix.flatten(order="F")
#         gl_proj_matrix = self.proj_matrix.flatten(order="F")
#
#         result = self.p.getCameraImage(
#             width=self.intrinsic.width,
#             height=self.intrinsic.height,
#             viewMatrix=gl_view_matrix,
#             projectionMatrix=gl_proj_matrix,
#             renderer=pybullet.ER_TINY_RENDERER,
#             shadow=0,
#         )
#
#         rgb, z_buffer = result[2][:, :, :3], result[3]
#         depth = (
#                 1.0 * self.far * self.near / (self.far - (self.far - self.near) * z_buffer)
#         )
#         segm = np.uint8(result[4])
#         if return_seg:
#             return rgb, depth, segm
#         return rgb, depth
#
#
# def _build_projection_matrix(intrinsic, near, far):
#     perspective = np.array(
#         [
#             [intrinsic.fx, 0.0, -intrinsic.cx, 0.0],
#             [0.0, intrinsic.fy, -intrinsic.cy, 0.0],
#             [0.0, 0.0, near + far, near * far],
#             [0.0, 0.0, -1.0, 0.0],
#         ]
#     )
#     ortho = _gl_ortho(0.0, intrinsic.width, intrinsic.height, 0.0, near, far)
#     return np.matmul(ortho, perspective)
#
#
# def _gl_ortho(left, right, bottom, top, near, far):
#     ortho = np.diag(
#         [2.0 / (right - left), 2.0 / (top - bottom), -2.0 / (far - near), 1.0]
#     )
#     ortho[0, 3] = -(right + left) / (right - left)
#     ortho[1, 3] = -(top + bottom) / (top - bottom)
#     ortho[2, 3] = -(far + near) / (far - near)
#     return ortho


class Camera2(object):
    """Virtual RGB-D camera based on the PyBullet camera interface.
        Attributes:

    """

    def __init__(self, bullet_client):
        """
        Args:
            bullet_client: pybullet client
        """
        self.p = bullet_client

    def render_camera(self, config):
        """Render RGB-D image with specified camera configuration."""
        # OpenGL camera settings.
        # use time as seed
        self._random = np.random.RandomState(int(time.time()))
        look_dir = np.float32([0, 0, 1]).reshape(3, 1)
        up_dir = np.float32([0, -1, 0]).reshape(3, 1)
        rotation = self.p.getMatrixFromQuaternion(config['rotation'])
        rot_m = np.float32(rotation).reshape(3, 3)
        look_dir = (rot_m @ look_dir).reshape(-1)
        up_dir = (rot_m @ up_dir).reshape(-1)
        lookat = config['position'] + look_dir
        focal_len = config['intrinsics'][0]
        z_near, z_far = config['zrange']
        view_m = self.p.computeViewMatrix(config['position'], lookat, up_dir)
        # def decompose_view_matrix(view_matrix):
        #     """ Decompose the view matrix to camera position and orientation """
        #     view_matrix = np.array(view_matrix).reshape(4, 4).T
        #     rot = view_matrix[:3, :3]
        #     trans = view_matrix[:3, 3]
        #     cam_pos = -np.matmul(rot.T, trans)
        #     cam_rot = rot.T  # Transpose of rotation matrix gives orientation
        #     return cam_pos, cam_rot
        #
        # cam_pos, cam_rot = decompose_view_matrix(view_m)
        #
        # # Draw camera position
        # self.p.addUserDebugText('Camera', cam_pos, textColorRGB=[1, 0, 0])
        #
        # # Axis length for visualization
        # axis_length = 0.3
        #
        # # Draw camera orientation axes
        # # X-axis in red, Y-axis in green, Z-axis in blue
        # self.p.addUserDebugLine(cam_pos, cam_pos + cam_rot[:, 0] * axis_length, [1, 0, 0])
        # self.p.addUserDebugLine(cam_pos, cam_pos + cam_rot[:, 1] * axis_length, [0, 1, 0])
        # self.p.addUserDebugLine(cam_pos, cam_pos + cam_rot[:, 2] * axis_length, [0, 0, 1])

        fov_h = (config['image_size'][0] / 2) / focal_len
        fov_h = 180 * np.arctan(fov_h) * 2 / np.pi

        # Notes: 1) FOV is vertical FOV 2) aspect must be float
        aspect_ratio = config['image_size'][1] / config['image_size'][0]
        proj_m = self.p.computeProjectionMatrixFOV(fov_h, aspect_ratio, z_near, z_far)

        # Render with OpenGL camera settings.
        _, _, color, depth, segm = self.p.getCameraImage(
            height=config['image_size'][0],
            width=config['image_size'][1],
            viewMatrix=view_m,
            projectionMatrix=proj_m,
            shadow=0,
            flags=self.p.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX,
            # renderer=self.p.ER_BULLET_HARDWARE_OPENGL)
            renderer=self.p.ER_TINY_RENDERER)

        # Get color image.
        color_image_size = (config['image_size'][0], config['image_size'][1], 4)
        color = np.array(color, dtype=np.uint8).reshape(color_image_size)
        color = color[:, :, :3]  # remove alpha channel

        # Get depth image.
        depth_image_size = (config['image_size'][0], config['image_size'][1])
        zbuffer = np.array(depth).reshape(depth_image_size)
        depth = (z_far + z_near - (2. * zbuffer - 1.) * (z_far - z_near))
        depth = (2. * z_near * z_far) / depth

        # Get segmentation image.
        segm = np.uint8(segm).reshape(depth_image_size)
        # self.draw_camera_visualization(viewm, self.p)
        
        ## debug ##
        # depth = apply_dex_noise(depth)
        ## debug ##

        return color, depth, segm

import skimage

def apply_dex_noise(img,
                gamma_shape=1000,
                gamma_scale=0.001,
                gp_sigma=0.005,
                gp_scale=4.0,
                gp_rate=0.5):
    gamma_noise = np.random.gamma(gamma_shape, gamma_scale)
    img = img * gamma_noise
    if np.random.rand() < gp_rate:
        h, w = img.shape[:2]
        gp_sample_height = int(h / gp_scale)
        gp_sample_width = int(w / gp_scale)
        gp_num_pix = gp_sample_height * gp_sample_width
        gp_noise = np.random.randn(gp_sample_height, gp_sample_width) * gp_sigma
        gp_noise = skimage.transform.resize(gp_noise,
                                            img.shape[:2],
                                            order=1,
                                            anti_aliasing=False,
                                            mode="constant")
        
        img += gp_noise
    return img