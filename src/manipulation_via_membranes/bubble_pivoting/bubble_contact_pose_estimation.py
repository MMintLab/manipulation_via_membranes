#!/usr/bin/env python3

import os
import tf
import threading
import trimesh

from arc_utilities.tf2wrapper import TF2Wrapper
from std_msgs.msg import Header
from visualization_msgs.msg import Marker

from mmint_tools.camera_tools.pointcloud_utils import pack_o3d_pcd
from wsg_50_utils.wsg_50_gripper import WSG50Gripper
from manipulation_via_membranes.bubble_pose_estimation.bubble_pc_reconstruction import BubblePCReconsturctorDepth, BubblePCReconsturctorTreeSearch
from mmint_camera_utils.ros_utils.marker_publisher import MarkerPublisher
from mmint_camera_utils.ros_utils.publisher_wrapper import PublisherWrapper

package_path = project_path = os.path.join(os.path.dirname(os.path.abspath(__file__)).split('/bubble_pivoting')[0], 'bubble_pivoting')

class BubbleContactPoseEstimator(object):

    def __init__(self, reconstruction_frame='grasp_frame', imprint_th=0.005, icp_th=0.01, rate=5.0, view=False, 
                verbose=False, object_name='allen', estimation_type='icp2d', plane_frame='med_base', 
                reconstruction='depth', gripper_width=None, marker_name='ycb_spatula_model'):
        self.object_name = object_name
        self.reconstruction_frame = reconstruction_frame
        self.imprint_th = imprint_th
        self.icp_th = icp_th
        self.rate = rate
        self.view = view
        self.verbose = verbose
        self.estimation_type = estimation_type
        self.gripper_width = gripper_width
        self.marker_name = marker_name
        self.model_mesh = trimesh.load(os.path.join(package_path, 'markers/{}.{}'.format(self.marker_name, 'stl')))
        self.model_pc, _ = trimesh.sample.sample_surface(self.model_mesh, count=1000)
        self.plane_frame = plane_frame
        try:
            rospy.init_node('bubble_contact_pose_estimator')
        except (rospy.exceptions.ROSInitException, rospy.exceptions.ROSException):
            pass
        self.tf2_listener = TF2Wrapper()
        self.gripper = WSG50Gripper()
        self.reconstructor = self._get_reconstructor(reconstruction)
        path='package://bubble_pivoting/markers/{}.{}'.format(self.marker_name, 'stl')
        self.contact_marker_publisher = MarkerPublisher(topic_name='estimated_contact_object', marker_type=Marker.MESH_RESOURCE, path=path)
        self.contact_cylinder_publisher = MarkerPublisher(topic_name='estimated_contact_cylinder', marker_type=Marker.CYLINDER)
        self.contact_point_publisher = MarkerPublisher(topic_name='estimated_contact_point', marker_type=Marker.SPHERE)
        self.contact_model_pc_publisher = PublisherWrapper(topic_name='contact_model_pc', msg_type=PointCloud2)
        self.marker_publisher = MarkerPublisher(topic_name='estimated_object', marker_type=Marker.MESH_RESOURCE, path=path, frame_id='grasp_frame')
        self._create_point(publisher=self.contact_point_publisher)
        self._create_cylinder(publisher=self.contact_cylinder_publisher)
        self._create_mesh_marker(publisher=self.marker_publisher)

        self.lock = threading.Lock()

        self.tf_broadcaster = tf.TransformBroadcaster()
        self.tool_estimated_contact_pose = None
        self.tool_estimated_pose = None
        self.contact_point = None
        self.axis = None
        self.transformed_model_pc = None
        self.alive = True
        self.calibrate()
        self.target_pc= self.reconstructor.get_imprint()
        self.target_pcd = pack_o3d_pcd(self.target_pc)
        self.estimate_pose(verbose=self.verbose, contact=True)
        # self.estimate_pose(verbose=self.verbose, contact=False)
        rospy.spin()


    def _get_reconstructor(self, reconstruction_key):
        reconstructors = {
            'depth': BubblePCReconsturctorDepth,
            'tree': BubblePCReconsturctorTreeSearch,
        }
        if reconstruction_key not in reconstructors:
            raise KeyError('No reconstructor found for key {} -- Possible keys: {}'.format(reconstruction_key, reconstructors.keys()))
        Reconstructor = reconstructors[reconstruction_key]
        reconstructor = Reconstructor(reconstruction_frame=self.reconstruction_frame, threshold=self.imprint_th, object_name=self.object_name, estimation_type=self.estimation_type,
                              view=self.view, verbose=self.verbose, percentile=0.005)
        return reconstructor

    def calibrate(self):
        info_msg = 'Press enter to calibrate --'
        if self.gripper_width is not None:
            info_msg += '\n\t>>> We will open the gripper!\t'
        _ = input(info_msg)
        if self.gripper_width is not None:
            # Open the gripper
            self.gripper.open_gripper()
        self.reconstructor.reference()
        info_msg = 'Calibration done! {}\nPress enter to continue :)'
        additional_msg = ''
        if self.gripper_width is not None:
            additional_msg = '\n We will close the gripper to a width {}mm'.format(self.gripper_width)
        _ = input(info_msg.format(additional_msg))
        if self.gripper_width is not None:
            # move gripper to gripper_width
            self.gripper.move(self.gripper_width, speed=50.0)

    def _evaluate_loss_function(self, model_points):
        target_points = self.target_pcd.points
        # Estimate Correspondences
        tree = KDTree(model_points)
        corr_distances, cp_indxs = tree.query(target_points)
        # Apply correspondences
        model_points_corr = model_points[cp_indxs] # corresponed points in model to the scene points
        # ICP loss function
        cost = np.mean(np.linalg.norm(target_points-model_points_corr, axis=1)**2)
        return cost

    # Given an initial guess for the pose coming from icp, moves the tool along handle axis until contact 
    # It also applies orientation perturbation and evaluates icp loss function to find the contact pose that better matches imprint
    def _get_contact_pose_plane_frame(self, pose_pf, tool_axis, plane_normal_axis):
        best_tf = np.empty((4,4))
        best_tf[3,:] = np.array([0,0,0,1])
        marker_axis_pf = pose_pf[:3, :3] @ tool_axis  # in the plane frame

        transformed_object = self.model_pc @ pose_pf[:3,:3].T + pose_pf[:3,3]
        h = np.min(transformed_object[:,2])
        cos_angle = np.dot(marker_axis_pf, plane_normal_axis)
        dist = -h/cos_angle
        best_tf[:3,:3] = pose_pf[:3,:3]
        best_tf[:3,3] = pose_pf[:3,3] + dist * marker_axis_pf
        object_adjusted = transformed_object + dist * marker_axis_pf
        contact_point_idx = np.argmin(object_adjusted[:,2])
        contact_point = object_adjusted[contact_point_idx]
        axis = marker_axis_pf
        self.transformed_model_pc = object_adjusted
        min_loss = self._evaluate_loss_function(model_points=object_adjusted)

        ft_rotation_axis = np.cross(marker_axis_pf, plane_normal_axis)
        ft_rotation_axis /= np.linalg.norm(ft_rotation_axis)

        for i in range(10):
            ft_angle = np.pi / 500 * (i-5)
            ft_rotation = tr.rotation_matrix(ft_angle, ft_rotation_axis)[:3,:3]
            ft_transformed_object = (self.model_pc @ pose_pf[:3,:3].T @ ft_rotation.T) + pose_pf[:3,3]
            ft_marker_axis_pf = ft_rotation @ marker_axis_pf 
            ft_cos_angle = np.dot(ft_marker_axis_pf, plane_normal_axis)
            ft_h = np.min(ft_transformed_object[:,2])
            ft_dist = -ft_h / ft_cos_angle
            ft_object_adjusted = ft_transformed_object + ft_dist * ft_marker_axis_pf
            loss = self._evaluate_loss_function(model_points=ft_object_adjusted)
            if loss <= min_loss:
                min_loss = loss
                best_tf[:3,:3] = ft_rotation @ pose_pf[:3,:3]
                best_tf[:3,3] = pose_pf[:3,3] + ft_dist * ft_marker_axis_pf
                contact_point_idx = np.argmin(ft_object_adjusted[:,2])
                contact_point = ft_object_adjusted[contact_point_idx]
                axis = ft_marker_axis_pf
                self.transformed_model_pc = ft_object_adjusted
        pc_header = Header()
        pc_header.frame_id = 'world'
        transformed_model_pc_i = pc2.create_cloud_xyz32(pc_header, self.transformed_model_pc)
        self.contact_model_pc_publisher.data = transformed_model_pc_i
        return best_tf, contact_point, axis


    def _estimate_contact_pose(self, verbose=False, return_original=False):
        # Transformation given by icp (or chosen method)
        init_tr = self.reconstructor.estimate_pose(threshold=self.icp_th, view=self.view, verbose=verbose)
        marker_axis = np.array([0,0,1])
        plane_normal_axis = np.array([0,0,1])  # axis normal to the plane in plane_frame

        # get tf between plane_frame and tool_frame
        marker_parent_frame_pf = self.tf2_listener.get_transform(parent=self.plane_frame, child=self.reconstruction_frame)
        
        tool_pose_pf = marker_parent_frame_pf @ init_tr # in plane frame

        estimated_tr, contact_point, axis = self._get_contact_pose_plane_frame(pose_pf=tool_pose_pf, tool_axis=marker_axis, plane_normal_axis=plane_normal_axis)
        estimated_quat = tr.quaternion_from_matrix(estimated_tr)
        estimated_trans = estimated_tr[:3,3]
        self.tf2_listener.send_transform(translation=contact_point, quaternion=[0,0,0,1], parent=self.plane_frame, child='tool_contact_point', is_static=False)
        self.tf2_listener.send_transform(translation=estimated_trans, quaternion=estimated_quat, parent='world', child='tool_frame', is_static=False)
        if return_original:
            return estimated_tr, contact_point, axis, init_tr
        else:
            return estimated_tr, contact_point, axis

    def estimate_pose(self, contact=True, verbose=False):
        while not rospy.is_shutdown():
            try:
                if contact:
                    icp_tr, contact_point, axis, original_tr = self._estimate_contact_pose(verbose=verbose, return_original=True)
                    with self.lock:
                        # update the tool_estimated_contact_pose
                        t = icp_tr[:3, 3]
                        q = tr.quaternion_from_matrix(icp_tr)
                        self.tool_estimated_contact_pose = np.concatenate([t, q])
                        self.contact_marker_publisher.pose = self.tool_estimated_contact_pose
                        self.contact_cylinder_publisher.pose = self.tool_estimated_contact_pose
                        self.contact_point_publisher.pose = np.concatenate([contact_point, np.zeros(4)], axis=0)
                        self.axis = axis
                # else:
                if True:
                    # icp_tr = self.reconstructor.estimate_pose(threshold=self.icp_th, view=self.view, verbose=verbose)
                    icp_tr = original_tr
                    with self.lock:
                        # update the tool_estimated_pose
                        t = icp_tr[:3, 3]
                        q = tr.quaternion_from_matrix(icp_tr)
                        self.tool_estimated_pose = np.concatenate([t, q])
                        self.marker_publisher.pose = self.tool_estimated_pose

            except rospy.ROSInterruptException:
                self.finish()
                break
           

    def _create_cylinder(self, publisher=None):
        scale_x = 2*self.reconstructor.radius
        scale_y = 2*self.reconstructor.radius
        scale_z = 2*self.reconstructor.height 
        publisher.scale = ((scale_x, scale_y, scale_z))
        # set color
        color_r = 158/255.
        color_g = 232/255.
        color_b = 217/255.
        color_a = 1.0
        publisher.marker_color = ((color_r, color_g, color_b, color_a))

    def _create_mesh_marker(self, publisher=None):
        # scale_x = 2*self.reconstructor.radius
        # scale_y = 2*self.reconstructor.radius
        # scale_z = 2*self.reconstructor.height
        # publisher.scale = ((scale_x, scale_y, scale_z))
        # set color
        color_r = 158/255.
        color_g = 232/255.
        color_b = 217/255.
        color_a = 1.0
        publisher.marker_color = ((color_r, color_g, color_b, color_a))

    def _create_point(self, publisher=None):
        scale_x = 0.01
        scale_y = 0.01
        scale_z = 0.01 
        publisher.scale = ((scale_x, scale_y, scale_z))
        # set color
        color_r = 255/255.
        color_g = 0
        color_b = 0
        color_a = 1.0
        publisher.marker_color = ((color_r, color_g, color_b, color_a))



if __name__ == '__main__':

    # Continuous  pose estimator:
    # view = False
    view = True
    # imprint_th = 0.0048 # for pen with gw 15
    # imprint_th = 0.0048 # for allen with gw 12
    imprint_th = 0.0053 # for marker with gw 20
    # imprint_th = 0.006 # for spatula with gripper width of 15mm
    icp_th = 1. # consider all points
    icp_th = 0.005 # for allen key

    bpe = BubbleContactPoseEstimator(view=view, imprint_th=imprint_th, icp_th=icp_th, rate=5., verbose=view)