#! /usr/bin/env python
import copy

import rospy
import numpy as np
import threading
from tf import transformations as tr

from geometry_msgs.msg import WrenchStamped
from visualization_msgs.msg import Marker

from arc_utilities.listener import Listener
from arc_utilities.tf2wrapper import TF2Wrapper
from victor_hardware_interface_msgs.msg import ControlMode
from mmint_camera_utils.recorders.wrench_recorder import WrenchRecorder
from bubble_utils.bubble_med.bubble_med import BubbleMed
from manipulation_via_membranes.bubble_contact_point_estimation.contact_point_marker_publisher import ContactPointMarkerPublisher


class ToolContactPointEstimator(object):

    def __init__(self, force_threshold=4.0):
        self.grasp_pose_joints = [0.7613740469101997, 1.1146166859754167, -1.6834551714751782, -1.6882417308401203,
                             0.47044861033517205, 0.8857417788890095, 0.8497585444122142]
        self.force_threshold = force_threshold
        rospy.init_node('tool_contact_point_estimator', anonymous=True)
        self.med = self._get_med()
        self.wrench_listener = Listener('/med/wrench', WrenchStamped)
        self.wrench_recorder = WrenchRecorder('/med/wrench', wait_for_data=True)
        self.tf2_listener = TF2Wrapper()
        self.calibration_wrench = None
        self.contact_point_marker_publisher = ContactPointMarkerPublisher()

    def get_contact_point(self, ref_frame='med_base'):
        contact_point_tf = self.tf2_listener.get_transform(parent=ref_frame, child='tool_contact_point')
        contact_xyz = contact_point_tf[:3,3]
        return contact_xyz

    def set_grasp_pose(self):
        self.med.plan_to_joint_config(self.med.arm_group, self.grasp_pose_joints)

    def _get_med(self):
        med = BubbleMed(display_goals=False)
        med.connect()
        return med

    def get_wrench(self):
        wrench_stamped_wrist = self.wrench_listener.get(block_until_data=True)
        wrench_stamped_world = self.tf2_listener.transform_to_frame(wrench_stamped_wrist, target_frame='world',
                                                               timeout=rospy.Duration(nsecs=int(5e8)))
        return wrench_stamped_world

    def _down_stop_signal(self, feedback):
        wrench_stamped_world = self.get_wrench()
        measured_fz = wrench_stamped_world.wrench.force.z
        calibration_fz = self.calibration_wrench.wrench.force.z
        fz =  measured_fz - calibration_fz
        flag_force = np.abs(fz) >= np.abs(self.force_threshold)
        if flag_force:
            print('force z: {} (measured: {}, calibration: {}) --- flag: {}'.format(fz, measured_fz, calibration_fz, flag_force))
            # activate contact detector pub
            self.contact_point_marker_publisher.show = True
        return flag_force

    def _up_signal(self, feedback):
        wrench_stamped_world = self.get_wrench()
        measured_fz = wrench_stamped_world.wrench.force.z
        calibration_fz = self.calibration_wrench.wrench.force.z
        fz = measured_fz - calibration_fz
        flag_no_force = np.abs(fz) < np.abs(self.force_threshold)
        if flag_no_force:
            self.contact_point_marker_publisher.show = False # deactivate the force flag
        out_flag = False
        return out_flag

    def lower_down(self):
        lowering_z = 0.065 # we could go as low as 0.06
        # Update the calibration
        self.calibration_wrench = self.get_wrench()
        self.med.set_xyz_cartesian(z_value=lowering_z, frame_id='grasp_frame', ref_frame='med_base', stop_condition=self._down_stop_signal)

    def raise_up(self):
        z_value = 0.35
        self.med.set_xyz_cartesian(z_value=z_value, frame_id='grasp_frame', ref_frame='med_base',
                                   stop_condition=self._up_signal)

    def rotate_along_axis_angle(self, axis, angle, frame='grasp_frame', **kwargs):
        # get current frame pose:
        current_frame_pose = self.tf2_listener.get_transform('world', frame)
        current_frame_pos = current_frame_pose[:3, 3].copy()
        rotation_matrix = tr.quaternion_matrix(tr.quaternion_about_axis(angle, axis))
        new_pose_matrix = rotation_matrix @ current_frame_pose
        new_position = current_frame_pos
        new_quat = tr.quaternion_from_matrix(new_pose_matrix)
        target_pose = np.concatenate([new_position, new_quat])
        self.med.plan_to_pose(self.med.arm_group, frame, target_pose=list(target_pose), frame_id='world',**kwargs)

    def get_tool_angle_axis(self, ref_frame='med_base'):
        tool_frame_rf = self.tf2_listener.get_transform(parent=ref_frame, child='tool_frame')
        z_axis = np.array([0, 0, 1])
        tool_axis_tf = np.array([0,0,-1]) # Tool axis on the tool_frame
        tool_axis_tf_h = np.append(tool_axis_tf,0)
        tool_axis_rf_h = tool_frame_rf @ tool_axis_tf_h
        tool_axis_rf = tool_axis_rf_h[:-1] # remove homogeneous vector coordinates
        tool_angle = np.arccos(np.dot(tool_axis_rf, z_axis))
        _rot_axis = np.cross(tool_axis_rf, z_axis)
        rot_axis = _rot_axis / np.linalg.norm(_rot_axis)
        # TODO: account for tool_axis parallel to z_axis
        return tool_angle, rot_axis

    def estimate_motion(self):
        self.med.set_control_mode(ControlMode.JOINT_POSITION, vel=0.1)
        self.set_grasp_pose()
        rospy.sleep(2.0)
        self.lower_down()
        rospy.sleep(3.0)
        # move on the plane
        move_dist = 0.2
        self.med.cartesian_delta_motion([0,move_dist,0])
        rospy.sleep(2.0)
        self.raise_up()
        rospy.sleep(2.0)
        # rotate
        self.rotate_along_axis_angle(axis=np.array([1,0,0]), angle=np.pi/8, frame='grasp_frame')
        rospy.sleep(2.0)
        # lower down again
        self.lower_down()
        rospy.sleep(4.0)
        self.raise_up()
        rospy.sleep(2.0)
        # rotate
        self.rotate_along_axis_angle(axis=np.array([1, 0, 0]), angle=-np.pi / 8, frame='grasp_frame')
        rospy.sleep(2.0)
        # lower down again
        self.lower_down()
        rospy.sleep(4.0)
        self.raise_up()
        self.med.rotation_along_axis_point_angle(axis=np.array([1, 0,0]), angle=np.pi/4, frame_id='grasp_frame')
        rospy.sleep(2.0)
        self.lower_down()
        rospy.sleep(2.0)
        self.med.cartesian_delta_motion([0, -move_dist, 0])
        rospy.sleep(2.0)
        contact_point = self.get_contact_point()
        self.med.rotation_along_axis_point_angle(axis=np.array([1, 0, 0]), angle=np.pi/8, point=contact_point)
        rospy.sleep(2.0)
        self.raise_up()
        rospy.sleep(2.0)
        self.med.cartesian_delta_motion([0, move_dist, 0])
        rospy.sleep(2.0)
        self.lower_down()
        rospy.sleep(2.0)
        contact_point = self.get_contact_point()
        self.med.rotation_along_axis_point_angle(axis=np.array([1, 0, 0]), angle=-np.pi / 5, point=contact_point)
        rospy.sleep(2.0)
        self.raise_up()

    def rotation_test_motion(self):
        num_steps = 20
        self.med.set_control_mode(ControlMode.JOINT_POSITION, vel=0.1)
        self.set_grasp_pose()
        self.med.cartesian_delta_motion([0, 0.2, 0])
        rospy.sleep(2.0)
        self.lower_down()
        rospy.sleep(3.0)
        contact_point = self.get_contact_point()
        self.med.rotation_along_axis_point_angle(axis=np.array([1, 0, 0]), angle=np.pi / 4, point=contact_point, num_steps=num_steps, pos_tol=0.001, ori_tol=0.005)
        rospy.sleep(3.0)
        contact_point = self.get_contact_point()
        self.med.rotation_along_axis_point_angle(axis=np.array([1, 0, 0]), angle=-np.pi / 4, point=contact_point, num_steps=num_steps, pos_tol=0.001, ori_tol=0.005)
        rospy.sleep(3.0)
        contact_point = self.get_contact_point()
        self.med.rotation_along_axis_point_angle(axis=np.array([1, 0, 0]), angle=np.pi / 4, point=contact_point,
                                                 num_steps=num_steps, pos_tol=0.001, ori_tol=0.005)
        rospy.sleep(3.0)
        contact_point = self.get_contact_point()
        self.med.rotation_along_axis_point_angle(axis=np.array([0, 1, 0]), angle=np.pi / 15, point=contact_point,
                                                 num_steps=num_steps, pos_tol=0.001, ori_tol=0.005)
        rospy.sleep(3.0)
        contact_point = self.get_contact_point()
        self.med.rotation_along_axis_point_angle(axis=np.array([0, 1, 0]), angle=-2*np.pi / 15, point=contact_point,
                                                 num_steps=num_steps, pos_tol=0.001, ori_tol=0.005)
        rospy.sleep(3.0)
        contact_point = self.get_contact_point()
        self.med.rotation_along_axis_point_angle(axis=np.array([0, 1, 0]), angle=np.pi / 15, point=contact_point,
                                                 num_steps=num_steps, pos_tol=0.001, ori_tol=0.005)
        rospy.sleep(3.0)
        contact_point = self.get_contact_point()
        self.med.rotation_along_axis_point_angle(axis=np.array([1, 0, 0]), angle=-np.pi / 5, point=contact_point,
                                                 num_steps=num_steps, pos_tol=0.001, ori_tol=0.005)
        rospy.sleep(2.0)
        self.raise_up()

    def pivoting_test(self):
        num_steps = 20
        self.med.set_control_mode(ControlMode.JOINT_POSITION, vel=0.1)
        rospy.sleep(2.0)
        contact_point = self.get_contact_point()
        print('pivoting along: ', contact_point)
        self.med.rotation_along_axis_point_angle_fixed_orientation(axis=np.array([1, 0, 0]), angle=np.pi / 3, point=contact_point, num_steps=num_steps)
        rospy.sleep(3.0)


    def compensate_tool(self):
        num_steps = 20
        self.med.set_control_mode(ControlMode.JOINT_POSITION, vel=0.1)
        self.set_grasp_pose()
        self.med.cartesian_delta_motion([0, 0.2, 0])
        rospy.sleep(2.0)
        self.lower_down()
        rospy.sleep(3.0)
        for i in range(20):
            contact_point = self.get_contact_point()
            angle, axis = self.get_tool_angle_axis()

            self.med.rotation_along_axis_point_angle(axis=axis, angle=angle, point=contact_point,
                                                     num_steps=num_steps, pos_tol=0.001, ori_tol=0.005)
            rospy.sleep(3.0)
        self.raise_up()
        rospy.sleep(2.0)

    def close(self):
        self.contact_point_marker_publisher.finish()


def contact_point_estimation_with_actions():
    force_threshold = 5.0
    tcpe = ToolContactPointEstimator(force_threshold=force_threshold)
    # tcpe.estimate_motion()
    # tcpe.rotation_test_motion()
    tcpe.compensate_tool()
    tcpe.close()

def test():
    force_threshold = 5.0
    tcpe = ToolContactPointEstimator(force_threshold=force_threshold)
    tcpe.pivoting_test()
    tcpe.close()



if __name__ == '__main__':
    # contact_point_estimation_with_actions()
    test()