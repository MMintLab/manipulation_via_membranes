#! /usr/bin/env python
import copy
from ctypes import pointer

from trimesh import parent

import rospy
import numpy as np
from itertools import combinations
import threading
from tf import transformations as tr

from geometry_msgs.msg import WrenchStamped
from visualization_msgs.msg import Marker
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2
from control_msgs.msg import FollowJointTrajectoryFeedback


from arc_utilities.listener import Listener
from arc_utilities.tf2wrapper import TF2Wrapper
from bubble_utils.bubble_med.bubble_med import BubbleMed



class ToolPivotingContactPoint(object):
    def __init__(self, goal_angle_difference=np.pi/2, pivoting_angle= np.pi/6, force_threshold=4.0, limit_height=0.1, max_force=12):
        self.force_threshold = force_threshold
        self.max_force = max_force
        try:
            rospy.init_node('jacobian_tool_pivoting')
        except (rospy.exceptions.ROSInitException, rospy.exceptions.ROSException):
            pass
        self.tf2_listener = TF2Wrapper()
        self.model_listener = Listener('contact_model_pc', PointCloud2)
        self.wrench_listener = Listener('/med/wrench', WrenchStamped)
        self.model_pc = self.get_model_pc()
        self.object_length = 0.3 #self.get_object_length()
        self.limit_angle = np.arcsin(limit_height/self.object_length)
        self.pivoting_angle = pivoting_angle
        self.goal_angle_difference = goal_angle_difference
        self.med = self._get_med()
        
    def _get_med(self):
        med = BubbleMed(display_goals=False)
        med.connect()
        return med

    def get_tool_frame(self, ref_frame='med_base'):
        tool_frame_tf = self.tf2_listener.get_transform(parent=ref_frame, child='tool_frame')
        return tool_frame_tf
    
    def get_contact_point(self, ref_frame='med_base'):
        contact_point_tf = self.tf2_listener.get_transform(parent=ref_frame, child='tool_contact_point')
        contact_xyz = contact_point_tf[:3,3]
        return contact_xyz
    
    def get_model_pc(self):
        model_pc = self.model_listener.get(block_until_data=True)
        model_points = np.array(list(pc2.read_points(model_pc)))
        return model_points

    def get_object_length(self):
        max_distance = 0
        for pair in combinations(self.model_pc,2):
            distance = np.linalg.norm(pair[1]-pair[0])
            max_distance = np.maximum(max_distance, distance)
        return max_distance

    def raise_to_free_space(self):
        z_value = self.object_length + 0.1
        motion = self.med.set_xyz_cartesian(z_value=z_value, frame_id='grasp_frame', ref_frame='med_base')     

    def get_angle_difference(self, parent_axis=np.array([0,0,-1]), child_axis=np.array([0,0,-1])):
        parent_axis = parent_axis/np.linalg.norm(parent_axis)
        child_axis = child_axis/np.linalg.norm(child_axis)
        cos_angle = np.dot(parent_axis, child_axis)
        angle = np.arccos(cos_angle)
        if (any(parent_axis < 0) and np.cross(parent_axis, child_axis)[0] < 0) or (any(parent_axis > 0) and np.cross(parent_axis, child_axis)[0] > 0) :
            angle *= -1        
        return angle

    def get_tool_axis(self, tool_axis=np.array([0,0,1]), ref_frame='med_base'):
        tool_frame = self.get_tool_frame(ref_frame=ref_frame)
        tool_axis_rf = tool_frame[:3,:3] @ tool_axis # in the reference frame
        return tool_axis_rf

    def prepare_pivoting(self):
        num_steps = 3
        tool_axis = self.get_tool_axis()
        tool_angle = self.get_angle_difference(child_axis=tool_axis)
        tool_axis_gf = self.get_tool_axis(ref_frame='grasp_frame')
        tool_angle_gf = self.get_angle_difference(child_axis=tool_axis_gf, parent_axis=np.array([0, 0, 1]))
        grasp_frame_tf = self.tf2_listener.get_transform(parent='med_base', child='grasp_frame')
        rotating_point = grasp_frame_tf[:3,3]
        if (self.goal_angle_difference - tool_angle_gf) < 0:
            print('clockwise')
            motion = self.med.rotation_along_axis_point_angle(axis=np.array([1, 0, 0]), angle=tool_angle-self.pivoting_angle, point=rotating_point, num_steps=num_steps)
        elif (self.goal_angle_difference - tool_angle_gf) > 0:
            print('counter-clockwise')
            motion = self.med.rotation_along_axis_point_angle(axis=np.array([1, 0, 0]), angle=-(tool_angle-self.pivoting_angle), point=rotating_point, num_steps=num_steps)

    def pivot(self):
        tool_axis = self.get_tool_axis(tool_axis=np.array([0,0,1]), ref_frame='grasp_frame')
        goal_axis = np.array([0, np.sin(self.goal_angle_difference), np.cos(self.goal_angle_difference)])
        rotation_angle_goal = self.get_angle_difference(parent_axis=goal_axis, child_axis=tool_axis)
        tool_axis_wf = self.get_tool_axis(tool_axis=np.array([0,0,1]), ref_frame='med_base')
        limit_axis = np.array([0, np.cos(self.limit_angle), -np.sin(self.limit_angle)])
        rotation_angle_limit = self.get_angle_difference(parent_axis=limit_axis, child_axis=tool_axis_wf)
        angles = np.array([rotation_angle_goal, rotation_angle_limit])
        rotation_angle = -angles[np.argmin(abs(angles))]
        contact_point = self.get_contact_point()
        motion = self.med.rotation_along_axis_point_angle_fixed_orientation(axis=np.array([1, 0, 0]), angle=rotation_angle, point=contact_point, num_steps=3)
        # self.check_motion_execute(motion)

    def check_goal_position(self, orientation_tol=np.pi/10):
        tool_axis = self.get_tool_axis(tool_axis=np.array([0,0,1]), ref_frame='grasp_frame')
        goal_axis = np.array([0, np.sin(self.goal_angle_difference), np.cos(self.goal_angle_difference)])
        rotation_angle_goal = self.get_angle_difference(parent_axis=goal_axis, child_axis=tool_axis)     
        return abs(rotation_angle_goal) < orientation_tol

    def get_current_angle(self):
        tool_axis = self.get_tool_axis(tool_axis=np.array([0,0,1]), ref_frame='grasp_frame')
        current_angle = self.get_angle_difference(parent_axis=tool_axis, child_axis=np.array([0,0,1]))
        return current_angle

    def move_to_the_center(self, ref_frame='med_base'):
        self.med.set_execute(True)
        grasp_frame_tf = self.tf2_listener.get_transform(parent=ref_frame, child='grasp_frame')
        x_position = grasp_frame_tf[0,3]
        if x_position > 1 or x_position < 0.35:
            self.med.set_xyz_cartesian(x_value=0.65, frame_id='grasp_frame', ref_frame='med_base')
        y_position = grasp_frame_tf[1,3]
        if np.abs(y_position) > 0.3:
            self.med.set_xyz_cartesian(y_value=0, frame_id='grasp_frame', ref_frame='med_base')

    def check_motion_execute(self, motion):
        motion_plan = motion
        if not motion_plan.planning_result.success:
            self.move_to_the_center()
        self.med.set_execute(True)
        self.med.follow_arms_joint_trajectory(motion_plan.planning_result.plan.joint_trajectory, stop_condition=self.force_guard)

    def go_to_center(self):
        self.med.set_xyz_cartesian(x_value=0.65, frame_id='grasp_frame', ref_frame='med_base')
        self.med.set_xyz_cartesian(y_value=0, frame_id='grasp_frame', ref_frame='med_base')

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
    
    def lower_down(self, **kwargs):
        lowering_z = 0.065 # we could go as low as 0.06
        # Update the calibration
        self.calibration_wrench = self.get_wrench()
        self.med.set_xyz_cartesian(z_value=lowering_z, frame_id='grasp_frame', ref_frame='med_base', stop_condition=self._down_stop_signal, **kwargs)

    def get_wrench(self):
        wrench_stamped_wrist = self.wrench_listener.get(block_until_data=True)
        wrench_stamped_world = self.tf2_listener.transform_to_frame(wrench_stamped_wrist, target_frame='world',
                                                               timeout=rospy.Duration(nsecs=int(5e8)))
        return wrench_stamped_world

    def tune_threshold(self):
        import pdb; pdb.set_trace()
        tool_axis_gf = self.get_tool_axis(ref_frame='grasp_frame')
        tool_angle = np.abs(self.get_angle_difference(child_axis=tool_axis_gf, parent_axis=np.array([0, 0, 1])))
        if tool_angle > np.pi/5:
            print('Tilted')
            self.force_threshold = 3.5


    def force_guard(self, feedback: FollowJointTrajectoryFeedback):
        wrench_stamped: WrenchStamped = self.wrench_listener.get()
        force = wrench_stamped.wrench.force
        force_magn = np.linalg.norm(np.array([force.x, force.y, force.z]))
        flag_force = force_magn > self.max_force
        return flag_force
