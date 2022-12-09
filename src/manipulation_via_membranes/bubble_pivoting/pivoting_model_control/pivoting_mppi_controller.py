from matplotlib.pyplot import axis
import numpy as np
import os
import sys
import torch
import copy

from torch import true_divide
import rospy
import tf2_ros as tf
import tf.transformations as tr
from geometry_msgs.msg import TransformStamped
from pytorch_mppi import mppi
import matplotlib.pyplot as plt


from manipulation_via_membranes.bubble_learning.aux.img_trs.block_downsampling_tr import BlockDownSamplingTr
from manipulation_via_membranes.bubble_model_control.controllers.bubble_model_mppi_controler import BubbleModelMPPIController
from bubble_pivoting.data_collection.bubble_pivoting_env import BubblePivotingEnv
from arc_utilities.tf2wrapper import TF2Wrapper
from manipulation_via_membranes.bubble_learning.aux.orientation_trs import QuaternionToAxis
from manipulation_via_membranes.bubble_model_control.aux.format_observation import format_observation_sample
from manipulation_via_membranes.bubble_learning.aux.load_model import load_model_version
from bubble_drawing.bubble_model_control.aux.bubble_model_control_utils import batched_tensor_sample, get_transformation_matrix, tr_frame, convert_all_tfs_to_tensors, batched_matrix_to_euler_corrected
from manipulation_via_membranes.bubble_pivoting.pivoting_model_control.cost_functions import CostFunction
from manipulation_via_membranes.bubble_pivoting.pivoting_model_control.aux.pivoting_geometry import check_goal_position_from_estimated_pose, get_angle_difference, check_goal_position, get_tool_axis, get_tool_angle_gf
from mmint_camera_utils.ros_utils.marker_publisher import MarkerPublisher


class BubblePivotingPolicy(BubbleModelMPPIController):
    def __init__(self, model, data_name, version, policy, rand_goal, object_name, goal_angle, env, 
                 object_pose_estimator, check_object_pose_estimator, tool, *args, orientation_tol=.07,
                factor_x = 7, factor_y=7, reduction='mean', keys_to_tr=['init_imprint'], debug=False, **kwargs):
        self.data_name = data_name
        self.object_name = object_name
        self.policy = policy
        self.rand_goal = rand_goal
        self.tf2_listener = TF2Wrapper()
        self.block_downsample_tr = BlockDownSamplingTr(factor_x, factor_y, keys_to_tr)
        self.goal_marker_publisher = MarkerPublisher('tool_goal', marker_color=(0, 1.0, 0, 1.0), frame_id='grasp_frame')
        self.orientation_tol = orientation_tol
        self.goal_angle = goal_angle
        self.cost_function = CostFunction(policy=self.policy)
        self.cost_function.env = env
        self.quat_to_axis = QuaternionToAxis()
        self.cost_function.goal_angle = goal_angle
        self.env = env
        self.tool = tool
        self.debug = debug
        self.object_pose_estimator = object_pose_estimator
        self.check_object_pose_estimator = check_object_pose_estimator
        self.action_space = self.env._get_action_space()
        self.is_action_valid = self.env.is_action_valid
        if policy == 'learned_mppi':
            self.version = version
            self.loaded_model = load_model_version(model, data_name, self.version)
            self.loaded_model.eval()
            self.device = self.loaded_model.device       
            super().__init__(model=self.loaded_model, env=env, object_pose_estimator=object_pose_estimator, cost_function=self.cost_function, *args, debug=self.debug,**kwargs)
        self._init_ros_node()

    def _get_action_container(self):
        action_container, _ = self.env.get_action(self.action_space, self.is_action_valid)
        return action_container

    def check_goal_position_from_sample(self, goal_angle, orientation_tol):
        self.env.gripper.move(20.0)
        obs_sample_raw = self.env.get_observation()
        obs_sample = self.get_downsampled_obs(obs_sample_raw)
        obs_sample = self.format_sample_for_pose_estimation(obs_sample)
        if not 'final_imprint' in obs_sample.keys():
            obs_sample['final_imprint'] = obs_sample['init_imprint']
        estimated_pose = self.check_object_pose_estimator.estimate_pose(obs_sample)
        return check_goal_position_from_estimated_pose(goal_angle, orientation_tol, estimated_pose, obs_sample)

    def collect_data(self, controller_method, scene_name='pivoting_evaluation'):
        self.policy = controller_method
        if self.rand_goal:
            self.goal_angle = np.random.rand() * 10 * np.pi/6 - 5 * np.pi/6
            self.cost_function.goal_angle = self.goal_angle
        print('GOAL ANGLE: ', self.goal_angle)
        _, init_angle_diff = self.check_goal_position_from_sample(self.goal_angle, self.orientation_tol)
        init_angle = self.goal_angle - init_angle_diff
        done, angle_diff, tool_detected, steps = self.execute_policy()
        _, online_angle_diff = check_goal_position(self.goal_angle, self.orientation_tol)
        model_name = None if controller_method != 'learned_mppi' else self.loaded_model.get_name()
        version = None if controller_method != 'learned_mppi' else self.version
        if model_name == 'bubble_linear_dynamics_model':
            controller_method = 'linear_learned_mppi'
        elif model_name == 'object_pose_dynamics_model':
            controller_method = 'pose_learned_mppi'
        data_params = {
            'SceneName': scene_name,
            'ControllerMethod': controller_method,
            'Model': model_name,
            'Version': version,
            'Tool': self.tool, 
            'InitAngle': init_angle,
            'GoalAngle': self.goal_angle,
            'Achieved': done,
            'AngleDiff': angle_diff,
            'ToolDetected': tool_detected,
            'NSteps': steps,
            'OnlineAngleDiff': online_angle_diff
        }
        return data_params
            
    def _init_ros_node(self):
        try:
            rospy.init_node('pivoting_model_mmpi')
        except (rospy.exceptions.ROSInitException, rospy.exceptions.ROSException):
            pass


    def display_goal_angle(self, goal_angle):
        marker_length = 0.25
        marker_diameter = 0.015
        marker_pos = np.array([0.,0.,0.])
        marker_quat = tr.quaternion_about_axis(goal_angle, axis=np.array([1,0,0]))
        marker_pos[2] = np.cos(goal_angle) * marker_length*0.5
        marker_pos[1] = -np.sin(goal_angle) * marker_length * 0.5
        marker_pose = np.concatenate([marker_pos, marker_quat], axis=-1)

        self.goal_marker_publisher.marker_type = self.goal_marker_publisher.Marker.CYLINDER
        self.goal_marker_publisher.scale = (marker_diameter, marker_diameter, marker_length)

        self.goal_marker_publisher.data = marker_pose

    def execute_policy(self):
        if self.env.num_steps > 0:
            self.env.reset()
        print('Goal angle:', self.goal_angle) # TODO: Remove this line
        self.display_goal_angle(self.goal_angle)

        done = False
        stuck = False
        angle_diff = None
        tool = None
        steps = 0
        while not done and not stuck:
            pre_action = {}
            pre_action['roll'] = self.get_roll()           
            init_feedback = self.env.do_pre_action_init(pre_action)
            lower_feedback = self.env.do_pre_action_lower()
            if not lower_feedback['planning_success']:
                print('Lower plan failed, please erase data point.')
                stuck = True
                break
            init_pose = self.env.med.get_current_pose('grasp_frame',ref_frame='med_base')
            done, angle_diff = self.check_goal_position_from_sample(self.goal_angle, self.orientation_tol)
            angle_diff_prev = angle_diff
            tool_angle = self.goal_angle-angle_diff
            print('Goal angle: ', self.goal_angle)
            print('Init tool angle: ', tool_angle)
            if done:
                print('Goal angle achieved, going to grasp pose')
                self.env._do_pre_action_prepare(open_width=None)
                self.env.med.set_robot_conf('grasp_conf')
                break
            self.env._do_pre_action_prepare(open_width=None)
            obs_sample_raw = self.env.get_observation()
            self.cost_function.contact=True
            steps = 0
            for i in range(10):
                # tool = self.env.tool_detected_listener.get(block_until_data=True).data
                # if not tool:
                #     print('No tool detected, going to grasp pose')
                #     stuck = True
                #     break
                if self.policy == 'jacobian':
                    motion = self.pivot()
                    if motion == "Failed":
                        stuck = True
                        break
                    self.env.num_steps += 1 # fake the step count
                else:
                    self._update_action_space()
                    self.env._do_pre_action_prepare(open_width=None)
                    obs_sample = self.get_downsampled_obs(obs_sample_raw)
                    if self.policy == 'random':
                        action, _ = self.env.get_action(self.action_space, self.is_action_valid)
                    else:
                        action = self.control(obs_sample)
                    print('Action: ', action)
                    obs_sample_raw, reward, _, info = self.env.step(action)
                    if self.debug:
                        formatted_sample = self.get_downsampled_obs(obs_sample_raw)
                        self.visualize_prediction(formatted_sample)
                done, angle_diff = self.check_goal_position_from_sample(self.goal_angle, self.orientation_tol)
                steps += 1
                tool_angle = self.goal_angle-angle_diff
                print('Real tool angle gf after action: ', tool_angle)
                if done:
                    print('Goal angle achieved, going to grasp pose')
                    self.env._do_pre_action_prepare(open_width=None)
                    self.env.med.gripper.move(15.0, 0)
                    # self.env.med.set_robot_conf('grasp_conf')
                    break
                if np.sign(angle_diff_prev*angle_diff) < 0:
                    print('Overshot goal angle, going to grasp pose.')
                    stuck = True
                    break
                angle_diff_prev = angle_diff
                progress = None
                # if i == 0:
                #     progress = input('Is the pivoting making proggress? (y or n) ')
                if progress == 'n':
                    stuck = True
                    break
            if i == 9:
                stuck = True
        tool = self.env.tool_detected_listener.get(block_until_data=True).data
        # self.env._plan_to_pose(init_pose, supervision=False, avoid_readjusting=True)
        return done, angle_diff, tool, steps
        
    def get_roll(self):
        done, angle_diff = self.check_goal_position_from_sample(self.goal_angle, self.orientation_tol)
        direction = np.sign(angle_diff)
        tool_angle_gf = self.goal_angle-angle_diff
        roll = np.pi + direction * np.pi/4 - tool_angle_gf
        return roll        
        
    def pivot(self):
        # grasp_width= np.random.rand() * 35 + 5.0
        grasp_width = 25.0
        self.env.gripper.move(grasp_width)
        done, rotation_angle_goal = self.check_goal_position_from_sample(self.goal_angle, self.orientation_tol)
        tool_axis_wf = get_tool_axis(tool_axis=np.array([0,0,1]), ref_frame='med_base')
        limit_height = 0.1
        object_length = 0.3
        limit_angle = np.arcsin(limit_height/object_length)
        limit_axis = np.array([0, np.cos(limit_angle), -np.sin(limit_angle)])
        rotation_angle_limit = get_angle_difference(parent_axis=limit_axis, child_axis=tool_axis_wf)
        angles = np.array([rotation_angle_goal, rotation_angle_limit.item()])
        rotation_angle = -angles[np.argmin(abs(angles))]
        contact_point = self.get_contact_point()
        grasp_point = self.tf2_listener.get_transform(parent='med_base', child='grasp_frame')[:3,3]
        motion = self.env.med.rotation_along_axis_point_angle(axis=np.array([1, 0, 0]), angle=rotation_angle_goal, point=grasp_point, num_steps=3)
        return motion
        # motion = self.env.med.rotation_along_axis_point_angle_fixed_orientation(axis=np.array([1, 0, 0]), angle=rotation_angle, point=contact_point, num_steps=3)
        # self.check_motion_execute(motion)
        
    def get_contact_point(self, ref_frame='med_base'):
        contact_point_tf = self.tf2_listener.get_transform(parent=ref_frame, child='tool_contact_point')
        contact_xyz = contact_point_tf[:3,3]
        return contact_xyz

    def _update_action_space(self):
        self.action_space = self.env._get_action_space()
        if self.policy == 'learned_mppi':
            self.u_min, self.u_max = self._get_action_space_limits()

    def get_downsampled_obs(self, obs=None):
        if obs is None:
            obs = self.env.get_observation()
        format_obs = format_observation_sample(obs)
        downsampled_obs = self.block_downsample_tr(format_obs)
        return downsampled_obs


