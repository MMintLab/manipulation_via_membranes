from abc import abstractmethod

import numpy as np
import rospy
from collections import OrderedDict
import gym
import copy
import time
import tf.transformations as tr
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2
from std_msgs.msg import Bool
from victor_hardware_interface_msgs.msg import ControlMode

from control_msgs.msg import FollowJointTrajectoryFeedback
from arc_utilities.tf2wrapper import TF2Wrapper
from arc_utilities.listener import Listener
from mmint_camera_utils.recording_utils.data_recording_wrappers import DataSelfSavedWrapper
from manipulation_via_membranes.aux.action_spaces import RollSpace
from bubble_utils.bubble_envs.bubble_base_env import BubbleBaseEnv
from bubble_utils.bubble_med.bubble_med import BubbleMed
from wsg_50_utils.wsg_50_gripper import WSG50Gripper
from manipulation_via_membranes.bubble_pivoting.aux.load_confs import load_object_models
from manipulation_via_membranes.bubble_pivoting.aux.load_confs import load_object_params



class BubblePivotingBaseEnv(BubbleBaseEnv):

    def __init__(self, *args, impedance_mode=False, reactive=False, force_threshold=10.,
                 max_force=15, grasp_width_limits=(10.0, 30.0), 
                 delta_y_limits=(-.04, .04), delta_roll_limits=(-np.pi/6, np.pi/6),
                 tool, roll_limits=(np.pi/2,3*np.pi/2), **kwargs):
        self.gripper = WSG50Gripper()
        self.gripper_width = 20.
        self.max_force_felt = 0.
        self.max_force = max_force
        self.impedance_mode = impedance_mode
        self.reactive = reactive
        self.force_threshold = force_threshold
        self.grasp_width_limits = grasp_width_limits
        self.delta_y_limits = delta_y_limits
        self.delta_roll_limits = delta_roll_limits
        self.tool = tool
        self.tool_length_dict = self._get_tool_length_dict()
        self.tool_length = self._get_tool_length(tool)        
        self.roll_limits = roll_limits
        self.bubble_ref_obs = None
        self.init_action = None
        self.init_action_space = self._get_init_action_space()
        super().__init__(*args, **kwargs)
        self.tf2_listener = TF2Wrapper()
        self.model_listener = Listener('contact_model_pc', PointCloud2)
        self.tool_detected_listener = Listener('tool_detected', Bool)
        self.reset_pose = np.array([.5, -0.3, .3, np.pi, 0, np.pi])
        self.object_model = self._get_object_model(self.tool)
        self.reset()

    @classmethod
    def get_name(cls):
        return 'bubble_pivoting_base_env'

    def reset(self):
        self.med.gripper.move(15.0)
        self._plan_to_pose(self.reset_pose, supervision=False, avoid_readjusting=True, stop_condition=self.force_guard)
        _ = input('Press enter to open the gripper and calibrate the bubbles')
        self.med.open_gripper()
        # Calibrate
        self.bubble_ref_obs = self._get_bubble_observation()
        _ = input('Press enter to close the gripper')
        self.gripper.move(self.gripper_width, speed=50.0)
        rospy.sleep(2.0)
        print('Calibration is done')
        super().reset()

    @abstractmethod
    def initialize(self):
        pass

    @abstractmethod
    def _get_init_action_space(self):
        pass

    @abstractmethod
    def _get_action_space(self):
        pass

    def _get_med(self):
        med = BubbleMed(display_goals=False)
        med.connect()
        return med

    def _add_bubble_reference_to_observation(self, obs):
        keys_to_include = ['color_img', 'depth_img', 'point_cloud']
        for k, v in self.bubble_ref_obs.items():
            for kti in keys_to_include:
                if kti in k:
                    # do not add the saving method:
                    if isinstance(v, DataSelfSavedWrapper):
                        obs['{}_reference'.format(k)] = v.data # unwrap the data so it will not be saved with the observation. This avoid overriting reference and current state. Reference will be saved apart.
                    else:
                        obs['{}_reference'.format(k)] = v
        return obs

    def _get_tool_length_dict(self):
        length_dict = {}
        object_params = load_object_params()['objects']
        for k, v in object_params.items():
            params = v['params']
            length_i = 0
            for p, s in params.items():
                if 'length' in p:
                    length_i += float(s)
            length_dict[k] = length_i
            if v['function'] == 'generate_spoon':
                length_dict[k] = (float(params['handle_length']) + float(params['triangle_length']) +
                                  float(params['circle_radius']) - float(params['intersection_length']))
        return length_dict
        
    def _get_tool_length(self, tool):
        tool_length = self.tool_length_dict[tool]
        print('Tool length: ', tool_length)
        return tool_length

    def _get_object_model(self, object_code):
        object_models = load_object_models() # TODO: Consdier doing this more efficient to avoid having to load every time
        object_model_pcd = object_models[object_code]
        object_model = np.asarray(object_model_pcd.points)
        return object_model

    def _get_observation(self):
        obs = {}
        bubble_obs = self._get_bubble_observation()
        obs.update(bubble_obs)
        obs['wrench'] = self._get_wrench()
        obs['tfs'] = self._get_tfs()
        obs['tool'] = self.tool_detected_listener.get(block_until_data=True).data
        obs['max_force_felt'] = self.max_force_felt
        obs['tool_code'] = self.tool
        obs['object_model'] = self.object_model
        # add the reference state
        obs = self._add_bubble_reference_to_observation(obs)
        return obs

    def _get_observation_space(self):
        return None

    def _get_tf_frames(self):
        frames = super()._get_tf_frames()
        frames += ['tool_frame']
        return frames

    def _get_model_pc(self):
        model_pc = self.model_listener.get(block_until_data=True)
        model_points = np.array(list(pc2.read_points(model_pc)))
        return model_points

    def get_wrench(self):
        wrench_stamped_wrist = self.med_wrench_recorder.get_wrench()
        wrench_stamped_world = self.tf2_listener.transform_to_frame(wrench_stamped_wrist, target_frame='world',
                                                               timeout=rospy.Duration(nsecs=int(5e8)))
        return wrench_stamped_world

    def _plan_to_pose(self, pose, frame_id='med_base', supervision=False, stop_condition = None, avoid_readjusting=False, constrained_normal=None):
        plan_success = False
        execution_success = False
        plan_found = False
        while (not rospy.is_shutdown()) and not plan_found:
            if supervision:
                self.med.set_execute(False)
            if constrained_normal is not None:
                plan_result = self.med.plan_to_pose_constrained_plane(self.med.arm_group, 'grasp_frame', target_pose=list(pose), 
                                                plane_normal=constrained_normal, frame_id=frame_id, stop_condition=stop_condition)   
            else: 
                plan_result = self.med.plan_to_pose(self.med.arm_group, 'grasp_frame', target_pose=list(pose),
                                                    frame_id=frame_id, stop_condition=stop_condition)
            plan_success = plan_result.success
            execution_success = plan_result.execution_result.success
            if not plan_success:
                print('@' * 20 + '    Plan Failed    ' + '@' * 20)
            if supervision:
                for i in range(10):
                    k = input('Execute plan (y: yes, r: replan, f: finish): ')
                    if k == 'y':
                        self.med.set_execute(True)
                        execution_result = self.med.follow_arms_joint_trajectory(
                            plan_result.planning_result.plan.joint_trajectory)
                        execution_success = execution_result.success
                        plan_found = True
                        break
                    elif k == 'r':
                        break
                    elif k == 'f':
                        return plan_success, execution_success
                    else:
                        pass
            else:
                plan_found = True

        if not execution_success:
            print('-' * 20 + '    Execution Failed or contact made   ' + '-' * 20)
            
        return plan_success, execution_success


class BubblePivotingEnv(BubblePivotingBaseEnv):

    def initialize(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.action_space = self._get_action_space()


    @classmethod
    def get_name(cls):
        return 'bubble_pivoting_env'

    def _get_action_space(self):
        action_space_dict = OrderedDict()
        action_space_dict['grasp_width'] = gym.spaces.Box(low=self.grasp_width_limits[0], high=self.grasp_width_limits[1], shape=())
        current_pose = self.med.get_current_pose()
        delta_z_limits = (-(current_pose[2]-0.08), .02)
        action_space_dict['delta_y'] = gym.spaces.Box(low=self.delta_y_limits[0], high=self.delta_y_limits[1], shape=())
        action_space_dict['delta_z'] = gym.spaces.Box(low=delta_z_limits[0], high=delta_z_limits[1], shape=())
        action_space_dict['delta_roll'] = gym.spaces.Box(low=self.delta_roll_limits[0], high=self.delta_roll_limits[1], shape=())
        action_space = gym.spaces.Dict(action_space_dict)
        return action_space

    def final_pose_from_action(self, action, euler=False):
        current_pose = self.med.get_current_pose()
        delta_y = action['delta_y']
        delta_z = action['delta_z']
        movement_wf = delta_y * np.array([0,1,0]) + delta_z * np.array([0,0,1])
        delta_roll = action['delta_roll']
        orientation = np.array(tr.euler_from_quaternion(current_pose[3:], 'sxyz'))
        orientation[0] += delta_roll
        orientation[1] = 0
        orientation[2] = np.pi
        if not euler:
            orientation = tr.quaternion_from_euler(orientation[0], orientation[1], orientation[2], 'sxyz')
        final_pose_wf = np.concatenate([current_pose[:3]+movement_wf, orientation])
        return final_pose_wf

    def  init_pose_from_action(self, pre_action, euler=False):
        init_position = np.array([.55, 0, .35])
        roll = pre_action['roll']
        init_orientation = np.array([roll, 0, np.pi])
        if not euler:
            init_orientation = tr.quaternion_from_euler(init_orientation[0], init_orientation[1], init_orientation[2], 'sxyz')
        init_pose_wf = np.concatenate([init_position, init_orientation])
        return init_pose_wf

    def get_action(self, action_space, is_action_valid):
        valid_action = False
        action = None
        for i in range(400):
            action = action_space.sample()
            valid_action = is_action_valid(action)
            if valid_action:
                break
        return action, valid_action

    def _get_init_action_space(self, direction=None):
        action_space_dict = OrderedDict()
        action_space_dict['roll'] = RollSpace()    
        action_space = gym.spaces.Dict(action_space_dict)
        return action_space

    def is_action_valid(self, action):
        final_pose_wf = self.final_pose_from_action(action, euler=True)
        final_euler = final_pose_wf[3:] % (2*np.pi)
        final_quat = tr.quaternion_from_euler(final_euler[0], final_euler[1], final_euler[2], 'sxyz')
        movement_wf = action['delta_y'] * np.array([0,1,0]) + action['delta_z'] * np.array([0,0,1])
        current_pose = self.med.get_current_pose()

        if final_euler[0] < np.pi/2 or final_euler[0] > 3*np.pi/2:
            return False

        # Checking whether it is potentially a pivoting motion (it pushes against the table)
        initial_pc = self._get_model_pc()
        #TODO: substract euler angles
        movement_transformation = tr.quaternion_matrix(tr.quaternion_multiply(final_quat, tr.quaternion_inverse(current_pose[3:])))
        tool_frame_tf = self.tf2_listener.get_transform(parent='med_base', child='tool_frame')
        grasp_frame_tf = self.tf2_listener.get_transform(parent='med_base', child='grasp_frame')
        grasping_point_tf = grasp_frame_tf[:3,3]
        initial_pc_origin = initial_pc - grasping_point_tf
        transformed_pc_origin = initial_pc_origin @ movement_transformation[:3,:3].T
        self.transformed_model_pc = transformed_pc_origin + grasping_point_tf + movement_wf

        # Checking that the movement is not parallel to the tool axis
        tool_frame_tf = self.tf2_listener.get_transform(parent='med_base', child='tool_frame')
        tool_axis_wf = tool_frame_tf[:3,:3] @ np.array([0,0,1])
        dot_prod = np.abs(np.dot(movement_wf, tool_axis_wf))/np.linalg.norm(movement_wf)

        valid_action = min(self.transformed_model_pc[:,2]) < -0.03 #and dot_prod < 0.8
        return valid_action

    def is_pre_action_valid(self, pre_action, point_down=False):
        valid_pre_action = pre_action['roll'] >= np.pi/2 and pre_action['roll'] <= 3*np.pi/2
        return valid_pre_action

    def down_stop_signal(self, feedback):
        wrench_stamped_world = self.get_wrench()
        measured_fz = wrench_stamped_world.wrench.force.z
        calibration_fz = self.calibration_wrench.wrench.force.z
        fz = measured_fz - calibration_fz
        flag_force = fz >= np.abs(self.force_threshold)

        tool_frame_wf = self.tf2_listener.get_transform(parent='med_base', child='tool_frame')
        tool_axis_wf = tool_frame_wf[:3,:3] @ np.array([0,0,1])
        cos_angle = np.dot(tool_axis_wf, np.array([0,0,-1]))
        tool_height = (self.tool_length-0.08)* cos_angle
        current_height = self.tf2_listener.get_transform(parent='med_base', child='grasp_frame')[2,3]
        flag_dist = current_height < tool_height
        # if flag_force:
        #     print('force z: {} (measured: {}, calibration: {}) --- flag: {}'.format(fz, measured_fz, calibration_fz, flag_force))
        print('force z: {} (measured: {}, calibration: {}) --- flag: {}'.format(fz, measured_fz, calibration_fz, flag_force))
        if flag_dist:
            print('Current height: {}, Tool height: {}'.format(current_height, tool_height))
        stop_signal = (flag_force and current_height < 0.5) or flag_dist
        # stop_signal = (flag_force and current_height < 0.5)
        return stop_signal

    def force_guard(self, feedback: FollowJointTrajectoryFeedback):
        wrench_stamped = self.get_wrench()
        force = wrench_stamped.wrench.force
        force_magn = np.linalg.norm(np.array([force.x, force.y, force.z]))
        flag_force = force_magn > self.max_force
        self.max_force_felt = np.maximum(self.max_force_felt, force_magn)
        current_pose = self.med.get_current_pose()
        if flag_force:
            print('force: {} --- flag: {}'.format(force_magn, flag_force))
        return flag_force and current_pose[2] < 0.8

    def do_pre_action_init(self, pre_action):
        # Firm grasp to hold tool
        # action_init_width = self.gripper_width
        action_init_width = 15.0
        self.gripper.move(action_init_width)
        initial_pose = self.init_pose_from_action(pre_action)
        self.med.set_control_mode(control_mode=ControlMode.JOINT_POSITION, vel=0.15)
        print('Going to initial position')
        planning_result = self._plan_to_pose(initial_pose, supervision=False, avoid_readjusting=True, stop_condition=self.force_guard)
        action_feedback = {
            'planning_success': planning_result[0],
            'execution_success': planning_result[1],
        }
        return action_feedback

    def do_pre_action_lower(self):
        lowering_z = 0.065 # we could go as low as 0.06
        # Update the calibration
        time.sleep(1.0)
        self.calibration_wrench = self.get_wrench()
        init_pose = self.med.get_current_pose()
        desired_pose = copy.deepcopy(init_pose)
        desired_pose[2] = lowering_z
        # self.med.set_control_mode(control_mode=ControlMode.JOINT_POSITION, vel=0.1)
        self.med.set_control_mode(control_mode=ControlMode.JOINT_POSITION, vel=0.05)
        print('Lowering down')

        planning_result = self._plan_to_pose(desired_pose, frame_id='med_base', 
                                             supervision=False, stop_condition=self.down_stop_signal, constrained_normal=np.array([1,0,0]))
        action_feedback = {
            'planning_success': planning_result[0],
            'execution_success': planning_result[1] or self.med.get_current_pose()[2] < 0.3,
        }
        return action_feedback

    def _do_pre_action_prepare(self, open_width=32.5):
        # Open gripper so the tool falls into contact
        if open_width is not None:
            self.gripper.move(open_width)
            rospy.sleep(2)        
        # Firm grasp to record imprint. 
        self.gripper.move(self.gripper_width)       
        rospy.sleep(0.5)

    def _do_action(self, action):
        grasping_width_i = action['grasp_width']
        final_pose_i = self.final_pose_from_action(action)
        # Custom grasp to perform pivoting
        self.med.set_control_mode(control_mode=ControlMode.JOINT_POSITION, vel=0.1)
        self.gripper.move(grasping_width_i)
        self.max_force_felt = 0
        print('Pivoting')
        planning_result = self._plan_to_pose(final_pose_i, frame_id='med_base', 
                                             supervision=False, stop_condition=self.force_guard, constrained_normal=np.array([1,0,0]))
        self.med.set_control_mode(control_mode=ControlMode.JOINT_POSITION, vel=0.15)
        action_feedback = {
            'planning_success': planning_result[0],
            'execution_success': planning_result[1],
        }
        self._do_pre_action_prepare(open_width=None)
        return action_feedback

    def step(self, a):
        info = {}
        action_feedback = self._do_action(a)
        info.update(action_feedback)
        observation = self.get_observation()
        info['max_force_felt'] = observation['max_force_felt']
        info['tool'] = observation['tool']
        done = self._is_done(observation, a)
        reward = self._get_reward(a, observation)
        self.num_steps += 1
        return observation, reward, done, info

    def no_tool_reset(self):
        self._plan_to_pose(self.reset_pose, supervision=False, avoid_readjusting=True, stop_condition=self.force_guard)
        info_msg = '\n\t>>> We will open the gripper!\t'
        _ = input(info_msg)
        self.med.set_grasping_force(25.0)
        self.gripper.open_gripper()
        additional_msg = '\n We will close the gripper to a width {}mm'.format(self.gripper_width)
        _ = input(additional_msg)
        self.gripper.move(self.gripper_width, speed=50.0)
        rospy.sleep(0.5)

