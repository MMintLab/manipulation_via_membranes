import numpy as np
from collections import defaultdict
import pandas as pd
import rospy
import time
from tqdm import tqdm
import abc

from tf import transformations as tr
from arc_utilities.listener import Listener
from sensor_msgs.msg import JointState
from bubble_utils.bubble_data_collection.bubble_data_collection_base import BubbleDataCollectionBase
from mmint_camera_utils.camera_utils.camera_parsers import RealSenseCameraParser
from mmint_camera_utils.camera_utils.camera_parsers import PicoFlexxCameraParser
from victor_hardware_interface_msgs.msg import ControlMode


class GelsightComparisonDataCollectionBase(BubbleDataCollectionBase):
    """
    collect data using the robot in impedance mode and makeing contact into a surface while grasping a marker
    """
    def __init__(self, *args, sensor_name='bubbles', manual_motion=False, force_threshold=10.0, delta_move=0.005, max_num_steps=100, **kwargs):
        self.manual_motion = manual_motion
        self.force_threshold = force_threshold
        self.delta_move = delta_move
        self.max_num_steps = max_num_steps
        self.sensor_name = sensor_name
        self.num_start_steps = 10
        self.num_calibration_steps = 15
        self.grasp_widths = {'bubbles': 20, 'gelsight': 25}
        if sensor_name == 'bubbles':
            right = True
            left = True
        else:
            right = False
            left = False
        self.reference_fc = None
        super().__init__(*args, right=right, left=left, **kwargs)
        self.init_pose = self._get_init_pose()
        self.realsense_parser = RealSenseCameraParser(camera_indx=1, scene_name=self.scene_name, save_path=self.save_path, verbose=False)

        self.joint_listener = Listener('/med/joint_states', JointState, wait_for_data=True)
        self.joint_sequence = None
        self.initialized = False

    @abc.abstractmethod
    def _get_init_pose(self):
        # Return the initial pose
        pass

    @abc.abstractmethod
    def _set_cartesian_impedance(self):
        pass

    @abc.abstractmethod
    def _motion_step(self, indx):
        # perform a single step of the motion
        pass

    @abc.abstractmethod
    def _check_wrench(self, wrench):
        # check wheter we meet the wrench requirements
        pass

    def _record_reference_state(self):
        self.reference_fc = self.get_new_filecode()
        self._record(self.reference_fc)

    # Add recording of joints
    def _get_legend_column_names(self):
        """
        Return a list containing the column names of the datalegend
        Returns:
        """
        column_names = ['Time', 'Scene', 'SensorName', 'Sequence', 'SequenceIndx', 'FileCode', 'ReferenceFC', 'JointState']
        return column_names

    def _get_legend_lines(self, data_params):
        """
        Return a list containing the values to log inot the data legend for the data sample with file code filecode
        Args:
            data_params: <dict> containg parameters of the collected data
        Returns:
        """
        column_names = self._get_legend_column_names()
        lines = np.array([data_params[cn] for cn in column_names], dtype=object).T
        return lines

    def _get_sequence_indx(self):
        seq_indx = 0
        dl = pd.read_csv(self.datalegend_path)
        sequence_indxs = dl['Sequence']
        if len(sequence_indxs) > 0:
            seq_indx = np.max(sequence_indxs) + 1
        return seq_indx

    def _get_wrench(self, *args, **kwargs):
        wrench_stamped = self.med.get_wrench(*args, **kwargs)
        wrench = [
            wrench_stamped.wrench.force.x,
            wrench_stamped.wrench.force.y,
            wrench_stamped.wrench.force.z,
            wrench_stamped.wrench.torque.x,
            wrench_stamped.wrench.torque.y,
            wrench_stamped.wrench.torque.z,
        ]
        return np.asarray(wrench)

    def _record(self, fc=None):
        super()._record(fc=fc)
        self.realsense_parser.record(fc=fc)

    def _init_data_collection_seq(self):
        print('Recording started. Moving the robot to the grasp pose')
        self.med.set_robot_conf('grasp_conf')
        _ = input('Press enter to open the gripper')
        self.med.open_gripper()
        self._record_reference_state()
        _ = input('Press to close the gripper')
        self.med.gripper.move(width=self.grasp_widths[self.sensor_name], speed=50.0)
        print('Strating to collect data')
        # Go to the initial position:
        self.med.plan_to_pose(self.med.arm_group, 'grasp_frame', list(self.init_pose), frame_id='med_base')
        # Set impedance mode
        self._set_cartesian_impedance()

    def _finish_data_collection_seq(self):
        self.med.set_joint_position_control()
        self.med.plan_to_pose(self.med.arm_group, 'grasp_frame', list(self.init_pose), frame_id='med_base')

    def _calibration_steps(self, num_start_steps, num_calibration_steps):
        calibration_wrenches = []
        for indx in range(num_start_steps):
            self._motion_step(indx)
            rospy.sleep(1.0)
        for indx in range(num_calibration_steps):
            self._motion_step(num_start_steps + indx)
            rospy.sleep(3.0)
            calibration_wrench_i = self._get_wrench()
            calibration_wrenches.append(calibration_wrench_i)

        # obtain the calibration wrench
        offset_wrench = np.mean(np.stack(calibration_wrenches, axis=0), axis=0)
        return offset_wrench

    def _collect_data_sample(self, params=None):
        """
        Collect and save data to the designed path in self.data_path
        Args:
            params:
        Returns: <dict> containing the parameters of the collected sample
        """
        data_params = defaultdict(list)
        joint_states = []
        seq_indx = self._get_sequence_indx()

        self._init_data_collection_seq()

        init_wrench = self._get_wrench()

        # calibration steps:
        offset_wrench = self._calibration_steps(self.num_start_steps, self.num_calibration_steps)

        print('Offset wrench: ', offset_wrench)

        for indx in tqdm(range(self.max_num_steps)):
            self._motion_step(indx+self.num_start_steps+self.num_calibration_steps)
            rospy.sleep(3.0)
            # RECORD:
            fc_i = self.get_new_filecode()
            self._record(fc=fc_i)
            joints_i = self.joint_listener.get(block_until_data=True)
            joint_states.append(joints_i)
            data_params['Time'].append(time.time())
            data_params['Sequence'].append(seq_indx)
            data_params['SequenceIndx'].append(indx)
            data_params['Scene'].append(self.scene_name)
            data_params['SensorName'].append(self.sensor_name)
            data_params['FileCode'].append(fc_i)
            data_params['ReferenceFC'].append(self.reference_fc)
            data_params['JointState'].append(joints_i.position)
            current_wrench = self._get_wrench()
            compensated_wrench = current_wrench - offset_wrench
            if self.manual_motion:
                print('Measured wrench: {} -- (raw: {})'.format(compensated_wrench, current_wrench))
                user_input = input('Continue? ')
                if user_input == 'y':
                    pass
                elif user_input == 'n':
                    break
                else:
                    print('command not acceptable')
                    pass
            else:
                is_wrench_limit_reached = self._check_wrench(compensated_wrench)
                if is_wrench_limit_reached:
                    break
                else:
                    pass

        self._finish_data_collection_seq()
        return data_params


class GelsightComparisonTopDownDataCollection(GelsightComparisonDataCollectionBase):
    """
    collect data using the robot in impedance mode and makeing contact into a surface while grasping a marker
    """
    def _get_init_pose(self):
        init_pose = np.array([0.6, 0., 0.25, -np.cos(np.pi / 4), np.cos(np.pi / 4), 0, 0])
        return init_pose

    def _set_cartesian_impedance(self):
        self.med.set_cartesian_impedance(velocity=1, x_stiffness=5000, y_stiffness=5000, z_stiffnes=100)

    def _motion_step(self, indx):
        pose_i = self.init_pose.copy()
        pose_i[2] = pose_i[2] - (indx + 1) * self.delta_move
        self.med.cartesian_impedance_raw_motion(pos=pose_i[:3], quat=pose_i[3:], frame_id='grasp_frame',
                                                ref_frame='med_base')

    def _check_wrench(self, wrench):
        wrench_value = np.abs(wrench[2])
        print('\n', wrench_value, '\n')
        is_wrench_limit_reached = wrench_value >= self.force_threshold
        return is_wrench_limit_reached


class GelsightComparisonSidewaysDataCollection(GelsightComparisonTopDownDataCollection):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    # def _init_data_collection_seq(self):
    #     super()._init_data_collection_seq()
    #     pre_contact_pose = self._get_init_pose() - np.array([0, 0.2, 0, 0, 0, 0, 0])
    #     self.med.cartesian_impedance_raw_motion(pos=pre_contact_pose[:3], quat=pre_contact_pose[3:], frame_id='grasp_frame',
    #                                             ref_frame='med_base')
    def _init_data_collection_seq(self):
        print('Recording started. Moving the robot to the grasp pose')
        self.med.plan_to_pose(self.med.arm_group, 'grasp_frame', list(self.init_pose), frame_id='med_base')
        _ = input('Press enter to open the gripper')
        self.med.open_gripper()
        self._record_reference_state()
        _ = input('Press to close the gripper')
        self.med.gripper.move(width=self.grasp_widths[self.sensor_name], speed=50.0)
        print('Strating to collect data')
        # Set impedance mode
        self._set_cartesian_impedance()

    def _get_init_pose(self):
        if self.sensor_name == 'bubbles':
            init_pose = np.array([0.55, -0.23, 0.15, -np.cos(np.pi / 4), np.cos(np.pi / 4), 0, 0])
        elif self.sensor_name == 'gelsight':
            init_pose = np.array([0.55, -0.23, 0.1, -np.cos(np.pi / 4), np.cos(np.pi / 4), 0, 0])
        else:
            raise NotImplementedError
        return init_pose

    def _set_cartesian_impedance(self):
        self.med.set_cartesian_impedance(velocity=1, x_stiffness=5000, y_stiffness=500, z_stiffnes=5000)

    def _motion_step(self, indx):
        pose_i = self.init_pose.copy()
        pose_i[1] = pose_i[1] + (indx + 1) * self.delta_move
        self.med.cartesian_impedance_raw_motion(pos=pose_i[:3], quat=pose_i[3:], frame_id='grasp_frame',
                                                ref_frame='med_base')

    def _check_wrench(self, wrench):
        wrench_value = np.abs(wrench[1])
        print('\n', wrench_value, '\n')
        is_wrench_limit_reached = wrench_value >= self.force_threshold
        return is_wrench_limit_reached


class GelsightComparisonRotationDataCollection(GelsightComparisonTopDownDataCollection):
    def __init__(self, *args, **kwargs):
        self.contact_init_pose = None
        super().__init__(*args, **kwargs)
        self.num_start_steps = 8 # here, this is the number of steps until making contact
        self.num_calibration_steps = 5 # here we first calibrate and then do the start steps

    def _init_data_collection_seq(self):
        print('Recording started. Moving the robot to the grasp pose')
        self.med.plan_to_pose(self.med.arm_group, 'grasp_frame', list(self.init_pose), frame_id='med_base')
        _ = input('Press enter to open the gripper')
        self.med.open_gripper()
        self._record_reference_state()
        _ = input('Press to close the gripper')
        self.med.gripper.move(width=self.grasp_widths[self.sensor_name], speed=50.0)
        print('Strating to collect data')
        # Set impedance mode
        self._set_cartesian_impedance()

    def _get_init_pose(self):
        init_pose = np.array([0.65, 0.1, 0.15, -np.cos(np.pi / 4), np.cos(np.pi / 4), 0, 0])
        return init_pose

    def _get_contact_init_pose(self):
        contact_init_pose = np.array([0.58, 0.1, 0.15, -np.cos(np.pi / 4), np.cos(np.pi / 4), 0, 0])
        return contact_init_pose

    def _set_cartesian_impedance(self):
        # Cartesian impedance is no good for rotations, therefore we try with joint impedance:
        self.med.set_control_mode(ControlMode.JOINT_IMPEDANCE, vel=0.1)
        # self.med.set_cartesian_impedance(velocity=1, x_stiffness=500, y_stiffness=5000, z_stiffnes=5000)

    def _motion_step(self, indx):
        print('motion_step')
        pose_i = self.contact_init_pose.copy()
        # pose_i = self._get_contact_init_pose()
        quat_i = pose_i[3:]
        step_angle = self.delta_move*(indx+1-self.num_start_steps-self.num_calibration_steps)
        delta_quat_i = tr.quaternion_about_axis(angle=step_angle, axis=np.array([0,1,0]))
        pose_i[3:] = tr.quaternion_multiply(delta_quat_i, quat_i)
        # self.med.cartesian_impedance_raw_motion(pos=pose_i[:3], quat=quat_i, frame_id='grasp_frame', ref_frame='med_base')
        # import pdb; pdb.set_trace()
        # self.med.get_current_pose()
        # self.med.set_execute(False)
        plan_and_execution_result = self.med.plan_to_pose(self.med.arm_group, 'grasp_frame', target_pose=list(pose_i), frame_id='med_base')
        # import pdb; pdb.set_trace()
        # self.med.rotation_along_axis_point_angle(axis=np.array([0, 1, 0]), angle=step_angle)
        # self.med.set_execute(True)
        # import pdb; pdb.set_trace()
        # x = 0
        pass

    def _init_motion_step(self, indx):
        pose_i = self.init_pose.copy()
        pose_i[0] = pose_i[0] - (indx+1)*0.005
        # self.med.cartesian_impedance_raw_motion(pos=pose_i[:3], quat=pose_i[3:], frame_id='grasp_frame', ref_frame='med_base')
        self.med.plan_to_pose(self.med.arm_group, 'grasp_frame', target_pose=list(pose_i), frame_id='med_base')

        return pose_i

    def _calibration_steps(self, num_start_steps, num_calibration_steps):
        calibration_wrenches = []
        pose_i = self.init_pose
        for i in range(num_calibration_steps):
            pose_i = self._init_motion_step(i)
            rospy.sleep(3.0)
            calibration_wrench_i = self._get_wrench()  # wrench in the grasp frame!
            calibration_wrenches.append(calibration_wrench_i)
        offset_wrench = np.mean(np.stack(calibration_wrenches, axis=0), axis=0)
        for i in range(num_start_steps):
            pose_i = self._init_motion_step(i+num_calibration_steps)
            rospy.sleep(3.0)
            wrench_i = self._get_wrench() - offset_wrench # wrench in the grasp frame!
            print('\n {}'.format(wrench_i))
            if np.abs(wrench_i[1]) > 1.5:
                break
        self.contact_init_pose = pose_i.copy()
        return offset_wrench

    def _check_wrench(self, wrench):
        wrench_value = np.abs(wrench[1])
        print('\n', wrench_value, '\n')
        is_wrench_limit_reached = wrench_value >= self.force_threshold
        return is_wrench_limit_reached

    def _get_wrench(self, *args, **kwargs):
        wrench = super()._get_wrench(frame_id='grasp_frame')
        return wrench



def simple_3_image_recording():
    from bubble_utils.bubble_med.bubble_med import BubbleMed

    rospy.init_node('simple_3_image_recording')
    scene_name = 'camera_recording'
    data_path = '/home/mmint/Desktop/bubble_vs_gelsight_top_down_calibration_data'
    save_path = data_path
    camera_name_right = 'pico_flexx_right'
    camera_parser_right = PicoFlexxCameraParser(camera_name=camera_name_right,
                                                         scene_name=scene_name, save_path=save_path,
                                                         verbose=False)
    camera_name_left = 'pico_flexx_left'
    camera_parser_left = PicoFlexxCameraParser(camera_name=camera_name_left,
                                                        scene_name=scene_name, save_path=save_path,
                                                        verbose=False)
    med = BubbleMed()
    # reference:
    _ = input('press enter to open gripper')
    med.gripper.open_gripper()
    camera_parser_left.record()
    camera_parser_right.record()
    _ = input('press enter to record')
    med.gripper.move(20, 50)
    # no deformation:
    camera_parser_left.record()
    camera_parser_right.record()
    for i in range(5):
        # deformation:
        _ = input('press enter to record')
        camera_parser_left.record()
        camera_parser_right.record()


def top_down_data_collection(sensor_name):
    dc = GelsightComparisonTopDownDataCollection(
        data_path='/home/mmint/Desktop/bubble_vs_gelsight_top_down_calibration_data',
        # dc = GelsightComparisonSidewaysDataCollection(
        #     data_path='/home/mmint/Desktop/bubble_vs_gelsight_sideways_calibration_data',
        scene_name=sensor_name,
        sensor_name=sensor_name,
        manual_motion=False,
    )
    dc.collect_data(num_data=5)


def sideways_data_collection(sensor_name):
    dc = GelsightComparisonSidewaysDataCollection(
        force_threshold=5.0,
        data_path='/home/mmint/Desktop/bubble_vs_gelsight_sideways_calibration_data',
        scene_name=sensor_name,
        sensor_name=sensor_name,
        manual_motion=False,
    )
    dc.collect_data(num_data=5)


def rotation_data_collection(sensor_name):
    dc = GelsightComparisonRotationDataCollection(
        force_threshold=5.0,
        data_path='/home/mmint/Desktop/bubble_vs_gelsight_rotation_calibration_data',
        scene_name=sensor_name,
        sensor_name=sensor_name,
        delta_move=np.deg2rad(3.0),
        manual_motion=False,
        max_num_steps=10,
    )
    dc.collect_data(num_data=5)


# DEBUG:
if __name__ == '__main__':

    # sensor_name = 'bubbles'
    sensor_name = 'gelsight'
    # top_down_data_collection(sensor_name)
    sideways_data_collection(sensor_name)
    # rotation_data_collection(sensor_name)
    # simple_3_image_recording()




