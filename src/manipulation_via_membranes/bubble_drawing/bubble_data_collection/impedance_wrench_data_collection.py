import numpy as np
import rospy
import time
import pandas as pd
from collections import defaultdict
from tqdm import tqdm
import copy

from arc_utilities.listener import Listener
from sensor_msgs.msg import JointState

from bubble_utils.bubble_data_collection.med_data_collection_base import MedDataCollectionBase


class PoseBuffer(object):

    def __init__(self, num_x, num_y, limits, h_value, quat=None):
        self.num_x = num_x
        self.num_y = num_y
        self.limits = limits
        self.h_value = h_value
        self.quat = quat
        self.indx = 0
        self.indexing_order = None
        self.poses = None
        self.reset()

    def reset(self):
        self.indx = 0
        self.indexing_order = self._get_indexing_order()
        self.poses = self._get_poses()

    def _get_poses(self):
        # return a matrix of xy position an with z values as h_value and orientation as self.quat
        # out shape (num_points_x, num_points_y,
        x_values = np.linspace(self.limits[0][0], self.limits[1][0], num=self.num_x)
        y_values = np.linspace(self.limits[0][1], self.limits[1][1], num=self.num_y)
        xy_matrix = np.stack(np.meshgrid(x_values, y_values), axis=-1)
        zquat = np.insert(self.quat, 0, self.h_value)
        z_quat_matrix = np.expand_dims(zquat, axis=[0,1]).repeat(self.num_x, axis=0).repeat(self.num_y, axis=1)
        poses_matrix = np.concatenate([xy_matrix, z_quat_matrix], axis=-1)
        return poses_matrix

    def _get_indexing_order(self):
        # x-sweeping
        indexing_order = []
        x_indxs = np.arange(self.num_x)
        y_indxs = np.arange(self.num_y)
        for y_i in y_indxs:
            if y_i % 2 == 0:
                line_xs = x_indxs
            else:
                line_xs = np.flip(x_indxs)
            line_indxs = np.stack([line_xs, y_i*np.ones_like(line_xs)], axis=-1)
            indexing_order.append(line_indxs)
        indexing_order = np.concatenate(indexing_order, axis=0)
        indexing_order = indexing_order.astype(np.int32)
        return indexing_order

    @property
    def remaining_count(self):
        count = self.__len__() - self.indx % self.__len__()
        return count

    def __len__(self):
        return len(self.indexing_order)

    def __iter__(self):
        self.indx = 0
        return self

    def __next__(self):
        out = self.get_current_pose()
        self.indx += 1
        return out

    def get_current_pose(self):
        wrapped_indx = self._wrap_index(self.indx)
        current_pose = self.__getitem__(wrapped_indx)
        return current_pose

    def get_current_indx(self):
        wrapped_indx = self._wrap_index(self.indx)
        current_indx = self.indexing_order[wrapped_indx]
        return current_indx

    def _wrap_index(self, indx):
        run_length = self.__len__() *2 - 2 # do not repeat end points
        run_indx = indx // run_length
        w_indx = indx % run_length
        direction = w_indx // self.__len__()
        if direction == 0:
            wrapped_indx =  w_indx
        else:
            wrapped_indx = run_length - w_indx
        # print('INDX: {}, wrapped_indx: {}, w_indx: {}, run_indx: {}, run_length: {}, direction: {}'.format(indx, wrapped_indx, w_indx, run_indx, run_length, direction))
        return wrapped_indx

    def __getitem__(self, item):
        index_i = self.indexing_order[item]
        pose_i = self.poses[index_i[0], index_i[1]]
        return pose_i


class ImpedanceWrenchDataCollection(MedDataCollectionBase):
    """
    Have the robot grasp a marker and move it along the whiteboard and collect data.
     - Fixed orientation of the end-effector.
     - Assume that it is in contact all time.
     * We expect to have the wrench readings to be fairly constant. High values in z because of the plane normal direction
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.grasp_width = 10
        self.pose_buffer = self._get_pose_buffer()
        # test_x = [self.pose_buffer._wrap_index(i) for i in range(20)]
        self.joint_listener = Listener('/med/joint_states', JointState, wait_for_data=True)
        self.joint_sequence = None
        self.initialized = False
        self.h = None
        self.med.home_robot()

    def _initialize_grasp(self):
        self.med.set_joint_position_control(vel=0.1)
        self.med.home_robot()
        self.med.set_robot_conf('grasp_conf')
        _ = input('Press enter to open the gripper and calibrate the bubbles')
        self.med.open_gripper()
        _ = input('Press enter to close the gripper')
        self.med.gripper.move(self.grasp_width)
        rospy.sleep(2.0)
        print('Calibration is done')
        self.med.home_robot()

    def _get_pose_buffer(self):
        num_x = 10
        num_y = 10
        x_lims = np.array([0.5, 0.7])
        y_lims = np.array([-0.1, 0.1])
        limits = np.stack([x_lims, y_lims], axis=1)
        h_value = 0.1
        quat = np.array([-np.cos(np.pi / 4), np.cos(np.pi / 4), 0, 0])
        pose_buffer = PoseBuffer(num_x=num_x, num_y=num_y, limits=limits, h_value=h_value, quat=quat)
        return pose_buffer

    def _init_drawing(self, pose_i):
        z_init = 0.15
        self.med.set_joint_position_control(vel=0.1)
        self.med.home_robot()
        self._initialize_grasp()
        init_pose = pose_i.copy()
        init_pose[2] = z_init
        # set the initial position and orientation
        self._plan_to_pose(init_pose, supervision=False)

        # start impedance mode and lower it down.
        self.med.cartesian._timeout_per_m = 500
        self.med.set_cartesian_impedance(0.25, z_stiffnes=2000)
        reached = self.med.cartesian_move(pose_i[:3], quat=pose_i[3:])
        self.med.cartesian.timeout_per_m = 500
        current_pose = self.med.get_current_pose()
        self.h = current_pose[2]
        return reached

    # Add recording of joints
    def _get_legend_column_names(self):
        """
        Return a list containing the column names of the datalegend
        Returns:
        """
        column_names = ['Time', 'Sequence', 'SequenceIndx', 'FileCode', 'JointState', 'Reached']
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
        if len(sequence_indxs)>0:
            seq_indx = np.max(sequence_indxs)+1
        return seq_indx

    def _collect_data_sample(self, params=None):
        """
        Collect and save data to the designed path in self.data_path
        Args:
            params:
        Returns: <dict> containing the parameters of the collected sample
        """
        data_params = defaultdict(list)
        joint_states = []
        seq_indx = self.pose_buffer.get_current_indx()
        indx = self.pose_buffer.indx
        pose_i = next(self.pose_buffer)
        if not self.initialized:
            reached = self._init_drawing(pose_i)
            self.initialized = True
        else:
            # Move the robot to the desired pose
            pose_x = pose_i.copy()
            pose_x[2] = self.h -0.02
            print(pose_x[:2])
            reached = self.med.cartesian_move(pose_x[:3], quat=pose_x[3:])
            pass

        rospy.sleep(1.0)
        fc_i = self.get_new_filecode()
        self._record(fc=fc_i)
        joints_i = self.joint_listener.get(block_until_data=True)
        joint_states.append(joints_i)
        data_params['Time'].append(time.time())
        data_params['Sequence'].append(seq_indx)
        data_params['SequenceIndx'].append(indx)
        data_params['FileCode'].append(fc_i)
        data_params['JointState'].append(joints_i.position)
        data_params['Reached'].append(reached)

        return data_params


# DEBUG:

if __name__ == '__main__':
    data_path = '/home/mmint/Desktop/bubbles_drawing_wrench_calibration'
    scene_name = 'drawing_wrench_veesa'
    dc = ImpedanceWrenchDataCollection(data_path=data_path, scene_name=scene_name)
    dc.collect_data(18)