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


class MedWrenchDataCollection(MedDataCollectionBase):

    def __init__(self, *args, **kwargs):
        self.home_conf = [0, 0.432, 0, -1.584, 0, 0.865, 0] # drawing home
        super().__init__(*args, **kwargs)
        self.joint_listener = Listener('/med/joint_states', JointState, wait_for_data=True)
        self.joint_sequence = None

    def home_robot(self):
        self.med.plan_to_joint_config(self.med.arm_group, self.home_conf)

    # Add recording of joints
    def _get_legend_column_names(self):
        """
        Return a list containing the column names of the datalegend
        Returns:
        """
        column_names = ['Time', 'Sequence', 'SequenceIndx', 'FileCode', 'JointState']
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
        # home robot
        self.home_robot()
        # Perform a sequence of predefined positions
        data_params = defaultdict(list)
        # grid_collection
        num_points_x = 3#5
        num_points_y = 3#7
        num_points_z = 3#7

        _xs = np.linspace(0.45, 0.65, num_points_x)
        _ys = np.linspace(-.3, .3, num_points_y)
        _zs = np.linspace(.1, .3, num_points_z)
        xs, ys, zs = np.meshgrid(_xs, _ys, _zs)
        positions = np.stack([xs.flatten(), ys.flatten(), zs.flatten()], axis=-1)
        joint_states = []
        quat = np.array([-np.cos(np.pi / 4), np.cos(np.pi / 4), 0, 0])
        sequence_i = self._get_sequence_indx()

        def __record(seq_i, indx):
            rospy.sleep(1.0)
            fc_i = self.get_new_filecode()
            self._record(fc=fc_i)
            joints_i = self.joint_listener.get(block_until_data=True)
            joint_states.append(joints_i)
            data_params['Time'].append(time.time())
            data_params['Sequence'].append(seq_i)
            data_params['SequenceIndx'].append(indx)
            data_params['FileCode'].append(fc_i)
            data_params['JointState'].append(joints_i.position)

        num_positions = 2*len(positions)
        with tqdm(total=num_positions, bar_format='(Seq: {postfix[3]}) - {postfix[0]} {postfix[1]}/{postfix[2]}{l_bar}{bar}| {n:.2f}/{total_fmt} [{elapsed}<{remaining}]', postfix=['Forward', 1, len(positions), sequence_i]) as pbar:
            if self.joint_sequence is None:
                for i, position_i in enumerate(positions):
                    pbar.postfix[1] = i+1
                    pose_i = np.concatenate([position_i, quat])
                    self.home_robot()
                    plan_success, execution_success = self._plan_to_pose(pose_i, supervision=self.supervision)
                    __record(sequence_i, i)
                    pbar.update()
                self.joint_sequence = copy.deepcopy(joint_states)
            else:
                for i, joint_state_i in enumerate(self.joint_sequence):
                    pbar.postfix[1] = i + 1
                    self.med.plan_to_joint_config(self.med.arm_group, joint_state_i.position[:-2])  # ONly the arm joints
                    __record(sequence_i, i)
                    pbar.update()
            # Repeat the sequence in the oposite direction
            pbar.postfix[0] = 'Backwards'
            for i, joint_state_i in enumerate(reversed(self.joint_sequence)):
                pbar.postfix[1] = i + 1
                self.med.plan_to_joint_config(self.med.arm_group, joint_state_i.position[:-2]) # ONly the arm joints
                __record(sequence_i, i+len(positions))
                pbar.update()


        return data_params