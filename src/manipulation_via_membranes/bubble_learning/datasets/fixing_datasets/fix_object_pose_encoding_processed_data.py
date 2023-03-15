import torch
import os
import numpy as np
from tqdm import tqdm
from bubble_tools.bubble_datasets import transform_processed_dataset
from manipulation_via_membranes.bubble_learning.datasets.bubble_drawing_dataset import BubbleDrawingDataset
from manipulation_via_membranes.bubble_pivoting.datasets.bubble_pivoting_dataset import BubblePivotingDownsampledDataset
from manipulation_via_membranes.bubble_learning.aux.orientation_trs import QuaternionToAxis


class InverseTr(object):
    def __init__(self, tr):
        self.tr = tr

    def __call__(self, sample):
        sample_tr = self.tr.inverse(sample)
        return sample_tr

    def inverse(self, sample_tr):
        sample = self.tr(sample_tr)
        return sample


class SplitPoseTr(object):
    def __init__(self, keys_to_tr=None):
        self.keys_to_tr = keys_to_tr

    def __call__(self, sample, replace=False):
        keys_to_tr = self._get_keys_to_tr(sample)
        for key in keys_to_tr:
            value = sample[key]
            pos, quat = self._tr(value)
            sample['{}_pos'.format(key)] = pos
            sample['{}_quat'.format(key)] = quat
            if replace:
                sample.pop(key)
        return sample

    def inverse(self, sample, replace=False):
        keys_to_tr = self._get_keys_to_tr(sample)
        for key in keys_to_tr:
            pos_key = '{}_pos'.format(key)
            quat_key = '{}_quat'.format(key)
            if pos_key in sample and quat_key in sample:
                pose = self._tr_inv(sample[pos_key], sample[quat_key])
                sample[key] = pose
                if replace:
                    sample.pop(pos_key)
                    sample.pop(quat_key)
        return sample

    def _get_keys_to_tr(self, sample):
        if self.keys_to_tr is None:
            keys_to_tr = [key for key in sample.keys() if 'pose' in key]
        else:
            keys_to_tr = [k for k in self.keys_to_tr if ((k in sample) or ('{}_pos'.format(k) in sample) or ('{}_quat'.format(k) in sample))]
        return keys_to_tr

    def _tr(self, pose):
        # splint into pos and quat
        pos = pose[..., :3]
        quat = pose[..., 3:]
        return pos, quat

    def _tr_inv(self, pos, quat):
        if torch.is_tensor(pos) and torch.is_tensor(quat):
            pose = torch.cat([pos, quat], dim=-1)
        else:
            pose = np.concatenate([pos, quat], axis=-1)
        return pose


class EncodeObjectPoseAsAxisAngleTr(object):
    def __init__(self):
        self.keys_to_tr = ['init_object_pose', 'final_object_pose']
        self.split_pose_tr = SplitPoseTr(keys_to_tr=self.keys_to_tr)
        self.quat_to_axis_tr = QuaternionToAxis(keys_to_tr=['{}_quat'.format(k) for k in self.keys_to_tr])

    def __call__(self, sample):
        sample = self.split_pose_tr(sample) # split pose into pos and quat
        sample = self.quat_to_axis_tr(sample) # encode only the quaternion part
        sample = self.split_pose_tr.inverse(sample, replace=True) # restore the pose from pos and quat
        return sample

    def inverse(self, sample):
        sample = self.split_pose_tr(sample) # split pose into pos and quat
        sample = self.quat_to_axis_tr.inverse(sample) # encode only the quaternion part
        sample = self.split_pose_tr.inverse(sample, replace=True) # restore the pose from pos and quat
        return sample


def fix_object_pose(dataset, init_indx=0, last_indx=0, indxs=None):
    """
    Convert object pose encoding on 'init_object_pose' and 'final_object_pose' to axis angle orientation encoding instead of quaternion
    :param dataset:
    :return:
    """
    encode_object_pose_as_axis_angle_tr = EncodeObjectPoseAsAxisAngleTr()
    if indxs is None:
        indxs = np.arange(start=init_indx, stop=(last_indx - 1) % len(dataset) + 1)
    for indx in tqdm(indxs):
        sample_i = dataset[indx]
        sample_i = encode_object_pose_as_axis_angle_tr(sample_i)
        # save
        save_path_i = os.path.join(dataset.processed_data_path, 'data_{}.pt'.format(indx))
        torch.save(sample_i, save_path_i)


def fix_object_pose_drawing_data(data_name):
    encode_object_pose_as_axis_angle_tr = EncodeObjectPoseAsAxisAngleTr()
    dataset = BubbleDrawingDataset(
        data_name=data_name,
        downsample_factor_x=7,
        downsample_factor_y=7,
        downsample_reduction='mean')
    # fix wrench
    trs = [encode_object_pose_as_axis_angle_tr]
    transform_processed_dataset(dataset, trs)


def fix_object_pose_pivoting_data(data_name):
    encode_object_pose_as_axis_angle_tr = EncodeObjectPoseAsAxisAngleTr()
    dataset = BubblePivotingDownsampledDataset(
        data_name=data_name,
        downsample_factor_x=7,
        downsample_factor_y=7,
        downsample_reduction='mean')
    # fix wrench
    trs = [encode_object_pose_as_axis_angle_tr]
    transform_processed_dataset(dataset, trs)


if __name__ == '__main__':
    data_path = '/home/mik/Datasets/bubble_datasets'

    fix_object_pose_drawing_data(os.path.join(data_path, 'drawing_data_one_direction'))
    fix_object_pose_drawing_data(os.path.join(data_path, 'drawing_data_line'))
    fix_object_pose_pivoting_data(os.path.join(data_path, 'bubble_pivoting_data'))
