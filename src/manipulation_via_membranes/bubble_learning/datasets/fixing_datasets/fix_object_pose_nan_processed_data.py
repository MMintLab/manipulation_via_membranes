import torch
import os
import copy
import numpy as np
from tqdm import tqdm
from bubble_utils.bubble_datasets.transform_processed_dataset import transform_processed_dataset
from manipulation_via_membranes.aux.load_confs import load_object_models as load_object_models_drawing
from manipulation_via_membranes.bubble_pivoting.aux.load_confs import load_object_models as load_object_models_pivoting
from manipulation_via_membranes.bubble_learning.datasets.bubble_drawing_dataset import BubbleDrawingDataset
from manipulation_via_membranes.bubble_pivoting.datasets.bubble_pivoting_dataset import BubblePivotingDataset, BubblePivotingDownsampledDataset
from bubble_utils.bubble_datasets.data_transformations import TensorTypeTr
from manipulation_via_membranes.bubble_learning.aux.orientation_trs import QuaternionToAxis


class FixNanObjectPoseTr(object):
    def __init__(self):
        self.keys_to_tr = ['init_object_pose', 'final_object_pose', 'object_pose']

    def __call__(self, sample):
        for key in self.keys_to_tr:
            if key in sample:
                value = sample[key]
                sample[key] = self._tr(value)
        return sample

    def inverse(self, sample):
        return sample # No inverse

    def _tr(self, pose):
        pose_tr = copy.deepcopy(pose)
        pose_tr[torch.where(torch.isnan(pose_tr))] = 0 # replace nans by 0
        return pose_tr


def fix_nan_object_pose(dataset, init_indx=0, last_indx=0, indxs=None):
    """
    Convert object pose encoding on 'init_object_pose' and 'final_object_pose' to axis angle orientation encoding instead of quaternion
    :param dataset:
    :return:
    """
    fix_nan_object_pose_tr = FixNanObjectPoseTr()
    if indxs is None:
        indxs = np.arange(start=init_indx, stop=(last_indx - 1) % len(dataset) + 1)
    for indx in tqdm(indxs):
        sample_i = dataset[indx]
        sample_i = fix_nan_object_pose_tr(sample_i)
        # save
        save_path_i = os.path.join(dataset.processed_data_path, 'data_{}.pt'.format(indx))
        torch.save(sample_i, save_path_i)


def fix_object_pose_drawing_data(data_name):
    fix_nan_object_pose_tr = FixNanObjectPoseTr()
    dataset = BubbleDrawingDataset(
        data_name=data_name,
        downsample_factor_x=7,
        downsample_factor_y=7,
        downsample_reduction='mean')
    # fix wrench
    trs = [fix_nan_object_pose_tr]
    transform_processed_dataset(dataset, trs)


def fix_object_pose_pivoting_data(data_name):
    fix_nan_object_pose_tr = FixNanObjectPoseTr()
    dataset = BubblePivotingDownsampledDataset(
        data_name=data_name,
        downsample_factor_x=7,
        downsample_factor_y=7,
        downsample_reduction='mean')
    # fix wrench
    trs = [fix_nan_object_pose_tr]
    transform_processed_dataset(dataset, trs)


if __name__ == '__main__':
    data_path = '/home/mik/Datasets/bubble_datasets'

    fix_object_pose_drawing_data(os.path.join(data_path, 'drawing_data_one_direction'))
    fix_object_pose_drawing_data(os.path.join(data_path, 'drawing_data_line'))
    fix_object_pose_pivoting_data(os.path.join(data_path, 'bubble_pivoting_data'))
    fix_object_pose_pivoting_data(os.path.join(data_path, 'bubble_pivoting_data_wide_rotations'))
