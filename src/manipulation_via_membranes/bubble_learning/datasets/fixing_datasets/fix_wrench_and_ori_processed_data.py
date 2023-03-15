import torch
import os
import numpy as np
from tqdm import tqdm
from bubble_tools.bubble_datasets import transform_processed_dataset
from bubble_tools.bubble_datasets import TensorTypeTr
from manipulation_via_membranes.bubble_learning.datasets.bubble_drawing_dataset import BubbleDrawingDataset
from manipulation_via_membranes.bubble_pivoting.datasets.bubble_pivoting_dataset import BubblePivotingDownsampledDataset
from manipulation_via_membranes.bubble_learning.aux.orientation_trs import QuaternionToAxis


def fix_wrench(dataset, wrench_frame=None, init_indx=0, last_indx=0, indxs=None):
    if indxs is None:
        indxs = np.arange(start=init_indx, stop=(last_indx - 1) % len(dataset) + 1)
    for indx in tqdm(indxs):
        sample_code = dataset.sample_codes[indx]
        sample_i = dataset[indx]
        dl_line = dataset.dl.iloc[sample_code]
        scene_name = dl_line['Scene']
        init_fc = dl_line['InitialStateFC']
        final_fc = dl_line['FinalStateFC']
        init_wrench = dataset._get_wrench(fc=init_fc, scene_name=scene_name, frame_id=wrench_frame)
        final_wrench = dataset._get_wrench(fc=final_fc, scene_name=scene_name, frame_id=wrench_frame)
        sample_i['init_wrench'] = init_wrench.flatten()
        sample_i['final_wrench'] = final_wrench.flatten()
        # save
        save_path_i = os.path.join(dataset.processed_data_path, 'data_{}.pt'.format(indx))
        torch.save(sample_i, save_path_i)


def fix_wench_and_ori_drawing_data(data_name):
    tensor_type_tr = TensorTypeTr(dtype=torch.float32)
    quat_to_axis_tr = QuaternionToAxis()
    dataset = BubbleDrawingDataset(
        data_name=data_name,
        downsample_factor_x=7,
        downsample_factor_y=7,
        downsample_reduction='mean')
    # fix wrench
    fix_wrench(dataset, wrench_frame='med_base')
    trs = [quat_to_axis_tr, tensor_type_tr]
    transform_processed_dataset(dataset, trs)


def fix_wench_and_ori_pivoting_data(data_name):
    tensor_type_tr = TensorTypeTr(dtype=torch.float32)
    quat_to_axis_tr = QuaternionToAxis()
    dataset = BubblePivotingDownsampledDataset(
        data_name=data_name,
        downsample_factor_x=7,
        downsample_factor_y=7,
        downsample_reduction='mean')
    # fix wrench
    fix_wrench(dataset, wrench_frame='med_base')
    trs = [quat_to_axis_tr, tensor_type_tr]
    transform_processed_dataset(dataset, trs)


if __name__ == '__main__':
    data_path = '/home/mik/Datasets/bubble_datasets'

    fix_wench_and_ori_drawing_data(os.path.join(data_path, 'drawing_data_one_direction'))
    fix_wench_and_ori_drawing_data(os.path.join(data_path, 'drawing_data_line'))
    fix_wench_and_ori_pivoting_data(os.path.join(data_path, 'bubble_pivoting_data'))
