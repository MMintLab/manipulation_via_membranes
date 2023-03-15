import os
import torch
from bubble_tools.bubble_datasets import CombinedDataset
from manipulation_via_membranes.bubble_pivoting.datasets.bubble_pivoting_dataset import BubblePivotingDownsampledDataset
from manipulation_via_membranes.bubble_learning.aux.orientation_trs import QuaternionToAxis
from manipulation_via_membranes.bubble_learning.datasets.fixing_datasets.fix_object_pose_encoding_processed_data import EncodeObjectPoseAsAxisAngleTr


class PivotingCombinedDataset(CombinedDataset):

    def __init__(self, data_name, downsample_factor_x=7, downsample_factor_y=7, wrench_frame='med_base', downsample_reduction='mean', transformation=None, dtype=None, load_cache=True,contribute_mode=False, clean_if_error=True, **kwargs):
        self.data_dir = data_name # it assumes that all datasets are found at the same directory called data_dir
        self.downsample_factor_x = downsample_factor_x
        self.downsample_factor_y = downsample_factor_y
        self.downsample_reduction = downsample_reduction
        self.wrench_frame = wrench_frame
        self.dtype = dtype
        self.transformation = transformation
        self.load_cache = load_cache
        self.contribute_mode = contribute_mode
        self.clean_if_error = clean_if_error
        datasets = self._get_datasets()
        super().__init__(datasets, data_name=self.data_dir, **kwargs)

    @classmethod
    def get_name(self):
        return 'pivoting_combined_dataset'

    def _get_datasets(self):
        datasets = []
        pivoting_dataset = BubblePivotingDownsampledDataset(
            data_name=os.path.join(self.data_dir, 'bubble_pivoting_data'),
            downsample_factor_x=self.downsample_factor_x,
            downsample_factor_y=self.downsample_factor_y,
            downsample_reduction=self.downsample_reduction,
            wrench_frame=self.wrench_frame,
            dtype=self.dtype,
            transformation=self.transformation,
            load_cache=self.load_cache,
            contribute_mode=self.contribute_mode,
            clean_if_error=self.clean_if_error,
        )
        datasets.append(pivoting_dataset)
        pivoting_dataset_wide_rot = BubblePivotingDownsampledDataset(
            data_name=os.path.join(self.data_dir, 'bubble_pivoting_data_wide_rotations'),
            downsample_factor_x=self.downsample_factor_x,
            downsample_factor_y=self.downsample_factor_y,
            downsample_reduction=self.downsample_reduction,
            wrench_frame=self.wrench_frame,
            dtype=self.dtype,
            transformation=self.transformation,
            load_cache=self.load_cache,
            contribute_mode=self.contribute_mode,
            clean_if_error=self.clean_if_error,
        )
        datasets.append(pivoting_dataset_wide_rot)

        # Make them combined datasets:
        return datasets



if __name__ == '__main__':
    trs = [QuaternionToAxis(), EncodeObjectPoseAsAxisAngleTr()]
    pivoting_combined_dataset = PivotingCombinedDataset('/home/mireiaplanaslisbona/Documents/research', transformation=trs, dtype=torch.float32)
    d0 = pivoting_combined_dataset[0]
    print(d0)