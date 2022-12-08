import os

from bubble_utils.bubble_datasets.combined_dataset import CombinedDataset
from manipulation_via_membranes.bubble_learning.datasets.bubble_drawing_dataset import BubbleDrawingDataset


class DrawingDataset(CombinedDataset):

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
        super().__init__(datasets, data_name=os.path.join(self.data_dir, 'drawing_dataset'), **kwargs)

    @classmethod
    def get_name(self):
        return 'drawing_dataset'

    def _get_datasets(self):
        datasets = []
        drawing_dataset_line = BubbleDrawingDataset(
            data_name=os.path.join(self.data_dir, 'drawing_data_one_direction'),
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
        datasets.append(drawing_dataset_line)
        drawing_dataset_one_dir = BubbleDrawingDataset(
            data_name=os.path.join(self.data_dir, 'drawing_data_line'),
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
        datasets.append(drawing_dataset_one_dir)

        # Make them combined datasets:
        return datasets



if __name__ == '__main__':
    drawing_combined_dataset = DrawingDataset('/home/mik/Datasets/bubble_datasets')
    d0 = drawing_combined_dataset[0]
    print(d0)