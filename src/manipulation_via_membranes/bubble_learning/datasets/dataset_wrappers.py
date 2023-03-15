import numpy as np

from bubble_tools.bubble_datasets import BubbleDatasetBase
from manipulation_via_membranes.bubble_learning.aux.img_trs.block_downsampling_tr import BlockDownSamplingTr
from mmint_tools.wrapping_utils.wrapping_utils import ClassWrapper


def imprint_downsampled_dataset(cls):
    """ This does not really work -- TODO: Fix this class """
    # class Wrapper(AttributeWrapper):
    class Wrapper(ClassWrapper):
        def __init__(self, *args, downsample_factor_x=5, downsample_factor_y=5, downsample_reduction='mean', **kwargs):
            self.downsample_factor_x = downsample_factor_x
            self.downsample_factor_y = downsample_factor_y
            self.downsample_reduction = downsample_reduction
            self.block_mean_downsampling_tr = BlockDownSamplingTr(factor_x=downsample_factor_x,
                                                                  factor_y=downsample_factor_y,
                                                                  reduction=self.downsample_reduction)  # downsample all imprint values

            # add the block_mean_downsampling_tr to the tr list
            if 'transformation' in kwargs:
                if type(kwargs['transformation']) in (list, tuple):
                    kwargs['transformation'] = list(kwargs['transformation']) + [self.block_mean_downsampling_tr]
                else:
                    print('')
                    raise AttributeError('Not supportes trasformations: {} type {}'.format(kwargs['transformation'],
                                                                                           type(kwargs[
                                                                                                    'transformation'])))
            else:
                kwargs['transformation'] = [self.block_mean_downsampling_tr]
            wrapped_obj = cls(*args, **kwargs)
            super().__init__(wrapped_obj)

        @classmethod
        def get_name(self):
            return '{}_downsampled'.format(self.wrapped_object.get_name())

    return Wrapper


class BubbleImprintCombinedDatasetWrapper(BubbleDatasetBase):
    """
    Creates a new dataset from the original wrapped dataset which is the the dataset in 2 so samples contain 'imprint' which is the combination of 'init_imprint' and 'final_imprint'
    """
    def __init__(self, dataset):
        self.dataset = dataset
        super().__init__(data_name=dataset.data_path)

    def _get_sample_codes(self):
        filecodes = np.arange(2*len(self.dataset))
        return filecodes

    def _get_sample(self, sample_code):
        true_fc = sample_code // 2
        sample = self.dataset[true_fc]
        if sample_code % 2 == 0:
            # sample is the initial
            key = 'init'
        else:
            key = 'final'
        sample['imprint'] = sample['{}_imprint'.format(key)]
        sample['object_pose'] = sample['{}_object_pose'.format(key)]
        sample['wrench'] = sample['{}_wrench'.format(key)]
        sample['pos'] = sample['{}_pos'.format(key)]
        sample['quat'] = sample['{}_quat'.format(key)]
        return sample

    def get_name(self):
        name = '{}_imprint_combined'.format(self.dataset.name)
        return name