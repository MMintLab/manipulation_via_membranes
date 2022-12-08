import numpy as np

from bubble_utils.bubble_datasets.bubble_dataset_base import BubbleDatasetBase


class BubbleSimpleDataset(BubbleDatasetBase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @classmethod
    def get_name(self):
        return 'bubble_simple_dataset'

    def _get_sample(self, sample_code):
        # fc: index of the line in the datalegend (self.dl) of the sample
        dl_line = self.dl.iloc[sample_code]
        scene_name = dl_line['Scene']
        fc = dl_line['FC']
        time = dl_line['Time']

        # Load the state:
        depth_r = self._load_bubble_depth_img(fc=fc, scene_name=scene_name, camera_name='right')
        depth_l = self._load_bubble_depth_img(fc=fc, scene_name=scene_name, camera_name='left')
        color_r = self._load_bubble_color_img(fc=fc, scene_name=scene_name, camera_name='right')
        color_l = self._load_bubble_color_img(fc=fc, scene_name=scene_name, camera_name='left')

        # camera info
        camera_info_r = self._load_bubble_camera_info_depth(scene_name=scene_name, camera_name='right')
        camera_info_l = self._load_bubble_camera_info_depth(scene_name=scene_name, camera_name='left')

        sample_simple = {
            'time': time,
            'depth_r': depth_r,
            'depth_l': depth_l,
            'color_r': color_r,
            'color_l': color_l,
            'camera_info_r': camera_info_r,
            'camera_info_l': camera_info_l,
        }
        return sample_simple


# DEBUG:

if __name__ == '__main__':
    from collections import defaultdict
    import matplotlib.pyplot as plt
    from tqdm import tqdm
    from matplotlib import cm
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    from bubble_utils.bubble_tools.bubble_img_tools import process_bubble_img

    data_name = '/home/mmint/Desktop/bubble_calibration_data'
    dataset = BubbleSimpleDataset(data_name=data_name)
    print('Dataset Name: ', dataset.name)
    print('Dataset Length:', len(dataset))
    sample_0 = dataset[0]
    print('Sample 0:', sample_0)
    keys_to_concatenate = ['time', 'depth_r', 'depth_l', 'color_r', 'color_l']
    concatenated_keys = defaultdict(list)
    for i, sample_i in enumerate(tqdm(dataset)):
        for key in keys_to_concatenate:
            concatenated_keys[key].append(sample_i[key])

    # compute means (along the stacked axis, which is the 0):
    means = {}
    stds = {}

    filter = True
    axis_on = True

    for key in concatenated_keys.keys():
        concatenated_keys[key] = np.stack(concatenated_keys[key]).squeeze()
        means[key] = np.mean(concatenated_keys[key], axis=0)
        stds[key] = np.std(concatenated_keys[key], axis=0)

    # plot the statistics
    cmap = cm.get_cmap('jet')
    fig, axes = plt.subplots(2, 2, figsize=(14,9))
    # plt left mean
    ax_i = axes[0, 0]
    if not axis_on:
        ax_i.axis('off')
    img_ar_i = means['depth_l']
    if filter:
        img_ar_i = process_bubble_img(img_ar_i)
    im_i = ax_i.imshow(img_ar_i)
    ax_i.set_title('Mean Depth Left [m]')
    divider = make_axes_locatable(ax_i)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im_i, cax=cax)

    ax_i = axes[0, 1]
    if not axis_on:
        ax_i.axis('off')
    img_ar_i = means['depth_r']
    if filter:
        img_ar_i = process_bubble_img(img_ar_i)
    im_i = ax_i.imshow(img_ar_i)
    ax_i.set_title('Mean Depth Right [m]')
    divider = make_axes_locatable(ax_i)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im_i, cax=cax)

    ax_i = axes[1, 0]
    if not axis_on:
        ax_i.axis('off')
    img_ar_i = stds['depth_l']
    if filter:
        img_ar_i = process_bubble_img(img_ar_i)
    im_i = ax_i.imshow(img_ar_i)
    ax_i.set_title('Std Depth Left [m]')
    divider = make_axes_locatable(ax_i)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im_i, cax=cax)

    ax_i = axes[1, 1]
    if not axis_on:
        ax_i.axis('off')
    img_ar_i = stds['depth_r']
    if filter:
        img_ar_i = process_bubble_img(img_ar_i)
    im_i = ax_i.imshow(img_ar_i)
    ax_i.set_title('Std Depth Right [m]')
    divider = make_axes_locatable(ax_i)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im_i, cax=cax)

    name = ''
    if filter:
        name += '_filter'
    if axis_on:
        name += '_axis'
    plt.savefig('/home/mmint/Desktop/bubble_calibration{}.png'.format(name))







