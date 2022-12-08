import numpy as np
import tf.transformations as tr

from bubble_utils.bubble_datasets.bubble_dataset_base import BubbleDatasetBase
from manipulation_via_membranes.bubble_learning.aux.img_trs.block_downsampling_tr import BlockDownSamplingTr
from manipulation_via_membranes.aux.load_confs import load_object_models
from manipulation_via_membranes.bubble_pose_estimation.bubble_pc_reconstruction import BubblePCReconstructorOfflineDepth
from mmint_camera_utils.ros_utils.utils import matrix_to_pose, pose_to_matrix


class BubbleDrawingDataset(BubbleDatasetBase):

    def __init__(self, *args, wrench_frame=None, tf_frame='grasp_frame', view=False,  downsample_factor_x=1, downsample_factor_y=1, downsample_reduction='mean', **kwargs):
        self.downsample_factor_x = downsample_factor_x
        self.downsample_factor_y = downsample_factor_y
        self.downsample_reduction = downsample_reduction
        self.block_mean_downsampling_tr = BlockDownSamplingTr(factor_x=downsample_factor_x,
                                                              factor_y=downsample_factor_y,
                                                              reduction=self.downsample_reduction)  # downsample all imprint values
        # add the block_mean_downsampling_tr to the tr list
        kwargs = self._add_transformation_to_kwargs(kwargs, self.block_mean_downsampling_tr)

        self.wrench_frame = wrench_frame
        self.tf_frame = tf_frame
        self.view = view
        super().__init__(*args, **kwargs)

    @classmethod
    def get_name(self):
        return 'bubble_drawing_dataset'

    def _get_sample(self, sample_code):
        # fc: index of the line in the datalegend (self.dl) of the sample
        dl_line = self.dl.iloc[sample_code]
        scene_name = dl_line['Scene']
        undef_fc = int(dl_line['UndeformedFC'])
        init_fc = int(dl_line['InitialStateFC'])
        final_fc = int(dl_line['FinalStateFC'])
        # Load initial state:
        init_imprint_r = self._get_depth_imprint(undef_fc=undef_fc, def_fc=init_fc, scene_name=scene_name, camera_name='right')
        init_imprint_l = self._get_depth_imprint(undef_fc=undef_fc, def_fc=init_fc, scene_name=scene_name, camera_name='left')
        init_imprint = np.stack([init_imprint_r, init_imprint_l], axis=0)
        init_wrench = self._get_wrench(fc=init_fc, scene_name=scene_name, frame_id=self.wrench_frame)
        # Final State
        final_imprint_r = self._get_depth_imprint(undef_fc=undef_fc, def_fc=final_fc, scene_name=scene_name, camera_name='right')
        final_imprint_l = self._get_depth_imprint(undef_fc=undef_fc, def_fc=final_fc, scene_name=scene_name, camera_name='left')
        final_imprint = np.stack([final_imprint_r, final_imprint_l], axis=0)
        final_wrench = self._get_wrench(fc=final_fc, scene_name=scene_name, frame_id=self.wrench_frame)

        init_tf = self._get_tfs(init_fc, scene_name=scene_name, frame_id=self.tf_frame)
        final_tf = self._get_tfs(final_fc, scene_name=scene_name, frame_id=self.tf_frame)
        init_pos = init_tf[..., :3]
        init_quat = init_tf[..., 3:]
        final_pos = final_tf[..., :3]
        final_quat = final_tf[..., 3:]

        undef_depth_r = self._load_bubble_depth_img(fc=undef_fc, scene_name=scene_name, camera_name='right')
        undef_depth_l = self._load_bubble_depth_img(fc=undef_fc, scene_name=scene_name, camera_name='left')
        init_def_depth_r = self._load_bubble_depth_img(fc=init_fc, scene_name=scene_name, camera_name='right')
        init_def_depth_l = self._load_bubble_depth_img(fc=init_fc, scene_name=scene_name, camera_name='left')
        final_def_depth_r = self._load_bubble_depth_img(fc=final_fc, scene_name=scene_name, camera_name='right')
        final_def_depth_l = self._load_bubble_depth_img(fc=final_fc, scene_name=scene_name, camera_name='left')

        # load tf from cameras to grasp frame (should
        all_tfs = self._load_tfs(init_fc, scene_name)

        # Action:
        action_fc = sample_code
        action = self._get_action(action_fc)

        # camera info
        camera_info_r = self._load_bubble_camera_info_depth(scene_name=scene_name, camera_name='right', fc=undef_fc)
        camera_info_l = self._load_bubble_camera_info_depth(scene_name=scene_name, camera_name='left', fc=undef_fc)

        object_code = self._get_object_code(sample_code)
        object_model = self._get_object_model(object_code)
        init_object_pose = self._estimate_object_pose(init_def_depth_r, init_def_depth_l, undef_depth_r, undef_depth_l, camera_info_r, camera_info_l, all_tfs)
        final_object_pose = self._estimate_object_pose(final_def_depth_r, final_def_depth_l, undef_depth_r, undef_depth_l, camera_info_r, camera_info_l, all_tfs)
        sample_simple = {
            'init_imprint': init_imprint,
            'init_wrench': init_wrench,
            'init_pos': init_pos,
            'init_quat': init_quat,
            'final_imprint': final_imprint,
            'final_wrench': final_wrench,
            'final_pos': final_pos,
            'final_quat': final_quat,
            'object_code': object_code,
            'object_model': object_model,
            'init_object_pose': init_object_pose,
            'final_object_pose': final_object_pose,
            'action': action,
            'undef_depth_r': undef_depth_r,
            'undef_depth_l': undef_depth_l,
            'camera_info_r': camera_info_r,
            'camera_info_l': camera_info_l,
            'all_tfs': all_tfs,
        }
        sample = self._reshape_sample(sample_simple)
        sample = self._compute_delta_sample(sample) # Add delta values to sample

        return sample

    def _get_action(self, fc):
        # TODO: Load from file instead of the logged values in the dl
        dl_line = self.dl.iloc[fc]
        action_column_names = ['rotation', 'length', 'grasp_width']
        action_i = dl_line[action_column_names].values.astype(np.float64)
        # direction = dl_line['direction']
        # length = dl_line['length']
        # action_i = length * np.array([np.cos(direction), np.sin(direction)])
        return action_i

    def _estimate_object_pose(self, def_r, def_l, ref_r, ref_l, camera_info_r, camera_info_l, all_tfs):
        reconstructor = BubblePCReconstructorOfflineDepth(object_name='marker', estimation_type='icp2d', view=self.view, percentile=0.005)
        reconstructor.threshold = 0.
        reconstructor.references['left'] = ref_l
        reconstructor.references['right'] = ref_r
        reconstructor.references['left_frame'] = 'pico_flexx_left_optical_frame'
        reconstructor.references['right_frame'] = 'pico_flexx_right_optical_frame'
        reconstructor.camera_info['left'] = camera_info_l
        reconstructor.camera_info['right'] = camera_info_r
        reconstructor.depth_r['img'] = def_r
        reconstructor.depth_l['img'] = def_l
        reconstructor.depth_r['frame'] = 'pico_flexx_right_optical_frame'
        reconstructor.depth_l['frame'] = 'pico_flexx_left_optical_frame'
        reconstructor.add_tfs(all_tfs)
        pose_matrix = reconstructor.estimate_pose(threshold=0, view=self.view) # Homogeneous
        pose = matrix_to_pose(pose_matrix)
        return pose

    def _get_object_code(self, fc):
        dl_line = self.dl.iloc[fc]
        object_code = dl_line['marker_init']
        return object_code

    def _get_object_model(self, object_code):
        object_models = load_object_models() # TODO: Consdier doing this more efficient to avoid having to load every time
        object_model_pcd = object_models[object_code]
        object_model = np.asarray(object_model_pcd.points)
        return object_model

    def _compute_delta_sample(self, sample):
        # TODO: improve this computation
        input_keys = ['imprint', 'wrench', 'pos']
        time_keys = ['init', 'final']
        for in_key in input_keys:
            sample['delta_{}'.format(in_key)] = sample['final_{}'.format(in_key)] - sample['init_{}'.format(in_key)]
        # for quaternion, compute the delta quaternion q_delta @ q_init = q_final <=> q_delta
        sample['delta_quat'] = tr.quaternion_multiply(sample['final_quat'], tr.quaternion_inverse(sample['init_quat']))
        return sample

    def _reshape_sample(self, sample):
        input_keys = ['wrench', 'pos', 'quat']
        time_keys = ['init', 'final']
        # reshape the imprint
        for time_key in time_keys:
            imprint_x = sample['{}_imprint'.format(time_key)]
            sample['{}_imprint'.format(time_key)] = imprint_x.transpose((0, 3, 1, 2)).reshape(-1, *imprint_x.shape[1:3])
            for in_key in input_keys:
                sample['{}_{}'.format(time_key, in_key)] = sample['{}_{}'.format(time_key, in_key)].flatten()
        sample['action'] = sample['action'].flatten()
        return sample

    def _add_transformation_to_kwargs(self, kwargs, tr):
        if 'transformation' in kwargs:
            if type(kwargs['transformation']) in (list, tuple):
                kwargs['transformation'] = list(kwargs['transformation']) + [tr]
            else:
                print('')
                raise AttributeError('Not supportes trasformations: {} type {}'.format(kwargs['transformation'],
                                                                                       type(kwargs['transformation'])))
        else:
            kwargs['transformation'] = [tr]
        return kwargs


# DEBUG
if __name__ == '__main__':
    # data_name = '/home/mmint/Desktop/drawing_data_cartesian'
    data_name = '/home/mmint/Desktop/test_drawing_data'
    dataset = BubbleDrawingDataset(data_name=data_name, wrench_frame='med_base', tf_frame='grasp_frame', view=False)
    # dataset = BubbleDrawingDownsampledDataset(data_name=data_name, wrench_frame='med_base', tf_frame='grasp_frame',downsample_factor_x=7, downsample_factor_y=7, downsample_reduction='mean')
    print('Dataset Name: ', dataset.name)
    print('Dataset Length:', len(dataset))
    sample_0 = dataset[0]
    print('Sample 0:', sample_0)

    # # save downsampled and undownsampled images
    # import matplotlib.pyplot as plt
    # # sample_indxs = np.random.randint(0,len(dataset),5)
    # sample_indxs = [0,1,2,3,4]
    #
    # for sample_indx in sample_indxs:
    #     sample_i = dataset[sample_indx]
    #     img_o = sample_i['init_imprint_undownsampled'].reshape(-1, sample_i['init_imprint_undownsampled'].shape[-1])
    #     img_d = sample_i['init_imprint'].reshape(-1, sample_i['init_imprint'].shape[-1])
    #
    #     fig, axes = plt.subplots(1, 2)
    #     axes[0].imshow(img_o)
    #     axes[0].set_title('Original Resolution')
    #     axes[1].imshow(img_d)
    #     axes[1].set_title('Image Downsampled (Avg Pooling)')
    #
    #     plt.savefig('/home/mmint/Desktop/resolution_comparison_{}.png'.format(sample_indx))
