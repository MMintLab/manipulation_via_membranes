#!/usr/bin/env python

import numpy as np
from numpy.lib.type_check import imag
from tf import transformations as tr
import json

from bubble_utils.bubble_datasets.bubble_dataset_base import BubbleDatasetBase
import scipy.optimize as opt
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from manipulation_via_membranes.bubble_learning.aux.img_trs.block_downsampling_tr import BlockDownSamplingTr
from manipulation_via_membranes.bubble_pose_estimation.bubble_pc_reconstruction import BubblePCReconstructorOfflineDepth
from mmint_camera_utils.ros_utils.utils import matrix_to_pose, pose_to_matrix
from mmint_camera_utils.ros_utils.utils import matrix_to_pose, pose_to_matrix
from manipulation_via_membranes.bubble_pivoting.aux.load_confs import load_object_models



class BubblePivotingDataset(BubbleDatasetBase):

    def __init__(self, *args, wrench_frame=None, scene_name=None, tf_frame='grasp_frame', filters=[], trans_threshold=0.1, view=False, **kwargs):
        self.wrench_frame = wrench_frame
        self.tf_frame = tf_frame
        self.scene_name = scene_name
        self.filters = filters
        self.trans_threshold = trans_threshold
        self.view = view
        super().__init__(*args, **kwargs)

    @classmethod
    def get_name(self):
        return 'bubble_pivoting_dataset'

    def _get_sample(self, sample_code):
        # fc: index of the line in the datalegend (self.dl) of the sample
        # TODO: Change names to my categories and maybe record depth too
        dl_line = self.dl.iloc[sample_code]
        scene_name = dl_line['Scene']
        undef_fc = dl_line['UndeformedFC']
        init_fc = dl_line['InitialStateFC']
        final_fc = dl_line['FinalStateFC']
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

        # TODO: Add frames on sample
        init_tf = self._get_tfs(init_fc, scene_name=scene_name, frame_id=self.tf_frame)
        final_tf = self._get_tfs(final_fc, scene_name=scene_name, frame_id=self.tf_frame)
        self.init_pos = init_tf[..., :3]
        self.init_quat = init_tf[..., 3:]
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

        # camera info
        camera_info_r = self._load_bubble_camera_info_depth(scene_name=scene_name, camera_name='right', fc=undef_fc)
        camera_info_l = self._load_bubble_camera_info_depth(scene_name=scene_name, camera_name='left', fc=undef_fc)
        # fitted_gaussian_init = self._fit_gaussian(init_imprint)    
        # fitted_gaussian_final = self._fit_gaussian(final_imprint)

        # Additional info
        info = dl_line['Info']
        info = info.replace(",", "\',")
        info = info.replace(": ", ": \'")
        info = info.replace("}", "\'}")
        info = info.replace("\'", "\"")
        info_dic = json.loads(info)
        max_force_felt = float(info_dic['max_force_felt'])
        tool = bool(info_dic['tool'])

        # Action:
        action_fc = sample_code
        action = self._get_action(action_fc)
        action = action.astype(np.float)

        object_code = self._get_object_code(sample_code)
        object_model = self._get_object_model(object_code)
        init_object_pose = self._estimate_object_pose(init_def_depth_r, init_def_depth_l, undef_depth_r, undef_depth_l, camera_info_r, camera_info_l, all_tfs)
        final_object_pose = self._estimate_object_pose(final_def_depth_r, final_def_depth_l, undef_depth_r, undef_depth_l, camera_info_r, camera_info_l, all_tfs)

        sample_simple = {
            'init_imprint': init_imprint,
            'init_wrench': init_wrench,
            'init_pos': self.init_pos,
            'init_quat': self.init_quat,
            'final_imprint': final_imprint,
            'final_wrench': final_wrench,
            'final_pos': final_pos,
            'final_quat': final_quat,
            'object_code': object_code,
            'object_model': object_model,
            'init_object_pose': init_object_pose,
            'final_object_pose': final_object_pose,
            'action': action,
            'tool_detected': tool,
            'max_force_felt': max_force_felt,
            'undef_depth_r': undef_depth_r,
            'undef_depth_l': undef_depth_l,
            'camera_info_r': camera_info_r,
            'camera_info_l': camera_info_l,
            'all_tfs': all_tfs,
        }
        sample = self._reshape_sample(sample_simple)
        sample = self._compute_delta_sample(sample) # Add delta values to sample
        #sample = self._create_imprint_img(sample)
        return sample

    def _get_relative_tf(self, parent_frame_tf, child_frame_tf):
        child_tf_matrix_wf = pose_to_matrix(child_frame_tf)
        parent_tf_matrix_wf = pose_to_matrix(parent_frame_tf)
        relative_tf_matrix = np.linalg.inv(parent_tf_matrix_wf) @ child_tf_matrix_wf
        quat = tr.quaternion_from_matrix(relative_tf_matrix)
        t = relative_tf_matrix[:3,3]
        relative_pose = np.concatenate([t, quat])
        return relative_pose

    def _get_action(self, fc):
        # TODO: Load from file instead of the logged values in the dl
        action_column_names = ['grasp_width', 'delta_y', 'delta_z', 'delta_roll']
        dl_line = self.dl.iloc[fc]
        action_i = dl_line[action_column_names].values.astype(np.float64)
        return action_i

    def string_to_array(self, string):
        s = string.replace('\n', '')
        s = s.replace('[', '')
        s = s.replace(']', '')
        array = np.fromstring(s, dtype=np.float64, sep='  ')
        return array

    def _estimate_object_pose(self, def_r, def_l, ref_r, ref_l, camera_info_r, camera_info_l, all_tfs):
        reconstructor = BubblePCReconstructorOfflineDepth(object_name='marker', estimation_type='icp2d', view=self.view, percentile=0.005)
        reconstructor.threshold = 0.0
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
        object_code = dl_line['Scene']
        return object_code

    def _get_object_model(self, object_code):
        object_models = load_object_models() # TODO: Consdier doing this more efficient to avoid having to load every time
        object_model_pcd = object_models[object_code]
        object_model = np.asarray(object_model_pcd.points)
        return object_model

    def _get_category(self, fc, category):
        dl_line = self.dl.iloc[fc]
        value = dl_line[category]
        return np.asarray([value])

    def _compute_delta_sample(self, sample):
        # TODO: improve this computation
        input_keys = ['wrench', 'pos']
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
            sample['{}_imprint'.format(time_key)] = imprint_x.transpose((0,3,1,2)).reshape(-1, *imprint_x.shape[1:3])
            for in_key in input_keys:
                sample['{}_{}'.format(time_key, in_key)] = sample['{}_{}'.format(time_key, in_key)].flatten()
        sample['action'] = sample['action'].flatten()
        return sample

    def _create_imprint_img(self, sample):
        time_keys = ['init', 'final']
        for time_key in time_keys:
            image = []
            for i in np.arange(2):
                fig = plt.figure()
                plt.imshow(sample['{}_imprint'.format(time_key)][i], interpolation='nearest')
                canvas = FigureCanvas(fig)
                ax = fig.gca()
                ax.axis('off')
                canvas.draw()
                buf = canvas.tostring_rgb()
                image_i = np.frombuffer(buf, dtype='uint8')
                w, h = canvas.get_width_height()
                image_i = np.fromstring(buf, dtype=np.uint8).reshape(h, w, 3)
                image.append(image_i)
                plt.close()
            sample['{}_imprint_img'.format(time_key)] = np.concatenate(image)
        return sample

    def _fit_gaussian(self, imprint):
        params = np.zeros((2, 7))
        for j, image_j in enumerate(imprint):
            shape = image_j.shape
            x = np.linspace(0, shape[1]-1, shape[1])
            y = np.linspace(0, shape[0]-1, shape[0])
            x, y = np.meshgrid(x, y)
            xy = [x, y]

            initial_guess = (5,shape[1]/2,shape[0]/2,20,40,0,10)
            # plt.imshow(image_j[:,:,0])
            # plt.show()
            popt, pcov = opt.curve_fit(self.twoD_Gaussian_alt, xy, image_j.flatten(), p0=initial_guess, maxfev=100000)
            params[j] = popt
            # self.print_twoD_Gaussian(xy, popt[0], popt[1], popt[2], popt[3], popt[4], popt[5], popt[6])
        params = params.flatten()
        return np.array(params, dtype=np.float32)


    def twoD_Gaussian(self, xy, amplitude, xo, yo, sigma_x, sigma_y, theta, offset):
        xo = float(xo)
        yo = float(yo)    
        a = (np.cos(theta)**2)/(2*sigma_x**2) + (np.sin(theta)**2)/(2*sigma_y**2)
        b = -(np.sin(2*theta))/(4*sigma_x**2) + (np.sin(2*theta))/(4*sigma_y**2)
        c = (np.sin(theta)**2)/(2*sigma_x**2) + (np.cos(theta)**2)/(2*sigma_y**2)
        g = offset + amplitude*np.exp( - (a*((xy[0]-xo)**2) + 2*b*(xy[0]-xo)*(xy[1]-yo) + c*((xy[1]-yo)**2)))
        return g.ravel()

    def twoD_Gaussian_alt(self, xy, amplitude, xo, yo, sigma_x, sigma_y, theta, offset):
        xo = float(xo)
        yo = float(yo)
        k = 1 / (2*(1-theta**2))
        g = offset + amplitude*np.exp( - k * ((xy[0]-xo)**2/ sigma_x**2 - 2*theta*(xy[0]-xo)*(xy[1]-yo)/(sigma_x*sigma_y) + (xy[1]-yo)**2/sigma_y**2))
        return g.ravel()

    def print_twoD_Gaussian(self, xy, amplitude, xo, yo, sigma_x, sigma_y, theta, offset):
        xo = float(xo)
        yo = float(yo)    
        a = (np.cos(theta)**2)/(2*sigma_x**2) + (np.sin(theta)**2)/(2*sigma_y**2)
        b = -(np.sin(2*theta))/(4*sigma_x**2) + (np.sin(2*theta))/(4*sigma_y**2)
        c = (np.sin(theta)**2)/(2*sigma_x**2) + (np.cos(theta)**2)/(2*sigma_y**2)
        g = offset + amplitude*np.exp( - (a*((xy[0]-xo)**2) + 2*b*(xy[0]-xo)*(xy[1]-yo) + c*((xy[1]-yo)**2)))
        plt.imshow(g)
        plt.show()
        return g.ravel()  

    def _get_sample_codes(self):
        """
        Return a list containing the data filecodes.
        Overwrite the function in case the data needs to be filtered.
        By default we load all the filecodes we find in the datalegend
        :return:
        """
        fcs = np.arange(len(self.dl))
        ind = []
        ind_tool_detected = None
        ind_scene = None
        ind_outliers =[]
        if 'tool_detected' in self.filters:
            ind_tool_detected = np.where(self.dl['ToolDetected'])[0].tolist()
        else: 
            ind_tool_detected = list(fcs)
        if self.scene_name is not None:
            ind_scene = np.where(self.dl['Scene']==self.scene_name)[0].tolist()
        else:
            ind_scene = list(fcs)
        if 'trans_outliers' in self.filters:
            for i, fc in enumerate(fcs):
                dl_line = self.dl.iloc[fc]
                init_fc = dl_line['InitialFC']
                final_fc = dl_line['FinalFC']
                scene_name = dl_line['Scene']
                init_tf = self._get_tfs(init_fc, scene_name=scene_name, frame_id=self.tf_frame)
                final_tf = self._get_tfs(final_fc, scene_name=scene_name, frame_id=self.tf_frame)             
                init_tool_tf_wf = self._get_tfs(init_fc, scene_name=scene_name, frame_id='tool_frame')
                init_tool_tf_gf = self._get_relative_tf(init_tf, init_tool_tf_wf)
                init_tool_trans = init_tool_tf_gf[1:3]
                final_tool_tf_wf = self._get_tfs(final_fc, scene_name=scene_name, frame_id='tool_frame')
                final_tool_tf_gf = self._get_relative_tf(final_tf, final_tool_tf_wf)
                final_tool_trans = final_tool_tf_gf[1:3]
                if (np.linalg.norm(final_tool_trans - init_tool_trans) < self.trans_threshold):
                    ind_outliers.append(i)
        else:
            ind_outliers = list(fcs)
        ind = list(set(ind_tool_detected) & set(ind_scene) & set(ind_outliers))
        fcs = fcs[ind]
        return fcs


class BubblePivotingDownsampledDataset(BubblePivotingDataset):
    def __init__(self, *args, downsample_factor_x=5, downsample_factor_y=5, downsample_reduction='mean', **kwargs):
        self.downsample_factor_x = downsample_factor_x
        self.downsample_factor_y = downsample_factor_y
        self.downsample_reduction = downsample_reduction
        self.block_mean_downsampling_tr = BlockDownSamplingTr(factor_x=downsample_factor_x, factor_y=downsample_factor_y, reduction=self.downsample_reduction) #downsample all imprint values
        # add the block_mean_downsampling_tr to the tr list
        if 'transformation' in kwargs:
            if type(kwargs['transformation']) in (list, tuple):
                kwargs['transformation'] = list(kwargs['transformation']) + [self.block_mean_downsampling_tr]
            else:
                print('')
                raise AttributeError('Not supportes trasformations: {} type {}'.format(kwargs['transformation'], type(kwargs['transformation'])))
        else:
            kwargs['transformation'] = [self.block_mean_downsampling_tr]
        super().__init__(*args, **kwargs)

    @classmethod
    def get_name(self):
        return 'bubble_pivoting_downsampled_dataset'


class BubblePivotingDownsampledCombinedDataset(BubblePivotingDownsampledDataset):
    """"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @classmethod
    def get_name(self):
        return 'bubble_pivoting_downsampled_combined_dataset'

    def _get_sample_codes(self):
        # duplicate the filecodes:
        fcs = np.arange(2 * len(super()._get_sample_codes()))
        return fcs

    def _get_sample(self, indx):
        # fc: index of the line in the datalegend (self.dl) of the sample
        true_indx = indx // 2
        dl_line = self.dl.iloc[true_indx]
        sample = super()._get_sample(true_indx)
        if indx % 2 == 0:
            # sample is the initial
            sample['imprint'] = sample['init_imprint']
        else:
            sample['imprint'] = sample['final_imprint']
        return sample

# DEBUG:

if __name__ == '__main__':
    data_name = '/home/mireia/Documents/research/bubble_pivoting_data'
    dataset = BubblePivotingDataset(data_name=data_name)
    print('Dataset Name: ', dataset.name)
    print('Dataset Length:', len(dataset))
    sample_0 = dataset[0]
    print('Sample 0:', sample_0)
