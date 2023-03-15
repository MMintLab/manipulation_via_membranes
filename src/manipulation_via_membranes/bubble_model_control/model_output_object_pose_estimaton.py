import numpy as np
import torch
import pytorch3d.transforms as batched_trs
import einops
from abc import abstractmethod

from manipulation_via_membranes.bubble_learning.aux.img_trs.block_upsampling_tr import BlockUpSamplingTr
from manipulation_via_membranes.aux.load_confs import load_bubble_reconstruction_params, load_object_models
from bubble_tools.bubble_tools.bubble_img_tools import unprocess_bubble_img

from bubble_drawing.bubble_pose_estimation.batched_pytorch_icp import icp_2d_masked, pc_batched_tr
from mmint_tools.camera_tools.img_utils import project_depth_image
from mmint_tools.camera_tools.pointcloud_utils import get_projection_tr, project_pc
from bubble_utils.bubble_tools.bubble_pc_tools import get_imprint_mask
from manipulation_via_membranes.bubble_learning.aux.load_model import load_model_version
from manipulation_via_membranes.bubble_learning.models.icp_approximation_model import ICPApproximationModel, FakeICPApproximationModel
from manipulation_via_membranes.bubble_learning.aux.orientation_trs import QuaternionToAxis


class ModelOutputObjectPoseEstimationBase(object):
    def __init__(self):
        pass

    def estimate_pose(self, sample):
        # upsample the imprints
        estimated_pose = self._estimate_pose(sample)
        return estimated_pose

    @abstractmethod
    def _estimate_pose(self, sample):
        # Return estimated object pose [x, y, z, qx, qy, qz, qw]
        pass


class BatchedModelOutputObjectPoseEstimationBase(ModelOutputObjectPoseEstimationBase):
    def _estimate_pose(self, batched_sample):
        """
        Estimate the object pose from the imprints using icp 2D. We compute it in parallel on batched operations
        :param sample: The sample is expected to be batched, i.e. all values (batch_size, original_size_1, ..., original_size_n)
        :return:
        """
        all_tfs = batched_sample['all_tfs']

        gf_X_objpose = self._estimate_object_pose(batched_sample)

        # Compute object pose in world frame
        wf_X_gf = self._get_transformation_matrix(all_tfs, 'med_base', 'grasp_frame').type(gf_X_objpose.dtype)
        wf_X_objpose = wf_X_gf @ gf_X_objpose

        # convert it to pose format [xs, ys, zs, qxs, qyx, qzs, qws]
        estimated_pos = wf_X_objpose[..., :3, 3]
        _estimated_quat = batched_trs.matrix_to_quaternion(wf_X_objpose[..., :3, :3]) # (qw,qx,qy,qz)
        estimated_quat = torch.index_select(_estimated_quat, dim=-1, index=torch.LongTensor([1, 2, 3, 0]))# (qx,qy,qz,qw)
        estimated_poses = torch.cat([estimated_pos, estimated_quat], dim=-1)
        return estimated_poses

    def _get_transformation_matrix(self, all_tfs, source_frame, target_frame):
        w_X_sf = all_tfs[source_frame]
        w_X_tf = all_tfs[target_frame]
        sf_X_w = torch.linalg.inv(w_X_sf)
        sf_X_tf = sf_X_w @ w_X_tf
        return sf_X_tf

    @abstractmethod
    def _estimate_object_pose(self, batched_sample):
        # returns the objec_pose estimation on the grasp frame.
        # Pose encoded as a 4x4 homogeneous transforamtion gf_X_objpose.
        # output_shape: (batched_size, 4, 4)
        pass


def axis_angle_pose_to_homogeneous_pose(axis_angle_pose):
    hom_pos = torch.zeros(axis_angle_pose.shape[:-1] + (4,4), dtype=axis_angle_pose.dtype, device=axis_angle_pose.device)
    hom_pos[..., 3, 3] = 1
    hom_pos[..., :3, 3] = axis_angle_pose[...,:3]
    hom_pos[..., :3, :3] = batched_trs.axis_angle_to_matrix(axis_angle_pose[...,3:])
    return hom_pos

def homogeneous_pose_to_axis_angle(homogeneous_pose):
    pos = homogeneous_pose[..., :3, 3]
    quat_wxyz = batched_trs.matrix_to_quaternion(homogeneous_pose[..., :3, :3])
    quat_xyzw = torch.index_select(quat_wxyz, dim=-1, index=torch.LongTensor([1, 2, 3, 0]))
    axis_angle = QuaternionToAxis._tr(quat_xyzw)
    axis_angle_pose = torch.cat([pos, axis_angle], dim=-1)
    return axis_angle_pose

class End2EndModelOutputObjectPoseEstimation(BatchedModelOutputObjectPoseEstimationBase):
    """
    ObjectPoseEstimation for pose-to-pose dynamics model
    """
    def _estimate_object_pose(self, sample):
        estimated_pose_axis_angle_gf = sample['final_object_pose']  # final_object_pose in grasp frame. It is also encoded as axis-angle
        gf_X_objpose =  axis_angle_pose_to_homogeneous_pose(estimated_pose_axis_angle_gf)
        return gf_X_objpose


class ICPApproximationModelOutputObjectPoseEstimation(BatchedModelOutputObjectPoseEstimationBase):
    """
    Approximates the ICP estimation with a learned NN model. This targets faster pose estimation.
    """
    def __init__(self, *args, model_name='icp_approximation_model', load_version=0, model_data_path=None, **kwargs):
        self.model_name = model_name
        self.load_version = load_version
        self.model_data_path = model_data_path
        super().__init__(*args, **kwargs)
        self.icp_approx_model = self._load_icp_approx_model()

    def _estimate_object_pose(self, batched_sample):
        predicted_imprint = batched_sample['final_imprint']
        estimated_pose_axis_angle_gf = self.icp_approx_model(predicted_imprint)
        gf_X_objpose = axis_angle_pose_to_homogeneous_pose(estimated_pose_axis_angle_gf)
        return gf_X_objpose

    def _load_icp_approx_model(self):
        Model = self._get_Model(self.model_name)
        if self.model_name in ['fake_icp_approximation_model']:
            icp_approx_model = Model() # fake models do not need loading
        else:
            icp_approx_model = load_model_version(Model, self.model_data_path, self.load_version)
        return icp_approx_model

    def _get_Model(self, model_name):
        all_models = [ICPApproximationModel, FakeICPApproximationModel]
        all_model_names = [m.get_name() for m in all_models]
        if not model_name in all_model_names:
            raise AttributeError('Model name for ICPApproximationModelOutputObjectPoseEstimation Not Supported.\n We support: {} (provided: {})'.format(all_model_names, model_name))
        model_indx = all_model_names.index(model_name)
        Model = all_models[model_indx]
        return Model


class BatchedModelOutputObjectPoseEstimation(BatchedModelOutputObjectPoseEstimationBase):
    """ ICP POSE ESTIMATION. Work with pytorch tensors"""
    def __init__(self, *args, device=None, imprint_selection='threshold', imprint_percentile=0.1, object_name='marker', factor_x=1, factor_y=1, method='bilinear', **kwargs):
        self.imprint_selection = imprint_selection
        self.imprint_percentile = imprint_percentile
        if device is None:
            device = torch.device('cpu')
        self.device = device
        self.object_name = object_name
        self.reconstruction_params = load_bubble_reconstruction_params()
        self.object_params = self.reconstruction_params[self.object_name]
        self.imprint_threshold = self.object_params['imprint_th']['depth']
        self.icp_threshold = self.object_params['icp_th']
        self.block_upsample_tr = BlockUpSamplingTr(factor_x=factor_x, factor_y=factor_y, method=method,
                                                   keys_to_tr=['final_imprint'])
        super().__init__(*args, **kwargs)
        self.model_pcs = load_object_models()

    def _upsample_sample(self, sample):
        # Upsample output
        sample_up = self.block_upsample_tr(sample)
        return sample_up

    def _estimate_object_pose(self, batched_sample_raw):
        batched_sample = self._upsample_sample(batched_sample_raw)
        all_tfs = batched_sample['all_tfs']

        # Get imprints from sample
        predicted_imprint = batched_sample['final_imprint']
        imprint_pred_r = predicted_imprint[:, 0]
        imprint_pred_l = predicted_imprint[:, 1]

        # unprocess the imprints (add padding to move them back to the original shape)
        imprint_pred_r = unprocess_bubble_img(imprint_pred_r.unsqueeze(-1)).squeeze(-1)  # ref frame:  -- (N, w, h)
        imprint_pred_l = unprocess_bubble_img(imprint_pred_l.unsqueeze(-1)).squeeze(-1)  # ref frame:  -- (N, w, h)
        imprint_frame_r = 'pico_flexx_right_optical_frame'
        imprint_frame_l = 'pico_flexx_left_optical_frame'

        depth_ref_r = batched_sample['undef_depth_r'].squeeze(-1)  # (N, w, h)
        depth_ref_l = batched_sample['undef_depth_l'].squeeze(-1)  # (N, w, h)
        depth_def_r = depth_ref_r - imprint_pred_r  # CAREFUL: Imprint is defined as undef_depth_img - def_depth_img
        depth_def_l = depth_ref_l - imprint_pred_l  # CAREFUL: Imprint is defined as undef_depth_img - def_depth_img

        # Project imprints to get point coordinates
        Ks_r = batched_sample['camera_info_r']['K']
        Ks_l = batched_sample['camera_info_l']['K']
        pc_r = project_depth_image(depth_def_r, Ks_r)  # (N, w, h, n_coords) -- n_coords=3
        pc_l = project_depth_image(depth_def_l, Ks_l)  # (N, w, h, n_coords) -- n_coords=3

        # Convert imprint point coordinates to grasp frame
        gf_X_ifr = self._get_transformation_matrix(all_tfs, 'grasp_frame', imprint_frame_r)
        gf_X_ifl = self._get_transformation_matrix(all_tfs, 'grasp_frame', imprint_frame_l)
        pc_shape = pc_r.shape
        pc_r_gf = pc_batched_tr(pc_r.view((pc_shape[0], -1, pc_shape[-1])), gf_X_ifr[..., :3, :3],
                                gf_X_ifr[..., :3, 3]).view(pc_shape)
        pc_l_gf = pc_batched_tr(pc_l.view((pc_shape[0], -1, pc_shape[-1])), gf_X_ifl[..., :3, :3],
                                gf_X_ifl[..., :3, 3]).view(pc_shape)
        pc_gf = torch.stack([pc_r_gf, pc_l_gf], dim=1)  # (N, n_impr, w, h, n_coords)

        # Load object model model
        model_pc = np.asarray(self.model_pcs[self.object_name].points)
        model_pc = torch.tensor(model_pc).to(predicted_imprint.device)

        # Project points to 2d
        projection_axis = (1, 0, 0)
        projection_tr = torch.tensor(get_projection_tr(projection_axis))  # (4,4)
        pc_gf_projected = project_pc(pc_gf, projection_axis)  # (N, n_impr, w, h, n_coords)
        pc_gf_2d = pc_gf_projected[..., :2]  # only 2d coordinates
        pc_model_projected = project_pc(model_pc, projection_axis).unsqueeze(0).repeat_interleave(pc_gf.shape[0], dim=0)

        # Apply ICP 2d
        num_iterations = 20
        pc_scene = pc_gf_2d  # pc_scene: (N, n_impr, w, h, n_coords)
        # Compute mask -- filter out points
        depth_ref = torch.stack([depth_ref_r, depth_ref_l], dim=1)  # (N, n_impr, w, h)
        depth_def = torch.stack([depth_def_r, depth_def_l], dim=1)  # (N, n_impr, w, h)
        pc_scene_mask = self._get_pc_mask(depth_def, depth_ref)
        pc_scene_mask = pc_scene_mask.unsqueeze(-1).repeat_interleave(2, dim=-1)  # (N, n_impr, w, h, n_coords)
        pc_model_projected_2d = pc_model_projected[..., :2]  # pc_model: (N, n_model_points, n_coords)

        # Apply ICP:
        device = self.device
        pc_model_projected_2d = self._filter_model_pc(pc_model_projected_2d)
        pc_scene, pc_scene_mask = self._filter_scene_pc(pc_scene, pc_scene_mask)
        # print(torch.sum(pc_scene_mask.reshape(pc_scene_mask.shape[0], -1), dim=1)) # report number of points per scene
        pc_model_projected_2d = pc_model_projected_2d.type(torch.float).to(device)  # This call takes almost 2 sec
        pc_scene = pc_scene.type(torch.float).to(device)
        pc_scene_mask = pc_scene_mask.to(device)

        Rs, ts = icp_2d_masked(pc_model_projected_2d, pc_scene, pc_scene_mask, num_iter=num_iterations)
        Rs = Rs.cpu()
        ts = ts.cpu()
        # Obtain object pose in grasp frame
        projected_ic_tr = torch.zeros(ts.shape[:-1] + (4, 4))
        projected_ic_tr[..., :2, :2] = Rs
        projected_ic_tr[..., :2, 3] = ts
        projected_ic_tr[..., 2, 2] = 1
        projected_ic_tr[..., 3, 3] = 1
        projection_tr = projection_tr.type(torch.float)
        unproject_tr = torch.linalg.inv(projection_tr)
        gf_X_objpose = torch.einsum('ji,kil->kjl', unproject_tr,
                                    torch.einsum('kij,jl->kil', projected_ic_tr, projection_tr))
        return gf_X_objpose

    def _filter_model_pc(self, model_pc):
        # model_pc (N, num_model_points, space_dim)
        model_pc = model_pc[:, ::20, :] # TODO: Find a better way to downsample the model
        return model_pc

    def _filter_scene_pc(self, pc_scene, pc_scene_mask):
        if self.imprint_selection == 'threshold':
            pc_scene = pc_scene[:, :, ::5, ::5, :]
            pc_scene_mask = pc_scene_mask[:, :, ::5, ::5, :]
            pc_scene = einops.rearrange(pc_scene, 'N i w h c -> N (i w h) c')
            pc_scene_mask = einops.rearrange(pc_scene_mask, 'N i w h c -> N (i w h) c')
        elif self.imprint_selection == 'percentile':
            original_shape = pc_scene.shape
            # when filtering using the percentile, since we select a constant number of points per imprint pair, we can filter out the points not belonging to the, since all batches will have the same number of poiints
            pc_scene = einops.rearrange(pc_scene, 'N i w h c -> N (i w h) c')
            pc_scene_mask = einops.rearrange(pc_scene_mask, 'N i w h c -> N (i w h) c').to(torch.bool)
            # select the masked points
            pc_scene = torch.masked_select(pc_scene, pc_scene_mask).reshape(original_shape[0], -1, original_shape[-1])  # (N, num_points, c)
            pc_scene_mask = torch.masked_select(pc_scene_mask, pc_scene_mask).reshape(original_shape[0], -1, original_shape[-1])  # (N, num_points, c)
        else:
            return NotImplementedError(
                'Imprint selection method {} not implemented yet.'.format(self.imprint_selection))

        return pc_scene, pc_scene_mask

    def _get_pc_mask(self, depth_def, depth_ref):
        # depth_def: (N, n_impr, w, h)
        # depth_ref: (N, n_impr, w, h)
        if self.imprint_selection == 'threshold':
            imprint_threshold = self.imprint_threshold
            pc_scene_mask = torch.tensor(get_imprint_mask(depth_ref, depth_def, imprint_threshold))
        elif self.imprint_selection == 'percentile':
            # select the points with larger deformation. In total will select top self.imprint_percentile*100%
            delta_depth = einops.rearrange(depth_ref - depth_def, 'N n w h -> N (n w h)')
            num_points = np.prod(depth_def.shape[1:])
            k = int(np.floor(self.imprint_percentile * num_points))
            top_k_vals, top_k_indxs = torch.topk(delta_depth, k, dim=1)
            pc_scene_mask = torch.zeros_like(delta_depth)
            pc_scene_mask = pc_scene_mask.scatter(1, top_k_indxs, torch.ones_like(top_k_vals))
            pc_scene_mask = pc_scene_mask.reshape(depth_ref.shape)
            # TODO: Add some check in case that there is no points deformed. Check it by a mimimum deform threshold. Retrun a mask of zeros in that case.
        else:
            return NotImplementedError('Imprint selection method {} not implemented yet.'.format(self.imprint_selection))

        return pc_scene_mask




























# TODO: Fix the class below:
# class ModelOutputObjectPoseEstimation(ModelOutputObjectPoseEstimationBase):
#
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.reconstructor = self._get_reconstructor()
#
#     def _get_reconstructor(self):
#         reconstructor = BubblePCReconstructorOfflineDepth(threshold=self.imprint_threshold,
#                                                           object_name=self.object_name,
#                                                           estimation_type='icp2d')
#         return reconstructor
#
#     def _estimate_pose(self, sample):
#         camera_info_r = sample['camera_info_r']
#         camera_info_l = sample['camera_info_l']
#         all_tfs = sample['all_tfs']
#         # obtain reference (undeformed) depths
#         ref_depth_img_r = sample['undef_depth_r'].squeeze()
#         ref_depth_img_l = sample['undef_depth_l'].squeeze()
#
#         predicted_imprint = sample['final_imprint']
#         imprint_pred_r, imprint_pred_l = predicted_imprint
#
#         # unprocess the imprints (add padding to move them back to the original shape)
#         imprint_pred_r = unprocess_bubble_img(np.expand_dims(imprint_pred_r, -1)).squeeze(-1)
#         imprint_pred_l = unprocess_bubble_img(np.expand_dims(imprint_pred_l, -1)).squeeze(-1)
#
#         deformed_depth_r = ref_depth_img_r - imprint_pred_r  # CAREFUL: Imprint is defined as undef_depth_img - def_depth_img
#         deformed_depth_l = ref_depth_img_l - imprint_pred_l
#
#         # THIS hacks the ways to obtain data for the reconstructor
#         ref_depth_img_l = np.expand_dims(ref_depth_img_l, -1)
#         ref_depth_img_r = np.expand_dims(ref_depth_img_r, -1)
#         deformed_depth_r = np.expand_dims(deformed_depth_r, -1)
#         deformed_depth_l = np.expand_dims(deformed_depth_l, -1)
#         self.reconstructor.references['left'] = ref_depth_img_l
#         self.reconstructor.references['right'] = ref_depth_img_r
#         self.reconstructor.depth_r = {'img': deformed_depth_r, 'frame': 'pico_flexx_right_optical_frame'}
#         self.reconstructor.depth_l = {'img': deformed_depth_l, 'frame': 'pico_flexx_left_optical_frame'}
#         self.reconstructor.camera_info['right'] = camera_info_r
#         self.reconstructor.camera_info['left'] = camera_info_l
#         # compute transformations from camera frames to grasp frame and transform the
#         self.reconstructor.add_tfs(all_tfs)
#         # estimate pose
#         estimated_pose = self.reconstructor.estimate_pose(self.icp_threshold)
#         # transform it to pos, quat instead of matrix
#         estimated_pos = estimated_pose[:3,3]
#         estimated_quat = tr.quaternion_from_matrix(estimated_pose)
#         estimated_pose = np.concatenate([estimated_pos, estimated_quat])
#         return estimated_pose




