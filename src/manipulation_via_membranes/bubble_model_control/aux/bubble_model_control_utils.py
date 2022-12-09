import abc
import torch
import numpy as np
import copy
import tf.transformations as tr
import pytorch3d.transforms as batched_trs


def convert_all_tfs_to_tensors(all_tfs):
    """
    Convert a DataFrame object containing the tfs with respect a common frame into a dictionary of tensor transformation matrices
    :param all_tfs: DataFrame
    :return:
    """
    # Transform a DF into a dictionary of homogeneous transformations matrices (4x4)
    converted_all_tfs = {}
    parent_frame = all_tfs['parent_frame'][0] # Assume that are all teh same
    child_frames = all_tfs['child_frame']
    converted_all_tfs[parent_frame] = np.eye(4) # Transformation to itself is the identity
    all_poses = all_tfs[['x', 'y', 'z', 'qx', 'qy', 'qz', 'qw']]
    for i, child_frame_i in enumerate(child_frames):
        pose_i = all_poses.iloc[i]
        X_i = tr.quaternion_matrix(pose_i[3:])
        X_i[:3, 3] = pose_i[:3]
        converted_all_tfs[child_frame_i] = X_i
    return converted_all_tfs


def tr_frame(all_tfs, frame_name, X, fixed_frame_names):
    # Apply tf to the frame_name and modify all other tf for the fixed frames to that tf frame
    # all_tfs: dict of tfs
    # frame_name: str for the frame to apply X
    # X (aka fn_X_fn_new) trasformation to be applied along frame_name
    # fixed_frame_names: list of strs containing the names of the frames that we also need to transform because they are rigid to the frame_name frame.
    new_tfs = {}
    w_X_fn = all_tfs[frame_name]
    w_X_fn_new = w_X_fn @ X
    new_tfs[frame_name] = w_X_fn_new
    # Apply the new transformation to all fixed frames
    for ff_i in fixed_frame_names:
        if ff_i == frame_name:
            continue
        w_X_ffi = all_tfs[ff_i]
        fn_X_ffi = get_transformation_matrix(all_tfs, source_frame=frame_name, target_frame=ff_i)
        w_X_ffi_new = w_X_fn_new @ fn_X_ffi
        new_tfs[ff_i] = w_X_ffi_new
    all_tfs.update(new_tfs)
    return all_tfs


def get_transformation_matrix(all_tfs, source_frame, target_frame):
    w_X_sf = all_tfs[source_frame]
    w_X_tf = all_tfs[target_frame]
    sf_X_w = torch.linalg.inv(w_X_sf)
    sf_X_tf = sf_X_w @ w_X_tf
    return sf_X_tf


def batched_tensor_sample(sample, batch_size=None, device=None):
    # sample is a dictionary of
    if device is None:
        device = torch.device('cpu')
    batched_sample = {}
    for k_i, v_i in sample.items():
        if type(v_i) is dict:
            batched_sample_i = batched_tensor_sample(v_i, batch_size=batch_size, device=device)
            batched_sample[k_i] = batched_sample_i
        elif type(v_i) is np.ndarray:
            batched_sample_i = torch.tensor(v_i).to(device)
            if batch_size is not None:
                batched_sample_i = batched_sample_i.unsqueeze(0).repeat_interleave(batch_size, dim=0)
            batched_sample[k_i] = batched_sample_i
        elif type(v_i) in [int, float]:
            batched_sample_i = torch.tensor([v_i]).to(device)
            if batch_size is not None:
                batched_sample_i = batched_sample_i.unsqueeze(0).repeat_interleave(batch_size, dim=0)
            batched_sample[k_i] = batched_sample_i
        elif type(v_i) is torch.Tensor:
            batched_sample_i = v_i.to(device)
            if batch_size is not None:
                batched_sample_i = batched_sample_i.unsqueeze(0).repeat_interleave(batch_size, dim=0)
            batched_sample[k_i] = batched_sample_i.to(device)
        else:
            batched_sample[k_i] = v_i
    return batched_sample

def batched_matrix_to_euler_corrected(batched_matrix):
    # Transform a batched matrix into euler angles with 'sxyz' convention
    euler_unordered = batched_trs.matrix_to_euler_angles(batched_matrix, 'ZYX')
    euler_ordered = torch.index_select(euler_unordered, dim=-1, index=torch.LongTensor([2, 1, 0]))
    return euler_ordered