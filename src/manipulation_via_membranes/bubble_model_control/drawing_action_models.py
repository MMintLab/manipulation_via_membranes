import abc
import torch
import numpy as np
import copy
import tf.transformations as tr
import pytorch3d.transforms as batched_trs

from manipulation_via_membranes.bubble_model_control.aux.bubble_model_control_utils import batched_tensor_sample, get_transformation_matrix, tr_frame, convert_all_tfs_to_tensors
from manipulation_via_membranes.bubble_learning.aux.orientation_trs import QuaternionToAxis


def drawing_one_dir_grasp_pose_correction(position, orientation, action):
    # NOTE: Orientations can either be quaternions or axis-angle
    # position: 3d position with reference on the med_base of the grasp_frame
    # orientation: orientation of the grasp_frame with respect to the med_base
    # action: ['rotation', 'length', 'grasp_width'] to be commanded to the robot
    #   - rotation: rotation along the gripper x axis in rad.
    #   - length: movement along the drawing direction (intersection of the med_base_pane and the plane perpendicular to the grasp_frame x_axis.
    #   - grasp_width: adjustement of the grasp width (no needed here)
    action_names = ['rotation', 'length', 'grasp_width']
    ori_quat = QuaternionToAxis._tr_inv(orientation)
    x_axis = torch.tensor([1,0,0], dtype=orientation.dtype, device=orientation.device).unsqueeze(0).repeat_interleave(orientation.shape[0], dim=0)
    axis_rot = x_axis
    axis_angle_rot = action[..., 0].unsqueeze(-1).repeat_interleave(3,dim=-1)*axis_rot
    q_rot = QuaternionToAxis._tr_inv(axis_angle_rot)
    q_next = batched_trs.quaternion_multiply(q_rot, ori_quat)
    orientation_next = QuaternionToAxis._tr(q_next)
    z_axis = torch.tensor([0,0,1], dtype=orientation.dtype, device=orientation.device).unsqueeze(0).repeat_interleave(orientation.shape[0], dim=0)
    w_R_gf = batched_trs.axis_angle_to_matrix(orientation)
    grasp_plane_normal_wf = torch.einsum('kij,kj->ki', w_R_gf, x_axis)
    moving_axis = torch.cross(z_axis, grasp_plane_normal_wf)
    position_delta = action[..., 1:2] * moving_axis
    position_next = position + position_delta
    return position_next, orientation_next


def drawing_action_model_one_dir(state_samples, actions):
    """
    ACTION MODEL FOR BubbleOneDirectionDrawingEnv.
    Simulates the effects of an action to the tfs.
    :param state_samples: dictionary of batched states representing a sample of a state
    :param actions: batched actions to be applied to the state_sample
    :return:
    """
    state_samples_corrected = state_samples
    action_names = ['rotation', 'length', 'grasp_width']
    rotations = actions[..., 0]
    lengths = actions[..., 1]
    grasp_widths = actions[..., 2] * 0.001  # the action grasp width is in mm
    # Rotation is a rotation about the x axis of the grasp_frame
    # Length is a translation motion of length 'length' of the grasp_frame on the xy med_base plane along the intersection with teh yz grasp frame plane
    # grasp_width is the width of the
    all_tfs = state_samples_corrected['all_tfs']  # Tfs from world frame ('med_base') to the each of teh frame names
    frame_names = all_tfs.keys()

    rigid_ee_frames = ['grasp_frame', 'med_kuka_link_ee', 'wsg50_finger_left', 'pico_flexx_left_link',
                       'pico_flexx_left_optical_frame', 'pico_flexx_right_link', 'pico_flexx_right_optical_frame']
    wf_X_gf = all_tfs['grasp_frame']
    # Move Gripper:
    # (move wsg_50_finger_{right,left} along x direction)
    gf_X_fl = get_transformation_matrix(all_tfs, 'grasp_frame', 'wsg50_finger_left')
    gf_X_fr = get_transformation_matrix(all_tfs, 'grasp_frame', 'wsg50_finger_right')
    X_finger_left = torch.eye(4).unsqueeze(0).repeat_interleave(actions.shape[0], dim=0).type(torch.double)
    X_finger_right = torch.eye(4).unsqueeze(0).repeat_interleave(actions.shape[0], dim=0).type(torch.double)
    current_half_width_l = -gf_X_fl[..., 0, 3] - 0.009
    current_half_width_r = gf_X_fr[..., 0, 3] - 0.009
    X_finger_left[..., 0, 3] = -(0.5 * grasp_widths - current_half_width_l).type(torch.double)
    X_finger_right[..., 0, 3] = -(0.5 * grasp_widths - current_half_width_r).type(torch.double)
    all_tfs = tr_frame(all_tfs, 'wsg50_finger_left', X_finger_left,
                       ['pico_flexx_left_link', 'pico_flexx_left_optical_frame'])
    all_tfs = tr_frame(all_tfs, 'wsg50_finger_right', X_finger_right,
                       ['pico_flexx_right_link', 'pico_flexx_right_optical_frame'])
    # Move Grasp frame on the plane amount 'length; and rotate the Grasp frame along x direction a 'rotation'  amount
    rot_axis = torch.tensor([1, 0, 0]).unsqueeze(0).repeat_interleave(actions.shape[0], dim=0).type(torch.double)
    angle_axis = rotations.unsqueeze(-1).repeat_interleave(3, dim=-1) * rot_axis
    X_gf_rot = torch.eye(4).unsqueeze(0).repeat_interleave(actions.shape[0], dim=0).type(torch.double)
    X_gf_rot[..., :3, :3] = batched_trs.axis_angle_to_matrix(angle_axis)  # rotation along x axis
    # compute translation
    z_axis = torch.tensor([0, 0, 1]).unsqueeze(0).repeat_interleave(actions.shape[0], dim=0).type(torch.double)
    y_dir_gf = torch.tensor([0, -1, 0]).unsqueeze(0).repeat_interleave(actions.shape[0], dim=0).type(torch.double)
    y_dir_wf = torch.einsum('kij,kj->ki', wf_X_gf[..., :3, :3], y_dir_gf)
    y_dir_wf_perp = torch.einsum('ki,ki->k', y_dir_wf, z_axis).unsqueeze(-1).repeat_interleave(3, dim=-1) * z_axis
    drawing_dir_wf = y_dir_wf - y_dir_wf_perp
    drawing_dir_wf = drawing_dir_wf / torch.linalg.norm(drawing_dir_wf, dim=1).unsqueeze(-1).repeat_interleave(3,
                                                                                                               dim=-1)  # normalize
    drawing_dir_gf = torch.einsum('kij,kj->ki', torch.linalg.inv(wf_X_gf[..., :3, :3]), drawing_dir_wf)
    trans_gf = lengths.unsqueeze(-1).repeat_interleave(3, dim=-1) * drawing_dir_gf
    X_gf_trans = torch.eye(4).unsqueeze(0).repeat_interleave(actions.shape[0], dim=0).type(torch.double).type(
        torch.double)
    X_gf_trans[..., :3, 3] = trans_gf
    all_tfs = tr_frame(all_tfs, 'grasp_frame', X_gf_trans, rigid_ee_frames)
    all_tfs = tr_frame(all_tfs, 'grasp_frame', X_gf_rot, rigid_ee_frames)
    state_samples_corrected['all_tfs'] = all_tfs

    return state_samples_corrected