import torch
import numpy as np
from manipulation_via_membranes.bubble_model_control.aux.bubble_model_control_utils import tr_frame, batched_matrix_to_euler_corrected
import pytorch3d.transforms as batched_trs
from manipulation_via_membranes.bubble_learning.aux.orientation_trs import EulerToAxis


def pivoting_action_model(state_samples, actions):       
    """
    ACTION MODEL FOR BubblePivotingEnv.
    Simulates the effects of an action to the tfs.
    :param state_samples: dictionary of batched states representing a sample of a state
    :param actions: batched actions to be applied to the state_sample
    :return:
    """ 
    # actions: tuple of tensors of shape (N, parameter dimension)
    num_actions = actions.shape[0] if len(actions.shape) > 1 else 1
    state_samples_corrected = state_samples
    # action_names = ['grasp_width', 'delta_y', 'delta_z', 'delta_roll']
    translations = torch.zeros_like(state_samples['all_tfs']['grasp_frame'][...,:3,3])
    translations[..., 1] = actions[..., 1] # delta_y
    translations[..., 2] = actions[..., 2] # delta_z
    rotations = torch.zeros_like(translations)
    rotations[..., 0] = actions[..., 3] # delta_roll
    all_tfs = state_samples_corrected['all_tfs'] # Tfs from world frame ('med_base') to the each of ten frame names
    
    # ### Debug ###
    # import copy
    # prev = copy.deepcopy(all_tfs['grasp_frame'])
    # ## Debug ###
    
    rigid_ee_frames = ['grasp_frame', 'med_kuka_link_ee', 'wsg50_finger_left', 'pico_flexx_left_link', 'pico_flexx_left_optical_frame', 'pico_flexx_right_link', 'pico_flexx_right_optical_frame', 'tool_frame']
    X_gf_wf = torch.zeros(4,4).unsqueeze(0).repeat_interleave(num_actions, dim=0).type(torch.double)
    X_gf_wf[...,:3,:3] = batched_trs.euler_angles_to_matrix(torch.index_select(-rotations, dim=-1, index=torch.LongTensor([2, 1, 0])), 'ZYX')
    X_gf_wf[...,:3,3] = translations
    gf_X_wf = torch.inverse(state_samples['all_tfs']['grasp_frame'])
    X_trans_gf = torch.einsum('kij,kjl->kil', gf_X_wf, X_gf_wf)
    X_trans_gf[...,3,3] = 1
    X_trans_gf[...,:3,:3] = torch.eye(3).unsqueeze(0).repeat_interleave(num_actions, dim=0).type(torch.double)
    X_gf_wf[...,3,3] = 1
    aux = torch.einsum('kij,kjl->kil', torch.inverse(state_samples['all_tfs']['grasp_frame']), X_gf_wf)
    X_rot_gf = torch.einsum('kij,kjl->kil', aux, state_samples['all_tfs']['grasp_frame'])
    X_rot_gf[...,:3,3] = 0
    all_tfs = tr_frame(all_tfs, 'grasp_frame', X_trans_gf, rigid_ee_frames)
    all_tfs = tr_frame(all_tfs, 'grasp_frame', X_rot_gf, rigid_ee_frames)
    
    # ### Debug ### 
    # curr = all_tfs['grasp_frame']
    # print('Translation ok', torch.norm(prev[...,1:3,3] + actions[...,1:3] - curr[...,1:3,3]) < 0.01)
    # import tf.transformations as tr
    
    # roll_prev = tr.euler_from_matrix(prev[0])
    # roll_curr = tr.euler_from_matrix(curr[0])
    # print('Rotation ok', torch.norm(roll_prev[0] + actions[0][-1] - roll_curr[0]) < 0.01 )
    # ### Debug ### 
    
    
    return state_samples_corrected

def pivoting_grasp_pose_correction(position, orientation, action):
    euler_to_axis = EulerToAxis()
    # Update position
    next_y = position[...,1] + action[..., 1] # y + delta_y
    next_z = position[...,2] + action[..., 2] # z + delta_z
    position_new = torch.stack((position[..., 0], next_y, next_z)).t()
    # Update orientation
    next_euler_sxyz = euler_to_axis.axis_angle_to_euler_sxyz(orientation)
    next_euler_sxyz[..., 0] += action[...,3] # current_roll + delta_roll
    orientation_axis_angle_new = euler_to_axis.euler_sxyz_to_axis_angle(next_euler_sxyz) 
    return position_new, orientation_axis_angle_new
