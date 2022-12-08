import numpy as np
import torch
import pytorch3d.transforms as batched_trs

from arc_utilities.tf2wrapper import TF2Wrapper


def get_angle_difference(parent_axis=np.array([0,0,-1]), child_axis=np.array([0,0,-1])):
    parent_axis = parent_axis/np.linalg.norm(parent_axis)
    child_axis = child_axis/np.array([np.linalg.norm(child_axis, axis=-1)]).T
    cos_angle = np.matmul(child_axis, parent_axis)
    angle = np.arccos(cos_angle)
    mask = np.ones_like(angle)
    if len(mask.shape) == 0:
        mask = np.array([mask])
    mask[np.where(np.cross(parent_axis, child_axis)[...,0] < 0)] = -1
    angle = angle * mask     
    return angle

def check_goal_position(goal_angle, orientation_tol):
    tool_axis = get_tool_axis(tool_axis=np.array([0,0,1]), ref_frame='grasp_frame')
    tool_axis_wf = get_tool_axis(tool_axis=np.array([0,0,1]), ref_frame='med_base')
    goal_axis = np.array([0, np.sin(-goal_angle), np.cos(-goal_angle)])
    if tool_axis_wf[2] > 0:
        tool_axis *= -1
    tool_angle_gf = get_angle_difference(parent_axis=np.array([0, 0, 1]), child_axis=tool_axis)
    # print('Real tool_angle_gf: ', tool_angle_gf)
    # print('Difference until the goal angle (deg): ', (goal_angle-tool_angle_gf)*180/np.pi)   
    return abs(goal_angle-tool_angle_gf) < orientation_tol, goal_angle-tool_angle_gf

def check_goal_position_from_estimated_pose(goal_angle, orientation_tol, estimated_pose_wf, state_sample):
    tool_angle_gf = get_tool_angle_gf(estimated_pose_wf, state_sample)
    return (abs(goal_angle-tool_angle_gf) < orientation_tol).item(), (goal_angle-tool_angle_gf).item()

def get_tool_axis(tool_axis=np.array([0,0,1]), ref_frame='med_base'):
    tf2_listener = TF2Wrapper()
    tool_frame = tf2_listener.get_transform(parent=ref_frame, child='tool_frame')
    tool_axis = tool_frame[:3,:3] @ tool_axis # in the reference frame
    return tool_axis

def get_tool_angle_gf(estimated_pose_wf, state_samples):
    estimated_poses_wxyz = torch.index_select(estimated_pose_wf[:, 3:], dim=-1, index=torch.LongTensor([3, 0, 1, 2]))
    wf_X_tf =  batched_trs.quaternion_to_matrix(estimated_poses_wxyz)
    gf_X_wf = torch.inverse(state_samples['all_tfs']['grasp_frame'][...,:3,:3]).type(torch.float)
    if len(gf_X_wf.shape) < 3:
        gf_X_wf = gf_X_wf.unsqueeze(0)
    gf_X_tf = torch.bmm(gf_X_wf, wf_X_tf)
    tool_axis_gf = gf_X_tf @ torch.tensor([0,0,1.0])
    
    tool_axis_wf = wf_X_tf @ torch.tensor([0,0,1.0])
    # Get rid of ambiguity assuming that the tool faces always down 
    mask = (tool_axis_wf[..., 2] < 0) * 2 - 1
    tool_axis_gf *= mask.unsqueeze(-1)
    
    tool_angle_gf = get_angle_difference(child_axis=tool_axis_gf.detach().numpy(), parent_axis=np.array([0, 0, 1]))  
    tool_angle_gf = torch.from_numpy(tool_angle_gf) 
    # ### DEBUG ###
    # tool_axis_gf_list = get_tool_axis(tool_axis=np.array([0,0,1]), ref_frame='grasp_frame')
    # tool_axis_wf = torch.inverse(wf_X_tf) @ torch.tensor([0,0,1.0])
    # angle_diff = get_angle_difference(child_axis=tool_axis_gf.detach().numpy(), parent_axis=np.array([0, 0, 1]))
    # angle_diff_list = get_angle_difference(child_axis=tool_axis_gf_list, parent_axis=np.array([0, 0, 1]))
    # if angle_diff.shape[0] < 2:
    #     print('Angle mismatch in degrees: ', np.rad2deg(angle_diff_list-angle_diff))
    #     print('Tool angle gf: ', tool_angle_gf)
    #     print('Tool angle gf from listener: ', angle_diff_list)
    
    # ### DEBUG ###
    
    return tool_angle_gf

def get_tool_translation_gf(estimated_pose_wf, state_samples):
    translation_wf = torch.zeros_like(estimated_pose_wf[..., :4])
    translation_wf[...,:3] = estimated_pose_wf[:, :3]
    gf_X_wf = torch.inverse(state_samples['all_tfs']['grasp_frame'][...,:3,:3]).type(torch.float)
    translation_gf = torch.bmm(gf_X_wf, translation_wf)[...,:3]
    return translation_gf
    