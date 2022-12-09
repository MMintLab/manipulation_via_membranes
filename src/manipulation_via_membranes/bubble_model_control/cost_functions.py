import numpy as np
import torch
import rospy
import pytorch3d.transforms as batched_tr


def only_position_cost_function(estimated_poses, states, prev_states, actions):
    # Only position ----------------------------------------
    goal_xyz = torch.zeros(3)
    estimated_xyz = estimated_poses[:, :3]
    cost = torch.linalg.norm(estimated_xyz-goal_xyz, axis=1)
    return cost


def vertical_tool_cost_function(estimated_poses, states, prev_states, actions):
    # Only position ----------------------------------------
    # goal_xyz = np.zeros(3)
    # estimated_xyz = estimated_poses[:, :3]
    # cost = np.linalg.norm(estimated_xyz-goal_xyz, axis=1)

    # Only orientation, using model points ------------
    # tool axis is z, so we want tool frame z axis to be aligned with the world z axis
    estimated_pos = estimated_poses[:, :3]  # (x, y, z)
    estimated_q = estimated_poses[:, 3:]  # (qx,qy,qz,qw)
    estimated_qwxyz = torch.index_select(estimated_q, dim=-1, index=torch.LongTensor([3, 0, 1, 2]))  # (qw, qx,qy,qz)
    estimated_R = batched_tr.quaternion_to_matrix(
        estimated_qwxyz)  # careful! batched_tr quat is [qw,qx,qy,qz], we work as [qx,qy,qz,qw]
    z_axis = torch.tensor([0., 0, 1.]).unsqueeze(0).repeat_interleave(estimated_R.shape[0], dim=0).float()
    tool_z_axis_wf = torch.einsum('kij,kj->ki', estimated_R, z_axis)
    ori_cost = 1 - torch.abs(torch.einsum('ki,ki->k', z_axis,
                                          tool_z_axis_wf))  # TO MINIMIZE (perfect case tool axis parallel to z axis, aka dot(z_axis, tool_z_axis)=1)
    ori_cost = torch.nan_to_num(ori_cost, nan=10.0)
    is_nan_action = torch.any(torch.isnan(actions), dim=1)
    is_nan_pose = torch.any(torch.isnan(estimated_pos), dim=1)
    cost = ori_cost + 10 * is_nan_action + 10 * is_nan_pose
    return cost*100