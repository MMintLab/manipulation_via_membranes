import numpy as np
from bubble_tools.bubble_tools.bubble_img_tools import process_bubble_img
from manipulation_via_membranes.bubble_learning.aux.orientation_trs import QuaternionToAxis


def format_observation_sample(obs_sample):
    formatted_obs_sample = {}
    # add imprints: -------
    init_imprint_r = obs_sample['bubble_depth_img_right_reference'] - obs_sample['bubble_depth_img_right']
    init_imprint_l = obs_sample['bubble_depth_img_left_reference'] - obs_sample['bubble_depth_img_left']
    formatted_obs_sample['init_imprint'] = process_bubble_img(np.stack([init_imprint_r, init_imprint_l], axis=0))[...,0]
    wrench_frames = [w.header.frame_id for w in obs_sample['wrench']]
    wrench_indx = wrench_frames.index('med_base')
    formatted_obs_sample['init_wrench'] = np.array([obs_sample['wrench'][wrench_indx].wrench.force.x,
                                            obs_sample['wrench'][wrench_indx].wrench.force.y,
                                            obs_sample['wrench'][wrench_indx].wrench.force.z,
                                            obs_sample['wrench'][wrench_indx].wrench.torque.x,
                                            obs_sample['wrench'][wrench_indx].wrench.torque.y,
                                            obs_sample['wrench'][wrench_indx].wrench.torque.z])
    formatted_obs_sample['init_pos'] = obs_sample['tfs'][obs_sample['tfs']['child_frame'] == 'grasp_frame'][['x','y','z']].values[0]
    quaternion = obs_sample['tfs'][obs_sample['tfs']['child_frame'] == 'grasp_frame'][['qx','qy','qz','qw']].values[0]
    quat_to_axis = QuaternionToAxis()
    formatted_obs_sample['init_quat'] = quat_to_axis._tr(quaternion)
    formatted_obs_sample['init_object_pose'] = np.array([0, 0, 0, 0, 0, 0])
    key_map = {
        'tfs': 'all_tfs',
        'bubble_camera_info_depth_left': 'camera_info_l',
        'bubble_camera_info_depth_right': 'camera_info_r',
        'bubble_depth_img_right_reference': 'undef_depth_r',
        'bubble_depth_img_left_reference': 'undef_depth_l',
        'object_model': 'object_model'
    }
    for k_old, k_new in key_map.items():
        formatted_obs_sample[k_new] = obs_sample[k_old]
    
    # apply the key_map
    return formatted_obs_sample