import trimesh
import torch
import os
import numpy as np
from mmint_camera_utils.ros_utils.publisher_wrapper import PublisherWrapper
from sensor_msgs.msg import  PointCloud2
import sensor_msgs.point_cloud2 as pc2
import pytorch3d.transforms as batched_trs
import tf.transformations as tr
from std_msgs.msg import Header
from manipulation_via_membranes.bubble_pivoting.pivoting_model_control.aux.pivoting_geometry import check_goal_position, get_angle_difference, get_tool_angle_gf

package_path = project_path = os.path.join(os.path.dirname(os.path.abspath(__file__)).split('/bubble_pivoting')[0], 'bubble_pivoting')

class CostFunction(object):
    def __init__(self, policy):
        self.policy = policy
        self._contact = None
        self._goal_angle = None
        self._pivoting_angle = None
        self._env = None
        self.marker_name = 'cylinder_model'
        self.model_mesh = trimesh.load(os.path.join(package_path, 'markers/{}.{}'.format(self.marker_name, 'stl')))
        self.model_pc, _ = trimesh.sample.sample_surface(self.model_mesh, count=1000)
        self.model_pc = torch.tensor(self.model_pc)
        self.next_model_pc_publisher = PublisherWrapper(topic_name='next_model_pc', msg_type=PointCloud2)

    def __call__(self, estimated_poses, states, prev_states, actions):
        return self.pivoting_cost_function(estimated_poses, states, prev_states, actions)

        
    @property
    def contact(self):
        return self._contact

    @contact.setter
    def contact(self, value):
        self._contact = value
            
    @property
    def pivoting_angle(self):
        return self._pivoting_angle

    @pivoting_angle.setter
    def pivoting_angle(self, value):
        self._pivoting_angle = value
            
    @property
    def goal_angle(self):
        return self._goal_angle

    @goal_angle.setter
    def goal_angle(self, value):
        self._goal_angle = value

    @property
    def env(self):
        return self._env

    @env.setter
    def env(self, value):
        self._env = value 

    def pivoting_cost_function(self, estimated_poses, state_samples, prev_states, actions):
        piv_coef = 10000
        height_coef = 30000
        roll_coef = 20000
        
        # TODO: penalize if the next imprint has low amount of points
        max_def, _ = torch.max(state_samples['final_imprint'].flatten(start_dim=1), dim=1)
        
        # Penalize overshooting: angle difference adds double cost if there is overshooting
        tool_angle_gf =  get_tool_angle_gf(estimated_poses, state_samples)
        _, diff = check_goal_position(self._goal_angle, 0.05)
        overshot = (torch.sign(self._goal_angle - tool_angle_gf) != torch.sign(torch.from_numpy(diff))).float()
        # delta_roll = actions[...,3]
        cost = torch.abs(self._goal_angle - tool_angle_gf) * (overshot+1.)
        
        
        wf_grasp_frame = state_samples['all_tfs']['grasp_frame']
        gripper_height = wf_grasp_frame[...,2,3]
        invalid_height = gripper_height < 0.06        
        euler = batched_trs.matrix_to_euler_angles(wf_grasp_frame[...,:3,:3], 'ZYX')
        euler_sxyz = torch.index_select(euler, dim=-1, index=torch.LongTensor([2,1,0]))
        roll = euler_sxyz[..., 0] % (2*np.pi)
        invalid_roll = torch.logical_or(roll < np.pi/2, roll > 3*np.pi/2)
        
        # Checking whether it is potentially a pivoting motion (it pushes against the table)
        wf_tool_frame_prev = prev_states['all_tfs']['tool_frame'][0]
        wf_grasp_frame_prev = prev_states['all_tfs']['grasp_frame'][0]
        tool_axis_wf_prev = torch.inverse(wf_tool_frame_prev[:3,:3]) @ np.array([0,0,1])
        transformed_object = torch.einsum('ij,lj->li',wf_tool_frame_prev[:3,:3], self.model_pc) + wf_tool_frame_prev[:3,3].unsqueeze(0)
        h, _ = torch.min(transformed_object[:,2], axis=0)
        cos_angle = np.dot(tool_axis_wf_prev, np.array([0,0,1]))
        dist = -h/cos_angle  
        object_adjusted = transformed_object + dist * tool_axis_wf_prev
        initial_pc = object_adjusted   
        movement_wf = torch.zeros((actions.shape[0], 3))
        movement_wf[..., 1] = actions[...,1] # delta_y
        movement_wf[..., 2] = actions[...,2] # delta_z
        rotation_euler = torch.zeros_like(movement_wf)
        rotation_euler[..., 0] = actions[...,3] # delta_roll
        rotation_matrices = batched_trs.euler_angles_to_matrix(torch.index_select(-rotation_euler, dim=-1, index=torch.LongTensor([2, 1, 0])), 'ZYX')
        initial_pc_origin = initial_pc - wf_grasp_frame_prev[:3,3]
        transformed_pc_origin = torch.einsum('kij,lj->kli', rotation_matrices.type(torch.double), initial_pc_origin.type(torch.double))
        transformed_model_pc = transformed_pc_origin + wf_grasp_frame_prev[:3,3] + movement_wf.unsqueeze(1)
        min_z, _ = torch.min(transformed_model_pc[:,:,2], dim=1)
        action_not_pivoting = min_z > 0
        
        cost += piv_coef * action_not_pivoting + height_coef * invalid_height + roll_coef * invalid_roll
        if min(cost) > 100000:
            print('Warning: tool will be lifted')
        # pc_header = Header()
        # pc_header.frame_id = 'world'
        # transformed_model_pc_i = pc2.create_cloud_xyz32(pc_header, initial_pc_origin)
        # self.next_model_pc_publisher.data = transformed_model_pc_i
        return cost
