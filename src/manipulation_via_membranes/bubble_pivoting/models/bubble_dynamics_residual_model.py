
import torch
import torch.nn as nn
import pytorch_lightning as pl
import torchvision
import numpy as np
import os
import sys
import cv2

from manipulation_via_membranes.bubble_learning.models.aux.fc_module import FCModule
from manipulation_via_membranes.bubble_learning.aux.pose_loss import PlanarBoxPoseLoss



class PivotingBubbleDynamicsResidualModelEstimatedPose (pl.LightningModule):
    """
    Model designed to model the bubbles dynamics.
    Given s_t, and a_t, it produces ∆s, where s_{t+1} = s_t + ∆s
     * Here s_t is composed by:
        - Depth image from each of the bubbles
        - Wrench information
        - End effector pose: Position and orientation of the end effector
    * The depth images are embedded into a vector which is later concatenated with the wrench and pose information
    """
    def __init__(self, input_sizes, num_fcs=2, fc_h_dim=100, skip_layers=None, lr=1e-4, dataset_params=None, activation='relu'):
        super().__init__()
        self.input_sizes = input_sizes
        self.num_fcs = num_fcs
        self.fc_h_dim = fc_h_dim
        self.skip_layers = skip_layers
        self.lr = lr
        self.dataset_params = dataset_params
        self.activation = activation

        self.dyn_model = self._get_dyn_model()

        self.loss = None #TODO: Define the loss function
        self.mse_loss = nn.MSELoss()
        self.box_size = np.array([0.015, 0.06])
        self.pbp_loss = PlanarBoxPoseLoss(box_size=self.box_size)

        self.save_hyperparameters()

    @classmethod
    def get_name(cls):
        return 'bubble_dynamics_residual_model'

    @property
    def name(self):
        return self.get_name()


    def _get_dyn_model(self):
        sizes = self._get_sizes()
        trans_size = sizes['tool_trans']
        rot_size = sizes['tool_rot']
        wrench_size = sizes['wrench']
        pos_size = sizes['position']
        quat_size = sizes['orientation']
        pose_size = pos_size + quat_size
        action_size = sizes['action']
        td_size = sizes['tool_detected']
        force_size = sizes['max_force']
        dyn_output_size = wrench_size + pose_size + trans_size + rot_size
        dyn_input_size = dyn_output_size + action_size
        dyn_model_sizes = [dyn_input_size] + [self.fc_h_dim]*self.num_fcs + [dyn_output_size]
        dyn_model = FCModule(sizes=dyn_model_sizes, skip_layers=self.skip_layers, activation=self.activation)
        return dyn_model

    def forward(self, wrench, position, orientation, tool_trans, tool_rot, action):
        sizes = self._get_sizes()
        trans_size = sizes['tool_trans']
        rot_size = sizes['tool_rot']
        wrench_size = sizes['wrench']
        position_size = sizes['position']
        quat_size = sizes['orientation']
        td_size = sizes['tool_detected']
        force_size = sizes['max_force']
        dyn_input = torch.cat([wrench, position, orientation, tool_trans, tool_rot, action], dim=-1)
        dyn_output = self.dyn_model(dyn_input)
        wrench_delta, pos_delta, quat_delta, trans_delta, rotation_delta = torch.split(dyn_output, [wrench_size, position_size, quat_size, trans_size, rot_size], dim=-1)
        return wrench_delta, pos_delta, quat_delta, trans_delta, rotation_delta

    def _get_sizes(self):
        tool_trans_size = np.prod(self.input_sizes['init_tool_trans'])
        tool_rot_size = np.prod(self.input_sizes['init_tool_rot'])
        wrench_size = np.prod(self.input_sizes['init_wrench'])
        pose_size = np.prod(self.input_sizes['init_pos'])
        quat_size = + np.prod(self.input_sizes['init_quat'])
        action_size = np.prod(self.input_sizes['action'])
        tool_detected_size = np.prod(self.input_sizes['tool_detected'])
        max_force_size = np.prod(self.input_sizes['maximum_force_felt'])
        sizes = {'tool_trans': tool_trans_size,
                 'tool_rot': tool_rot_size,
                 'wrench': wrench_size,
                 'position': pose_size,
                 'orientation': quat_size,
                 'action': action_size,
                 'tool_detected': tool_detected_size,
                 'max_force': max_force_size
                 }
        return sizes

    def step(self, batch, batch_idx, phase):
        wrench_t = batch['init_wrench']
        pos_t = batch['init_pos']
        quat_t = batch['init_quat']
        trans_t = batch['init_tool_trans']
        rot_t = batch['init_tool_rot']
        td_t = batch['tool_detected']
        force_t= batch['maximum_force_felt']
        action = batch['action']
        wrench_d_gth = batch['delta_wrench']
        pos_d_gth = batch['delta_pos']
        quat_d_gth = batch['delta_quat']
        trans_d_gth = batch['delta_tool_trans']
        rot_d_gth = batch['delta_tool_rot']
        td_gth = batch['tool_detected']
        force_d_gth = batch['maximum_force_felt']

        wrench_delta, pos_delta, quat_delta, trans_delta, rotation_delta = self.forward(wrench_t, pos_t, quat_t, trans_t,
                                                                                        rot_t, action)

        loss =  self._compute_loss(wrench_delta, pos_delta, quat_delta, trans_delta, rotation_delta, wrench_d_gth, pos_d_gth, quat_d_gth, trans_d_gth, rot_d_gth)

        wrench_loss = self._compute_wrench_loss(wrench_delta, wrench_d_gth)

        height = 0.06 * 100 / 0.15
        width = 0.015 * 100 / 0.15
        images = []
        for i,_ in enumerate(trans_delta):
            img = np.zeros([100,100,3],dtype=np.uint8)
            img.fill(100)
            center_x_p = 50 + trans_delta[i][0] * 10 / 0.15
            center_y_p = 50 + trans_delta[i][1] * 10 / 0.15
            color_p = (255,0,0)
            self.draw_angled_rec(center_x_p, center_y_p, width, height, rotation_delta[i].item(), color_p, img)
            center_x_gth = 50 + trans_d_gth[i][0] * 10 / 0.15
            center_y_gth = 50 + trans_d_gth[i][1] * 10 / 0.15
            color_gth = (0,0,255)
            self.draw_angled_rec(center_x_gth, center_y_gth, width, height, rot_d_gth[i].item(), color_gth, img)
            img = torch.tensor(img)
            img = img.permute(2,0,1)
            images.append(img)

        grid = torchvision.utils.make_grid(images)
        self.logger.experiment.add_image('pose_estimation_{}'.format(phase), grid, self.global_step)

        self.log('batch', batch_idx)
        self.log('{}_loss'.format(phase), loss)
        self.log('{}_force_x_loss'.format(phase), wrench_loss[0])
        self.log('{}_force_y_loss'.format(phase), wrench_loss[1])
        self.log('{}_force_z_loss'.format(phase), wrench_loss[2])
        self.log('{}_torque_x_loss'.format(phase), wrench_loss[3])
        self.log('{}_torque_y_loss'.format(phase), wrench_loss[4])
        self.log('{}_torque_z_loss'.format(phase), wrench_loss[5])
        return loss

    def draw_angled_rec(self, x0, y0, width, height, angle, color, img):
        b = np.cos(angle) * 0.5
        a = np.sin(angle) * 0.5
        pt0 = (int(x0 - a * height - b * width),
            int(y0 + b * height - a * width))
        pt1 = (int(x0 + a * height - b * width),
            int(y0 - b * height - a * width))
        pt2 = (int(2 * x0 - pt0[0]), int(2 * y0 - pt0[1]))
        pt3 = (int(2 * x0 - pt1[0]), int(2 * y0 - pt1[1]))

        cv2.line(img, pt0, pt1, color, 3)
        cv2.line(img, pt1, pt2, color, 3)
        cv2.line(img, pt2, pt3, color, 3)
        cv2.line(img, pt3, pt0, color, 3)

    def validation_step(self, val_batch, batch_idx):
        return self.step(val_batch, batch_idx, 'validation')

    def training_step(self, train_batch, batch_idx):
        return self.step(train_batch, batch_idx, 'training')

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def _compute_loss(self, wrench_delta, pos_delta, quat_delta, trans_delta, rotation_delta,
                      wrench_d_gth, pos_d_gth, quat_d_gth, trans_d_gth, rot_d_gth): # TODO: Add inputs
        pose_1 = torch.cat((trans_delta, rotation_delta), dim=-1)
        pose_2 = torch.cat((trans_d_gth, rot_d_gth), dim=-1)
        pose_loss = self.pbp_loss(pose_1, pose_2)
        # trans_d_zeros = torch.zeros_like(trans_delta)
        # trans_d_gth_zeros = torch.zeros_like(trans_d_gth)
        # pose_rot_1 = torch.cat((trans_d_zeros, rotation_delta), dim=-1)
        # pose_rot_2 = torch.cat((trans_d_gth_zeros, rot_d_gth), dim=-1)
        # pose_rot_loss = self.pbp_loss(pose_rot_1, pose_rot_2)
        trans_loss = self.mse_loss(quat_delta, quat_d_gth)
        rot_loss = self.mse_loss(pos_delta, pos_d_gth)
        wrench_loss = self.mse_loss(wrench_delta, wrench_d_gth)
        loss = pose_loss + wrench_loss/100


        return loss

    def _compute_wrench_loss(self, wrench_delta, wrench_d_gth):
        wrench_loss = np.zeros(6)
        for i in np.arange(6):
            wrench_loss[i] = self.mse_loss(wrench_delta[i], wrench_d_gth[i])
        return wrench_loss

