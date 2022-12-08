
import torch
import torch.nn as nn
import pytorch_lightning as pl
import torchvision
import numpy as np
import torch.nn.functional as F
from matplotlib import cm
import os
import sys
import cv2
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

from manipulation_via_membranes.bubble_learning.models.aux.fc_module import FCModule
from manipulation_via_membranes.bubble_pivoting.fg_loss import FittedGaussianPoseLoss



class PivotingBubbleDynamicsResidualModelFittedGaussian(pl.LightningModule):
    """
    Model designed to model the bubbles dynamics.
    Given s_t, and a_t, it produces ∆s, where s_{t+1} = s_t + ∆s
     * Here s_t is composed by:
        - Depth image from each of the bubbles
        - Wrench information
        - End effector pose: Position and orientation of the end effector
    * The depth images are embedded into a vector which is later concatenated with the wrench and pose information
    """
    def __init__(self, input_sizes, num_fcs=2, fc_h_dim=100, skip_layers=None, lr=1e-4, dataset_params=None, activation='relu', n_points=20):
        super().__init__()
        self.input_sizes = input_sizes
        self.fitted_gaussian_size = 14
        self.num_fcs = num_fcs
        self.fc_h_dim = fc_h_dim
        self.skip_layers = skip_layers
        self.lr = lr
        self.dataset_params = dataset_params
        self.activation = activation

        self.dyn_model = self._get_dyn_model()

        self.loss = None #TODO: Define the loss function
        self.n_points = n_points
        self.mse_loss = nn.MSELoss()
        self.save_hyperparameters()

    @classmethod
    def get_name(cls):
        return 'bubble_dynamics_residual_model'

    @property
    def name(self):
        return self.get_name()


    def _get_dyn_model(self):
        sizes = self._get_sizes()
        wrench_size = sizes['wrench']
        pos_size = sizes['position']
        quat_size = sizes['orientation']
        pose_size = pos_size + quat_size
        action_size = sizes['action']
        td_size = sizes['tool_detected']
        force_size = sizes['max_force']
        fitted_gaussian_size = sizes['fitted_gaussian']
        dyn_output_size = wrench_size + pose_size + fitted_gaussian_size
        dyn_input_size = dyn_output_size + action_size
        dyn_model_sizes = [dyn_input_size] + [self.fc_h_dim]*self.num_fcs + [dyn_output_size]
        dyn_model = FCModule(sizes=dyn_model_sizes, skip_layers=self.skip_layers, activation=self.activation)
        return dyn_model

    def forward(self, fitted_gaussian_input, wrench, position, orientation, action):
        sizes = self._get_sizes()
        wrench_size = sizes['wrench']
        position_size = sizes['position']
        quat_size = sizes['orientation']
        td_size = sizes['tool_detected']
        force_size = sizes['max_force']
        fitted_gaussian_size = sizes['fitted_gaussian']
        dyn_input = torch.cat([fitted_gaussian_input, wrench, position, orientation, action], dim=-1)
        dyn_output = self.dyn_model(dyn_input)
        fitted_gaussian_delta, wrench_delta, pos_delta, quat_delta = torch.split(dyn_output, [fitted_gaussian_size, wrench_size, position_size, quat_size], dim=-1)
        return fitted_gaussian_delta, wrench_delta, pos_delta, quat_delta

    def _get_sizes(self):
        tool_trans_size = np.prod(self.input_sizes['init_tool_trans'])
        tool_rot_size = np.prod(self.input_sizes['init_tool_rot'])
        wrench_size = np.prod(self.input_sizes['init_wrench'])
        pose_size = np.prod(self.input_sizes['init_pos'])
        quat_size = + np.prod(self.input_sizes['init_quat'])
        action_size = np.prod(self.input_sizes['action'])
        tool_detected_size = np.prod(self.input_sizes['tool_detected'])
        max_force_size = np.prod(self.input_sizes['maximum_force_felt'])
        fitted_gaussian_size = self.fitted_gaussian_size
        sizes = {'fitted_gaussian': fitted_gaussian_size,
                 'tool_trans': tool_trans_size,
                 'tool_rot': tool_rot_size,
                 'wrench': wrench_size,
                 'position': pose_size,
                 'orientation': quat_size,
                 'action': action_size,
                 'tool_detected': tool_detected_size,
                 'max_force': max_force_size
                 }
        return sizes

    def _step(self, batch, batch_idx, phase):
        imprint_t = batch['init_imprint']
        #imprint_t_img = batch['init_imprint_img']
        self.imprint_shape = imprint_t.shape
        self.fg_loss = FittedGaussianPoseLoss(self.imprint_shape[2:], n_points=self.n_points)
        wrench_t = batch['init_wrench']
        pos_t = batch['init_pos']
        quat_t = batch['init_quat']
        td_t = batch['tool_detected']
        force_t= batch['maximum_force_felt']
        action = batch['action']
        final_imprint = batch['final_imprint']
        #final_imprint_img = batch['final_imprint_img']
        wrench_d_gth = batch['delta_wrench']
        pos_d_gth = batch['delta_pos']
        quat_d_gth = batch['delta_quat']
        td_gth = batch['tool_detected']
        force_d_gth = batch['maximum_force_felt']
        fitted_gaussian_input = batch['fitted_gaussian_init'] 
        fitted_gaussian_out_gth = batch['fitted_gaussian_final']
        init_grid = self._get_image_grid(imprint_t, [fitted_gaussian_input])
        predicted_gaussian, wrench_delta, pos_delta, quat_delta = self.forward(fitted_gaussian_input, wrench_t, pos_t, quat_t, action)
        loss = self._compute_loss(predicted_gaussian , wrench_delta, pos_delta, quat_delta, 
                                  fitted_gaussian_out_gth, wrench_d_gth, pos_d_gth, quat_d_gth, final_imprint)

        # wrench_loss = self._compute_wrench_loss(wrench_delta, wrench_d_gth)
        final_grid = self._get_image_grid(final_imprint, [fitted_gaussian_out_gth, predicted_gaussian])
        if self.global_step == 0:
            self.logger.experiment.add_image('init_grid_{}'.format(phase), init_grid, self.global_step)       
        if self.global_step % 10 == 0:
            self.logger.experiment.add_image('final_grid_{}'.format(phase), final_grid, self.global_step)

        self.log('batch', batch_idx)
        self.log('{}_loss'.format(phase), loss)
        # self.log('{}_force_x_loss'.format(phase), wrench_loss[0])
        # self.log('{}_force_y_loss'.format(phase), wrench_loss[1])
        # self.log('{}_force_z_loss'.format(phase), wrench_loss[2])
        # self.log('{}_torque_x_loss'.format(phase), wrench_loss[3])
        # self.log('{}_torque_y_loss'.format(phase), wrench_loss[4])
        # self.log('{}_torque_z_loss'.format(phase), wrench_loss[5])
        return loss

    # def _get_image_grid(self, batched_img):
    #     # swap the axis so the grid is (batch_size, num_channels, h, w)
    #     # batched_img = batched_img.permute(0,2,3,1)
    #     # batched_img_arr = np.array(batched_img)
    #     # images = []
    #     # for batch_i in batched_img_arr:
    #     #     image = []
    #     #     for image_i in batch_i:
    #     #         fig = plt.figure()
    #     #         plt.imshow(image_i, interpolation='nearest')
    #     #         canvas = FigureCanvas(fig)
    #     #         ax = fig.gca()
    #     #         ax.axis('off')
    #     #         canvas.draw()
    #     #         buf = canvas.tostring_rgb()
    #     #         image_i = np.frombuffer(buf, dtype='uint8')
    #     #         w, h = canvas.get_width_height()
    #     #         image_i = np.fromstring(buf, dtype=np.uint8).reshape(h, w, 3)
    #     #         image.append(image_i)
    #     #         plt.close()
    #     #     images.append(np.concatenate(image))
    #     # images_tensor = torch.tensor(images).to(self.device)
    #     # images_tensor = images_tensor.permute(0,3,1,2)
    #     images_tensor = torch.tensor(batched_img)
    #     grid_img = torchvision.utils.make_grid(images_tensor)
    #     return grid_img

    # def _get_image_grid(self, batched_img, fitted_gaussian):
    #     # swap the axis so the grid is (batch_size, num_channels, h, w)
    #     batched_img = batched_img.permute(0,3,1,2)
    #     # batched_img = batched_img.permute(0,4,2,3,1)
    #     # desired_shape = [x for x in batched_img.shape]
    #     # desired_shape[-2] *= batched_img.shape[-1]
    #     # desired_shape = desired_shape[:-1]
    #     # for i, img in enumerate(batched_img):
    #     #     plt.imshow(img)
    #     grid_img = torchvision.utils.make_grid(batched_img)
    #     return grid_img

    # def _get_image_grid(self, batched_img, fitted_gaussian, cmap='jet'):
    #     # reshape the batched_img to have the same imprints one above the other
    #     batched_img = batched_img.detach().cpu()
    #     # Add gaussian
    #     cnt = 0
    #     points_r = np.array((self.fg_loss.twoD_Gaussian_points(fitted_gaussian[:,:7], 40)).type(torch.int))
    #     points_l = np.array((self.fg_loss.twoD_Gaussian_points(fitted_gaussian[:,7:], 40)).type(torch.int))
    #     for j, batch in enumerate(batched_img):
    #         for i, bubble in enumerate(batch):
    #             if i == 0:
    #                 for point in points_r[j]:
    #                     batched_img[j, i, point[0], point[1]] = 2e-2
    #                     cnt += 1
    #             if i == 1:
    #                 for point in points_l[j]:
    #                     batched_img[j, i, point[0], point[1]] = 2e-2
    #                     cnt += 1



    #     print('Counter:', cnt)
    #     batched_img_r = batched_img.reshape(*batched_img.shape[:1],-1,*batched_img.shape[3:]) # (batch_size, 2*W, H)
    #     # Add padding
    #     padding_pixels = 5
    #     batched_img_padded = F.pad(input=batched_img_r,
    #                                pad=(padding_pixels, padding_pixels, padding_pixels, padding_pixels),
    #                                mode='constant',
    #                                value=0)
    #     batched_img_cmap = self._cmap_tensor(batched_img_padded, cmap=cmap) # size (..., w,h, 3)
    #     num_dims = len(batched_img_cmap.shape)
    #     grid_input = batched_img_cmap.permute(*np.arange(num_dims-3), -1, -3, -2)
    #     grid_img = torchvision.utils.make_grid(grid_input)
    #     return grid_img





    def _get_image_grid(self, batched_img, fitted_gaussians, cmap='jet'):
        # reshape the batched_img to have the same imprints one above the other
        batched_img = batched_img.detach().cpu()
        img_shape = batched_img.shape[-2:]
        # Add gaussian
        batched_img_cmap = self._cmap_tensor(batched_img, cmap=cmap) # size (..., w,h, 3)
        for k, fg in enumerate(fitted_gaussians):
            points_r = (self.fg_loss.twoD_Gaussian_points(fg[:,:7], 100)).type(torch.int)
            points_l = (self.fg_loss.twoD_Gaussian_points(fg[:,7:], 100)).type(torch.int)
            for j, batch in enumerate(batched_img):
                for i, bubble in enumerate(batch):
                    if i == 0:
                        for point in points_r[j]:
                            if point[0] < img_shape[0] and point[0] > 0 and point[1] < img_shape[1] and point[1] > 0:
                                batched_img_cmap[j, i, point[0], point[1], :] = 0.0
                                batched_img_cmap[j, i, point[0], point[1], k] = 1.0
                    if i == 1:
                        for point in points_l[j]:
                            if point[0] < img_shape[0] and point[0] > 0 and point[1] < img_shape[1] and point[1] > 0:
                                batched_img_cmap[j, i, point[0], point[1], :] = 0.0
                                batched_img_cmap[j, i, point[0], point[1], k] = 1.0
        batched_img_r = batched_img_cmap.reshape(*batched_img_cmap.shape[:1],-1,*batched_img_cmap.shape[3:]) # (batch_size, 2*W, H)
        # Add padding
        padding_pixels = 5
        # batched_img_padded = F.pad(input=batched_img_r,
        #                            pad=(padding_pixels, padding_pixels, padding_pixels, padding_pixels),
        #                            mode='constant',
        #                            value=0)
        num_dims = len(batched_img_r.shape)
        grid_input = batched_img_r.permute(*np.arange(num_dims-3), -1, -3, -2)
        grid_img = torchvision.utils.make_grid(grid_input)
        return grid_img





    def _cmap_tensor(self, img_tensor, cmap='jet'):
        cmap = cm.get_cmap(cmap)
        mapped_img_ar = cmap(img_tensor/torch.max(img_tensor)) # (..,w,h,4)
        mapped_img_ar = mapped_img_ar[..., :3] # (..,w,h,3) -- get rid of the alpha value
        mapped_img = torch.tensor(mapped_img_ar).to(self.device)
        return mapped_img

    def validation_step(self, val_batch, batch_idx):
        return self._step(val_batch, batch_idx, 'validation')

    def training_step(self, train_batch, batch_idx):
        return self._step(train_batch, batch_idx, 'training')

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def _compute_loss(self, fitted_gaussian_predicted, wrench_delta, pos_delta, quat_delta, 
                      fitted_gaussian_out_gth, wrench_d_gth, pos_d_gth, quat_d_gth, final_imprint):
        wrench_loss = self.mse_loss(wrench_delta, wrench_d_gth)
        #fitted_gaussian_loss = self.fg_loss(fitted_gaussian_predicted, fitted_gaussian_out_gth)
        mean_loss_1 = self.mse_loss(fitted_gaussian_predicted[:,1:3], fitted_gaussian_out_gth[:,1:3])
        mean_loss_2 = self.mse_loss(fitted_gaussian_predicted[:,8:10], fitted_gaussian_out_gth[:,1:3])
        param_loss = self.mse_loss(fitted_gaussian_predicted, fitted_gaussian_out_gth)
        prob_loss_1 = self.probabilistic_loss(fitted_gaussian_predicted[:,:7], final_imprint[:,0])
        prob_loss_2 = self.probabilistic_loss(fitted_gaussian_predicted[:,7:], final_imprint[:,1])
        loss = prob_loss_1 + prob_loss_2
        return loss

    def probabilistic_loss(self, fitted_gaussian_predicted, final_imprint):
        batch_size = fitted_gaussian_predicted.shape[0]
        img_shape = final_imprint.shape[1:]
        # Create grid
        x = np.linspace(0, img_shape[1]-1, img_shape[1])
        y = np.linspace(0, img_shape[0]-1, img_shape[0])
        x, y = (torch.tensor(np.meshgrid(x, y))).to(self.device)
        x = torch.tile(x,(batch_size,1,1))
        y = torch.tile(y,(batch_size,1,1))
        # Load parameters
        amplitude = fitted_gaussian_predicted[:,0]
        xo = fitted_gaussian_predicted[:,1]
        yo = fitted_gaussian_predicted[:,2]
        sigma_x = fitted_gaussian_predicted[:,3]
        sigma_y = fitted_gaussian_predicted[:,4]
        theta = fitted_gaussian_predicted[:,5]
        offset = fitted_gaussian_predicted[:,6]

        amplitude = (torch.tile(amplitude, (img_shape[0],img_shape[1],1))).permute(2,0,1)
        xo = (torch.tile(xo, (img_shape[0],img_shape[1],1))).permute(2,0,1)
        yo = (torch.tile(yo, (img_shape[0],img_shape[1],1))).permute(2,0,1)
        sigma_x = (torch.tile(sigma_x, (img_shape[0],img_shape[1],1))).permute(2,0,1)
        sigma_y = (torch.tile(sigma_y, (img_shape[0],img_shape[1],1))).permute(2,0,1)
        theta = (torch.tile(theta, (img_shape[0],img_shape[1],1))).permute(2,0,1)
        offset = (torch.tile(offset, (img_shape[0],img_shape[1],1))).permute(2,0,1)

        A = (torch.cos(theta)**2)/(2*sigma_x**2) + (torch.sin(theta)**2)/(2*sigma_y**2)
        B = -(torch.sin(2*theta))/(4*sigma_x**2) + (torch.sin(2*theta))/(4*sigma_y**2)
        C = (torch.sin(theta)**2)/(2*sigma_x**2) + (torch.cos(theta)**2)/(2*sigma_y**2)
        # Calculate gaussian
        g = amplitude*torch.exp( - (A*((x-xo)**2) + 2*B*(x-xo)*(y-yo) + C*((y-yo)**2)))

        loss = self.mse_loss(torch.norm(final_imprint-g, dim=-1), torch.norm(g-g, dim=-1))
        return loss



    def _compute_wrench_loss(self, wrench_delta, wrench_d_gth):
        wrench_loss = np.zeros(6)
        for i in np.arange(6):
            wrench_loss[i] = self.mse_loss(wrench_delta[i], wrench_d_gth[i])
        return wrench_loss

