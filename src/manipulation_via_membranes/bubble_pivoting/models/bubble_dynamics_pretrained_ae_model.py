import pdb

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import matplotlib.pyplot as plt
from matplotlib import cm
import torchvision
import numpy as np
import os
import sys
from copy import deepcopy

from manipulation_via_membranes.bubble_learning.models.aux.fc_module import FCModule
from manipulation_via_membranes.bubble_learning.models.bubble_dynamics_model import BubbleDynamicsModel


class PivotingBubbleDynamicsPretrainedAEModel(BubbleDynamicsModel):
    """
    Model designed to model the bubbles dynamics.
    Given s_t, and a_t, it produces ∆s, where s_{t+1} = s_t + ∆s
     * Here s_t is composed by:
        - Depth image from each of the bubbles
    * The depth images are embedded into a vector which is later concatenated with the wrench and pose information
    """

    @classmethod
    def get_name(cls):
        return 'bubble_dynamics_pivoting_pretrained_autoencoder_model'

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
        dyn_output_size = self.img_embedding_size + wrench_size
        dyn_input_size = dyn_output_size + action_size + pose_size
        dyn_model_sizes = [dyn_input_size] + [self.fc_h_dim]*self.num_fcs + [dyn_output_size]
        dyn_model = FCModule(sizes=dyn_model_sizes, skip_layers=self.skip_layers, activation=self.activation)
        return dyn_model

    def query(self, imprint, wrench, pos, ori, action):
        sizes = self._get_sizes()
        # Encode imprint
        if self.load_norm:
            imprint_input_emb = self.autoencoder.img_encoder(imprint)
        else:
            imprint_input_emb = self.autoencoder.encode(imprint)

        state_dyn_input = torch.cat([imprint_input_emb, wrench, pos, ori], dim=-1)
        dyn_input = torch.cat([state_dyn_input, action], dim=-1)
        output_dyn_delta = self.dyn_model(dyn_input)
        delta_imprint_output_emb, delta_wrench = torch.split(output_dyn_delta, (self.img_embedding_size, sizes['wrench']), dim=-1)
        imprint_output_emb = imprint_input_emb + delta_imprint_output_emb
        wrench_next = wrench + delta_wrench

        # Decode imprint
        if self.load_norm:
            imprint_next = self.autoencoder.img_decoder(imprint_output_emb)
        else:
            imprint_next = self.autoencoder.decode(imprint_output_emb)

        return imprint_next, wrench_next

    def forward(self, imprint, wrench, pos, ori, action):
        if self.load_norm:
            # apply the normalization to the imprint input, and then to the output, because the model has been trained on the normalized space.
            imprint_t = self.autoencoder._norm_imprint(imprint)
            imprint_next, wrench_next = self.query(imprint_t, wrench, pos, ori, action)
            imprint_next = self.autoencoder._denorm_imprint(imprint_next)
        else:
            imprint_next, wrench_next = self.query(imprint, action)
        return imprint_next, wrench_next

    def _get_sizes(self):
        imprint_size = self.input_sizes['init_imprint']
        wrench_size = np.prod(self.input_sizes['init_wrench'])
        pose_size = np.prod(self.input_sizes['init_pos'])
        orientation_size = np.prod(self.input_sizes['init_quat'])
        action_size = np.prod(self.input_sizes['action'])
        sizes = {'imprint': imprint_size,
                 'wrench': wrench_size,
                 'position': pose_size,
                 'orientation': orientation_size,
                 'action': action_size
                 }
        return sizes 

    def _get_state_items(self):
        input_items = ['imprint', 'wrench', 'position', 'orientation']
        return input_items
    
    def _step(self, batch, batch_idx, phase='train'):
        imprint_t = batch['init_imprint']
        imprint_next = batch['final_imprint']
        wrench_t = batch['init_wrench']
        position_t = batch['init_pos']
        orientation_t = batch['init_quat']
        imprint_next = batch['final_imprint']
        wrench_next = batch['final_wrench']
        action = batch['action']
        if self.load_norm:
            # normalize the tensors
            imprint_t = self.autoencoder._norm_imprint(imprint_t)
            imprint_next = self.autoencoder._norm_imprint(imprint_next)

        imprint_next_rec, wrench_next_rec = self.query(imprint_t, wrench_t, position_t, orientation_t, action)

        loss = self._compute_loss(imprint_next_rec, imprint_next)
        self.log('{}_batch'.format(phase), batch_idx)
        self.log('{}_loss'.format(phase), loss)

        if batch_idx == 0:
            if self.current_epoch == 0:
                init_grid = self._get_image_grid(imprint_t, imprint_t.detach())
                gth_grid = self._get_image_grid(imprint_next, imprint_next.detach())
                self.logger.experiment.add_image('init_imprint_{}'.format(phase), init_grid, self.global_step)
                self.logger.experiment.add_image('next_imprint_gt_{}'.format(phase), gth_grid, self.global_step)
            predicted_grid = self._get_image_grid(imprint_next_rec, imprint_next_rec.detach())
            self.logger.experiment.add_image('next_imprint_predicted_{}'.format(phase), predicted_grid, self.global_step)
        return loss

    def _get_image_grid(self, batched_img, ref_img=None, cmap='jet'):
        # reshape the batched_img to have the same imprints one above the other
        batched_img = batched_img.detach().cpu()
        batched_img_r = batched_img.reshape(*batched_img.shape[:1], -1, *batched_img.shape[3:]) # (batch_size, 2*W, H)
        # Apply cmap and add padding
        padding_pixels = 3
        batched_img_cmap = self._cmap_tensor(batched_img_r, ref_img, cmap=cmap) # size (..., w,h, 3)
        num_dims = len(batched_img_cmap.shape)
        batched_img_cmap_p = batched_img_cmap.permute(*np.arange(num_dims-3), -1, -3, -2)
        batched_img_padded = F.pad(input=batched_img_cmap_p,
                                   pad=(padding_pixels, padding_pixels, padding_pixels, padding_pixels),
                                   mode='constant',
                                   value=0)
        grid_img = torchvision.utils.make_grid(batched_img_padded)
        return grid_img

    def _cmap_tensor(self, img_tensor, ref_img=None, cmap='jet'):
        cmap = cm.get_cmap(cmap)
        mapped_img_ar = cmap(self._normalize_img(img_tensor, ref_img))  # (..,w,h,4)
        mapped_img_ar = mapped_img_ar[..., :3] # (..,w,h,3) -- get rid of the alpha value
        mapped_img = torch.tensor(mapped_img_ar).to(self.device)
        return mapped_img

    def _normalize_img(self, img_tensor, ref_img=None):
        if ref_img is None:
            ref_img = deepcopy(img_tensor)
        img_tensor = img_tensor - torch.min(ref_img)
        img_tensor = img_tensor/(torch.max(ref_img) - torch.min(ref_img))
        return img_tensor

    def _compute_loss(self, imprint_rec, imprint_gth):
        imprint_reconstruction_loss = self.mse_loss(imprint_rec, imprint_gth)
        loss = imprint_reconstruction_loss
        return loss