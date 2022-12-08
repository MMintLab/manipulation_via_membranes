import torch
import torch.nn as nn
from manipulation_via_membranes.bubble_learning.models.bubble_dynamics_model_base import BubbleDynamicsModelBase


class BubbleDynamicsModel(BubbleDynamicsModelBase):

    def __init__(self, *args, input_batch_norm=True, **kwargs):
        self.input_batch_norm = input_batch_norm
        super().__init__(*args, **kwargs)
        sizes = self._get_sizes()
        self.dyn_input_batch_norm = nn.BatchNorm1d(num_features=sizes['dyn_input_size']) # call eval() to freeze the mean and std estimation
        self.save_hyperparameters()

    @classmethod
    def get_name(cls):
        return 'bubble_dynamics_model'

    def _get_dyn_input_size(self, sizes):
        dyn_input_size = self.img_embedding_size + sizes['init_wrench'] + sizes['init_pos'] + sizes[
            'init_quat'] + self.object_embedding_size + sizes['action']
        return dyn_input_size

    def _get_dyn_output_size(self, sizes):
        dyn_output_size = self.img_embedding_size + sizes['init_wrench']
        return dyn_output_size

    def forward(self, imprint, wrench, pos, ori, object_model, action):
        sizes = self._get_sizes()
        imprint_input_emb = self.autoencoder.encode(imprint) # (B, imprint_emb_size)
        obj_model_emb = self.object_embedding_module(object_model) # (B, imprint_emb_size)
        state_dyn_input = torch.cat([imprint_input_emb, wrench], dim=-1)
        dyn_input = torch.cat([state_dyn_input, pos, ori, obj_model_emb, action], dim=-1)
        if self.input_batch_norm:
            dyn_input = self.dyn_input_batch_norm(dyn_input)
        state_dyn_output_delta = self.dyn_model(dyn_input)
        state_dyn_output = state_dyn_input + state_dyn_output_delta
        imprint_emb_next, wrench_next = torch.split(state_dyn_output, (self.img_embedding_size, sizes['init_wrench']), dim=-1)
        imprint_next = self.autoencoder.decode(imprint_emb_next)
        return imprint_next, wrench_next

    def get_state_keys(self):
        state_keys = ['init_imprint', 'init_wrench', 'init_pos', 'init_quat',
                      'object_model']
        return state_keys
    
    def get_input_keys(self):
        input_keys = ['init_imprint', 'init_wrench', 'init_pos', 'init_quat',
                      'object_model']
        return input_keys

    def get_model_output_keys(self):
        output_keys = ['init_imprint', 'init_wrench']
        return output_keys

    def get_next_state_map(self):
        next_state_map = {
            'init_imprint': 'final_imprint',
            'init_wrench': 'final_wrench',

        }
        return next_state_map

    def _compute_loss(self, imprint_rec, wrench_rec, imprint_gth, wrench_gth):
        imprint_reconstruction_loss = self.mse_loss(imprint_rec, imprint_gth)
        wrench_reconstruction_loss = self.mse_loss(wrench_rec, wrench_gth)
        loss = imprint_reconstruction_loss + 0.000001 * wrench_reconstruction_loss
        return loss





